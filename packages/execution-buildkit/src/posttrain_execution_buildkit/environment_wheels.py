"""Build bounded, immutable wheels from verified environment Git sources."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from posttrain.common import ContractError
from posttrain.execution_pack import EnvironmentWheelRequest, ProjectEnvironmentSourceRequest

from .git_sources import (
    LockedGitSubdirectory,
    MaterializedGitSource,
    MaterializedGitSources,
    _tree_digest,
)

_SCHEMA = "posttrain.environment-wheel-lock.v1"
_DEFAULT_MAX_PACKAGES = 64
_DEFAULT_MAX_WHEEL_BYTES = 512 * 1024 * 1024


class WheelBuildGateway(Protocol):
    """Bounded command boundary for building one package into one directory."""

    def build(self, package_root: Path, output_directory: Path) -> None: ...


class UvWheelBuildCli:
    """Build a wheel with uv without invoking a shell."""

    def __init__(self, executable: str = "uv") -> None:
        self._executable = executable

    def build(self, package_root: Path, output_directory: Path) -> None:
        environment = {
            name: value
            for name in (
                "HOME",
                "HTTPS_PROXY",
                "HTTP_PROXY",
                "NO_PROXY",
                "PATH",
                "REQUESTS_CA_BUNDLE",
                "SSL_CERT_FILE",
            )
            if (value := os.environ.get(name)) is not None
        }
        result = subprocess.run(
            [
                self._executable,
                "build",
                "--no-config",
                "--no-sources",
                "--no-python-downloads",
                "--wheel",
                "--out-dir",
                str(output_directory),
                str(package_root),
            ],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"uv build --wheel failed with exit code {result.returncode}")


@dataclass(frozen=True, slots=True)
class LockedEnvironmentWheel:
    package: str
    repository: str | None
    revision: str | None
    subdirectory: str | None
    source_tree_digest: str
    wheel_filename: str
    wheel_sha256: str
    wheel_size_bytes: int
    source_kind: str = "git"
    project_path: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "package": self.package,
            "source_kind": self.source_kind,
            "repository": self.repository,
            "revision": self.revision,
            "subdirectory": self.subdirectory,
            "source_tree_digest": self.source_tree_digest,
            "wheel_filename": self.wheel_filename,
            "wheel_sha256": self.wheel_sha256,
            "wheel_size_bytes": self.wheel_size_bytes,
            "project_path": self.project_path,
        }


@dataclass(frozen=True, slots=True)
class EnvironmentWheelLock:
    packages: tuple[LockedEnvironmentWheel, ...]
    schema: str = _SCHEMA

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "packages": [package.as_dict() for package in self.packages],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class MaterializedEnvironmentWheel:
    path: Path
    lock: LockedEnvironmentWheel


@dataclass(frozen=True, slots=True)
class MaterializedEnvironmentWheels:
    wheels: tuple[MaterializedEnvironmentWheel, ...]
    lock: EnvironmentWheelLock


class ImmutableEnvironmentWheelBuilder:
    """Build exactly one bounded wheel for every selected package root."""

    def __init__(
        self,
        *,
        output_root: Path,
        gateway: WheelBuildGateway | None = None,
        max_packages: int = _DEFAULT_MAX_PACKAGES,
        max_wheel_bytes: int = _DEFAULT_MAX_WHEEL_BYTES,
    ) -> None:
        if not output_root.is_absolute():
            raise ContractError("environment wheel output root must be absolute")
        if max_packages < 1 or max_wheel_bytes < 1:
            raise ContractError("environment wheel limits must be positive")
        self._output_root = output_root
        self._gateway = gateway or UvWheelBuildCli()
        self._max_packages = max_packages
        self._max_wheel_bytes = max_wheel_bytes

    def build(
        self,
        sources: MaterializedGitSources,
        requests: Sequence[EnvironmentWheelRequest],
    ) -> MaterializedEnvironmentWheels:
        selected = _normalize_requests(requests, max_packages=self._max_packages)
        source_index = {(source.lock.repository, source.lock.revision): source for source in sources.sources}

        wheels: list[MaterializedEnvironmentWheel] = []
        for request in selected:
            source = source_index.get((request.repository, request.revision))
            if source is None:
                raise ContractError(f"environment package source was not materialized: {request.package}")
            selected_lock = _selected_subdirectory(source, request)
            _verify_materialized_source(source)
            package_root = (
                source.root if request.subdirectory == "." else source.root.joinpath(*request.subdirectory.split("/"))
            )
            _verify_selected_source(package_root, selected_lock)
            pyproject = package_root / "pyproject.toml"
            if not pyproject.is_file() or pyproject.is_symlink():
                raise ContractError(f"environment package root has no regular pyproject.toml: {request.package}")
            _verify_project_name(pyproject, request.package)
            wheel = self._build_one(package_root)
            try:
                _verify_materialized_source(source)
                _verify_selected_source(package_root, selected_lock)
                wheel_digest = _file_digest(wheel)
                wheel_size = wheel.stat().st_size
                _verify_wheel_filename(wheel.name, request.package)
                if wheel_size > self._max_wheel_bytes:
                    raise ContractError(f"environment wheel exceeds {self._max_wheel_bytes} bytes: {request.package}")
                retained = self._retain_wheel(wheel, wheel_digest)
                lock = LockedEnvironmentWheel(
                    package=request.package,
                    repository=request.repository,
                    revision=request.revision,
                    subdirectory=request.subdirectory,
                    source_tree_digest=selected_lock.tree_digest,
                    wheel_filename=retained.name,
                    wheel_sha256=wheel_digest,
                    wheel_size_bytes=wheel_size,
                )
            finally:
                wheel.unlink(missing_ok=True)
                wheel.parent.rmdir()
            wheels.append(MaterializedEnvironmentWheel(path=retained, lock=lock))

        result = tuple(wheels)
        return MaterializedEnvironmentWheels(
            wheels=result,
            lock=EnvironmentWheelLock(packages=tuple(wheel.lock for wheel in result)),
        )

    def build_project_sources(
        self,
        sources: Mapping[str, Path],
        requests: Sequence[ProjectEnvironmentSourceRequest],
    ) -> MaterializedEnvironmentWheels:
        """Build wheels from already-snapshotted project package roots."""

        wheels: list[MaterializedEnvironmentWheel] = []
        for request in sorted(requests, key=lambda item: (item.package, item.path)):
            try:
                package_root = sources[request.path]
            except KeyError as error:
                raise ContractError(f"project environment source was not materialized: {request.path}") from error
            if not package_root.is_absolute() or package_root.is_symlink() or not package_root.is_dir():
                raise ContractError(f"project environment source is not a regular directory: {request.path}")
            if _tree_digest(package_root) != request.tree_digest:
                raise ContractError(f"project environment source differs from its planned digest: {request.path}")
            pyproject = package_root / "pyproject.toml"
            if not pyproject.is_file() or pyproject.is_symlink():
                raise ContractError(f"environment package root has no regular pyproject.toml: {request.package}")
            _verify_project_name(pyproject, request.package)
            wheel = self._build_one(package_root)
            try:
                if _tree_digest(package_root) != request.tree_digest:
                    raise ContractError(f"project environment source changed during wheel build: {request.path}")
                wheel_digest = _file_digest(wheel)
                wheel_size = wheel.stat().st_size
                _verify_wheel_filename(wheel.name, request.package)
                if wheel_size > self._max_wheel_bytes:
                    raise ContractError(f"environment wheel exceeds {self._max_wheel_bytes} bytes: {request.package}")
                retained = self._retain_wheel(wheel, wheel_digest)
                lock = LockedEnvironmentWheel(
                    package=request.package,
                    repository=None,
                    revision=None,
                    subdirectory=None,
                    source_tree_digest=request.tree_digest,
                    wheel_filename=retained.name,
                    wheel_sha256=wheel_digest,
                    wheel_size_bytes=wheel_size,
                    source_kind="project-path",
                    project_path=request.path,
                )
            finally:
                wheel.unlink(missing_ok=True)
                wheel.parent.rmdir()
            wheels.append(MaterializedEnvironmentWheel(retained, lock))
        result = tuple(wheels)
        return MaterializedEnvironmentWheels(result, EnvironmentWheelLock(tuple(wheel.lock for wheel in result)))

    def _build_one(self, package_root: Path) -> Path:
        self._output_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".environment-wheel-build-", dir=self._output_root))
        try:
            staged_source = temporary / package_root.name
            shutil.copytree(package_root, staged_source)
            build_output = temporary / "output"
            build_output.mkdir()
            self._gateway.build(staged_source, build_output)
            entries = list(build_output.iterdir())
            meaningful_entries = [path for path in entries if not _is_uv_output_marker(path)]
            wheels = [
                path
                for path in meaningful_entries
                if path.suffix == ".whl" and path.is_file() and not path.is_symlink()
            ]
            if len(meaningful_entries) != 1 or len(wheels) != 1:
                raise ContractError("environment wheel build must emit exactly one regular wheel")
            retention_root = Path(tempfile.mkdtemp(prefix=".environment-wheel-retain-", dir=self._output_root))
            retained_staging = retention_root / wheels[0].name
            wheels[0].replace(retained_staging)
            return retained_staging
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def _retain_wheel(self, staged: Path, digest: str) -> Path:
        destination_root = self._output_root / digest
        destination_root.mkdir(parents=True, exist_ok=True)
        destination = destination_root / staged.name
        if destination.exists():
            if not destination.is_file() or destination.is_symlink() or _file_digest(destination) != digest:
                raise ContractError("retained environment wheel contains dirty drift")
        else:
            staged.replace(destination)
        return destination


def _normalize_requests(
    requests: Sequence[EnvironmentWheelRequest],
    *,
    max_packages: int,
) -> tuple[EnvironmentWheelRequest, ...]:
    if not requests:
        raise ContractError("at least one environment wheel is required")
    if len(requests) > max_packages:
        raise ContractError(f"environment wheel selection exceeds {max_packages} packages")

    by_package: dict[str, EnvironmentWheelRequest] = {}
    by_normalized_package: dict[str, str] = {}
    by_source_root: dict[tuple[str, str, str], str] = {}
    for request in requests:
        existing = by_package.get(request.package)
        if existing is not None and existing != request:
            raise ContractError(f"environment package identity selects multiple source roots: {request.package}")
        normalized_package = _normalized_package(request.package)
        spelling = by_normalized_package.get(normalized_package)
        if spelling is not None and spelling != request.package:
            raise ContractError("environment package identities collide after normalization")
        source_root = (request.repository, request.revision, request.subdirectory)
        other_package = by_source_root.get(source_root)
        if other_package is not None and other_package != request.package:
            raise ContractError("environment source root has conflicting package identities")
        by_package[request.package] = request
        by_normalized_package[normalized_package] = request.package
        by_source_root[source_root] = request.package
    if len(by_package) > max_packages:
        raise ContractError(f"environment wheel selection exceeds {max_packages} packages")
    return tuple(
        sorted(
            by_package.values(),
            key=lambda request: (
                request.package,
                request.repository,
                request.revision,
                request.subdirectory,
            ),
        )
    )


def _selected_subdirectory(
    source: MaterializedGitSource,
    request: EnvironmentWheelRequest,
) -> LockedGitSubdirectory:
    selected = {subdirectory.path: subdirectory for subdirectory in source.lock.subdirectories}.get(
        request.subdirectory
    )
    if selected is None:
        raise ContractError(f"environment package root was not selected in the Git lock: {request.package}")
    return selected


def _verify_materialized_source(source: MaterializedGitSource) -> None:
    observed = _tree_digest(source.root)
    if observed != source.lock.source_tree_digest:
        raise ContractError("materialized Git source contains filesystem drift")


def _verify_selected_source(
    package_root: Path,
    selected: LockedGitSubdirectory,
) -> None:
    if _tree_digest(package_root) != selected.tree_digest:
        raise ContractError("selected environment package contains filesystem drift")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_uv_output_marker(path: Path) -> bool:
    return path.name == ".gitignore" and path.is_file() and not path.is_symlink() and path.read_bytes() == b"*"


def _verify_project_name(pyproject: Path, expected: str) -> None:
    try:
        document = tomllib.loads(pyproject.read_text())
        observed = document["project"]["name"]
    except (KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        raise ContractError("environment pyproject must declare a static project.name") from error
    if not isinstance(observed, str) or _normalized_package(observed) != _normalized_package(expected):
        raise ContractError(f"environment package identity does not match pyproject: {expected}")


def _verify_wheel_filename(filename: str, expected: str) -> None:
    distribution, separator, _remainder = filename.partition("-")
    if separator != "-" or _normalized_package(distribution) != _normalized_package(expected):
        raise ContractError(f"environment wheel filename does not match package identity: {expected}")


def _normalized_package(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()
