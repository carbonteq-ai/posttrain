"""Provider-neutral value contracts used while packing actual-job images."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol
from urllib.parse import unquote, urlsplit

from posttrain.common import ContractError, JsonValue
from posttrain.data import DatasetLoadPlan
from posttrain.execution import (
    DatasetPackageLock,
    EnvironmentPackageLock,
    RuntimeDependencyLock,
)

_FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_PACKAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class GitSourceRequest:
    """One immutable repository revision and its required package roots."""

    repository: str
    revision: str
    subdirectories: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_repository(self.repository)
        if not _FULL_COMMIT.fullmatch(self.revision):
            raise ContractError("Git source revision must be a full lowercase commit SHA")
        if not self.subdirectories:
            raise ContractError("Git source must select at least one subdirectory")
        normalized = tuple(_validate_subdirectory(value) for value in self.subdirectories)
        if normalized != self.subdirectories:
            raise ContractError("Git source subdirectories must be normalized")
        if len(set(normalized)) != len(normalized):
            raise ContractError("Git source subdirectories must be unique")


@dataclass(frozen=True, slots=True)
class EnvironmentWheelRequest:
    """Logical package identity bound to one selected immutable source root."""

    package: str
    repository: str
    revision: str
    subdirectory: str

    def __post_init__(self) -> None:
        if not _PACKAGE.fullmatch(self.package):
            raise ContractError("environment package identity is invalid")
        GitSourceRequest(
            repository=self.repository,
            revision=self.revision,
            subdirectories=(self.subdirectory,),
        )


@dataclass(frozen=True, slots=True)
class DatasetPackRequest:
    """One inert dataset selection to materialize during the pack phase."""

    seat_name: str
    selection: DatasetLoadPlan

    def __post_init__(self) -> None:
        if (
            not self.seat_name
            or self.seat_name != self.seat_name.strip()
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,127}", self.seat_name)
        ):
            raise ContractError("dataset seat name is invalid")

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "seat_name": self.seat_name,
            "selection_id": self.selection.id,
            "revision": self.selection.revision,
            "kind": self.selection.kind,
            "source": dict(self.selection.source),
            "format": self.selection.format,
        }


@dataclass(frozen=True, slots=True)
class SourcePackage:
    """One already-filtered source tree selected for an actual-job image."""

    root: Path
    install_roots: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.root.is_absolute():
            raise ContractError("source package root must be absolute")
        normalized = tuple(_validate_subdirectory(value) for value in self.install_roots)
        if normalized != self.install_roots:
            raise ContractError("source install roots must be normalized")
        if len(set(normalized)) != len(normalized) or tuple(sorted(normalized)) != normalized:
            raise ContractError("source install roots must be unique and canonically ordered")


@dataclass(frozen=True, slots=True)
class MaterializedEnvironmentPackage:
    """A verified wheel paired with its portable package lock."""

    path: Path
    lock: EnvironmentPackageLock

    def __post_init__(self) -> None:
        if not self.path.is_absolute():
            raise ContractError("materialized environment wheel path must be absolute")


@dataclass(frozen=True, slots=True)
class MaterializedRuntimeDependency:
    """One interpreter-specific requirements file and its portable identity."""

    path: Path
    lock: RuntimeDependencyLock

    def __post_init__(self) -> None:
        if not self.path.is_absolute():
            raise ContractError("materialized runtime dependency path must be absolute")


@dataclass(frozen=True, slots=True)
class MaterializedEnvironments:
    """All selected environment wheels and their combined runtime lock."""

    packages: tuple[MaterializedEnvironmentPackage, ...]
    runtime_requirements: Path
    runtime_dependencies_digest: str
    runtime_dependencies: tuple[MaterializedRuntimeDependency, ...] = ()

    def __post_init__(self) -> None:
        if not self.runtime_requirements.is_absolute():
            raise ContractError("runtime requirements path must be absolute")
        if not re.fullmatch(r"[0-9a-f]{64}", self.runtime_dependencies_digest):
            raise ContractError("runtime dependencies digest must be SHA-256")
        if tuple(sorted(self.packages, key=lambda package: package.lock.package)) != self.packages:
            raise ContractError("materialized environment packages must be canonically ordered")
        roles = tuple(item.lock.role for item in self.runtime_dependencies)
        if len(set(roles)) != len(roles) or roles != tuple(sorted(roles)):
            raise ContractError("materialized runtime dependencies must have canonical unique roles")


class EnvironmentPackager(Protocol):
    """Adapter port hiding Git, wheel, and dependency-resolution mechanics."""

    def package(
        self,
        *,
        git_sources: tuple[GitSourceRequest, ...],
        wheel_requests: tuple[EnvironmentWheelRequest, ...],
        kind_profile: str,
        output_root: Path,
    ) -> MaterializedEnvironments: ...


class MaterializedDatasetPackagesPort(Protocol):
    @property
    def root(self) -> Path: ...

    @property
    def locks(self) -> tuple[DatasetPackageLock, ...]: ...


class DatasetPackager(Protocol):
    """Port implemented by the provider-neutral immutable dataset packager."""

    def package(
        self,
        requests: Sequence[DatasetPackRequest],
        *,
        output_root: Path,
    ) -> MaterializedDatasetPackagesPort: ...


def _validate_repository(value: str) -> None:
    if not value or value != value.strip():
        raise ContractError("Git source repository must be a canonical HTTPS URL")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or unquote(parsed.path) != parsed.path
    ):
        raise ContractError("Git source repository must be a secret-free canonical HTTPS URL")
    expected_netloc = parsed.hostname
    if parsed.port is not None:
        expected_netloc = f"{expected_netloc}:{parsed.port}"
    segments = parsed.path.split("/")[1:]
    if (
        parsed.netloc != expected_netloc
        or not segments
        or any(not part or part in {".", ".."} for part in segments)
        or parsed.path.endswith("/")
    ):
        raise ContractError("Git source repository URL is not canonical")


def _validate_subdirectory(value: str) -> str:
    if not value or "\\" in value or value != value.strip():
        raise ContractError("Git source subdirectory must be a normalized POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".."} for part in path.parts):
        raise ContractError("Git source subdirectory must remain inside its repository")
    normalized = path.as_posix()
    if normalized != value or (normalized != "." and "." in path.parts):
        raise ContractError("Git source subdirectory must be normalized")
    if ".git" in path.parts:
        raise ContractError("Git metadata cannot be selected as a source subdirectory")
    return normalized
