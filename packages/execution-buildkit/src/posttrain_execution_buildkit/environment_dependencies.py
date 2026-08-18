"""Resolve selected environment wheels into one portable, hash-locked set."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path
from typing import Protocol

from posttrain.common import ContractError

from .environment_wheels import MaterializedEnvironmentWheels

_SCHEMA = "posttrain.environment-dependency-lock.v2"
_CONSTRAINT_PROFILE_SCHEMA = "posttrain.kind-dependency-constraints.v1"
_PROFILE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_PACKAGE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CONSTRAINT_PACKAGE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")
_HASH = re.compile(r"^--hash=sha256:([0-9a-f]{64})$")
_PINNED_REQUIREMENT = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*"
    r"(?:\[[A-Za-z0-9,._-]+\])?"
    r"==[^\s;\\]+(?:\s*;\s*.+)?$"
)
_FULL_GIT_REVISION = re.compile(r"git\+https://[^@\s]+@[0-9a-f]{40}(?:#[^\s]+)?$")
_VENDORED_WHEEL = re.compile(
    r"^(?P<package>[A-Za-z0-9][A-Za-z0-9._-]*)\s*@\s*"
    r"file:///opt/posttrain/vendor/(?P<filename>[A-Za-z0-9][A-Za-z0-9._-]*\.whl)"
    r"#sha256=(?P<digest>[0-9a-f]{64})$"
)
_URL_USERINFO = re.compile(r"https?://[^/\s:@]+(?::[^@/\s]*)?@")
_SENSITIVE_QUERY = re.compile(
    r"[?&](?:access[_-]?token|api[_-]?key|auth|credential|password|secret|token)=",
    re.IGNORECASE,
)
_SENSITIVE_OPTION = re.compile(
    r"^\s*--[^=\s]*(?:credential|password|secret|token)[^=\s]*=",
    re.IGNORECASE,
)
_DEFAULT_MAX_CONSTRAINT_BYTES = 4 * 1024 * 1024
_DEFAULT_MAX_LOCK_BYTES = 16 * 1024 * 1024
_DEFAULT_MAX_REQUIREMENTS = 4096
_PYTHON_VERSION = "3.13.12"
_PYTHON_PLATFORM = "x86_64-unknown-linux-gnu"
_PYTHON_EXECUTABLE = "/opt/posttrain/venv/bin/python"
_WHEEL_DIRECTORY = "wheels/environments"
_LOCK_FILENAME = "environment-dependencies.lock.txt"
_INDEX_ENVIRONMENT_NAMES = ("UV_INDEX_PASSWORD", "UV_INDEX_URL", "UV_INDEX_USERNAME")


class DependencyResolutionError(ContractError):
    """The complete environment set cannot produce a safe immutable lock."""


class DependencyCompileGateway(Protocol):
    """Non-shell boundary for one combined dependency-resolution invocation."""

    def compile(
        self,
        *,
        requirements: Path,
        constraints: Path,
        output: Path,
        working_directory: Path,
        python_version: str,
        python_platform: str,
        provided_packages: tuple[str, ...],
    ) -> None: ...


class UvDependencyCompileCli:
    """Compile all environment wheels together with a fixed uv target."""

    def __init__(
        self,
        executable: str = "uv",
        *,
        index_environment: Mapping[str, str] | None = None,
        runtime_vendor_root: Path | None = None,
    ) -> None:
        self._executable = executable
        supplied = dict(index_environment or {})
        unknown = sorted(set(supplied) - set(_INDEX_ENVIRONMENT_NAMES))
        if unknown:
            raise ContractError(f"unsupported dependency-index environment names: {', '.join(unknown)}")
        for name, value in supplied.items():
            if not isinstance(value, str) or not value or "\x00" in value:
                raise ContractError(f"dependency-index environment value is invalid: {name}")
        index_url = supplied.get("UV_INDEX_URL")
        if index_url is not None and (
            not index_url.startswith(("http://", "https://"))
            or _URL_USERINFO.search(index_url)
            or _SENSITIVE_QUERY.search(index_url)
        ):
            raise ContractError("dependency-index URL must be credential-free HTTP(S)")
        self._index_environment = supplied
        if runtime_vendor_root is not None and not runtime_vendor_root.is_absolute():
            raise ContractError("runtime vendor root must be absolute")
        self._runtime_vendor_root = runtime_vendor_root

    def compile(
        self,
        *,
        requirements: Path,
        constraints: Path,
        output: Path,
        working_directory: Path,
        python_version: str,
        python_platform: str,
        provided_packages: tuple[str, ...],
    ) -> None:
        host_constraints = _materialize_host_constraints(
            constraints,
            working_directory=working_directory,
            runtime_vendor_root=self._runtime_vendor_root,
        )
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
        environment.update(self._index_environment)
        arguments = [
            self._executable,
            "pip",
            "compile",
            "--generate-hashes",
            "--no-header",
            "--no-annotate",
            "--no-config",
            "--no-progress",
            "--color",
            "never",
            "--system-certs",
            "--index-strategy",
            "unsafe-best-match",
            "--python-version",
            python_version,
            "--python-platform",
            python_platform,
        ]
        for package in provided_packages:
            arguments.extend(("--no-emit-package", package))
        arguments.extend(
            [
                "--constraint",
                host_constraints.name,
                "--output-file",
                output.name,
                requirements.name,
            ]
        )
        result = subprocess.run(
            arguments,
            cwd=working_directory,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise DependencyResolutionError(f"uv dependency resolution failed with exit code {result.returncode}")


def _materialize_host_constraints(
    constraints: Path,
    *,
    working_directory: Path,
    runtime_vendor_root: Path | None,
) -> Path:
    """Map image-owned wheel URLs to verified temporary host copies for uv."""

    contents = constraints.read_text(encoding="utf-8")
    matches = tuple(
        match
        for line in contents.splitlines()
        if (match := _VENDORED_WHEEL.fullmatch(line.strip())) is not None
    )
    if not matches:
        return constraints
    if runtime_vendor_root is None or not runtime_vendor_root.is_dir():
        raise DependencyResolutionError("runtime vendored wheel root is unavailable on the packaging host")

    replacements: dict[str, str] = {}
    for match in matches:
        filename = match.group("filename")
        expected_digest = match.group("digest")
        source = runtime_vendor_root / filename
        if not source.is_file() or source.is_symlink() or _file_digest(source) != expected_digest:
            raise DependencyResolutionError("runtime vendored wheel differs from its immutable constraint")
        wheel_name, wheel_version = _wheel_identity(source)
        if re.sub(r"[-_.]+", "-", wheel_name).lower() != re.sub(
            r"[-_.]+", "-", match.group("package")
        ).lower():
            raise DependencyResolutionError("runtime vendored wheel package identity is inconsistent")
        replacements[match.group(0)] = f"{match.group('package')}=={wheel_version}"

    rewritten = "\n".join(replacements.get(line.strip(), line) for line in contents.splitlines()) + "\n"
    host_constraints = working_directory / "host-kind-constraints.txt"
    host_constraints.write_text(rewritten, encoding="utf-8")
    return host_constraints


def _wheel_identity(path: Path) -> tuple[str, str]:
    try:
        with zipfile.ZipFile(path) as archive:
            metadata = tuple(
                item
                for item in archive.infolist()
                if item.filename.endswith(".dist-info/METADATA") and not item.is_dir()
            )
            if len(metadata) != 1 or metadata[0].file_size > 1024 * 1024:
                raise DependencyResolutionError("runtime vendored wheel metadata is invalid")
            document = Parser().parsestr(archive.read(metadata[0]).decode("utf-8"))
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as error:
        raise DependencyResolutionError("runtime vendored wheel metadata is unreadable") from error
    name = document.get("Name", "").strip()
    version = document.get("Version", "").strip()
    if not _PACKAGE_NAME.fullmatch(name) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+!-]*", version) is None:
        raise DependencyResolutionError("runtime vendored wheel metadata identity is invalid")
    return name, version


@dataclass(frozen=True, slots=True)
class KindDependencyConstraints:
    """Secret-free, expanded constraints selected by one immutable kind image."""

    profile: str
    contents: str
    provided_packages: tuple[str, ...] = ()
    role: str = "control"
    python_version: str = _PYTHON_VERSION
    python_platform: str = _PYTHON_PLATFORM
    python_executable: str = _PYTHON_EXECUTABLE
    requirements_filename: str = _LOCK_FILENAME

    def __post_init__(self) -> None:
        if not _PROFILE.fullmatch(self.profile):
            raise ContractError("kind dependency profile is invalid")
        normalized = _normalize_constraints(self.contents)
        object.__setattr__(self, "contents", normalized)
        object.__setattr__(
            self,
            "provided_packages",
            _normalize_provided_packages(self.provided_packages),
        )
        if self.role not in {"control", "backend"}:
            raise ContractError("kind dependency role must be control or backend")
        if re.fullmatch(r"3[.][0-9]+(?:[.][0-9]+)?", self.python_version) is None:
            raise ContractError("kind dependency Python version is invalid")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", self.python_platform):
            raise ContractError("kind dependency Python platform is invalid")
        if not self.python_executable.startswith("/opt/") or not self.python_executable.endswith("/bin/python"):
            raise ContractError("kind dependency interpreter must be a capsule-owned Python path")
        if self.requirements_filename != _LOCK_FILENAME and (
            re.fullmatch(
                r"runtime[.][a-z]+[.]requirements[.]txt",
                self.requirements_filename,
            )
            is None
            or self.role not in self.requirements_filename
        ):
            raise ContractError("kind dependency requirements filename must identify its role")

    @property
    def constraints_sha256(self) -> str:
        return hashlib.sha256(self.contents.encode()).hexdigest()

    @property
    def constrained_packages(self) -> tuple[str, ...]:
        """Return normalized package names declared by this exact constraint set."""

        packages: list[str] = []
        for raw_line in self.contents.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            match = _CONSTRAINT_PACKAGE_NAME.match(line)
            if match is None:
                raise ContractError("kind dependency constraint must start with a package name")
            packages.append(re.sub(r"[-_.]+", "-", match.group()).lower())
        if len(packages) != len(set(packages)):
            raise ContractError("kind dependency constraints must name each package at most once")
        return tuple(sorted(packages))

    @property
    def digest(self) -> str:
        payload = {
            "schema": _CONSTRAINT_PROFILE_SCHEMA,
            "profile": self.profile,
            "constraints_sha256": self.constraints_sha256,
            "provided_packages": self.provided_packages,
            "role": self.role,
            "python_version": self.python_version,
            "python_platform": self.python_platform,
            "python_executable": self.python_executable,
            "requirements_filename": self.requirements_filename,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class EnvironmentDependencyLock:
    """Portable identity for one combined, target-specific resolution."""

    kind_profile: str
    kind_constraints_sha256: str
    constraint_profile_sha256: str
    provided_packages: tuple[str, ...]
    environment_wheel_lock_sha256: str
    requirements_sha256: str
    requirements_size_bytes: int
    requirement_count: int
    role: str = "control"
    python_version: str = _PYTHON_VERSION
    python_platform: str = _PYTHON_PLATFORM
    python_executable: str = _PYTHON_EXECUTABLE
    wheel_directory: str = _WHEEL_DIRECTORY
    requirements_filename: str = _LOCK_FILENAME
    schema: str = _SCHEMA

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "kind_profile": self.kind_profile,
            "kind_constraints_sha256": self.kind_constraints_sha256,
            "constraint_profile_sha256": self.constraint_profile_sha256,
            "provided_packages": list(self.provided_packages),
            "role": self.role,
            "environment_wheel_lock_sha256": self.environment_wheel_lock_sha256,
            "python_version": self.python_version,
            "python_platform": self.python_platform,
            "python_executable": self.python_executable,
            "wheel_directory": self.wheel_directory,
            "requirements_filename": self.requirements_filename,
            "requirements_sha256": self.requirements_sha256,
            "requirements_size_bytes": self.requirements_size_bytes,
            "requirement_count": self.requirement_count,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class MaterializedEnvironmentDependencyLock:
    path: Path
    lock: EnvironmentDependencyLock


class ImmutableEnvironmentDependencyCompiler:
    """Resolve and retain one bounded lock for all selected environments."""

    def __init__(
        self,
        *,
        output_root: Path,
        gateway: DependencyCompileGateway | None = None,
        max_constraint_bytes: int = _DEFAULT_MAX_CONSTRAINT_BYTES,
        max_lock_bytes: int = _DEFAULT_MAX_LOCK_BYTES,
        max_requirements: int = _DEFAULT_MAX_REQUIREMENTS,
    ) -> None:
        if not output_root.is_absolute():
            raise ContractError("environment dependency output root must be absolute")
        if max_constraint_bytes < 1 or max_lock_bytes < 1 or max_requirements < 1:
            raise ContractError("environment dependency limits must be positive")
        self._output_root = output_root
        self._gateway = gateway or UvDependencyCompileCli()
        self._max_constraint_bytes = max_constraint_bytes
        self._max_lock_bytes = max_lock_bytes
        self._max_requirements = max_requirements

    def compile(
        self,
        wheels: MaterializedEnvironmentWheels,
        constraints: KindDependencyConstraints,
    ) -> MaterializedEnvironmentDependencyLock:
        if len(constraints.contents.encode()) > self._max_constraint_bytes:
            raise ContractError("kind dependency constraints exceed the configured byte limit")
        selected = _verify_wheels(wheels)
        if len(selected) > self._max_requirements:
            raise ContractError("environment wheel selection exceeds the requirement limit")

        self._output_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=".environment-dependency-compile-",
                dir=self._output_root,
            )
        )
        try:
            staged_wheels = temporary / _WHEEL_DIRECTORY
            staged_wheels.mkdir(parents=True)
            for wheel in selected:
                shutil.copyfile(wheel.path, staged_wheels / wheel.lock.wheel_filename)

            requirements = temporary / "environment-wheels.in"
            requirements.write_text(
                "".join(f"./{_WHEEL_DIRECTORY}/{wheel.lock.wheel_filename}\n" for wheel in selected),
                encoding="utf-8",
            )
            constraint_file = temporary / "kind-constraints.txt"
            constraint_file.write_text(constraints.contents, encoding="utf-8")
            output = temporary / constraints.requirements_filename

            self._gateway.compile(
                requirements=requirements,
                constraints=constraint_file,
                output=output,
                working_directory=temporary,
                python_version=constraints.python_version,
                python_platform=constraints.python_platform,
                provided_packages=constraints.provided_packages,
            )
            _verify_wheels(wheels)
            raw = _read_bounded_regular_file(output, self._max_lock_bytes)
            canonical, count = _canonicalize_lock(
                raw,
                wheels=selected,
                max_requirements=self._max_requirements,
            )
            requirements_digest = hashlib.sha256(canonical).hexdigest()
            retained = self._retain(
                canonical,
                requirements_digest,
                filename=constraints.requirements_filename,
            )
            lock = EnvironmentDependencyLock(
                kind_profile=constraints.profile,
                kind_constraints_sha256=constraints.constraints_sha256,
                constraint_profile_sha256=constraints.digest,
                provided_packages=constraints.provided_packages,
                environment_wheel_lock_sha256=wheels.lock.digest,
                requirements_sha256=requirements_digest,
                requirements_size_bytes=len(canonical),
                requirement_count=count,
                role=constraints.role,
                python_version=constraints.python_version,
                python_platform=constraints.python_platform,
                python_executable=constraints.python_executable,
                requirements_filename=constraints.requirements_filename,
            )
            return MaterializedEnvironmentDependencyLock(path=retained, lock=lock)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def _retain(self, contents: bytes, digest: str, *, filename: str) -> Path:
        destination_root = self._output_root / digest
        destination_root.mkdir(parents=True, exist_ok=True)
        destination = destination_root / filename
        if destination.exists():
            if not destination.is_file() or destination.is_symlink() or destination.read_bytes() != contents:
                raise ContractError("retained environment dependency lock contains dirty drift")
        else:
            descriptor, staging_name = tempfile.mkstemp(
                prefix=f".{filename}.",
                dir=destination_root,
            )
            os.close(descriptor)
            staging = Path(staging_name)
            try:
                staging.write_bytes(contents)
                staging.replace(destination)
            finally:
                staging.unlink(missing_ok=True)
        return destination


def _normalize_constraints(contents: str) -> str:
    normalized = contents.replace("\r\n", "\n").replace("\r", "\n")
    if "\x00" in normalized:
        raise ContractError("kind dependency constraints contain a NUL byte")
    normalized = normalized.rstrip() + "\n"
    if normalized == "\n":
        raise ContractError("kind dependency constraints cannot be empty")
    for line in normalized.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if _URL_USERINFO.search(stripped) or _SENSITIVE_QUERY.search(stripped) or _SENSITIVE_OPTION.search(stripped):
            raise ContractError("kind dependency constraints must not contain secrets")
        if (
            lowered.startswith(("-r ", "--requirement ", "-c ", "--constraint "))
            or ("file://" in lowered and _VENDORED_WHEEL.fullmatch(stripped) is None)
            or stripped.startswith(("/", "\\"))
            or "${" in stripped
        ):
            raise ContractError("kind dependency constraints must be expanded and portable")
        if "git+" in lowered and not _FULL_GIT_REVISION.search(stripped):
            raise ContractError("kind dependency Git constraints require a full immutable commit")
    return normalized


def _normalize_provided_packages(
    packages: tuple[str, ...],
) -> tuple[str, ...]:
    normalized: list[str] = []
    for package in packages:
        if not isinstance(package, str) or not _PACKAGE_NAME.fullmatch(package):
            raise ContractError("kind dependency provided packages must be plain package names")
        canonical = re.sub(r"[-_.]+", "-", package).lower()
        normalized.append(canonical)
    if len(normalized) != len(set(normalized)):
        raise ContractError("kind dependency provided packages must be unique after normalization")
    return tuple(sorted(normalized))


def _verify_wheels(wheels: MaterializedEnvironmentWheels):
    if not wheels.wheels:
        raise ContractError("at least one materialized environment wheel is required")
    expected_locks = tuple(wheel.lock for wheel in wheels.wheels)
    if expected_locks != wheels.lock.packages:
        raise ContractError("materialized environment wheel lock is inconsistent")

    filenames: set[str] = set()
    for wheel in wheels.wheels:
        if not wheel.path.is_absolute() or not wheel.path.is_file() or wheel.path.is_symlink():
            raise ContractError("environment wheel must be an absolute regular file")
        if wheel.path.name != wheel.lock.wheel_filename:
            raise ContractError("environment wheel filename does not match its lock")
        if wheel.lock.wheel_filename in filenames:
            raise ContractError("environment wheel filenames must be unique")
        filenames.add(wheel.lock.wheel_filename)
        if wheel.path.stat().st_size != wheel.lock.wheel_size_bytes:
            raise ContractError("environment wheel size does not match its lock")
        if _file_digest(wheel.path) != wheel.lock.wheel_sha256:
            raise ContractError("environment wheel digest does not match its lock")
    return tuple(
        sorted(
            wheels.wheels,
            key=lambda wheel: (
                wheel.lock.package,
                wheel.lock.wheel_filename,
                wheel.lock.wheel_sha256,
            ),
        )
    )


def _read_bounded_regular_file(path: Path, limit: int) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise DependencyResolutionError("dependency compiler did not emit one regular lock file")
    if path.stat().st_size > limit:
        raise DependencyResolutionError("compiled environment dependency lock exceeds the byte limit")
    return path.read_bytes()


def _canonicalize_lock(
    raw: bytes,
    *,
    wheels,
    max_requirements: int,
) -> tuple[bytes, int]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DependencyResolutionError("compiled environment dependency lock is not UTF-8") from error
    if "\x00" in text:
        raise DependencyResolutionError("compiled environment dependency lock contains a NUL byte")

    blocks = _logical_blocks(text)
    if not blocks or len(blocks) > max_requirements:
        raise DependencyResolutionError("compiled environment dependency lock has an invalid requirement count")

    selected_wheels = {f"./{_WHEEL_DIRECTORY}/{wheel.lock.wheel_filename}": wheel.lock.wheel_sha256 for wheel in wheels}
    observed_wheels: set[str] = set()
    canonical_blocks: list[tuple[str, tuple[str, ...]]] = []
    heads: set[str] = set()
    for block in blocks:
        head, hashes = _validate_block(block)
        if head in heads:
            raise DependencyResolutionError("compiled environment dependency lock contains a duplicate requirement")
        heads.add(head)
        expected_wheel_hash = selected_wheels.get(head)
        if head.startswith(f"./{_WHEEL_DIRECTORY}/"):
            if expected_wheel_hash is None:
                raise DependencyResolutionError("compiled dependency lock references an unselected wheel")
            if hashes != (expected_wheel_hash,):
                raise DependencyResolutionError("compiled dependency lock wheel hash does not match its source lock")
            observed_wheels.add(head)
        canonical_blocks.append((head, hashes))

    if observed_wheels != set(selected_wheels):
        raise DependencyResolutionError("compiled dependency lock omitted a selected environment wheel")

    lines: list[str] = []
    for head, hashes in sorted(canonical_blocks, key=lambda item: item[0].casefold()):
        lines.append(f"{head} \\")
        for index, digest in enumerate(hashes):
            suffix = " \\" if index < len(hashes) - 1 else ""
            lines.append(f"    --hash=sha256:{digest}{suffix}")
    canonical = ("\n".join(lines) + "\n").encode()
    return canonical, len(canonical_blocks)


def _logical_blocks(text: str) -> tuple[tuple[str, ...], ...]:
    blocks: list[tuple[str, ...]] = []
    current: list[str] = []
    continuing = False
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            if continuing:
                raise DependencyResolutionError("compiled dependency lock has a malformed continuation")
            continue
        current.append(line[:-1].rstrip() if line.endswith("\\") else line)
        continuing = line.endswith("\\")
        if not continuing:
            blocks.append(tuple(current))
            current = []
    if continuing or current:
        raise DependencyResolutionError("compiled dependency lock has a dangling continuation")
    return tuple(blocks)


def _validate_block(block: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    if len(block) < 2:
        raise DependencyResolutionError("every compiled dependency must have a sha256 hash")
    head, *options = block
    if "@" in head or "://" in head or "${" in head or head.startswith(("/", "\\", "-")):
        raise DependencyResolutionError("compiled dependency lock contains a mutable or non-portable source")
    if head.startswith(f"./{_WHEEL_DIRECTORY}/"):
        filename = head.removeprefix(f"./{_WHEEL_DIRECTORY}/")
        if not filename or "/" in filename or "\\" in filename or not filename.endswith(".whl"):
            raise DependencyResolutionError("compiled dependency lock contains an invalid wheel reference")
    else:
        if "/" in head or "\\" in head:
            raise DependencyResolutionError("compiled dependency lock contains a non-portable path")
        if not _PINNED_REQUIREMENT.fullmatch(head):
            raise DependencyResolutionError("compiled dependency lock contains an unpinned requirement")

    hashes: list[str] = []
    for option in options:
        match = _HASH.fullmatch(option)
        if match is None:
            raise DependencyResolutionError("compiled dependency lock contains an unsupported or unhashed option")
        hashes.append(match.group(1))
    if not hashes:
        raise DependencyResolutionError("every compiled dependency must have a sha256 hash")
    return head, tuple(sorted(set(hashes)))


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
