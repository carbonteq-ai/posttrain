"""Reusable job-package identity embedded in an actual-job OCI image."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Literal, cast

from posttrain.common import ContractError, JsonValue

from .contracts import RuntimeImageRef

_SCHEMA = "posttrain.job-package.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@-]{0,255}$")
_KIND_PROFILES = {
    "data.prepare": "supervised",
    "train.sft": "supervised",
    "train.dpo": "supervised",
    "train.grpo": "online-rl",
    "train.sampo": "online-rl",
    "train.distill": "online-rl",
    "eval.general": "eval",
    "eval.domain": "eval",
    "serve.benchmark": "serve",
    "serve.smoke": "serve",
    "model.transform": "transform",
}
JOB_PACKAGE_MANIFEST_PATH = PurePosixPath("/opt/posttrain/job/package.json")
JOB_PACKAGE_WORKER_COMMAND = (
    "posttrain-runtime",
    "execute",
    "--manifest",
    str(JOB_PACKAGE_MANIFEST_PATH),
)


@dataclass(frozen=True, slots=True)
class EnvironmentPackageLock:
    """One verified environment wheel built from immutable Git source."""

    package: str
    repository: str
    revision: str
    subdirectory: str
    tree_digest: str
    wheel_filename: str
    wheel_digest: str
    wheel_size_bytes: int

    def __post_init__(self) -> None:
        if not _IDENTITY.fullmatch(self.package):
            raise ContractError("environment package name is invalid")
        if not self.repository.startswith("https://"):
            raise ContractError("environment repository must use canonical HTTPS")
        if not _COMMIT.fullmatch(self.revision):
            raise ContractError("environment revision must be a full commit SHA")
        if self.subdirectory != ".":
            _relative_path(self.subdirectory, "environment subdirectory")
        _digest(self.tree_digest, "environment tree")
        if PurePosixPath(self.wheel_filename).name != self.wheel_filename or not self.wheel_filename.endswith(".whl"):
            raise ContractError("environment wheel filename must be a portable wheel filename")
        _digest(self.wheel_digest, "environment wheel")
        if self.wheel_size_bytes < 1:
            raise ContractError("environment wheel size must be positive")

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "package": self.package,
            "repository": self.repository,
            "revision": self.revision,
            "subdirectory": self.subdirectory,
            "tree_digest": self.tree_digest,
            "wheel_filename": self.wheel_filename,
            "wheel_digest": self.wheel_digest,
            "wheel_size_bytes": self.wheel_size_bytes,
        }


@dataclass(frozen=True, slots=True)
class StagedResourceLock:
    """One regular activation resource copied into an immutable job package."""

    name: str
    source_path: str
    staged_path: str
    digest: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not _IDENTITY.fullmatch(self.name):
            raise ContractError("activation resource name is invalid")
        source = PurePosixPath(self.source_path)
        if (
            not self.source_path
            or "\\" in self.source_path
            or source.is_absolute()
            or source.as_posix() != self.source_path
            or any(part in {"", ".", ".."} for part in source.parts)
        ):
            raise ContractError("activation resource source path is invalid")
        path = PurePosixPath(self.staged_path)
        if (
            not self.staged_path
            or "\\" in self.staged_path
            or path.is_absolute()
            or path.as_posix() != self.staged_path
            or any(part in {"", ".", ".."} for part in path.parts)
            or not path.is_relative_to(PurePosixPath("environment-resources"))
        ):
            raise ContractError("activation resource staged path is invalid")
        _digest(self.digest, "activation resource")
        if self.size_bytes < 0:
            raise ContractError("activation resource size cannot be negative")

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "name": self.name,
            "source_path": self.source_path,
            "staged_path": self.staged_path,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class EnvironmentActivationLock:
    """One selected environment binding backed by an installed package."""

    environment_id: str
    package: str
    kind: Literal["verifiers-config", "python-factory"]
    digest: str
    reference: str | None = None
    config: Mapping[str, JsonValue] | None = None
    resources: Mapping[str, StagedResourceLock] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _IDENTITY.fullmatch(self.environment_id):
            raise ContractError("environment activation id is invalid")
        if not _IDENTITY.fullmatch(self.package):
            raise ContractError("environment activation package is invalid")
        if self.kind == "verifiers-config":
            if self.reference is not None:
                raise ContractError("declarative Verifiers activation cannot name a Python factory")
            if self.config is None:
                raise ContractError("declarative Verifiers activation requires its JSON config")
            config = _freeze_json(dict(self.config))
            _json_bytes(
                _thaw_json(config),
                "environment activation config",
            )
            object.__setattr__(
                self,
                "config",
                cast(Mapping[str, JsonValue], config),
            )
        if self.kind == "python-factory" and (
            self.reference is None
            or ":" not in self.reference
            or any(character.isspace() for character in self.reference)
        ):
            raise ContractError("environment factory ref must be an import reference such as module:callable")
        if self.kind == "python-factory" and self.config is not None:
            raise ContractError("Python factory activation cannot include config")
        resources = dict(self.resources)
        if any(name != resource.name for name, resource in resources.items()):
            raise ContractError("activation resource mapping names must match their locks")
        if len({resource.staged_path for resource in resources.values()}) != len(resources):
            raise ContractError("activation resources must have distinct staged paths")
        object.__setattr__(self, "resources", MappingProxyType(dict(sorted(resources.items()))))
        activation_payload: dict[str, JsonValue] = {"kind": self.kind}
        if self.kind == "python-factory":
            activation_payload["reference"] = self.reference
        else:
            activation_payload["config"] = cast(
                JsonValue,
                _thaw_json(self.config or {}),
            )
            if self.resources:
                activation_payload["resources"] = cast(
                    JsonValue,
                {
                    name: {"source": {"kind": "project-path", "path": resource.source_path}}
                    for name, resource in self.resources.items()
                },
                )
        observed = hashlib.sha256(_json_bytes(activation_payload, "environment activation")).hexdigest()
        if self.digest != observed:
            raise ContractError("environment activation digest does not match its payload")

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "environment_id": self.environment_id,
            "package": self.package,
            "kind": self.kind,
            "digest": self.digest,
            "reference": self.reference,
            "config": (None if self.config is None else cast(JsonValue, _thaw_json(self.config))),
            "resources": {name: resource.to_payload() for name, resource in self.resources.items()},
        }


@dataclass(frozen=True, slots=True)
class DatasetPackageLock:
    """One immutable dataset snapshot copied into an actual-job image."""

    seat_name: str
    selection_id: str
    selection_revision: str
    dataset_revision: str
    kind: Literal["supervised", "preference"]
    schema_version: int
    digest: str
    package_path: str
    manifest_path: str
    size_bytes: int
    num_records: int | None = None

    def __post_init__(self) -> None:
        if not _IDENTITY.fullmatch(self.seat_name):
            raise ContractError("dataset seat name is invalid")
        if not _IDENTITY.fullmatch(self.selection_id):
            raise ContractError("dataset selection id is invalid")
        if not self.selection_revision.strip() or not self.dataset_revision.strip():
            raise ContractError("dataset revisions cannot be empty")
        if self.schema_version < 1:
            raise ContractError("dataset schema version must be positive")
        _digest(self.digest, "dataset")
        _relative_path(self.package_path, "dataset package path")
        _relative_path(self.manifest_path, "dataset manifest path")
        if self.size_bytes < 0:
            raise ContractError("dataset size cannot be negative")
        if self.num_records is not None and self.num_records < 0:
            raise ContractError("dataset record count cannot be negative")

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "seat_name": self.seat_name,
            "selection_id": self.selection_id,
            "selection_revision": self.selection_revision,
            "dataset_revision": self.dataset_revision,
            "kind": self.kind,
            "schema_version": self.schema_version,
            "digest": self.digest,
            "package_path": self.package_path,
            "manifest_path": self.manifest_path,
            "size_bytes": self.size_bytes,
            "num_records": self.num_records,
        }


@dataclass(frozen=True, slots=True)
class RuntimeDependencyLock:
    """One interpreter-specific, hash-locked actual-job dependency closure."""

    role: Literal["control", "backend"]
    python_version: str
    python_executable: str
    requirements_path: str
    requirements_digest: str
    resolution_digest: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"3[.][0-9]+(?:[.][0-9]+)?", self.python_version) is None:
            raise ContractError("runtime dependency Python version is invalid")
        executable = PurePosixPath(self.python_executable)
        if (
            not executable.is_absolute()
            or executable.as_posix() != self.python_executable
            or not executable.is_relative_to(PurePosixPath("/opt"))
            or executable.name != "python"
        ):
            raise ContractError("runtime dependency interpreter must be a normalized capsule path")
        requirements = _relative_path(
            self.requirements_path,
            "runtime dependency requirements path",
        )
        if not requirements.is_relative_to(PurePosixPath("locks")) or requirements.suffix != ".txt":
            raise ContractError("runtime dependency requirements must be a lock file below locks")
        _digest(self.requirements_digest, "runtime dependency requirements")
        _digest(self.resolution_digest, "runtime dependency resolution")

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "role": self.role,
            "python_version": self.python_version,
            "python_executable": self.python_executable,
            "requirements_path": self.requirements_path,
            "requirements_digest": self.requirements_digest,
            "resolution_digest": self.resolution_digest,
        }


@dataclass(frozen=True, slots=True)
class BackendRuntimeLock:
    """Capsule-owned veRL source, interpreter, lock, and projection identity."""

    backend: Literal["verl"]
    source_repository: str
    source_revision: str
    dependency_lock_path: str
    dependency_lock_digest: str
    working_directory: str
    projection_path: str
    projection_digest: str
    worker_module: str

    def __post_init__(self) -> None:
        if not self.source_repository.startswith("https://"):
            raise ContractError("backend runtime repository must use canonical HTTPS")
        if _COMMIT.fullmatch(self.source_revision) is None:
            raise ContractError("backend runtime source revision must be a full commit")
        for label, value in (
            ("dependency lock", self.dependency_lock_path),
            ("working directory", self.working_directory),
            ("projection", self.projection_path),
        ):
            path = PurePosixPath(value)
            if (
                not path.is_absolute()
                or path.as_posix() != value
                or not path.is_relative_to(PurePosixPath("/opt/posttrain-verl"))
            ):
                raise ContractError(f"backend runtime {label} must be a normalized capsule path")
        if self.dependency_lock_path != "/opt/posttrain-verl/release/uv.lock":
            raise ContractError("veRL dependency lock must use the capsule release path")
        if self.working_directory != "/opt/posttrain-verl/workdir":
            raise ContractError("veRL worktree must use the capsule workdir")
        if self.projection_path != "/opt/posttrain-verl/projection":
            raise ContractError("veRL projection must use the capsule projection path")
        _digest(self.dependency_lock_digest, "backend dependency lock")
        _digest(self.projection_digest, "backend worker projection")
        if self.worker_module != "posttrain.train.backends.verl.worker":
            raise ContractError("veRL worker module is unsupported")

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "backend": self.backend,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "dependency_lock_path": self.dependency_lock_path,
            "dependency_lock_digest": self.dependency_lock_digest,
            "working_directory": self.working_directory,
            "projection_path": self.projection_path,
            "projection_digest": self.projection_digest,
            "worker_module": self.worker_module,
        }


@dataclass(frozen=True, slots=True)
class JobPackageManifest:
    """Immutable job meaning shared by any number of execution attempts."""

    project_id: str
    work_package_id: str
    job_id: str
    job_definition_id: str
    job_kind: str
    resolved_inputs_digest: str
    framework_source_digest: str
    project_source_digest: str
    runtime_dependencies_digest: str
    code_requirements_digest: str
    resolved_config_digest: str
    project_config_digest: str
    universal_image: RuntimeImageRef
    kind_image: RuntimeImageRef
    runtime_variant: str
    runtime_dependency_locks: tuple[RuntimeDependencyLock, ...] = ()
    backend_runtime: BackendRuntimeLock | None = None
    environment_packages: tuple[EnvironmentPackageLock, ...] = ()
    environment_activations: tuple[EnvironmentActivationLock, ...] = ()
    datasets: tuple[DatasetPackageLock, ...] = ()
    expected_artifact_roles: tuple[str, ...] = ()
    worker_contract_version: str = "1"

    def __post_init__(self) -> None:
        for label, value in (
            ("project id", self.project_id),
            ("work package id", self.work_package_id),
            ("job id", self.job_id),
            ("job definition id", self.job_definition_id),
            ("job kind", self.job_kind),
            ("runtime variant", self.runtime_variant),
            ("worker contract version", self.worker_contract_version),
        ):
            if not _IDENTITY.fullmatch(value):
                raise ContractError(f"job package {label} is invalid")
        profile = _KIND_PROFILES.get(self.job_kind)
        if profile is None or not (self.runtime_variant == profile or self.runtime_variant.startswith(f"{profile}-")):
            raise ContractError("job package runtime variant must refine its logical kind profile")
        _digest(self.resolved_inputs_digest, "resolved inputs")
        _digest(self.framework_source_digest, "framework source")
        _digest(self.project_source_digest, "project source")
        _digest(self.runtime_dependencies_digest, "runtime dependencies")
        _digest(self.code_requirements_digest, "code requirements")
        _digest(self.resolved_config_digest, "resolved config")
        _digest(self.project_config_digest, "project config")
        roles = tuple(lock.role for lock in self.runtime_dependency_locks)
        if len(set(roles)) != len(roles) or roles != tuple(sorted(roles)):
            raise ContractError("runtime dependency locks must have unique, canonical roles")
        if self.runtime_variant.startswith("online-rl-verl-"):
            if roles != ("backend", "control"):
                raise ContractError("veRL runtime requires backend and control dependency locks")
            by_role = {lock.role: lock for lock in self.runtime_dependency_locks}
            control = by_role["control"]
            backend = by_role["backend"]
            if (
                control.python_version != "3.13.12"
                or control.python_executable != "/opt/posttrain/venv/bin/python"
                or backend.python_version != "3.13.12"
                or backend.python_executable != "/opt/posttrain-verl/bin/python"
            ):
                raise ContractError("veRL runtime dependency locks target the wrong interpreters")
            if self.backend_runtime is None:
                raise ContractError("veRL runtime requires capsule backend identity")
        elif self.backend_runtime is not None:
            raise ContractError("backend runtime identity is only valid for the veRL runtime variant")
        if len({item.package for item in self.environment_packages}) != len(self.environment_packages):
            raise ContractError("job package environment names must be unique")
        if len({item.environment_id for item in self.environment_activations}) != len(self.environment_activations):
            raise ContractError("job package environment activation ids must be unique")
        installed_packages = {item.package for item in self.environment_packages}
        missing_packages = sorted(
            {item.package for item in self.environment_activations if item.package not in installed_packages}
        )
        if missing_packages:
            raise ContractError(
                "job package environment activations reference missing packages: " + ", ".join(missing_packages)
            )
        if len({item.seat_name for item in self.datasets}) != len(self.datasets):
            raise ContractError("job package dataset seat names must be unique")
        if len(set(self.expected_artifact_roles)) != len(self.expected_artifact_roles):
            raise ContractError("job package artifact roles must be unique")
        if any(not role.strip() for role in self.expected_artifact_roles):
            raise ContractError("job package artifact roles cannot be empty")

    @property
    def package_key(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_payload(),
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "schema": _SCHEMA,
            "project_id": self.project_id,
            "work_package_id": self.work_package_id,
            "job_id": self.job_id,
            "job_definition_id": self.job_definition_id,
            "job_kind": self.job_kind,
            "resolved_inputs_digest": self.resolved_inputs_digest,
            "framework_source_digest": self.framework_source_digest,
            "project_source_digest": self.project_source_digest,
            "runtime_dependencies_digest": self.runtime_dependencies_digest,
            "code_requirements_digest": self.code_requirements_digest,
            "resolved_config_digest": self.resolved_config_digest,
            "project_config_digest": self.project_config_digest,
            "universal_image": self.universal_image.value,
            "kind_image": self.kind_image.value,
            "runtime_variant": self.runtime_variant,
            "runtime_dependency_locks": [item.to_payload() for item in self.runtime_dependency_locks],
            "backend_runtime": (self.backend_runtime.to_payload() if self.backend_runtime is not None else None),
            "environment_packages": [item.to_payload() for item in self.environment_packages],
            "environment_activations": [item.to_payload() for item in self.environment_activations],
            "datasets": [item.to_payload() for item in self.datasets],
            "expected_artifact_roles": list(self.expected_artifact_roles),
            "worker_contract_version": self.worker_contract_version,
        }

    def to_bytes(self) -> bytes:
        return (json.dumps(self.to_payload(), indent=2, sort_keys=True) + "\n").encode()

    @classmethod
    def from_payload(cls, payload: object) -> JobPackageManifest:
        if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
            raise ContractError("job package manifest schema is unsupported")
        allowed = {
            "schema",
            "project_id",
            "work_package_id",
            "job_id",
            "job_definition_id",
            "job_kind",
            "resolved_inputs_digest",
            "framework_source_digest",
            "project_source_digest",
            "runtime_dependencies_digest",
            "code_requirements_digest",
            "resolved_config_digest",
            "project_config_digest",
            "universal_image",
            "kind_image",
            "runtime_variant",
            "runtime_dependency_locks",
            "backend_runtime",
            "environment_packages",
            "environment_activations",
            "datasets",
            "expected_artifact_roles",
            "worker_contract_version",
        }
        if unknown := sorted(set(payload) - allowed):
            raise ContractError(f"job package manifest has unknown fields: {', '.join(unknown)}")
        try:
            environment_packages = payload.get("environment_packages", [])
            runtime_dependency_locks = payload.get("runtime_dependency_locks", [])
            environment_activations = payload.get("environment_activations", [])
            datasets = payload.get("datasets", [])
            roles = payload.get("expected_artifact_roles", [])
            if not isinstance(environment_packages, list):
                raise TypeError("environment packages")
            if not isinstance(runtime_dependency_locks, list):
                raise TypeError("runtime dependency locks")
            if not isinstance(environment_activations, list):
                raise TypeError("environment activations")
            if not isinstance(datasets, list):
                raise TypeError("datasets")
            if not isinstance(roles, list) or not all(isinstance(role, str) for role in roles):
                raise TypeError("artifact roles")
            return cls(
                project_id=str(payload["project_id"]),
                work_package_id=str(payload["work_package_id"]),
                job_id=str(payload["job_id"]),
                job_definition_id=str(payload["job_definition_id"]),
                job_kind=str(payload["job_kind"]),
                resolved_inputs_digest=str(payload["resolved_inputs_digest"]),
                framework_source_digest=str(payload["framework_source_digest"]),
                project_source_digest=str(payload["project_source_digest"]),
                runtime_dependencies_digest=str(payload["runtime_dependencies_digest"]),
                code_requirements_digest=str(payload["code_requirements_digest"]),
                resolved_config_digest=str(payload["resolved_config_digest"]),
                project_config_digest=str(payload["project_config_digest"]),
                universal_image=RuntimeImageRef(str(payload["universal_image"])),
                kind_image=RuntimeImageRef(str(payload["kind_image"])),
                runtime_variant=str(payload["runtime_variant"]),
                runtime_dependency_locks=tuple(_runtime_dependency_lock(item) for item in runtime_dependency_locks),
                backend_runtime=_backend_runtime_lock(payload.get("backend_runtime")),
                environment_packages=tuple(_environment_package_lock(item) for item in environment_packages),
                environment_activations=tuple(_environment_activation_lock(item) for item in environment_activations),
                datasets=tuple(_dataset_lock(item) for item in datasets),
                expected_artifact_roles=tuple(roles),
                worker_contract_version=str(payload["worker_contract_version"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("job package manifest fields are invalid") from error

    @classmethod
    def from_bytes(cls, value: bytes) -> JobPackageManifest:
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContractError("job package manifest is invalid JSON") from error
        return cls.from_payload(payload)


def _relative_path(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ContractError(f"{label} must be a normalized relative path")
    return path


def _digest(value: str, label: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ContractError(f"{label} digest must be SHA-256")


def _json_bytes(value: object, label: str) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    except (TypeError, ValueError) as error:
        raise ContractError(f"{label} must contain only JSON values") from error


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ContractError("JSON object keys must be strings")
        return MappingProxyType({key: _freeze_json(child) for key, child in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(child) for child in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


def _environment_package_lock(value: object) -> EnvironmentPackageLock:
    if not isinstance(value, dict):
        raise TypeError("environment lock")
    package = value.get("package")
    repository = value.get("repository")
    revision = value.get("revision")
    subdirectory = value.get("subdirectory")
    tree_digest = value.get("tree_digest")
    wheel_filename = value.get("wheel_filename")
    wheel_digest = value.get("wheel_digest")
    wheel_size_bytes = value.get("wheel_size_bytes")
    if not isinstance(wheel_size_bytes, int) or isinstance(wheel_size_bytes, bool):
        raise TypeError("environment wheel size")
    return EnvironmentPackageLock(
        package=_required_string(package, "environment package"),
        repository=_required_string(repository, "environment repository"),
        revision=_required_string(revision, "environment revision"),
        subdirectory=_required_string(subdirectory, "environment subdirectory"),
        tree_digest=_required_string(tree_digest, "environment tree digest"),
        wheel_filename=_required_string(wheel_filename, "environment wheel filename"),
        wheel_digest=_required_string(wheel_digest, "environment wheel digest"),
        wheel_size_bytes=wheel_size_bytes,
    )


def _environment_activation_lock(value: object) -> EnvironmentActivationLock:
    if not isinstance(value, dict):
        raise TypeError("environment activation lock")
    kind = value.get("kind")
    reference = value.get("reference")
    config = value.get("config")
    resources = value.get("resources", {})
    if kind not in {"verifiers-config", "python-factory"}:
        raise TypeError("environment activation kind")
    if reference is not None and not isinstance(reference, str):
        raise TypeError("environment activation reference")
    if config is not None and not isinstance(config, dict):
        raise TypeError("environment activation config")
    if not isinstance(resources, dict):
        raise TypeError("environment activation resources")
    return EnvironmentActivationLock(
        environment_id=_required_string(value.get("environment_id"), "environment activation id"),
        package=_required_string(value.get("package"), "environment activation package"),
        kind=kind,
        digest=_required_string(value.get("digest"), "environment activation digest"),
        reference=reference,
        config=config,
        resources={
            _required_string(name, "environment activation resource name"): _staged_resource_lock(resource)
            for name, resource in resources.items()
        },
    )


def _staged_resource_lock(value: object) -> StagedResourceLock:
    if not isinstance(value, dict):
        raise TypeError("activation resource lock")
    size_bytes = value.get("size_bytes")
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool):
        raise TypeError("activation resource size")
    return StagedResourceLock(
        name=_required_string(value.get("name"), "activation resource name"),
        source_path=_required_string(value.get("source_path"), "activation resource source path"),
        staged_path=_required_string(value.get("staged_path"), "activation resource staged path"),
        digest=_required_string(value.get("digest"), "activation resource digest"),
        size_bytes=size_bytes,
    )


def _runtime_dependency_lock(value: object) -> RuntimeDependencyLock:
    if not isinstance(value, dict):
        raise TypeError("runtime dependency lock")
    role = value.get("role")
    if role not in {"control", "backend"}:
        raise TypeError("runtime dependency role")
    return RuntimeDependencyLock(
        role=cast(Literal["control", "backend"], role),
        python_version=_required_string(
            value.get("python_version"),
            "runtime dependency Python version",
        ),
        python_executable=_required_string(
            value.get("python_executable"),
            "runtime dependency interpreter",
        ),
        requirements_path=_required_string(
            value.get("requirements_path"),
            "runtime dependency requirements path",
        ),
        requirements_digest=_required_string(
            value.get("requirements_digest"),
            "runtime dependency requirements digest",
        ),
        resolution_digest=_required_string(
            value.get("resolution_digest"),
            "runtime dependency resolution digest",
        ),
    )


def _backend_runtime_lock(value: object) -> BackendRuntimeLock | None:
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("backend") != "verl":
        raise TypeError("backend runtime lock")
    return BackendRuntimeLock(
        backend="verl",
        source_repository=_required_string(
            value.get("source_repository"),
            "backend runtime repository",
        ),
        source_revision=_required_string(
            value.get("source_revision"),
            "backend runtime source revision",
        ),
        dependency_lock_path=_required_string(
            value.get("dependency_lock_path"),
            "backend dependency lock path",
        ),
        dependency_lock_digest=_required_string(
            value.get("dependency_lock_digest"),
            "backend dependency lock digest",
        ),
        working_directory=_required_string(
            value.get("working_directory"),
            "backend working directory",
        ),
        projection_path=_required_string(
            value.get("projection_path"),
            "backend projection path",
        ),
        projection_digest=_required_string(
            value.get("projection_digest"),
            "backend projection digest",
        ),
        worker_module=_required_string(
            value.get("worker_module"),
            "backend worker module",
        ),
    )


def _dataset_lock(value: object) -> DatasetPackageLock:
    if not isinstance(value, dict):
        raise TypeError("dataset lock")
    seat_name = _required_string(value.get("seat_name"), "dataset seat name")
    selection_id = _required_string(value.get("selection_id"), "dataset selection id")
    selection_revision = _required_string(value.get("selection_revision"), "dataset selection revision")
    dataset_revision = _required_string(value.get("dataset_revision"), "dataset revision")
    kind = value.get("kind")
    schema_version = value.get("schema_version")
    digest = value.get("digest")
    package_path = _required_string(value.get("package_path"), "dataset package path")
    manifest_path = _required_string(value.get("manifest_path"), "dataset manifest path")
    size_bytes = value.get("size_bytes")
    num_records = value.get("num_records")
    if kind not in {"supervised", "preference"}:
        raise TypeError("dataset kind")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise TypeError("dataset schema version")
    if not isinstance(digest, str):
        raise TypeError("dataset digest")
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool):
        raise TypeError("dataset lock size")
    if num_records is not None and (not isinstance(num_records, int) or isinstance(num_records, bool)):
        raise TypeError("dataset lock record count")
    return DatasetPackageLock(
        seat_name=seat_name,
        selection_id=selection_id,
        selection_revision=selection_revision,
        dataset_revision=dataset_revision,
        kind=kind,
        schema_version=schema_version,
        digest=digest,
        package_path=package_path,
        manifest_path=manifest_path,
        size_bytes=size_bytes,
        num_records=num_records,
    )


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(label)
    return value


__all__ = [
    "JOB_PACKAGE_MANIFEST_PATH",
    "JOB_PACKAGE_WORKER_COMMAND",
    "DatasetPackageLock",
    "EnvironmentActivationLock",
    "EnvironmentPackageLock",
    "JobPackageManifest",
]
