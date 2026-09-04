"""Read-only semantic planning for one actual-job OCI image."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

from posttrain.common import ContractError, JsonValue
from posttrain.data import DatasetLoadPlan
from posttrain.environment import (
    EnvironmentBinding,
    ProjectPathActivationResource,
    ProjectPathEnvironmentSource,
    PythonFactoryActivation,
    VerifiersV1ConfigActivation,
)
from posttrain.eval import EvaluationPlan
from posttrain.execution import BackendRuntimeIdentity, EnvironmentActivationLock, RuntimeImageRef, StagedResourceLock
from posttrain.work import JobKind, PreparedWorkPackageJob, ResolvedSeats

from .contracts import (
    DatasetPackRequest,
    EnvironmentWheelRequest,
    GitSourceRequest,
    ProjectEnvironmentSourceRequest,
)
from .source_snapshot import ImmutableSourceSnapshotter, SourceSnapshotRequest

type JobKindProfile = Literal[
    "supervised",
    "online-rl",
    "eval",
    "serve",
    "transform",
]
type Compression = Literal["zstd"]

_PLAN_SCHEMA = "posttrain.job-pack-plan.v1"
_PUBLICATION_SCHEMA = "posttrain.job-publication.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@-]{0,255}$")
_OCI_REPOSITORY = re.compile(
    r"^[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]+)?"
    r"(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+$"
)
_PLATFORM = re.compile(r"^linux/(?:amd64|arm64)(?:/[a-z0-9._-]+)?$")

_KIND_PROFILES: Mapping[str, JobKindProfile] = {
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


@dataclass(frozen=True, slots=True)
class ImagePublicationSpec:
    """Registry-scoped publication settings, separate from package meaning."""

    repository: str
    platforms: tuple[str, ...] = ("linux/amd64",)
    compression: Compression = "zstd"
    compression_level: int = 3
    provenance: bool = True
    sbom: bool = True

    def __post_init__(self) -> None:
        if not _OCI_REPOSITORY.fullmatch(self.repository) or "@" in self.repository or "://" in self.repository:
            raise ContractError("publication repository must be a credential-free OCI repository")
        if (
            not self.platforms
            or len(set(self.platforms)) != len(self.platforms)
            or tuple(sorted(self.platforms)) != self.platforms
            or any(not _PLATFORM.fullmatch(value) for value in self.platforms)
        ):
            raise ContractError("publication platforms must be unique, sorted Linux OCI platforms")
        if self.compression != "zstd":
            raise ContractError("job image publication currently requires zstd")
        if not 1 <= self.compression_level <= 22:
            raise ContractError("zstd compression level must be between 1 and 22")
        if not self.provenance or not self.sbom:
            raise ContractError("job image publication requires provenance and an SBOM")

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "schema": _PUBLICATION_SCHEMA,
            "repository": self.repository,
            "platforms": list(self.platforms),
            "compression": self.compression,
            "compression_level": self.compression_level,
            "provenance": self.provenance,
            "sbom": self.sbom,
        }

    @classmethod
    def from_payload(cls, payload: object) -> ImagePublicationSpec:
        if not isinstance(payload, dict) or set(payload) != {
            "schema",
            "repository",
            "platforms",
            "compression",
            "compression_level",
            "provenance",
            "sbom",
        }:
            raise ContractError("image publication payload is invalid")
        if payload["schema"] != _PUBLICATION_SCHEMA:
            raise ContractError("image publication schema is unsupported")
        repository = payload["repository"]
        platforms = payload["platforms"]
        compression = payload["compression"]
        compression_level = payload["compression_level"]
        provenance = payload["provenance"]
        sbom = payload["sbom"]
        if (
            not isinstance(repository, str)
            or not isinstance(platforms, list)
            or not all(isinstance(item, str) for item in platforms)
            or compression != "zstd"
            or not isinstance(compression_level, int)
            or isinstance(compression_level, bool)
            or not isinstance(provenance, bool)
            or not isinstance(sbom, bool)
        ):
            raise ContractError("image publication payload has invalid field types")
        return cls(repository, tuple(platforms), compression, compression_level, provenance, sbom)


@dataclass(frozen=True, slots=True)
class JobPackSpec:
    """Immutable semantic inputs known without filesystem or network access."""

    project_id: str
    work_package_id: str
    job_id: str
    job_definition_id: str
    job_kind: JobKind
    kind_profile: JobKindProfile
    runtime_variant: str
    resolved_inputs_digest: str
    framework_source_digest: str
    project_source_digest: str
    universal_image: RuntimeImageRef
    kind_image: RuntimeImageRef
    datasets: tuple[DatasetPackRequest, ...]
    git_sources: tuple[GitSourceRequest, ...]
    environment_wheels: tuple[EnvironmentWheelRequest, ...]
    environment_activations: tuple[EnvironmentActivationLock, ...]
    expected_artifact_roles: tuple[str, ...]
    backend_runtime_identity: BackendRuntimeIdentity | None = None
    worker_contract_version: str = "1"
    family_registry_lock: Mapping[str, object] = field(default_factory=dict)
    project_environment_sources: tuple[ProjectEnvironmentSourceRequest, ...] = ()

    def __post_init__(self) -> None:
        for label, value in (
            ("project id", self.project_id),
            ("work package id", self.work_package_id),
            ("job id", self.job_id),
            ("job definition id", self.job_definition_id),
            ("worker contract version", self.worker_contract_version),
        ):
            if not _IDENTITY.fullmatch(value):
                raise ContractError(f"job pack {label} is invalid")
        if _KIND_PROFILES.get(self.job_kind) != self.kind_profile:
            raise ContractError(f"job kind {self.job_kind!r} requires profile {_KIND_PROFILES.get(self.job_kind)!r}")
        if not _IDENTITY.fullmatch(self.runtime_variant) or not (
            self.runtime_variant == self.kind_profile or self.runtime_variant.startswith(f"{self.kind_profile}-")
        ):
            raise ContractError("job pack runtime variant must refine its logical kind profile")
        if self.runtime_variant == "online-rl-verl-py313":
            if self.backend_runtime_identity is None:
                raise ContractError("veRL job pack requires immutable backend runtime identity")
        elif self.backend_runtime_identity is not None:
            raise ContractError("backend runtime identity is only valid for the veRL runtime variant")
        for label, digest in (
            ("resolved inputs", self.resolved_inputs_digest),
            ("framework source", self.framework_source_digest),
            ("project source", self.project_source_digest),
        ):
            if not _SHA256.fullmatch(digest):
                raise ContractError(f"{label} digest must be SHA-256")
        expected_sources = tuple(
            sorted(
                self.git_sources,
                key=lambda source: (
                    source.repository,
                    source.revision,
                    source.subdirectories,
                ),
            )
        )
        expected_datasets = tuple(
            sorted(
                self.datasets,
                key=lambda request: request.seat_name,
            )
        )
        expected_wheels = tuple(
            sorted(
                self.environment_wheels,
                key=lambda wheel: (
                    wheel.package,
                    wheel.repository,
                    wheel.revision,
                    wheel.subdirectory,
                ),
            )
        )
        expected_activations = tuple(
            sorted(
                self.environment_activations,
                key=lambda activation: activation.environment_id,
            )
        )
        if self.datasets != expected_datasets:
            raise ContractError("job pack datasets must be canonically ordered")
        if len({request.seat_name for request in self.datasets}) != len(self.datasets):
            raise ContractError("job pack dataset seat names must be unique")
        if self.git_sources != expected_sources:
            raise ContractError("job pack Git sources must be canonically ordered")
        if self.environment_wheels != expected_wheels:
            raise ContractError("job pack environment wheels must be canonically ordered")
        if self.environment_activations != expected_activations:
            raise ContractError("job pack environment activations must be canonically ordered")
        expected_project_sources = tuple(sorted(self.project_environment_sources, key=lambda source: source.path))
        if self.project_environment_sources != expected_project_sources:
            raise ContractError("job pack project environment sources must be canonically ordered")
        if len({activation.environment_id for activation in self.environment_activations}) != len(
            self.environment_activations
        ):
            raise ContractError("job pack environment activation ids must be unique")
        if (
            len(set(self.expected_artifact_roles)) != len(self.expected_artifact_roles)
            or tuple(sorted(self.expected_artifact_roles)) != self.expected_artifact_roles
        ):
            raise ContractError("expected artifact roles must be unique and sorted")
        if any(not role.strip() for role in self.expected_artifact_roles):
            raise ContractError("expected artifact roles cannot be empty")
        lock = dict(self.family_registry_lock)
        entries = lock.get("entries")
        digest = lock.get("digest")
        if lock and (
            not isinstance(entries, list) or not entries or not isinstance(digest, str) or not _SHA256.fullmatch(digest)
        ):
            raise ContractError("family registry lock must contain entries and a SHA-256 digest")
        try:
            json.dumps(lock, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as error:
            raise ContractError("family registry lock must contain only JSON values") from error
        object.__setattr__(self, "family_registry_lock", MappingProxyType(lock))

    @property
    def plan_key(self) -> str:
        """Identity of the immutable requests known before materialization."""

        return _digest(self.to_payload())

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "schema": _PLAN_SCHEMA,
            "project_id": self.project_id,
            "work_package_id": self.work_package_id,
            "job_id": self.job_id,
            "job_definition_id": self.job_definition_id,
            "job_kind": self.job_kind,
            "kind_profile": self.kind_profile,
            "runtime_variant": self.runtime_variant,
            "resolved_inputs_digest": self.resolved_inputs_digest,
            "framework_source_digest": self.framework_source_digest,
            "project_source_digest": self.project_source_digest,
            "universal_image": self.universal_image.value,
            "kind_image": self.kind_image.value,
            "datasets": [request.to_payload() for request in self.datasets],
            "git_sources": [
                {
                    "repository": source.repository,
                    "revision": source.revision,
                    "subdirectories": list(source.subdirectories),
                }
                for source in self.git_sources
            ],
            "environment_wheels": [
                {
                    "package": wheel.package,
                    "repository": wheel.repository,
                    "revision": wheel.revision,
                    "subdirectory": wheel.subdirectory,
                }
                for wheel in self.environment_wheels
            ],
            "environment_activations": [activation.to_payload() for activation in self.environment_activations],
            "project_environment_sources": [
                {"package": source.package, "path": source.path, "tree_digest": source.tree_digest}
                for source in self.project_environment_sources
            ],
            "expected_artifact_roles": list(self.expected_artifact_roles),
            "backend_runtime_identity": (
                self.backend_runtime_identity.to_payload() if self.backend_runtime_identity is not None else None
            ),
            "worker_contract_version": self.worker_contract_version,
            "family_registry_lock": cast(JsonValue, dict(self.family_registry_lock)),
        }


@dataclass(frozen=True, slots=True)
class JobPackPlan:
    """Complete read-only plan with separate package and publication identities."""

    spec: JobPackSpec
    publication: ImagePublicationSpec

    @property
    def plan_key(self) -> str:
        return self.spec.plan_key

    @property
    def publication_plan_key(self) -> str:
        return _digest(
            {
                "plan_key": self.plan_key,
                "publication": self.publication.to_payload(),
            }
        )


def environment_bindings(seats: ResolvedSeats) -> tuple[EnvironmentBinding, ...]:
    """Extract and deduplicate direct and evaluation-plan environment selections."""

    selected: list[EnvironmentBinding] = []
    for value in seats.values():
        if isinstance(value, EnvironmentBinding):
            selected.append(value)
        elif isinstance(value, EvaluationPlan):
            selected.extend(value.environments)

    by_id: dict[str, EnvironmentBinding] = {}
    for binding in selected:
        previous = by_id.get(binding.id)
        if previous is not None and previous != binding:
            raise ContractError(f"environment id selects conflicting bindings: {binding.id}")
        by_id[binding.id] = binding
    return tuple(sorted(by_id.values(), key=lambda binding: binding.id))


def plan_job_pack(
    prepared: PreparedWorkPackageJob,
    *,
    framework_source_digest: str,
    project_source_digest: str,
    universal_image: RuntimeImageRef,
    kind_image: RuntimeImageRef,
    publication: ImagePublicationSpec,
    runtime_variant: str | None = None,
    worker_contract_version: str = "1",
    family_registry_lock: Mapping[str, object] | None = None,
    project_root: Path | None = None,
    backend_runtime_identity: BackendRuntimeIdentity | None = None,
) -> JobPackPlan:
    """Derive an immutable plan without importing, fetching, building, or writing."""

    bindings = environment_bindings(prepared.seats)
    datasets = tuple(
        DatasetPackRequest(name, value)
        for name, value in sorted(prepared.seats.items())
        if isinstance(value, DatasetLoadPlan)
    )
    git_sources, wheels, project_sources = _environment_source_requests(bindings, project_root=project_root)
    activations = tuple(_activation_lock(binding, project_root=project_root) for binding in bindings)
    resolved_inputs_digest = _digest(dict(prepared.spec.resolved_inputs))
    profile = _KIND_PROFILES.get(prepared.recipe_job.kind)
    if profile is None:
        raise ContractError(f"job kind has no framework image profile: {prepared.recipe_job.kind}")
    spec = JobPackSpec(
        project_id=prepared.spec.project_id,
        work_package_id=prepared.spec.work_package_id,
        job_id=prepared.recipe_job.id,
        job_definition_id=prepared.definition.id,
        job_kind=prepared.recipe_job.kind,
        kind_profile=profile,
        runtime_variant=runtime_variant or profile,
        resolved_inputs_digest=resolved_inputs_digest,
        framework_source_digest=framework_source_digest,
        project_source_digest=project_source_digest,
        universal_image=universal_image,
        kind_image=kind_image,
        datasets=datasets,
        git_sources=git_sources,
        environment_wheels=wheels,
        environment_activations=activations,
        expected_artifact_roles=tuple(sorted(prepared.definition.required_artifact_roles)),
        backend_runtime_identity=backend_runtime_identity,
        worker_contract_version=worker_contract_version,
        family_registry_lock=family_registry_lock or {},
        project_environment_sources=project_sources,
    )
    return JobPackPlan(spec=spec, publication=publication)


def _environment_source_requests(
    bindings: tuple[EnvironmentBinding, ...],
    *,
    project_root: Path | None,
) -> tuple[
    tuple[GitSourceRequest, ...],
    tuple[EnvironmentWheelRequest, ...],
    tuple[ProjectEnvironmentSourceRequest, ...],
]:
    revisions: dict[str, str] = {}
    subdirectories: dict[tuple[str, str], set[str]] = {}
    wheels_by_package: dict[str, EnvironmentWheelRequest] = {}
    package_spellings: dict[str, str] = {}
    package_by_root: dict[tuple[str, str, str], str] = {}
    project_sources: list[ProjectEnvironmentSourceRequest] = []

    for binding in bindings:
        source = binding.source
        if isinstance(source, ProjectPathEnvironmentSource):
            if project_root is None:
                raise ContractError("planning a project-path environment requires the project root")
            root = project_root.resolve()
            package_root = (root / source.path).resolve()
            if (
                not package_root.is_relative_to(root)
                or package_root.is_symlink()
                or not (package_root / "pyproject.toml").is_file()
            ):
                raise ContractError(
                    f"project-path environment {source.package!r} must contain a regular pyproject.toml: {source.path}"
                )
            # A project environment is identified by the package tree itself.
            # Keep its project-relative path as provenance on the request, but
            # do not include that parent path in the content digest: the wheel
            # builder receives and verifies the selected package root.
            snapshot = SourceSnapshotRequest(root=package_root, includes=(".",), install_roots=(".",))
            digest = ImmutableSourceSnapshotter(
                cache_root=(root / ".posttrain" / "state" / "pack" / "sources")
            ).inspect(snapshot)
            project_sources.append(ProjectEnvironmentSourceRequest(source.package, source.path, digest))
            continue
        subdirectory = source.subdirectory or "."
        previous_revision = revisions.get(source.repository)
        if previous_revision is not None and previous_revision != source.revision:
            raise ContractError(
                "one job package cannot select multiple revisions of the same "
                f"environment repository: {source.repository}"
            )
        revisions[source.repository] = source.revision
        subdirectories.setdefault((source.repository, source.revision), set()).add(subdirectory)

        wheel = EnvironmentWheelRequest(
            package=source.package,
            repository=source.repository,
            revision=source.revision,
            subdirectory=subdirectory,
        )
        previous_wheel = wheels_by_package.get(source.package)
        if previous_wheel is not None and previous_wheel != wheel:
            raise ContractError(f"environment package identity selects multiple source roots: {source.package}")
        normalized_package = re.sub(
            r"[-_.]+",
            "-",
            source.package,
        ).lower()
        previous_spelling = package_spellings.get(normalized_package)
        if previous_spelling is not None and previous_spelling != source.package:
            raise ContractError("environment package identities collide after normalization")
        root = (source.repository, source.revision, subdirectory)
        previous_package = package_by_root.get(root)
        if previous_package is not None and previous_package != source.package:
            raise ContractError("environment source root has conflicting package identities")
        wheels_by_package[source.package] = wheel
        package_spellings[normalized_package] = source.package
        package_by_root[root] = source.package

    sources = tuple(
        GitSourceRequest(
            repository=repository,
            revision=revision,
            subdirectories=tuple(sorted(paths)),
        )
        for (repository, revision), paths in sorted(subdirectories.items())
    )
    wheels = tuple(
        sorted(
            wheels_by_package.values(),
            key=lambda wheel: (
                wheel.package,
                wheel.repository,
                wheel.revision,
                wheel.subdirectory,
            ),
        )
    )
    project = tuple(sorted(project_sources, key=lambda source: source.path))
    if len({source.package for source in project}) != len(project) or len({source.path for source in project}) != len(
        project
    ):
        raise ContractError("project environment package paths and identities must be unique")
    return sources, wheels, project


def activation_resource_sources(
    bindings: tuple[EnvironmentBinding, ...],
    *,
    project_root: Path,
) -> Mapping[tuple[str, str], Path]:
    """Resolve local files named by selected declarative environment resources."""

    root = project_root.resolve()
    sources: dict[tuple[str, str], Path] = {}
    for binding in bindings:
        activation = binding.activation
        if not isinstance(activation, VerifiersV1ConfigActivation):
            continue
        for name, resource in activation.resources.items():
            if not isinstance(resource, ProjectPathActivationResource):
                raise ContractError(f"environment {binding.id!r} has an unsupported activation resource")
            source = _project_resource_path(root, resource.path, binding.id, name)
            sources[(binding.id, name)] = source
        _reject_undeclared_project_paths(binding, root, sources)
    return MappingProxyType(sources)


def _activation_lock(
    binding: EnvironmentBinding,
    *,
    project_root: Path | None,
) -> EnvironmentActivationLock:
    reference = binding.activation.reference if isinstance(binding.activation, PythonFactoryActivation) else None
    config = dict(binding.activation.config) if not isinstance(binding.activation, PythonFactoryActivation) else None
    resources: Mapping[str, StagedResourceLock] = {}
    if isinstance(binding.activation, VerifiersV1ConfigActivation):
        if binding.activation.resources and project_root is None:
            raise ContractError(
                "planning an activation with project resources requires the project root; "
                "open the project before planning the job"
            )
        source_paths = activation_resource_sources((binding,), project_root=project_root) if project_root else {}
        resources = {
            name: StagedResourceLock(
                name=name,
                source_path=binding.activation.resources[name].path,
                staged_path=f"environment-resources/{binding.id}/{name}",
                digest=_file_digest(source),
                size_bytes=source.stat().st_size,
            )
            for (environment_id, name), source in source_paths.items()
            if environment_id == binding.id
        }
    return EnvironmentActivationLock(
        environment_id=binding.id,
        package=binding.source.package,
        kind=binding.activation.kind,
        digest=binding.activation.digest,
        reference=reference,
        config=config,
        resources=resources,
        qualification=binding.qualification,
    )


def _project_resource_path(root: Path, relative: str, environment_id: str, name: str) -> Path:
    source = (root / relative).resolve()
    if not source.is_relative_to(root) or not source.is_file() or source.is_symlink():
        raise ContractError(
            f"environment {environment_id!r} resource {name!r} must name a regular file under the project root: {relative}"
        )
    return source


def _reject_undeclared_project_paths(
    binding: EnvironmentBinding,
    root: Path,
    declared: Mapping[tuple[str, str], Path],
) -> None:
    activation = binding.activation
    if not isinstance(activation, VerifiersV1ConfigActivation):
        return
    declared_paths = set(declared.values())
    for value in _json_strings(activation.config):
        candidate = (root / value).resolve()
        if candidate in declared_paths or not candidate.is_relative_to(root) or not candidate.is_file():
            continue
        raise ContractError(
            f"environment {binding.id!r} activation references project file {value!r} directly; "
            "declare it under activation.resources and use { $resource: NAME } in the config"
        )


def _json_strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(item for child in value.values() for item in _json_strings(child))
    if isinstance(value, (tuple, list)):
        return tuple(item for child in value for item in _json_strings(child))
    return ()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(payload: Mapping[str, JsonValue]) -> str:
    try:
        encoded = json.dumps(
            dict(payload),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    except (TypeError, ValueError) as error:
        raise ContractError("job pack identity must contain only finite JSON values") from error
    return hashlib.sha256(encoded).hexdigest()
