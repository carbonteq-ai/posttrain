"""Provider-neutral checkpoint snapshot and publication contracts."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal, cast

from posttrain.common import ContractError, HubModelRef

_SHA256 = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class CheckpointSnapshotId:
    """Stable logical identity for one producer run and exact training step."""

    run_id: str
    global_step: int

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ContractError("checkpoint snapshot run_id cannot be empty")
        if self.global_step < 0:
            raise ContractError("checkpoint snapshot global_step cannot be negative")

    @property
    def value(self) -> str:
        return f"{self.run_id}/step-{self.global_step:08d}"


@dataclass(frozen=True, slots=True)
class CheckpointComponent:
    """One manifest entry; paths are relative and digests are content-bound."""

    role: str
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise ContractError("checkpoint component role cannot be empty")
        path = PurePosixPath(self.relative_path)
        if not self.relative_path.strip() or path.is_absolute() or ".." in path.parts:
            raise ContractError("checkpoint component path must be a relative safe path")
        if self.size_bytes < 0:
            raise ContractError("checkpoint component size cannot be negative")
        if not _SHA256.fullmatch(self.sha256):
            raise ContractError("checkpoint component sha256 must be a SHA-256 digest")


@dataclass(frozen=True, slots=True)
class CheckpointSnapshotManifest:
    """Versioned structural manifest for a complete or incomplete snapshot."""

    snapshot_id: CheckpointSnapshotId
    created_at: datetime
    training_backend: str
    backend_revision: str
    technique: str
    parameter_update_kind: Literal["lora", "qlora", "full"]
    base_model: HubModelRef
    renderer_id: str
    tokenizer_fingerprint: str | None
    trainer_checkpoint_schema: str
    components: tuple[CheckpointComponent, ...]
    complete: bool = False
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise ContractError("checkpoint manifest schema_version must be positive")
        if self.created_at.tzinfo is None:
            raise ContractError("checkpoint manifest created_at must be timezone-aware")
        for name, value in (
            ("training_backend", self.training_backend),
            ("backend_revision", self.backend_revision),
            ("technique", self.technique),
            ("renderer_id", self.renderer_id),
            ("trainer_checkpoint_schema", self.trainer_checkpoint_schema),
        ):
            if not value.strip():
                raise ContractError(f"checkpoint manifest {name} cannot be empty")
        if self.tokenizer_fingerprint is not None and not _SHA256.fullmatch(self.tokenizer_fingerprint):
            raise ContractError("checkpoint tokenizer_fingerprint must be a SHA-256 digest")
        paths = [component.relative_path for component in self.components]
        if len(paths) != len(set(paths)):
            raise ContractError("checkpoint manifest component paths must be unique")
        if self.complete and not self.components:
            raise ContractError("complete checkpoint manifests require components")

    @property
    def checkpoint_snapshot_id(self) -> str:
        return self.snapshot_id.value

    @property
    def global_step(self) -> int:
        return self.snapshot_id.global_step

    @property
    def total_bytes(self) -> int:
        return sum(component.size_bytes for component in self.components)


@dataclass(frozen=True, slots=True)
class CheckpointPublicationPolicy:
    """Bounded publication choices independent from local checkpoint cadence."""

    publish_recovery: bool = True
    publish_model_view: bool = True
    milestone_steps: tuple[int, ...] = ()
    publish_terminal: bool = True
    publish_on_cancel: bool = True
    failure_fatal: bool = True
    max_queue_items: int = 2
    max_queue_bytes: int = 8 * 1024**3
    drain_timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if any(step < 0 for step in self.milestone_steps):
            raise ContractError("checkpoint milestone steps cannot be negative")
        if tuple(sorted(set(self.milestone_steps))) != self.milestone_steps:
            raise ContractError("checkpoint milestone steps must be sorted and unique")
        if self.max_queue_items < 1 or self.max_queue_bytes < 1:
            raise ContractError("checkpoint publication queue bounds must be positive")
        if self.drain_timeout_seconds <= 0:
            raise ContractError("checkpoint drain timeout must be positive")
        if not self.publish_recovery and not self.publish_model_view:
            raise ContractError("checkpoint publication must select a recovery or model view")

    def selects_step(self, step: int, *, terminal: bool = False) -> bool:
        if step < 0:
            raise ContractError("checkpoint step cannot be negative")
        return terminal and self.publish_terminal or step in self.milestone_steps


@dataclass(frozen=True, slots=True)
class CheckpointProjectionCapability:
    """Backend assertion describing which checkpoint views are directly safe."""

    update_kind: Literal["lora", "qlora", "full"]
    recovery: bool
    model_view: bool
    requires_transform: bool = False

    def __post_init__(self) -> None:
        if not self.recovery and not self.model_view:
            raise ContractError("checkpoint capability must expose a supported view")
        if self.model_view and self.requires_transform:
            raise ContractError("a direct model view cannot also require a transform")


CheckpointView = Literal["recovery", "model"]


@dataclass(frozen=True, slots=True)
class CheckpointSelector:
    """Read or consume one exact view from a source run."""

    source_run_id: str
    step: int | Literal["latest-complete"]
    view: CheckpointView

    def __post_init__(self) -> None:
        if not self.source_run_id.strip():
            raise ContractError("checkpoint selector source_run_id cannot be empty")
        if isinstance(self.step, int) and self.step < 0:
            raise ContractError("checkpoint selector step cannot be negative")


@dataclass(frozen=True, slots=True)
class CheckpointArtifactRecord:
    """Small provider-neutral artifact record used by inspection and planning."""

    logical_name: str
    kind: str
    provider: str
    namespace: str
    name: str
    version: str
    digest: str | None
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.logical_name.strip() or not self.kind.strip():
            raise ContractError("checkpoint artifact record requires logical_name and kind")
        if not self.provider.strip() or not self.namespace.strip() or not self.name.strip() or not self.version.strip():
            raise ContractError("checkpoint artifact record requires an immutable reference")
        if self.digest is not None and not _SHA256.fullmatch(self.digest):
            raise ContractError("checkpoint artifact record digest must be a SHA-256 digest")

    @property
    def has_digest(self) -> bool:
        return self.digest is not None

    @property
    def snapshot_id(self) -> str | None:
        value = self.metadata.get("checkpoint_snapshot_id")
        return value if isinstance(value, str) and value.strip() else None

    @property
    def step(self) -> int | None:
        value = self.metadata.get("checkpoint_step", self.metadata.get("global_step"))
        return value if isinstance(value, int) and value >= 0 else None

    @property
    def view(self) -> CheckpointView | None:
        value = self.metadata.get("checkpoint_view")
        if value in {"recovery", "model"}:
            return cast(CheckpointView, value)
        if self.kind == "training-checkpoint":
            return "recovery"
        if self.kind in {"model-adapter", "model-weights"}:
            return "model"
        return None


@dataclass(frozen=True, slots=True)
class ResolvedCheckpointSource:
    """Immutable source identity carried into a consumer job plan."""

    selector: CheckpointSelector
    snapshot_id: str
    step: int
    view: CheckpointView
    artifact: CheckpointArtifactRecord

    @property
    def artifact_kind(self) -> str:
        return self.artifact.kind


@dataclass(frozen=True, slots=True)
class CheckpointInspection:
    """One grouped snapshot with bounded readiness information."""

    snapshot_id: str
    step: int
    recovery: CheckpointArtifactRecord | None
    model: CheckpointArtifactRecord | None

    @property
    def ready(self) -> bool:
        return self.recovery is not None and self.model is not None

    @property
    def recovery_ready(self) -> bool:
        """Whether an immutable recovery view is available for continuation."""

        return self.recovery is not None

    @property
    def model_ready(self) -> bool:
        """Whether an immutable model view is available for eval or inference."""

        return self.model is not None


def _record_from_mapping(value: Mapping[str, object]) -> CheckpointArtifactRecord:
    artifact = value.get("artifact")
    stored = artifact if isinstance(artifact, Mapping) else value
    provider_metadata = stored.get("provider_metadata", {})
    metadata: dict[str, object] = {}
    if isinstance(provider_metadata, Mapping):
        metadata.update(provider_metadata)
    raw_metadata = value.get("metadata")
    if isinstance(raw_metadata, Mapping):
        metadata.update(raw_metadata)
    return CheckpointArtifactRecord(
        logical_name=str(value.get("logical_name", "")),
        kind=str(value.get("kind", "")),
        provider=str(stored.get("provider", "")),
        namespace=str(stored.get("namespace", "")),
        name=str(stored.get("name", "")),
        version=str(stored.get("version", "")),
        digest=(str(stored["digest"]) if stored.get("digest") is not None else None),
        metadata=metadata,
    )


def _record_step(views: Mapping[str, CheckpointArtifactRecord]) -> int:
    steps = {record.step for record in views.values()}
    if len(steps) != 1 or None in steps:
        raise ContractError("checkpoint views must agree on one non-negative step")
    step = next(iter(steps))
    assert isinstance(step, int)
    return step


def inspect_checkpoint_artifacts(records: Iterable[Mapping[str, object]]) -> tuple[CheckpointInspection, ...]:
    """Group committed checkpoint views without downloading manifests or bytes."""

    grouped: dict[str, dict[str, CheckpointArtifactRecord]] = {}
    for raw in records:
        record = _record_from_mapping(raw)
        snapshot_id = record.snapshot_id
        view = record.view
        if snapshot_id is None or record.step is None or view is None:
            continue
        views = grouped.setdefault(snapshot_id, {})
        if view in views:
            raise ContractError(f"checkpoint snapshot {snapshot_id!r} has duplicate {view} views")
        views[view] = record
    inspections = [
        CheckpointInspection(snapshot_id, _record_step(views), views.get("recovery"), views.get("model"))
        for snapshot_id, views in grouped.items()
    ]
    return tuple(sorted(inspections, key=lambda item: (item.step, item.snapshot_id), reverse=True))


def resolve_checkpoint_artifacts(
    records: Sequence[Mapping[str, object]],
    selector: CheckpointSelector,
) -> ResolvedCheckpointSource:
    """Resolve an exact or latest-complete view from committed artifact edges."""

    inspections = inspect_checkpoint_artifacts(records)
    candidates = [
        inspection
        for inspection in inspections
        if (
            (inspection.recovery_ready and inspection.recovery is not None and inspection.recovery.has_digest)
            if selector.view == "recovery"
            else (inspection.model_ready and inspection.model is not None and inspection.model.has_digest)
        )
    ]
    if selector.step == "latest-complete":
        selected = candidates[0] if candidates else None
    else:
        selected = next((item for item in candidates if item.step == selector.step), None)
    if selected is None:
        raise ContractError(
            f"source run {selector.source_run_id!r} has no {selector.view} checkpoint at {selector.step!r}"
        )
    artifact = selected.recovery if selector.view == "recovery" else selected.model
    assert artifact is not None
    return ResolvedCheckpointSource(selector, selected.snapshot_id, selected.step, selector.view, artifact)
