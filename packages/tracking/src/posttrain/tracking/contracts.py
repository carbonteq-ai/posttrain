"""Provider-neutral tracking lifecycle and read contracts."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol

from posttrain.common import (
    ContractError,
    JsonValue,
    LocalArtifactRef,
    Observer,
    PublishedArtifact,
    StoredArtifactRef,
)
from posttrain.common.selections import validate_selection_id

if TYPE_CHECKING:
    from .models import (
        ArtifactSet,
        MetricSeries,
        RunDetail,
        RunQuery,
        RunSummary,
        TracePage,
        TraceQuery,
        TrackingCapabilities,
    )

type Stage = Literal["screen", "train", "qualify"]
type RunStatus = Literal["running", "succeeded", "partial", "failed", "cancelled", "unsupported"]
type RunOutcomeStatus = Literal["succeeded", "partial", "failed", "cancelled", "unsupported"]
type ArtifactIntegrityState = Literal["verified", "failed", "unsupported"]


@dataclass(frozen=True, slots=True)
class ArtifactIntegrityResult:
    """Bounded provider result for artifact presence and digest verification."""

    state: ArtifactIntegrityState
    checked_bytes: int = 0
    failures: tuple[str, ...] = ()
    deep: bool = False

    def __post_init__(self) -> None:
        if self.checked_bytes < 0:
            raise ContractError("artifact integrity checked_bytes cannot be negative")
        if any(not failure.strip() for failure in self.failures):
            raise ContractError("artifact integrity failures must be non-empty")


class ArtifactPublicationHandle(Protocol):
    """A queued artifact publication that can be awaited during finalization."""

    @property
    def submission_id(self) -> str | None: ...

    @property
    def state(self) -> Literal["pending", "uploading", "committed", "failed", "aborted"]: ...

    def wait(self, timeout: float | None = None) -> PublishedArtifact: ...


@dataclass(frozen=True, slots=True)
class ArtifactInput:
    """A named run input that a tracking backend must materialize."""

    reference: StoredArtifactRef
    kind: str

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ContractError("input artifact kind cannot be empty")


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Canonical run identity and immutable resolved inputs supplied by a host."""

    project_id: str
    work_package_id: str
    stage: Stage
    job_kind: str
    job_definition_version: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resolved_inputs: Mapping[str, JsonValue] = field(default_factory=dict)
    source_metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    artifacts: Mapping[str, ArtifactInput] = field(default_factory=dict)
    required_artifact_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_selection_id(self.project_id, "project id")
        validate_selection_id(self.work_package_id, "work package id")
        validate_selection_id(self.run_id, "run id")
        validate_selection_id(self.job_kind, "job kind")
        if not self.job_definition_version.strip():
            raise ContractError("job definition version cannot be empty")
        if any(not role.strip() for role in self.required_artifact_roles):
            raise ContractError("required artifact roles cannot be empty")
        if len(set(self.required_artifact_roles)) != len(self.required_artifact_roles):
            raise ContractError("required artifact roles must be unique")
        object.__setattr__(self, "resolved_inputs", MappingProxyType(dict(self.resolved_inputs)))
        object.__setattr__(self, "source_metadata", MappingProxyType(dict(self.source_metadata)))
        object.__setattr__(self, "artifacts", MappingProxyType(dict(self.artifacts)))


@dataclass(frozen=True, slots=True)
class RunError:
    """Safe failure information retained with a run outcome."""

    type: str
    message: str

    def __post_init__(self) -> None:
        if not self.type.strip() or not self.message.strip():
            raise ContractError("run errors require a type and safe message")


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """Canonical terminal state supplied once operation evidence is durable."""

    status: RunOutcomeStatus
    started_at: datetime
    finished_at: datetime
    error: RunError | None = None

    def __post_init__(self) -> None:
        if self.started_at.tzinfo is None or self.finished_at.tzinfo is None:
            raise ContractError("run outcome timestamps must be timezone-aware")
        if self.finished_at < self.started_at:
            raise ContractError("run outcome cannot finish before it starts")
        if self.status == "failed" and self.error is None:
            raise ContractError("failed run outcomes require safe error information")
        if self.status != "failed" and self.error is not None:
            raise ContractError("only failed run outcomes may carry error information")

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()


class TrackedRun(Observer, Protocol):
    """One open provider run that also accepts framework observations."""

    @property
    def run_id(self) -> str: ...

    def materialize_inputs(
        self,
        inputs: Mapping[str, ArtifactInput],
        root: Path,
    ) -> Mapping[str, LocalArtifactRef]: ...

    def finish(self, outcome: RunOutcome) -> None: ...

    def flush_artifacts(self, timeout: float | None = None) -> tuple[PublishedArtifact, ...]: ...


class TrackingBackend(Protocol):
    """Writer-side provider integration selected by a host."""

    def start_run(self, spec: RunSpec) -> TrackedRun: ...


class RunDataSource(Protocol):
    """Read-side provider integration consumed by job-aware views."""

    @property
    def capabilities(self) -> TrackingCapabilities: ...

    async def list_runs(self, query: RunQuery) -> tuple[RunSummary, ...]: ...

    async def get_run(self, run_id: str) -> RunDetail: ...

    async def metric_series(self, run_id: str, names: tuple[str, ...]) -> tuple[MetricSeries, ...]: ...

    async def traces(self, run_id: str, query: TraceQuery) -> TracePage: ...

    async def artifacts(self, run_id: str) -> ArtifactSet: ...

    async def verify_artifact(self, reference: StoredArtifactRef, *, deep: bool = False) -> ArtifactIntegrityResult: ...


__all__ = [
    "ArtifactInput",
    "ArtifactIntegrityResult",
    "ArtifactIntegrityState",
    "ArtifactPublicationHandle",
    "RunDataSource",
    "RunError",
    "RunOutcome",
    "RunOutcomeStatus",
    "RunSpec",
    "RunStatus",
    "Stage",
    "StoredArtifactRef",
    "TrackedRun",
    "TrackingBackend",
]
