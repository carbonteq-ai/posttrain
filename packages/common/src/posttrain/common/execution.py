"""Framework-neutral execution and observation contracts."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .artifacts import JsonValue, LocalArtifactRef, ProducedArtifact
from .errors import OperationCancelled
from .selections import validate_selection_id

Attributes = Mapping[str, JsonValue]
Clock = Callable[[], datetime]


@dataclass(frozen=True, slots=True)
class EventObservation:
    name: str
    occurred_at: datetime
    attributes: Attributes = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MetricObservation:
    name: str
    value: float
    step: int | None = None
    attributes: Attributes = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MetricBatchObservation:
    values: Mapping[str, float]
    step: int | None = None
    attributes: Attributes = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TraceObservation:
    trace_type: str
    external_id: str
    payload: Mapping[str, JsonValue]
    attributes: Attributes = field(default_factory=dict)


class Observer(Protocol):
    def event(self, observation: EventObservation) -> None: ...

    def metric(self, observation: MetricObservation) -> None: ...

    def metrics(self, observation: MetricBatchObservation) -> None: ...

    def trace(self, observation: TraceObservation) -> None: ...

    def artifact(self, artifact: ProducedArtifact) -> None: ...


class NullObserver:
    """Observer used when reusable operations run without a host platform."""

    def event(self, observation: EventObservation) -> None:
        del observation

    def metric(self, observation: MetricObservation) -> None:
        del observation

    def metrics(self, observation: MetricBatchObservation) -> None:
        del observation

    def trace(self, observation: TraceObservation) -> None:
        del observation

    def artifact(self, artifact: ProducedArtifact) -> None:
        del artifact


class CancellationToken:
    """Thread-safe cooperative cancellation shared by package operations."""

    def __init__(self) -> None:
        self._cancelled = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        self._cancelled.set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise OperationCancelled("operation was cancelled")


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class RunContext:
    """Canonical host-injected identity and observation context for one run."""

    project_id: str
    work_package_id: str
    run_id: str
    job_kind: str
    job_definition_version: str
    workspace: Path
    observer: Observer = field(default_factory=NullObserver)
    cancellation: CancellationToken = field(default_factory=CancellationToken)
    clock: Clock = utc_now
    source_metadata: Attributes = field(default_factory=dict)
    input_artifacts: Mapping[str, LocalArtifactRef] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_selection_id(self.project_id, "project id")
        validate_selection_id(self.work_package_id, "work package id")
        validate_selection_id(self.run_id, "run id")
        validate_selection_id(self.job_kind, "job kind")
        if not self.job_definition_version.strip():
            raise ValueError("job definition version cannot be empty")
        if not self.workspace.is_absolute():
            raise ValueError("run workspace must be absolute")

    @property
    def identity_attributes(self) -> dict[str, JsonValue]:
        return {
            "project_id": self.project_id,
            "work_package_id": self.work_package_id,
            "run_id": self.run_id,
            "job_kind": self.job_kind,
            "job_definition_version": self.job_definition_version,
        }

    def _attributes(self, attributes: Attributes | None = None) -> dict[str, JsonValue]:
        return {**self.identity_attributes, **dict(attributes or {})}

    def event(self, name: str, attributes: Attributes | None = None) -> None:
        self.cancellation.raise_if_cancelled()
        self._emit_event(name, attributes)

    def _emit_event(self, name: str, attributes: Attributes | None = None) -> None:
        self.observer.event(EventObservation(name, self.clock(), self._attributes(attributes)))

    @contextmanager
    def phase(
        self,
        name: str,
        attributes: Attributes | None = None,
    ) -> Iterator[str]:
        """Record one timestamped runtime phase without coupling to a provider.

        Phases may be nested. Readers assign a host sample to the most specific
        active interval, so an outer ``actor_update`` phase can contain a
        shorter ``rollout`` phase without double-counting the sample.
        """

        phase = name.strip()
        if not phase:
            raise ValueError("runtime phase name cannot be empty")
        phase_id = str(uuid.uuid4())
        phase_attributes = {**dict(attributes or {}), "phase": phase, "phase_id": phase_id}
        self.cancellation.raise_if_cancelled()
        self._emit_event("runtime_phase_started", phase_attributes)
        try:
            yield phase_id
        except BaseException as error:
            self._emit_event(
                "runtime_phase_failed",
                {**phase_attributes, "error_type": type(error).__name__},
            )
            raise
        else:
            self._emit_event("runtime_phase_completed", phase_attributes)

    def metric(
        self,
        name: str,
        value: float,
        *,
        step: int | None = None,
        attributes: Attributes | None = None,
    ) -> None:
        self.cancellation.raise_if_cancelled()
        self.observer.metric(MetricObservation(name, float(value), step, self._attributes(attributes)))

    def metrics(
        self,
        values: Mapping[str, float],
        *,
        step: int | None = None,
        attributes: Attributes | None = None,
    ) -> None:
        self.cancellation.raise_if_cancelled()
        self.observer.metrics(
            MetricBatchObservation(
                {name: float(value) for name, value in values.items()},
                step,
                self._attributes(attributes),
            )
        )

    def trace(self, observation: TraceObservation) -> None:
        self.cancellation.raise_if_cancelled()
        self.observer.trace(replace(observation, attributes=self._attributes(observation.attributes)))

    def artifact(self, artifact: ProducedArtifact) -> None:
        self.cancellation.raise_if_cancelled()
        self.observer.artifact(replace(artifact, metadata=self._attributes(artifact.metadata)))

    def input_artifact(self, name: str) -> LocalArtifactRef:
        try:
            return self.input_artifacts[name]
        except KeyError as error:
            available = ", ".join(sorted(self.input_artifacts)) or "none"
            raise KeyError(f"input artifact {name!r} is unavailable; materialized inputs: {available}") from error
