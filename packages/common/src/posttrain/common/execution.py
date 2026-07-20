"""Framework-neutral execution and observation contracts."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .artifacts import JsonValue, ProducedArtifact
from .errors import OperationCancelled
from .jobs import Invocation, Job, JobAction, RunAttempt

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
class TraceObservation:
    trace_type: str
    external_id: str
    payload: Mapping[str, JsonValue]
    attributes: Attributes = field(default_factory=dict)


class Observer(Protocol):
    def event(self, observation: EventObservation) -> None: ...

    def metric(self, observation: MetricObservation) -> None: ...

    def trace(self, observation: TraceObservation) -> None: ...

    def artifact(self, artifact: ProducedArtifact) -> None: ...


class NullObserver:
    """Observer used when reusable operations run without a host platform."""

    def event(self, observation: EventObservation) -> None:
        del observation

    def metric(self, observation: MetricObservation) -> None:
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
class ExecutionContext:
    job: Job
    action: JobAction
    invocation: Invocation
    attempt: RunAttempt
    workspace: Path
    observer: Observer = field(default_factory=NullObserver)
    cancellation: CancellationToken = field(default_factory=CancellationToken)
    clock: Clock = utc_now

    def __post_init__(self) -> None:
        if self.action.job_id != self.job.id:
            raise ValueError("action job_id must match the execution job")
        if not self.workspace.is_absolute():
            raise ValueError("execution workspace must be absolute")

    def event(self, name: str, attributes: Attributes | None = None) -> None:
        self.cancellation.raise_if_cancelled()
        self.observer.event(EventObservation(name, self.clock(), attributes or {}))

    def metric(
        self,
        name: str,
        value: float,
        *,
        step: int | None = None,
        attributes: Attributes | None = None,
    ) -> None:
        self.cancellation.raise_if_cancelled()
        self.observer.metric(MetricObservation(name, float(value), step, attributes or {}))

    def trace(self, observation: TraceObservation) -> None:
        self.cancellation.raise_if_cancelled()
        self.observer.trace(observation)

    def artifact(self, artifact: ProducedArtifact) -> None:
        self.cancellation.raise_if_cancelled()
        self.observer.artifact(artifact)
