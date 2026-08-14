"""Framework-neutral execution and observation contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol

from .artifacts import JsonValue, LocalArtifactRef, ProducedArtifact
from .errors import ContractError, OperationCancelled
from .selections import validate_selection_id

Attributes = Mapping[str, JsonValue]
Clock = Callable[[], datetime]
FactScalar = str | int | float | bool | None
FactState = Literal["complete", "partial", "unsupported"]
FactMeasure = int | float | None
SignalSourceKind = Literal[
    "llm_judge",
    "deterministic",
    "human",
    "environment",
    "group",
    "teacher",
    "composite",
    "unknown",
]

_FACT_NAME = re.compile(r"^[a-z][a-z0-9]*(?:[._/-][a-z0-9]+)*$")
_SIGNAL_SOURCE_KINDS = frozenset(
    {
        "llm_judge",
        "deterministic",
        "human",
        "environment",
        "group",
        "teacher",
        "composite",
        "unknown",
    }
)


def _validate_fact_name(value: str, field_name: str) -> None:
    if not _FACT_NAME.fullmatch(value):
        raise ContractError(f"{field_name} must be a lowercase stable identifier, got {value!r}")


def _validate_finite_measure(value: FactMeasure, field_name: str) -> None:
    if isinstance(value, bool) or (value is not None and not isinstance(value, (int, float))):
        raise ContractError(f"{field_name} must be numeric")
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(f"{field_name} must be finite")


def _bounded_text(value: str, field_name: str, *, maximum: int = 256) -> None:
    if not value.strip() or len(value) > maximum or any(ord(char) < 32 for char in value):
        raise ContractError(f"{field_name} must be non-empty bounded text")


@dataclass(frozen=True, slots=True)
class SignalSource:
    """Lightweight semantic provenance for one scored signal, never an execution record."""

    kind: SignalSourceKind = "unknown"
    id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _SIGNAL_SOURCE_KINDS:
            raise ContractError(f"unknown signal source kind {self.kind!r}")
        if self.id is not None:
            _bounded_text(self.id, "signal source id")


@dataclass(frozen=True, slots=True)
class TraceRewardComponent:
    """One named reward contribution with optional raw score, weight, and source."""

    name: str
    contribution: FactMeasure
    score: FactMeasure = None
    weight: FactMeasure = None
    source: SignalSource = field(default_factory=SignalSource)

    def __post_init__(self) -> None:
        _bounded_text(self.name, "trace reward component name")
        _validate_finite_measure(self.contribution, f"trace reward component {self.name!r} contribution")
        _validate_finite_measure(self.score, f"trace reward component {self.name!r} score")
        _validate_finite_measure(self.weight, f"trace reward component {self.name!r} weight")
        if not isinstance(self.source, SignalSource):
            raise ContractError("trace reward component source must be SignalSource")
        if self.contribution is not None and self.score is not None and self.weight is not None:
            expected = float(self.score) * float(self.weight)
            if not math.isclose(float(self.contribution), expected, rel_tol=1e-9, abs_tol=1e-12):
                raise ContractError("trace reward component contribution must equal score * weight")


@dataclass(frozen=True, slots=True)
class TraceFactSet:
    """Versioned scalar facts projected from one native trace.

    The envelope is provider-neutral. The integration that understands the
    native trace owns calculation; tracking providers only persist and query
    the supplied dimensions and measures.
    """

    namespace: str
    calculator_version: str
    dimensions: Mapping[str, FactScalar] = field(default_factory=dict)
    measures: Mapping[str, FactMeasure] = field(default_factory=dict)
    reward_components: tuple[TraceRewardComponent, ...] = ()
    provenance: Mapping[str, str] = field(default_factory=dict)
    state: FactState = "complete"

    def __post_init__(self) -> None:
        _validate_fact_name(self.namespace, "trace fact namespace")
        if not self.calculator_version.strip():
            raise ContractError("trace fact calculator version is required")
        for name, value in self.dimensions.items():
            _validate_fact_name(name, "trace fact dimension")
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ContractError(f"trace fact dimension {name!r} must be scalar")
            if isinstance(value, float) and not math.isfinite(value):
                raise ContractError(f"trace fact dimension {name!r} must be finite")
        for name, value in self.measures.items():
            _validate_fact_name(name, "trace fact measure")
            _validate_finite_measure(value, f"trace fact measure {name!r}")
        component_names = [component.name for component in self.reward_components]
        if len(set(component_names)) != len(component_names):
            raise ContractError("trace reward component names must be unique")
        if any(not isinstance(component, TraceRewardComponent) for component in self.reward_components):
            raise ContractError("trace reward components must be TraceRewardComponent values")
        for name, value in self.provenance.items():
            _validate_fact_name(name, "trace fact provenance field")
            if not value.strip():
                raise ContractError(f"trace fact provenance {name!r} cannot be empty")
        object.__setattr__(self, "dimensions", MappingProxyType(dict(self.dimensions)))
        object.__setattr__(self, "measures", MappingProxyType(dict(self.measures)))
        object.__setattr__(self, "reward_components", tuple(sorted(self.reward_components, key=lambda item: item.name)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def projection_id(self) -> str:
        """Deterministic identity for this immutable source projection."""

        payload = {
            "namespace": self.namespace,
            "calculator_version": self.calculator_version,
            "dimensions": dict(sorted(self.dimensions.items())),
            "measures": dict(sorted(self.measures.items())),
            "reward_components": [
                {
                    "name": component.name,
                    "contribution": component.contribution,
                    "score": component.score,
                    "weight": component.weight,
                    "source": {"kind": component.source.kind, "id": component.source.id},
                }
                for component in self.reward_components
            ],
            "provenance": dict(sorted(self.provenance.items())),
            "state": self.state,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


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
    facts: tuple[TraceFactSet, ...] = ()


@dataclass(frozen=True, slots=True)
class TraceFactUpdateObservation:
    """A later scalar fact projection keyed to an already-observed trace."""

    trace_type: str
    external_id: str
    facts: TraceFactSet
    attributes: Attributes = field(default_factory=dict)

    def __post_init__(self) -> None:
        _bounded_text(self.trace_type, "trace fact update trace type", maximum=64)
        _bounded_text(self.external_id, "trace fact update external id", maximum=768)
        if self.facts.reward_components:
            raise ContractError("trace fact updates cannot replace reward components")


class Observer(Protocol):
    def event(self, observation: EventObservation) -> None: ...

    def metric(self, observation: MetricObservation) -> None: ...

    def metrics(self, observation: MetricBatchObservation) -> None: ...

    def trace(self, observation: TraceObservation) -> None: ...

    def trace_fact_update(self, observation: TraceFactUpdateObservation) -> None: ...

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

    def trace_fact_update(self, observation: TraceFactUpdateObservation) -> None:
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
        active interval, so an outer ``training`` phase can contain a shorter
        ``rollout`` phase without double-counting the sample. A specific phase
        name must still describe its complete interval; for example,
        ``actor_update`` must not wrap rollout generation.
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

    def trace_fact_update(self, observation: TraceFactUpdateObservation) -> None:
        self.cancellation.raise_if_cancelled()
        self.observer.trace_fact_update(replace(observation, attributes=self._attributes(observation.attributes)))

    def artifact(self, artifact: ProducedArtifact) -> None:
        self.cancellation.raise_if_cancelled()
        self.observer.artifact(replace(artifact, metadata=self._attributes(artifact.metadata)))

    def input_artifact(self, name: str) -> LocalArtifactRef:
        try:
            return self.input_artifacts[name]
        except KeyError as error:
            available = ", ".join(sorted(self.input_artifacts)) or "none"
            raise KeyError(f"input artifact {name!r} is unavailable; materialized inputs: {available}") from error
