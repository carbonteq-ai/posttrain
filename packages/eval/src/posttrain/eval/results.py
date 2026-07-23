"""Typed results returned by evaluation operations."""

from __future__ import annotations

from dataclasses import dataclass

from posttrain.common import ProducedArtifact


@dataclass(frozen=True, slots=True)
class TraceSynchronization:
    observed: int
    emitted: int
    invalid: int
    failed_batches: int
    unsynchronized: int
    errors: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return self.invalid == 0 and self.unsynchronized == 0

    @property
    def status(self) -> str:
        return "complete" if self.complete else "partial"


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    plan_id: str
    environment_id: str
    model_id: str
    trace_ids: tuple[str, ...]
    native_artifact: ProducedArtifact
    synchronization: TraceSynchronization

    @property
    def rollout_count(self) -> int:
        return len(self.trace_ids)

    @property
    def status(self) -> str:
        return self.synchronization.status


__all__ = ["EvaluationResult", "TraceSynchronization"]
