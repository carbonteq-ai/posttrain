"""Typed results returned by evaluation operations."""

from __future__ import annotations

from dataclasses import dataclass

from posttrain.common import ProducedArtifact


@dataclass(frozen=True, slots=True)
class EvaluationPopulation:
    """Irreducible rollout counts retained at evaluation-run grain."""

    attempted: int
    complete: int
    failed: int
    truncated: int
    coverage_missing: int
    model_call_error_rollouts: int = 0
    model_call_http_400_rollouts: int = 0
    context_overflow_rollouts: int = 0

    def __post_init__(self) -> None:
        values = (
            self.attempted,
            self.complete,
            self.failed,
            self.truncated,
            self.coverage_missing,
            self.model_call_error_rollouts,
            self.model_call_http_400_rollouts,
            self.context_overflow_rollouts,
        )
        if any(value < 0 for value in values):
            raise ValueError("evaluation population counts must be non-negative")
        if any(
            value > self.attempted
            for value in (
                self.complete,
                self.failed,
                self.truncated,
                self.model_call_error_rollouts,
                self.model_call_http_400_rollouts,
                self.context_overflow_rollouts,
            )
        ):
            raise ValueError("evaluation outcome counts cannot exceed attempted rollouts")
        if self.model_call_http_400_rollouts > self.model_call_error_rollouts:
            raise ValueError("HTTP 400 rollouts cannot exceed model-call error rollouts")
        if self.context_overflow_rollouts > self.model_call_http_400_rollouts:
            raise ValueError("context-overflow rollouts cannot exceed HTTP 400 rollouts")


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
    population: EvaluationPopulation

    @property
    def rollout_count(self) -> int:
        return len(self.trace_ids)

    @property
    def status(self) -> str:
        if (
            self.synchronization.complete
            and self.population.coverage_missing == 0
            and self.population.failed == 0
            and self.population.truncated == 0
        ):
            return "complete"
        return "partial"


__all__ = ["EvaluationPopulation", "EvaluationResult", "TraceSynchronization"]
