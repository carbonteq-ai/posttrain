"""Telemetry owned by the TRL online policy-optimization adapter."""

from __future__ import annotations

import time
from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Any, cast

from posttrain.common import RunContext

from ...grpo_observations import GRPOObservationFeatures, normalize_grpo_metrics


class ActorUpdateTelemetry:
    """Own one bounded actor-update phase between rollout and optimizer step."""

    def __init__(self, context: RunContext) -> None:
        self._context = context
        self._phase: AbstractContextManager[str] | None = None
        self._started_at: float | None = None
        self._optimizer_step: int | None = None
        self._completed_durations: dict[int, float] = {}
        self._last_token_count = 0.0

    @property
    def active(self) -> bool:
        return self._phase is not None

    def start(self, optimizer_step: int) -> None:
        if optimizer_step <= 0:
            raise ValueError("actor optimizer step must be positive")
        if self._phase is not None:
            raise RuntimeError(
                f"actor update for step {self._optimizer_step} is still active; cannot start step {optimizer_step}"
            )
        phase = self._context.phase("actor_update", {"backend": "trl", "logical_step": optimizer_step})
        phase.__enter__()
        self._phase = phase
        self._started_at = time.perf_counter()
        self._optimizer_step = optimizer_step

    def ensure_started(self, optimizer_step: int) -> None:
        if self._phase is not None and self._optimizer_step == optimizer_step:
            return
        self.start(optimizer_step)

    def complete(self, optimizer_step: int) -> None:
        if self._phase is None:
            return
        if optimizer_step != self._optimizer_step:
            raise RuntimeError(f"actor update for step {self._optimizer_step} cannot complete at step {optimizer_step}")
        phase, started_at = self._reset()
        phase.__exit__(None, None, None)
        if started_at is not None:
            duration = time.perf_counter() - started_at
            self._completed_durations[optimizer_step] = duration
            self._context.metric("train/rl/time/actor_update_seconds", duration, step=optimizer_step)

    def record_tokens(self, optimizer_step: int, cumulative_tokens: float) -> None:
        completed_steps = [step for step in self._completed_durations if step <= optimizer_step]
        if not completed_steps:
            return
        duration = sum(self._completed_durations.pop(step) for step in completed_steps)
        token_delta = cumulative_tokens - self._last_token_count
        self._last_token_count = cumulative_tokens
        if duration <= 0 or token_delta < 0:
            return
        self._context.metrics(
            {
                "train/rl/actor_processed_tokens": token_delta,
                "train/rl/actor_tokens_per_second": token_delta / duration,
            },
            step=optimizer_step,
        )

    def fail(self, error: BaseException) -> None:
        if self._phase is None:
            return
        phase, _started_at = self._reset()
        phase.__exit__(type(error), error, error.__traceback__)

    def _reset(self) -> tuple[AbstractContextManager[str], float | None]:
        phase = self._phase
        if phase is None:
            raise RuntimeError("actor update telemetry is not active")
        started_at = self._started_at
        self._phase = None
        self._started_at = None
        self._optimizer_step = None
        return phase, started_at


def actor_update_callback_type(imports: Mapping[str, Any], telemetry: ActorUpdateTelemetry) -> type[Any]:
    parent = imports["TrainerCallback"]

    class ActorUpdateCallback(parent):
        def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            del args
            telemetry.complete(int(state.global_step))
            return control

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: Mapping[str, object] | None = None,
            **_: Any,
        ) -> Any:
            del args
            token_count = (logs or {}).get("num_tokens")
            if isinstance(token_count, int | float) and not isinstance(token_count, bool):
                telemetry.record_tokens(int(state.global_step), float(token_count))
            return control

    return ActorUpdateCallback


def actor_update_trainer_type(parent: type[Any], telemetry: ActorUpdateTelemetry) -> type[Any]:
    """Start actor telemetry after TRL prepares the retained rollout batch."""

    class ActorUpdateTrainer(parent):
        def _prepare_inputs(self, generation_batch: dict[str, Any]) -> dict[str, Any]:
            prepared = super()._prepare_inputs(generation_batch)
            if self.model.training:
                telemetry.ensure_started(int(self.state.global_step) + 1)
            return cast(dict[str, Any], prepared)

    return ActorUpdateTrainer


def normalize_live_metrics(
    step: int,
    native: Mapping[str, object],
    features: GRPOObservationFeatures,
) -> Mapping[str, float]:
    """Normalize one TRL actor record while preserving retained-group signal."""

    metrics = dict(normalize_grpo_metrics(backend="trl", step=step, native=native, features=features).metrics)
    metrics.pop("train/step_time_seconds", None)
    return metrics


__all__ = [
    "ActorUpdateTelemetry",
    "actor_update_callback_type",
    "actor_update_trainer_type",
    "normalize_live_metrics",
]
