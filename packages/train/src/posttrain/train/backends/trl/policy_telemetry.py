"""Telemetry owned by the TRL online policy-optimization adapter."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Any, cast

from posttrain.common import RunContext

from ...grpo_observations import GRPOObservationFeatures, normalize_grpo_metrics
from .policy_probe import PostUpdateProbe


class ActorUpdateTelemetry:
    """Own one bounded actor-update phase between rollout and optimizer step."""

    def __init__(self, context: RunContext) -> None:
        self._context = context
        self._phase: AbstractContextManager[str] | None = None
        self._started_at: float | None = None
        self._optimizer_step: int | None = None
        self._completed_durations: dict[int, float] = {}
        self._last_token_count = 0.0
        # Measured parts of the active phase: micro-step forward/backward and the optimizer step.
        self._parts: dict[str, float] = {}

    def add_part(self, name: str, seconds: float) -> None:
        if self._phase is not None:
            self._parts[name] = self._parts.get(name, 0.0) + seconds

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
            for name, seconds in self._parts.items():
                self._context.metric(f"train/rl/time/{name}_seconds", seconds, step=optimizer_step)
        self._parts = {}

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


def actor_update_callback_type(
    imports: Mapping[str, Any], telemetry: ActorUpdateTelemetry, probe: PostUpdateProbe | None = None
) -> type[Any]:
    parent = imports["TrainerCallback"]

    class ActorUpdateCallback(parent):
        _optimizer_started_at: float | None = None

        def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            del args, state
            _synchronize_cuda()
            self._optimizer_started_at = time.perf_counter()
            return control

        def on_optimizer_step(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            del args, state
            if self._optimizer_started_at is not None:
                _synchronize_cuda()
                telemetry.add_part("optimizer_step", time.perf_counter() - self._optimizer_started_at)
                self._optimizer_started_at = None
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            del args
            telemetry.complete(int(state.global_step))
            if probe is not None:
                probe.finish(skipped=bool(probe.trainer.accelerator.optimizer_step_was_skipped))
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


def actor_update_trainer_type(
    parent: type[Any], telemetry: ActorUpdateTelemetry, probe: PostUpdateProbe | None = None
) -> type[Any]:
    """Start actor telemetry after TRL prepares the retained rollout batch."""

    class ActorUpdateTrainer(parent):
        _microstep_started_at: float | None = None

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._posttrain_sampler_gap = SamplerGapAccumulator()
            self._posttrain_probe_inputs: dict[str, Any] | None = None
            super().__init__(*args, **kwargs)
            if probe is not None:
                if self.accelerator.num_processes != 1 or self.args.use_liger_kernel:
                    raise ValueError("post-update probing requires one process and use_liger_kernel=False")
                import torch

                if any(isinstance(module, torch.nn.Dropout) and module.p > 0 for module in self.model.modules()):
                    raise ValueError("post-update probing requires dropout disabled")
                probe.trainer = self

        def _compute_loss(self, model: Any, inputs: dict[str, Any]) -> Any:
            self._posttrain_probe_inputs = inputs if probe is not None and self.model.training else None
            try:
                return super()._compute_loss(model, inputs)
            finally:
                self._posttrain_probe_inputs = None

        def _get_per_token_logps_and_entropies(self, *args: Any, **kwargs: Any) -> Any:
            result = super()._get_per_token_logps_and_entropies(*args, **kwargs)
            if probe is not None and self._posttrain_probe_inputs is not None:
                probe.capture(self._posttrain_probe_inputs, result[0])
            return result

        def _prepare_inputs(self, generation_batch: dict[str, Any]) -> dict[str, Any]:
            prepared = super()._prepare_inputs(generation_batch)
            if self.model.training:
                telemetry.ensure_started(int(self.state.global_step) + 1)
                # The first micro-step's inputs include the rollout; time forward/backward from here.
                _synchronize_cuda()
                self._microstep_started_at = time.perf_counter()
            return cast(dict[str, Any], prepared)

        def _prepare_active_sampling_inputs(self, candidate_inputs: Any) -> Any:
            try:
                return super()._prepare_active_sampling_inputs(candidate_inputs)
            except RuntimeError as error:
                rejections = getattr(self, "_posttrain_admission_rejections", ())
                if rejections:
                    error.add_note(f"rollout admission rejected groups: {sorted(rejections)}")
                raise

        def _vllm_importance_sampling_ratio(self, actor_logps: Any, sampling_logps: Any, mask: Any) -> Any:
            result = super()._vllm_importance_sampling_ratio(actor_logps, sampling_logps, mask)
            if self.model.training:
                # Called once per generation batch, or once per micro-batch when the
                # ratio comes from the training forward; pooled until the update logs.
                self._posttrain_sampler_gap.add(actor_logps, sampling_logps, mask)
            return result

        def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
            if getattr(self, "use_precomputed_advantages", False):
                # Native GRPO logs these before replacing them with supplied credit.
                # Scorable/truncated fractions remain valid population diagnostics.
                mode = "train" if self.model.training else "eval"
                for statistic in ("mean", "std", "abs_mean", "positive_fraction", "negative_fraction", "zero_fraction"):
                    self._metrics[mode].pop(f"advantages/{statistic}", None)
            if self.model.training:
                for name, value in self._posttrain_sampler_gap.flush().items():
                    self._metrics["train"][name].append(value)
            super().log(logs, start_time)

        def training_step(self, *args: Any, **kwargs: Any) -> Any:
            loss = super().training_step(*args, **kwargs)
            if self._microstep_started_at is not None:
                _synchronize_cuda()
                telemetry.add_part("actor_forward_backward", time.perf_counter() - self._microstep_started_at)
                self._microstep_started_at = None
            return loss

    return ActorUpdateTrainer


class SamplerGapAccumulator:
    """Pool the trainer-sampler log-probability gap over one logged update.

    TRL reports the mean and maximum per-token gap; the 99th percentile shows
    whether the tail that the importance-sampling cap truncates moved, and the
    mean absolute per-sequence sum is the sequence-level log-ratio that
    sequence importance sampling (SAMPO) uses.  Both are computed over every
    token and sequence of the update, however TRL splits it into calls.
    Tokens whose sampler log-probability is unavailable (NaN) are excluded, as
    in TRL.  Statistics are per process.
    """

    def __init__(self) -> None:
        self._tokens: list[Any] = []
        self._sequences: list[Any] = []

    def add(self, actor_logps: Any, sampling_logps: Any, mask: Any) -> None:
        import torch

        difference = (actor_logps.detach().float() - sampling_logps.detach().float()) * mask
        valid = mask.bool() & ~torch.isnan(difference)
        if not bool(valid.any()):
            return
        self._tokens.append(difference[valid].abs().cpu())
        self._sequences.append(torch.nan_to_num(difference, nan=0.0).sum(dim=-1)[valid.any(dim=-1)].cpu())

    def flush(self) -> dict[str, float]:
        import torch

        if not self._tokens:
            return {}
        tokens, sequences = torch.cat(self._tokens), torch.cat(self._sequences)
        self._tokens, self._sequences = [], []
        # Nearest-rank percentile: torch.quantile rejects inputs above 16M elements.
        rank = max(1, math.ceil(0.99 * tokens.numel()))
        return {
            "sampling/sampling_logp_difference/p99": float(tokens.kthvalue(rank).values),
            "sampling/sequence_logp_difference/abs_mean": float(sequences.abs().mean()),
        }


def _synchronize_cuda() -> None:
    """Wait for queued GPU work so wall-clock timings measure it, not its launch."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


# TRL computes these from its own scalar group advantages before a rollout
# function's precomputed advantages replace them. For SAMPO they describe credit
# the policy is never trained on, and they collide with the hierarchical
# credit statistics SAMPO reports under the same names at the same step.
SAMPO_SUPERSEDED_NATIVE_METRICS = frozenset(
    {
        "advantages/mean",
        "advantages/std",
        "advantages/abs_mean",
        "advantages/positive_fraction",
        "advantages/negative_fraction",
        "advantages/zero_fraction",
    }
)


def native_trainer_record(native: Mapping[str, object], *, resolved: bool, sampo: bool) -> dict[str, object]:
    """Drop native values whose meaning another posttrain writer owns at this step.

    The resolved engine reports its own objective loss, so native windowed loss
    is dropped. Ordinary SAMPO reports its supplied advantages, so TRL's
    unused scalar-advantage statistics are dropped.
    """
    dropped = {"loss"} if resolved else SAMPO_SUPERSEDED_NATIVE_METRICS if sampo else frozenset()
    return {key: value for key, value in native.items() if key not in dropped}


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
    "native_trainer_record",
    "normalize_live_metrics",
    "SAMPO_SUPERSEDED_NATIVE_METRICS",
    "SamplerGapAccumulator",
]
