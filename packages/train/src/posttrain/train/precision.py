"""Numeric precision of a training job: trainer compute, rollout sampler, and log-probabilities.

The trainer and the rollout sampler (vLLM) compute the same policy's token
log-probabilities with different kernels.  Rounding in their compute dtype makes
those log-probabilities disagree, which the truncated importance-sampling (IS)
correction then has to absorb.  These selections let a job choose the
precision on both sides:

* ``backend_options.training_precision`` on a training binding: ``bf16`` (the
  default and the only behaviour before this option existed) or ``fp16``.  In
  fp16 the frozen base model is loaded in float16, the trainer runs float16
  autocast with dynamic loss scaling, and the LoRA adapter parameters stay in
  float32 (PEFT's default), so the optimizer updates float32 master weights.
* ``backend_options.logits_float32`` on a training binding: when true, the
  trainer casts the language-model head's output to float32 before the
  log-softmax, so its log-probabilities are not rounded to the model dtype.
  The default (false) keeps the existing behaviour.
* ``backend_options.fp16_initial_loss_scale`` on a training binding: the
  dynamic loss scaler's starting scale in fp16 (default 1024, only with
  ``training_precision: fp16``). PyTorch starts at 65536 and halves the scale on
  every overflow; in online RL each overflow skips a whole rollout batch, so a
  scale that starts near the working range saves the first updates. Growth
  (doubling after a run of finite steps) keeps its default.
* ``engine.dtype`` on a rollout inference binding: the vLLM compute dtype
  (``bfloat16``, ``float16`` or ``float32``).  Without it vLLM follows the
  checkpoint, except that a TurboQuant KV cache requires float16.

This module is import-light (no torch) so plans, the CLI and tests can resolve
and display the selected precision without the training runtime.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

from posttrain.common import JsonValue

type TrainingPrecision = Literal["bf16", "fp16"]
type RolloutDtype = Literal["bfloat16", "float16", "float32"]

TRAINING_PRECISIONS: tuple[TrainingPrecision, ...] = ("bf16", "fp16")
ROLLOUT_DTYPES: tuple[RolloutDtype, ...] = ("bfloat16", "float16", "float32")
_MODEL_LOAD_DTYPE: dict[TrainingPrecision, str] = {"bf16": "bfloat16", "fp16": "float16"}
_CHECKPOINT_DTYPE: dict[str, RolloutDtype] = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
DEFAULT_FP16_INITIAL_LOSS_SCALE = 1024.0
FP16_INITIAL_LOSS_SCALE_ENV = "POSTTRAIN_FP16_INITIAL_LOSS_SCALE"
VERL_LOSS_SCALE_HOOK = "posttrain.train.backends.verl.loss_scale_hook.configure_initial_loss_scale"


def training_precision(backend_options: Mapping[str, JsonValue]) -> TrainingPrecision:
    """The trainer compute precision a training binding selects (default ``bf16``)."""

    value = backend_options.get("training_precision", "bf16")
    if value == "bf16":
        return "bf16"
    if value == "fp16":
        return "fp16"
    raise ValueError(f"training backend_options.training_precision must be one of {', '.join(TRAINING_PRECISIONS)}")


def logits_float32(backend_options: Mapping[str, JsonValue]) -> bool:
    """Whether the trainer computes log-probabilities from float32 logits (default false)."""

    value = backend_options.get("logits_float32", False)
    if not isinstance(value, bool):
        raise ValueError("training backend_options.logits_float32 must be a boolean")
    return value


def fp16_initial_loss_scale(backend_options: Mapping[str, JsonValue]) -> float | None:
    """The fp16 dynamic loss scaler's starting scale (default 1024), or None without fp16."""

    value = backend_options.get("fp16_initial_loss_scale")
    if training_precision(backend_options) != "fp16":
        if value is not None:
            raise ValueError("training backend_options.fp16_initial_loss_scale requires training_precision: fp16")
        return None
    if value is None:
        return DEFAULT_FP16_INITIAL_LOSS_SCALE
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
        raise ValueError("training backend_options.fp16_initial_loss_scale must be a finite positive number")
    return float(value)


def effective_logits_float32(backend_options: Mapping[str, JsonValue]) -> bool:
    """Whether the trainer takes log-probabilities from float32 logits.

    Always under ``training_precision: fp16``: float16 log-probabilities make
    TRL's loss arithmetic overflow (``exp`` of a log-ratio above about 11 is
    infinite in float16), so float16 training computes logits, log-softmax and
    every loss term in float32. ``logits_float32: false`` is rejected there.
    Otherwise the explicit option (default false).
    """

    explicit = backend_options.get("logits_float32")
    if training_precision(backend_options) == "fp16":
        if explicit is False:
            raise ValueError(
                "training_precision fp16 computes log-probabilities from float32 logits; remove logits_float32: false"
            )
        return True
    return logits_float32(backend_options)


def model_load_dtype(precision: TrainingPrecision) -> str:
    """The dtype the frozen base weights are loaded in for one trainer precision."""

    return _MODEL_LOAD_DTYPE[precision]


def rollout_dtype(
    engine: Mapping[str, JsonValue],
) -> tuple[RolloutDtype | None, Literal["binding", "turboquant", "checkpoint"]]:
    """The vLLM compute dtype a rollout engine selects and where it came from.

    Returns ``(dtype, source)``.  ``dtype`` is what the framework passes to vLLM
    (``None`` when it passes nothing and vLLM follows the checkpoint).  A
    TurboQuant KV cache only runs in float16, so it implies float16 and rejects
    any other explicit dtype.
    """

    requested = engine.get("dtype")
    kv_cache_dtype = engine.get("kv_cache_dtype")
    turboquant = isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("turboquant_")
    if requested is None:
        if turboquant:
            return "float16", "turboquant"
        return None, "checkpoint"
    if requested not in ROLLOUT_DTYPES:
        raise ValueError(f"rollout engine dtype must be one of {', '.join(ROLLOUT_DTYPES)}")
    if turboquant and requested != "float16":
        raise ValueError("a TurboQuant KV cache requires rollout engine dtype float16")
    dtype: RolloutDtype = requested  # pyright: ignore[reportAssignmentType]
    return dtype, "binding"


def verl_rollout_dtype(
    engine: Mapping[str, JsonValue],
) -> tuple[RolloutDtype, Literal["binding", "turboquant", "backend-default"]]:
    """The vLLM dtype the veRL backend passes (it always passes one).

    veRL's rollout configuration defaults to bfloat16 regardless of the
    checkpoint, and the backend selects float16 for a TurboQuant KV cache.
    """

    dtype, source = rollout_dtype(engine)
    if dtype is None:
        return "bfloat16", "backend-default"
    return dtype, "binding" if source == "binding" else "turboquant"


@dataclass(frozen=True, slots=True)
class ResolvedPrecision:
    """The numeric precision one online-RL job runs with, as recorded and displayed."""

    training: TrainingPrecision
    model_load_dtype: str
    loss_scaling: Literal["none", "dynamic"]
    logits_float32: bool
    rollout_dtype: str | None
    rollout_dtype_source: Literal["binding", "turboquant", "checkpoint", "backend-default"] | None
    backend: Literal["trl", "verl"] = "trl"
    initial_loss_scale: float | None = None

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "training_precision": self.training,
            "model_load_dtype": self.model_load_dtype,
            "loss_scaling": self.loss_scaling,
            "logits_float32": self.logits_float32,
            "rollout_dtype": self.rollout_dtype,
            "rollout_dtype_source": self.rollout_dtype_source,
            "backend": self.backend,
            "initial_loss_scale": self.initial_loss_scale,
        }

    def without_rollout(self) -> ResolvedPrecision:
        """The same trainer precision for a job without a vLLM rollout engine."""

        return replace(self, rollout_dtype=None, rollout_dtype_source=None)

    def summary(self) -> str:
        scaling = ""
        if self.loss_scaling == "dynamic":
            start = f" from {self.initial_loss_scale:g}" if self.initial_loss_scale is not None else ""
            scaling = f", dynamic loss scaling{start}"
        if self.backend == "verl":
            compute = "float16" if self.training == "fp16" else "bfloat16"
            trainer = f"trainer {self.training} (FSDP {compute} compute over float32 master weights{scaling})"
        else:
            logits = "float32 logits" if self.logits_float32 else f"{self.model_load_dtype} logits"
            trainer = (
                f"trainer {self.training} (base weights {self.model_load_dtype}{scaling}; log-probs from {logits})"
            )
        if self.rollout_dtype is None:
            return trainer
        return f"{trainer}; rollout vLLM {self.rollout_dtype} ({self.rollout_dtype_source})"


def resolve_precision(
    backend_options: Mapping[str, JsonValue],
    engine: Mapping[str, JsonValue] | None,
    weight_precision: str,
    *,
    backend: Literal["trl", "verl"] = "trl",
) -> ResolvedPrecision:
    """Resolve the trainer and rollout precision of one training job.

    TRL loads the frozen base in the compute dtype (LoRA adapters stay
    float32); veRL's FSDP keeps float32 master weights for every parameter and
    computes in the mixed-precision dtype.
    """

    training = training_precision(backend_options)
    source: Literal["binding", "turboquant", "checkpoint", "backend-default"]
    if engine is None:
        dtype, source = None, "checkpoint"
    elif backend == "verl":
        dtype, source = verl_rollout_dtype(engine)
    else:
        dtype, source = rollout_dtype(engine)
    resolved_rollout = dtype if dtype is not None else _CHECKPOINT_DTYPE.get(weight_precision, weight_precision)
    return ResolvedPrecision(
        training=training,
        model_load_dtype="float32" if backend == "verl" else model_load_dtype(training),
        loss_scaling="dynamic" if training == "fp16" else "none",
        # TRL float16 training always takes log-probabilities from float32 logits
        # (see effective_logits_float32). veRL runs its forward and loss under
        # autocast, whose log-softmax and exp are float32 already.
        logits_float32=(
            effective_logits_float32(backend_options) if backend == "trl" else logits_float32(backend_options)
        ),
        rollout_dtype=resolved_rollout,
        rollout_dtype_source=source,
        backend=backend,
        initial_loss_scale=fp16_initial_loss_scale(backend_options),
    )


def verl_mixed_precision_overrides(backend_options: Mapping[str, JsonValue]) -> list[str]:
    """Hydra overrides selecting veRL's FSDP mixed precision (none for the bf16 default).

    veRL's FSDP engine computes in ``mixed_precision.param_dtype`` over float32
    master weights and, for float16, creates a ``ShardedGradScaler`` with
    dynamic loss scaling (growth interval 400) that skips a step whose
    gradients overflow. The actor and the reference policy use the same
    precision so the KL term compares like with like.
    """

    initial_scale = fp16_initial_loss_scale(backend_options)
    if initial_scale is None:
        return []
    policy = "{param_dtype:fp16,reduce_dtype:fp32,buffer_dtype:fp32}"
    return [
        f"+actor_rollout_ref.actor.fsdp_config.mixed_precision={policy}",
        f"+actor_rollout_ref.ref.fsdp_config.mixed_precision={policy}",
        # veRL creates its ShardedGradScaler in the Ray actor processes with a fixed
        # constructor; the worker setup hook gives it the selected starting scale.
        f"++ray_kwargs.ray_init.runtime_env.env_vars.{FP16_INITIAL_LOSS_SCALE_ENV}='{initial_scale:g}'",
        f"++ray_kwargs.ray_init.runtime_env.worker_process_setup_hook={VERL_LOSS_SCALE_HOOK}",
    ]


__all__ = [
    "DEFAULT_FP16_INITIAL_LOSS_SCALE",
    "FP16_INITIAL_LOSS_SCALE_ENV",
    "ROLLOUT_DTYPES",
    "VERL_LOSS_SCALE_HOOK",
    "TRAINING_PRECISIONS",
    "ResolvedPrecision",
    "RolloutDtype",
    "TrainingPrecision",
    "fp16_initial_loss_scale",
    "effective_logits_float32",
    "logits_float32",
    "model_load_dtype",
    "resolve_precision",
    "rollout_dtype",
    "training_precision",
    "verl_mixed_precision_overrides",
    "verl_rollout_dtype",
]
