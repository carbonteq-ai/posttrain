"""Runtime mechanics of the TRL trainer precision selections.

Float16 training uses PyTorch's dynamic loss scaling: the loss is multiplied by
a scale before the backward pass so small gradients do not underflow, the
gradients are divided by it before the optimizer step, and when any gradient is
infinite or NaN the step is skipped and the scale halved.  One optimizer step
consumes one whole rollout batch in online RL, so every skipped step wastes a
batch; ``LossScaleMonitor`` records the scale and every skip as metrics.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

from posttrain.common import JsonValue, RunContext

from ...precision import logits_float32, training_precision

type MetricNormalizer = Callable[[int, Mapping[str, object]], Mapping[str, float]]


class LossScaleMonitor:
    """Record the dynamic loss scale and skipped optimizer steps of a float16 run.

    A step is skipped when the scaler found an infinite or NaN gradient. Accelerate
    reports that as ``step_was_skipped`` only for an optimizer the scaler does not
    call on overflow; a fused optimizer (Transformers' default ``adamw_torch_fused``)
    is always called and skips the update inside its kernel, so Accelerate reports
    ``False``. The scaler halves its scale exactly when it found an overflow, so a
    scale lower after the step than before it is the robust skip signal.
    """

    def __init__(self, context: RunContext) -> None:
        self._context = context
        self.optimizer_steps = 0
        self.skipped_steps = 0
        self.last_step_skipped = False
        self._scale_before_step: float | None = None

    def before_step(self, optimizer: Any) -> None:
        """Remember the loss scale before one optimizer step (a no-op without loss scaling)."""

        scaler = getattr(optimizer, "scaler", None)
        self._scale_before_step = float(scaler.get_scale()) if scaler is not None else None

    def observe(self, optimizer: Any, global_step: int) -> None:
        """Read the scaler after one optimizer step (a no-op without loss scaling)."""

        scaler = getattr(optimizer, "scaler", None)
        before, self._scale_before_step = self._scale_before_step, None
        if scaler is None:
            return
        scale = float(scaler.get_scale())
        skipped = bool(getattr(optimizer, "step_was_skipped", False)) or (before is not None and scale < before)
        self.optimizer_steps += 1
        self.last_step_skipped = skipped
        if skipped:
            self.skipped_steps += 1
            self._context.event(
                "optimizer_step_skipped",
                {"global_step": global_step, "loss_scale": scale, "reason": "non-finite gradients"},
            )
        self._context.metrics(
            {
                "train/loss_scale": scale,
                "train/optimizer_step_skipped": float(skipped),
                "train/optimizer_steps_skipped": float(self.skipped_steps),
            },
            step=global_step,
        )

    def finite_grad_norm(self, normalizer: MetricNormalizer) -> MetricNormalizer:
        """Drop the infinite gradient norm of a step the scaler skipped, before normalization.

        Transformers logs the unscaled gradient norm, which is infinite exactly
        when the scaler skips the step; that is expected loss-scaling behaviour,
        recorded by ``train/optimizer_step_skipped``, not a training failure.
        A non-finite norm on a step that was not skipped still fails the run.
        """

        def normalize(step: int, native: Mapping[str, object]) -> Mapping[str, float]:
            grad_norm = native.get("grad_norm")
            if (
                self.last_step_skipped
                and isinstance(grad_norm, int | float)
                and not isinstance(grad_norm, bool)
                and not math.isfinite(float(grad_norm))
            ):
                native = {name: value for name, value in native.items() if name != "grad_norm"}
            return normalizer(step, native)

        return normalize


def apply_initial_loss_scale(trainer: Any, initial_scale: float | None) -> None:
    """Start the trainer's fp16 GradScaler at ``initial_scale`` (a no-op without fp16).

    Accelerate creates the scaler with PyTorch's default 65536 when the trainer
    is built; the scale tensor is created lazily at the first scaled backward
    pass from ``_init_scale``, which is set here. A scaler that already has a
    scale (restored from a checkpoint) keeps it.
    """

    if initial_scale is None:
        return
    scaler = getattr(getattr(trainer, "accelerator", None), "scaler", None)
    if scaler is None:
        raise RuntimeError("fp16 training has no gradient scaler to start at the selected loss scale")
    if getattr(scaler, "_scale", None) is None:
        scaler._init_scale = float(initial_scale)


def loss_scale_callback_type(imports: Mapping[str, Any], monitor: LossScaleMonitor) -> type[Any]:
    parent = imports["TrainerCallback"]

    class LossScaleCallback(parent):
        def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, state
            optimizer = kwargs.get("optimizer")
            if optimizer is not None:
                monitor.before_step(optimizer)
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args
            optimizer = kwargs.get("optimizer")
            if optimizer is not None:
                monitor.observe(optimizer, int(state.global_step))
            return control

    return LossScaleCallback


def upcast_logits_to_float32(model: Any) -> Any:
    """Cast the language-model head's output to float32 before any log-softmax.

    The logits are still produced by the head in the model dtype; this removes
    the second rounding of the log-probabilities themselves, so the trainer's
    log-probabilities follow vLLM's sampler, which also takes model-dtype
    logits to float32 before its log-softmax.  Returns the hook handle.
    """

    head = model.get_output_embeddings()
    if head is None:
        raise ValueError("logits_float32 requires a model with an output embedding (language-model head)")

    def to_float32(_module: Any, _inputs: Any, output: Any) -> Any:
        return output.float()

    return head.register_forward_hook(to_float32)


def require_float32_trainable_parameters(model: Any) -> None:
    """Fail before training when a float16 run would step non-float32 parameters.

    The gradient scaler refuses to unscale float16 gradients, and stepping
    float16 weights loses small updates.  PEFT keeps LoRA adapters in float32
    over a float16 base both for a fresh adapter and for one resumed with
    ``is_trainable=True``; this check makes that invariant explicit.
    """

    wrong = [
        f"{name} ({parameter.dtype})"
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and str(parameter.dtype) != "torch.float32"
    ]
    if wrong:
        shown = ", ".join(wrong[:3]) + (f" and {len(wrong) - 3} more" if len(wrong) > 3 else "")
        raise ValueError(f"training_precision fp16 requires float32 trainable parameters; found {shown}")


def require_default_precision(backend_options: Mapping[str, JsonValue], technique: str) -> None:
    """Reject precision selections that only online RL implements."""

    if training_precision(backend_options) != "bf16" or logits_float32(backend_options):
        raise ValueError(
            f"training_precision fp16 and logits_float32 are qualified for TRL online RL only, not {technique}"
        )


__all__ = [
    "LossScaleMonitor",
    "apply_initial_loss_scale",
    "loss_scale_callback_type",
    "require_default_precision",
    "require_float32_trainable_parameters",
    "upcast_logits_to_float32",
]
