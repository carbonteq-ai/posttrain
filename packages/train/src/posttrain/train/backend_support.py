"""Online-RL settings a training backend does not implement.

A setting that a backend never reads must be rejected, not silently ignored:
the run would otherwise train with different behaviour than its settings and
recorded configuration say.
"""

from __future__ import annotations

import math
from dataclasses import fields

from .profiles import GRPOSettings, TrainingLoop

# Top-level GRPOSettings fields the veRL GRPO launch plan does not pass to veRL.
# Only their defaults are accepted, so a selection cannot claim behaviour veRL
# never applies.
_VERL_GRPO_FIXED_DEFAULTS: tuple[str, ...] = (
    "advantage_scaling",
    "importance_sampling_mode",
    "importance_sampling_clip_min",
    "importance_sampling_clip_max",
    "max_admission_attempts",
)


def verl_grpo_settings_problem(settings: GRPOSettings) -> str | None:
    """Explain the first GRPO setting the veRL backend would silently ignore, or None."""

    # OLMo 3 and its active sampling are mapped natively (docs/plan/verl-vortex-port.md,
    # phases 1-2); the worker also requires a fork revision that registers them.
    if settings.adaptive_curriculum is not None:
        return "adaptive_curriculum is currently supported by the TRL backend only"
    if settings.algorithm == "olmo3":
        # GRPOSettings fixes the OLMo 3 advantage scaling and sampler correction,
        # and the veRL worker maps both; group admission retries are not used.
        return None
    defaults = {item.name: item.default for item in fields(GRPOSettings) if item.name in _VERL_GRPO_FIXED_DEFAULTS}
    changed = [name for name in _VERL_GRPO_FIXED_DEFAULTS if getattr(settings, name) != defaults[name]]
    if changed:
        selected = ", ".join(f"{name}={getattr(settings, name)!r}" for name in changed)
        return (
            f"GRPO {selected} is currently supported by the TRL backend only; the veRL backend does not "
            "receive this setting, so keep the default " + ", ".join(f"{name}={defaults[name]!r}" for name in changed)
        )
    return None


# Transformers schedule names veRL 0.9.0.post4 reproduces exactly: its FSDP
# optimizer offers only "constant" (linear warmup, then constant) and "cosine".
# Transformers' "constant" never warms up, so it maps to veRL constant with zero
# warmup steps; "constant_with_warmup" maps to veRL constant with the same
# ceil(max_steps * warmup_ratio) warmup steps the TRL backend uses.
VERL_LR_SCHEDULES: frozenset[str] = frozenset({"constant", "constant_with_warmup"})


def verl_warmup_steps(loop: TrainingLoop) -> int:
    """Warmup steps veRL runs for a supported schedule, matching Transformers."""

    if loop.lr_scheduler_type == "constant":
        return 0
    return math.ceil(loop.max_steps * loop.warmup_ratio)


def verl_training_loop_problem(loop: TrainingLoop, *, rows_per_update: int, world_size: int) -> str | None:
    """Explain the first training-loop setting veRL cannot run as selected, or None.

    veRL runs one optimizer step per update over all ``rows_per_update``
    (prompt groups x generations) rows, in micro-batches of
    ``per_device_batch_size`` rows on each of ``world_size`` devices.
    """

    if loop.lr_scheduler_type not in VERL_LR_SCHEDULES:
        return (
            f"lr_scheduler_type {loop.lr_scheduler_type!r} is not available on the veRL backend, whose optimizer "
            "schedules the learning rate as constant (optionally after linear warmup) or cosine; select "
            "'constant' or 'constant_with_warmup'"
        )
    if loop.logging_steps != 1:
        return (
            f"logging_steps {loop.logging_steps} is not available on the veRL backend, which logs every update; use 1"
        )
    if world_size < 1:
        raise ValueError("veRL world size must be positive")
    micro = loop.per_device_batch_size
    if micro * loop.gradient_accumulation_steps != rows_per_update:
        return (
            f"per_device_batch_size x gradient_accumulation_steps ({micro} x {loop.gradient_accumulation_steps}) "
            f"must equal the {rows_per_update} rows of one update (prompt groups x generations) on the veRL backend"
        )
    if rows_per_update % (micro * world_size) != 0:
        return (
            f"the {rows_per_update} rows of one update do not split into micro-batches of per_device_batch_size "
            f"{micro} on {world_size} device(s) on the veRL backend"
        )
    return None


__all__ = ["VERL_LR_SCHEDULES", "verl_grpo_settings_problem", "verl_training_loop_problem", "verl_warmup_steps"]
