"""Online-RL settings a training backend does not implement.

A setting that a backend never reads must be rejected, not silently ignored:
the run would otherwise train with different behaviour than its settings and
recorded configuration say.
"""

from __future__ import annotations

import math

from .profiles import GRPOSettings, TrainingLoop


def verl_grpo_settings_problem(settings: GRPOSettings) -> str | None:
    """Explain a GRPO setting veRL cannot reproduce exactly as TRL runs it, or None.

    The CarbonTeq veRL fork reproduces TRL's objective, sampler correction (every
    mode and bound), advantage scaling, truncated-completion masking, group
    admission, candidate-batch DAPO dynamic sampling (including the curriculum's
    pool decision and batch-std scaling per candidate batch) and learning-rate
    schedules; the worker requires a fork revision that provides them. No GRPO
    setting is rejected.
    """

    del settings
    return None


# Transformers schedule names the CarbonTeq veRL fork reproduces exactly. Transformers'
# "constant" never warms up, so it maps to veRL constant with zero warmup steps;
# "constant_with_warmup" maps to veRL constant with the same ceil(max_steps *
# warmup_ratio) warmup steps the TRL backend uses; "linear" maps to the fork's
# Transformers-identical linear schedule (warmup, then linear decay to zero).
VERL_LR_SCHEDULES: frozenset[str] = frozenset({"constant", "constant_with_warmup", "linear"})


def verl_warmup_steps(loop: TrainingLoop) -> int:
    """Warmup steps veRL runs for a supported schedule, matching Transformers."""

    if loop.lr_scheduler_type == "constant":
        return 0
    # "constant_with_warmup" and "linear" warm up over ceil(max_steps * warmup_ratio) steps.
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
