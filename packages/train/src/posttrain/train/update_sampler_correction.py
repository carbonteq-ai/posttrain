"""Detached sampler/trainer correction on complete native action support.

This ratio is separate from current/old PPO ratios and their clipping. Sequence
correction uses the product over original eligible episode actions, even when an
optimizer occurrence selects only a turn or span. Packing never defines scope.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from .profiles import CAPOSettings, GDPOSettings, GRPOSettings, SAMPOSettings
from .update_records import InvalidPolicyUpdate, PopulationSnapshot, frozen_array

type SamplerCorrectionMode = Literal["token_truncate", "token_mask", "sequence_truncate", "sequence_mask"]


def sampler_correction_recipe(
    settings: GRPOSettings | SAMPOSettings | GDPOSettings | CAPOSettings,
) -> tuple[SamplerCorrectionMode, float | None, float | None]:
    """The (mode, lower, upper) correction recipe a typed selection declares."""
    if isinstance(settings, GRPOSettings | SAMPOSettings):
        return (
            settings.importance_sampling_mode,
            settings.importance_sampling_clip_min,
            settings.importance_sampling_clip_max,
        )
    if isinstance(settings, GDPOSettings | CAPOSettings):
        return "token_truncate", None, 3.0
    raise InvalidPolicyUpdate("unsupported sampler correction recipe")


def recipe_sampler_correction_weights(
    settings: GRPOSettings | SAMPOSettings | GDPOSettings | CAPOSettings,
    snapshot: PopulationSnapshot,
    old: np.ndarray,
    sampled: np.ndarray,
) -> np.ndarray:
    """Apply the normalizer's selected recipe consistently in either backend."""
    mode, lower, upper = sampler_correction_recipe(settings)
    return sampler_correction_weights(snapshot, old, sampled, mode=mode, lower=lower, upper=upper)


def sampler_correction_weights(
    snapshot: PopulationSnapshot,
    old: np.ndarray,
    sampled: np.ndarray,
    *,
    mode: Literal["token_truncate", "token_mask", "sequence_truncate", "sequence_mask"],
    lower: float | None,
    upper: float | None,
) -> np.ndarray:
    """Freeze exp(old-sampler) per position, with explicitly selected truncate/mask bounds.

    Inputs are log scores per population position, scored under the same
    declared temperature/distribution contract by the caller. This helper
    cannot authenticate model or sampler provenance. Bounds are applied in log
    space before exp, avoiding overflow without an implicit safety cap.
    Sequence modes use the sum over the episode's complete eligible support.
    """
    if mode not in {"token_truncate", "token_mask", "sequence_truncate", "sequence_mask"}:
        raise InvalidPolicyUpdate("unsupported sampler correction mode")
    bounds = (lower, upper)
    if any(
        value is not None and (type(value) not in (float, int) or not math.isfinite(value) or value <= 0)
        for value in bounds
    ) or (lower is not None and upper is not None and lower >= upper):
        raise InvalidPolicyUpdate("sampler correction bounds must be positive finite and ordered")
    if any(np.asarray(scores).dtype.kind not in "fiu" for scores in (old, sampled)):
        raise InvalidPolicyUpdate("sampler correction requires detached finite log scores")
    old, sampled = np.asarray(old, dtype=np.float64), np.asarray(sampled, dtype=np.float64)
    if old.shape != (snapshot.size,) or sampled.shape != (snapshot.size,):
        raise InvalidPolicyUpdate("sampler correction requires exact complete population score support")
    if not (np.isfinite(old).all() and np.isfinite(sampled).all()):
        raise InvalidPolicyUpdate("sampler correction requires detached finite log scores")
    deltas = old - sampled
    if not np.isfinite(deltas).all():
        raise InvalidPolicyUpdate("sampler correction log difference is not finite")
    if mode.startswith("sequence"):
        totals = np.zeros(len(snapshot.episodes), dtype=np.float64)
        np.add.at(totals, snapshot.episode_of, deltas)
        if not np.isfinite(totals).all():
            raise InvalidPolicyUpdate("sampler correction sequence log difference is not finite")
        deltas = totals[snapshot.episode_of]
    log_lower = -math.inf if lower is None else math.log(lower)
    log_upper = math.inf if upper is None else math.log(upper)
    below, above = deltas < log_lower, deltas > log_upper
    with np.errstate(over="ignore"):
        weights = np.exp(np.clip(deltas, log_lower, log_upper) if mode.endswith("truncate") else deltas)
    if mode.endswith("mask"):
        weights[below | above] = 0.0
    else:
        if lower is not None:
            weights[below] = lower
        if upper is not None:
            weights[above] = upper
    if not np.isfinite(weights).all():
        raise InvalidPolicyUpdate("uncapped sampler correction is not finite")
    return frozen_array(weights, np.float64)
