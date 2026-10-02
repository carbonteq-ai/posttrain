"""Detached sampler/trainer correction on complete native action support.

This ratio is separate from current/old PPO ratios and their clipping. Sequence
correction uses the product over original eligible episode actions, even when an
optimizer occurrence selects only a turn or span. Packing never defines scope.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from .profiles import CAPOSettings, GDPOSettings, GRPOSettings, SAMPOSettings
from .update_records import ActionRef, InvalidPolicyUpdate, PopulationSnapshot


def recipe_sampler_correction_weights(
    settings: GRPOSettings | SAMPOSettings | GDPOSettings | CAPOSettings,
    snapshot: PopulationSnapshot,
    old: Mapping[ActionRef, float],
    sampled: Mapping[ActionRef, float],
) -> Mapping[ActionRef, float]:
    """Apply the normalizer's selected recipe consistently in either backend."""
    if isinstance(settings, GRPOSettings | SAMPOSettings):
        mode, lower, upper = (settings.importance_sampling_mode,
                              settings.importance_sampling_clip_min, settings.importance_sampling_clip_max)
    elif isinstance(settings, GDPOSettings | CAPOSettings):
        mode, lower, upper = "token_truncate", None, 3.0
    else:
        raise InvalidPolicyUpdate("unsupported sampler correction recipe")
    return sampler_correction_weights(snapshot, old, sampled, mode=mode, lower=lower, upper=upper)


def sampler_correction_weights(
    snapshot: PopulationSnapshot,
    old: Mapping[ActionRef, float],
    sampled: Mapping[ActionRef, float],
    *,
    mode: Literal["token_truncate", "token_mask", "sequence_truncate", "sequence_mask"],
    lower: float | None,
    upper: float | None,
) -> Mapping[ActionRef, float]:
    """Freeze exp(old-sampler), with explicitly selected truncate/mask bounds.

    Inputs must cover exactly the population, with probabilities scored under
    the same declared temperature/distribution contract by the caller. This
    helper cannot authenticate model or sampler provenance. Bounds are applied
    in log space before exp, avoiding overflow without an implicit safety cap.
    """
    if mode not in {"token_truncate", "token_mask", "sequence_truncate", "sequence_mask"}:
        raise InvalidPolicyUpdate("unsupported sampler correction mode")
    bounds = (lower, upper)
    if any(value is not None and (type(value) not in (float, int) or not math.isfinite(value) or value <= 0)
           for value in bounds) or (lower is not None and upper is not None and lower >= upper):
        raise InvalidPolicyUpdate("sampler correction bounds must be positive finite and ordered")
    actions = tuple(record.action for record in snapshot.actions)
    if set(old) != set(actions) or set(sampled) != set(actions):
        raise InvalidPolicyUpdate("sampler correction requires exact complete population score support")
    if any(type(value) not in (float, int) or not math.isfinite(value)
           for scores in (old, sampled) for value in scores.values()):
        raise InvalidPolicyUpdate("sampler correction requires detached finite log scores")
    deltas = {action: old[action] - sampled[action] for action in actions}
    if any(not math.isfinite(value) for value in deltas.values()):
        raise InvalidPolicyUpdate("sampler correction log difference is not finite")
    if mode.startswith("sequence"):
        episodes: dict[tuple[str, str], list[ActionRef]] = {}
        for action in actions:
            episodes.setdefault((action.episode_id, action.branch_id), []).append(action)
        try:
            totals = {episode: math.fsum(deltas[action] for action in members) for episode, members in episodes.items()}
        except OverflowError as error:
            raise InvalidPolicyUpdate("sampler correction sequence log difference is not finite") from error
        deltas = {action: totals[(action.episode_id, action.branch_id)] for action in actions}
    log_lower = -math.inf if lower is None else math.log(lower)
    log_upper = math.inf if upper is None else math.log(upper)
    result = {}
    for action, delta in deltas.items():
        if mode.endswith("mask") and not log_lower <= delta <= log_upper:
            result[action] = 0.0
            continue
        if mode.endswith("truncate"):
            if delta < log_lower:
                result[action] = lower
                continue
            if delta > log_upper:
                result[action] = upper
                continue
        try:
            value = math.exp(delta)
        except OverflowError as error:
            raise InvalidPolicyUpdate("uncapped sampler correction is not finite") from error
        if not math.isfinite(value):
            raise InvalidPolicyUpdate("uncapped sampler correction is not finite")
        result[action] = value
    return MappingProxyType(result)
