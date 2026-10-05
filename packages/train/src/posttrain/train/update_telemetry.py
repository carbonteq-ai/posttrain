"""Backend-neutral telemetry for resolved policy-update populations.

The resolved engine collects one frozen population and then applies one or more
declared optimizer updates to it. Its evidence therefore has two scopes:

* collection metrics describe the admitted population once, at the first
  update that trains on it (reward level and spread, completion shape and, for
  SAMPO, the episode/turn credit hierarchy);
* update metrics describe the actions one applied optimizer update selected
  (credit support, PPO-ratio clipping, sampler-correction weights, entropy).

Both reuse the metric names the ordinary TRL/veRL paths already report, so
Observatory reads resolved and ordinary runs with one job definition. Nothing
here imports a trainer, tensor library or tracking provider.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from .online_rl import EnvironmentRollout
from .profiles import CAPOSettings, GDPOSettings, GRPOSettings, SAMPOSettings, shape_online_reward
from .sampo_advantages import ZERO_CREDIT_TOLERANCE, compute_sampo_advantages
from .update_credit import PreparedCredit
from .update_objectives import ResolvedObjectiveTerm

# TRL treats a group as zero-spread, and an advantage as zero, when
# torch.isclose(value, 0) holds; with default tolerances that is |value| <= 1e-8.
_ZERO = ZERO_CREDIT_TOLERANCE

SAMPO_CREDIT_ESTIMATOR_PREFIX = "sampo-credit@"


def _sample_std(values: Sequence[float]) -> float:
    """Bessel-corrected standard deviation, matching TRL's logged reward_std."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def collection_metrics(
    settings: GRPOSettings | SAMPOSettings | GDPOSettings | CAPOSettings,
    rollouts: Sequence[EnvironmentRollout],
    *,
    credit_estimator_id: str,
) -> dict[str, float]:
    """One value per metric for an admitted population, over its complete groups.

    Rewards are the algorithm (shaped) rewards credit is computed from, which is
    also what the ordinary TRL path logs as reward. Groups are the admitted
    prompt groups; a fresh resolved population holds each task in one group.
    SAMPO credit evidence is recomputed with the same estimator inputs as
    `SampoCreditEstimator` and only when that estimator prepared the credit;
    an injected process-credit estimator replaces it and is not described here.
    """

    if not rollouts:
        return {}
    if isinstance(settings, GRPOSettings | SAMPOSettings):
        rewards = [
            shape_online_reward(
                settings, rollout.reward, len(rollout.completion_ids), is_truncated=rollout.is_truncated
            )
            for rollout in rollouts
        ]
    else:
        rewards = [rollout.reward for rollout in rollouts]
    groups: dict[str, list[int]] = {}
    for index, rollout in enumerate(rollouts):
        groups.setdefault(rollout.example_id, []).append(index)
    group_stds = [_sample_std([rewards[index] for index in members]) for members in groups.values()]
    lengths = [sum(rollout.env_mask) for rollout in rollouts]
    values: dict[str, float] = {
        "train/rl/reward_mean": math.fsum(rewards) / len(rewards),
        "train/rl/reward_std": _sample_std(rewards),
        "train/rl/group_reward_std_mean": math.fsum(group_stds) / len(group_stds),
        "train/rl/group_zero_variance_fraction": sum(std <= _ZERO for std in group_stds) / len(group_stds),
        "train/rl/completion_tokens_mean": math.fsum(lengths) / len(lengths),
        "train/rl/completion_tokens_max": float(max(lengths)),
        "train/rl/completion_truncation_rate": sum(rollout.is_truncated for rollout in rollouts) / len(rollouts),
    }
    if isinstance(settings, SAMPOSettings) and credit_estimator_id.startswith(SAMPO_CREDIT_ESTIMATOR_PREFIX):
        # The estimator sees contiguous complete groups of shaped copies; the
        # retained native rewards stay unchanged.
        shaped = [replace(rollouts[index], reward=rewards[index]) for members in groups.values() for index in members]
        advantages = compute_sampo_advantages(settings, tuple(rollout.example_id for rollout in shaped), shaped)
        values.update(
            {name: mean for name, (mean, _) in advantages.credit_evidence(settings.step_advantage_weight).items()}
        )
    return values


def update_metrics(
    term: ResolvedObjectiveTerm,
    credit: PreparedCredit,
    *,
    episode_of: np.ndarray,
    ratios: np.ndarray,
    clipped: np.ndarray,
    sampler_correction: np.ndarray | None = None,
    correction_recipe: tuple[str, float | None, float | None] | None = None,
    entropies: np.ndarray | None = None,
    old_scores: np.ndarray | None = None,
    sampled_scores: np.ndarray | None = None,
) -> dict[str, float]:
    """Credit, clipping, correction, sampler gap and entropy over one update's selected policy actions.

    Every array is indexed by population position. `episode_of` is the
    snapshot's episode index per position. `clipped` marks the actions whose
    clipped surrogate was active and `ratios` holds their current/old ratio. A
    ratio above one was clipped at the upper bound (a positive advantage pushed
    past 1+epsilon_high, or the CISPO weight cap); one below one at the lower
    bound (a negative advantage past 1-epsilon_low). The first update on a fresh
    population evaluates at the old parameters, so clipping is non-zero only when
    several updates share one collection. Sampler-correction weights are the
    frozen detached weights multiplying each selected action's term; "clamped"
    counts weights held at a truncation bound or zeroed by a mask. With a
    sequence correction mode every action of an episode shares one weight, so
    the weight statistics count each touched episode once, as TRL reports
    sequence importance sampling. Credit within 1e-8 of zero (rounding residue
    from centring equal rewards) counts as zero, as in TRL's advantage
    statistics. Entropy is NaN at positions that were not scored.

    The sampler gap compares the frozen trainer old scores with the sampler's
    own log scores for the same original actions, which are the two inputs of
    the sampler correction. As in TRL, the token statistics are the mean, max
    and nearest-rank 99th percentile of |old - sampled| over the selected
    actions, and the sequence statistic is the mean over touched episodes of
    |sum(old - sampled)| over the episode's complete eligible support (the
    scope sequence correction uses).
    """

    selected = term.policy_positions
    if not selected.size:
        return {}
    advantages = credit.advantages[selected]
    count = int(selected.size)
    mean = math.fsum(advantages.tolist()) / count
    values: dict[str, float] = {
        "train/rl/advantage_mean": mean,
        "train/rl/advantage_abs_mean": math.fsum(np.abs(advantages).tolist()) / count,
        "train/rl/advantage_std": math.sqrt(math.fsum(((advantages - mean) ** 2).tolist()) / count),
        "train/rl/advantage_nonzero_fraction": int((np.abs(advantages) > _ZERO).sum()) / count,
        "train/rl/advantage_positive_fraction": int((advantages > _ZERO).sum()) / count,
        "train/rl/advantage_negative_fraction": int((advantages < -_ZERO).sum()) / count,
        "train/rl/advantage_zero_fraction": int((np.abs(advantages) <= _ZERO).sum()) / count,
    }
    was_clipped, ratio = clipped[selected], ratios[selected]
    values["train/rl/clip_fraction"] = int(was_clipped.sum()) / count
    values["train/rl/clip_fraction_high"] = int((was_clipped & (ratio > 1)).sum()) / count
    values["train/rl/clip_fraction_low"] = int((was_clipped & (ratio < 1)).sum()) / count
    if sampler_correction is not None:
        if correction_recipe is not None and correction_recipe[0].startswith("sequence"):
            _, first = np.unique(episode_of[selected], return_index=True)
            weights = sampler_correction[selected[np.sort(first)]].astype(np.float64)
        else:
            weights = sampler_correction[selected].astype(np.float64)
        values.update(
            {
                "train/rl/importance_sampling_ratio_mean": math.fsum(weights.tolist()) / weights.size,
                "train/rl/importance_sampling_ratio_min": float(weights.min()),
                "train/rl/importance_sampling_ratio_max": float(weights.max()),
            }
        )
        if correction_recipe is not None:
            mode, lower, upper = correction_recipe
            if mode.endswith("mask"):
                clamped = int((weights == 0.0).sum())
            else:
                held = np.zeros(weights.size, dtype=bool)
                if lower is not None:
                    held |= weights == lower
                if upper is not None:
                    held |= weights == upper
                clamped = int(held.sum())
            values["train/rl/importance_sampling_ratio_clamped_fraction"] = clamped / weights.size
    if old_scores is not None and sampled_scores is not None:
        values.update(_sampler_gap(selected, episode_of, old_scores, sampled_scores))
    if entropies is not None:
        scored = entropies[selected]
        scored = scored[np.isfinite(scored)]
        if scored.size:
            values["train/rl/entropy"] = math.fsum(scored.tolist()) / scored.size
    return values


def _sampler_gap(
    selected: np.ndarray,
    episode_of: np.ndarray,
    old: np.ndarray,
    sampled: np.ndarray,
) -> dict[str, float]:
    difference = np.asarray(old, dtype=np.float64) - np.asarray(sampled, dtype=np.float64)
    gaps = np.sort(np.abs(difference[selected]))
    # Nearest rank, as the ordinary TRL path reports its p99.
    rank = max(1, math.ceil(0.99 * gaps.size))
    touched = np.unique(episode_of[selected])
    sequences = [abs(math.fsum(difference[episode_of == episode].tolist())) for episode in touched]
    return {
        "train/rl/sampling_logp_delta_mean": math.fsum(gaps.tolist()) / gaps.size,
        "train/rl/sampling_logp_delta_max": float(gaps[-1]),
        "train/rl/sampling_logp_delta_p99": float(gaps[rank - 1]),
        "train/rl/sampling_sequence_logp_delta_abs_mean": math.fsum(sequences) / len(sequences),
    }


__all__ = ["SAMPO_CREDIT_ESTIMATOR_PREFIX", "collection_metrics", "update_metrics"]
