"""Backend-neutral SAMPO hierarchical advantage construction."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .online_rl import EnvironmentRollout
from .profiles import SAMPOSettings

_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class SAMPOAdvantages:
    """Token-aligned advantages plus evidence used to explain the update."""

    token_advantages: tuple[tuple[float, ...], ...]
    episode_advantages: tuple[float, ...]
    turn_advantages: tuple[tuple[float, ...], ...]
    anchor_group_sizes: tuple[tuple[int, ...], ...]
    used_sparse_rewards: tuple[bool, ...]


def compute_sampo_advantages(
    settings: SAMPOSettings,
    example_ids: Sequence[str],
    rollouts: Sequence[EnvironmentRollout],
) -> SAMPOAdvantages:
    """Compute GiGPO-style episode and anchor-state-relative turn advantages."""

    if len(example_ids) != len(rollouts) or not rollouts:
        raise ValueError("SAMPO example identities must align with a non-empty rollout batch")
    if len(rollouts) % settings.num_generations:
        raise ValueError("SAMPO requires complete prompt groups with exactly num_generations trajectories")
    for example_id, rollout in zip(example_ids, rollouts, strict=True):
        if example_id != rollout.example_id:
            raise ValueError("SAMPO rollout example identity does not match the requested group")
        if not math.isfinite(rollout.reward):
            raise ValueError("SAMPO requires finite trajectory rewards")
        if not rollout.turns:
            raise ValueError("SAMPO requires explicit sampled assistant-turn spans")
        covered = [False] * len(rollout.completion_ids)
        for turn in rollout.turns:
            covered[turn.completion_start : turn.completion_end] = [True] * (
                turn.completion_end - turn.completion_start
            )
        if tuple(covered) != rollout.env_mask:
            raise ValueError("SAMPO turn spans must cover every sampled policy token")
    grouped_indices = [
        list(range(start, start + settings.num_generations))
        for start in range(0, len(rollouts), settings.num_generations)
    ]
    for indices in grouped_indices:
        if len({example_ids[index] for index in indices}) != 1:
            raise ValueError("SAMPO prompt groups must be contiguous and share one example identity")

    episode = [0.0] * len(rollouts)
    for indices in grouped_indices:
        normalized = _center_and_scale(
            [rollouts[index].reward for index in indices],
            settings.advantage_normalization,
        )
        for index, value in zip(indices, normalized, strict=True):
            episode[index] = value

    returns_by_rollout: list[list[float]] = []
    sparse_flags: list[bool] = []
    for rollout in rollouts:
        explicit = [turn.step_reward for turn in rollout.turns]
        if all(value is None for value in explicit):
            rewards = [0.0] * len(explicit)
            rewards[-1] = rollout.reward
            sparse_flags.append(True)
        elif all(value is not None for value in explicit):
            rewards = []
            for value in explicit:
                assert value is not None
                rewards.append(value)
            sparse_flags.append(False)
        else:
            raise ValueError("SAMPO step rewards must be either complete or entirely absent")
        returns_by_rollout.append(_discounted_returns(rewards, settings.discount_gamma))

    anchors: dict[tuple[int, str], list[tuple[int, int, float]]] = defaultdict(list)
    for group_index, indices in enumerate(grouped_indices):
        for rollout_index in indices:
            rollout = rollouts[rollout_index]
            returns = returns_by_rollout[rollout_index]
            for turn_index, (turn, value) in enumerate(zip(rollout.turns, returns, strict=True)):
                anchors[(group_index, turn.anchor_state_key)].append((rollout_index, turn_index, value))

    turn_advantages = [[0.0] * len(rollout.turns) for rollout in rollouts]
    anchor_group_sizes = [[0] * len(rollout.turns) for rollout in rollouts]
    for members in anchors.values():
        normalized = _center_and_scale(
            [value for _, _, value in members],
            settings.advantage_normalization,
        )
        for (rollout_index, turn_index, _), value in zip(members, normalized, strict=True):
            turn_advantages[rollout_index][turn_index] = value
            anchor_group_sizes[rollout_index][turn_index] = len(members)

    token_advantages: list[tuple[float, ...]] = []
    for rollout_index, rollout in enumerate(rollouts):
        values = [0.0] * len(rollout.completion_ids)
        for turn_index, turn in enumerate(rollout.turns):
            combined = episode[rollout_index] + (
                settings.step_advantage_weight * turn_advantages[rollout_index][turn_index]
            )
            values[turn.completion_start : turn.completion_end] = [combined] * (
                turn.completion_end - turn.completion_start
            )
        token_advantages.append(tuple(values))

    return SAMPOAdvantages(
        token_advantages=tuple(token_advantages),
        episode_advantages=tuple(episode),
        turn_advantages=tuple(tuple(values) for values in turn_advantages),
        anchor_group_sizes=tuple(tuple(values) for values in anchor_group_sizes),
        used_sparse_rewards=tuple(sparse_flags),
    )


def _discounted_returns(rewards: Sequence[float], gamma: float) -> list[float]:
    result = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        result[index] = running
    return result


def _center_and_scale(values: Sequence[float], normalization: str) -> list[float]:
    mean = sum(values) / len(values)
    centered = [value - mean for value in values]
    if normalization == "mean":
        return centered
    if len(centered) == 1:
        return [0.0]
    variance = sum(value * value for value in centered) / (len(centered) - 1)
    scale = math.sqrt(variance)
    return [value / (scale + _EPSILON) for value in centered]


__all__ = ["SAMPOAdvantages", "compute_sampo_advantages"]
