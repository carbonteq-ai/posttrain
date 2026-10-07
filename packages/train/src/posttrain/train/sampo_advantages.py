"""Backend-neutral SAMPO hierarchical advantage construction."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .online_rl import EnvironmentRollout
from .profiles import SAMPOSettings

_EPSILON = 1e-6
# A turn advantage smaller than this carries no usable relative signal.
_INFORMATIVE = 1e-9
# Credit within this distance of zero is reported as zero, matching TRL's
# torch.isclose(advantage, 0) convention. Centring a group of equal rewards
# leaves rounding residue (around 1e-17), which is not positive or negative credit.
ZERO_CREDIT_TOLERANCE = 1e-8


@dataclass(frozen=True, slots=True)
class SAMPOAdvantages:
    """Token-aligned advantages plus evidence used to explain the update."""

    token_advantages: tuple[tuple[float, ...], ...]
    episode_advantages: tuple[float, ...]
    turn_advantages: tuple[tuple[float, ...], ...]
    anchor_group_sizes: tuple[tuple[int, ...], ...]
    used_sparse_rewards: tuple[bool, ...]
    sampled_token_advantages: tuple[float, ...]
    # Goal-relative turn credit (settings.goal_credit); zeros when it is off.
    goal_advantages: tuple[tuple[float, ...], ...] = ()
    # verified-sign: turns that first achieved a goal, and turns whose sign the verified
    # outcome changed relative to episode + anchor credit.
    verified_turns: tuple[tuple[bool, ...], ...] = ()
    sign_protected_turns: tuple[tuple[bool, ...], ...] = ()
    # anchor_fallback: turns grouped by their environment state key.
    fallback_anchor_turns: tuple[tuple[bool, ...], ...] = ()

    def hierarchy_evidence(self, step_advantage_weight: float) -> dict[str, tuple[float, int]]:
        """Per-update (mean, count) pairs that show where SAMPO's credit comes from.

        Advantages are centred within each prompt group and anchor group, so
        their plain means are zero by construction. Magnitudes and shares are
        the readable evidence: how much credit each level carries, how much of
        a turn's credit is its own, and how many turns had anything to compare.
        """

        turns = [
            (episode, turn, size)
            for episode, turn_values, sizes in zip(
                self.episode_advantages, self.turn_advantages, self.anchor_group_sizes, strict=True
            )
            for turn, size in zip(turn_values, sizes, strict=True)
        ]
        evidence = {
            "train/rl/episode_advantage_abs_mean": (
                math.fsum(abs(value) for value in self.episode_advantages) / len(self.episode_advantages),
                len(self.episode_advantages),
            ),
        }
        if not turns:
            return evidence
        count = len(turns)
        episode_credit = math.fsum(abs(episode) for episode, _, _ in turns)
        turn_credit = math.fsum(abs(step_advantage_weight * turn) for _, turn, _ in turns)
        evidence.update(
            {
                "train/rl/turn_advantage_abs_mean": (math.fsum(abs(turn) for _, turn, _ in turns) / count, count),
                "train/rl/turn_advantage_informative_fraction": (
                    sum(abs(turn) > _INFORMATIVE for _, turn, _ in turns) / count,
                    count,
                ),
                "train/rl/singleton_anchor_fraction": (sum(size == 1 for _, _, size in turns) / count, count),
            }
        )
        goal_values = [value for values in self.goal_advantages for value in values]
        goal_credit = math.fsum(abs(value) for value in goal_values)
        if goal_values:
            evidence["train/rl/goal_turn_credit_abs_mean"] = (goal_credit / len(goal_values), len(goal_values))
            evidence["train/rl/goal_credited_turn_fraction"] = (
                sum(value > _INFORMATIVE for value in goal_values) / len(goal_values),
                len(goal_values),
            )
            evidence["train/rl/harm_debited_turn_fraction"] = (
                sum(value < -_INFORMATIVE for value in goal_values) / len(goal_values),
                len(goal_values),
            )
        fallback_flags = [flag for flags in self.fallback_anchor_turns for flag in flags]
        if fallback_flags:
            evidence["train/rl/fallback_anchor_turn_fraction"] = (
                sum(fallback_flags) / len(fallback_flags),
                len(fallback_flags),
            )
        verified = [flag for flags in self.verified_turns for flag in flags]
        if verified:
            evidence["train/rl/verified_turn_fraction"] = (sum(verified) / len(verified), len(verified))
            protected = [flag for flags in self.sign_protected_turns for flag in flags]
            evidence["train/rl/sign_protected_turn_fraction"] = (sum(protected) / len(protected), len(protected))
        if episode_credit + turn_credit + goal_credit > 0:
            evidence["train/rl/turn_credit_share"] = (
                (turn_credit + goal_credit) / (episode_credit + turn_credit + goal_credit),
                count,
            )
        return evidence

    def credit_evidence(self, step_advantage_weight: float) -> dict[str, tuple[float, int]]:
        """Per-population (mean, count) for every SAMPO credit metric except sampled-token pooling.

        The centred episode and turn means stay for continuity with older
        readers; `hierarchy_evidence` supplies the readable magnitudes and
        shares. Both backends and both update engines report this one set.
        """

        flat_turns = [value for values in self.turn_advantages for value in values]
        flat_sizes = [value for values in self.anchor_group_sizes for value in values]
        evidence = {
            "train/rl/episode_advantage_mean": (
                math.fsum(self.episode_advantages) / len(self.episode_advantages),
                len(self.episode_advantages),
            ),
            "train/rl/sparse_reward_projection_fraction": (
                sum(self.used_sparse_rewards) / len(self.used_sparse_rewards),
                len(self.used_sparse_rewards),
            ),
        }
        if flat_turns:
            evidence["train/rl/turn_advantage_mean"] = (math.fsum(flat_turns) / len(flat_turns), len(flat_turns))
            evidence["train/rl/anchor_group_size_mean"] = (sum(flat_sizes) / len(flat_sizes), len(flat_sizes))
        evidence.update(self.hierarchy_evidence(step_advantage_weight))
        return evidence

    def policy_credit_evidence(self) -> dict[str, tuple[float, int]]:
        """Pool actual supplied advantages over sampled actions, excluding tool/padding tokens."""
        tokens = self.sampled_token_advantages
        count = len(tokens)
        return {
            "train/rl/advantage_mean": (math.fsum(tokens) / count, count),
            "train/rl/advantage_abs_mean": (math.fsum(abs(value) for value in tokens) / count, count),
            "train/rl/advantage_positive_fraction": (
                sum(value > ZERO_CREDIT_TOLERANCE for value in tokens) / count,
                count,
            ),
            "train/rl/advantage_negative_fraction": (
                sum(value < -ZERO_CREDIT_TOLERANCE for value in tokens) / count,
                count,
            ),
            "train/rl/advantage_zero_fraction": (
                sum(abs(value) <= ZERO_CREDIT_TOLERANCE for value in tokens) / count,
                count,
            ),
        }


def anchor_identities(rollouts: Sequence[EnvironmentRollout], *, fallback: bool) -> list[list[str]]:
    """Effective anchor identity of every turn of one prompt group's rollouts.

    Without fallback this is the exact observation key. With fallback, a turn whose exact key
    no other turn of the group shares, and that declares an environment state key, is grouped by
    that state key instead (only with other such turns).
    """

    if not fallback:
        return [[turn.anchor_state_key for turn in rollout.turns] for rollout in rollouts]
    exact = Counter(turn.anchor_state_key for rollout in rollouts for turn in rollout.turns)
    return [
        [
            f"environment-state:{turn.state_key}"
            if exact[turn.anchor_state_key] == 1 and turn.state_key is not None
            else turn.anchor_state_key
            for turn in rollout.turns
        ]
        for rollout in rollouts
    ]


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
    fallback = settings.anchor_fallback == "environment-state"
    fallback_turns = [[False] * len(rollout.turns) for rollout in rollouts]
    for group_index, indices in enumerate(grouped_indices):
        identities = anchor_identities([rollouts[index] for index in indices], fallback=fallback)
        for rollout_index, turn_identities in zip(indices, identities, strict=True):
            returns = returns_by_rollout[rollout_index]
            for turn_index, (identity, value) in enumerate(zip(turn_identities, returns, strict=True)):
                anchors[(group_index, identity)].append((rollout_index, turn_index, value))
                fallback_turns[rollout_index][turn_index] = identity.startswith("environment-state:")

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

    goal_advantages = [[0.0] * len(rollout.turns) for rollout in rollouts]
    verified_turns = [[False] * len(rollout.turns) for rollout in rollouts]
    sign_protected = [[False] * len(rollout.turns) for rollout in rollouts]
    if settings.goal_credit in {"group-relative", "verified-sign"}:
        scale = settings.goal_credit_scale
        for indices in grouped_indices:
            # p_g: the share of the group's attempts in which some turn first achieved goal g.
            reached: dict[str, int] = defaultdict(int)
            for rollout_index in indices:
                for key in {key for turn in rollouts[rollout_index].turns for key, _ in turn.goal_credits}:
                    reached[key] += 1
            for rollout_index in indices:
                for turn_index, turn in enumerate(rollouts[rollout_index].turns):
                    goal = scale * math.fsum(
                        weight * (1.0 - reached[key] / len(indices)) for key, weight in turn.goal_credits
                    )
                    if settings.goal_credit == "group-relative":
                        goal_advantages[rollout_index][turn_index] = goal - turn.harm_debit
                        continue
                    # verified-sign: the turn's own verified outcome decides its sign.
                    base = episode[rollout_index] + (
                        settings.step_advantage_weight * turn_advantages[rollout_index][turn_index]
                    )
                    if turn.harm_debit > 0:
                        total = min(base + goal, 0.0) - turn.harm_debit
                    elif turn.goal_credits:
                        total = max(base, 0.0) + goal
                    else:
                        total = base
                    goal_advantages[rollout_index][turn_index] = total - base
                    verified_turns[rollout_index][turn_index] = bool(turn.goal_credits)
                    sign_protected[rollout_index][turn_index] = (total > 0) != (base > 0) and total != 0.0
    elif any(turn.goal_credits or turn.harm_debit for rollout in rollouts for turn in rollout.turns):
        raise ValueError("SAMPO turns carry goal or harm credit but goal_credit is off")

    token_advantages: list[tuple[float, ...]] = []
    for rollout_index, rollout in enumerate(rollouts):
        values = [0.0] * len(rollout.completion_ids)
        for turn_index, turn in enumerate(rollout.turns):
            combined = (
                episode[rollout_index]
                + settings.step_advantage_weight * turn_advantages[rollout_index][turn_index]
                + goal_advantages[rollout_index][turn_index]
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
        goal_advantages=(tuple(tuple(values) for values in goal_advantages) if settings.goal_credit != "none" else ()),
        fallback_anchor_turns=(tuple(tuple(flags) for flags in fallback_turns) if fallback else ()),
        verified_turns=(
            tuple(tuple(flags) for flags in verified_turns) if settings.goal_credit == "verified-sign" else ()
        ),
        sign_protected_turns=(
            tuple(tuple(flags) for flags in sign_protected) if settings.goal_credit == "verified-sign" else ()
        ),
        sampled_token_advantages=tuple(
            value
            for values, rollout in zip(token_advantages, rollouts, strict=True)
            for value, selected in zip(values, rollout.env_mask, strict=True)
            if selected
        ),
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


__all__ = ["ZERO_CREDIT_TOLERANCE", "SAMPOAdvantages", "compute_sampo_advantages"]
