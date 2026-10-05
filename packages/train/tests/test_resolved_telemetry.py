"""Resolved-engine telemetry values equal an independent hand computation.

The fixture population has three complete groups of three episodes:

* group a has dense per-turn step rewards, a truncated episode and turns that
  share and do not share anchor states;
* group b has identical rewards, so it carries no learning signal at all;
* group c has sparse terminal rewards and a truncated episode.

The reference below is written from the SAMPO definition (GiGPO episode
credit plus anchor-state-relative discounted turn credit), not by calling the
estimator, and is checked against the resolved collection metrics, the real
SAMPO credit estimator, the per-update metrics and the ordinary (legacy) TRL
SAMPO update means.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
from posttrain.common import TraceObservation
from posttrain.train.online_rl import AgenticTurn, EnvironmentRollout
from posttrain.train.profiles import SAMPOSettings, TrainingLoop
from posttrain.train.sampo_advantages import compute_sampo_advantages
from posttrain.train.update_credit import NativeCreditRows, SampoCreditEstimator, prepare_credit
from posttrain.train.update_records import (
    ActionRecord,
    ActionRef,
    ConditioningView,
    PolicyVersions,
    PopulationRelation,
    PopulationSnapshot,
)
from posttrain.train.update_sampler_correction import sampler_correction_weights
from posttrain.train.update_telemetry import collection_metrics, update_metrics

GAMMA = 0.95
TRUNCATION_PENALTY = 0.5
# TRL's torch.isclose(advantage, 0) boundary for zero credit.
ZERO = 1e-8


def _credit_fractions(tokens: list[float]) -> dict[str, float]:
    count = len(tokens)
    return {
        "train/rl/advantage_positive_fraction": sum(value > ZERO for value in tokens) / count,
        "train/rl/advantage_negative_fraction": sum(value < -ZERO for value in tokens) / count,
        "train/rl/advantage_zero_fraction": sum(abs(value) <= ZERO for value in tokens) / count,
    }


# (task, reward, truncated, ((anchor, step reward or None), ...))
EPISODES: tuple[tuple[str, float, bool, tuple[tuple[str, float | None], ...]], ...] = (
    ("task-a", 1.0, False, (("s0", 0.2), ("s1", 0.5))),
    ("task-a", 0.4, False, (("s0", 0.0), ("s2", 0.3))),
    ("task-a", 0.0, True, (("s0", 0.1), ("s1", 0.0), ("s3", 0.2))),
    ("task-b", 0.5, False, (("t0", None), ("t1", None))),
    ("task-b", 0.5, False, (("t0", None), ("t1", None))),
    ("task-b", 0.5, False, (("t0", None), ("t2", None))),
    ("task-c", 1.0, False, (("u0", None),)),
    ("task-c", 0.0, True, (("u0", None), ("u1", None))),
    ("task-c", 1.0, False, (("u0", None), ("u1", None))),
)


def _settings(normalization: str, weight: float) -> SAMPOSettings:
    return SAMPOSettings(
        id="telemetry-fixture",
        loop=TrainingLoop(max_steps=1, max_length=64, per_device_batch_size=9),
        num_prompts_per_step=3,
        num_generations=3,
        max_prompt_length=8,
        max_completion_length=32,
        discount_gamma=GAMMA,
        step_advantage_weight=weight,
        advantage_normalization=cast(Any, normalization),
        truncation_penalty=TRUNCATION_PENALTY,
    )


def _rollouts(episodes=EPISODES) -> tuple[EnvironmentRollout, ...]:
    """Each turn samples two tokens; one observation token separates turns."""
    rollouts = []
    for index, (task, reward, truncated, turns) in enumerate(episodes):
        mask: list[bool] = []
        spans = []
        for position, (anchor, step_reward) in enumerate(turns):
            if position:
                mask.append(False)
            spans.append(AgenticTurn(len(mask), len(mask) + 2, anchor, step_reward))
            mask.extend((True, True))
        rollouts.append(
            EnvironmentRollout(
                example_id=task,
                prompt_ids=(1, 2),
                completion_ids=tuple(range(3, 3 + len(mask))),
                sampling_logprobs=(-0.5,) * len(mask),
                env_mask=tuple(mask),
                reward=reward,
                is_truncated=truncated,
                trace=TraceObservation("verifiers", f"trace-{index}", {}),
                turns=tuple(spans),
            )
        )
    return tuple(rollouts)


def _normalize(values: list[float], normalization: str) -> list[float]:
    mean = sum(values) / len(values)
    if normalization == "mean":
        return [value - mean for value in values]
    if len(values) == 1:
        return [0.0]
    scale = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return [(value - mean) / (scale + 1e-6) for value in values]


def _stdev(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _reference(normalization: str, weight: float, episodes=EPISODES) -> dict[str, Any]:
    shaped = [reward - (TRUNCATION_PENALTY if truncated else 0.0) for _, reward, truncated, _ in episodes]
    groups: dict[str, list[int]] = {}
    for index, (task, *_rest) in enumerate(episodes):
        groups.setdefault(task, []).append(index)
    episode = [0.0] * len(episodes)
    for members in groups.values():
        for index, value in zip(members, _normalize([shaped[i] for i in members], normalization), strict=True):
            episode[index] = value
    # Discounted turn returns: explicit step rewards, or the shaped episode
    # reward projected onto the final turn when no step reward exists.
    returns: list[list[float]] = []
    for index, (_, _, _, turns) in enumerate(episodes):
        rewards = [step for _, step in turns]
        if all(step is None for step in rewards):
            rewards = [0.0] * (len(turns) - 1) + [shaped[index]]
        running, values = 0.0, []
        for step in reversed(rewards):
            assert step is not None
            running = step + GAMMA * running
            values.append(running)
        returns.append(values[::-1])
    anchors: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for index, (task, _, _, turns) in enumerate(episodes):
        for position, (anchor, _) in enumerate(turns):
            anchors.setdefault((task, anchor), []).append((index, position))
    turn = [[0.0] * len(item[3]) for item in episodes]
    size = [[0] * len(item[3]) for item in episodes]
    for members in anchors.values():
        for (index, position), value in zip(
            members, _normalize([returns[i][p] for i, p in members], normalization), strict=True
        ):
            turn[index][position] = value
            size[index][position] = len(members)
    flat_turns = [(index, value) for index, values in enumerate(turn) for value in values]
    flat_sizes = [value for values in size for value in values]
    episode_credit = sum(abs(episode[index]) for index, _ in flat_turns)
    turn_credit = sum(abs(weight * value) for _, value in flat_turns)
    group_stds = [_stdev([shaped[i] for i in members]) for members in groups.values()]
    lengths = [2 * len(item[3]) for item in episodes]
    # Every sampled token of a turn carries episode + weight * turn credit.
    tokens = [episode[index] + weight * value for index, values in enumerate(turn) for value in values for _ in "ab"]
    return {
        "shaped": shaped,
        "tokens": tokens,
        "collection": {
            "train/rl/reward_mean": sum(shaped) / len(shaped),
            "train/rl/reward_std": _stdev(shaped),
            "train/rl/group_reward_std_mean": sum(group_stds) / len(group_stds),
            "train/rl/group_zero_variance_fraction": sum(std == 0 for std in group_stds) / len(group_stds),
            "train/rl/completion_tokens_mean": sum(lengths) / len(lengths),
            "train/rl/completion_tokens_max": float(max(lengths)),
            "train/rl/completion_truncation_rate": sum(item[2] for item in episodes) / len(episodes),
            "train/rl/episode_advantage_mean": sum(episode) / len(episode),
            "train/rl/episode_advantage_abs_mean": sum(map(abs, episode)) / len(episode),
            "train/rl/turn_advantage_mean": sum(value for _, value in flat_turns) / len(flat_turns),
            "train/rl/turn_advantage_abs_mean": sum(abs(value) for _, value in flat_turns) / len(flat_turns),
            "train/rl/anchor_group_size_mean": sum(flat_sizes) / len(flat_sizes),
            "train/rl/sparse_reward_projection_fraction": sum(
                all(step is None for _, step in item[3]) for item in episodes
            )
            / len(episodes),
            "train/rl/turn_advantage_informative_fraction": sum(abs(value) > 1e-9 for _, value in flat_turns)
            / len(flat_turns),
            "train/rl/singleton_anchor_fraction": sum(value == 1 for value in flat_sizes) / len(flat_sizes),
            "train/rl/turn_credit_share": turn_credit / (episode_credit + turn_credit),
        },
    }


CASES = [("mean", 1.0), ("mean_std", 1.0), ("mean", 0.5)]


def test_fixture_contains_a_zero_signal_group_and_shared_anchors():
    expected = _reference("mean", 1.0)
    # Group b's identical rewards produce zero spread and zero credit throughout.
    assert expected["collection"]["train/rl/group_zero_variance_fraction"] == pytest.approx(1 / 3)
    # Centring equal values leaves rounding residue (about 1e-17), not credit.
    assert expected["tokens"][14:26] == pytest.approx([0.0] * 12, abs=1e-15)
    # Shaping: the two truncated episodes lose the 0.5 penalty.
    assert expected["shaped"] == [1.0, 0.4, -0.5, 0.5, 0.5, 0.5, 1.0, -0.5, 1.0]
    assert expected["collection"]["train/rl/reward_mean"] == pytest.approx(3.9 / 9)


@pytest.mark.parametrize(("normalization", "weight"), CASES)
def test_resolved_sampo_collection_metrics_equal_hand_computation(normalization, weight):
    settings = _settings(normalization, weight)
    expected = _reference(normalization, weight)["collection"]

    actual = collection_metrics(settings, _rollouts(), credit_estimator_id="sampo-credit@2:fixture")

    assert actual == pytest.approx(expected, abs=1e-12)
    # A replaced (process) estimator is not described as SAMPO credit.
    replaced = collection_metrics(settings, _rollouts(), credit_estimator_id="critique-credit@1")
    assert "train/rl/turn_credit_share" not in replaced
    assert replaced["train/rl/reward_mean"] == pytest.approx(expected["train/rl/reward_mean"])


def _native_population(settings: SAMPOSettings):
    rollouts = _rollouts()
    coordinates = tuple(
        tuple(
            ActionRef(f"episode-{index}", "branch", f"turn-{index}", position) if eligible else None
            for position, eligible in enumerate(rollout.env_mask)
        )
        for index, rollout in enumerate(rollouts)
    )
    actions = tuple(action for row in coordinates for action in row if action is not None)
    snapshot = PopulationSnapshot(
        "telemetry-fixture",
        "native:fixture",
        "native-fixture-digest",
        tuple(ActionRecord(action, "context", "native:node") for action in actions),
        (ConditioningView("context", "native:fixture", "tokens", "attention", "positions", "template@1", "digest", 8),),
        (),
        (PopulationRelation("population", "prompt-group", actions, "complete", actions),),
        PolicyVersions("sampler@1", "old@1", "current@1", None),
        "selector@1",
    )
    rows = NativeCreditRows(rollouts, coordinates, ("native-fixture-digest",))
    credit = prepare_credit(snapshot, SampoCreditEstimator(settings, rows, ("population",)))
    return snapshot, actions, credit


@pytest.mark.parametrize(("normalization", "weight"), CASES)
def test_real_sampo_credit_and_update_metrics_equal_hand_computation(normalization, weight):
    settings = _settings(normalization, weight)
    reference = _reference(normalization, weight)
    snapshot, actions, credit = _native_population(settings)
    tokens = reference["tokens"]

    # The estimator that trains the policy produces the hand-computed credit,
    # and the telemetry gate recognizes its identity.
    assert [value.advantage for value in credit.values] == pytest.approx(tokens, abs=1e-12)
    assert credit.estimator_id.startswith("sampo-credit@")

    # Sampler correction: token truncation at 2.0, no lower bound (the SAMPO default).
    deltas = [(0.0, 0.5, -0.5, 2.0, -3.0)[index % 5] for index in range(len(actions))]
    old = {action: -1.0 for action in actions}
    sampled = {action: -1.0 - delta for action, delta in zip(actions, deltas, strict=True)}
    correction = sampler_correction_weights(snapshot, old, sampled, mode="token_truncate", lower=None, upper=2.0)
    weights = [min(math.exp(delta), 2.0) for delta in deltas]
    assert [correction[action] for action in actions] == pytest.approx(weights)

    # Two clipped actions: one above and one below the unit ratio.
    clipped = {actions[0]: 1.5, actions[3]: 0.6}
    entropies = {action: 0.1 * (index % 4) for index, action in enumerate(actions)}
    term = cast(Any, SimpleNamespace(policy_weights=tuple(SimpleNamespace(action=action) for action in actions)))

    values = update_metrics(
        term,
        credit,
        clipped_ratios=clipped,
        sampler_correction=correction,
        correction_recipe=("token_truncate", None, 2.0),
        entropies=entropies,
        old_scores=old,
        sampled_scores=sampled,
    )

    count = len(tokens)
    mean = sum(tokens) / count
    assert values == pytest.approx(
        {
            "train/rl/advantage_mean": mean,
            "train/rl/advantage_abs_mean": sum(map(abs, tokens)) / count,
            "train/rl/advantage_std": math.sqrt(sum((value - mean) ** 2 for value in tokens) / count),
            "train/rl/advantage_nonzero_fraction": sum(abs(value) > ZERO for value in tokens) / count,
            **_credit_fractions(tokens),
            "train/rl/clip_fraction": 2 / count,
            "train/rl/clip_fraction_high": 1 / count,
            "train/rl/clip_fraction_low": 1 / count,
            "train/rl/importance_sampling_ratio_mean": sum(weights) / count,
            "train/rl/importance_sampling_ratio_min": math.exp(-3.0),
            "train/rl/importance_sampling_ratio_max": 2.0,
            "train/rl/importance_sampling_ratio_clamped_fraction": deltas.count(2.0) / count,
            "train/rl/entropy": sum(entropies.values()) / count,
            **_hand_sampler_gap(actions, deltas),
        },
        abs=1e-12,
    )
    # Group b contributes twelve zero-credit tokens despite rounding residue.
    assert values["train/rl/advantage_zero_fraction"] >= 12 / count


def _hand_sampler_gap(actions, deltas) -> dict[str, float]:
    """|old - sampled| per selected token; |sum(old - sampled)| per touched episode."""
    gaps = sorted(abs(delta) for delta in deltas)
    episodes: dict[str, float] = {}
    for action, delta in zip(actions, deltas, strict=True):
        episodes[action.episode_id] = episodes.get(action.episode_id, 0.0) + delta
    return {
        "train/rl/sampling_logp_delta_mean": sum(gaps) / len(gaps),
        "train/rl/sampling_logp_delta_max": max(gaps),
        # Nearest-rank 99th percentile: ceil(0.99 * n)-th smallest.
        "train/rl/sampling_logp_delta_p99": gaps[math.ceil(0.99 * len(gaps)) - 1],
        "train/rl/sampling_sequence_logp_delta_abs_mean": sum(map(abs, episodes.values())) / len(episodes),
    }


def test_sequence_correction_reports_one_weight_per_episode_and_full_episode_gaps():
    settings = _settings("mean", 1.0)
    snapshot, actions, credit = _native_population(settings)
    episodes = sorted({action.episode_id for action in actions})
    # A constant per-episode token gap gives each episode a known sequence log ratio.
    per_episode = {
        episode: (0.1, -0.2, 0.05, 0.4, -0.3, 0.0, 0.2, -0.05, 0.15)[i] for i, episode in enumerate(episodes)
    }
    deltas = [per_episode[action.episode_id] for action in actions]
    old = {action: -2.0 for action in actions}
    sampled = {action: -2.0 - delta for action, delta in zip(actions, deltas, strict=True)}
    correction = sampler_correction_weights(snapshot, old, sampled, mode="sequence_truncate", lower=None, upper=2.0)
    # The update selects only the first turn's tokens of every episode, but
    # sequence statistics use each touched episode's complete support.
    selected = [action for action in actions if action.token_index < 2]
    term = cast(Any, SimpleNamespace(policy_weights=tuple(SimpleNamespace(action=action) for action in selected)))

    values = update_metrics(
        term,
        credit,
        clipped_ratios={},
        sampler_correction=correction,
        correction_recipe=("sequence_truncate", None, 2.0),
        old_scores=old,
        sampled_scores=sampled,
    )

    lengths = {episode: sum(action.episode_id == episode for action in actions) for episode in episodes}
    sequence = {episode: per_episode[episode] * lengths[episode] for episode in episodes}
    weights = [min(math.exp(value), 2.0) for value in sequence.values()]
    gaps = sorted(abs(per_episode[action.episode_id]) for action in selected)
    assert values["train/rl/importance_sampling_ratio_mean"] == pytest.approx(sum(weights) / len(weights))
    assert values["train/rl/importance_sampling_ratio_min"] == pytest.approx(min(weights))
    assert values["train/rl/importance_sampling_ratio_max"] == 2.0
    assert values["train/rl/importance_sampling_ratio_clamped_fraction"] == pytest.approx(
        sum(math.exp(value) > 2.0 for value in sequence.values()) / len(episodes)
    )
    assert values["train/rl/sampling_logp_delta_mean"] == pytest.approx(sum(gaps) / len(gaps))
    assert values["train/rl/sampling_logp_delta_max"] == pytest.approx(0.4)
    assert values["train/rl/sampling_sequence_logp_delta_abs_mean"] == pytest.approx(
        sum(map(abs, sequence.values())) / len(episodes)
    )


def test_update_metrics_count_masked_correction_and_cover_only_selected_actions():
    settings = _settings("mean", 1.0)
    snapshot, actions, credit = _native_population(settings)
    deltas = [(0.0, 1.0, -1.0)[index % 3] for index in range(len(actions))]
    old = {action: 0.0 for action in actions}
    sampled = {action: -delta for action, delta in zip(actions, deltas, strict=True)}
    correction = sampler_correction_weights(snapshot, old, sampled, mode="token_mask", lower=0.5, upper=2.0)
    selected = actions[:6]
    term = cast(Any, SimpleNamespace(policy_weights=tuple(SimpleNamespace(action=action) for action in selected)))

    values = update_metrics(
        term,
        credit,
        clipped_ratios={actions[-1]: 2.0},
        sampler_correction=correction,
        correction_recipe=("token_mask", 0.5, 2.0),
    )

    # e^1 and e^-1 fall outside [0.5, 2.0] and are masked to zero.
    assert values["train/rl/importance_sampling_ratio_clamped_fraction"] == pytest.approx(4 / 6)
    assert values["train/rl/importance_sampling_ratio_min"] == 0.0
    assert values["train/rl/importance_sampling_ratio_max"] == 1.0
    # A clipped action outside the selected update is not counted.
    assert values["train/rl/clip_fraction"] == 0.0
    assert "train/rl/entropy" not in values
    assert "train/rl/sampling_logp_delta_mean" not in values  # no sampled scores supplied
    empty = cast(Any, SimpleNamespace(policy_weights=()))
    assert update_metrics(empty, credit, clipped_ratios={}) == {}


@pytest.mark.parametrize(("normalization", "weight"), CASES)
def test_ordinary_trl_sampo_update_means_equal_hand_computation(normalization, weight):
    """The legacy TRL SAMPO path's own credit metrics agree with the same reference."""
    from posttrain.train.backends.trl.policy_rollouts import _sampo_update_means

    settings = _settings(normalization, weight)
    reference = _reference(normalization, weight)
    rollouts = _rollouts()
    # The ordinary collector hands the estimator shaped copies, in group order.
    shaped = [replace(rollout, reward=value) for rollout, value in zip(rollouts, reference["shaped"], strict=True)]
    advantages = compute_sampo_advantages(settings, tuple(rollout.example_id for rollout in shaped), shaped)

    means = {name: mean for name, (mean, _) in _sampo_update_means(advantages, settings).items()}

    hierarchy = {
        name: value
        for name, value in reference["collection"].items()
        if "advantage" in name or "anchor" in name or "sparse" in name or "credit" in name
    }
    tokens = reference["tokens"]
    count = len(tokens)
    assert means == pytest.approx(
        {
            **hierarchy,
            "train/rl/advantage_mean": sum(tokens) / count,
            "train/rl/advantage_abs_mean": sum(map(abs, tokens)) / count,
            **_credit_fractions(tokens),
        },
        abs=1e-12,
    )


def test_native_record_drops_values_owned_by_another_writer():
    from posttrain.train.backends.trl.policy_telemetry import native_trainer_record

    native = {"loss": 1.0, "grad_norm": 2.0, "advantages/abs_mean": 0.7, "reward": 0.4}
    # Ordinary SAMPO: TRL's scalar GRPO advantages are never trained on and
    # would overwrite SAMPO's own credit statistics at the same step.
    assert native_trainer_record(native, resolved=False, sampo=True) == {"loss": 1.0, "grad_norm": 2.0, "reward": 0.4}
    assert native_trainer_record(native, resolved=False, sampo=False) == native
    # Resolved: the native windowed loss is not the resolved objective.
    assert native_trainer_record(native, resolved=True, sampo=True) == {
        "grad_norm": 2.0,
        "advantages/abs_mean": 0.7,
        "reward": 0.4,
    }
