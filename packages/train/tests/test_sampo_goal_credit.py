"""Goal-relative turn credit: verified goals and harms reach the turn that produced them."""

from __future__ import annotations

import pytest
from posttrain.common import TraceObservation
from posttrain.train import AgenticTurn, EnvironmentRollout, SAMPOSettings, TrainingLoop, compute_sampo_advantages


def _settings(**changes) -> SAMPOSettings:
    values = {
        "id": "sampo-goal-test",
        "loop": TrainingLoop(max_steps=1, max_length=8, per_device_batch_size=4),
        "num_generations": 4,
        "max_prompt_length": 2,
        "max_completion_length": 6,
        "goal_credit": "group-relative",
    }
    values.update(changes)
    return SAMPOSettings(**values)


def _rollout(suffix: str, *, reward=0.5, first=(), second=(), harm=0.0, steps=(0.0, 0.0)) -> EnvironmentRollout:
    """Two turns; each attempt sees its own observations, so every turn is a singleton anchor."""
    return EnvironmentRollout(
        example_id="task-1",
        prompt_ids=(1, 2),
        completion_ids=(3, 4, 5, 6, 7, 8),
        sampling_logprobs=(-0.1,) * 6,
        env_mask=(True, True, False, False, True, True),
        reward=reward,
        is_truncated=False,
        trace=TraceObservation("test", f"trace-{suffix}", {}),
        turns=(
            AgenticTurn(0, 2, f"first-{suffix}", steps[0], goal_credits=first),
            AgenticTurn(4, 6, f"second-{suffix}", steps[1], goal_credits=second, harm_debit=harm),
        ),
    )


def _group(*rollouts):
    return compute_sampo_advantages(_settings(), ("task-1",) * len(rollouts), rollouts)


def test_a_rare_goal_credits_its_singleton_turn_by_how_few_attempts_reached_it():
    result = _group(
        _rollout("a", second=(("record:task-written", 0.25),)),
        _rollout("b"),
        _rollout("c"),
        _rollout("d"),
    )
    # Every turn is a singleton anchor, so the observation-anchored turn credit is zero ...
    assert all(value == 0 for values in result.turn_advantages for value in values)
    # ... but the goal still reaches the turn that achieved it: 0.25 * (1 - 1/4).
    assert result.goal_advantages[0] == pytest.approx((0.0, 0.1875))
    assert result.goal_advantages[1:] == ((0.0, 0.0),) * 3
    assert result.token_advantages[0] == pytest.approx((0.0, 0.0, 0.0, 0.0, 0.1875, 0.1875))


def test_a_goal_every_attempt_reached_earns_nothing():
    goal = (("obligation:read:instance", 0.25),)
    result = _group(*(_rollout(name, first=goal) for name in "abcd"))
    assert result.goal_advantages == ((0.0, 0.0),) * 4


def test_a_goal_counts_once_per_attempt_wherever_it_was_achieved():
    goal = "obligation:read:instance"
    result = _group(
        _rollout("a", first=((goal, 0.25),)),
        _rollout("b", second=((goal, 0.25),)),
        _rollout("c"),
        _rollout("d"),
    )
    assert result.goal_advantages[0] == pytest.approx((0.125, 0.0))
    assert result.goal_advantages[1] == pytest.approx((0.0, 0.125))


def test_a_harm_debits_its_turn_even_when_every_attempt_caused_it():
    result = _group(*(_rollout(name, harm=0.1) for name in "abcd"))
    assert result.goal_advantages == ((0.0, -0.1),) * 4
    evidence = result.hierarchy_evidence(1.0)
    assert evidence["train/rl/harm_debited_turn_fraction"][0] == pytest.approx(0.5)


def test_goal_credit_is_reported_beside_the_hierarchy():
    result = _group(_rollout("a", second=(("record:task-written", 0.25),)), _rollout("b"), _rollout("c"), _rollout("d"))
    evidence = result.hierarchy_evidence(1.0)
    assert evidence["train/rl/goal_credited_turn_fraction"][0] == pytest.approx(1 / 8)
    assert evidence["train/rl/goal_turn_credit_abs_mean"][0] == pytest.approx(0.1875 / 8)


def test_turn_outcomes_without_the_setting_are_refused():
    rollouts = (_rollout("a", harm=0.1), _rollout("b"), _rollout("c"), _rollout("d"))
    with pytest.raises(ValueError, match="goal_credit is off"):
        compute_sampo_advantages(_settings(goal_credit="none"), ("task-1",) * 4, rollouts)


def test_turn_goal_credits_are_validated():
    with pytest.raises(ValueError, match="unique keys"):
        AgenticTurn(0, 2, "key", goal_credits=(("g", 0.1), ("g", 0.2)))
    with pytest.raises(ValueError, match="positive"):
        AgenticTurn(0, 2, "key", goal_credits=(("g", 0.0),))
    with pytest.raises(ValueError, match="harm debit"):
        AgenticTurn(0, 2, "key", harm_debit=-0.1)


def _signed(*rollouts, scale=4.0):
    settings = _settings(goal_credit="verified-sign", goal_credit_scale=scale)
    return compute_sampo_advantages(settings, ("task-1",) * len(rollouts), rollouts)


def test_verified_goal_turn_is_never_pushed_negative_by_its_attempts_failure():
    goal = (("record:task-written", 0.25),)
    result = _signed(
        _rollout("loser", reward=0.0, second=goal),
        _rollout("b", reward=1.0, second=goal),
        _rollout("c", reward=1.0),
        _rollout("d", reward=1.0),
    )
    # The losing attempt's episode advantage is -0.75; its goal turn keeps max(-0.75, 0) plus
    # 4 * 0.25 * (1 - 2/4) = 0.5, and its other turn keeps the episode signal.
    first, second = (sum(result.token_advantages[0][i : i + 2]) / 2 for i in (0, 4))
    assert first == pytest.approx(-0.75)
    assert second == pytest.approx(0.5)
    assert result.sign_protected_turns[0] == (False, True)
    assert result.verified_turns[0] == (False, True)


def test_harmful_turn_is_never_pushed_positive_by_its_attempts_success_even_with_a_goal():
    result = _signed(
        _rollout("winner", reward=1.0, second=(("record:task-written", 0.25),), harm=0.1),
        _rollout("b", reward=0.0),
        _rollout("c", reward=0.0),
        _rollout("d", reward=0.0),
    )
    second = result.token_advantages[0][4]
    assert second == pytest.approx(-0.1)  # min(0.75 + 4 * 0.25 * 0.75, 0) - 0.1
    assert result.token_advantages[0][0] == pytest.approx(0.75)  # the clean turn keeps its episode credit
    # The harm forced the goal turn negative, so it is not a turn the rule protected.
    assert result.verified_turns[0] == (False, False)
    assert result.hierarchy_evidence(1.0)["train/rl/verified_turn_fraction"][0] == 0.0


def test_verified_sign_scales_the_goal_term_and_reports_protection():
    goal = (("record:task-written", 0.25),)
    rollouts = (
        _rollout("a", reward=1.0, second=goal),
        _rollout("b", reward=0.0),
        _rollout("c", reward=0.0),
        _rollout("d", reward=0.0),
    )
    small, large = _signed(*rollouts, scale=1.0), _signed(*rollouts, scale=4.0)
    assert large.token_advantages[0][4] - small.token_advantages[0][4] == pytest.approx(3 * 0.25 * 0.75)
    evidence = large.hierarchy_evidence(1.0)
    assert evidence["train/rl/verified_turn_fraction"][0] == pytest.approx(1 / 8)
    assert "train/rl/sign_protected_turn_fraction" in evidence


def test_goal_credit_scale_requires_goal_credit_and_a_positive_value():
    with pytest.raises(ValueError, match="requires goal credit"):
        _settings(goal_credit="none", goal_credit_scale=2.0)
    with pytest.raises(ValueError, match="finite and positive"):
        _settings(goal_credit="verified-sign", goal_credit_scale=0.0)


def _keyed(suffix, reward, state, *, first_exact=None):
    rollout = _rollout(suffix, reward=reward)
    first, second = rollout.turns
    return replace_turns(
        rollout,
        (
            AgenticTurn(
                first.completion_start,
                first.completion_end,
                first_exact or first.anchor_state_key,
                0.0,
                state_key=state,
            ),
            AgenticTurn(
                second.completion_start,
                second.completion_end,
                second.anchor_state_key,
                reward,
                state_key=f"{state}-later",
            ),
        ),
    )


def replace_turns(rollout, turns):
    from dataclasses import replace

    return replace(rollout, turns=turns)


def test_anchor_fallback_compares_would_be_singletons_that_share_a_state():
    rollouts = (_keyed("a", 1.0, "s1"), _keyed("b", 0.0, "s1"), _keyed("c", 0.5, "s2"), _keyed("d", 0.5, "s3"))
    off = compute_sampo_advantages(_settings(goal_credit="none"), ("task-1",) * 4, rollouts)
    on = compute_sampo_advantages(
        _settings(goal_credit="none", anchor_fallback="environment-state"), ("task-1",) * 4, rollouts
    )
    assert all(value == 0 for values in off.turn_advantages for value in values)  # every exact anchor is a singleton
    # a and b reached state s1 by different routes; their first turns are now compared.
    assert on.turn_advantages[0][0] > 0 > on.turn_advantages[1][0]
    assert on.anchor_group_sizes[0][0] == 2 and on.anchor_group_sizes[2][0] == 1
    assert on.fallback_anchor_turns[0] == (True, True)
    assert on.hierarchy_evidence(1.0)["train/rl/fallback_anchor_turn_fraction"][0] == 1.0


def test_anchor_fallback_leaves_exact_groups_alone():
    rollouts = (
        _keyed("a", 1.0, "s1", first_exact="shared"),
        _keyed("b", 0.0, "s1", first_exact="shared"),
        _keyed("c", 0.5, "s1"),
        _keyed("d", 0.5, "s9"),
    )
    on = compute_sampo_advantages(
        _settings(goal_credit="none", anchor_fallback="environment-state"), ("task-1",) * 4, rollouts
    )
    # a and b share an exact anchor and stay a pair; c's first turn has no exact sibling and no
    # other fallback turn with state s1, so it stays alone rather than joining the exact pair.
    assert on.anchor_group_sizes[0][0] == 2 and on.anchor_group_sizes[1][0] == 2
    assert on.anchor_group_sizes[2][0] == 1
    assert on.fallback_anchor_turns[0][0] is False and on.fallback_anchor_turns[2][0] is True
