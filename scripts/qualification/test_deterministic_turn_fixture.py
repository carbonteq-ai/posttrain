"""Independent expected fixtures, including observation and template holes."""

from types import SimpleNamespace

import pytest
from deterministic_turn_fixture import ANNOTATION_KEY, SCORER_DIGEST, DeterministicTurnFixture
from posttrain.common import TraceObservation
from posttrain.train import RewardComponentProjection, RewardProjection
from posttrain.train.turn_rewards import native_turn_map


def trace(rollout_id):
    nodes = [
        SimpleNamespace(message={"role": "user"}, sampled=False, token_ids=[1], mask=[False]),
        SimpleNamespace(message={"role": "assistant"}, sampled=True, token_ids=[2, 3, 4], mask=[True, False, True]),
        SimpleNamespace(message={"role": "tool"}, sampled=False, token_ids=[5], mask=[False]),
        SimpleNamespace(message={"role": "assistant"}, sampled=True, token_ids=[6], mask=[True]),
    ]
    branch = SimpleNamespace(
        nodes=nodes, sampled_mask=[False, True, False, True, False, True], token_ids=[1, 2, 3, 4, 5, 6]
    )
    return SimpleNamespace(id="trace", branches=[branch], reward=0.75, info={"posttrain_rollout_id": rollout_id})


def test_repeatable_complete_evidence_without_mutating_native_rewards_or_tokens():
    first, second = trace("rollout"), trace("rollout")
    fixture = DeterministicTurnFixture()
    fixture(first)
    fixture(second)
    assert first.info == second.info
    assert first.reward == 0.75
    assert first.branches[0].token_ids == [1, 2, 3, 4, 5, 6]
    assert first.branches[0].sampled_mask == [False, True, False, True, False, True]
    panel = first.info[ANNOTATION_KEY]
    assert panel["synthetic"] is True
    assert [item["turn_id"] for item in panel["assessments"]] == ["assistant-0", "assistant-1"]
    with pytest.raises(ValueError, match="already exists"):
        fixture(first)


def test_four_cases_and_signed_scores_are_exercised():
    panels = {}
    for index in range(100):
        item = trace(str(index))
        DeterministicTurnFixture()(item)
        panel = item.info[ANNOTATION_KEY]
        panels[panel["case"]] = panel
    assert set(panels) == {0, 1, 2, 3}
    expected_scores = {0: [0.25, -0.5], 1: [-0.75, 1.0], 2: [0.0, 0.0], 3: [1.0, -0.25]}
    expected_errors = {0: [], 1: ["assistant-0"], 2: ["assistant-0", "assistant-1"], 3: ["assistant-1"]}
    for case, panel in panels.items():
        assert [item["components"][0]["value"] for item in panel["assessments"]] == expected_scores[case]
        assert panel["erroneous_turn_ids"] == expected_errors[case]


def test_fixture_projection_never_credits_observation_or_template_holes():
    item = trace("rollout")
    item.info["posttrain_prompt_group_id"] = "group"
    DeterministicTurnFixture()(item)
    projection = RewardProjection(
        "fixture",
        "1",
        (RewardComponentProjection("score", "turn_mean", "fixture_a"),),
        scorer_digest=SCORER_DIGEST,
        turns_info_key=ANNOTATION_KEY,
        turn_error_key="erroneous_turn_ids",
    )
    turns = native_turn_map(item.branches[0])
    result = projection.project(
        TraceObservation(trace_type="verifiers", external_id=item.id, payload={"info": item.info}),
        scalar_reward=item.reward,
        turn_ids=tuple(turn.id for turn in turns),
        native_turns=turns,
    )
    assert result.process is not None
    mask = result.process.error_mask((True, False, True, False, True))
    assert mask[1] is False and mask[3] is False
    errors = item.info[ANNOTATION_KEY]["erroneous_turn_ids"]
    assert mask == ("assistant-0" in errors, False, "assistant-0" in errors, False, "assistant-1" in errors)
