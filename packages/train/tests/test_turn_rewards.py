"""Turn scores require complete native coverage and explicit reduction."""

from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest
from posttrain.common import TraceObservation
from posttrain.train.reward_evidence import InvalidRewardEvidence, RewardStatus, RewardValue
from posttrain.train.reward_projection import RewardComponentProjection, RewardProjection
from posttrain.train.turn_rewards import TURN_PROJECTION, TurnAssessment, native_turn_map
from pydantic import TypeAdapter


def branch():
    nodes = [
        SimpleNamespace(message={"role": "user"}, sampled=False, token_ids=[1], mask=[False]),
        SimpleNamespace(message={"role": "assistant"}, sampled=True, token_ids=[2, 3, 4], mask=[True, False, True]),
        SimpleNamespace(message={"role": "tool"}, sampled=False, token_ids=[5], mask=[False]),
        SimpleNamespace(message={"role": "assistant"}, sampled=True, token_ids=[6], mask=[True]),
    ]
    return SimpleNamespace(
        nodes=nodes,
        sampled_mask=[value for node in nodes for value in node.mask],
        token_ids=[value for node in nodes for value in node.token_ids],
    )


def test_native_turns_keep_noncontiguous_policy_positions_and_observation_holes():
    turns = native_turn_map(branch())
    assert [turn.id for turn in turns] == ["assistant-0", "assistant-1"]
    assert turns[0].token_spans == ((0, 1), (2, 3))
    assert turns[1].token_spans == ((4, 5),)
    assert [turn.node_index for turn in turns] == [1, 3]


def selection(reduction: Literal["turn_mean", "turn_sum"] = "turn_mean"):
    return RewardProjection(
        "turns",
        "1",
        (RewardComponentProjection("quality", reduction, "custom"),),
        scorer_digest="a" * 64,
        turns_info_key="ratings",
    )


def observation(ids=("assistant-1", "assistant-0"), status: RewardStatus = "valid", **overrides):
    envelope = {
        "trace_id": "trace",
        "branch_id": "0",
        "projection_id": TURN_PROJECTION,
        "scorer_digest": "a" * 64,
        "assessments": [
            asdict(
                TurnAssessment(
                    turn_id, (RewardValue("custom", status, score if status == "valid" else None),), "native/ratings"
                )
            )
            for turn_id, score in zip(ids, (0.0, 1.0), strict=False)
        ],
        **overrides,
    }
    return TraceObservation(
        trace_type="verifiers",
        external_id="trace",
        payload={
            "info": {
                "posttrain_prompt_group_id": "group",
                "posttrain_rollout_id": "rollout",
                "posttrain_scorer_digest": "a" * 64,
                "ratings": envelope,
            }
        },
    )


@pytest.mark.parametrize(("reduction", "expected"), [("turn_mean", 0.5), ("turn_sum", 1.0)])
def test_reduction_is_explicit_and_judge_order_does_not_change_native_identity(reduction, expected):
    result = selection(reduction).project(observation(), scalar_reward=99, turn_ids=("assistant-0", "assistant-1"))
    assert result.require_components(("quality",)) == (expected,)
    adapter = TypeAdapter(RewardProjection)
    assert adapter.validate_json(adapter.dump_json(selection(reduction))) == selection(reduction)


@pytest.mark.parametrize("ids", [("assistant-0",), ("assistant-0", "assistant-0"), ("assistant-0", "foreign")])
def test_incomplete_duplicate_or_foreign_turn_ids_fail(ids):
    with pytest.raises(InvalidRewardEvidence, match="exactly once"):
        selection().project(observation(ids), scalar_reward=0, turn_ids=("assistant-0", "assistant-1"))


@pytest.mark.parametrize("status", ["abstained", "failed", "inapplicable"])
def test_unavailable_turn_score_is_not_replaced_by_zero(status):
    with pytest.raises(InvalidRewardEvidence, match="no valid"):
        selection().project(observation(status=status), scalar_reward=0, turn_ids=("assistant-0", "assistant-1"))


@pytest.mark.parametrize("field", ["trace_id", "branch_id", "projection_id", "scorer_digest"])
def test_foreign_evidence_identity_fails(field):
    with pytest.raises(InvalidRewardEvidence, match="identity"):
        selection().project(observation(**cast(dict[str, Any], {field: "foreign"})),
                            scalar_reward=0, turn_ids=("assistant-0", "assistant-1"))


def test_mask_disagreement_and_nonpolicy_ownership_fail():
    native = branch()
    native.sampled_mask[-1] = False
    with pytest.raises(InvalidRewardEvidence):
        native_turn_map(native)
    native = branch()
    native.nodes[1].message["role"] = "tool"
    with pytest.raises(InvalidRewardEvidence, match="only sampled assistant"):
        native_turn_map(native)


def test_matching_masks_cannot_hide_changed_token_identities():
    native = branch()
    native.token_ids[1] = 999
    with pytest.raises(InvalidRewardEvidence, match="token identities disagree"):
        native_turn_map(native)


def test_direct_turn_selection_preserves_distinct_scores_without_group_identity():
    projection = replace(selection(), turn_reward_key="custom", turn_reward_includes_terminal_outcome=False)
    trace = observation()
    info = trace.payload["info"]
    assert isinstance(info, dict)
    info.pop("posttrain_prompt_group_id")
    info.pop("posttrain_rollout_id")
    assert projection.project_turn_rewards(trace, ("assistant-0", "assistant-1")) == (1.0, 0.0)
    assert selection().project_turn_rewards(trace, ()) is None


def test_selected_direct_turn_rewards_never_fall_back_on_partial_coverage():
    projection = replace(selection(), turn_reward_key="custom", turn_reward_includes_terminal_outcome=False)
    with pytest.raises(InvalidRewardEvidence, match="exactly once"):
        projection.project_turn_rewards(observation(("assistant-0",)), ("assistant-0", "assistant-1"))


def test_namespaced_judges_do_not_overwrite_each_others_identity():
    trace = observation()
    info = trace.payload["info"]
    assert isinstance(info, dict)
    info.pop("posttrain_scorer_digest")
    info["posttrain_scorer_digests"] = {"ratings": "a" * 64, "other": "b" * 64}
    assert selection().project(trace, scalar_reward=0, turn_ids=("assistant-0", "assistant-1")).require_components(
        ("quality",)
    ) == (0.5,)


def test_direct_turn_selection_requires_explicit_terminal_outcome_semantics():
    with pytest.raises(InvalidRewardEvidence, match="declare whether"):
        replace(selection(), turn_reward_key="custom")


def test_capo_maps_error_turn_ids_to_native_sampled_union_without_observations():
    projection = replace(selection(), turn_error_key="errors")
    turns = native_turn_map(branch())
    result = projection.project(
        observation(errors=["assistant-0"]),
        scalar_reward=1,
        turn_ids=tuple(turn.id for turn in turns),
        native_turns=turns,
    )
    assert result.process is not None
    assert result.process.error_spans == ((0, 1), (2, 3))
    assert result.process.error_mask((True, False, True, False, True)) == (True, False, True, False, False)


@pytest.mark.parametrize("errors", [None, ["foreign"], ["assistant-0", "assistant-0"], [True]])
def test_capo_does_not_accept_missing_duplicate_or_unknown_turn_errors(errors):
    turns = native_turn_map(branch())
    with pytest.raises(InvalidRewardEvidence, match="unique known native"):
        replace(selection(), turn_error_key="errors").project(
            observation(errors=errors),
            scalar_reward=1,
            turn_ids=tuple(turn.id for turn in turns),
            native_turns=turns,
        )
