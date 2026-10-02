"""Evidence contracts reject lost provenance before any backend is activated."""

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
from posttrain.common import TraceObservation
from posttrain.train.reward_evidence import InvalidRewardEvidence, RewardValue, SpanAssessment
from posttrain.train.update_records import (
    ActionInterval,
    ActionRecord,
    ActionRef,
    ConditioningView,
    InvalidPolicyUpdate,
    PolicyVersions,
    PopulationRelation,
    PopulationSnapshot,
    SemanticSpan,
)


def population() -> PopulationSnapshot:
    # Token index 1 is an observation, hence absent from eligible actions.
    actions = tuple(ActionRef("ep", "branch", "turn", index) for index in (0, 2, 3))
    contexts = tuple(
        ConditioningView(
            name,
            f"native:{name}",
            f"tokens:{name}",
            f"mask:{name}",
            f"positions:{name}",
            "template@1",
            f"digest:{name}",
            size,
        )
        for name, size in (("original", 6), ("rolling-window", 4))
    )
    return PopulationSnapshot(
        "population",
        "native:episode",
        "native-digest",
        tuple(
            ActionRecord(action, "original" if action.token_index == 0 else "rolling-window", "native:node")
            for action in actions
        ),
        contexts,
        (
            SemanticSpan(
                "thinking", "reasoning", "extract@1", (ActionInterval(actions[0], 1), ActionInterval(actions[1], 4))
            ),
            SemanticSpan("step-2", "reasoning", "extract@1", (ActionInterval(actions[2], 4),)),
        ),
        (
            PopulationRelation("prompt", "prompt-group", actions, "complete", actions),
            PopulationRelation("anchor", "anchor-state", actions[1:], "complete", actions[1:]),
        ),
        PolicyVersions("sampler@1", "old@1", "current@1", "reference@1"),
        "selector@1",
    )


def test_overlapping_multi_interval_spans_select_original_actions_once() -> None:
    snapshot = population()
    assert snapshot.select_spans(("thinking", "step-2")) == tuple(record.action for record in snapshot.actions)
    assert snapshot.actions[0].conditioning_id == "original"
    assert snapshot.actions[1].conditioning_id == "rolling-window"
    assert snapshot.relations[1].members == snapshot.relations[0].members[1:]
    assert snapshot.digest == population().digest
    assert replace(snapshot, versions=replace(snapshot.versions, current="current@2")).digest != snapshot.digest


def test_spans_cannot_select_observation_or_another_branch() -> None:
    snapshot = population()
    for action in (ActionRef("ep", "branch", "turn", 1), ActionRef("ep", "other", "turn", 0)):
        with pytest.raises(InvalidPolicyUpdate, match="ineligible"):
            replace(snapshot, spans=(SemanticSpan("bad", "reasoning", "extract@1", (ActionInterval(action, 2),)),))


def test_missing_actual_context_and_duplicate_actions_rejected() -> None:
    snapshot = population()
    with pytest.raises(InvalidPolicyUpdate, match="conditioning"):
        replace(snapshot, conditioning=snapshot.conditioning[:1])
    with pytest.raises(InvalidPolicyUpdate, match="unique eligible"):
        replace(snapshot, actions=snapshot.actions + snapshot.actions[:1])
    with pytest.raises(InvalidPolicyUpdate, match="unknown semantic"):
        snapshot.select_spans(("missing",))


def test_complete_relation_cannot_hide_missing_members() -> None:
    relation = population().relations[0]
    with pytest.raises(InvalidPolicyUpdate, match="lacks expected"):
        replace(relation, members=relation.members[:1])
    assert replace(relation, members=relation.members[:1], completeness="partial").completeness == "partial"


def test_assessment_preserves_unavailable_score_and_observation_scope() -> None:
    assessment = SpanAssessment(
        "judgment:1",
        "thinking",
        (RewardValue("quality", "abstained"),),
        "scorer@1",
        "input:prefix",
        "prefix",
        "quality",
        "scorer-config-digest",
    )
    assert assessment.components[0].value is None
    assert assessment.observation_scope == "prefix"
    with pytest.raises(InvalidRewardEvidence, match="observation scope"):
        replace(assessment, observation_scope="unknown")  # type: ignore[arg-type]
    with pytest.raises(InvalidRewardEvidence, match="unique"):
        replace(assessment, components=assessment.components * 2)


@pytest.mark.parametrize("index", [-1, True, 1.5])
def test_invalid_action_coordinate(index: int) -> None:
    with pytest.raises(InvalidPolicyUpdate, match="token index"):
        ActionRef("episode", "branch", "turn", index)


def test_native_bridge_opt_in_retains_actual_paths_without_changing_legacy_rows(monkeypatch) -> None:
    from posttrain.train.integrations.verifiers import VerifiersEnvironmentRolloutBridge

    nodes = [
        SimpleNamespace(parent=None, sampled=False, message={"role": "user"}, token_ids=[1, 2], mask=[False, False]),
        SimpleNamespace(parent=0, sampled=True, message={"role": "assistant"}, token_ids=[3, 4], mask=[False, True]),
        SimpleNamespace(parent=None, sampled=False, message={"role": "system"}, token_ids=[10], mask=[False]),
        SimpleNamespace(parent=2, sampled=False, message={"role": "tool"}, token_ids=[11, 12], mask=[False, False]),
        SimpleNamespace(parent=3, sampled=True, message={"role": "assistant"}, token_ids=[13, 14], mask=[False, True]),
    ]
    trace = SimpleNamespace(
        id="native-trace", reward=1.0, nodes=nodes, calls=[SimpleNamespace(node=1), SimpleNamespace(node=4)]
    )
    # Existing training projection may flatten selected sampled nodes. Its rows
    # must not be mistaken for the later call's original conditioning path.
    branch = SimpleNamespace(
        index=7,
        nodes=[nodes[index] for index in (0, 1, 3, 4)],
        token_ids=[1, 2, 3, 4, 11, 12, 13, 14],
        sampled_mask=[False, False, False, True, False, False, False, True],
        logprobs=[0.0] * 8,
    )
    monkeypatch.setattr("posttrain.train.integrations.verifiers._project_training_branch", lambda _: branch)
    observation = TraceObservation("verifiers", "native-trace", {}, {"is_truncated": False, "example_id": "task-1"})
    bridge = SimpleNamespace(technique="grpo", reward_projection=None, policy_update_context_contract=None)
    legacy = VerifiersEnvironmentRolloutBridge._project(
        cast(VerifiersEnvironmentRolloutBridge, bridge), trace, observation
    )
    bridge.policy_update_context_contract = "causal-text@1"
    selected = VerifiersEnvironmentRolloutBridge._project(
        cast(VerifiersEnvironmentRolloutBridge, bridge), trace, observation
    )
    assert legacy.conditioning_records == ()
    assert selected.prompt_ids == legacy.prompt_ids
    assert selected.completion_ids == legacy.completion_ids
    assert selected.env_mask == legacy.env_mask
    assert selected.selected_branch_id == "7"
    assert selected.conditioning_records[1].prefix_node_indices == (2, 3)
