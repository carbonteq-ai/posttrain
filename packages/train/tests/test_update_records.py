"""Evidence contracts reject lost provenance before any backend is activated."""

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
from posttrain.common import TraceObservation
from posttrain.train.reward_evidence import InvalidRewardEvidence, RewardValue, SpanAssessment
from posttrain.train.update_records import (
    ActionInterval,
    ActionRef,
    ActionSelection,
    ConditioningView,
    InvalidPolicyUpdate,
    PolicyVersions,
    PopulationRelation,
    PopulationSnapshot,
    SemanticSpan,
)


def view(name: str, size: int, sampled: tuple[int, ...]) -> ConditioningView:
    return ConditioningView(
        name,
        "native:episode",
        f"tokens:{name}",
        f"mask:{name}",
        f"positions:{name}",
        "template@1",
        f"digest:{name}",
        size,
        "ep",
        "branch",
        sampled,
    )


def population() -> PopulationSnapshot:
    # Three positions: "original" samples native tokens 0 and 2 (token 1 is an
    # observation, hence absent from eligible actions); "rolling-window" samples
    # token 0 under its own, shorter conditioning context.
    original, window = view("original", 6, (0, 2)), view("rolling-window", 4, (0,))
    first, second, third = (
        ActionRef("ep", "branch", "original", 0),
        ActionRef("ep", "branch", "original", 2),
        ActionRef("ep", "branch", "rolling-window", 0),
    )
    return PopulationSnapshot(
        "population",
        "native:episode",
        "native-digest",
        (original, window),
        (
            SemanticSpan(
                "thinking",
                "reasoning",
                "extract@1",
                (ActionInterval(first, 1), ActionInterval(second, 3), ActionInterval(third, 1)),
            ),
            SemanticSpan("step-2", "reasoning", "extract@1", (ActionInterval(third, 1),)),
        ),
        (
            PopulationRelation(
                "prompt", "prompt-group", ("original", "rolling-window"), "complete", ("original", "rolling-window")
            ),
            PopulationRelation("anchor", "anchor-state", ("rolling-window",), "complete", ("rolling-window",)),
        ),
        PolicyVersions("sampler@1", "old@1", "current@1", "reference@1"),
        "selector@1",
    )


def test_overlapping_multi_interval_spans_select_original_actions_once() -> None:
    snapshot = population()
    assert snapshot.size == 3
    assert snapshot.select_spans(("thinking", "step-2")).tolist() == [True, True, True]
    assert snapshot.select_spans(("step-2",)).tolist() == [False, False, True]
    assert snapshot.select_roles(("reasoning",)).tolist() == [True, True, True]
    assert ActionSelection("spans", ("step-2",)).resolve(snapshot).tolist() == [False, False, True]
    assert ActionSelection().resolve(snapshot).all()
    # Canonical order: views in admission order, native indices ascending within a view.
    assert [(action.turn_id, action.token_index) for action in snapshot.actions()] == [
        ("original", 0),
        ("original", 2),
        ("rolling-window", 0),
    ]
    assert snapshot.action(1) == ActionRef("ep", "branch", "original", 2)
    assert snapshot.positions(tuple(reversed(snapshot.actions()))).tolist() == [2, 1, 0]
    assert snapshot.view_of.tolist() == [0, 0, 1]
    assert snapshot.view_positions(1) == slice(2, 3)
    assert snapshot.relations[1].members == snapshot.relations[0].members[1:]
    assert snapshot.digest == population().digest
    assert replace(snapshot, versions=replace(snapshot.versions, current="current@2")).digest != snapshot.digest


def test_spans_cannot_select_observation_or_another_branch() -> None:
    snapshot = population()
    for action in (ActionRef("ep", "branch", "original", 1), ActionRef("ep", "other", "original", 0)):
        with pytest.raises(InvalidPolicyUpdate, match="ineligible"):
            replace(snapshot, spans=(SemanticSpan("bad", "reasoning", "extract@1", (ActionInterval(action, 2),)),))
    with pytest.raises(InvalidPolicyUpdate, match="ineligible"):
        snapshot.positions((ActionRef("ep", "branch", "original", 1),))


def test_missing_actual_context_and_duplicate_actions_rejected() -> None:
    snapshot = population()
    with pytest.raises(InvalidPolicyUpdate, match="absent sampled turns"):
        replace(snapshot, conditioning=snapshot.conditioning[:1])
    with pytest.raises(InvalidPolicyUpdate, match="conditioning identities must be unique"):
        replace(snapshot, conditioning=snapshot.conditioning + snapshot.conditioning[:1])
    with pytest.raises(InvalidPolicyUpdate, match="ascending unique"):
        replace(snapshot.conditioning[0], sampled=(0, 2, 2))
    with pytest.raises(InvalidPolicyUpdate, match="ascending unique"):
        replace(snapshot.conditioning[0], sampled=())
    with pytest.raises(InvalidPolicyUpdate, match="different population evidence"):
        replace(snapshot, conditioning=(replace(snapshot.conditioning[0], native_ref="native:other"),))
    with pytest.raises(InvalidPolicyUpdate, match="unknown semantic"):
        snapshot.select_spans(("missing",))


def test_complete_relation_cannot_hide_missing_members() -> None:
    relation = population().relations[0]
    with pytest.raises(InvalidPolicyUpdate, match="lacks expected"):
        replace(relation, members=relation.members[:1])
    assert replace(relation, members=relation.members[:1], completeness="partial").completeness == "partial"
    with pytest.raises(InvalidPolicyUpdate, match="cannot duplicate"):
        replace(relation, members=relation.members * 2)


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
