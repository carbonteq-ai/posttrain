from dataclasses import replace

import pytest
from posttrain.common import TraceObservation
from posttrain.environment.verifiers_conditioning import NativeConditioningRecord
from posttrain.train.online_rl import BehaviorPolicySpan
from posttrain.train.update_credit import SampoCreditEstimator, prepare_credit
from posttrain.train.update_evidence import population_from_rollouts
from posttrain.train.update_records import InvalidPolicyUpdate, PolicyVersions

from .test_sampo import _rollout, _settings


def native_rollout(group: str, index: int, reward: float):
    identity = f"{group}-{index}"
    trace_id = f"trace-{identity}"
    return replace(
        _rollout(reward, identity),
        example_id=f"task-{group}",
        trace=TraceObservation(
            "verifiers",
            trace_id,
            {
                "info": {
                    "posttrain_episode_id": f"episode-{identity}",
                    "posttrain_prompt_group_id": group,
                    "posttrain_rollout_id": f"occurrence-{identity}",
                }
            },
        ),
        behavior_policy=BehaviorPolicySpan(3, 3),
        conditioning_records=(
            NativeConditioningRecord(trace_id, 1, (0,), (1, 2), 5, f"input-a-{identity}", "causal-text@1"),
            NativeConditioningRecord(trace_id, 4, (2, 3), (1, 2), 6, f"input-b-{identity}", "causal-text@1"),
        ),
        selected_branch_id="7",
        conditioning_completion_indices=((0, 1), (4, 5)),
    )


def assemble(rollouts):
    return population_from_rollouts(
        tuple(rollouts),
        population_id="population",
        native_evidence_ref="native:episodes",
        native_evidence_digest="native-digest",
        template_revision="template@1",
        versions=PolicyVersions("sampler@3", "old@3", "current@3", None),
        sampler_step=3,
        num_generations=2,
        selector_digest="all@1",
    )


def test_interleaved_groups_preserve_native_context_and_delegate_sampo_credit():
    a0, a1 = native_rollout("a", 0, 1.0), native_rollout("a", 1, 0.0)
    b0, b1 = native_rollout("b", 0, 0.0), native_rollout("b", 1, 1.0)
    snapshot, rows = assemble((a0, b0, a1, b1))
    assert rows.rollouts == (a0, a1, b0, b1)
    assert snapshot.size == 16
    assert rows.positions[0][2:4].tolist() == [-1, -1]
    assert rows.positions[0][4] >= 0
    action = snapshot.action(int(rows.positions[0][4]))
    assert action.token_index == 1
    assert action.branch_id == "7"
    assert snapshot.conditioning[1].context_tokens == 6
    assert '"prefix_nodes":[2,3]' in snapshot.conditioning[1].token_ids_ref
    groups = tuple(relation.id for relation in snapshot.relations if relation.kind == "prompt-group")
    credit = prepare_credit(snapshot, SampoCreditEstimator(_settings(), rows, groups))
    assert credit.advantages.tolist() == pytest.approx(
        [0.975, 0.975, 1, 1, -0.975, -0.975, -1, -1, -0.975, -0.975, -1, -1, 0.975, 0.975, 1, 1]
    )


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda row: replace(row, behavior_policy=BehaviorPolicySpan(2, 3)), "synchronous"),
        (
            lambda row: replace(
                row, conditioning_records=(), selected_branch_id=None, conditioning_completion_indices=()
            ),
            "legacy flattened",
        ),
        (lambda row: replace(row, example_id="other-task"), "example identity"),
        (
            lambda row: replace(
                row, turns=tuple(replace(turn, completion_end=turn.completion_end - 1) for turn in row.turns)
            ),
            "coordinates disagree",
        ),
    ],
)
def test_population_rejects_unproved_provenance(mutation, message):
    with pytest.raises(InvalidPolicyUpdate, match=message):
        assemble((native_rollout("a", 0, 1), mutation(native_rollout("a", 1, 0))))


def test_incomplete_and_duplicate_episodes_rejected():
    row = native_rollout("a", 0, 1)
    with pytest.raises(InvalidPolicyUpdate, match="complete prompt groups"):
        assemble((row,))
    with pytest.raises(InvalidPolicyUpdate, match="one policy trajectory"):
        assemble((row, row))


def test_fresh_population_cannot_duplicate_a_task_across_complete_groups():
    first = (native_rollout("a", 0, 1), native_rollout("a", 1, 0))
    repeated = tuple(
        replace(native_rollout("b", index, reward), example_id=first[0].example_id)
        for index, reward in enumerate((1, 0))
    )
    with pytest.raises(InvalidPolicyUpdate, match="one task in multiple prompt groups"):
        assemble(first + repeated)
