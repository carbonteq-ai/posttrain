"""Complete-group replacement and cross-rank failure agreement."""

import asyncio
from dataclasses import replace

import pytest
from posttrain.common import TraceObservation
from posttrain.train.online_rl import EnvironmentRollout, PartialRolloutBatchError, RolloutBatch
from posttrain.train.profiles import ActiveGroupSampling, GDPOSettings, GRPOSettings, TrainingLoop
from posttrain.train.reward_admission import admit_reward_groups, admit_rollout_groups
from posttrain.train.reward_evidence import InvalidRewardEvidence, RewardEvidence, RewardValue


def settings():
    return GDPOSettings(
        id="test",
        component_names=("score",),
        component_weights=(1.0,),
        num_prompts_per_step=2,
        loop=TrainingLoop(max_steps=1, gradient_accumulation_steps=4),
    )


def batch():
    return RolloutBatch(("a", "a", "b", "b"), 1, "policy", ("g1", "g1", "g2", "g2"), ("r1", "r2", "r3", "r4"))


def rows(selected):
    result = []
    for task, group, identity in zip(
        selected.example_ids, selected.prompt_group_ids, selected.rollout_ids, strict=True
    ):
        trace = TraceObservation(trace_type="fixture", external_id=identity, payload={})
        evidence = RewardEvidence(group, identity, identity, "0", "projection@1", (RewardValue("score", "valid", 0.0),))
        result.append(
            EnvironmentRollout(task, (1,), (2,), (-1.0,), (True,), 0.0, False, trace, reward_evidence=evidence)
        )
    return result


def test_replace_whole_bad_group_but_retain_other_zero_reward_group():
    calls = []

    def collect(selected):
        calls.append(selected)
        result = rows(selected)
        if len(calls) == 1:
            result[0] = replace(result[0], is_truncated=True)
        return result

    result = admit_reward_groups(batch(), settings(), collect, lambda failures: failures)
    assert result.rounds == 2
    assert result.attempted_rollouts == 6
    assert calls[1].example_ids == ("a", "a")
    evidence = [row.reward_evidence for row in result.rollouts]
    assert all(item is not None for item in evidence)
    assert [item.rollout_id for item in evidence if item is not None] == [
        "r1/attempt/1",
        "r2/attempt/1",
        "r3/attempt/0",
        "r4/attempt/0",
    ]


def test_local_exception_is_exchanged_before_retry_and_exhaustion():
    exchanged = []

    def collect(selected):
        raise RuntimeError("local detail")

    def gather(failures):
        exchanged.append(failures)
        return failures

    with pytest.raises(InvalidRewardEvidence, match="exhausted 3 attempts"):
        admit_reward_groups(batch(), settings(), collect, gather)
    assert len(exchanged) == 3
    assert {group for group, _ in exchanged[0]} == {"g1", "g2"}


def test_partial_batch_failure_retries_only_its_complete_prompt_group():
    calls = []

    def collect(selected):
        calls.append(selected)
        result = rows(selected)
        if len(calls) == 1:
            raise PartialRolloutBatchError(
                "one occurrence failed",
                completed={0: result[0], 2: result[2], 3: result[3]},
                failures={1: "judge_timeout"},
            )
        return result

    result = admit_reward_groups(batch(), settings(), collect, lambda failures: failures)
    assert result.rounds == 2
    assert result.attempted_rollouts == 6
    assert calls[1].example_ids == ("a", "a")
    evidence = [row.reward_evidence for row in result.rollouts]
    assert [item.rollout_id for item in evidence if item is not None] == [
        "r1/attempt/1",
        "r2/attempt/1",
        "r3/attempt/0",
        "r4/attempt/0",
    ]


def test_scalar_group_admission_replaces_only_group_with_failed_occurrence():
    calls = []
    scalar_settings = GRPOSettings(
        id="scalar",
        num_prompts_per_step=2,
        num_generations=2,
        max_admission_attempts=2,
        loop=TrainingLoop(max_steps=1, gradient_accumulation_steps=4),
    )

    def scalar_rows(selected):
        result = []
        for task, identity in zip(selected.example_ids, selected.rollout_ids, strict=True):
            trace = TraceObservation(trace_type="fixture", external_id=identity, payload={})
            result.append(EnvironmentRollout(task, (1,), (2,), (-1.0,), (True,), 0.0, False, trace))
        return result

    def collect(selected):
        calls.append(selected)
        result = scalar_rows(selected)
        if len(calls) == 1:
            raise PartialRolloutBatchError(
                "one occurrence failed",
                completed={0: result[0], 2: result[2], 3: result[3]},
                failures={1: "episode_timeout"},
            )
        return result

    result = admit_rollout_groups(batch(), scalar_settings, collect, lambda failures: failures)

    assert result.rounds == 2
    assert result.retained_positions == (0, 1, 2, 3)
    assert result.attempted_rollouts == 6
    assert result.rejected_groups == 1
    assert result.failed_rollouts == 1
    assert calls[1].example_ids == ("a", "a")
    assert [row.trace.external_id for row in result.rollouts] == [
        "r1/attempt/1",
        "r2/attempt/1",
        "r3/attempt/0",
        "r4/attempt/0",
    ]


@pytest.mark.parametrize("algorithm", ["grpo", "olmo3"])
def test_active_sampling_drops_failed_complete_group_for_outer_refill(algorithm):
    scalar_settings = GRPOSettings(
        id="olmo3",
        algorithm=algorithm,
        num_prompts_per_step=2,
        num_generations=2,
        max_admission_attempts=2,
        advantage_scaling="none",
        importance_sampling_mode="token_truncate",
        importance_sampling_clip_min=None,
        importance_sampling_clip_max=2.0,
        active_sampling=ActiveGroupSampling(max_candidate_batches=2) if algorithm == "olmo3" else None,
        loop=TrainingLoop(max_steps=1, gradient_accumulation_steps=4),
    )
    selected = batch()
    result_rows = rows(selected)

    def collect(_selected):
        raise PartialRolloutBatchError(
            "one occurrence failed",
            completed={0: result_rows[0], 2: result_rows[2], 3: result_rows[3]},
            failures={1: "episode_error"},
        )

    result = admit_rollout_groups(
        selected,
        scalar_settings,
        collect,
        lambda failures: failures,
        max_attempts=1,
        retain_complete_on_exhaustion=True,
    )

    assert result.rounds == 1
    assert result.attempted_rollouts == 4
    assert result.failed_rollouts == 1
    assert result.rejected_groups == 1
    assert result.retained_positions == (2, 3)
    assert [rollout.example_id for rollout in result.rollouts] == ["b", "b"]


def test_structured_admission_drops_failed_complete_group_without_retry():
    selected = batch()

    def collect(current):
        result_rows = rows(current)
        raise PartialRolloutBatchError(
            "one structured occurrence failed",
            completed={0: result_rows[0], 2: result_rows[2], 3: result_rows[3]},
            failures={1: "judge_timeout"},
        )

    result = admit_rollout_groups(
        selected,
        settings(),
        collect,
        lambda failures: failures,
        max_attempts=1,
        retain_complete_on_exhaustion=True,
    )

    assert result.rounds == 1
    assert result.attempted_rollouts == 4
    assert result.failed_rollouts == 1
    assert result.rejected_groups == 1
    assert result.retained_positions == (2, 3)
    assert [rollout.example_id for rollout in result.rollouts] == ["b", "b"]


def test_rank_without_pending_rows_still_participates_in_failure_exchange():
    calls = []
    exchanges = []

    def collect(selected):
        calls.append(selected)
        return rows(selected)

    def gather(failures):
        exchanges.append(failures)
        return [("remote-group", "missing evidence")] if len(exchanges) == 1 else []

    result = admit_reward_groups(batch(), settings(), collect, gather)
    assert len(calls) == 1
    assert len(exchanges) == 2
    assert result.rounds == 2


def test_cancellation_is_not_a_failed_reward():
    def collect(selected):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        admit_reward_groups(batch(), settings(), collect, lambda failures: failures)
