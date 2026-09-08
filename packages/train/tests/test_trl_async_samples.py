import queue
from collections import defaultdict
from dataclasses import replace
from unittest.mock import Mock

import pytest
from posttrain.common import TraceObservation
from posttrain.train.backends.trl.async_samples import (
    AsyncRolloutRecord,
    BehaviorPolicySpan,
    InvalidAsyncSampleGroup,
    project_async_group,
)
from posttrain.train.online_rl import EnvironmentRollout
from trl.experimental.async_grpo.async_grpo_trainer import RolloutQueueDataset
from trl.experimental.async_grpo.async_rollout_worker import RolloutSample


def record(
    occurrence: str,
    *,
    version: int = 4,
    end_version: int | None = None,
    example_id: str = "task-1",
    truncated: bool = False,
) -> AsyncRolloutRecord:
    rollout = EnvironmentRollout(
        example_id=example_id,
        prompt_ids=(10, 11),
        completion_ids=(20, 21, 22),
        sampling_logprobs=(-0.2, -9.0, -0.4),
        env_mask=(True, False, True),
        reward=0.75,
        is_truncated=truncated,
        trace=TraceObservation("verifiers", f"trace-{occurrence}", {}),
        behavior_policy=BehaviorPolicySpan(version, version if end_version is None else end_version),
    )
    return AsyncRolloutRecord(
        occurrence_id=occurrence,
        rollout=rollout,
        prompt_messages=({"role": "user", "content": "do the task"},),
        completion_messages=({"role": "assistant", "content": "done"},),
    )


def test_projects_exact_tokens_masks_logprobs_and_native_type():
    samples = project_async_group(
        (record("a"), record("b")),
        (1.25, -1.25),
        group_id=9,
        expected_group_size=2,
    )

    assert all(isinstance(sample, RolloutSample) for sample in samples)
    assert samples[0].input_ids == [10, 11, 20, 21, 22]
    assert samples[0].completion_mask == [0, 0, 1, 0, 1]
    assert samples[0].old_log_probs == [0.0, 0.0, -0.2, 0.0, -0.4]
    assert samples[0].model_version == 4
    assert samples[0].advantage == 1.25
    assert samples[0].metrics == {
        "reward": 0.75,
        "behavior_policy_start": 4.0,
        "behavior_policy_end": 4.0,
    }


def test_behavior_policy_span_merges_turn_provenance():
    assert BehaviorPolicySpan(4, 5).merge(BehaviorPolicySpan(3, 7)) == BehaviorPolicySpan(3, 7)


def test_rejects_rollout_without_behavior_policy_provenance():
    source = record("a")
    missing = replace(source, rollout=replace(source.rollout, behavior_policy=None))

    with pytest.raises(InvalidAsyncSampleGroup, match="behavior policy span"):
        project_async_group((missing,), (0.0,), group_id=1, expected_group_size=1)


@pytest.mark.parametrize(
    ("records", "advantages", "message"),
    [
        ((record("a"),), (0.0,), "incomplete"),
        ((record("a"), record("a")), (0.0, 0.0), "occurrence ids"),
        ((record("a"), record("b", version=5)), (0.0, 0.0), "same behavior policy"),
        ((record("a", truncated=True), record("b")), (0.0, 0.0), "truncated"),
    ],
)
def test_rejects_group_before_publishing_any_partial_samples(records, advantages, message):
    with pytest.raises(InvalidAsyncSampleGroup, match=message):
        project_async_group(records, advantages, group_id=1, expected_group_size=2)


def test_native_queue_consumer_drops_stale_sample_and_yields_fresh_sample(monkeypatch):
    # The native logger expects a trainer-created Accelerate state. Keep this
    # unit test focused on the real queue consumer's filtering behavior.
    monkeypatch.setattr("trl.experimental.async_grpo.async_grpo_trainer.logger", Mock())
    stale = project_async_group((record("old", version=1),), (0.5,), group_id=1, expected_group_size=1)[0]
    fresh = project_async_group((record("new", version=4),), (0.75,), group_id=2, expected_group_size=1)[0]
    rollout_queue: queue.Queue = queue.Queue()
    rollout_queue.put(stale)
    rollout_queue.put(fresh)
    metrics = defaultdict(list)
    dataset = RolloutQueueDataset(
        rollout_queue,
        model_version_fn=lambda: 4,
        check_health_fn=lambda _: None,
        stale_after_s=10.0,
        metrics=metrics,
        max_staleness=2,
        poll_interval_s=0.01,
    )

    yielded = next(iter(dataset))

    assert yielded["input_ids"] == fresh.input_ids
    assert yielded["completion_mask"] == fresh.completion_mask
    assert yielded["old_log_probs"] == fresh.old_log_probs
    assert yielded["advantage"] == 0.75
    assert metrics["sample/dropped_stale_total"] == [1.0]


def test_episode_may_span_updates_while_staleness_uses_group_start_version():
    samples = project_async_group(
        (record("a", version=4, end_version=5), record("b", version=4, end_version=6)),
        (0.5, -0.5),
        group_id=3,
        expected_group_size=2,
    )

    assert [item.model_version for item in samples] == [4, 4]
