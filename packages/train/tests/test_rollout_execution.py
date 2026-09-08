from types import SimpleNamespace

import pytest
from posttrain.train.rollout_execution import (
    CollectionExecutionError,
    CollectionKey,
    EpisodeKey,
    EpisodeOutcome,
    EpisodeStatus,
    RolloutExecutionConfig,
    validate_execution_config,
    validate_outcome_identity,
)


def key(*, example_id: str = "task-1") -> EpisodeKey:
    return EpisodeKey(
        collection=CollectionKey("run-1", "collection-1", "policy-1"),
        example_id=example_id,
        group_id="group-1",
        occurrence_id="rollout-1",
        seed=7,
    )


def test_execution_config_keeps_worker_capacity_inside_global_limit():
    config = RolloutExecutionConfig(env_workers=4, episodes_per_worker=8)
    validate_execution_config(config, global_limit=32, effective_cpus=4)

    with pytest.raises(ValueError, match="exceeds the global rollout limit"):
        validate_execution_config(config, global_limit=31, effective_cpus=4)
    with pytest.raises(ValueError, match="native-thread reservation"):
        validate_execution_config(config, global_limit=32, effective_cpus=3)


def test_completed_outcome_requires_scheduled_identity():
    scheduled = key()
    completed = EpisodeOutcome(
        key=scheduled,
        status=EpisodeStatus.COMPLETED,
        rollout=SimpleNamespace(example_id="task-1", reward_evidence=None),
    )
    validate_outcome_identity(scheduled, completed)

    mismatched = EpisodeOutcome(
        key=scheduled,
        status=EpisodeStatus.COMPLETED,
        rollout=SimpleNamespace(example_id="task-2", reward_evidence=None),
    )
    with pytest.raises(CollectionExecutionError, match="example id"):
        validate_outcome_identity(scheduled, mismatched)


def test_terminal_outcomes_do_not_fabricate_rollouts():
    with pytest.raises(ValueError, match="completed"):
        EpisodeOutcome(key=key(), status=EpisodeStatus.COMPLETED)
    with pytest.raises(ValueError, match="non-completed"):
        EpisodeOutcome(
            key=key(),
            status=EpisodeStatus.FAILED,
            rollout=SimpleNamespace(example_id="task-1"),
            error="tool failed",
        )
