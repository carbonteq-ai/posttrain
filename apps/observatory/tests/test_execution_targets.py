"""Execution-target projection tests."""

from posttrain_observatory.execution_targets import (
    execution_target_capacity,
    execution_target_contexts,
)


def test_execution_targets_keep_roles_and_aggregate_declared_capacity() -> None:
    targets = execution_target_contexts(
        {
            "execution_targets": {
                "schema_version": 1,
                "targets": [
                    {
                        "selection_id": "targets/cuda-24gb",
                        "revision": "3",
                        "roles": ["training", "rollout_inference"],
                        "device_class": "nvidia-cuda",
                        "memory_gb": 24,
                        "placement": {"world_size": 2},
                        "host_constraints": {"driver": ">=570"},
                    }
                ],
            }
        }
    )

    assert len(targets) == 1
    assert targets[0].roles == ("rollout_inference", "training")
    assert targets[0].device_count == 2
    assert targets[0].memory_bytes_per_device == 24 * 1024**3
    assert targets[0].aggregate_memory_bytes == 48 * 1024**3
    assert execution_target_capacity(targets) == ("available", 48 * 1024**3)


def test_legacy_target_id_remains_visible_without_inventing_capacity() -> None:
    targets = execution_target_contexts(
        {
            "training": {
                "selection_id": "training/legacy",
                "resolved": {"target_id": "targets/local-cuda-8gb"},
            }
        }
    )

    assert targets[0].selection_id == "targets/local-cuda-8gb"
    assert targets[0].state == "partial"
    assert targets[0].aggregate_memory_bytes is None
    assert execution_target_capacity(targets) == ("unavailable", None)


def test_conflicting_target_capacities_are_explicitly_ambiguous() -> None:
    targets = execution_target_contexts(
        {
            "execution_targets": {
                "schema_version": 1,
                "targets": [
                    {
                        "selection_id": "targets/train",
                        "revision": "1",
                        "roles": ["training"],
                        "device_class": "cuda",
                        "memory_gb": 24,
                        "placement": {"world_size": 1},
                        "host_constraints": {},
                    },
                    {
                        "selection_id": "targets/rollout",
                        "revision": "1",
                        "roles": ["rollout_inference"],
                        "device_class": "cuda",
                        "memory_gb": 48,
                        "placement": {"world_size": 1},
                        "host_constraints": {},
                    },
                ],
            }
        }
    )

    assert execution_target_capacity(targets) == ("ambiguous", None)
