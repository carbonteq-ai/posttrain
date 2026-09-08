from types import SimpleNamespace

import pytest
from posttrain.train.backends.trl.policy_config import _rollout_execution_config


def request(*, options=None, max_concurrent=32, backend="vllm@0.25.1", mode="colocate", sleep=True):
    return SimpleNamespace(
        training=SimpleNamespace(backend_options=options or {}),
        inference=SimpleNamespace(
            backend=backend,
            engine={"mode": mode, "sleep_during_optimization": sleep},
        ),
        bridge=SimpleNamespace(max_concurrent=max_concurrent),
    )


def test_trl_rollout_execution_is_opt_in_and_capacity_bounded():
    assert _rollout_execution_config(request()) is None
    execution = _rollout_execution_config(
        request(
            options={
                "rollout_execution": {
                    "env_workers": 4,
                    "episodes_per_worker": 8,
                    "worker_native_threads": 1,
                }
            }
        )
    )
    assert execution is not None
    assert execution.episode_capacity == 32

    with pytest.raises(ValueError, match="exceeds the global rollout limit"):
        _rollout_execution_config(
            request(
                max_concurrent=31,
                options={
                    "rollout_execution": {
                        "env_workers": 4,
                        "episodes_per_worker": 8,
                        "worker_native_threads": 1,
                    }
                },
            )
        )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"env_workers": 1}, "missing"),
        (
            {"env_workers": 1, "episodes_per_worker": 1, "worker_native_threads": 1, "extra": 1},
            "unknown",
        ),
        ({"env_workers": True, "episodes_per_worker": 1, "worker_native_threads": 1}, "must be an integer"),
    ],
)
def test_trl_rollout_execution_rejects_ambiguous_topologies(values, message):
    with pytest.raises(ValueError, match=message):
        _rollout_execution_config(request(options={"rollout_execution": values}))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"backend": "transformers@5.0.0"}, "requires a vLLM"),
        ({"mode": "server"}, "requires colocated vLLM"),
        ({"sleep": False}, "sleep_during_optimization=true"),
        ({"max_concurrent": None}, "declare max_concurrent"),
    ],
)
def test_trl_rollout_execution_rejects_unsupported_runtime_ownership(overrides, message):
    values = {"env_workers": 1, "episodes_per_worker": 1, "worker_native_threads": 1}
    with pytest.raises(ValueError, match=message):
        _rollout_execution_config(request(options={"rollout_execution": values}, **overrides))
