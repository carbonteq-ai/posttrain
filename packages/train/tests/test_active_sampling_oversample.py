"""Active-sampling oversampling: settings, capacity guards, and TRL argument translation."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

import pytest
from posttrain.common import CatalogRef
from posttrain.train import SAMPOSettings, TrainingLoop
from posttrain.train.backends.trl.policy_config import (
    _online_rl_arguments,
    validate_oversampled_round_capacity,
)
from posttrain.train.backends.trl.policy_optimization import _trainer_arguments
from posttrain.train.catalog_schema import decode_training_selection
from posttrain.train.profiles import ActiveGroupSampling
from posttrain.train.reward_projection import RewardComponentProjection, RewardProjection
from posttrain.train.reward_recovery import reward_contract_digest
from posttrain.train.rollout_execution import oversampled_round_capacity_error


def _settings(**active_sampling: int) -> SAMPOSettings:
    # 24 prompts x 6 generations, the LFM2.5-2.6B SAMPO shape.
    return SAMPOSettings(
        id="sampo-oversample",
        loop=TrainingLoop(max_steps=1, max_length=8, per_device_batch_size=1, gradient_accumulation_steps=144),
        num_prompts_per_step=24,
        num_generations=6,
        max_prompt_length=2,
        max_completion_length=6,
        active_sampling=ActiveGroupSampling(**{"max_candidate_batches": 6, **active_sampling}),
    )


def _request(
    settings: SAMPOSettings,
    *,
    engine: dict[str, Any] | None = None,
    max_concurrent: int = 168,
    workers: tuple[int, int] | None = (12, 14),
) -> Any:
    options = {}
    if workers is not None:
        options["rollout_execution"] = {
            "env_workers": workers[0],
            "episodes_per_worker": workers[1],
            "worker_native_threads": 1,
        }
    return SimpleNamespace(
        settings=settings,
        policy=SimpleNamespace(provenance={}, form="foundation"),
        training=SimpleNamespace(
            backend="trl@1.12.0.post11",
            backend_options=options,
            update=SimpleNamespace(kind="lora"),
            runtime=SimpleNamespace(nodes=1, devices_per_node=1),
        ),
        inference=SimpleNamespace(
            backend="vllm@0.29.1",
            sampling={},
            engine={
                "mode": "colocate",
                "request_mode": "async" if workers is not None else "batch",
                "sleep_during_optimization": True,
                "max_num_seqs": 168,
                **(engine or {}),
            },
        ),
        bridge=SimpleNamespace(max_concurrent=max_concurrent),
    )


def test_oversampling_settings_count_prompt_groups_and_must_fit_the_reservation() -> None:
    assert ActiveGroupSampling(6) == ActiveGroupSampling(6, oversample=0, oversample_refill=0)
    with pytest.raises(ValueError, match="non-negative prompt groups"):
        ActiveGroupSampling(6, oversample=-1)
    with pytest.raises(ValueError, match="non-negative prompt groups"):
        ActiveGroupSampling(6, oversample_refill=-1)
    # A one-batch reservation cannot hold the target plus any extra first-round group.
    with pytest.raises(ValueError, match="needs 25 candidate prompts .* reserves 24"):
        _settings(max_candidate_batches=1, oversample=1)
    assert _settings(max_candidate_batches=2, oversample=24).active_sampling.oversample == 24
    # Refill oversampling is cut to what remains of the reservation at run time.
    assert _settings(max_candidate_batches=1, oversample_refill=5).active_sampling.oversample_refill == 5


def test_catalog_decodes_oversampling_settings() -> None:
    data = {
        "selection_type": "sampo-settings",
        "id": "sampo-oversample",
        "loop": {"max_steps": 1, "max_length": 8, "per_device_batch_size": 2},
        "max_prompt_length": 2,
        "max_completion_length": 6,
        "active_sampling": {"max_candidate_batches": 5, "oversample": 1, "oversample_refill": 2},
    }
    settings = decode_training_selection(CatalogRef("training", "sampo-oversample"), data, {})
    assert isinstance(settings, SAMPOSettings)
    assert settings.active_sampling == ActiveGroupSampling(5, oversample=1, oversample_refill=2)

    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        decode_training_selection(
            CatalogRef("training", "sampo-negative"),
            {**data, "active_sampling": {"max_candidate_batches": 5, "oversample": -1}},
            {},
        )


def _capacity(**overrides: Any) -> str | None:
    values: dict[str, Any] = {
        "num_prompts_per_step": 24,
        "num_generations": 6,
        "oversample": 4,
        "vllm_max_num_seqs": 168,
        "environment_max_concurrent": 168,
        "worker_slots": (12, 14),
    }
    values.update(overrides)
    return oversampled_round_capacity_error(**values)


def test_first_round_capacity_is_checked_against_every_limit_independently() -> None:
    assert _capacity() is None
    vllm = _capacity(vllm_max_num_seqs=144)
    assert vllm is not None
    assert "active_sampling oversample 4 needs 168 concurrent episodes" in vllm
    assert "rollout inference engine max_num_seqs is 144" in vllm
    assert "environment" not in vllm and "env_workers" not in vllm
    environment = _capacity(environment_max_concurrent=144)
    assert environment is not None and "environment max_concurrent is 144" in environment
    assert "max_num_seqs" not in environment
    workers = _capacity(worker_slots=(12, 12))
    assert workers is not None and "env_workers x episodes_per_worker is 12 x 12 = 144" in workers
    assert "max_num_seqs" not in workers
    everything = _capacity(vllm_max_num_seqs=144, environment_max_concurrent=144, worker_slots=(12, 12))
    assert everything is not None and everything.count(" is ") == 3


def test_capacity_is_not_checked_without_first_round_oversampling() -> None:
    # Exact refill never exceeds the configured batch, and refill oversampling is
    # capped at the first round at run time, so neither needs extra concurrency.
    assert _capacity(oversample=0, vllm_max_num_seqs=1, environment_max_concurrent=1, worker_slots=(1, 1)) is None
    assert _capacity(vllm_max_num_seqs=None, environment_max_concurrent=None, worker_slots=None) is None


def test_trl_receives_oversampling_only_when_selected(tmp_path) -> None:
    exact = _online_rl_arguments(_request(_settings()), tmp_path, {})
    assert "active_sampling_oversample" not in exact
    assert "active_sampling_oversample_refill" not in exact

    arguments = _online_rl_arguments(_request(_settings(oversample=4, oversample_refill=2)), tmp_path, {})
    assert arguments["active_sampling_max_batches"] == 6
    assert arguments["active_sampling_oversample"] == 4
    assert arguments["active_sampling_oversample_refill"] == 2


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"engine": {"max_num_seqs": 144}}, "engine max_num_seqs is 144"),
        ({"max_concurrent": 144, "workers": (12, 12)}, "environment max_concurrent is 144"),
        ({"workers": (12, 12)}, "12 x 12 = 144"),
    ],
)
def test_trainer_start_rejects_an_oversampled_round_the_rollout_topology_cannot_run(tmp_path, changes, message):
    request = _request(_settings(oversample=4), **changes)
    with pytest.raises(ValueError, match=message):
        _online_rl_arguments(request, tmp_path, {})


def test_trainer_start_derives_the_default_vllm_limit_and_rechecks_the_resolved_engine() -> None:
    # Without max_num_seqs TRL admits one generation batch (24 x 6) per process.
    request = _request(_settings(oversample=4), engine={"max_num_seqs": None})
    with pytest.raises(ValueError, match="max_num_seqs is 144"):
        validate_oversampled_round_capacity(request)
    fits = _request(_settings(oversample=4))
    validate_oversampled_round_capacity(fits)
    with pytest.raises(ValueError, match="max_num_seqs is 150"):
        validate_oversampled_round_capacity(fits, vllm_max_num_seqs=150)
    # Refill oversampling alone never raises the largest round.
    validate_oversampled_round_capacity(
        _request(_settings(oversample_refill=10), engine={"max_num_seqs": 144}, max_concurrent=144, workers=(12, 12))
    )


def test_earlier_trl_releases_reject_oversampling_with_the_required_version() -> None:
    @dataclass
    class ReleasedConfig:
        active_sampling_max_batches: int = 10
        extra: dict[str, Any] = field(default_factory=dict)

    request = cast(Any, SimpleNamespace(training=SimpleNamespace(backend="trl@1.12.0.post10")))
    assert isinstance(_trainer_arguments(ReleasedConfig, {"active_sampling_max_batches": 6}, request), ReleasedConfig)
    with pytest.raises(RuntimeError, match="trl@1.12.0.post10 does not provide active_sampling_oversample; .*post11"):
        _trainer_arguments(ReleasedConfig, {"active_sampling_oversample": 4}, request)


def test_reward_contract_digest_ignores_oversampling() -> None:
    from posttrain.environment import EnvironmentBinding, EnvironmentSource, SamplingPolicy, VerifiersV1ConfigActivation

    environment = EnvironmentBinding(
        "environments/turns",
        "tool-use",
        EnvironmentSource("test", "https://example.test/env", "a" * 40),
        VerifiersV1ConfigActivation({"taskset": {}}),
        SamplingPolicy(max_tokens=128),
        num_tasks=200,
    )
    projection = RewardProjection("test", "1", (RewardComponentProjection("outcome", "scalar"),))

    def digest(settings: SAMPOSettings) -> str:
        request = SimpleNamespace(
            settings=settings, environment=environment, bridge=SimpleNamespace(reward_projection=projection)
        )
        return reward_contract_digest(cast(Any, request))

    exact = digest(_settings())
    assert digest(_settings(oversample=4, oversample_refill=2)) == exact
    assert digest(_settings(max_candidate_batches=5)) != exact
