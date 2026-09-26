from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from posttrain.common import TraceObservation
from posttrain.train import (
    AgenticTurn,
    EnvironmentRollout,
    SAMPOSettings,
    TrainingLoop,
    compute_sampo_advantages,
)
from posttrain.train.api import _sampo_backend
from posttrain.train.backends.trl.policy_config import _online_rl_arguments


def _settings(**changes) -> SAMPOSettings:
    values = {
        "id": "sampo-test",
        "loop": TrainingLoop(max_steps=1, max_length=8, per_device_batch_size=2),
        "max_prompt_length": 2,
        "max_completion_length": 6,
    }
    values.update(changes)
    return SAMPOSettings(**values)


def _rollout(reward: float, suffix: str, *, step_rewards=(None, None)) -> EnvironmentRollout:
    return EnvironmentRollout(
        example_id="task-1",
        prompt_ids=(1, 2),
        completion_ids=(3, 4, 5, 6, 7, 8),
        sampling_logprobs=(-0.1,) * 6,
        env_mask=(True, True, False, False, True, True),
        reward=reward,
        is_truncated=False,
        trace=TraceObservation("test", f"trace-{suffix}", {}),
        turns=(
            AgenticTurn(0, 2, "observation-a", step_rewards[0]),
            AgenticTurn(4, 6, "observation-b", step_rewards[1]),
        ),
    )


def test_sampo_combines_episode_and_anchor_relative_sparse_turn_advantages() -> None:
    result = compute_sampo_advantages(
        _settings(),
        ("task-1", "task-1"),
        (_rollout(1.0, "good"), _rollout(0.0, "bad")),
    )

    assert result.episode_advantages == pytest.approx((0.5, -0.5))
    assert result.turn_advantages[0] == pytest.approx((0.475, 0.5))
    assert result.turn_advantages[1] == pytest.approx((-0.475, -0.5))
    assert result.anchor_group_sizes == ((2, 2), (2, 2))
    assert result.used_sparse_rewards == (True, True)
    assert result.token_advantages[0] == pytest.approx((0.975, 0.975, 0.0, 0.0, 1.0, 1.0))
    assert result.token_advantages[1] == pytest.approx((-0.975, -0.975, 0.0, 0.0, -1.0, -1.0))


def test_sampo_rejects_partial_step_reward_evidence() -> None:
    with pytest.raises(ValueError, match="complete or entirely absent"):
        compute_sampo_advantages(
            _settings(),
            ("task-1", "task-1"),
            (
                _rollout(1.0, "good", step_rewards=(0.2, None)),
                _rollout(0.0, "bad"),
            ),
        )


def test_sampo_mean_std_uses_group_sample_standard_deviation() -> None:
    result = compute_sampo_advantages(
        _settings(advantage_normalization="mean_std"),
        ("task-1", "task-1"),
        (_rollout(1.0, "good"), _rollout(0.0, "bad")),
    )

    assert result.episode_advantages == pytest.approx((2**-0.5, -(2**-0.5)), rel=1e-5)


def test_sampo_requires_complete_generation_groups() -> None:
    with pytest.raises(ValueError, match="complete prompt groups"):
        compute_sampo_advantages(_settings(), ("task-1",), (_rollout(1.0, "one"),))


def test_sampo_keeps_repeated_example_occurrences_as_distinct_prompt_groups() -> None:
    settings = _settings(
        num_prompts_per_step=2,
        loop=TrainingLoop(max_steps=1, max_length=8, per_device_batch_size=4),
    )
    result = compute_sampo_advantages(
        settings,
        ("task-1",) * 4,
        (
            _rollout(1.0, "first-good"),
            _rollout(0.0, "first-bad"),
            _rollout(0.0, "second-bad"),
            _rollout(1.0, "second-good"),
        ),
    )

    assert result.episode_advantages == pytest.approx((0.5, -0.5, -0.5, 0.5))
    assert result.anchor_group_sizes == ((2, 2), (2, 2), (2, 2), (2, 2))


def test_agentic_turns_must_cover_only_sampled_tokens() -> None:
    with pytest.raises(ValueError, match="model-sampled"):
        EnvironmentRollout(
            example_id="task-1",
            prompt_ids=(1,),
            completion_ids=(2, 3),
            sampling_logprobs=(-0.1, 0.0),
            env_mask=(True, False),
            reward=1.0,
            is_truncated=False,
            trace=TraceObservation("test", "trace-invalid", {}),
            turns=(AgenticTurn(0, 2, "observation"),),
        )


def test_sampo_turns_must_cover_every_sampled_token() -> None:
    incomplete = EnvironmentRollout(
        example_id="task-1",
        prompt_ids=(1,),
        completion_ids=(2, 3),
        sampling_logprobs=(-0.1, -0.1),
        env_mask=(True, True),
        reward=1.0,
        is_truncated=False,
        trace=TraceObservation("test", "trace-incomplete", {}),
        turns=(AgenticTurn(0, 1, "observation"),),
    )

    with pytest.raises(ValueError, match="cover every sampled"):
        compute_sampo_advantages(
            _settings(
                loop=TrainingLoop(max_steps=1, max_length=8, per_device_batch_size=2),
            ),
            ("task-1", "task-1"),
            (incomplete, incomplete),
        )


def test_trl_sampo_selects_sequence_clipping_and_precomputed_advantages(tmp_path) -> None:
    request = cast(
        Any,
        SimpleNamespace(
            settings=_settings(),
            policy=SimpleNamespace(provenance={}),
            training=SimpleNamespace(backend_options={}),
            inference=SimpleNamespace(backend="transformers@1", sampling={}, engine={}),
        ),
    )

    arguments = _online_rl_arguments(request, tmp_path, {})

    assert arguments["loss_type"] == "grpo"
    assert arguments["importance_sampling_level"] == "sequence"
    assert arguments["use_precomputed_advantages"] is True
    assert arguments["dynamic_sampling"] is False
    assert arguments["active_sampling"] is True
    assert arguments["epsilon"] == 0.003
    assert arguments["epsilon_high"] == 0.004


def test_verl_sampo_uses_the_hierarchical_backend_adapter() -> None:
    assert _sampo_backend("verl@candidate").__module__ == "posttrain.train.backends.verl.launcher"


def test_sampo_refills_like_vortex_with_active_sampling() -> None:
    from posttrain.train.profiles import ActiveGroupSampling

    assert _settings().active_sampling == ActiveGroupSampling(3)
    assert _settings(active_sampling=ActiveGroupSampling(max_candidate_batches=4)).max_collection_attempts == 4
    assert not hasattr(_settings(), "dynamic_sampling")
    with pytest.raises(ValueError, match="admission attempts"):
        _settings(max_admission_attempts=0)


def test_catalog_decodes_sampo_active_sampling_and_curriculum() -> None:
    from posttrain.common import CatalogRef
    from posttrain.train.catalog_schema import decode_training_selection
    from posttrain.train.profiles import ActiveGroupSampling

    data = {
        "selection_type": "sampo-settings",
        "id": "sampo-active",
        "loop": {"max_steps": 1, "max_length": 8, "per_device_batch_size": 2},
        "max_prompt_length": 2,
        "max_completion_length": 6,
        "active_sampling": {"max_candidate_batches": 5},
        "adaptive_curriculum": {"class_field": "domain", "policy": "yield_first", "seed": 7},
        "max_admission_attempts": 2,
    }
    settings = decode_training_selection(CatalogRef("training", "sampo-active"), data, {})
    assert isinstance(settings, SAMPOSettings)
    assert settings.active_sampling == ActiveGroupSampling(max_candidate_batches=5)
    assert settings.adaptive_curriculum is not None and settings.adaptive_curriculum.policy == "yield_first"
    assert settings.max_admission_attempts == 2

    from pydantic import ValidationError

    legacy = {**data, "dynamic_sampling": {"max_candidate_batches": 3}}
    with pytest.raises(ValidationError, match="dynamic_sampling"):
        decode_training_selection(CatalogRef("training", "sampo-legacy"), legacy, {})


def test_trl_sampo_active_sampling_refills_only_missing_groups(tmp_path) -> None:
    from posttrain.train.profiles import ActiveGroupSampling

    request = cast(
        Any,
        SimpleNamespace(
            settings=_settings(active_sampling=ActiveGroupSampling(max_candidate_batches=4)),
            policy=SimpleNamespace(provenance={}),
            training=SimpleNamespace(backend_options={}),
            inference=SimpleNamespace(backend="transformers@1", sampling={}, engine={}),
        ),
    )

    arguments = _online_rl_arguments(request, tmp_path, {})

    assert arguments["dynamic_sampling"] is False
    assert arguments["active_sampling"] is True
    assert arguments["active_sampling_max_batches"] == 4
    assert arguments["active_sampling_reward_std_epsilon"] == 0.0
    assert arguments["use_precomputed_advantages"] is True
    assert arguments["importance_sampling_level"] == "sequence"


def _admission_rows(selected) -> list[EnvironmentRollout]:
    return [
        EnvironmentRollout(task, (1,), (2,), (-1.0,), (True,), 0.0, False, TraceObservation("fixture", identity, {}))
        for task, identity in zip(selected.example_ids, selected.rollout_ids, strict=True)
    ]


def test_sampo_admission_drops_a_failed_group_instead_of_the_update() -> None:
    from posttrain.train.online_rl import PartialRolloutBatchError, RolloutBatch
    from posttrain.train.reward_admission import admit_rollout_groups

    batch = RolloutBatch(("a", "a", "b", "b"), 1, "policy", ("g1", "g1", "g2", "g2"), ("r1", "r2", "r3", "r4"))

    def collect(selected):
        rows = _admission_rows(selected)
        raise PartialRolloutBatchError(
            "one rollout timed out", completed={0: rows[0], 2: rows[2], 3: rows[3]}, failures={1: "rollout timeout"}
        )

    result = admit_rollout_groups(
        batch, _settings(), collect, lambda failures: failures, retain_complete_on_exhaustion=True
    )
    assert result.retained_positions == (2, 3)
    assert [row.example_id for row in result.rollouts] == ["b", "b"]
    assert result.rejected_groups == 1


def test_sampo_admission_returns_no_groups_so_the_refill_draws_again() -> None:
    from posttrain.train.online_rl import RolloutBatch
    from posttrain.train.reward_admission import admit_rollout_groups

    batch = RolloutBatch(("a", "a"), 1, "policy", ("g1", "g1"), ("r1", "r2"))

    def collect(selected):
        raise RuntimeError("engine unavailable")

    result = admit_rollout_groups(
        batch, _settings(), collect, lambda failures: failures, max_attempts=1, retain_complete_on_exhaustion=True
    )
    assert result.rollouts == () and result.retained_positions == ()
    assert result.rejection_reasons == ("RuntimeError",)


def test_sampo_advantages_use_the_admitted_example_ids() -> None:
    from posttrain.train.backends.trl.policy_rollouts import _admitted_example_ids

    example_ids = ("a", "a", "b", "b")
    assert _admitted_example_ids(example_ids, None) == example_ids
    assert _admitted_example_ids(example_ids, (2, 3)) == ("b", "b")
    advantages = compute_sampo_advantages(
        _settings(),
        _admitted_example_ids(("x", "x", "task-1", "task-1"), (2, 3)),
        [_rollout(1.0, "a"), _rollout(0.0, "b")],
    )
    assert advantages.episode_advantages == (0.5, -0.5)


def _sampo_vllm_request(settings: SAMPOSettings) -> Any:
    return cast(
        Any,
        SimpleNamespace(
            settings=settings,
            policy=SimpleNamespace(provenance={}),
            training=SimpleNamespace(backend_options={}),
            inference=SimpleNamespace(backend="vllm@0.29.1.dev4", sampling={}, engine={"mode": "colocate"}),
        ),
    )


def test_sampo_vllm_correction_defaults_to_the_vortex_per_token_cap(tmp_path) -> None:
    arguments = _online_rl_arguments(_sampo_vllm_request(_settings()), tmp_path, {})

    assert arguments["vllm_importance_sampling_mode"] == "token_truncate"
    assert arguments["vllm_importance_sampling_clip_min"] is None
    assert arguments["vllm_importance_sampling_clip_max"] == 2.0
    # SAMPO's own policy ratio stays sequence-level.
    assert arguments["importance_sampling_level"] == "sequence"

    sequence = _settings(importance_sampling_mode="sequence_truncate", importance_sampling_clip_min=0.1)
    arguments = _online_rl_arguments(_sampo_vllm_request(sequence), tmp_path, {})
    assert arguments["vllm_importance_sampling_mode"] == "sequence_truncate"
    assert arguments["vllm_importance_sampling_clip_min"] == 0.1


def test_sampo_rejects_inconsistent_correction_and_truncation_settings() -> None:
    with pytest.raises(ValueError, match="minimum must be smaller"):
        _settings(importance_sampling_clip_min=2.0, importance_sampling_clip_max=2.0)
    with pytest.raises(ValueError, match="finite and positive"):
        _settings(importance_sampling_clip_max=0.0)
    with pytest.raises(ValueError, match="truncation penalty has no effect"):
        _settings(truncation_penalty=0.2, mask_truncated_completions=True)
    with pytest.raises(ValueError, match="finite positive"):
        _settings(truncation_penalty=-0.1)


def test_sampo_truncation_penalty_shapes_the_episode_reward() -> None:
    from posttrain.train.profiles import shape_online_reward

    settings = _settings(truncation_penalty=0.2)
    assert shape_online_reward(settings, 0.5, 100, is_truncated=True) == pytest.approx(0.3)
    assert shape_online_reward(settings, 0.5, 100, is_truncated=False) == 0.5
    assert shape_online_reward(_settings(), 0.5, 100, is_truncated=True) == 0.5


def test_anchor_state_key_ignores_per_attempt_identifiers() -> None:
    from posttrain.train.integrations.verifiers import _anchor_state_key

    first = {
        "role": "tool",
        "tool_call_id": "call_0",
        "content": '{"id": "bc9882d8-7e08-488c-b298-635c5014a043", "ok": true}',
    }
    second = {
        "role": "tool",
        "tool_call_id": "call_3",
        "content": '{"id": "0f1e2d3c-4b5a-6978-8a9b-0c1d2e3f4a5b", "ok": true}',
    }
    different = {
        "role": "tool",
        "tool_call_id": "call_0",
        "content": '{"id": "bc9882d8-7e08-488c-b298-635c5014a043", "ok": false}',
    }

    assert _anchor_state_key(first) == _anchor_state_key(second)
    assert _anchor_state_key(first) != _anchor_state_key(different)
