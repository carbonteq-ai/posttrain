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
from posttrain.train.backends.trl.grpo import _online_rl_arguments


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
            training=SimpleNamespace(backend_options={}),
            inference=SimpleNamespace(backend="transformers@1", sampling={}, engine={}),
        ),
    )

    arguments = _online_rl_arguments(request, tmp_path, {})

    assert arguments["loss_type"] == "grpo"
    assert arguments["importance_sampling_level"] == "sequence"
    assert arguments["use_precomputed_advantages"] is True
    assert arguments["dynamic_sampling"] is True
    assert arguments["epsilon"] == 0.003
    assert arguments["epsilon_high"] == 0.004


def test_verl_sampo_uses_the_hierarchical_backend_adapter() -> None:
    assert _sampo_backend("verl@candidate").__module__ == "posttrain.train.backends.verl.launcher"
