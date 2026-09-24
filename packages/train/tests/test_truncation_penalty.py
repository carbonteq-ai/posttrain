"""The optional GRPO truncation penalty ranks truncated rollouts below finished ones."""

from dataclasses import replace

import pytest
from posttrain.common import CatalogRef
from posttrain.train.catalog_schema import decode_training_selection
from posttrain.train.profiles import (
    QWEN35_GRPO_SMOKE,
    ActiveGroupSampling,
    GRPOSettings,
    shape_online_reward,
)


def _olmo3(**changes: object) -> GRPOSettings:
    return replace(
        QWEN35_GRPO_SMOKE,
        algorithm="olmo3",
        advantage_scaling="none",
        importance_sampling_mode="token_truncate",
        importance_sampling_clip_min=None,
        importance_sampling_clip_max=2.0,
        active_sampling=ActiveGroupSampling(max_candidate_batches=2),
        **changes,
    )


def test_penalty_is_off_by_default_and_keeps_task_reward() -> None:
    settings = _olmo3()
    assert settings.truncation_penalty is None
    assert shape_online_reward(settings, 0.5, 100, is_truncated=True) == 0.5


def test_penalty_applies_only_to_truncated_rollouts() -> None:
    settings = _olmo3(truncation_penalty=0.2)
    assert shape_online_reward(settings, 0.0, 100, is_truncated=False) == 0.0
    assert shape_online_reward(settings, 0.0, 100, is_truncated=True) == pytest.approx(-0.2)
    assert shape_online_reward(settings, 0.5, 100, is_truncated=True) == pytest.approx(0.3)
    # A zero-score truncated rollout now ranks below a zero-score finished one, so a
    # mixed failure group has reward spread and survives OLMo 3 active sampling.
    assert shape_online_reward(settings, 0.0, 100, is_truncated=True) < shape_online_reward(
        settings, 0.0, 100, is_truncated=False
    )


def test_penalty_stacks_after_dapo_soft_overlong_shaping() -> None:
    settings = replace(
        QWEN35_GRPO_SMOKE,
        algorithm="dapo",
        clip_epsilon_high=0.28,
        overlong_buffer_tokens=84,
        truncation_penalty=0.1,
    )
    # 384-token limit, 84-token buffer: 342 tokens is halfway into the buffer.
    assert shape_online_reward(settings, 1.0, 342, is_truncated=True) == pytest.approx(0.4)


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), float("inf")])
def test_penalty_must_be_finite_and_positive(value: float) -> None:
    with pytest.raises(ValueError, match="truncation penalty must be a finite positive number"):
        _olmo3(truncation_penalty=value)


def test_penalty_is_rejected_when_truncated_completions_are_masked() -> None:
    with pytest.raises(ValueError, match="no effect when truncated completions are masked"):
        _olmo3(truncation_penalty=0.2, mask_truncated_completions=True)


def test_catalog_passes_the_penalty_through() -> None:
    settings = decode_training_selection(
        CatalogRef("training", "test"),
        {
            "selection_type": "grpo-settings",
            "id": "test",
            "revision": "1",
            "loop": {"max_steps": 1, "gradient_accumulation_steps": 2},
            "truncation_penalty": 0.2,
        },
        {},
    )
    assert isinstance(settings, GRPOSettings)
    assert settings.truncation_penalty == 0.2
