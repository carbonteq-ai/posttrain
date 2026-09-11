"""Tests for provider-free training model resolution."""

from dataclasses import replace

import pytest
from posttrain.common import ExecutionTarget, InferenceBinding
from posttrain.common.variants import QWEN_35_2B
from posttrain.train import QWEN35_RENDERER, LoRAUpdate, TrainingBinding, resolve_training_model_configuration


def _training() -> TrainingBinding:
    return TrainingBinding(
        "training/qwen-test@1",
        "1",
        "trl@1",
        QWEN35_RENDERER,
        LoRAUpdate(),
        ExecutionTarget("target/test", "1", "cuda", 24),
    )


def _inference(*, reasoning_mode: str | None = None) -> InferenceBinding:
    return InferenceBinding(
        "inference/qwen-test@1",
        "1",
        QWEN_35_2B,
        "vllm@1",
        QWEN_35_2B.renderer.id,
        {"max_model_len": 1024, "gpu_memory_utilization": 0.8},
        {"max_tokens": 32},
        ExecutionTarget("target/test", "1", "cuda", 24),
        ("rollout",),
        reasoning_mode=reasoning_mode,
    )


def test_training_configuration_explains_update_and_reasoning() -> None:
    training = _training()
    inference = _inference()

    resolved = resolve_training_model_configuration(QWEN_35_2B, training, inference=inference)

    assert resolved.reasoning_mode == "off"
    assert {origin.path for origin in resolved.origins} >= {
        "policy.model",
        "policy.reasoning_mode",
        "policy.update.kind",
    }


def test_training_configuration_rejects_reasoning_mismatch() -> None:
    training = replace(_training(), renderer=replace(QWEN35_RENDERER, reasoning_mode="thinking"))
    inference = _inference()

    with pytest.raises(ValueError, match="reasoning mode must match"):
        resolve_training_model_configuration(QWEN_35_2B, training, inference=inference)
