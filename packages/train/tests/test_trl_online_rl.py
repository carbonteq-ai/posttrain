"""Tests for the TRL online-RL adapter."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from posttrain.common import ExecutionTarget
from posttrain.common.variants import GEMMA_4_E4B_IT, QWEN_35_2B
from posttrain.train import (
    GEMMA4_RENDERER,
    QWEN35_GRPO_SMOKE,
    QWEN35_RENDERER,
    JsonSchemaResponse,
    LoRAUpdate,
    PolicySampling,
    PolicyTurnRequest,
    QLoRAUpdate,
    TrainingBinding,
)
from posttrain.train.backends.trl.online_rl import TrlPolicyGenerator


class FakeRendered:
    token_ids = [1, 2]
    is_content = [False, True]

    def message_token_spans(self):
        return [(0, 2)]


class FakeRenderer:
    def render(self, messages, *, tools, add_generation_prompt):
        assert messages == [{"role": "user", "content": "hello"}]
        assert tools is None
        assert add_generation_prompt is True
        return FakeRendered()

    def parse_response(self, token_ids, *, tools):
        assert token_ids == [3, 4]
        assert tools is None
        return SimpleNamespace(content="answer", reasoning_content="reason", tool_calls=[])

    def get_stop_token_ids(self):
        return [4]


class FakeTrainer:
    temperature = 0.7
    top_p = 0.9

    def _generate_single_turn(self, prompt_ids, generation_config, extra):
        assert prompt_ids == [[1, 2]]
        assert generation_config is None
        assert extra == {}
        return [[3, 4]], [[-0.1, -0.2]]


def test_trl_policy_generator_reuses_loaded_trainer_and_preserves_exact_tokens(monkeypatch) -> None:
    monkeypatch.setattr("posttrain.train.backends.trl.online_rl.create_renderer", lambda *args: FakeRenderer())
    profile = replace(QWEN35_GRPO_SMOKE, max_completion_length=2)
    generator = TrlPolicyGenerator(FakeTrainer(), object(), QWEN_35_2B, profile, _training())

    result = asyncio.run(
        generator.generate(
            PolicyTurnRequest(
                messages=({"role": "user", "content": "hello"},),
                sampling=PolicySampling(max_tokens=2, temperature=0.7, top_p=0.9),
            )
        )
    )

    assert result.message == {"role": "assistant", "content": "answer", "reasoning_content": "reason"}
    assert result.prompt_ids == (1, 2)
    assert result.completion_ids == (3, 4)
    assert result.completion_logprobs == (-0.1, -0.2)
    assert result.finish_reason == "stop"
    assert result.raw_response == {
        "id": "posttrain-policy-turn",
        "object": "chat.completion",
        "created": 0,
        "choices": [{"index": 0, "message": result.message, "finish_reason": "stop"}],
    }


def test_trl_policy_generator_rejects_environment_sampling_drift(monkeypatch) -> None:
    monkeypatch.setattr("posttrain.train.backends.trl.online_rl.create_renderer", lambda *args: FakeRenderer())
    profile = replace(QWEN35_GRPO_SMOKE, max_completion_length=2)
    generator = TrlPolicyGenerator(FakeTrainer(), object(), QWEN_35_2B, profile, _training())

    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(
            generator.generate(
                PolicyTurnRequest(
                    messages=({"role": "user", "content": "hello"},),
                    sampling=PolicySampling(max_tokens=2, temperature=1.0, top_p=0.9),
                )
            )
        )


class StructuredTrainer(FakeTrainer):
    def __init__(self, *, fail: bool = False) -> None:
        self.generation_config = SimpleNamespace(max_new_tokens=99)
        self.vllm_generation = SimpleNamespace(generation_kwargs={"baseline": "kept"})
        self.fail = fail

    def _generate_single_turn(self, prompt_ids, generation_config, extra):
        assert prompt_ids == [[1, 2]]
        assert generation_config is None
        assert extra == {}
        assert self.generation_config.max_new_tokens == 2
        assert self.vllm_generation.generation_kwargs == {
            "baseline": "kept",
            "max_tokens": 2,
            "structured_outputs": {
                "json": {
                    "type": "object",
                    "properties": {"rules": {"type": "array"}},
                    "required": ["rules"],
                },
                "whitespace_pattern": r" ?",
            },
        }
        if self.fail:
            raise RuntimeError("generation failed")
        return [[3, 4]], [[-0.1, -0.2]]


@pytest.mark.parametrize("fail", [False, True])
def test_trl_policy_generator_applies_and_restores_staged_json_schema(monkeypatch, fail) -> None:
    monkeypatch.setattr("posttrain.train.backends.trl.online_rl.create_renderer", lambda *args: FakeRenderer())
    profile = replace(QWEN35_GRPO_SMOKE, max_completion_length=4)
    trainer = StructuredTrainer(fail=fail)
    generator = TrlPolicyGenerator(trainer, object(), GEMMA_4_E4B_IT, profile, _gemma_training())
    request = PolicyTurnRequest(
        messages=({"role": "user", "content": "hello"},),
        sampling=PolicySampling(max_tokens=2, temperature=0.7, top_p=0.9),
        response_format=JsonSchemaResponse(
            name="rules",
            schema={
                "type": "object",
                "properties": {"rules": {"type": "array", "uniqueItems": True}},
                "required": ["rules"],
            },
            strict=True,
        ),
    )

    if fail:
        with pytest.raises(RuntimeError, match="generation failed"):
            asyncio.run(generator.generate(request))
    else:
        result = asyncio.run(generator.generate(request))
        assert result.completion_ids == (3, 4)
    assert request.response_format.schema == {
        "type": "object",
        "properties": {"rules": {"type": "array", "uniqueItems": True}},
        "required": ["rules"],
    }
    assert trainer.generation_config.max_new_tokens == 99
    assert trainer.vllm_generation.generation_kwargs == {"baseline": "kept"}


def test_trl_policy_generator_rejects_stage_limit_above_global_cap(monkeypatch) -> None:
    monkeypatch.setattr("posttrain.train.backends.trl.online_rl.create_renderer", lambda *args: FakeRenderer())
    profile = replace(QWEN35_GRPO_SMOKE, max_completion_length=2)
    generator = TrlPolicyGenerator(FakeTrainer(), object(), QWEN_35_2B, profile, _training())

    with pytest.raises(ValueError, match="exceeds.*cap"):
        asyncio.run(
            generator.generate(
                PolicyTurnRequest(
                    messages=({"role": "user", "content": "hello"},),
                    sampling=PolicySampling(max_tokens=3, temperature=0.7, top_p=0.9),
                )
            )
        )


def _training() -> TrainingBinding:
    return TrainingBinding(
        "training/qwen3.5-test@1",
        "1",
        "trl@1.8.0",
        QWEN35_RENDERER,
        QLoRAUpdate(),
        ExecutionTarget("targets/test", "1", "nvidia-cuda", 8),
    )


def _gemma_training() -> TrainingBinding:
    return TrainingBinding(
        "training/gemma4-test@1",
        "1",
        "trl@1.8.0",
        GEMMA4_RENDERER,
        LoRAUpdate(),
        ExecutionTarget("targets/test", "1", "nvidia-cuda", 141),
    )
