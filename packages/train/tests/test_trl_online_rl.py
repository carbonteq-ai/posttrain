"""Tests for the TRL online-RL adapter."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from posttrain.common import ExecutionTarget
from posttrain.common.variants import GEMMA_4_E2B_IT, QWEN_35_2B
from posttrain.train import (
    GEMMA4_RENDERER,
    QWEN35_GRPO_SMOKE,
    QWEN35_RENDERER,
    PolicySampling,
    PolicyTurnRequest,
    QLoRAUpdate,
    TrainingBinding,
    TrainingLoop,
)
from posttrain.train.backends.trl.common import trainer_arguments
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
        assert token_ids in ([3, 4], [3])
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


class BatchFakeTrainer:
    temperature = 0.7
    top_p = 0.9

    def __init__(self) -> None:
        self.prompt_batches: list[list[list[int]]] = []

    def _generate_single_turn(self, prompt_ids, generation_config, extra):
        self.prompt_batches.append(prompt_ids)
        assert generation_config is None
        assert extra == {}
        return (
            [[3, 4] for _prompt_ids in prompt_ids],
            [[-0.1, -0.2] for _prompt_ids in prompt_ids],
        )


class DynamicFakeTrainer(BatchFakeTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.vllm_generation = SimpleNamespace(
            max_completion_length=99,
            generation_kwargs={"seed": 7},
        )
        self.observed: list[tuple[int, dict[str, object]]] = []

    def _generate_single_turn(self, prompt_ids, generation_config, extra):
        self.prompt_batches.append(prompt_ids)
        self.observed.append(
            (
                self.vllm_generation.max_completion_length,
                dict(self.vllm_generation.generation_kwargs),
            )
        )
        length = min(2, self.vllm_generation.max_completion_length)
        return (
            [[3, 4][:length] for _prompt_ids in prompt_ids],
            [[-0.1, -0.2][:length] for _prompt_ids in prompt_ids],
        )


def test_trl_checkpoint_steps_zero_disables_recovery_saves(tmp_path: Path) -> None:
    arguments = trainer_arguments(
        TrainingLoop(max_steps=2, checkpoint_steps=0),
        tmp_path,
    )

    assert arguments["save_strategy"] == "no"
    assert "save_steps" not in arguments


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
        "posttrain_generation": {
            "prompt_tokens": 2,
            "effective_max_tokens": 2,
        },
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


def test_trl_policy_generator_batches_concurrent_environment_turns(monkeypatch) -> None:
    monkeypatch.setattr("posttrain.train.backends.trl.online_rl.create_renderer", lambda *args: FakeRenderer())
    profile = replace(QWEN35_GRPO_SMOKE, max_completion_length=2)
    trainer = BatchFakeTrainer()
    generator = TrlPolicyGenerator(trainer, object(), QWEN_35_2B, profile, _training())
    request = PolicyTurnRequest(
        messages=({"role": "user", "content": "hello"},),
        sampling=PolicySampling(max_tokens=2, temperature=0.7, top_p=0.9),
    )

    async def generate_all():
        return await asyncio.gather(*(generator.generate(request) for _index in range(4)))

    results = asyncio.run(generate_all())

    assert trainer.prompt_batches == [[[1, 2], [1, 2], [1, 2], [1, 2]]]
    assert [result.completion_ids for result in results] == [(3, 4)] * 4
    assert [result.completion_logprobs for result in results] == [(-0.1, -0.2)] * 4


def test_trl_policy_generator_applies_dynamic_limit_and_strict_schema(monkeypatch) -> None:
    monkeypatch.setattr(
        "posttrain.train.backends.trl.online_rl.create_renderer",
        lambda *args: FakeRenderer(),
    )
    profile = replace(QWEN35_GRPO_SMOKE, max_prompt_length=8, max_completion_length=6)
    trainer = DynamicFakeTrainer()
    generator = TrlPolicyGenerator(trainer, object(), QWEN_35_2B, profile, _training())

    result = asyncio.run(
        generator.generate(
            PolicyTurnRequest(
                messages=({"role": "user", "content": "hello"},),
                sampling=PolicySampling(max_tokens=6, temperature=0.7, top_p=0.9),
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                            "additionalProperties": False,
                        },
                    },
                },
                max_prompt_tokens=2,
                max_sequence_tokens=3,
            )
        )
    )

    assert result.completion_ids == (3,)
    assert result.finish_reason == "length"
    assert trainer.observed == [
        (
            1,
            {
                "seed": 7,
                "structured_outputs": {
                    "json": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    }
                },
                "min_tokens": 1,
            },
        )
    ]
    assert trainer.vllm_generation.max_completion_length == 99
    assert trainer.vllm_generation.generation_kwargs == {"seed": 7}


def test_trl_policy_generator_bounds_gemma_json_whitespace(monkeypatch) -> None:
    monkeypatch.setattr(
        "posttrain.train.backends.trl.online_rl.create_renderer",
        lambda *args: FakeRenderer(),
    )
    profile = replace(QWEN35_GRPO_SMOKE, max_prompt_length=8, max_completion_length=6)
    trainer = DynamicFakeTrainer()
    generator = TrlPolicyGenerator(
        trainer,
        object(),
        GEMMA_4_E2B_IT,
        profile,
        replace(_training(), renderer=GEMMA4_RENDERER),
    )

    asyncio.run(
        generator.generate(
            PolicyTurnRequest(
                messages=({"role": "user", "content": "hello"},),
                sampling=PolicySampling(max_tokens=6, temperature=0.7, top_p=0.9),
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                            "additionalProperties": False,
                        },
                    },
                },
                max_prompt_tokens=2,
                max_sequence_tokens=8,
            )
        )
    )

    assert trainer.observed == [
        (
            6,
            {
                "seed": 7,
                "structured_outputs": {
                    "json": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                    "whitespace_pattern": r" ?",
                },
                "min_tokens": 1,
            },
        )
    ]


def test_trl_policy_generator_preserves_larger_structured_output_minimum(monkeypatch) -> None:
    monkeypatch.setattr(
        "posttrain.train.backends.trl.online_rl.create_renderer",
        lambda *args: FakeRenderer(),
    )
    profile = replace(QWEN35_GRPO_SMOKE, max_prompt_length=8, max_completion_length=6)
    trainer = DynamicFakeTrainer()
    trainer.vllm_generation.generation_kwargs["min_tokens"] = 2
    generator = TrlPolicyGenerator(trainer, object(), QWEN_35_2B, profile, _training())

    asyncio.run(
        generator.generate(
            PolicyTurnRequest(
                messages=({"role": "user", "content": "hello"},),
                sampling=PolicySampling(max_tokens=6, temperature=0.7, top_p=0.9),
                response_format={
                    "type": "json_object",
                },
                max_prompt_tokens=2,
                max_sequence_tokens=8,
            )
        )
    )

    assert trainer.observed == [
        (
            6,
            {
                "seed": 7,
                "min_tokens": 2,
                "structured_outputs": {"json_object": True},
            },
        )
    ]
    assert trainer.vllm_generation.generation_kwargs == {"seed": 7, "min_tokens": 2}


def test_trl_policy_generator_drains_turns_queued_while_waiting_for_the_lock(monkeypatch) -> None:
    monkeypatch.setattr("posttrain.train.backends.trl.online_rl.create_renderer", lambda *args: FakeRenderer())
    profile = replace(QWEN35_GRPO_SMOKE, max_completion_length=2)
    trainer = BatchFakeTrainer()
    generator = TrlPolicyGenerator(trainer, object(), QWEN_35_2B, profile, _training())
    request = PolicyTurnRequest(
        messages=({"role": "user", "content": "hello"},),
        sampling=PolicySampling(max_tokens=2, temperature=0.7, top_p=0.9),
    )

    async def generate_with_late_turn():
        await generator._lock.acquire()  # noqa: SLF001 - force the flush to yield at its lock boundary
        first = asyncio.create_task(generator.generate(request))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        second = asyncio.create_task(generator.generate(request))
        await asyncio.sleep(0)
        generator._lock.release()  # noqa: SLF001 - pair with the controlled acquisition above
        return await asyncio.wait_for(asyncio.gather(first, second), timeout=1)

    results = asyncio.run(generate_with_late_turn())

    assert trainer.prompt_batches == [[[1, 2]], [[1, 2]]]
    assert [result.completion_ids for result in results] == [(3, 4), (3, 4)]


def _training() -> TrainingBinding:
    return TrainingBinding(
        "training/qwen3.5-test@1",
        "1",
        "trl@1.8.0",
        QWEN35_RENDERER,
        QLoRAUpdate(),
        ExecutionTarget("targets/test", "1", "nvidia-cuda", 8),
    )
