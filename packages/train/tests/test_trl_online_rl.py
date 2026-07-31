"""Tests for the TRL online-RL adapter."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from posttrain.common import ExecutionTarget
from posttrain.common.variants import QWEN_35_2B
from posttrain.train import (
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
