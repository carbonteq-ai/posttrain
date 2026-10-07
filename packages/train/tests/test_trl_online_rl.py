"""Tests for the TRL online-RL adapter."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from posttrain.common import ExecutionTarget
from posttrain.common.variants import LFM_25_26B, QWEN_35_2B
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

    def parse_response(self, token_ids, *, tools, prompt_ids=None):
        assert token_ids == [3, 4]
        assert tools is None
        return SimpleNamespace(content="answer", reasoning_content="reason", tool_calls=[])

    def get_stop_token_ids(self):
        return [4]

    def bridge_to_next_turn(self, previous_prompt_ids, previous_completion_ids, new_messages, *, tools):
        return None


class FakeTrainer:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    min_p: float | None = None
    repetition_penalty: float = 1.0
    args: SimpleNamespace = SimpleNamespace(generation_kwargs=None)

    def _generate_single_turn(self, prompt_ids, generation_config, extra):
        assert prompt_ids == [[1, 2]]
        assert generation_config is None
        assert extra == {}
        return [[3, 4]], [[-0.1, -0.2]]


class BatchFakeTrainer:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    min_p: float | None = None
    repetition_penalty: float = 1.0
    args: SimpleNamespace = SimpleNamespace(generation_kwargs=None)

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


def test_trl_generated_call_evidence_reaches_native_trace(monkeypatch) -> None:
    vf = pytest.importorskip("verifiers.v1")
    if not hasattr(vf, "GeneratedCallAttempt"):
        pytest.skip("requires native assessment candidate checkout")
    from posttrain.train.integrations.verifiers import _PolicyClient
    from renderers.base import ParsedToolCall
    from verifiers.v1.assessment_source import capture_trace_source
    from verifiers.v1.dialects.chat import ChatDialect
    from verifiers.v1.graph import prepare_turn

    class CallRenderer(FakeRenderer):
        def parse_response(self, token_ids, *, tools, prompt_ids=None):
            return SimpleNamespace(
                content=None,
                reasoning_content=None,
                tool_calls=[ParsedToolCall(raw="send()", name="send", arguments={}, token_span=(0, 1))],
            )

    monkeypatch.setattr("posttrain.train.backends.trl.online_rl.create_renderer", lambda *args: CallRenderer())
    generator = TrlPolicyGenerator(
        FakeTrainer(), object(), QWEN_35_2B, replace(QWEN35_GRPO_SMOKE, max_completion_length=2), _training()
    )
    response = asyncio.run(
        _PolicyClient(generator).get_response(
            ChatDialect(),
            {"messages": [{"role": "user", "content": "hello"}]},
            "model",
            SimpleNamespace(max_tokens=2, temperature=0.7, top_p=0.9),
        )
    )
    assert response.tokens.completion_ids == [3, 4]
    assert response.tokens.generated_calls[0].emitted_call_index == 0
    from posttrain.train.integrations.verifiers_generation import decode_parser_evidence, encode_parser_evidence

    parsed = CallRenderer().parse_response([3, 4], tools=None)
    evidence = encode_parser_evidence(parsed, (3, 4), response.message.model_dump(), CallRenderer(), configuration={})
    assert evidence is not None
    altered = dict(evidence)
    altered["producer"] = {"renderer": "substituted"}
    with pytest.raises(ValueError, match="producer digest"):
        decode_parser_evidence(altered)
    trace = vf.Trace(
        episode_id="episode",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(prompt="hello")),
    )
    index = prepare_turn(trace, [vf.UserMessage(content="hello")]).commit(response)
    source = capture_trace_source(trace)
    subject = vf.SubjectRef(
        kind="call",
        snapshot_id=source.snapshot_id,
        episode_id="episode",
        trace_id=trace.id,
        node_index=index,
        node_content_digest=source.nodes[index].node_content_digest,
        call_index=0,
    )
    restored = vf.WireTrace.model_validate(trace.to_record())
    assert restored.nodes[index].generated_call_producer == response.tokens.generated_call_producer
    projected = vf.project_subject(subject, source, restored)
    assert projected.status == "exact_call"
    assert [(item.start, item.end) for item in projected.intervals] == [(0, 1)]
    assert restored.nodes[index].token_ids == [3, 4]
    assert restored.nodes[index].mask == [True, True]
    from posttrain.environment.verifiers_conditioning import (
        materialize_native_conditioning,
        native_conditioning_records,
    )
    from verifiers.v1.trace import ModelCall

    restored.calls.append(ModelCall(node=index, model="model", usage=response.usage))
    record = native_conditioning_records(restored, sampled_node_indices=(index,), context_contract="causal-text@1")[0]
    conditioning = materialize_native_conditioning(restored, record)
    assert conditioning.token_ids == tuple(response.tokens.prompt_ids + response.tokens.completion_ids)
    local_support = set(range(projected.intervals[0].start, projected.intervals[0].end))
    assert tuple(physical for local, physical in conditioning.action_positions if local in local_support) == (2,)


@pytest.mark.parametrize("version", [True, 2, "1"])
def test_present_parser_evidence_rejects_unknown_versions(version) -> None:
    from posttrain.train.integrations.verifiers_generation import decode_parser_evidence

    with pytest.raises(ValueError, match="version"):
        decode_parser_evidence({"kind": "verifiers.generated-calls", "schema_version": version, "attempts": []})


@pytest.mark.parametrize(
    "emitted_arguments,duplicate,expected",
    [('{"id": 1}', False, (0,)), ('{"id": 2}', False, (None,)), ('{"id": 1}', True, (None, None))],
)
def test_parser_transport_keeps_ambiguous_and_repaired_links_unavailable(emitted_arguments, duplicate, expected):
    vf = pytest.importorskip("verifiers.v1")
    if not hasattr(vf, "GeneratedCallProducer"):
        pytest.skip("requires native assessment candidate checkout")
    from posttrain.train.integrations.verifiers_generation import decode_parser_evidence, encode_parser_evidence
    from renderers.base import ParsedToolCall

    count = 2 if duplicate else 1
    parsed = SimpleNamespace(
        tool_calls=[
            ParsedToolCall(id="same", raw="send(1)", name="send", arguments={"id": 1}, token_span=(i, i + 1))
            for i in range(count)
        ]
    )
    message = {"tool_calls": [{"id": "same", "name": "send", "arguments": emitted_arguments} for _ in range(count)]}
    configuration = {"tools": [{"description": "référence"}]}
    sidecar = encode_parser_evidence(parsed, (3, 4), message, FakeRenderer(), configuration=configuration)
    assert sidecar is not None
    configuration["tools"][0]["description"] = "rewritten"
    decoded = decode_parser_evidence(sidecar)
    assert tuple(item.emitted_call_index for item in decoded.attempts) == expected
    assert tuple(item.token_span for item in decoded.attempts) == tuple((i, i + 1) for i in range(count))
    assert "référence" in decoded.producer.descriptor_json


class BlockingFakeTrainer(BatchFakeTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def _generate_single_turn(self, prompt_ids, generation_config, extra):
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release the blocking trainer")
        return super()._generate_single_turn(prompt_ids, generation_config, extra)


def test_trl_checkpoint_steps_zero_disables_recovery_saves(tmp_path: Path) -> None:
    arguments = trainer_arguments(
        TrainingLoop(max_steps=2, checkpoint_steps=0),
        tmp_path,
    )

    assert arguments["save_strategy"] == "no"
    assert "save_steps" not in arguments


def test_trl_constant_with_warmup_scheduler_is_forwarded(tmp_path: Path) -> None:
    arguments = trainer_arguments(
        TrainingLoop(max_steps=20, warmup_ratio=0.5, lr_scheduler_type="constant_with_warmup"),
        tmp_path,
    )

    assert arguments["warmup_steps"] == 10
    assert arguments["lr_scheduler_type"] == "constant_with_warmup"


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


def test_resolved_generator_requires_native_sampler_scores_and_opts_in(monkeypatch) -> None:
    monkeypatch.setattr("posttrain.train.backends.trl.online_rl.create_renderer", lambda *args: FakeRenderer())
    profile = replace(QWEN35_GRPO_SMOKE, max_completion_length=2)
    with pytest.raises(ValueError, match="cannot retain native sampled"):
        TrlPolicyGenerator(FakeTrainer(), object(), QWEN_35_2B, profile, _training(), retain_generation_logprobs=True)

    class ReceiptTrainer(FakeTrainer):
        def _generate_single_turn(self, prompt_ids, generation_config, extra, *, return_generation_logprobs=False):
            assert return_generation_logprobs is True
            return super()._generate_single_turn(prompt_ids, generation_config, extra)

    generator = TrlPolicyGenerator(
        ReceiptTrainer(), object(), QWEN_35_2B, profile, _training(), retain_generation_logprobs=True
    )
    result = asyncio.run(
        generator.generate(
            PolicyTurnRequest(
                messages=({"role": "user", "content": "hello"},),
                sampling=PolicySampling(max_tokens=2, temperature=0.7, top_p=0.9),
            )
        )
    )
    assert result.completion_logprobs == (-0.1, -0.2)


def test_trl_lfm_tool_cycle_keeps_sampled_prefix_and_appends_only_new_tool_messages(monkeypatch) -> None:
    from renderers import RenderedTokens
    from renderers.catalog_models import bridge_lfm25_tool_cycle

    tokenizer = SimpleNamespace(bos_token_id=99, eos_token_id=4, encode=lambda text, **kwargs: [10])

    class LfmRenderer:
        # The fork's LFM25Renderer bridges tool results itself.
        def bridge_to_next_turn(self, previous_prompt_ids, previous_completion_ids, new_messages, *, tools):
            return bridge_lfm25_tool_cycle(self, tokenizer, previous_prompt_ids, previous_completion_ids, new_messages)

        def render(self, messages, *, tools, add_generation_prompt):
            assert messages == [{"role": "tool", "content": "created", "tool_call_id": "call_0"}]
            assert tools is None
            assert add_generation_prompt is True
            return RenderedTokens(
                token_ids=[99, 20, 21],
                message_indices=[-1, 0, -1],
                message_roles=["tool"],
                message_tool_names=[None],
            )

        def parse_response(self, token_ids, *, tools, prompt_ids):
            # The rendered prompt reaches the parser so a prefilled thought is attributed.
            assert prompt_ids == [1, 2, 3, 4, 10, 20, 21]
            return SimpleNamespace(content="answer", reasoning_content="reason", tool_calls=[])

        def get_stop_token_ids(self):
            return [4]

    class LfmTrainer(FakeTrainer):
        def _generate_single_turn(self, prompt_ids, generation_config, extra):
            assert prompt_ids == [[1, 2, 3, 4, 10, 20, 21]]
            assert generation_config is None
            assert extra == {}
            return [[3, 4]], [[-0.1, -0.2]]

    renderer = LfmRenderer()
    monkeypatch.setattr("posttrain.train.backends.trl.online_rl.create_renderer", lambda *args: renderer)
    generator = TrlPolicyGenerator(
        LfmTrainer(),
        tokenizer,
        LFM_25_26B,
        replace(QWEN35_GRPO_SMOKE, max_completion_length=2),
        replace(_training(), renderer=replace(_training().renderer, model_family="lfm2.5", implementation="default")),
    )

    result = asyncio.run(
        generator.generate(
            PolicyTurnRequest(
                messages=(
                    {"role": "user", "content": "create"},
                    {"role": "assistant", "content": None, "tool_calls": []},
                    {"role": "tool", "content": "created", "tool_call_id": "call_0"},
                ),
                sampling=PolicySampling(max_tokens=2, temperature=0.7, top_p=0.9),
                previous_prompt_ids=(1, 2),
                previous_completion_ids=(3,),
                tail_start=2,
            )
        )
    )

    assert result.prompt_ids == (1, 2, 3, 4, 10, 20, 21)
    assert result.prompt_message_spans == (None, None, (5, 6))


def test_trl_policy_generator_preserves_rejected_call_in_native_message(monkeypatch) -> None:
    renderer = FakeRenderer()
    renderer.parse_response = lambda *args, **kwargs: SimpleNamespace(
        content="",
        reasoning_content="Check the task.",
        tool_calls=[
            SimpleNamespace(name="asana_get_task", status=SimpleNamespace(value="invalid_json"), token_span=(0, 1)),
        ],
    )
    monkeypatch.setattr("posttrain.train.backends.trl.online_rl.create_renderer", lambda *args: renderer)
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "<tool_call>invalid attempt</tool_call>")
    generator = TrlPolicyGenerator(
        FakeTrainer(),
        tokenizer,
        QWEN_35_2B,
        replace(QWEN35_GRPO_SMOKE, max_completion_length=2),
        _training(),
    )
    result = asyncio.run(
        generator.generate(
            PolicyTurnRequest(
                messages=({"role": "user", "content": "hello"},),
                sampling=PolicySampling(max_tokens=2, temperature=0.7, top_p=0.9),
            )
        )
    )
    assert result.message["content"] == "<tool_call>invalid attempt</tool_call>"
    assert result.message["reasoning_content"] == "Check the task."
    assert "tool_calls" not in result.message
    provider_state = cast(list[dict[str, Any]], result.message["provider_state"])
    assert provider_state[0]["status"] == "invalid_json"
    raw_response = cast(dict[str, Any], result.raw_response)
    choices = cast(list[dict[str, Any]], raw_response["choices"])
    assert choices[0]["message"] == result.message
    assert result.completion_ids == (3, 4)
    assert result.completion_logprobs == (-0.1, -0.2)


def test_resolved_trl_preserves_native_train_client_admission(monkeypatch) -> None:
    from posttrain.train.update_plan import PolicyExecutionBudget, PolicyUpdateSchedule, PolicyUpdateSettings

    renderer = FakeRenderer()
    renderer.parse_response = lambda *args, **kwargs: SimpleNamespace(
        content="",
        reasoning_content="Check the task.",
        tool_calls=[
            SimpleNamespace(
                name="asana_get_task",
                arguments={"task_id": "bad"},
                id="call_0",
                status=SimpleNamespace(value="invalid_json"),
                token_span=(0, 1),
            )
        ],
    )
    monkeypatch.setattr("posttrain.train.backends.trl.online_rl.create_renderer", lambda *args: renderer)
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "<tool_call>invalid attempt</tool_call>")
    settings = replace(
        QWEN35_GRPO_SMOKE,
        max_completion_length=2,
        loop=replace(QWEN35_GRPO_SMOKE.loop, per_device_batch_size=1, gradient_accumulation_steps=1),
        policy_updates=PolicyUpdateSettings(PolicyUpdateSchedule("episode", 2), PolicyExecutionBudget(2, 8192, 100000)),
    )
    generator = TrlPolicyGenerator(FakeTrainer(), tokenizer, QWEN_35_2B, settings, _training())
    result = asyncio.run(
        generator.generate(
            PolicyTurnRequest(
                messages=({"role": "user", "content": "hello"},),
                sampling=PolicySampling(max_tokens=2, temperature=0.7, top_p=0.9),
            )
        )
    )

    assert result.message["tool_calls"] == [
        {"id": "call_0", "name": "asana_get_task", "arguments": '{"task_id": "bad"}'}
    ]
    assert "provider_state" not in result.message
    assert result.raw_response is not None
    choices = cast(list[dict[str, Any]], result.raw_response["choices"])
    assert choices[0]["finish_reason"] == "stop"
    assert result.completion_ids == (3, 4)
    assert result.completion_logprobs == (-0.1, -0.2)


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


def test_trl_policy_generator_checks_complete_sampling_policy(monkeypatch) -> None:
    monkeypatch.setattr("posttrain.train.backends.trl.online_rl.create_renderer", lambda *args: FakeRenderer())
    profile = replace(QWEN35_GRPO_SMOKE, max_completion_length=2)
    trainer = FakeTrainer()
    trainer.top_k = 20
    trainer.min_p = 0.0
    trainer.repetition_penalty = 1.1
    trainer.args = SimpleNamespace(generation_kwargs={"presence_penalty": 1.5})
    generator = TrlPolicyGenerator(trainer, object(), QWEN_35_2B, profile, _training())

    result = asyncio.run(
        generator.generate(
            PolicyTurnRequest(
                messages=({"role": "user", "content": "hello"},),
                sampling=PolicySampling(
                    max_tokens=2,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=20,
                    min_p=0.0,
                    repetition_penalty=1.1,
                    presence_penalty=1.5,
                ),
            )
        )
    )

    assert result.finish_reason == "stop"

    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(
            generator.generate(
                PolicyTurnRequest(
                    messages=({"role": "user", "content": "hello"},),
                    sampling=PolicySampling(
                        max_tokens=2,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=20,
                        min_p=0.0,
                        repetition_penalty=1.1,
                        presence_penalty=0.0,
                    ),
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


def test_trl_policy_generation_does_not_block_environment_event_loop(monkeypatch) -> None:
    monkeypatch.setattr("posttrain.train.backends.trl.online_rl.create_renderer", lambda *args: FakeRenderer())
    profile = replace(QWEN35_GRPO_SMOKE, max_completion_length=2)
    trainer = BlockingFakeTrainer()
    generator = TrlPolicyGenerator(trainer, object(), QWEN_35_2B, profile, _training())
    request = PolicyTurnRequest(
        messages=({"role": "user", "content": "hello"},),
        sampling=PolicySampling(max_tokens=2, temperature=0.7, top_p=0.9),
    )

    async def generate_while_environment_progresses():
        generation = asyncio.create_task(generator.generate(request))
        while not trainer.entered.is_set():
            await asyncio.sleep(0)
        # MCP and environment coroutines share this loop in the direct bridge.
        # They must remain schedulable while colocated vLLM is generating.
        environment_progressed = asyncio.Event()
        asyncio.get_running_loop().call_soon(environment_progressed.set)
        await asyncio.wait_for(environment_progressed.wait(), timeout=0.1)
        assert not generation.done()
        trainer.release.set()
        return await asyncio.wait_for(generation, timeout=1)

    result = asyncio.run(generate_while_environment_progresses())

    assert result.completion_ids == (3, 4)
    assert trainer.prompt_batches == [[[1, 2]]]


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


def test_trl_generator_attributes_prefilled_lfm_thought_with_real_renderer(monkeypatch) -> None:
    from pathlib import Path

    from posttrain.train import LFM25_RENDERER

    transformers = pytest.importorskip("transformers")
    pytest.importorskip("renderers")
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--LiquidAI--LFM2.5-2.6B/snapshots"
        / "654f9463ce32b05d0429d76fe1f580b27d4c1ac0"
    )
    if not snapshot.exists():
        pytest.skip("requires the cached immutable LFM2.5-2.6B tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    sampled = tokenizer.encode("I should answer briefly.</think>Four.<|im_end|>", add_special_tokens=False)

    class ThinkingTrainer(FakeTrainer):
        def _generate_single_turn(self, prompt_ids, generation_config, extra):
            # LFM2.5-2.6B prefills <think> at the end of its generation prompt.
            assert tokenizer.decode(prompt_ids[0][-1:]) == "<think>"
            return [sampled], [[-0.1] * len(sampled)]

    generator = TrlPolicyGenerator(
        ThinkingTrainer(),
        tokenizer,
        LFM_25_26B,
        replace(QWEN35_GRPO_SMOKE, max_completion_length=len(sampled)),
        replace(_training(), renderer=LFM25_RENDERER),
    )
    result = asyncio.run(
        generator.generate(
            PolicyTurnRequest(
                messages=({"role": "user", "content": "Two plus two?"},),
                sampling=PolicySampling(max_tokens=len(sampled), temperature=0.7, top_p=0.9),
            )
        )
    )
    assert result.reasoning_tokens == len(
        tokenizer.encode("I should answer briefly.</think>", add_special_tokens=False)
    )
    assert "</think>" not in str(result.message.get("content"))
