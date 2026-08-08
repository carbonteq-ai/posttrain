"""Tests for the Verifiers-to-GRPO integration bridge."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from itertools import count, islice
from types import SimpleNamespace
from typing import Any, TypeVar, cast

import pytest
from posttrain.catalog import open_catalog
from posttrain.common import CatalogRef, InferenceBinding, ModelVariant
from posttrain.eval import EnvironmentBinding
from posttrain.train import (
    GRPORequest,
    GRPOSettings,
    OnPolicyDistillationRequest,
    OnPolicyDistillationSettings,
    PolicySampling,
    PolicyTurnResult,
    RolloutBatch,
    TrainingBinding,
    build_verifiers_distillation_request,
    build_verifiers_grpo_request,
)
from posttrain.train.integrations import (
    VerifiersEnvironmentRolloutBridge,
    preflight_verifiers_environment,
)
from posttrain.train.integrations.verifiers import (
    _apply_verifiers_runtime_compatibility,
    _load_selected_tasks,
    _PolicyClient,
    load_verifiers_bridge_snapshot,
)

pytest.importorskip("verifiers")

from verifiers.v1 import (
    AgentInfo,
    AssistantMessage,
    MessageNode,
    Sampling,
    TaskData,
    ToolMessage,
    Trace,
    TraceTask,
    UserMessage,
)
from verifiers.v1.dialects import ChatDialect

T = TypeVar("T")


def test_verifiers_runtime_compatibility_pins_null_harness_mcp_v1(monkeypatch) -> None:
    from verifiers.v1.harnesses.null import harness as null_harness

    monkeypatch.setattr(
        null_harness,
        "PROGRAM_SOURCE",
        '# dependencies = ["openai", "mcp", "httpx", "tenacity"]',
    )

    _apply_verifiers_runtime_compatibility()

    assert '"mcp>=1.24.0,<2"' in null_harness.PROGRAM_SOURCE


class FakeGenerator:
    def __init__(self) -> None:
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return PolicyTurnResult(
            message={"role": "assistant", "content": "done"},
            prompt_ids=(1, 2),
            completion_ids=(3, 4),
            completion_logprobs=(-0.1, -0.2),
            finish_reason="stop",
            prompt_message_spans=((0, 2),),
            prompt_is_content=(False, True),
            raw_response={"id": "response-1"},
        )


def _trace(task: Any, suffix: int) -> Trace:
    trace = Trace(
        id=f"trace-{suffix}",
        task=TraceTask(type=type(task).__name__, data=task.data),
        agent=AgentInfo(model="model-profile-v1", sampling=Sampling(temperature=1.0, max_tokens=32)),
        nodes=[
            MessageNode(
                message=UserMessage(content=str(task.data.prompt)),
                token_ids=[1, 2],
                mask=[False, False],
            ),
            MessageNode(
                parent=0,
                message=AssistantMessage(content="calling"),
                sampled=True,
                token_ids=[3, 4],
                mask=[True, True],
                logprobs=[-0.1, -0.2],
            ),
            MessageNode(
                parent=1,
                message=ToolMessage(tool_call_id="call-1", content="result"),
                token_ids=[5, 6],
                mask=[False, False],
            ),
            MessageNode(
                parent=2,
                message=AssistantMessage(content="done"),
                sampled=True,
                token_ids=[7, 8],
                mask=[True, True],
                logprobs=[-0.3, -0.4],
            ),
        ],
        is_completed=True,
        stop_condition="agent_completed",
    )
    trace.record_reward("native", 1.0)
    return trace


class FakeEpisode:
    def __init__(self, task: object, count: int) -> None:
        self.task = task
        self.count = count

    async def run(self):
        return [_trace(self.task, index) for index in range(self.count)]


class FakeEnvironment:
    @asynccontextmanager
    async def serving(self):
        yield

    def episode(self, task, context, n=1):
        assert context.model == "model-profile-v1"
        return FakeEpisode(task, n)


class ConcurrentFakeEpisode(FakeEpisode):
    def __init__(self, environment: ConcurrentFakeEnvironment, task: object, count: int) -> None:
        super().__init__(task, count)
        self.environment = environment

    async def run(self):
        self.environment.active += 1
        self.environment.max_active = max(self.environment.max_active, self.environment.active)
        if self.environment.active == 2:
            self.environment.started.set()
        await asyncio.wait_for(self.environment.started.wait(), timeout=1)
        try:
            return await super().run()
        finally:
            self.environment.active -= 1


class ConcurrentFakeEnvironment(FakeEnvironment):
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.started = asyncio.Event()

    def episode(self, task, context, n=1):
        assert context.model == "model-profile-v1"
        return ConcurrentFakeEpisode(self, task, n)


class InfiniteAlphabetStyleTaskset:
    INFINITE = True

    def __init__(self) -> None:
        self.select_calls: list[tuple[int | None, bool]] = []

    def load(self):
        for index in count(41, 2):
            yield SimpleNamespace(
                data=SimpleNamespace(
                    idx=index,
                    prompt=None,
                    info={"num_turns": 3},
                )
            )

    def select(self, num_tasks: int | None = None, shuffle: bool = False):
        self.select_calls.append((num_tasks, shuffle))
        assert num_tasks is not None
        return list(islice(self.load(), num_tasks))


class FiniteTaskset:
    INFINITE = False

    def __init__(self) -> None:
        self.tasks = [SimpleNamespace(data=SimpleNamespace(idx=index, prompt=f"task-{index}")) for index in range(6)]

    def load(self):
        return self.tasks

    def select(self, *_args, **_kwargs):
        raise AssertionError("finite tasksets must retain seeded index sampling")


def add_shaping(trace: Trace) -> None:
    trace.record_reward("shape", 0.5, 0.1)


def test_infinite_taskset_uses_bounded_selection_and_native_task_identities() -> None:
    taskset = InfiniteAlphabetStyleTaskset()
    environment = SimpleNamespace(
        id="multi-turn-alphabet-sort",
        num_tasks=3,
        parameters={"sampling_seed": 17},
    )

    selected = _load_selected_tasks(
        cast(Any, environment),
        cast(Any, lambda: SimpleNamespace(taskset=taskset)),
    )

    assert taskset.select_calls == [(3, False)]
    assert tuple(selected) == (41, 43, 45)
    assert [task.data.info["num_turns"] for task in selected.values()] == [3, 3, 3]


def test_finite_taskset_preserves_seeded_position_sampling() -> None:
    taskset = FiniteTaskset()
    environment = SimpleNamespace(
        id="finite",
        num_tasks=3,
        parameters={"sampling_seed": 17},
    )

    selected = _load_selected_tasks(
        cast(Any, environment),
        cast(Any, lambda: SimpleNamespace(taskset=taskset)),
    )

    assert tuple(selected) == (2, 3, 4)
    assert [task.data.prompt for task in selected.values()] == [
        "task-2",
        "task-3",
        "task-4",
    ]


def test_policy_client_preserves_exact_turn_tokens_and_response() -> None:
    generator = FakeGenerator()
    client = _PolicyClient(generator)

    response = asyncio.run(
        client.get_response(
            ChatDialect(),
            {"messages": [{"role": "user", "content": "hello"}]},
            "model-profile-v1",
            Sampling(temperature=1.0, max_tokens=32),
            session_id="session-1",
        )
    )

    assert generator.requests[0].messages[0]["content"] == "hello"
    assert generator.requests[0].session_id == "session-1"
    assert response.tokens.prompt_ids == [1, 2]
    assert response.tokens.completion_ids == [3, 4]
    assert response.tokens.completion_logprobs == [-0.1, -0.2]
    assert response.model == "model-profile-v1"
    assert response.raw == {"id": "response-1"}


@pytest.mark.parametrize("technique", ["grpo", "sampo", "distill"])
def test_native_bridge_projects_multiturn_masks_rewards_and_trace_artifact(tmp_path, technique) -> None:
    task = SimpleNamespace(data=TaskData(idx=7, prompt="Arbitrary environment prompt"))
    bridge = VerifiersEnvironmentRolloutBridge(
        dataset_id="custom/train-v1",
        revision="revision",
        tasks={7: task},
        environment_factory=FakeEnvironment,
        trace_path=tmp_path / "traces.jsonl",
        environment_id="custom-v1",
        run_id="run-1",
        sampling=PolicySampling(max_tokens=32),
        technique=technique,
        enrichers=(add_shaping,),
    )

    rollouts = asyncio.run(
        bridge.run(
            RolloutBatch(
                example_ids=("train/000007", "train/000007"),
                step=3,
                model_id="model-profile-v1",
            ),
            FakeGenerator(),
        )
    )
    artifacts = bridge.finalize()
    evidence = bridge.evidence()

    assert bridge.dataset.examples[0].prompt == "Arbitrary environment prompt"
    assert len(rollouts) == 2
    assert rollouts[0].prompt_ids == (1, 2)
    assert rollouts[0].completion_ids == (3, 4, 5, 6, 7, 8)
    assert rollouts[0].env_mask == (True, True, False, False, True, True)
    assert rollouts[0].sampling_logprobs == (-0.1, -0.2, 0.0, 0.0, -0.3, -0.4)
    assert rollouts[0].reward == 1.05
    assert rollouts[0].is_truncated is False
    if technique == "sampo":
        assert tuple((turn.completion_start, turn.completion_end) for turn in rollouts[0].turns) == ((0, 2), (4, 6))
        assert rollouts[0].turns[0].anchor_state_key != rollouts[0].turns[1].anchor_state_key
    else:
        assert rollouts[0].turns == ()
    assert rollouts[0].trace.payload["run"] == {"type": "train", "id": "run-1", "step": 3}
    info = cast(dict[str, object], rollouts[0].trace.payload["info"])
    assert info["example_id"] == "train/000007"
    assert len(artifacts) == 1
    assert artifacts[0].metadata["trace_count"] == 2
    assert artifacts[0].metadata["technique"] == technique
    assert len(evidence.traces) == 2
    assert evidence.traces[0].external_id == rollouts[0].trace.external_id
    assert evidence.traces[0].attributes["environment_id"] == "custom-v1"
    assert len(evidence.metrics) == 1
    assert evidence.metrics[0].step == 3
    assert evidence.metrics[0].values["train/rl/rollouts_attempted"] == 2
    assert evidence.metrics[0].values["train/rl/rollouts_completed"] == 2
    assert evidence.metrics[0].values["train/rl/reward_std"] == pytest.approx(0.0)


def test_native_bridge_runs_distinct_prompt_groups_concurrently(tmp_path) -> None:
    environment = ConcurrentFakeEnvironment()
    tasks = {
        7: SimpleNamespace(data=TaskData(idx=7, prompt="Prompt seven")),
        8: SimpleNamespace(data=TaskData(idx=8, prompt="Prompt eight")),
    }
    bridge = VerifiersEnvironmentRolloutBridge(
        dataset_id="custom/train-v1",
        revision="revision",
        tasks=tasks,
        environment_factory=lambda: environment,
        trace_path=tmp_path / "traces.jsonl",
        environment_id="custom-v1",
        run_id="run-1",
        sampling=PolicySampling(max_tokens=32),
        max_concurrent=2,
    )

    rollouts = asyncio.run(
        bridge.run(
            RolloutBatch(
                example_ids=("train/000007", "train/000008", "train/000007", "train/000008"),
                step=3,
                model_id="model-profile-v1",
            ),
            FakeGenerator(),
        )
    )

    assert environment.max_active == 2
    assert [rollout.example_id for rollout in rollouts] == [
        "train/000007",
        "train/000008",
        "train/000007",
        "train/000008",
    ]


def test_native_bridge_portable_snapshot_reconstructs_without_live_environment_state(tmp_path) -> None:
    task = SimpleNamespace(data=TaskData(idx=7, prompt="Portable environment prompt"))
    bridge = VerifiersEnvironmentRolloutBridge(
        dataset_id="custom/train-v1",
        revision="revision",
        tasks={7: task},
        environment_factory=FakeEnvironment,
        trace_path=tmp_path / "traces.jsonl",
        environment_id="custom-v1",
        run_id="run-1",
        sampling=PolicySampling(max_tokens=32),
        technique="distill",
        enrichers=(add_shaping,),
    )
    snapshot = tmp_path / "bridge.pkl"

    bridge.write_portable_snapshot(snapshot)
    restored = load_verifiers_bridge_snapshot(snapshot)

    assert restored.dataset == bridge.dataset
    assert restored.environment_id == bridge.environment_id
    assert restored.trace_path == bridge.trace_path
    assert restored.technique == "distill"


def test_catalog_environment_builds_public_grpo_and_distillation_requests(tmp_path) -> None:
    catalog = open_catalog(scope="bridge-test")

    def selection(family, selection_id, expected: type[T]) -> T:
        value = catalog.resolve(CatalogRef(family, selection_id)).value
        assert isinstance(value, expected)
        return value

    environment = selection("environment", "gsm8k-distill-train", EnvironmentBinding)
    config = preflight_verifiers_environment(environment)
    tasks = {0: SimpleNamespace(data=SimpleNamespace(prompt="What is 2 + 2?"))}

    grpo_request = build_verifiers_grpo_request(
        policy=selection("model", "models/qwen3.5-2b@bf16", ModelVariant),
        environment=environment,
        settings=selection("training", "qwen3.5-2b/grpo-smoke-v3", GRPOSettings),
        training=selection("training", "training/qwen3.5-trl-lora@1", TrainingBinding),
        inference=selection(
            "inference",
            "inference/qwen3.5-2b-vllm-rollout@1",
            InferenceBinding,
        ),
        trace_path=tmp_path / "grpo-traces.jsonl",
        run_id="grpo-run",
        tasks=tasks,
    )
    distillation_request = build_verifiers_distillation_request(
        student=selection("model", "models/qwen3.5-0.8b@bf16", ModelVariant),
        teacher=selection("model", "models/qwen3.5-2b@bf16", ModelVariant),
        environment=environment,
        settings=selection(
            "training",
            "qwen3.5-0.8b/on-policy-distill-smoke-v1",
            OnPolicyDistillationSettings,
        ),
        training=selection(
            "training",
            "training/qwen3.5-0.8b-trl-distill-lora@1",
            TrainingBinding,
        ),
        rollout_inference=selection(
            "inference",
            "inference/qwen3.5-0.8b-vllm-distill-rollout@1",
            InferenceBinding,
        ),
        teacher_inference=selection(
            "inference",
            "inference/qwen3.5-2b-vllm-teacher-score@1",
            InferenceBinding,
        ),
        trace_path=tmp_path / "distill-traces.jsonl",
        run_id="distill-run",
        tasks=tasks,
    )

    assert config["taskset"]["split"] == "train"
    assert isinstance(grpo_request, GRPORequest)
    assert isinstance(distillation_request, OnPolicyDistillationRequest)
    assert grpo_request.bridge.dataset.examples[0].prompt == "What is 2 + 2?"
    assert distillation_request.bridge.dataset.examples[0].prompt == "What is 2 + 2?"
