"""Contract tests for the isolated veRL backend adapter."""

from __future__ import annotations

import asyncio
import importlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from posttrain.common import (
    AppendOnlyJsonlTailer,
    EventObservation,
    ExecutionTarget,
    InferenceBinding,
    LocalArtifactRef,
    MetricBatchObservation,
    MetricObservation,
    ProducedArtifact,
    RunContext,
    TraceFactUpdateObservation,
    TraceObservation,
)
from posttrain.common.variants import LFM_25_12B_THINKING, QWEN_35_2B
from posttrain.data import RolloutDataset, RolloutExample
from posttrain.train import (
    LFM25_RENDERER,
    QWEN35_RENDERER,
    ActiveGroupSampling,
    AdaptiveCurriculum,
    DynamicGroupSampling,
    FullParameterUpdate,
    GRPORequest,
    GRPOSettings,
    LoRAUpdate,
    OnPolicyDistillationRequest,
    OnPolicyDistillationSettings,
    PolicySampling,
    PolicyTurnRequest,
    SAMPORequest,
    SAMPOSettings,
    TrainingBinding,
    TrainingLoop,
    TrainingRuntime,
    verl_grpo_settings_problem,
    verl_training_loop_problem,
)
from posttrain.train.api import _distillation_backend, _grpo_backend, _sampo_backend
from posttrain.train.backends.verl.contracts import VerlLaunchManifest, VerlWorkerResult
from posttrain.train.backends.verl.launcher import (
    _backend_result,
    _isolated_environment,
    _record_failure_artifacts,
    _record_failure_artifacts_best_effort,
    _record_trace_sync_receipt,
    _replay_grpo_metrics,
    _replay_trace_fact_updates,
    _runtime_timeout,
    _start_isolated_worker,
    build_distillation_launch_plan,
    build_grpo_launch_plan,
    build_sampo_launch_plan,
    build_structured_launch_plan,
    grpo_algorithm_payload,
)
from posttrain.train.backends.verl.metrics import (
    VerlRolloutRewardRecord,
    read_verl_metric_records,
    read_verl_rollout_reward_records,
)
from posttrain.train.backends.verl.reward_fields import (
    shaped_rollout_reward,
    streaming_reward_extra_info,
    training_response_mask,
)
from posttrain.train.backends.verl.worker import (
    _last_metrics,
    _uses_turboquant,
    _write_agent_config,
    _write_dataset,
    build_hydra_overrides,
)
from posttrain.train.online_rl import BehaviorPolicySpan, EnvironmentRollout
from posttrain.train.profiles import shape_online_reward
from pydantic import ValidationError


@dataclass(frozen=True)
class FakeEnvironment:
    id: str = "envs/test@1"
    revision: str = "1"


class FakeBridge:
    max_concurrent = 32
    dataset = RolloutDataset(
        "test-rollouts-v1",
        "a" * 40,
        (RolloutExample("train/000000", "Solve 2 + 2.", {}),),
    )

    async def run(self, batch, generator):  # pragma: no cover - not used by translation tests
        raise NotImplementedError

    def finalize(self):
        return ()


@dataclass
class CaptureObserver:
    metrics_seen: list[MetricBatchObservation] = field(default_factory=list)
    artifacts_seen: list[ProducedArtifact] = field(default_factory=list)
    trace_fact_updates: list[TraceFactUpdateObservation] = field(default_factory=list)

    def event(self, observation: EventObservation) -> None:
        del observation

    def metric(self, observation: MetricObservation) -> None:
        self.metrics_seen.append(MetricBatchObservation({observation.name: observation.value}, observation.step))

    def metrics(self, observation: MetricBatchObservation) -> None:
        self.metrics_seen.append(observation)

    def trace(self, observation: TraceObservation) -> None:
        del observation

    def trace_fact_update(self, observation: TraceFactUpdateObservation) -> None:
        self.trace_fact_updates.append(observation)

    def artifact(self, artifact: ProducedArtifact) -> None:
        self.artifacts_seen.append(artifact)


def _context(tmp_path: Path, observer: CaptureObserver) -> RunContext:
    return RunContext(
        project_id="projects/test",
        work_package_id="work-packages/grpo-test",
        run_id="runs/verl-grpo-test",
        job_kind="train.grpo",
        job_definition_version="1",
        workspace=tmp_path.resolve(),
        observer=observer,
    )


def _target(identifier: str) -> ExecutionTarget:
    return ExecutionTarget(identifier, "1", "nvidia-cuda", 80, {"world_size": 2})


def _training(*, family: str = "qwen3.5", update=None) -> TrainingBinding:
    renderer = QWEN35_RENDERER if family == "qwen3.5" else LFM25_RENDERER
    return TrainingBinding(
        "training/verl-test@1",
        "1",
        "verl@a35908c",
        renderer,
        update or FullParameterUpdate(),
        _target("targets/train"),
        runtime=TrainingRuntime(
            global_batch_size=2,
            nodes=1,
            devices_per_node=2,
            parameter_offload=True,
            optimizer_offload=True,
        ),
        backend_options={
            "python_executable": "/opt/posttrain-verl/bin/python",
            "working_directory": "/opt/src/verl",
            "source_revision": "a35908ca3c9632859c58d6a2855d858918ae21dc",
            "attention_implementation": "sdpa",
        },
    )


def _inference(model, *, purpose=("rollout",), identifier="inference/rollout@1") -> InferenceBinding:
    return InferenceBinding(
        identifier,
        "1",
        model,
        "vllm@0.18.0",
        model.renderer_contract,
        {
            "max_model_len": 384,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.4,
            "kv_cache_memory_bytes": 64 * 1024 * 1024,
        },
        {"max_tokens": 128, "temperature": 1.0, "top_p": 1.0},
        _target(f"targets/{purpose[0]}"),
        purpose,
    )


def _grpo_request(*, model=QWEN_35_2B, family="qwen3.5", update=None) -> GRPORequest:
    return GRPORequest(
        policy=model,
        bridge=FakeBridge(),
        settings=GRPOSettings(
            "settings/grpo-test@1",
            TrainingLoop(
                max_steps=1,
                max_length=384,
                per_device_batch_size=1,
                gradient_accumulation_steps=2,
                lr_scheduler_type="constant",
            ),
            max_prompt_length=256,
            max_completion_length=128,
        ),
        environment=FakeEnvironment(),
        training=_training(family=family, update=update),
        inference=_inference(model),
    )


def _sampo_request() -> SAMPORequest:
    return SAMPORequest(
        policy=QWEN_35_2B,
        bridge=FakeBridge(),
        settings=SAMPOSettings(
            "settings/sampo-test@1",
            TrainingLoop(
                max_steps=1,
                max_length=384,
                per_device_batch_size=1,
                gradient_accumulation_steps=2,
                lr_scheduler_type="constant",
            ),
            max_prompt_length=256,
            max_completion_length=128,
        ),
        environment=FakeEnvironment(),
        training=_training(),
        inference=_inference(QWEN_35_2B),
    )


def test_backend_resolver_exposes_general_verl_product_for_both_operations() -> None:
    assert _grpo_backend("verl@a35908c").__module__ == "posttrain.train.backends.verl.launcher"
    assert _sampo_backend("verl@a35908c").__module__ == "posttrain.train.backends.verl.launcher"
    assert _distillation_backend("verl@a35908c").__module__ == "posttrain.train.backends.verl.launcher"
    with pytest.raises(ValueError, match="unsupported GRPO"):
        _grpo_backend("unknown@1")


def test_verl_policy_generator_preserves_complete_sampling_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "posttrain.train.backends.verl.agent_loop"
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    for package in ("verl", "verl.experimental", "verl.experimental.agent_loop"):
        module = ModuleType(package)
        module.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, package, module)
    verl_agent_loop = ModuleType("verl.experimental.agent_loop.agent_loop")
    verl_agent_loop.__dict__["AgentLoopBase"] = object
    verl_agent_loop.__dict__["AgentLoopMetrics"] = object
    verl_agent_loop.__dict__["AgentLoopOutput"] = object
    monkeypatch.setitem(sys.modules, "verl.experimental.agent_loop.agent_loop", verl_agent_loop)

    class FakeRenderer:
        def render(self, messages, *, tools, add_generation_prompt):
            assert messages == [{"role": "user", "content": "hello"}]
            assert tools is None
            assert add_generation_prompt is True
            return SimpleNamespace(
                token_ids=(1, 2),
                message_token_spans=lambda: ((0, 2),),
                is_content=(True, True),
            )

        def parse_response(self, token_ids, *, tools):
            assert token_ids == [3, 4]
            assert tools is None
            return SimpleNamespace(content="done", reasoning_content=None, tool_calls=())

        def get_stop_token_ids(self):
            return ()

    renderers = ModuleType("renderers")
    renderer_configs: list[object] = []
    renderers.__dict__["Qwen35RendererConfig"] = lambda *, enable_thinking: {"enable_thinking": enable_thinking}
    renderers.__dict__["DefaultRendererConfig"] = lambda: {"default": True}
    renderers.__dict__["create_renderer"] = lambda tokenizer, config: (renderer_configs.append(config), FakeRenderer())[
        1
    ]
    monkeypatch.setitem(sys.modules, "renderers", renderers)

    class ServerManager:
        def __init__(self) -> None:
            self.request: dict[str, object] | None = None
            self.spans = iter(((3, 3), (3, 4), (4, 5)))

        async def generate(self, **kwargs):
            self.request = kwargs
            start, end = next(self.spans)
            return SimpleNamespace(
                token_ids=(3, 4),
                log_probs=(-0.1, -0.2),
                extra_fields={"min_global_steps": start, "max_global_steps": end},
            )

    agent_loop = importlib.import_module(module_name)
    server = ServerManager()
    generator = agent_loop.VerlPolicyGenerator(server, object(), enable_thinking=False)
    agent_loop.VerlPolicyGenerator(
        server,
        object(),
        enable_thinking=False,
        renderer_implementation="default",
    )
    assert renderer_configs == [{"enable_thinking": False}, {"default": True}]
    sampling = PolicySampling(
        max_tokens=32,
        temperature=0.7,
        top_p=0.95,
        top_k=20,
        min_p=0.01,
        repetition_penalty=1.1,
        presence_penalty=1.5,
    )

    result = asyncio.run(
        generator.generate(PolicyTurnRequest(messages=({"role": "user", "content": "hello"},), sampling=sampling))
    )

    assert result.completion_ids == (3, 4)
    assert result.behavior_policy == BehaviorPolicySpan(3, 3)
    assert generator.behavior_policy == BehaviorPolicySpan(3, 3)
    assert server.request is not None
    assert server.request["sampling_params"] == {
        "max_tokens": 32,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.01,
        "repetition_penalty": 1.1,
        "presence_penalty": 1.5,
        "logprobs": True,
    }
    asyncio.run(
        generator.generate(
            PolicyTurnRequest(
                messages=({"role": "user", "content": "hello"},),
                sampling=replace(sampling, min_p=None, presence_penalty=0.0),
            )
        )
    )
    parameters = server.request["sampling_params"]
    assert isinstance(parameters, dict)
    assert "min_p" not in parameters
    assert parameters["repetition_penalty"] == 1.1
    assert parameters["presence_penalty"] == 0.0

    generator.set_sampling_overrides(
        {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": 16,
            "logprobs": True,
        }
    )
    asyncio.run(
        generator.generate(PolicyTurnRequest(messages=({"role": "user", "content": "hello"},), sampling=sampling))
    )
    assert server.request is not None
    assert server.request["sampling_params"] == {
        "max_tokens": 16,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.01,
        "repetition_penalty": 1.1,
        "presence_penalty": 1.5,
        "logprobs": True,
    }
    assert generator.behavior_policy == BehaviorPolicySpan(3, 5)
    generator.set_sampling_overrides({"max_tokens": 33})
    with pytest.raises(ValueError, match="exceeds the environment output limit"):
        asyncio.run(
            generator.generate(PolicyTurnRequest(messages=({"role": "user", "content": "hello"},), sampling=sampling))
        )
    sys.modules.pop(module_name, None)


def test_verl_policy_generator_reports_bridged_spans_over_the_full_message_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bridged turn's spans index the full message list, as the TRL generator's do."""

    real_renderers = pytest.importorskip("renderers")
    module_name = "posttrain.train.backends.verl.agent_loop"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    for package in ("verl", "verl.experimental", "verl.experimental.agent_loop"):
        module = ModuleType(package)
        module.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, package, module)
    verl_agent_loop = ModuleType("verl.experimental.agent_loop.agent_loop")
    for name in ("AgentLoopBase", "AgentLoopMetrics", "AgentLoopOutput"):
        verl_agent_loop.__dict__[name] = object
    monkeypatch.setitem(sys.modules, "verl.experimental.agent_loop.agent_loop", verl_agent_loop)

    class FakeRenderer:
        def render(self, messages, *, tools, add_generation_prompt):
            raise AssertionError("a bridged turn must not re-render the conversation")

        def bridge_to_next_turn(self, prompt_ids, completion_ids, new_messages, *, tools):
            # Prefix (1, 2, 3, 4) retained; two tool results and the generation prompt follow.
            assert (prompt_ids, completion_ids) == ([1, 2], [3, 4])
            assert [message["role"] for message in new_messages] == ["tool", "tool"]
            return real_renderers.RenderedTokens(
                token_ids=[1, 2, 3, 4, 7, 8, 9, 10, 11],
                message_indices=[-1, -1, -1, -1, 0, 0, 1, 1, -1],
                message_roles=["tool", "tool"],
            )

        def parse_response(self, token_ids, *, tools):
            return SimpleNamespace(content="done", reasoning_content=None, tool_calls=())

        def get_stop_token_ids(self):
            return (6,)

    renderers = ModuleType("renderers")
    renderers.__dict__["Qwen35RendererConfig"] = lambda **kwargs: kwargs
    renderers.__dict__["DefaultRendererConfig"] = lambda **kwargs: kwargs
    renderers.__dict__["create_renderer"] = lambda tokenizer, config: FakeRenderer()
    renderers.__dict__["RenderedTokens"] = real_renderers.RenderedTokens
    monkeypatch.setitem(sys.modules, "renderers", renderers)

    class ServerManager:
        async def generate(self, **kwargs):
            return SimpleNamespace(token_ids=(5, 6), log_probs=(-0.1, -0.2), extra_fields={})

    agent_loop = importlib.import_module(module_name)
    generator = agent_loop.VerlPolicyGenerator(ServerManager(), object(), enable_thinking=False)
    messages = (
        {"role": "user", "content": "find"},
        {"role": "assistant", "content": "calling"},
        {"role": "tool", "content": "first"},
        {"role": "tool", "content": "second"},
    )
    result = asyncio.run(
        generator.generate(
            PolicyTurnRequest(
                messages=messages,
                sampling=PolicySampling(max_tokens=32, temperature=1.0, top_p=1.0),
                previous_prompt_ids=(1, 2),
                previous_completion_ids=(3, 4),
                tail_start=2,
            )
        )
    )

    # Before the fix veRL reported ((4, 6), (6, 8)): one span per new message,
    # which Verifiers reads as the spans of messages 0 and 1.
    assert result.prompt_message_spans == (None, None, (4, 6), (6, 8))
    sys.modules.pop(module_name, None)


def test_qwen35_grpo_translation_is_deterministic_and_backend_neutral(tmp_path: Path) -> None:
    request = _grpo_request(update=LoRAUpdate(rank=16, alpha=32))
    output_dir = tmp_path / "trainer"
    output_dir.mkdir()

    first = build_grpo_launch_plan(request, output_dir)
    second = build_grpo_launch_plan(request, output_dir)

    assert first == second
    assert first.backend.startswith("verl@")
    assert first.operation == "grpo"
    assert first.payload.policy is not None
    assert first.payload.policy.family == "qwen3.5"
    assert first.payload.training.update.model_dump() == {
        "kind": "lora",
        "rank": 16,
        "alpha": 32,
        "dropout": 0.0,
        "target_modules": "all-linear",
    }
    assert first.payload.training.renderer.model_dump() == {
        "id": "qwen3.5-off-v1",
        "implementation": "qwen3.5",
        "reasoning_mode": "off",
    }
    assert first.payload.environment.examples[0].id == "train/000000"
    assert first.command[0] == "/opt/posttrain-verl/bin/python"


def test_verl_launch_manifest_round_trips_and_rejects_schema_drift(tmp_path: Path) -> None:
    plan = build_grpo_launch_plan(_grpo_request(), tmp_path)
    path = tmp_path / "posttrain-verl-launch.json"

    plan.write(path)

    assert VerlLaunchManifest.read(path) == plan
    drifted = plan.model_dump()
    drifted["payload"]["training"]["runtime"]["unknown_setting"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        VerlLaunchManifest.model_validate(drifted)


def test_verl_launch_manifest_rejects_operation_role_mismatch(tmp_path: Path) -> None:
    payload = build_grpo_launch_plan(_grpo_request(), tmp_path).model_dump()
    payload["operation"] = "distill"

    with pytest.raises(ValidationError, match="distillation manifest requires student and teacher"):
        VerlLaunchManifest.model_validate(payload)


def test_grpo_worker_maps_prompt_groups_generations_and_kl_without_importing_verl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = replace(_grpo_request(), settings=replace(_grpo_request().settings, beta=0.02))
    plan = build_grpo_launch_plan(request, tmp_path)
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")

    overrides = build_hydra_overrides(
        plan,
        tmp_path / "rollouts.parquet",
        tmp_path / "agent-loop.json",
        tmp_path / "checkpoints",
    )
    assert "data.train_batch_size=1" in overrides
    assert "actor_rollout_ref.rollout.n=2" in overrides
    assert "actor_rollout_ref.actor.ppo_mini_batch_size=1" in overrides
    assert "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" in overrides
    assert "actor_rollout_ref.actor.use_kl_loss=true" in overrides
    assert "actor_rollout_ref.actor.kl_loss_coef=0.02" in overrides
    assert "actor_rollout_ref.rollout.load_format=dummy" in overrides
    assert "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=67108864" in overrides
    assert "trainer.max_actor_ckpt_to_keep=1" in overrides
    assert "trainer.max_critic_ckpt_to_keep=1" in overrides
    assert "trainer.resume_mode=disable" in overrides
    assert not any(value.startswith("trainer.resume_from_path=") for value in overrides)
    assert "trainer.logger=['console','file']" in overrides


@pytest.mark.parametrize(
    ("schedule", "warmup_ratio", "warmup_steps"),
    [("constant", 0.0, 0), ("constant", 0.2, 0), ("constant_with_warmup", 0.0, 0), ("constant_with_warmup", 0.2, 2)],
)
def test_verl_worker_maps_the_training_loop_exactly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    schedule: str,
    warmup_ratio: float,
    warmup_steps: int,
) -> None:
    request = _grpo_request()
    loop = replace(
        request.settings.loop,
        max_steps=10,
        lr_scheduler_type=schedule,  # type: ignore[arg-type]
        warmup_ratio=warmup_ratio,
        seed=1729,
    )
    plan = build_grpo_launch_plan(replace(request, settings=replace(request.settings, loop=loop)), tmp_path)
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")

    overrides = build_hydra_overrides(
        plan, tmp_path / "rollouts.parquet", tmp_path / "agent-loop.json", tmp_path / "checkpoints"
    )

    # Transformers "constant" never warms up; "constant_with_warmup" warms up for
    # ceil(max_steps * warmup_ratio) steps, then veRL holds the rate constant.
    assert plan.payload.training.loop.lr_scheduler_type == schedule
    assert "actor_rollout_ref.actor.optim.lr_scheduler_type=constant" in overrides
    assert f"actor_rollout_ref.actor.optim.lr_warmup_steps={warmup_steps}" in overrides
    assert "actor_rollout_ref.actor.optim.weight_decay=0.0" in overrides
    for key in (
        "data.seed",
        "actor_rollout_ref.rollout.seed",
        "actor_rollout_ref.actor.data_loader_seed",
        "actor_rollout_ref.actor.fsdp_config.seed",
        "actor_rollout_ref.ref.fsdp_config.seed",
    ):
        assert f"{key}=1729" in overrides


def test_verl_worker_uses_the_per_device_batch_as_its_micro_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _grpo_request()
    settings = replace(
        request.settings,
        num_prompts_per_step=2,
        num_generations=4,
        loop=replace(request.settings.loop, per_device_batch_size=2, gradient_accumulation_steps=4),
    )
    training = replace(request.training, runtime=replace(request.training.runtime, global_batch_size=8))
    plan = build_grpo_launch_plan(replace(request, settings=settings, training=training), tmp_path)
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")

    overrides = build_hydra_overrides(
        plan, tmp_path / "rollouts.parquet", tmp_path / "agent-loop.json", tmp_path / "checkpoints"
    )

    # One optimizer step per update over all 8 rows, two rows per device per micro-batch.
    assert "actor_rollout_ref.actor.ppo_mini_batch_size=2" in overrides
    assert "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2" in overrides
    assert "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2" in overrides


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"lr_scheduler_type": "linear"}, "lr_scheduler_type 'linear' is not available on the veRL backend"),
        ({"logging_steps": 2}, "logging_steps 2 is not available on the veRL backend"),
        (
            {"per_device_batch_size": 2, "gradient_accumulation_steps": 1},
            "the 2 rows of one update do not split into micro-batches of per_device_batch_size 2 on 2 device(s)",
        ),
    ],
)
def test_verl_rejects_training_loops_it_cannot_run_exactly(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    request = _grpo_request()
    loop = replace(request.settings.loop, **changes)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=re.escape(message)):
        build_grpo_launch_plan(replace(request, settings=replace(request.settings, loop=loop)), tmp_path)


def test_verl_training_loop_problem_checks_the_batch_split() -> None:
    loop = TrainingLoop(
        max_steps=1, per_device_batch_size=2, gradient_accumulation_steps=3, lr_scheduler_type="constant"
    )

    assert verl_training_loop_problem(loop, rows_per_update=6, world_size=1) is None
    assert "(2 x 3) must equal the 8 rows" in str(verl_training_loop_problem(loop, rows_per_update=8, world_size=1))
    assert "on 2 device(s)" in str(verl_training_loop_problem(loop, rows_per_update=6, world_size=2))


def test_grpo_worker_maps_bounded_rollout_execution_to_native_verl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # The native-thread reservation is checked against the host CPUs; fix them so the test runs anywhere.
    monkeypatch.setattr("posttrain.train.rollout_execution.effective_cpu_count", lambda: 64)
    request = _grpo_request()
    training = replace(
        request.training,
        backend_options={
            **request.training.backend_options,
            "source_revision": "5dbf667c99b29db613d1dfcded1ed90440ef6311",
            "rollout_execution": {
                "env_workers": 4,
                "episodes_per_worker": 8,
                "worker_native_threads": 1,
            },
        },
    )
    plan = build_grpo_launch_plan(replace(request, training=training), tmp_path)
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")

    overrides = build_hydra_overrides(
        plan,
        tmp_path / "rollouts.parquet",
        tmp_path / "agent-loop.json",
        tmp_path / "checkpoints",
    )

    assert "actor_rollout_ref.rollout.agent.num_workers=4" in overrides
    assert "actor_rollout_ref.rollout.agent.num_cpus_per_worker=1" in overrides
    assert "actor_rollout_ref.rollout.agent.max_concurrent_episodes=32" in overrides
    assert "actor_rollout_ref.rollout.agent.max_concurrent_episodes_per_worker=8" in overrides
    assert "trainer.v1.sampler.refill_all_failed_groups=True" in overrides


def test_grpo_worker_rejects_rollout_capacity_above_environment_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _grpo_request()
    training = replace(
        request.training,
        backend_options={
            **request.training.backend_options,
            "source_revision": "5dbf667c99b29db613d1dfcded1ed90440ef6311",
            "rollout_execution": {
                "env_workers": 5,
                "episodes_per_worker": 8,
                "worker_native_threads": 1,
            },
        },
    )
    plan = build_grpo_launch_plan(replace(request, training=training), tmp_path)
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")

    with pytest.raises(ValueError, match="environment worker capacity exceeds"):
        build_hydra_overrides(
            plan,
            tmp_path / "rollouts.parquet",
            tmp_path / "agent-loop.json",
            tmp_path / "checkpoints",
        )


def test_grpo_worker_rejects_bounded_execution_on_legacy_verl_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _grpo_request()
    training = replace(
        request.training,
        backend_options={
            **request.training.backend_options,
            "rollout_execution": {
                "env_workers": 4,
                "episodes_per_worker": 8,
                "worker_native_threads": 1,
            },
        },
    )
    plan = build_grpo_launch_plan(replace(request, training=training), tmp_path)
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")

    with pytest.raises(ValueError, match="does not support bounded rollout_execution"):
        build_hydra_overrides(
            plan,
            tmp_path / "rollouts.parquet",
            tmp_path / "agent-loop.json",
            tmp_path / "checkpoints",
        )


def test_grpo_prompt_shuffle_is_explicit_and_backend_neutral(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = replace(_grpo_request(), settings=replace(_grpo_request().settings, shuffle_prompts=True))
    plan = build_grpo_launch_plan(request, tmp_path)
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")

    overrides = build_hydra_overrides(
        plan,
        tmp_path / "rollouts.parquet",
        tmp_path / "agent-loop.json",
        tmp_path / "checkpoints",
    )

    assert plan.payload.algorithm.shuffle_prompts is True
    assert "data.shuffle=true" in overrides


def test_verl_checkpoint_steps_zero_keeps_only_terminal_model_save(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _grpo_request()
    request = replace(
        request,
        settings=replace(
            request.settings,
            loop=replace(request.settings.loop, checkpoint_steps=0),
        ),
    )
    plan = build_grpo_launch_plan(request, tmp_path)
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")

    overrides = build_hydra_overrides(
        plan,
        tmp_path / "rollouts.parquet",
        tmp_path / "agent-loop.json",
        tmp_path / "checkpoints",
    )

    assert f"trainer.save_freq={request.settings.loop.max_steps + 1}" in overrides


POST5_REVISION = "9fd6e7a31396ba33a29233cc869ab05b0a9e5a80"
POST4_REVISION = "54124edfb8d0b73694696400cf07a76a14d9be65"
# codex/vortex-active-sampling: post5 plus round-based active sampling (unreleased).
ACTIVE_SAMPLING_REVISION = "6c7295cd411c4d3973ddc206e43816560c842336"


def _with_revision(request, revision: str, **options: object):
    backend_options = {**request.training.backend_options, "source_revision": revision, **options}
    return replace(request, training=replace(request.training, backend_options=backend_options))


def _structured_request(monkeypatch, algorithm: str):
    from posttrain.train import CAPORequest, CAPOSettings, GDPORequest, GDPOSettings

    base = _sampo_request()
    from posttrain.train import RewardComponentProjection, RewardProjection

    monkeypatch.setattr(
        FakeBridge,
        "reward_projection",
        RewardProjection(
            "fixture",
            "1",
            (RewardComponentProjection("outcome", "scalar"), RewardComponentProjection("quality", "metric", "quality")),
            "resolved_credit",
        ),
        raising=False,
    )
    loop = base.settings.loop
    settings = (
        GDPOSettings(
            id="gdpo",
            loop=loop,
            component_names=("outcome", "quality"),
            component_weights=(1.0, 2.0),
            shuffle_prompts=True,
        )
        if algorithm == "gdpo"
        else CAPOSettings(id="capo", loop=loop, shuffle_prompts=True)
    )
    return (
        GDPORequest(base.policy, base.bridge, settings, base.environment, base.training, base.inference)
        if isinstance(settings, GDPOSettings)
        else CAPORequest(base.policy, base.bridge, settings, base.environment, base.training, base.inference)
    )


@pytest.mark.parametrize("algorithm", ["gdpo", "capo"])
def test_structured_algorithms_select_explicit_native_loss_and_evidence_contract(monkeypatch, tmp_path, algorithm):
    request = _with_revision(_structured_request(monkeypatch, algorithm), POST5_REVISION)
    from posttrain.train.backends.trl.policy_config import _online_rl_arguments

    trl_arguments = _online_rl_arguments(request, tmp_path / "trl", {})
    assert trl_arguments["shuffle_dataset"] is True
    assert trl_arguments["vllm_importance_sampling_mode"] == "token_truncate"
    assert trl_arguments["vllm_importance_sampling_clip_min"] is None
    assert trl_arguments["vllm_importance_sampling_clip_max"] == 3.0
    plan = build_structured_launch_plan(request, tmp_path)
    from posttrain.train.backends.verl.launcher import _grpo_runtime_attributes

    assert _grpo_runtime_attributes(request, plan)["online_rl_algorithm"] == algorithm
    assert _grpo_runtime_attributes(request, plan)["shuffle_prompts"] is True
    assert plan.payload.algorithm.shuffle_prompts is True
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")
    overrides = build_hydra_overrides(plan, tmp_path / "data", tmp_path / "agent", tmp_path / "checkpoints")
    assert f"algorithm.adv_estimator={algorithm}" in overrides
    assert "data.shuffle=true" in overrides
    assert "actor_rollout_ref.actor.policy_loss.loss_mode=token_clip" in overrides
    assert "actor_rollout_ref.actor.kl_loss_type=k3_unclipped" in overrides
    assert "algorithm.filter_groups.enable=false" in overrides
    assert "+algorithm.structured_rewards.max_admission_attempts=3" in overrides
    config_path = tmp_path / "agent.json"
    _write_agent_config(plan.payload, config_path)
    config = json.loads(config_path.read_text())
    assert config[0]["structured_algorithm"] == algorithm


@pytest.mark.parametrize("algorithm", ["gdpo", "capo"])
def test_structured_algorithms_fail_before_verl_starts_on_a_fork_without_their_loss(monkeypatch, tmp_path, algorithm):
    """Regression: post4 registers neither token_clip nor k3_unclipped, so GDPO/CAPO
    crashed at veRL's first actor update. The worker now rejects such a revision."""

    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")
    request = _with_revision(_structured_request(monkeypatch, algorithm), POST4_REVISION)
    plan = build_structured_launch_plan(request, tmp_path)

    with pytest.raises(ValueError, match=f"{POST4_REVISION} does not register k3_unclipped, token_clip"):
        build_hydra_overrides(plan, tmp_path / "data", tmp_path / "agent", tmp_path / "checkpoints")

    # A dirty candidate checkout is identified by its content digest; it is not gated here.
    dirty = _with_revision(_structured_request(monkeypatch, algorithm), POST4_REVISION, source_dirty=True)
    build_hydra_overrides(
        build_structured_launch_plan(dirty, tmp_path), tmp_path / "data", tmp_path / "agent", tmp_path / "checkpoints"
    )


def _pinned_verl_revision() -> str:
    import tomllib

    import posttrain.runtime_images as runtime_images

    profile = (
        Path(runtime_images.__file__).parent / "containers" / "posttrain-job-kinds" / "verl-py313" / "profile.toml"
    )
    return tomllib.loads(profile.read_text(encoding="utf-8"))["fork_revision"]


@pytest.mark.parametrize(
    "operation",
    [
        "gdpo",
        "capo",
        pytest.param(
            "olmo3",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason=(
                    "the veRL job kind pins 0.9.0.post5, which has the OLMo 3 loss but not active sampling; "
                    "remove this marker in the commit that pins the active-sampling release"
                ),
            ),
        ),
    ],
)
def test_pinned_verl_fork_registers_every_native_name_posttrain_requests(monkeypatch, tmp_path, operation):
    from posttrain.train.backends.verl.worker import fork_native_names, requested_fork_native_names

    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")
    revision = _pinned_verl_revision()
    if operation == "olmo3":
        manifest = _olmo3_manifest(tmp_path)
        manifest = VerlLaunchManifest.model_validate({**manifest.model_dump(), "backend_source_revision": revision})
        manifest.payload.training.backend_options["source_revision"] = revision
    else:
        request = _with_revision(_structured_request(monkeypatch, operation), revision)
        manifest = build_structured_launch_plan(request, tmp_path)
    # Ask for the names with the gate lifted, then require the pinned fork to register them.
    requested = requested_fork_native_names(
        build_hydra_overrides(
            VerlLaunchManifest.model_validate(
                {**manifest.model_dump(), "backend_source_revision": ACTIVE_SAMPLING_REVISION}
            ),
            tmp_path / "data",
            tmp_path / "agent",
            tmp_path / "checkpoints",
        )
    )
    expected = {"token_clip", "k3_unclipped"} | ({"active_sampling"} if operation == "olmo3" else set())
    assert requested == expected
    assert requested <= fork_native_names(revision), f"pinned veRL {revision} lacks {sorted(requested)}"


def test_verl_rejects_sampo_without_active_sampling(tmp_path):
    with pytest.raises(ValueError, match="VORTEX active sampling"):
        build_sampo_launch_plan(_sampo_request(), tmp_path)


def test_verl_dapo_uses_core_trainer_and_maps_all_dynamic_sampling_controls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _grpo_request()
    settings = replace(
        request.settings,
        algorithm="dapo",
        dynamic_sampling=DynamicGroupSampling(max_candidate_batches=7),
        overlong_buffer_tokens=32,
    )
    plan = build_grpo_launch_plan(replace(request, settings=settings), tmp_path)
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")

    overrides = build_hydra_overrides(
        plan,
        tmp_path / "rollouts.parquet",
        tmp_path / "agent-loop.json",
        tmp_path / "checkpoints",
    )

    assert "actor_rollout_ref.actor.loss_agg_mode=token-mean" in overrides
    assert "actor_rollout_ref.actor.clip_ratio_low=0.2" in overrides
    assert "actor_rollout_ref.actor.clip_ratio_high=0.28" in overrides
    assert "data.gen_batch_size=1" in overrides
    assert "algorithm.filter_groups.enable=true" in overrides
    assert "algorithm.filter_groups.metric=seq_reward" in overrides
    assert "algorithm.filter_groups.max_num_gen_batches=7" in overrides


def _olmo3_settings(settings: GRPOSettings, **changes: object) -> GRPOSettings:
    values: dict[str, object] = {
        "algorithm": "olmo3",
        "advantage_scaling": "none",
        "importance_sampling_mode": "token_truncate",
        "importance_sampling_clip_min": None,
        "importance_sampling_clip_max": 2.0,
        "active_sampling": ActiveGroupSampling(max_candidate_batches=4),
    }
    values.update(changes)
    return replace(settings, **values)


def _olmo3_request(*, max_num_seqs: int | None = None, **changes: object) -> GRPORequest:
    request = _with_revision(_grpo_request(), ACTIVE_SAMPLING_REVISION)
    if max_num_seqs is not None:
        engine = {**request.inference.engine, "max_num_seqs": max_num_seqs}
        request = replace(request, inference=replace(request.inference, engine=engine))
    return replace(request, settings=_olmo3_settings(request.settings, **changes))


def _olmo3_manifest(tmp_path: Path, *, max_num_seqs: int | None = None, **changes: object) -> VerlLaunchManifest:
    return build_grpo_launch_plan(_olmo3_request(max_num_seqs=max_num_seqs, **changes), tmp_path)


def test_verl_accepts_olmo3_and_maps_its_active_sampling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")
    oversampled = ActiveGroupSampling(max_candidate_batches=6, oversample=1, oversample_refill=2)
    with pytest.raises(ValueError, match="needs 4 concurrent episodes for the first round"):
        _olmo3_manifest(tmp_path, active_sampling=oversampled)
    manifest = _olmo3_manifest(tmp_path, max_num_seqs=4, active_sampling=oversampled)
    algorithm = manifest.payload.algorithm
    assert (algorithm.active_sampling, algorithm.active_sampling_max_candidate_batches) == (True, 6)
    assert (algorithm.active_sampling_oversample, algorithm.active_sampling_oversample_refill) == (1, 2)

    overrides = build_hydra_overrides(manifest, tmp_path / "r.parquet", tmp_path / "a.json", tmp_path / "c")

    for expected in (
        "algorithm.active_sampling.enable=true",
        "algorithm.active_sampling.max_candidate_batches=6",
        "algorithm.active_sampling.oversample=1",
        "algorithm.active_sampling.oversample_refill=2",
        "algorithm.active_sampling.reward_std_epsilon=0.0",
        "algorithm.active_sampling.metric=seq_reward",
    ):
        assert expected in overrides
    assert not any(value.startswith("algorithm.filter_groups.") for value in overrides)


def test_verl_olmo3_requires_a_fork_revision_with_active_sampling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")
    post5 = VerlLaunchManifest.model_validate(
        {**_olmo3_manifest(tmp_path).model_dump(), "backend_source_revision": POST5_REVISION}
    )

    with pytest.raises(ValueError, match=f"{POST5_REVISION} does not register active_sampling"):
        build_hydra_overrides(post5, tmp_path / "r.parquet", tmp_path / "a.json", tmp_path / "c")


@pytest.mark.parametrize(
    "override",
    ["algorithm.active_sampling.enable=false", "+algorithm.active_sampling.oversample=9"],
)
def test_verl_backend_options_cannot_replace_active_sampling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, override: str
) -> None:
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")
    data = _olmo3_manifest(tmp_path).model_dump()
    data["payload"]["training"]["backend_options"]["hydra_overrides"] = [override]

    with pytest.raises(ValueError, match="cannot replace selected"):
        build_hydra_overrides(
            VerlLaunchManifest.model_validate(data), tmp_path / "r.parquet", tmp_path / "a.json", tmp_path / "c"
        )


def test_verl_olmo3_manifest_requires_active_sampling(tmp_path: Path) -> None:
    data = _olmo3_manifest(tmp_path).model_dump()
    data["payload"]["algorithm"]["active_sampling"] = None

    with pytest.raises(ValidationError, match="fixed objective settings"):
        VerlLaunchManifest.model_validate(data)


def test_verl_maps_the_olmo3_objective_to_native_token_clip_and_rollout_correction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")
    manifest = _olmo3_manifest(tmp_path, beta=0.005)

    overrides = build_hydra_overrides(
        manifest, tmp_path / "rollouts.parquet", tmp_path / "agent-loop.json", tmp_path / "checkpoints"
    )

    for expected in (
        "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        "actor_rollout_ref.actor.clip_ratio_low=0.2",
        "actor_rollout_ref.actor.clip_ratio_high=0.272",
        "actor_rollout_ref.actor.policy_loss.loss_mode=token_clip",
        "actor_rollout_ref.actor.use_kl_loss=true",
        "actor_rollout_ref.actor.kl_loss_coef=0.005",
        "actor_rollout_ref.actor.kl_loss_type=k3_unclipped",
        "algorithm.norm_adv_by_std_in_grpo=false",
        "algorithm.rollout_correction.rollout_is=token",
        "algorithm.rollout_correction.rollout_is_threshold=2.0",
        "algorithm.rollout_correction.rollout_is_batch_normalize=false",
        "algorithm.rollout_correction.rollout_rs=null",
        "algorithm.rollout_correction.bypass_mode=false",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        "algorithm.use_kl_in_reward=False",
    ):
        assert expected in overrides
    assert not any(value.startswith("algorithm.filter_groups.") for value in overrides)


def test_verl_olmo3_requires_a_fork_revision_with_token_clip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")
    manifest = _olmo3_manifest(tmp_path)
    legacy = VerlLaunchManifest.model_validate({**manifest.model_dump(), "backend_source_revision": "a" * 40})

    with pytest.raises(
        ValueError, match="does not register active_sampling, k3_unclipped, token_clip, which the grpo objective"
    ):
        build_hydra_overrides(legacy, tmp_path / "r.parquet", tmp_path / "a.json", tmp_path / "c")


def test_verl_olmo3_manifest_rejects_a_changed_recipe(tmp_path: Path) -> None:
    data = _olmo3_manifest(tmp_path).model_dump()
    data["payload"]["algorithm"]["normalize_advantage_by_std"] = True

    with pytest.raises(ValidationError, match="fixed objective settings"):
        VerlLaunchManifest.model_validate(data)


def test_verl_grpo_keeps_its_historical_advantage_and_correction_mapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")
    plan = build_grpo_launch_plan(_grpo_request(), tmp_path)
    data = plan.model_dump()
    data["payload"]["algorithm"]["rollout_importance_sampling"] = "token"
    with pytest.raises(ValidationError, match="only for OLMo 3"):
        VerlLaunchManifest.model_validate(data)

    overrides = build_hydra_overrides(plan, tmp_path / "r.parquet", tmp_path / "a.json", tmp_path / "c")
    assert "actor_rollout_ref.actor.kl_loss_type=low_var_kl" in overrides
    assert "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean" in overrides
    assert not any(value.startswith("algorithm.rollout_correction") for value in overrides)
    assert not any(value.startswith("algorithm.norm_adv_by_std_in_grpo") for value in overrides)


@pytest.mark.parametrize(
    "override",
    [
        "algorithm.rollout_correction.rollout_is=sequence",
        "+algorithm.rollout_correction.rollout_is_threshold=5.0",
        "algorithm.norm_adv_by_std_in_grpo=true",
    ],
)
def test_verl_backend_options_cannot_replace_the_olmo3_objective(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, override: str
) -> None:
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")
    data = _olmo3_manifest(tmp_path).model_dump()
    data["payload"]["training"]["backend_options"]["hydra_overrides"] = [override]
    manifest = VerlLaunchManifest.model_validate(data)

    with pytest.raises(ValueError, match="cannot replace selected"):
        build_hydra_overrides(manifest, tmp_path / "r.parquet", tmp_path / "a.json", tmp_path / "c")


def test_verl_accepts_the_truncation_penalty_and_passes_it_to_the_agent_loop(tmp_path: Path) -> None:
    request = _grpo_request()
    plan = build_grpo_launch_plan(
        replace(request, settings=replace(request.settings, truncation_penalty=0.2)), tmp_path
    )
    assert plan.payload.algorithm.truncation_penalty == 0.2

    _write_agent_config(plan.payload, tmp_path / "agent-loop.json")
    config = json.loads((tmp_path / "agent-loop.json").read_text(encoding="utf-8"))
    assert config[0]["truncation_penalty"] == 0.2


def _shaping_rollout(reward: float, completion_tokens: int, *, truncated: bool) -> EnvironmentRollout:
    return EnvironmentRollout(
        example_id="task-1",
        prompt_ids=(1, 2),
        completion_ids=tuple(range(completion_tokens)),
        sampling_logprobs=(-0.1,) * completion_tokens,
        env_mask=(True,) * completion_tokens,
        reward=reward,
        is_truncated=truncated,
        trace=TraceObservation("test", "trace-shaping", {}),
    )


@pytest.mark.parametrize("algorithm", ["grpo", "dapo", "olmo3"])
@pytest.mark.parametrize(("reward", "tokens", "truncated"), [(0.0, 40, True), (0.5, 120, True), (1.0, 100, False)])
def test_verl_reward_shaping_is_the_trl_rule(algorithm: str, reward: float, tokens: int, truncated: bool) -> None:
    base = _grpo_request().settings
    if algorithm == "olmo3":
        settings = _olmo3_settings(base, truncation_penalty=0.2)
    elif algorithm == "dapo":
        settings = replace(base, algorithm="dapo", overlong_buffer_tokens=32, truncation_penalty=0.2)
    else:
        settings = replace(base, truncation_penalty=0.2)
    payload = grpo_algorithm_payload(settings)

    shaped = shaped_rollout_reward(
        _shaping_rollout(reward, tokens, truncated=truncated),
        max_completion_tokens=payload["max_completion_length"],
        overlong_buffer_tokens=payload["overlong_buffer_tokens"],
        overlong_penalty_factor=payload["overlong_penalty_factor"],
        truncation_penalty=payload["truncation_penalty"],
    )

    assert shaped == shape_online_reward(settings, reward, tokens, is_truncated=truncated)
    if truncated and algorithm != "dapo":
        assert shaped == pytest.approx(reward - 0.2)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"adaptive_curriculum": AdaptiveCurriculum(class_field="category")},
            "adaptive_curriculum is currently supported by the TRL backend only",
        ),
        ({"advantage_scaling": "batch"}, "advantage_scaling='batch' is currently supported by the TRL backend only"),
        ({"advantage_scaling": "none"}, "advantage_scaling='none' is currently supported by the TRL backend only"),
        ({"importance_sampling_mode": "token_mask"}, "importance_sampling_mode='token_mask'"),
        ({"importance_sampling_clip_min": None}, "importance_sampling_clip_min=None"),
        ({"importance_sampling_clip_max": 2.0}, "importance_sampling_clip_max=2.0"),
        ({"max_admission_attempts": 1}, "max_admission_attempts=1 is currently supported by the TRL backend only"),
    ],
)
def test_verl_rejects_grpo_settings_it_would_silently_ignore(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    request = _grpo_request()
    settings = replace(request.settings, **changes)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=re.escape(message)):
        build_grpo_launch_plan(replace(request, settings=settings), tmp_path)


def test_verl_grpo_settings_problem_accepts_the_defaults() -> None:
    assert verl_grpo_settings_problem(_grpo_request().settings) is None


def test_verl_maps_shared_checkpoint_retention_and_explicit_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = (tmp_path / "prior run" / "global_step_4").resolve()
    checkpoint.mkdir(parents=True)
    request = _grpo_request()
    request = replace(
        request,
        settings=replace(
            request.settings,
            loop=replace(request.settings.loop, checkpoint_steps=3, checkpoint_limit=2),
        ),
        resume_from=LocalArtifactRef(checkpoint, "a" * 64),
    )
    plan = build_grpo_launch_plan(request, tmp_path)
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")

    overrides = build_hydra_overrides(
        plan,
        tmp_path / "rollouts.parquet",
        tmp_path / "agent-loop.json",
        tmp_path / "checkpoints",
    )

    assert "trainer.save_freq=3" in overrides
    assert "trainer.max_actor_ckpt_to_keep=2" in overrides
    assert "trainer.max_critic_ckpt_to_keep=2" in overrides
    assert "trainer.resume_mode=resume_path" in overrides
    assert f"trainer.resume_from_path={json.dumps(str(checkpoint))}" in overrides


def test_verl_lora_rollout_loads_the_immutable_base_and_syncs_only_adapters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _grpo_request(update=LoRAUpdate(rank=8, alpha=16))
    plan = build_grpo_launch_plan(request, tmp_path)
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")

    overrides = build_hydra_overrides(
        plan,
        tmp_path / "rollouts.parquet",
        tmp_path / "agent-loop.json",
        tmp_path / "checkpoints",
    )

    assert "actor_rollout_ref.rollout.load_format=safetensors" in overrides
    assert "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" in overrides
    assert 'actor_rollout_ref.model.target_modules="all-linear"' in overrides


def test_verl_rollout_passes_selected_kv_cache_dtype_to_vllm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _grpo_request(update=LoRAUpdate(rank=8, alpha=16))
    inference = replace(
        request.inference,
        engine={
            **request.inference.engine,
            "max_model_len": 32_768,
            "max_num_batched_tokens": 4_096,
            "kv_cache_dtype": "turboquant_k8v4",
            "enable_chunked_prefill": True,
        },
        sampling={**request.inference.sampling, "max_tokens": 24_576},
    )
    settings = replace(
        request.settings,
        loop=replace(request.settings.loop, max_length=32_768),
        max_prompt_length=8_192,
        max_completion_length=24_576,
    )
    plan = build_grpo_launch_plan(
        replace(request, inference=inference, settings=settings),
        tmp_path,
    )
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")

    overrides = build_hydra_overrides(
        plan,
        tmp_path / "rollouts.parquet",
        tmp_path / "agent-loop.json",
        tmp_path / "checkpoints",
    )

    assert "actor_rollout_ref.rollout.max_model_len=32768" in overrides
    assert "actor_rollout_ref.rollout.max_num_batched_tokens=4096" in overrides
    assert "actor_rollout_ref.rollout.dtype=float16" in overrides
    assert "actor_rollout_ref.rollout.enable_chunked_prefill=true" in overrides
    assert ("+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=turboquant_k8v4") in overrides
    assert _uses_turboquant(plan.payload)


def test_verl_rollout_maps_native_mtp_without_enabling_mtp_loss(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _grpo_request(update=LoRAUpdate(rank=8, alpha=16))
    inference = replace(
        request.inference,
        engine={
            **request.inference.engine,
            "speculative_config": {"method": "mtp", "num_speculative_tokens": 1},
        },
    )
    plan = build_grpo_launch_plan(replace(request, inference=inference), tmp_path)
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")

    overrides = build_hydra_overrides(
        plan,
        tmp_path / "rollouts.parquet",
        tmp_path / "agent-loop.json",
        tmp_path / "checkpoints",
    )

    assert "actor_rollout_ref.model.mtp.enable=true" in overrides
    assert "actor_rollout_ref.model.mtp.enable_train=false" in overrides
    assert "actor_rollout_ref.model.mtp.enable_rollout=true" in overrides
    assert "actor_rollout_ref.model.mtp.method=mtp" in overrides
    assert "actor_rollout_ref.model.mtp.num_speculative_tokens=1" in overrides
    assert "actor_rollout_ref.rollout.disable_log_stats=false" in overrides


@pytest.mark.parametrize(
    ("speculative_config", "message"),
    [
        ({"method": "draft_model", "num_speculative_tokens": 1}, "only native MTP"),
        ({"method": "mtp", "num_speculative_tokens": 0}, "positive integer"),
    ],
)
def test_verl_rollout_rejects_unqualified_speculative_modes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    speculative_config: dict[str, object],
    message: str,
) -> None:
    request = _grpo_request()
    inference = replace(
        request.inference,
        engine={**request.inference.engine, "speculative_config": speculative_config},
    )
    plan = build_grpo_launch_plan(replace(request, inference=inference), tmp_path)
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")

    with pytest.raises(ValueError, match=message):
        build_hydra_overrides(
            plan,
            tmp_path / "rollouts.parquet",
            tmp_path / "agent-loop.json",
            tmp_path / "checkpoints",
        )


def test_verl_backend_options_append_native_hydra_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    request = _grpo_request()
    training = replace(
        request.training,
        backend_options={
            **request.training.backend_options,
            "hydra_overrides": ["actor_rollout_ref.actor.use_torch_compile=true"],
        },
    )
    plan = build_grpo_launch_plan(replace(request, training=training), tmp_path)
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")

    overrides = build_hydra_overrides(
        plan,
        tmp_path / "rollouts.parquet",
        tmp_path / "agent-loop.json",
        tmp_path / "checkpoints",
    )

    assert overrides[-1] == "actor_rollout_ref.actor.use_torch_compile=true"


def test_verl_backend_options_cannot_replace_selected_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    request = _grpo_request()
    training = replace(
        request.training,
        backend_options={
            **request.training.backend_options,
            "hydra_overrides": ["actor_rollout_ref.model.path=/unselected/model"],
        },
    )
    plan = build_grpo_launch_plan(replace(request, training=training), tmp_path)
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")

    with pytest.raises(ValueError, match="cannot replace selected"):
        build_hydra_overrides(
            plan,
            tmp_path / "rollouts.parquet",
            tmp_path / "agent-loop.json",
            tmp_path / "checkpoints",
        )


@pytest.mark.parametrize(
    "override",
    [
        "trainer.save_freq=10",
        "trainer.max_actor_ckpt_to_keep=10",
        "trainer.resume_mode=auto",
        "trainer.resume_from_path=/tmp/unselected",
    ],
)
def test_verl_backend_options_cannot_replace_shared_checkpoint_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    override: str,
) -> None:
    request = _grpo_request()
    training = replace(
        request.training,
        backend_options={
            **request.training.backend_options,
            "hydra_overrides": [override],
        },
    )
    plan = build_grpo_launch_plan(replace(request, training=training), tmp_path)
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")

    with pytest.raises(ValueError, match="checkpoint"):
        build_hydra_overrides(
            plan,
            tmp_path / "rollouts.parquet",
            tmp_path / "agent-loop.json",
            tmp_path / "checkpoints",
        )


def test_verl_metric_parser_accepts_v1_inline_console_format(tmp_path: Path) -> None:
    log = tmp_path / "verl-native.log"
    log.write_text(
        "step:1 - actor/pg_loss:0.0 - training/global_step:1 - training/num_turns/mean:np.float64(2.0)\n",
        encoding="utf-8",
    )

    assert _last_metrics(log) == {
        "step": 1.0,
        "actor/pg_loss": 0.0,
        "training/global_step": 1.0,
        "training/num_turns/mean": 2.0,
    }


def test_verl_structured_metric_sidecar_is_monotonic_and_backend_neutral(tmp_path: Path) -> None:
    path = tmp_path / "verl-metrics.jsonl"
    path.write_text(
        "\n".join(
            (
                json.dumps({"step": 1, "data": {"critic/rewards/mean": 0.25}}),
                json.dumps({"step": 2, "data": {"actor/pg_loss": -0.1}}),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    records = read_verl_metric_records(path)

    assert tuple(record.step for record in records) == (1, 2)
    assert records[0].data == {"critic/rewards/mean": 0.25}


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ('{"step": 2, "data": {}}\n{"step": 1, "data": {}}\n', "monotonic"),
        ('{"step": 1, "data": {"reward": NaN}}\n', "non-finite"),
        ('{"step": 1, "data": []}\n', "data object"),
        ("", "empty"),
    ],
)
def test_verl_structured_metric_sidecar_rejects_invalid_records(
    tmp_path: Path,
    lines: str,
    message: str,
) -> None:
    path = tmp_path / "verl-metrics.jsonl"
    path.write_text(lines, encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=message):
        read_verl_metric_records(path)


def test_verl_result_uses_validated_structured_metrics_as_native_summary(tmp_path: Path) -> None:
    output = tmp_path / "trainer"
    model = output / "model"
    checkpoint = output / "checkpoints" / "global_step_1"
    metrics = output / "verl-metrics.jsonl"
    retention = output / "retention-manifest.json"
    model.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    metrics.write_text(
        json.dumps({"step": 1, "data": {"actor/pg_loss": -0.1}}) + "\n",
        encoding="utf-8",
    )
    retention.write_text('{"schema_version": 1, "status": "completed"}\n', encoding="utf-8")
    payload = VerlWorkerResult.model_validate(
        {
            "summary": {
                "global_step": 1,
                "train_loss": -0.1,
                "runtime_seconds": 2.0,
                "samples_per_second": 1.0,
                "steps_per_second": 0.5,
            },
            "model_dir": model,
            "recovery_checkpoint": checkpoint,
            "metrics_file": metrics,
            "retention_manifest": retention,
        }
    )

    backend, records = _backend_result(payload, output)

    assert backend.summary_file == metrics
    assert backend.retention_manifest == retention
    assert records[0].data == {"actor/pg_loss": -0.1}

    outside = tmp_path / "outside.jsonl"
    outside.write_text(metrics.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="inside the run output"):
        _backend_result(payload.model_copy(update={"metrics_file": outside}), output)
    with pytest.raises(ValueError, match="inside the run output"):
        _backend_result(payload.model_copy(update={"retention_manifest": outside}), output)


def test_verl_failure_preserves_native_diagnostics(tmp_path: Path) -> None:
    observer = CaptureObserver()
    context = _context(tmp_path, observer)
    output = tmp_path / "trainer"
    output.mkdir()
    plan = build_grpo_launch_plan(_grpo_request(), output)
    plan.write(output / "posttrain-verl-launch.json")
    (output / "posttrain-verl.log").write_text("worker failure\n", encoding="utf-8")
    (output / "verl-native.log").write_text("native failure\n", encoding="utf-8")

    _record_failure_artifacts(context, plan, output)

    assert [artifact.name for artifact in observer.artifacts_seen] == [
        "training/diagnostics/verl/grpo/launch-manifest",
        "training/diagnostics/verl/grpo/worker-log",
        "training/diagnostics/verl/grpo/native-log",
    ]
    assert all(not artifact.required for artifact in observer.artifacts_seen)


def test_verl_failure_diagnostics_do_not_replace_the_worker_error(tmp_path: Path) -> None:
    class BackpressuredObserver(CaptureObserver):
        def artifact(self, artifact: ProducedArtifact) -> None:
            del artifact
            raise RuntimeError("artifact queue is full")

    context = _context(tmp_path, BackpressuredObserver())
    output = tmp_path / "trainer"
    output.mkdir()
    plan = build_grpo_launch_plan(_grpo_request(), output)
    plan.write(output / "posttrain-verl-launch.json")
    (output / "posttrain-verl.log").write_text("worker failure\\n", encoding="utf-8")

    _record_failure_artifacts_best_effort(context, plan, output)


def test_verl_structured_records_replay_through_shared_grpo_names(tmp_path: Path) -> None:
    request = _grpo_request()
    path = tmp_path / "verl-metrics.jsonl"
    path.write_text(
        json.dumps(
            {
                "step": 1,
                "data": {
                    "critic/rewards/mean": 0.75,
                    "critic/rewards/std": 0.25,
                    "actor/pg_loss": -0.1,
                    "actor/grad_norm": 0.5,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    observer = CaptureObserver()

    _replay_grpo_metrics(
        _context(tmp_path, observer),
        request,
        read_verl_metric_records(path),
    )

    values = observer.metrics_seen[-1].values
    assert values == {
        "train/rl/reward_mean": 0.75,
        "train/rl/reward_std": 0.25,
        "train/rl/policy_loss": -0.1,
        "train/grad_norm": 0.5,
    }
    assert observer.metrics_seen[-1].step == 1
    assert observer.metrics_seen[-1].attributes["training_backend"] == "verl"


def test_verl_replays_trace_keyed_algorithm_rewards_from_the_parent_process(tmp_path: Path) -> None:
    observer = CaptureObserver()

    _replay_trace_fact_updates(
        _context(tmp_path, observer),
        (VerlRolloutRewardRecord("trace-1", 4, 0.8, 0.6),),
    )

    update = observer.trace_fact_updates[0]
    assert update.external_id == "trace-1"
    assert update.facts.measures == {"algorithm_reward": 0.6}
    # Trackio accepts an enrichment only when it supplies algorithm_reward alone.
    assert update.facts.dimensions == {}
    assert update.facts.reward_components == ()
    assert update.attributes["optimizer_step"] == 4


def test_verl_rollout_reward_journal_is_validated(tmp_path: Path) -> None:
    journal = tmp_path / "verl-rollout-rewards.jsonl"
    journal.write_text(
        json.dumps({"trace_id": "trace-1", "step": 4, "task_reward": 0.8, "algorithm_reward": 0.6}) + "\n",
        encoding="utf-8",
    )

    assert read_verl_rollout_reward_records(journal) == (VerlRolloutRewardRecord("trace-1", 4, 0.8, 0.6),)


def test_verl_mtp_partial_runtime_counters_fail_closed(tmp_path: Path) -> None:
    request = _grpo_request()
    request = replace(
        request,
        inference=replace(
            request.inference,
            engine={
                **request.inference.engine,
                "speculative_config": {"method": "mtp", "num_speculative_tokens": 1},
            },
        ),
    )
    path = tmp_path / "verl-metrics.jsonl"
    path.write_text(
        json.dumps(
            {
                "step": 1,
                "data": {
                    "rollout/spec_accept_rate": 0.8,
                    "rollout/spec_accept_length": 1.8,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="partial MTP evidence"):
        _replay_grpo_metrics(
            _context(tmp_path, CaptureObserver()),
            request,
            read_verl_metric_records(path),
        )


def test_verl_isolated_environment_does_not_forward_tracking_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    monkeypatch.setenv("TRACKIO_API_KEY", "secret")
    monkeypatch.setenv("POSTTRAIN_KEEP", "yes")
    monkeypatch.setenv("VIRTUAL_ENV", "/opt/posttrain/venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/opt/posttrain/venv")
    monkeypatch.setenv("PATH", "/opt/posttrain/venv/bin:/usr/bin")
    projection = tmp_path / "projection"
    projection.mkdir()
    monkeypatch.setenv("POSTTRAIN_VERL_PYTHONPATH", str(projection))
    monkeypatch.setenv("PYTHONPATH", "/existing/pythonpath")

    environment = _isolated_environment(Path("/opt/posttrain-verl/bin/python"))

    assert "WANDB_API_KEY" not in environment
    assert "TRACKIO_API_KEY" not in environment
    assert "VIRTUAL_ENV" not in environment
    assert "UV_PROJECT_ENVIRONMENT" not in environment
    assert environment["POSTTRAIN_KEEP"] == "yes"
    assert environment["PATH"] == ("/opt/posttrain-verl/bin:/opt/posttrain/venv/bin:/usr/bin")
    assert environment["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0"
    assert environment["PYTHONPATH"] == str(projection)
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONSAFEPATH"] == "1"
    assert "/existing/pythonpath" not in environment["PYTHONPATH"]


def test_verl_isolated_environment_rejects_missing_worker_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "POSTTRAIN_VERL_PYTHONPATH",
        str(tmp_path / "missing"),
    )

    with pytest.raises(RuntimeError, match="packaged absolute"):
        _isolated_environment(Path("/opt/posttrain-verl/bin/python"))


def test_verl_popen_receives_only_the_selected_interpreter_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = build_grpo_launch_plan(_grpo_request(), tmp_path)
    manifest = tmp_path / "posttrain-verl-launch.json"
    captured: dict[str, object] = {}

    class Process:
        pass

    def popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return Process()

    monkeypatch.setenv("VIRTUAL_ENV", "/opt/posttrain/venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/opt/posttrain/venv")
    monkeypatch.setenv("PATH", "/opt/posttrain/venv/bin:/usr/bin")
    monkeypatch.setattr(
        "posttrain.train.backends.verl.launcher.subprocess.Popen",
        popen,
    )
    with (tmp_path / "worker.log").open("w", encoding="utf-8") as stream:
        process = _start_isolated_worker(
            plan,
            manifest=manifest,
            stdout=stream,
        )

    assert isinstance(process, Process)
    assert captured["command"] == plan.command
    assert captured["cwd"] == str(plan.working_directory)
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert "VIRTUAL_ENV" not in environment
    assert "UV_PROJECT_ENVIRONMENT" not in environment
    assert environment["PATH"].startswith("/opt/posttrain-verl/bin:")
    assert environment["POSTTRAIN_VERL_MANIFEST"] == str(manifest)
    assert captured["start_new_session"] is True


def test_verl_runtime_deadline_is_explicit_and_positive() -> None:
    request = _grpo_request()

    assert _runtime_timeout(request) is None
    assert (
        _runtime_timeout(
            replace(
                request,
                training=replace(request.training, runtime=replace(request.training.runtime, timeout_seconds=90)),
            )
        )
        == 90.0
    )
    with pytest.raises(ValueError, match="finite positive number"):
        replace(request.training.runtime, timeout_seconds=0)


def test_verl_trace_sync_receipt_is_compact_and_payload_free(tmp_path: Path) -> None:
    observer = CaptureObserver()
    context = _context(tmp_path, observer)
    plan = build_grpo_launch_plan(_grpo_request(), tmp_path)
    path = tmp_path / "traces.jsonl"
    tailer = AppendOnlyJsonlTailer(path, lambda _record: None)
    path.write_text('{"id":"trace-1","secret":"must-not-appear"}\n', encoding="utf-8")
    tailer.poll()

    _record_trace_sync_receipt(context, plan, tmp_path, tailer)

    receipt = tmp_path / "posttrain-verl-trace-sync.json"
    assert receipt.is_file()
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["emitted_records"] == 1
    assert payload["complete"] is True
    assert "secret" not in receipt.read_text(encoding="utf-8")
    assert observer.artifacts_seen[-1].name.endswith("trace-sync-receipt")


def test_verl_agent_loop_honors_selected_reasoning_mode(tmp_path: Path) -> None:
    path = tmp_path / "agent-loop.json"
    request = _grpo_request()
    training = replace(request.training, renderer=replace(request.training.renderer, reasoning_mode="thinking"))
    payload = build_grpo_launch_plan(replace(request, training=training), tmp_path).payload

    _write_agent_config(payload, path)

    config = json.loads(path.read_text(encoding="utf-8"))
    assert config[0]["enable_thinking"] is True
    assert config[0]["renderer_implementation"] == "qwen3.5"


def test_verl_streaming_reward_exposes_dynamic_filter_metric() -> None:
    assert streaming_reward_extra_info(
        task_reward=0.75,
        algorithm_reward=0.5,
    ) == {
        "seq_reward": 0.5,
        "task_reward": 0.75,
    }


def test_sampo_rejects_masked_truncation_for_bounded_replacement() -> None:
    with pytest.raises(RuntimeError, match="SAMPO requires replacement"):
        training_response_mask(
            (True, True, False),
            is_truncated=True,
            mask_truncated_completions=True,
            requires_complete_group=True,
        )


def test_non_sampo_truncation_remains_fully_masked() -> None:
    assert training_response_mask(
        (True, True, False),
        is_truncated=True,
        mask_truncated_completions=True,
        requires_complete_group=False,
    ) == [0, 0, 0]


def test_verl_preflight_rejects_models_outside_current_qwen35_qualification(tmp_path: Path) -> None:
    request = _grpo_request(model=LFM_25_12B_THINKING, family="lfm2.5")
    with pytest.raises(ValueError, match="currently qualifies only qwen3.5"):
        build_grpo_launch_plan(request, tmp_path)


def test_qwen35_distillation_translation_uses_native_exact_token_k1_loss(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fingerprint = "f" * 64
    student = replace(QWEN_35_2B, tokenizer_fingerprint=fingerprint)
    teacher = replace(
        QWEN_35_2B,
        id="models/qwen3.5-2b-teacher@test",
        tokenizer_fingerprint=fingerprint,
    )
    request = OnPolicyDistillationRequest(
        student=student,
        teacher=teacher,
        bridge=FakeBridge(),
        settings=OnPolicyDistillationSettings(
            "settings/distill-test@1",
            TrainingLoop(
                max_steps=1,
                max_length=384,
                per_device_batch_size=1,
                gradient_accumulation_steps=2,
                lr_scheduler_type="constant",
            ),
            num_generations=2,
            max_prompt_length=256,
            max_completion_length=128,
        ),
        environment=FakeEnvironment(),
        training=_training(update=FullParameterUpdate()),
        rollout_inference=replace(
            _inference(student),
            engine={
                **_inference(student).engine,
                "max_model_len": 32_768,
                "max_num_batched_tokens": 4_096,
                "kv_cache_dtype": "turboquant_k8v4",
                "enable_chunked_prefill": True,
            },
        ),
        teacher_inference=replace(
            _inference(
                teacher,
                purpose=("teacher-score",),
                identifier="inference/teacher@1",
            ),
            engine={
                "max_model_len": 32_768,
                "max_num_batched_tokens": 4_096,
                "max_num_seqs": 2,
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.4,
                "kv_cache_dtype": "turboquant_k8v4",
                "enable_chunked_prefill": True,
            },
        ),
    )

    plan = build_distillation_launch_plan(request, tmp_path)

    assert plan.operation == "distill"
    assert plan.payload.student is not None
    assert plan.payload.teacher is not None
    assert plan.payload.student.family == "qwen3.5"
    assert plan.payload.teacher.family == "qwen3.5"
    assert plan.payload.algorithm.model_dump(exclude_none=True) == {
        "advantage_estimator": "grpo",
        "loss_mode": "k1",
        "use_policy_gradient": True,
        "use_task_rewards": False,
        "temperature": 1.0,
        "num_prompts_per_step": 1,
        "num_generations": 2,
        "max_prompt_length": 256,
        "max_completion_length": 128,
    }
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")
    overrides = build_hydra_overrides(
        plan,
        tmp_path / "rollouts.parquet",
        tmp_path / "agent-loop.json",
        tmp_path / "checkpoints",
    )
    assert "distillation.enable_resource_pool=False" in overrides
    assert "distillation.distillation_loss.loss_mode=k1" in overrides
    assert "distillation.distillation_loss.use_policy_gradient=true" in overrides
    assert "distillation.distillation_loss.use_task_rewards=false" in overrides
    assert "distillation.teacher_models.teacher_model.inference.max_model_len=32768" in overrides
    assert "distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=4096" in overrides
    assert "distillation.teacher_models.teacher_model.inference.enable_chunked_prefill=true" in overrides
    assert (
        "+distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.kv_cache_dtype=turboquant_k8v4"
    ) in overrides

    default_teacher_scoring = plan.payload.teacher_scoring
    assert default_teacher_scoring is not None
    default_context_plan = plan.model_copy(
        update={
            "payload": plan.payload.model_copy(
                update={
                    "teacher_scoring": default_teacher_scoring.model_copy(
                        update={"engine": {}},
                    ),
                },
            ),
        },
    )
    default_context_overrides = build_hydra_overrides(
        default_context_plan,
        tmp_path / "default-rollouts.parquet",
        tmp_path / "default-agent-loop.json",
        tmp_path / "default-checkpoints",
    )
    assert "distillation.teacher_models.teacher_model.inference.max_model_len=385" in default_context_overrides
    assert "distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=385" in default_context_overrides


def test_verl_worker_cycles_small_dataset_to_one_complete_prompt_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = build_grpo_launch_plan(_grpo_request(), tmp_path)
    payload = plan.payload.model_copy(
        update={
            "algorithm": plan.payload.algorithm.model_copy(
                update={"num_prompts_per_step": 2},
            ),
        },
    )
    captured_rows: list[dict[str, object]] = []

    class FakeDataset:
        @classmethod
        def from_list(cls, rows):
            captured_rows.extend(rows)
            return SimpleNamespace(to_parquet=lambda path: None)

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(Dataset=FakeDataset))

    _write_dataset(payload, tmp_path / "rollouts.parquet")

    assert len(captured_rows) == 2
    assert [row["example_id"] for row in captured_rows] == ["train/000000", "train/000000"]


def test_verl_rollout_honours_the_bindings_prefix_caching_and_eager_choice(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")
    payload = build_grpo_launch_plan(_grpo_request(), tmp_path).model_dump()

    def overrides_for(engine: dict[str, object]) -> list[str]:
        payload["payload"]["rollout"]["engine"] = {**payload["payload"]["rollout"]["engine"], **engine}
        manifest = VerlLaunchManifest.model_validate(payload)
        return build_hydra_overrides(
            manifest, tmp_path / "rollouts.parquet", tmp_path / "agent-loop.json", tmp_path / "checkpoints"
        )

    chosen = overrides_for({"enable_prefix_caching": True, "enforce_eager": False})
    assert "actor_rollout_ref.rollout.enable_prefix_caching=true" in chosen
    assert "actor_rollout_ref.rollout.enforce_eager=false" in chosen
    omitted = overrides_for({"enable_prefix_caching": False})
    assert "actor_rollout_ref.rollout.enable_prefix_caching=false" in omitted


def _adapter_request(tmp_path: Path, *, rank: int = 8, beta: float = 0.01, kl_reference: str = "base") -> GRPORequest:
    adapter = tmp_path / "adapter"
    adapter.mkdir(exist_ok=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"r": rank, "lora_alpha": 16}))
    started = replace(
        QWEN_35_2B,
        artifact=LocalArtifactRef(adapter.resolve(), "a" * 64),
        form="adapter",
        revision=None,
        parent=QWEN_35_2B.id,
    )
    request = _grpo_request(model=started, update=LoRAUpdate(rank=8, alpha=16))
    return replace(request, settings=replace(request.settings, beta=beta, kl_reference=kl_reference))


def test_verl_continues_a_trained_adapter_on_its_foundation_weights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _adapter_request(tmp_path)
    plan = build_grpo_launch_plan(request, tmp_path / "out")
    policy = plan.payload.policy
    assert policy is not None and policy.base is not None
    assert (policy.base.repo_id, policy.base.revision) == (QWEN_35_2B.base.repo_id, QWEN_35_2B.base.revision)
    monkeypatch.setattr(
        "posttrain.train.backends.verl.worker._model_path",
        lambda artifact: str(artifact.path) if artifact.kind == "local" else "/models/foundation",
    )

    overrides = build_hydra_overrides(
        plan, tmp_path / "rollouts.parquet", tmp_path / "agent-loop.json", tmp_path / "checkpoints"
    )
    # The actor and rollout load the foundation; the actor attaches the starting adapter.
    assert "actor_rollout_ref.model.path=/models/foundation" in overrides
    assert f"actor_rollout_ref.model.lora_adapter_path={json.dumps(str(tmp_path / 'adapter'))}" in overrides
    assert "actor_rollout_ref.model.lora_rank=8" in overrides
    assert "actor_rollout_ref.actor.use_kl_loss=true" in overrides

    fresh = build_hydra_overrides(
        build_grpo_launch_plan(_grpo_request(update=LoRAUpdate(rank=8, alpha=16)), tmp_path / "fresh"),
        tmp_path / "rollouts.parquet",
        tmp_path / "agent-loop.json",
        tmp_path / "checkpoints",
    )
    assert not any(value.startswith("actor_rollout_ref.model.lora_adapter_path=") for value in fresh)


def test_verl_adapter_continuation_rejects_what_it_cannot_honor(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot hold a frozen copy of the starting adapter"):
        build_grpo_launch_plan(_adapter_request(tmp_path, kl_reference="start"), tmp_path / "out")
    # Without a KL penalty there is no reference to hold.
    build_grpo_launch_plan(_adapter_request(tmp_path, beta=0.0, kl_reference="start"), tmp_path / "out")
    with pytest.raises(ValueError, match="starting adapter has LoRA rank 4 but the training binding selects rank 8"):
        build_grpo_launch_plan(_adapter_request(tmp_path, rank=4), tmp_path / "out")
    request = _adapter_request(tmp_path)
    with pytest.raises(ValueError, match="full-parameter updates cannot continue from an unmerged PEFT adapter"):
        build_grpo_launch_plan(replace(request, training=_training()), tmp_path / "out")


def _validated_manifest(tmp_path: Path, worktree: Path, revision: str, **options: object) -> VerlLaunchManifest:
    request = _grpo_request()
    training = replace(
        request.training,
        backend_options={
            **request.training.backend_options,
            "working_directory": str(worktree),
            "source_revision": revision,
            **options,
        },
    )
    return build_grpo_launch_plan(replace(request, training=training), tmp_path / "out")


def test_verl_worker_accepts_the_kind_images_source_snapshot_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The 0.4.12 veRL kind removes .git and records the revision; the worker must not run git."""

    from posttrain.train.backends.verl import worker as verl_worker

    revision = "9fd6e7a31396ba33a29233cc869ab05b0a9e5a80"
    snapshot = tmp_path / "workdir"
    snapshot.mkdir()
    (snapshot / ".posttrain-source-revision").write_text(revision + "\n")
    monkeypatch.setattr("importlib.metadata.version", lambda name: "0.9.0.post5")

    def no_git(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"git must not run for an immutable snapshot: {args}")

    monkeypatch.setattr(verl_worker.subprocess, "run", no_git)
    verl_worker._validate_runtime(_validated_manifest(tmp_path, snapshot, revision, source_dirty=False))

    with pytest.raises(RuntimeError, match=f"is at {revision}, expected immutable revision 0{{40}}"):
        verl_worker._validate_runtime(_validated_manifest(tmp_path, snapshot, "0" * 40))
    # A snapshot is clean by construction, so a dirty selection cannot match it.
    with pytest.raises(RuntimeError, match="dirty state is False, expected True"):
        verl_worker._validate_runtime(_validated_manifest(tmp_path, snapshot, revision, source_dirty=True))
    (snapshot / ".git").mkdir()
    with pytest.raises(RuntimeError, match="unexpectedly retains Git metadata"):
        verl_worker._validate_runtime(_validated_manifest(tmp_path, snapshot, revision))


def test_verl_worker_reads_a_development_checkout_with_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from posttrain.train.backends.verl import worker as verl_worker

    checkout = tmp_path / "verl"
    checkout.mkdir()
    environment = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
        "HOME": str(tmp_path),
        "PATH": "/usr/bin:/bin",
    }
    for command in (["git", "init", "-q"], ["git", "commit", "-q", "--allow-empty", "-m", "base"]):
        subprocess.run(command, cwd=checkout, check=True, env=environment)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=checkout, check=True, capture_output=True, text=True
    ).stdout.strip()
    monkeypatch.setattr("importlib.metadata.version", lambda name: "0.9.0.post5")

    verl_worker._validate_runtime(_validated_manifest(tmp_path, checkout, revision, source_dirty=False))
    (checkout / "patch.py").write_text("changed = True\n")
    with pytest.raises(RuntimeError, match="dirty state is True, expected False"):
        verl_worker._validate_runtime(_validated_manifest(tmp_path, checkout, revision, source_dirty=False))


def test_verl_launcher_rejects_olmo3_on_a_fork_without_active_sampling(tmp_path: Path) -> None:
    request = _with_revision(_olmo3_request(), POST5_REVISION)

    with pytest.raises(ValueError, match=f"{POST5_REVISION} has no round-based active sampling"):
        build_grpo_launch_plan(request, tmp_path)
    # A dirty candidate checkout is identified by its content digest and not gated here.
    build_grpo_launch_plan(_with_revision(_olmo3_request(), POST5_REVISION, source_dirty=True), tmp_path)
