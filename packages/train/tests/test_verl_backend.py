"""Contract tests for the isolated veRL backend adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from posttrain.common import (
    EventObservation,
    ExecutionTarget,
    InferenceBinding,
    MetricBatchObservation,
    MetricObservation,
    ProducedArtifact,
    RunContext,
    TraceObservation,
)
from posttrain.common.variants import LFM_25_12B_THINKING, QWEN_35_2B
from posttrain.data import RolloutDataset, RolloutExample
from posttrain.train import (
    LFM25_RENDERER,
    QWEN35_RENDERER,
    FullParameterUpdate,
    GRPORequest,
    GRPOSettings,
    LoRAUpdate,
    OnPolicyDistillationRequest,
    OnPolicyDistillationSettings,
    TrainingBinding,
    TrainingLoop,
)
from posttrain.train.api import _distillation_backend, _grpo_backend
from posttrain.train.backends.verl.launcher import (
    _backend_result,
    _isolated_environment,
    _record_failure_artifacts,
    _replay_grpo_metrics,
    _runtime_timeout,
    build_distillation_launch_plan,
    build_grpo_launch_plan,
)
from posttrain.train.backends.verl.metrics import read_verl_metric_records
from posttrain.train.backends.verl.worker import (
    _last_metrics,
    _uses_turboquant,
    _write_agent_config,
    build_hydra_overrides,
)


@dataclass(frozen=True)
class FakeEnvironment:
    id: str = "envs/test@1"
    revision: str = "1"


class FakeBridge:
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

    def event(self, observation: EventObservation) -> None:
        del observation

    def metric(self, observation: MetricObservation) -> None:
        self.metrics_seen.append(MetricBatchObservation({observation.name: observation.value}, observation.step))

    def metrics(self, observation: MetricBatchObservation) -> None:
        self.metrics_seen.append(observation)

    def trace(self, observation: TraceObservation) -> None:
        del observation

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
        runtime={
            "global_batch_size": 2,
            "nodes": 1,
            "devices_per_node": 2,
            "parameter_offload": True,
            "optimizer_offload": True,
        },
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
            TrainingLoop(max_steps=1, max_length=384, per_device_batch_size=2),
            max_prompt_length=256,
            max_completion_length=128,
        ),
        environment=FakeEnvironment(),
        training=_training(family=family, update=update),
        inference=_inference(model),
    )


def test_backend_resolver_exposes_general_verl_product_for_both_operations() -> None:
    assert _grpo_backend("verl@a35908c").__module__ == "posttrain.train.backends.verl.launcher"
    assert _distillation_backend("verl@a35908c").__module__ == "posttrain.train.backends.verl.launcher"
    with pytest.raises(ValueError, match="unsupported GRPO"):
        _grpo_backend("unknown@1")


def test_qwen35_grpo_translation_is_deterministic_and_backend_neutral(tmp_path: Path) -> None:
    request = _grpo_request(update=LoRAUpdate(rank=16, alpha=32))
    output_dir = tmp_path / "trainer"
    output_dir.mkdir()

    first = build_grpo_launch_plan(request, output_dir)
    second = build_grpo_launch_plan(request, output_dir)

    assert first == second
    assert first.backend.startswith("verl@")
    assert first.operation == "grpo"
    assert first.payload["policy"]["family"] == "qwen3.5"
    assert first.payload["training"]["update"] == {
        "kind": "lora",
        "rank": 16,
        "alpha": 32,
        "dropout": 0.0,
        "target_modules": "all-linear",
    }
    assert first.payload["training"]["renderer"] == {
        "id": "qwen3.5-off-v1",
        "implementation": "qwen3.5",
        "reasoning_mode": "off",
    }
    assert first.payload["environment"]["examples"][0]["id"] == "train/000000"
    assert first.command[0] == "/opt/posttrain-verl/bin/python"


def test_grpo_worker_maps_prompt_groups_generations_and_kl_without_importing_verl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = replace(_grpo_request(), settings=replace(_grpo_request().settings, beta=0.02))
    plan = build_grpo_launch_plan(request, tmp_path)
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")

    overrides = build_hydra_overrides(
        {
            "operation": plan.operation,
            "payload": plan.payload,
        },
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
    assert "trainer.logger=['console','file']" in overrides


def test_verl_lora_rollout_loads_the_immutable_base_and_syncs_only_adapters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _grpo_request(update=LoRAUpdate(rank=8, alpha=16))
    plan = build_grpo_launch_plan(request, tmp_path)
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")

    overrides = build_hydra_overrides(
        {"operation": plan.operation, "payload": plan.payload},
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
        {"operation": plan.operation, "payload": plan.payload},
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
        {"operation": plan.operation, "payload": plan.payload},
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
            {"operation": plan.operation, "payload": plan.payload},
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
        {"operation": plan.operation, "payload": plan.payload},
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
            {"operation": plan.operation, "payload": plan.payload},
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
    model.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    metrics.write_text(
        json.dumps({"step": 1, "data": {"actor/pg_loss": -0.1}}) + "\n",
        encoding="utf-8",
    )
    payload = {
        "summary": {
            "global_step": 1,
            "train_loss": -0.1,
            "runtime_seconds": 2.0,
            "samples_per_second": 1.0,
            "steps_per_second": 0.5,
        },
        "model_dir": str(model),
        "recovery_checkpoint": str(checkpoint),
        "metrics_file": str(metrics),
    }

    backend, records = _backend_result(payload, output / "posttrain-result.json", output)

    assert backend.summary_file == metrics
    assert records[0].data == {"actor/pg_loss": -0.1}

    outside = tmp_path / "outside.jsonl"
    outside.write_text(metrics.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="inside the run output"):
        _backend_result({**payload, "metrics_file": str(outside)}, output / "result.json", output)


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
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    monkeypatch.setenv("TRACKIO_API_KEY", "secret")
    monkeypatch.setenv("POSTTRAIN_KEEP", "yes")

    environment = _isolated_environment()

    assert "WANDB_API_KEY" not in environment
    assert "TRACKIO_API_KEY" not in environment
    assert environment["POSTTRAIN_KEEP"] == "yes"
    assert environment["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0"


def test_verl_runtime_deadline_is_explicit_and_positive() -> None:
    request = _grpo_request()

    assert _runtime_timeout(request) is None
    assert (
        _runtime_timeout(
            replace(
                request, training=replace(request.training, runtime={**request.training.runtime, "timeout_seconds": 90})
            )
        )
        == 90.0
    )
    with pytest.raises(ValueError, match="positive number"):
        _runtime_timeout(
            replace(
                request, training=replace(request.training, runtime={**request.training.runtime, "timeout_seconds": 0})
            )
        )


def test_verl_agent_loop_honors_selected_reasoning_mode(tmp_path: Path) -> None:
    path = tmp_path / "agent-loop.json"
    payload = {
        "environment": {"bridge_snapshot": "/tmp/bridge.pkl"},
        "training": {"renderer": {"reasoning_mode": "thinking"}},
    }

    _write_agent_config(payload, path)

    assert '"enable_thinking": true' in path.read_text(encoding="utf-8")


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
            TrainingLoop(max_steps=1, max_length=384, per_device_batch_size=2),
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
    assert plan.payload["student"]["family"] == "qwen3.5"
    assert plan.payload["teacher"]["family"] == "qwen3.5"
    assert plan.payload["algorithm"] == {
        "advantage_estimator": "grpo",
        "loss_mode": "k1",
        "use_policy_gradient": False,
        "use_task_rewards": False,
        "temperature": 1.0,
        "num_prompts_per_step": 1,
        "num_generations": 2,
        "max_prompt_length": 256,
        "max_completion_length": 128,
    }
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")
    overrides = build_hydra_overrides(
        {"operation": plan.operation, "payload": plan.payload},
        tmp_path / "rollouts.parquet",
        tmp_path / "agent-loop.json",
        tmp_path / "checkpoints",
    )
    assert "distillation.teacher_models.teacher_model.inference.max_model_len=32768" in overrides
    assert "distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=4096" in overrides
    assert "distillation.teacher_models.teacher_model.inference.enable_chunked_prefill=true" in overrides
    assert (
        "+distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.kv_cache_dtype=turboquant_k8v4"
    ) in overrides
