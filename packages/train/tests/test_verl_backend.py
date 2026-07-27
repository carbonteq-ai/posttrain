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
    LocalArtifactRef,
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
    DynamicGroupSampling,
    FullParameterUpdate,
    GRPORequest,
    GRPOSettings,
    LoRAUpdate,
    OnPolicyDistillationRequest,
    OnPolicyDistillationSettings,
    SAMPORequest,
    SAMPOSettings,
    TrainingBinding,
    TrainingLoop,
    TrainingRuntime,
)
from posttrain.train.api import _distillation_backend, _grpo_backend, _sampo_backend
from posttrain.train.backends.verl.contracts import VerlLaunchManifest, VerlWorkerResult
from posttrain.train.backends.verl.launcher import (
    _backend_result,
    _isolated_environment,
    _record_failure_artifacts,
    _replay_grpo_metrics,
    _runtime_timeout,
    _start_isolated_worker,
    build_distillation_launch_plan,
    build_grpo_launch_plan,
    build_sampo_launch_plan,
)
from posttrain.train.backends.verl.metrics import read_verl_metric_records
from posttrain.train.backends.verl.worker import (
    _last_metrics,
    _uses_turboquant,
    _write_agent_config,
    build_hydra_overrides,
)
from pydantic import ValidationError


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
            TrainingLoop(max_steps=1, max_length=384, per_device_batch_size=2),
            max_prompt_length=256,
            max_completion_length=128,
        ),
        environment=FakeEnvironment(),
        training=_training(family=family, update=update),
        inference=_inference(model),
    )


def _sampo_request() -> SAMPORequest:
    training = _training()
    training = replace(
        training,
        backend_options={
            **training.backend_options,
            "dynamic_sampling_recipe_working_directory": "/opt/src/verl-recipe",
            "dynamic_sampling_recipe_source_revision": "2" * 40,
        },
    )
    return SAMPORequest(
        policy=QWEN_35_2B,
        bridge=FakeBridge(),
        settings=SAMPOSettings(
            "settings/sampo-test@1",
            TrainingLoop(max_steps=1, max_length=384, per_device_batch_size=2),
            max_prompt_length=256,
            max_completion_length=128,
        ),
        environment=FakeEnvironment(),
        training=training,
        inference=_inference(QWEN_35_2B),
    )


def test_backend_resolver_exposes_general_verl_product_for_both_operations() -> None:
    assert _grpo_backend("verl@a35908c").__module__ == "posttrain.train.backends.verl.launcher"
    assert _sampo_backend("verl@a35908c").__module__ == "posttrain.train.backends.verl.launcher"
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


def test_verl_sampo_maps_hierarchical_advantages_gspo_and_dynamic_sampling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _sampo_request()
    plan = build_sampo_launch_plan(request, tmp_path)
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")

    overrides = build_hydra_overrides(
        plan,
        tmp_path / "rollouts.parquet",
        tmp_path / "agent-loop.json",
        tmp_path / "checkpoints",
    )
    agent_config_path = tmp_path / "sampo-agent-loop.json"
    _write_agent_config(plan.payload, agent_config_path)
    agent_config = json.loads(agent_config_path.read_text(encoding="utf-8"))

    assert plan.operation == "sampo"
    assert plan.payload.algorithm.advantage_estimator == "sampo"
    assert plan.recipe_working_directory == Path("/opt/src/verl-recipe")
    assert plan.recipe_source_revision == "2" * 40
    assert "algorithm.adv_estimator=sampo" in overrides
    assert "algorithm.sampo.discount_gamma=0.95" in overrides
    assert "algorithm.sampo.step_advantage_weight=1.0" in overrides
    assert "algorithm.sampo.advantage_normalization=mean" in overrides
    assert "actor_rollout_ref.actor.policy_loss.loss_mode=gspo" in overrides
    assert "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean" in overrides
    assert "actor_rollout_ref.actor.clip_ratio_low=0.003" in overrides
    assert "actor_rollout_ref.actor.clip_ratio_high=0.004" in overrides
    assert "algorithm.filter_groups.enable=true" in overrides
    assert "algorithm.filter_groups.max_num_gen_batches=3" in overrides
    assert agent_config[0]["emit_sampo_metadata"] is True


def test_verl_sampo_requires_the_pinned_dynamic_sampling_recipe(tmp_path: Path) -> None:
    request = _sampo_request()
    request = replace(
        request,
        training=replace(
            request.training,
            backend_options={
                key: value
                for key, value in request.training.backend_options.items()
                if not key.startswith("dynamic_sampling_recipe_")
            },
        ),
    )

    with pytest.raises(ValueError, match="dynamic_sampling_recipe_working_directory"):
        build_sampo_launch_plan(request, tmp_path)


def test_verl_dapo_uses_pinned_recipe_and_maps_all_dynamic_sampling_controls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _grpo_request()
    training = replace(
        request.training,
        backend_options={
            **request.training.backend_options,
            "dapo_recipe_working_directory": "/opt/src/verl-recipe",
            "dapo_recipe_source_revision": "2" * 40,
        },
    )
    settings = replace(
        request.settings,
        algorithm="dapo",
        dynamic_sampling=DynamicGroupSampling(max_candidate_batches=7),
        overlong_buffer_tokens=32,
    )
    plan = build_grpo_launch_plan(replace(request, training=training, settings=settings), tmp_path)
    monkeypatch.setattr("posttrain.train.backends.verl.worker._model_path", lambda model: "/models/qwen35")

    overrides = build_hydra_overrides(
        plan,
        tmp_path / "rollouts.parquet",
        tmp_path / "agent-loop.json",
        tmp_path / "checkpoints",
    )

    assert plan.recipe_working_directory == Path("/opt/src/verl-recipe")
    assert plan.recipe_source_revision == "2" * 40
    assert "actor_rollout_ref.actor.loss_agg_mode=token-mean" in overrides
    assert "actor_rollout_ref.actor.clip_ratio_low=0.2" in overrides
    assert "actor_rollout_ref.actor.clip_ratio_high=0.28" in overrides
    assert "data.gen_batch_size=1" in overrides
    assert "algorithm.filter_groups.enable=true" in overrides
    assert "algorithm.filter_groups.metric=seq_reward" in overrides
    assert "algorithm.filter_groups.max_num_gen_batches=7" in overrides


def test_verl_dapo_dynamic_sampling_requires_a_pinned_recipe(tmp_path: Path) -> None:
    request = _grpo_request()
    settings = replace(
        request.settings,
        algorithm="dapo",
        dynamic_sampling=DynamicGroupSampling(),
    )

    with pytest.raises(ValueError, match="dynamic_sampling_recipe_working_directory"):
        build_grpo_launch_plan(replace(request, settings=settings), tmp_path)


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


def test_verl_agent_loop_honors_selected_reasoning_mode(tmp_path: Path) -> None:
    path = tmp_path / "agent-loop.json"
    request = _grpo_request()
    training = replace(request.training, renderer=replace(request.training.renderer, reasoning_mode="thinking"))
    payload = build_grpo_launch_plan(replace(request, training=training), tmp_path).payload

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
    assert plan.payload.student is not None
    assert plan.payload.teacher is not None
    assert plan.payload.student.family == "qwen3.5"
    assert plan.payload.teacher.family == "qwen3.5"
    assert plan.payload.algorithm.model_dump(exclude_none=True) == {
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
        plan,
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
