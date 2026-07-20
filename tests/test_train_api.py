from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from posttrain.common import (
    EventObservation,
    ExecutionContext,
    Invocation,
    Job,
    JobAction,
    MetricBatchObservation,
    MetricObservation,
    ModelVariant,
    ProducedArtifact,
    RunAttempt,
    TraceObservation,
)
from posttrain.common.profiles import QWEN_35_2B
from posttrain.train import (
    QWEN35_DPO_SMOKE,
    QWEN35_GRPO_MTP_SMOKE,
    QWEN35_GRPO_SMOKE,
    QWEN35_RENDERER,
    QWEN35_SFT_SMOKE,
    DPORequest,
    GRPOProfile,
    GRPORequest,
    GRPORolloutProfile,
    PreferenceDataset,
    PreferenceExample,
    RolloutDataset,
    RolloutExample,
    SFTRequest,
    SupervisedDataset,
    SupervisedExample,
    TrainingLoop,
    TrainingRollout,
    dpo,
    grpo,
    sft,
)
from posttrain.train.backends.trl.common import BackendTrainingResult, trainer_lifecycle
from posttrain.train.backends.trl.grpo import _grpo_arguments, _rollout_function
from posttrain.train.results import TrainingSummary


@dataclass
class Observer:
    events: list[EventObservation] = field(default_factory=list)
    metrics_seen: list[MetricBatchObservation] = field(default_factory=list)
    artifacts: list[ProducedArtifact] = field(default_factory=list)
    traces: list[TraceObservation] = field(default_factory=list)

    def event(self, observation: EventObservation) -> None:
        self.events.append(observation)

    def metric(self, observation: MetricObservation) -> None:
        self.metrics_seen.append(MetricBatchObservation({observation.name: observation.value}, observation.step))

    def metrics(self, observation: MetricBatchObservation) -> None:
        self.metrics_seen.append(observation)

    def trace(self, observation: TraceObservation) -> None:
        self.traces.append(observation)

    def artifact(self, artifact: ProducedArtifact) -> None:
        self.artifacts.append(artifact)


def _supervised() -> SupervisedDataset:
    return SupervisedDataset(
        "gsm8k-sft-smoke-v1",
        "a" * 40,
        (SupervisedExample("gsm8k/train/0", "What is 2 + 2?", "#### 4"),),
    )


def _preferences() -> PreferenceDataset:
    return PreferenceDataset(
        "gsm8k-dpo-smoke-v1",
        "a" * 40,
        (
            PreferenceExample(
                "gsm8k/train/0",
                "What is 2 + 2?",
                "#### 4",
                "#### 5",
                1.0,
                0.0,
                rejected_trace_id="trace-1",
            ),
        ),
    )


def _rollouts() -> RolloutDataset:
    return RolloutDataset(
        "gsm8k-grpo-smoke-v1",
        "a" * 40,
        (RolloutExample("gsm8k/train/0", "What is 2 + 2?", {"task_index": 0}),),
    )


@dataclass
class FakeRLBridge:
    dataset: RolloutDataset = field(default_factory=_rollouts)

    async def run(self, batch, generator) -> tuple[TrainingRollout, ...]:
        del generator
        return tuple(
            TrainingRollout(
                example_id=example_id,
                prompt_ids=(1, 2),
                completion_ids=(10, 11, 12),
                sampling_logprobs=(-0.1, -0.2, -0.3),
                env_mask=(True, True, True),
                reward=1.0,
                is_truncated=False,
                trace=TraceObservation(
                    "test",
                    f"trace-{index}",
                    {"example_id": example_id, "step": batch.step},
                ),
            )
            for index, example_id in enumerate(batch.example_ids)
        )

    def finalize(self) -> tuple[ProducedArtifact, ...]:
        return ()


def _context(workspace: Path, observer: Observer) -> ExecutionContext:
    return ExecutionContext(
        Job("gsm8k-posttraining", "b" * 40, "GSM8K post-training"),
        JobAction("gsm8k-posttraining", "sft", "training-sft"),
        Invocation.new(),
        RunAttempt.new(),
        workspace,
        observer,
    )


def _backend(_: ExecutionContext, __: object, output_dir: Path) -> BackendTrainingResult:
    root = output_dir.parent
    adapter = root / "adapter"
    checkpoint = output_dir / "checkpoint-2"
    adapter.mkdir()
    checkpoint.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoint / "trainer_state.json").write_text("{}")
    summary_file = root / "training-summary.json"
    summary_file.write_text("{}")
    return BackendTrainingResult(TrainingSummary(2, 0.5, 1.0, 2.0, 2.0), adapter, checkpoint, summary_file)


def test_trainer_lifecycle_closes_distributed_runtime_after_failure() -> None:
    closed: list[bool] = []
    trainer = SimpleNamespace(accelerator=SimpleNamespace(end_training=lambda: closed.append(True)))

    with pytest.raises(RuntimeError, match="training failed"):
        with trainer_lifecycle(trainer):
            raise RuntimeError("training failed")

    assert closed == [True]


def test_sft_operation_separates_adapter_recovery_and_summary_artifacts() -> None:
    observer = Observer()
    with tempfile.TemporaryDirectory() as raw:
        context = _context(Path(raw).resolve(), observer)
        result = sft(
            context,
            SFTRequest(ModelVariant.foundation(QWEN_35_2B), _supervised(), QWEN35_SFT_SMOKE),
            runner=_backend,
        )
    assert result.model.format == "peft-adapter"
    assert result.summary.global_step == 2
    assert [artifact.kind for artifact in observer.artifacts] == [
        "model-adapter",
        "training-checkpoint",
        "training-summary",
    ]
    assert observer.events[-1].name == "training_completed"


def test_dpo_operation_preserves_preference_dataset_identity() -> None:
    observer = Observer()
    with tempfile.TemporaryDirectory() as raw:
        context = _context(Path(raw).resolve(), observer)
        result = dpo(
            context,
            DPORequest(ModelVariant.foundation(QWEN_35_2B), _preferences(), QWEN35_DPO_SMOKE),
            runner=_backend,
        )
    assert result.technique == "dpo"
    assert result.model_artifact.metadata["dataset_id"] == "gsm8k-dpo-smoke-v1"


def test_grpo_operation_reuses_training_artifact_contract() -> None:
    observer = Observer()
    with tempfile.TemporaryDirectory() as raw:
        context = _context(Path(raw).resolve(), observer)
        result = grpo(
            context,
            GRPORequest(
                ModelVariant.foundation(QWEN_35_2B),
                FakeRLBridge(),
                QWEN35_GRPO_SMOKE,
            ),
            runner=_backend,
        )
    assert result.technique == "grpo"
    assert result.model_artifact.metadata["dataset_id"] == "gsm8k-grpo-smoke-v1"


def test_grpo_backend_configures_one_generation_schedule_control(tmp_path: Path) -> None:
    request = GRPORequest(
        ModelVariant.foundation(QWEN_35_2B),
        FakeRLBridge(),
        QWEN35_GRPO_SMOKE,
    )

    arguments = _grpo_arguments(request, tmp_path, {"enable_thinking": False})

    assert arguments["generation_batch_size"] == 2
    assert "steps_per_generation" not in arguments
    assert arguments["max_completion_length"] == 384
    assert arguments["use_vllm"] is True
    assert arguments["vllm_mode"] == "colocate"
    assert arguments["vllm_enable_sleep_mode"] is True
    assert arguments["vllm_max_model_length"] == 640
    assert arguments["vllm_weight_name_prefix"] is None
    assert arguments["vllm_weight_sync_mode"] == "lora"
    assert arguments["vllm_importance_sampling_mode"] == "sequence_truncate"
    assert arguments["vllm_importance_sampling_clip_min"] == 0.1
    assert arguments["vllm_importance_sampling_clip_max"] == 3.0
    assert arguments["vllm_engine_kwargs"] == {
        "language_model_only": True,
        "skip_mm_profiling": True,
        "kv_cache_memory_bytes": 64 * 1024 * 1024,
    }
    assert arguments["vllm_speculative_config"] is None

    mtp_request = GRPORequest(
        ModelVariant.foundation(QWEN_35_2B),
        FakeRLBridge(),
        QWEN35_GRPO_MTP_SMOKE,
    )
    mtp_arguments = _grpo_arguments(mtp_request, tmp_path, {"enable_thinking": False})
    assert mtp_arguments["vllm_speculative_config"] == {
        "method": "qwen3_next_mtp",
        "num_speculative_tokens": 2,
    }


def test_grpo_profile_requires_engine_window_to_cover_declared_generation_bounds() -> None:
    with pytest.raises(ValueError, match="model length must cover"):
        GRPOProfile(
            "qwen3.5-2b/invalid-window-v1",
            "qwen3.5",
            QWEN35_RENDERER,
            TrainingLoop(max_steps=1, per_device_batch_size=2),
            max_prompt_length=256,
            max_completion_length=384,
            rollout=GRPORolloutProfile(
                "qwen3.5-2b/invalid-window-v1",
                "vllm",
                vllm_mode="colocate",
                gpu_memory_utilization=0.2,
                max_model_length=512,
                importance_sampling_mode="sequence_truncate",
                importance_sampling_clip_min=0.1,
                importance_sampling_clip_max=3.0,
            ),
        )


def test_vllm_rollout_requires_explicit_importance_sampling_strategy() -> None:
    with pytest.raises(ValueError, match="explicit importance-sampling"):
        GRPORolloutProfile(
            "qwen3.5-2b/missing-is-v1",
            "vllm",
            vllm_mode="colocate",
            gpu_memory_utilization=0.2,
            max_model_length=512,
        )


def test_trl_rollout_adapter_preserves_identity_rewards_masks_and_native_traces(monkeypatch, tmp_path: Path) -> None:
    observer = Observer()
    context = _context(tmp_path.resolve(), observer)
    request = GRPORequest(
        ModelVariant.foundation(QWEN_35_2B),
        FakeRLBridge(),
        QWEN35_GRPO_SMOKE,
    )
    monkeypatch.setattr(
        "posttrain.train.backends.trl.online_rl.TrlPolicyGenerator",
        lambda *args: object(),
    )

    output = _rollout_function(context, request, object())(
        [[{"role": "user", "content": "What is 2 + 2?"}]],
        SimpleNamespace(state=SimpleNamespace(global_step=3)),
        inputs=[{"example_id": "gsm8k/train/0"}],
    )

    assert output["rollout_reward"] == [1.0]
    assert output["prompt_ids"] == [[1, 2]]
    assert output["completion_ids"] == [[10, 11, 12]]
    assert output["env_mask"] == [[True, True, True]]
    assert output["is_truncated"] == [False]
    assert observer.traces[0].external_id == "trace-0"
    assert observer.traces[0].payload["example_id"] == "gsm8k/train/0"
    assert observer.traces[0].payload["step"] == 3
    assert observer.traces[0].attributes["technique"] == "grpo"
    assert observer.traces[0].attributes["model_profile_id"] == QWEN_35_2B.id


def test_preference_contract_rejects_unordered_or_identical_pairs() -> None:
    with pytest.raises(ValueError, match="strictly greater"):
        PreferenceExample("bad", "p", "yes", "no", 0.0, 1.0)
    with pytest.raises(ValueError, match="must differ"):
        PreferenceExample("bad", "p", "same", "same", 1.0, 0.0)
