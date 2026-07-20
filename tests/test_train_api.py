from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

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
    QWEN35_SFT_SMOKE,
    CompletedRollout,
    DPORequest,
    GRPORequest,
    PreferenceDataset,
    PreferenceExample,
    RolloutDataset,
    RolloutExample,
    RolloutScore,
    SFTRequest,
    SupervisedDataset,
    SupervisedExample,
    dpo,
    grpo,
    sft,
)
from posttrain.train.backends.trl.common import BackendTrainingResult
from posttrain.train.backends.trl.grpo import _grpo_arguments, _reward_function
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
class FakeRLEnvironment:
    dataset: RolloutDataset = field(default_factory=_rollouts)

    async def score(self, rollout: CompletedRollout) -> RolloutScore:
        return RolloutScore(
            1.0,
            TraceObservation("test", rollout.example_id, {"completion": rollout.completion}),
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
                FakeRLEnvironment(),
                QWEN35_GRPO_SMOKE,
            ),
            runner=_backend,
        )
    assert result.technique == "grpo"
    assert result.model_artifact.metadata["dataset_id"] == "gsm8k-grpo-smoke-v1"


def test_grpo_backend_configures_one_generation_schedule_control(tmp_path: Path) -> None:
    request = GRPORequest(
        ModelVariant.foundation(QWEN_35_2B),
        FakeRLEnvironment(),
        QWEN35_GRPO_SMOKE,
    )

    arguments = _grpo_arguments(request, tmp_path, {"enable_thinking": False})

    assert arguments["generation_batch_size"] == 2
    assert "steps_per_generation" not in arguments
    assert arguments["use_vllm"] is True
    assert arguments["vllm_mode"] == "colocate"
    assert arguments["vllm_enable_sleep_mode"] is True
    assert arguments["vllm_weight_name_prefix"] is None
    assert arguments["vllm_weight_sync_mode"] == "lora"
    assert arguments["vllm_engine_kwargs"] == {
        "language_model_only": True,
        "skip_mm_profiling": True,
        "kv_cache_memory_bytes": 64 * 1024 * 1024,
    }
    assert arguments["vllm_speculative_config"] is None

    mtp_request = GRPORequest(
        ModelVariant.foundation(QWEN_35_2B),
        FakeRLEnvironment(),
        QWEN35_GRPO_MTP_SMOKE,
    )
    mtp_arguments = _grpo_arguments(mtp_request, tmp_path, {"enable_thinking": False})
    assert mtp_arguments["vllm_speculative_config"] == {
        "method": "qwen3_next_mtp",
        "num_speculative_tokens": 2,
    }


def test_trl_reward_adapter_preserves_rollout_identity_and_observes_native_trace(tmp_path: Path) -> None:
    observer = Observer()
    context = _context(tmp_path.resolve(), observer)
    request = GRPORequest(
        ModelVariant.foundation(QWEN_35_2B),
        FakeRLEnvironment(),
        QWEN35_GRPO_SMOKE,
    )

    rewards = asyncio.run(
        _reward_function(context, request)(
            completions=[[{"role": "assistant", "content": "#### 4"}]],
            completion_ids=[[10, 11, 12]],
            example_id=["gsm8k/train/0"],
            trainer_state=type("TrainerState", (), {"global_step": 3})(),
        )
    )

    assert rewards == [1.0]
    assert observer.traces[0].external_id == "gsm8k/train/0"
    assert observer.traces[0].payload["completion"] == "#### 4"
    assert observer.traces[0].attributes["technique"] == "grpo"
    assert observer.traces[0].attributes["model_profile_id"] == QWEN_35_2B.id


def test_preference_contract_rejects_unordered_or_identical_pairs() -> None:
    with pytest.raises(ValueError, match="strictly greater"):
        PreferenceExample("bad", "p", "yes", "no", 0.0, 1.0)
    with pytest.raises(ValueError, match="must differ"):
        PreferenceExample("bad", "p", "same", "same", 1.0, 0.0)
