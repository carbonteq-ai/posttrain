"""Tests for the training package API."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from posttrain.common import (
    Catalog,
    CatalogRef,
    EventObservation,
    ExecutionTarget,
    HubModelRef,
    InferenceBinding,
    JsonValue,
    LocalArtifactRef,
    MetricBatchObservation,
    MetricObservation,
    ModelVariant,
    ProducedArtifact,
    RunContext,
    TraceObservation,
)
from posttrain.common.variants import QWEN_35_2B
from posttrain.data import (
    DatasetDescriptor,
    PreferenceDataset,
    PreferenceExample,
    RolloutDataset,
    RolloutExample,
    SupervisedDataset,
    SupervisedExample,
)
from posttrain.train import (
    QWEN35_DPO_SMOKE,
    QWEN35_GRPO_MTP_SMOKE,
    QWEN35_GRPO_SMOKE,
    QWEN35_RENDERER,
    QWEN35_SFT_SMOKE,
    DPORequest,
    DynamicGroupSampling,
    EnvironmentRollout,
    EnvironmentRolloutEvidence,
    FullParameterUpdate,
    GRPOObservationFeatures,
    GRPORequest,
    GRPOSettings,
    LoRAUpdate,
    OnPolicyDistillationRequest,
    OnPolicyDistillationSettings,
    QLoRAUpdate,
    QuantizationPlan,
    SFTRequest,
    SFTSettings,
    SFTValidationSettings,
    TrainingBinding,
    TrainingLoop,
    TrainingParallelism,
    TrainingRuntime,
    TransformRequest,
    distill,
    dpo,
    grpo,
    normalize_grpo_metrics,
    sft,
    transform,
)
from posttrain.train.backends.trl.common import BackendTrainingResult, callback_type, trainer_lifecycle
from posttrain.train.backends.trl.distillation import (
    _distillation_arguments,
    _teacher_server_command,
)
from posttrain.train.backends.trl.distillation import (
    _rollout_function as _distillation_rollout_function,
)
from posttrain.train.backends.trl.grpo import (
    _configure_liger_loss,
    _grpo_arguments,
    _grpo_runtime_attributes,
    _rollout_function,
)
from posttrain.train.catalog_schema import TrainingRuntimeSchema
from posttrain.train.results import TrainingSummary
from pydantic import ValidationError


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
        (
            SupervisedExample(
                "gsm8k/train/0",
                (
                    {"role": "user", "content": "What is 2 + 2?"},
                    {"role": "assistant", "content": "#### 4"},
                ),
                (1,),
            ),
        ),
    )


def _validation_supervised() -> SupervisedDataset:
    return SupervisedDataset(
        "gsm8k-sft-validation-v1",
        "b" * 40,
        (
            SupervisedExample(
                "gsm8k/validation/0",
                (
                    {"role": "user", "content": "What is 3 + 3?"},
                    {"role": "assistant", "content": "#### 6"},
                ),
                (1,),
            ),
        ),
    )


def _preferences() -> PreferenceDataset:
    return PreferenceDataset(
        "gsm8k-dpo-smoke-v1",
        "a" * 40,
        (
            PreferenceExample(
                "gsm8k/train/0",
                ({"role": "user", "content": "What is 2 + 2?"},),
                ({"role": "assistant", "content": "#### 4"},),
                ({"role": "assistant", "content": "#### 5"},),
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

    async def run(self, batch, generator) -> tuple[EnvironmentRollout, ...]:
        del generator
        return tuple(
            EnvironmentRollout(
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


@dataclass
class FailingFinalizeBridge(FakeRLBridge):
    def finalize(self) -> tuple[ProducedArtifact, ...]:
        raise RuntimeError("finalize failed")


@dataclass
class TrackingFinalizeBridge(FakeRLBridge):
    finalized: bool = False

    def finalize(self) -> tuple[ProducedArtifact, ...]:
        self.finalized = True
        return ()


@dataclass
class EvidenceReplayBridge(FakeRLBridge):
    def evidence(self) -> EnvironmentRolloutEvidence:
        return EnvironmentRolloutEvidence(
            metrics=(
                MetricBatchObservation(
                    {
                        "train/rl/rollouts_completed": 8.0,
                        "train/rl/reward_std": 0.25,
                    },
                    step=3,
                    attributes={"observation_source": "verifiers"},
                ),
            ),
        )


@dataclass(frozen=True)
class FakeEnvironment:
    id: str = "gsm8k-train-candidates"
    revision: str = "a" * 40


@dataclass
class LazySupervisedSource:
    dataset: SupervisedDataset = field(default_factory=_supervised)
    loads: int = 0

    @property
    def descriptor(self) -> DatasetDescriptor:
        return self.dataset.descriptor

    def load(self) -> SupervisedDataset:
        self.loads += 1
        return self.dataset


def _context(workspace: Path, observer: Observer) -> RunContext:
    return _run_context(workspace, observer)


def _run_context(
    workspace: Path,
    observer: Observer,
    *,
    job_kind: str = "train/sft",
    run_id: str = "runs/sft-smoke",
) -> RunContext:
    return RunContext(
        project_id="projects/gsm8k",
        work_package_id="work-packages/gsm8k-slice-3",
        run_id=run_id,
        job_kind=job_kind,
        job_definition_version="1",
        workspace=workspace,
        observer=observer,
    )


def _target(identifier: str = "targets/local-cuda-8gb") -> ExecutionTarget:
    return ExecutionTarget(identifier, "1", "nvidia-cuda", 8, {"world_size": 1})


def _training(
    *, target: ExecutionTarget | None = None, update: FullParameterUpdate | LoRAUpdate | QLoRAUpdate | None = None
) -> TrainingBinding:
    return TrainingBinding(
        "training/qwen3.5-trl-test@1",
        "1",
        "trl@1.8.0",
        QWEN35_RENDERER,
        update or QLoRAUpdate(),
        target or _target(),
        TrainingParallelism(sequence_length_divisor=2),
        TrainingRuntime(global_batch_size=2),
    )


def _inference(
    model: ModelVariant,
    *,
    target: ExecutionTarget | None = None,
    max_model_len: int = 640,
    speculative: bool = False,
    kv_cache_dtype: str | None = None,
) -> InferenceBinding:
    engine: dict[str, JsonValue] = {
        "mode": "colocate",
        "sleep_during_optimization": True,
        "gpu_memory_utilization": 0.2,
        "tensor_parallel_size": 1,
        "max_model_len": max_model_len,
        "text_only": True,
        "skip_mm_profiling": True,
        "enforce_eager": True,
        "kv_cache_memory_bytes": 64 * 1024 * 1024,
        "weight_sync_mode": "lora",
    }
    if speculative:
        engine["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": 1,
        }
    if kv_cache_dtype is not None:
        engine["kv_cache_dtype"] = kv_cache_dtype
    return InferenceBinding(
        "inference/qwen3.5-2b-grpo-test@1",
        "1",
        model,
        "vllm@0.25.1",
        model.renderer_contract,
        engine,
        {"max_tokens": 384, "temperature": 0.8, "top_p": 1.0},
        target or _target("targets/rollout-cuda-8gb"),
        ("rollout",),
    )


def _teacher_inference(model: ModelVariant) -> InferenceBinding:
    return InferenceBinding(
        "inference/qwen3.5-teacher-score-test@1",
        "1",
        model,
        "vllm@0.25.1",
        model.renderer_contract,
        {"base_url": "http://teacher.invalid:8000", "max_model_len": 640},
        {"temperature": 1.0},
        _target("targets/teacher-cuda-8gb"),
        ("teacher-score",),
    )


def test_training_runtime_is_closed_and_validates_normalized_values() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        TrainingRuntime(global_batch_size=0)
    with pytest.raises(ValueError, match="finite positive number"):
        TrainingRuntime(timeout_seconds=float("inf"))
    with pytest.raises(ValidationError, match="extra_forbidden"):
        TrainingRuntimeSchema.model_validate({"global_batch_size": 2, "use_liger_kernel": True})
    with pytest.raises(ValidationError, match="int_type"):
        TrainingRuntimeSchema.model_validate({"nodes": "1"})


def _distillation_request() -> OnPolicyDistillationRequest:
    fingerprint = "f" * 64
    student = replace(QWEN_35_2B, tokenizer_fingerprint=fingerprint)
    teacher = replace(
        QWEN_35_2B,
        id="models/qwen3.5-2b-teacher@test",
        tokenizer_fingerprint=fingerprint,
    )
    settings = OnPolicyDistillationSettings(
        "qwen3.5/on-policy-distill-test@1",
        TrainingLoop(max_steps=1, max_length=640, per_device_batch_size=2),
        num_generations=2,
        max_prompt_length=256,
        max_completion_length=384,
    )
    return OnPolicyDistillationRequest(
        student=student,
        teacher=teacher,
        bridge=FakeRLBridge(),
        settings=settings,
        environment=FakeEnvironment(),
        training=_training(),
        rollout_inference=_inference(student),
        teacher_inference=_teacher_inference(teacher),
    )


def test_teacher_server_command_preserves_memory_and_kv_selections() -> None:
    request = _distillation_request()
    inference = replace(
        request.teacher_inference,
        engine={
            **request.teacher_inference.engine,
            "gpu_memory_utilization": 0.69,
            "dtype": "bfloat16",
            "kv_cache_dtype": "fp8",
            "enable_prefix_caching": True,
            "enforce_eager": True,
        },
    )

    command = _teacher_server_command(
        replace(request, teacher_inference=inference),
        host="127.0.0.1",
        port=8000,
    )

    assert command[-7:] == (
        "--enforce_eager",
        "--dtype",
        "bfloat16",
        "--kv_cache_dtype",
        "fp8",
        "--enable_prefix_caching",
        "true",
    )


def _backend(_: RunContext, __: object, *args: object) -> BackendTrainingResult:
    output_dir = args[-1]
    assert isinstance(output_dir, Path)
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


@pytest.mark.parametrize(
    "values",
    [
        (0, 0.5, 1.0, 2.0, 2.0),
        (1, float("nan"), 1.0, 2.0, 2.0),
        (1, 0.5, -1.0, 2.0, 2.0),
        (1, 0.5, 1.0, -2.0, 2.0),
    ],
)
def test_completed_training_summary_rejects_invalid_values(values: tuple[int, float, float, float, float]) -> None:
    with pytest.raises(ValueError):
        TrainingSummary(*values)


def test_backend_result_requires_workspace_scoped_existing_outputs(tmp_path: Path) -> None:
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    outside = (tmp_path / "outside").resolve()
    outside.mkdir()
    summary = workspace / "summary.json"
    summary.write_text("{}")
    result = BackendTrainingResult(TrainingSummary(1, 0.5, 1.0, 1.0, 1.0), outside, None, summary)

    with pytest.raises(ValueError, match="inside the run workspace"):
        result.validate(workspace)

    missing_model = workspace / "missing-model"
    result = replace(result, model_dir=missing_model)
    with pytest.raises(FileNotFoundError, match="missing-model"):
        result.validate(workspace)


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
            SFTRequest(QWEN_35_2B, _supervised(), QWEN35_SFT_SMOKE, _training()),
            runner=_backend,
        )
    assert result.model.form == "adapter"
    assert result.model.parent == result.source_model.id
    assert result.model.provenance["parameter_update_kind"] == "qlora"
    assert result.model_artifact.name.endswith("/sft/qlora/adapter")
    assert result.summary.global_step == 2
    assert [artifact.kind for artifact in observer.artifacts] == [
        "model-adapter",
        "training-checkpoint",
        "training-summary",
    ]
    assert observer.events[-1].name == "training_completed"


def test_training_operation_records_retention_manifest(tmp_path: Path) -> None:
    observer = Observer()

    def backend_with_retention(context, request, dataset, validation_dataset, output_dir):
        result = _backend(context, request, dataset, validation_dataset, output_dir)
        manifest = context.workspace / "retention-manifest.json"
        manifest.write_text('{"schema_version": 1, "status": "completed"}\n', encoding="utf-8")
        return replace(result, retention_manifest=manifest)

    sft(
        _context(tmp_path.resolve(), observer),
        SFTRequest(QWEN_35_2B, _supervised(), QWEN35_SFT_SMOKE, _training()),
        runner=backend_with_retention,
    )

    assert [artifact.kind for artifact in observer.artifacts] == [
        "model-adapter",
        "training-checkpoint",
        "training-summary",
        "training-retention-manifest",
    ]


def test_lora_and_full_updates_materialize_distinct_model_forms_and_artifact_kinds(tmp_path: Path) -> None:
    lora_observer = Observer()
    lora = sft(
        _context(tmp_path / "lora", lora_observer),
        SFTRequest(QWEN_35_2B, _supervised(), QWEN35_SFT_SMOKE, _training(update=LoRAUpdate())),
        runner=_backend,
    )
    full_observer = Observer()
    full = sft(
        _context(tmp_path / "full", full_observer),
        SFTRequest(QWEN_35_2B, _supervised(), QWEN35_SFT_SMOKE, _training(update=FullParameterUpdate())),
        runner=_backend,
    )

    assert lora.model.form == "adapter"
    assert lora.model_artifact.kind == "model-adapter"
    assert lora.model_artifact.name.endswith("/sft/lora/adapter")
    assert full.model.form == "full-finetuned"
    assert full.model_artifact.kind == "model-weights"
    assert full.model_artifact.name.endswith("/sft/full/weights")


def test_training_preflight_rejects_double_quantization_and_update_mode_drift(tmp_path: Path) -> None:
    quantized = replace(
        QWEN_35_2B,
        id="models/qwen3.5-2b@awq-test",
        form="weight-quantized",
        quantization={"method": "awq", "bits": 4},
        parent=QWEN_35_2B.id,
    )
    with pytest.raises(ValueError, match="persistent weight-quantized"):
        SFTRequest(quantized, _supervised(), QWEN35_SFT_SMOKE, _training())

    local = tmp_path / "adapter"
    local.mkdir()
    adapter = replace(
        QWEN_35_2B,
        id="models/qwen3.5-2b/sft-qlora-test",
        artifact=LocalArtifactRef(local, "a" * 64),
        form="adapter",
        revision=None,
        digest="a" * 64,
        parent=QWEN_35_2B.id,
        provenance={"parameter_update_kind": "qlora"},
    )
    with pytest.raises(ValueError, match="produced with 'qlora'"):
        SFTRequest(adapter, _supervised(), QWEN35_SFT_SMOKE, _training(update=LoRAUpdate()))


def test_canonical_sft_request_uses_settings_and_explicit_training_binding() -> None:
    observer = Observer()
    source = LazySupervisedSource()
    with tempfile.TemporaryDirectory() as raw:
        result = sft(
            _run_context(Path(raw).resolve(), observer),
            SFTRequest(
                model=QWEN_35_2B,
                data=source,
                settings=QWEN35_SFT_SMOKE,
                training=_training(),
            ),
            runner=_backend,
        )

    assert isinstance(QWEN35_SFT_SMOKE, SFTSettings)
    assert source.loads == 1
    assert result.model.parent == QWEN_35_2B.id
    assert observer.events[0].attributes["training_settings_id"] == QWEN35_SFT_SMOKE.id
    assert observer.events[0].attributes["work_package_id"] == "work-packages/gsm8k-slice-3"


def test_transform_materializes_catalog_resolvable_child_variant(tmp_path: Path) -> None:
    observer = Observer()
    source = QWEN_35_2B
    plan = QuantizationPlan(
        "qwen3.5-2b/awq-4bit-test-v1",
        "1",
        "awq",
        "awq-test-recipe-v1",
        "a" * 64,
        "int4-group128",
        output_weight_precision="int4",
        output_quantization={"bits": 4, "group_size": 128},
    )

    def runner(context, request, output_dir):
        del context, request
        output_dir.mkdir(parents=True)
        (output_dir / "model.safetensors").write_bytes(b"quantized-weights")
        (output_dir / "posttrain-quantization-summary.json").write_text(
            '{"runtime_versions":{"llmcompressor":"0.12.1.dev71+g655b83d1"}}',
            encoding="utf-8",
        )
        return output_dir

    result = transform(
        _run_context(tmp_path.resolve(), observer),
        TransformRequest(source, plan, _target(), "models/qwen3.5-2b@awq-int4-test"),
        runner=runner,
    )
    catalog = Catalog.open(
        {"layer_id": "framework-test", "model": {source.id: source}},
        overlays=(
            {
                "layer_id": "run-output-test",
                "model": {result.model.id: result.model},
            },
        ),
        scope="projects/gsm8k",
    )
    resolved = catalog.resolve(CatalogRef("model", result.model.id))

    assert resolved.value == result.model
    assert resolved.source_layer == "overlay"
    assert result.model.form == "weight-quantized"
    assert result.model.parent == source.id
    assert result.model.quantization["method"] == "awq"
    assert result.artifact.reference == result.model.artifact
    assert result.artifact.metadata["runtime_versions"] == {"llmcompressor": "0.12.1.dev71+g655b83d1"}
    assert result.model.provenance["runtime_versions"] == {"llmcompressor": "0.12.1.dev71+g655b83d1"}


def test_sft_materializes_trainer_neutral_source_once_and_passes_snapshot_to_backend() -> None:
    observer = Observer()
    source = LazySupervisedSource()
    snapshots: list[SupervisedDataset] = []

    def backend(context, request, dataset, validation_dataset, output_dir):
        assert validation_dataset is None
        snapshots.append(dataset)
        return _backend(context, request, dataset, output_dir)

    with tempfile.TemporaryDirectory() as raw:
        context = _context(Path(raw).resolve(), observer)
        sft(
            context,
            SFTRequest(QWEN_35_2B, source, QWEN35_SFT_SMOKE, _training()),
            runner=backend,
        )

    assert source.loads == 1
    assert snapshots == [source.dataset]
    assert observer.events[0].attributes["dataset_schema_version"] == 1


def test_sft_materializes_disjoint_validation_source_and_records_identity() -> None:
    observer = Observer()
    train_source = LazySupervisedSource()
    validation_source = LazySupervisedSource(_validation_supervised())
    snapshots: list[tuple[SupervisedDataset, SupervisedDataset | None]] = []

    def backend(context, request, dataset, validation_dataset, output_dir):
        snapshots.append((dataset, validation_dataset))
        return _backend(context, request, dataset, validation_dataset, output_dir)

    settings = SFTSettings(
        "qwen3.5-2b/sft-validation-test",
        TrainingLoop(max_steps=2),
        validation=SFTValidationSettings(steps=1, on_start=True),
    )
    with tempfile.TemporaryDirectory() as raw:
        result = sft(
            _context(Path(raw).resolve(), observer),
            SFTRequest(
                QWEN_35_2B,
                train_source,
                settings,
                _training(),
                validation_data=validation_source,
            ),
            runner=backend,
        )

    assert train_source.loads == 1
    assert validation_source.loads == 1
    assert snapshots == [(train_source.dataset, validation_source.dataset)]
    assert observer.events[0].attributes["validation_dataset_id"] == validation_source.dataset.id
    assert result.model_artifact.metadata["validation_dataset_revision"] == validation_source.dataset.revision


def test_sft_requires_validation_source_and_schedule_together() -> None:
    settings = SFTSettings(
        "qwen3.5-2b/sft-validation-test",
        TrainingLoop(max_steps=2),
        validation=SFTValidationSettings(steps=1),
    )
    with pytest.raises(ValueError, match="selected together"):
        SFTRequest(QWEN_35_2B, _supervised(), settings, _training())
    with pytest.raises(ValueError, match="selected together"):
        SFTRequest(
            QWEN_35_2B,
            _supervised(),
            QWEN35_SFT_SMOKE,
            _training(),
            validation_data=_validation_supervised(),
        )


def test_sft_rejects_example_overlap_between_train_and_validation() -> None:
    settings = SFTSettings(
        "qwen3.5-2b/sft-validation-test",
        TrainingLoop(max_steps=2),
        validation=SFTValidationSettings(steps=1),
    )
    with tempfile.TemporaryDirectory() as raw:
        with pytest.raises(ValueError, match="datasets overlap"):
            sft(
                _context(Path(raw).resolve(), Observer()),
                SFTRequest(
                    QWEN_35_2B,
                    _supervised(),
                    settings,
                    _training(),
                    validation_data=_supervised(),
                ),
                runner=_backend,
            )


def test_sft_rejects_source_that_changes_its_declared_identity() -> None:
    source = LazySupervisedSource()

    class DriftedSource:
        descriptor = DatasetDescriptor("drifted", "declared", "supervised", num_examples=1)

        def load(self):
            return source.dataset

    with tempfile.TemporaryDirectory() as raw:
        context = _context(Path(raw).resolve(), Observer())
        with pytest.raises(ValueError, match="declared descriptor"):
            sft(
                context,
                SFTRequest(
                    QWEN_35_2B,
                    DriftedSource(),
                    QWEN35_SFT_SMOKE,
                    _training(),
                ),
                runner=_backend,
            )


def test_dpo_operation_preserves_preference_dataset_identity() -> None:
    observer = Observer()
    with tempfile.TemporaryDirectory() as raw:
        context = _context(Path(raw).resolve(), observer)
        result = dpo(
            context,
            DPORequest(QWEN_35_2B, _preferences(), QWEN35_DPO_SMOKE, _training()),
            runner=_backend,
        )
    assert result.technique == "dpo"
    assert result.model_artifact.metadata["dataset_id"] == "gsm8k-dpo-smoke-v1"


def test_grpo_operation_reuses_training_artifact_contract() -> None:
    observer = Observer()
    model = QWEN_35_2B
    target = _target()
    rollout_target = _target("targets/rollout-cuda-8gb")
    inference = _inference(model, target=rollout_target)
    with tempfile.TemporaryDirectory() as raw:
        context = _run_context(
            Path(raw).resolve(),
            observer,
            job_kind="train/grpo",
            run_id="runs/grpo-smoke",
        )
        result = grpo(
            context,
            GRPORequest(
                policy=model,
                bridge=FakeRLBridge(),
                settings=QWEN35_GRPO_SMOKE,
                environment=FakeEnvironment(),
                training=_training(target=target),
                inference=inference,
            ),
            runner=_backend,
        )
    assert result.technique == "grpo"
    assert result.model_artifact.metadata["dataset_id"] == "gsm8k-grpo-smoke-v1"
    assert observer.events[0].attributes["target_id"] == target.id
    assert observer.events[0].attributes["rollout_target_id"] == rollout_target.id


def test_grpo_replays_trace_evidence_after_training_without_decreasing_metric_step(
    tmp_path: Path,
) -> None:
    observer = Observer()
    model = QWEN_35_2B
    request = GRPORequest(
        policy=model,
        bridge=EvidenceReplayBridge(),
        settings=QWEN35_GRPO_SMOKE,
        environment=FakeEnvironment(),
        training=_training(),
        inference=_inference(model),
    )

    def backend(
        context: RunContext,
        value: GRPORequest,
        output_dir: Path,
    ) -> BackendTrainingResult:
        context.metrics({"train/loss": 0.25}, step=15)
        return _backend(context, value, output_dir)

    grpo(_context(tmp_path, observer), request, runner=backend)

    replay = next(batch for batch in observer.metrics_seen if "train/rl/reward_std" in batch.values)
    assert "train/rl/rollouts_completed" not in replay.values
    assert replay.step is None
    assert replay.attributes["source_step"] == 3
    assert replay.attributes["observation_source"] == "verifiers"


def test_grpo_replays_trace_population_when_verl_does_not_emit_it_live(
    tmp_path: Path,
) -> None:
    observer = Observer()
    model = QWEN_35_2B
    request = GRPORequest(
        policy=model,
        bridge=EvidenceReplayBridge(),
        settings=QWEN35_GRPO_SMOKE,
        environment=FakeEnvironment(),
        training=replace(_training(), backend="verl@0.7.0"),
        inference=_inference(model),
    )

    grpo(_context(tmp_path, observer), request, runner=_backend)

    replay = next(batch for batch in observer.metrics_seen if "train/rl/reward_std" in batch.values)
    assert replay.values["train/rl/rollouts_completed"] == 8.0
    assert replay.step is None
    assert replay.attributes["source_step"] == 3


def test_distillation_operation_records_teacher_student_and_native_trace_contract() -> None:
    observer = Observer()
    request = _distillation_request()
    with tempfile.TemporaryDirectory() as raw:
        context = _run_context(
            Path(raw).resolve(),
            observer,
            job_kind="train.distill",
            run_id="runs/distill-smoke",
        )
        result = distill(context, request, runner=_backend)

    assert result.technique == "distill"
    assert result.source_model.id == request.student.id
    assert result.model.tokenizer_fingerprint == request.student.tokenizer_fingerprint
    assert result.teacher_scoring is not None
    assert result.teacher_scoring.teacher.id == request.teacher.id
    assert result.teacher_scoring.mode == "exact-token"
    assert result.teacher_scoring.inference_binding_id == request.teacher_inference.id
    assert result.teacher_scoring.inference_binding_revision == request.teacher_inference.revision
    assert result.teacher_scoring.backend == request.teacher_inference.backend
    assert observer.events[0].attributes["teacher_model_variant_id"] == request.teacher.id
    assert any("train/distill/loss" in batch.values for batch in observer.metrics_seen)
    assert all("train/distill/teacher_failures" not in batch.values for batch in observer.metrics_seen)


def test_online_training_preserves_backend_failure_when_trace_finalization_also_fails(tmp_path: Path) -> None:
    request = GRPORequest(
        policy=QWEN_35_2B,
        bridge=FailingFinalizeBridge(),
        settings=QWEN35_GRPO_SMOKE,
        environment=FakeEnvironment(),
        training=_training(),
        inference=_inference(QWEN_35_2B),
    )

    def fail_backend(_: RunContext, __: GRPORequest, ___: Path) -> BackendTrainingResult:
        raise RuntimeError("training failed")

    with pytest.raises(RuntimeError, match="training failed") as captured:
        grpo(_context(tmp_path, Observer()), request, runner=fail_backend)

    assert captured.value.__notes__ == ["environment trace finalization also failed: RuntimeError: finalize failed"]


def test_online_training_validates_backend_outputs_before_finalizing_traces(tmp_path: Path) -> None:
    bridge = TrackingFinalizeBridge()
    request = GRPORequest(
        policy=QWEN_35_2B,
        bridge=bridge,
        settings=QWEN35_GRPO_SMOKE,
        environment=FakeEnvironment(),
        training=_training(),
        inference=_inference(QWEN_35_2B),
    )

    def invalid_backend(_: RunContext, __: GRPORequest, output_dir: Path) -> BackendTrainingResult:
        outside = output_dir.parents[3] / "outside-model"
        outside.mkdir()
        summary = output_dir.parent / "summary.json"
        summary.write_text("{}")
        return BackendTrainingResult(TrainingSummary(1, 0.5, 1.0, 1.0, 1.0), outside, None, summary)

    with pytest.raises(ValueError, match="inside the run workspace"):
        grpo(_context(tmp_path, Observer()), request, runner=invalid_backend)

    assert bridge.finalized is False


def test_distillation_request_rejects_missing_or_mismatched_tokenizer_fingerprints() -> None:
    request = _distillation_request()
    with pytest.raises(ValueError, match="immutable.*fingerprints"):
        replace(request, student=replace(request.student, tokenizer_fingerprint=None))
    with pytest.raises(ValueError, match="identical.*token-id"):
        replace(request, teacher=replace(request.teacher, tokenizer_fingerprint="e" * 64))


def test_distillation_request_rejects_wrong_teacher_purpose_and_model() -> None:
    request = _distillation_request()
    with pytest.raises(ValueError, match="teacher-score purpose"):
        replace(request, teacher_inference=replace(request.teacher_inference, purpose=("eval",)))
    with pytest.raises(ValueError, match="must select the teacher"):
        replace(request, teacher_inference=replace(request.teacher_inference, model=request.student))


def test_distillation_backend_fixes_fully_on_policy_reverse_kl_contract(tmp_path: Path) -> None:
    request = _distillation_request()
    arguments = _distillation_arguments(request, tmp_path, "http://teacher.invalid:8000")

    assert arguments["lmbda"] == 1.0
    assert arguments["beta"] == 1.0
    assert arguments["reverse_kl_top_1_mode"] == "sampled"
    assert arguments["loss_top_k"] == 1
    assert arguments["teacher_model_server_url"] == "http://teacher.invalid:8000"
    assert arguments["use_vllm"] is True
    assert arguments["vllm_weight_sync_mode"] == "lora"
    assert arguments["generation_batch_size"] == request.settings.num_prompts_per_step


def test_distillation_backend_configures_colocated_transformers_teacher(
    tmp_path: Path,
) -> None:
    request = _distillation_request()
    assert isinstance(request.teacher.artifact, HubModelRef)
    local_teacher = replace(
        request.teacher_inference,
        backend="transformers@4.57.6",
        engine={"mode": "colocate", "dtype": "bfloat16"},
    )

    arguments = _distillation_arguments(
        replace(request, teacher_inference=local_teacher),
        tmp_path,
        None,
    )

    assert arguments["use_teacher_server"] is False
    assert arguments["teacher_model_server_url"] is None
    assert arguments["teacher_model_revision"] == request.teacher.artifact.revision
    assert arguments["teacher_model_init_kwargs"] == {
        "revision": request.teacher.artifact.revision,
        "dtype": "bfloat16",
    }


def test_distillation_backend_translates_mtp_and_turboquant_rollout_options(tmp_path: Path) -> None:
    request = _distillation_request()
    rollout = _inference(
        request.student,
        max_model_len=640,
        speculative=True,
        kv_cache_dtype="turboquant_k8v4",
    )
    arguments = _distillation_arguments(
        replace(request, rollout_inference=rollout), tmp_path, "http://teacher.invalid:8000"
    )

    assert arguments["vllm_speculative_config"] == {"method": "mtp", "num_speculative_tokens": 1}
    assert arguments["vllm_engine_kwargs"] == {
        "language_model_only": True,
        "skip_mm_profiling": True,
        "enforce_eager": True,
        "kv_cache_memory_bytes": 64 * 1024 * 1024,
        "kv_cache_dtype": "turboquant_k8v4",
        "dtype": "float16",
        "disable_log_stats": False,
    }


@pytest.mark.parametrize(
    ("speculative_config", "message"),
    [
        ({"method": "draft_model", "num_speculative_tokens": 1}, "only native MTP"),
        ({"method": "mtp", "num_speculative_tokens": 0}, "positive integer"),
    ],
)
def test_trl_backend_rejects_unsupported_speculative_configuration_before_trainer_construction(
    speculative_config: dict[str, JsonValue], message: str, tmp_path: Path
) -> None:
    request = _distillation_request()
    inference = _inference(request.student)
    inference = replace(inference, engine={**inference.engine, "speculative_config": speculative_config})

    with pytest.raises(ValueError, match=message):
        _distillation_arguments(replace(request, rollout_inference=inference), tmp_path, "http://teacher.invalid")


def test_distillation_rollout_adapter_preserves_fresh_identity_masks_and_traces(monkeypatch, tmp_path: Path) -> None:
    observer = Observer()
    context = _context(tmp_path.resolve(), observer)
    request = _distillation_request()
    monkeypatch.setattr(
        "posttrain.train.backends.trl.online_rl.TrlPolicyGenerator",
        lambda *args: object(),
    )
    from posttrain.train import DistillationBatchLedger

    policy_revision = request.student.revision
    assert policy_revision is not None
    ledger = DistillationBatchLedger(policy_revision)
    rollout = _distillation_rollout_function(context, request, object(), ledger)
    trainer = SimpleNamespace(state=SimpleNamespace(global_step=3))
    inputs = [{"example_id": "gsm8k/train/0"}]

    output = rollout([[{"role": "user", "content": "What is 2 + 2?"}]], trainer, inputs=inputs)

    assert output["prompt_ids"] == [[1, 2]]
    assert output["prompt_lengths"] == [2]
    assert output["completion_ids"] == [[10, 11, 12]]
    assert output["completion_loss_mask"] == [[True, True, True]]
    assert output["logprobs"] == [[-0.1, -0.2, -0.3]]
    assert output["rollout_ids"] == ["trace-0"]
    assert observer.traces[0].attributes["technique"] == "distill"
    assert observer.events[-1].name == "distillation_batch_consumed"
    assert any(batch.values.get("train/distill/scored_tokens") == 3 for batch in observer.metrics_seen)

    with pytest.raises(ValueError, match="already been consumed"):
        rollout([[{"role": "user", "content": "What is 2 + 2?"}]], trainer, inputs=inputs)


def test_grpo_rollout_adapter_emits_population_and_throughput_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observer = Observer()
    context = _run_context(
        tmp_path.resolve(),
        observer,
        job_kind="train.grpo",
        run_id="runs/grpo-observations",
    )
    model = QWEN_35_2B
    request = GRPORequest(
        model,
        FakeRLBridge(),
        QWEN35_GRPO_SMOKE,
        FakeEnvironment(),
        _training(),
        _inference(model),
    )
    monkeypatch.setattr(
        "posttrain.train.backends.trl.online_rl.TrlPolicyGenerator",
        lambda *args: object(),
    )
    rollout = _rollout_function(context, request, object())

    output = rollout(
        [[{"role": "user", "content": "What is 2 + 2?"}]],
        SimpleNamespace(state=SimpleNamespace(global_step=3)),
        inputs=[{"example_id": "gsm8k/train/0"}],
    )

    assert output["rollout_reward"] == [1.0]
    values = observer.metrics_seen[-1].values
    assert values["train/rl/rollouts_attempted"] == 1
    assert values["train/rl/rollouts_completed"] == 1
    assert values["train/rl/rollouts_failed"] == 0
    assert values["train/rl/rollouts_truncated"] == 0
    assert values["train/rl/rollouts_unscorable"] == 0
    assert values["train/rl/time/rollout_seconds"] > 0
    assert values["train/rl/rollout_tokens_per_second"] > 0
    assert observer.metrics_seen[-1].step == 3


def test_grpo_callback_emits_normalized_names_without_trl_vocabulary(tmp_path: Path) -> None:
    observer = Observer()
    context = _run_context(
        tmp_path.resolve(),
        observer,
        job_kind="train.grpo",
        run_id="runs/grpo-normalized",
    )
    features = GRPOObservationFeatures(reference_kl_enabled=True)
    callback = callback_type(
        context,
        {"TrainerCallback": object},
        metric_normalizer=lambda step, native: (
            normalize_grpo_metrics(
                backend="trl",
                step=step,
                native=native,
                features=features,
            ).metrics
        ),
    )()

    callback.on_log(
        SimpleNamespace(max_grad_norm=1.0),
        SimpleNamespace(global_step=4),
        SimpleNamespace(),
        logs={"reward": 0.5, "reward_std": 0.25, "kl": 0.01},
    )

    values = observer.metrics_seen[-1].values
    assert values["train/rl/reward_mean"] == 0.5
    assert values["train/rl/reward_std"] == 0.25
    assert values["train/rl/kl"] == 0.01
    assert "train/reward" not in values


def test_grpo_backend_configures_one_generation_schedule_control(tmp_path: Path) -> None:
    model = QWEN_35_2B
    request = GRPORequest(
        model,
        FakeRLBridge(),
        QWEN35_GRPO_SMOKE,
        FakeEnvironment(),
        _training(),
        _inference(model),
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
        "enforce_eager": True,
        "kv_cache_memory_bytes": 64 * 1024 * 1024,
    }
    assert arguments["vllm_speculative_config"] is None

    multi_prompt_settings = replace(
        QWEN35_GRPO_SMOKE,
        loop=replace(QWEN35_GRPO_SMOKE.loop, per_device_batch_size=16),
        num_prompts_per_step=2,
        num_generations=8,
    )
    multi_prompt_request = GRPORequest(
        model,
        FakeRLBridge(),
        multi_prompt_settings,
        FakeEnvironment(),
        replace(_training(), runtime=TrainingRuntime(global_batch_size=16)),
        _inference(model),
    )
    multi_prompt_arguments = _grpo_arguments(
        multi_prompt_request,
        tmp_path,
        {"enable_thinking": False},
    )
    assert multi_prompt_arguments["generation_batch_size"] == 16

    optimized_training = replace(
        _training(),
        runtime=TrainingRuntime(global_batch_size=2),
        backend_options={
            "use_liger_kernel": True,
            "liger_loss_compiled": False,
            "logits_chunk_size": 128,
        },
    )
    optimized_request = replace(request, training=optimized_training)
    optimized_arguments = _grpo_arguments(
        optimized_request,
        tmp_path,
        {"enable_thinking": False},
    )
    assert optimized_arguments["use_liger_kernel"] is True
    assert optimized_arguments["logits_chunk_size"] == 128
    trainer = SimpleNamespace(liger_loss=SimpleNamespace(compiled=True))
    _configure_liger_loss(trainer, optimized_request)
    assert trainer.liger_loss.compiled is False
    assert _grpo_runtime_attributes(optimized_request)["liger_loss_compiled"] is False
    invalid_liger_request = replace(
        request,
        training=replace(
            _training(),
            backend_options={"use_liger_kernel": False, "liger_loss_compiled": False},
        ),
    )
    with pytest.raises(ValueError, match="requires use_liger_kernel=true"):
        _grpo_arguments(invalid_liger_request, tmp_path, {"enable_thinking": False})

    mtp_request = GRPORequest(
        model,
        FakeRLBridge(),
        QWEN35_GRPO_MTP_SMOKE,
        FakeEnvironment(),
        _training(),
        _inference(model, max_model_len=1_024, speculative=True),
    )
    mtp_arguments = _grpo_arguments(mtp_request, tmp_path, {"enable_thinking": False})
    assert mtp_arguments["vllm_speculative_config"] == {
        "method": "mtp",
        "num_speculative_tokens": 1,
    }
    assert mtp_arguments["vllm_engine_kwargs"]["disable_log_stats"] is False

    turboquant_request = replace(request, inference=_inference(model, kv_cache_dtype="turboquant_k8v4"))
    turboquant_arguments = _grpo_arguments(turboquant_request, tmp_path, {"enable_thinking": False})
    assert turboquant_arguments["vllm_engine_kwargs"]["kv_cache_dtype"] == "turboquant_k8v4"
    assert turboquant_arguments["vllm_engine_kwargs"]["dtype"] == "float16"

    dapo_request = replace(
        request,
        settings=replace(
            request.settings,
            algorithm="dapo",
            dynamic_sampling=DynamicGroupSampling(max_candidate_batches=4),
            overlong_buffer_tokens=64,
        ),
    )
    dapo_arguments = _grpo_arguments(dapo_request, tmp_path, {"enable_thinking": False})
    assert dapo_arguments["loss_type"] == "dapo"
    assert dapo_arguments["epsilon"] == 0.2
    assert dapo_arguments["epsilon_high"] == 0.28
    assert dapo_arguments["dynamic_sampling"] is True
    assert dapo_arguments["dynamic_sampling_max_batches"] == 4
    assert dapo_arguments["mask_truncated_completions"] is False


def test_grpo_runtime_event_attributes_describe_selected_acceleration_without_claiming_results() -> None:
    model = QWEN_35_2B
    request = GRPORequest(
        model,
        FakeRLBridge(),
        QWEN35_GRPO_MTP_SMOKE,
        FakeEnvironment(),
        _training(),
        _inference(
            model,
            max_model_len=1_024,
            speculative=True,
            kv_cache_dtype="turboquant_k8v4",
        ),
    )

    attributes = _grpo_runtime_attributes(request)

    assert attributes["training_backend"] == "trl"
    assert attributes["speculative_method"] == "mtp"
    assert attributes["num_speculative_tokens"] == 1
    assert attributes["kv_cache_dtype"] == "turboquant_k8v4"
    assert not any("accept" in key or "usage" in key for key in attributes)


def test_grpo_request_requires_engine_window_to_cover_declared_generation_bounds() -> None:
    model = QWEN_35_2B
    with pytest.raises(ValueError, match="model length must cover"):
        GRPORequest(
            model,
            FakeRLBridge(),
            QWEN35_GRPO_SMOKE,
            FakeEnvironment(),
            _training(),
            _inference(model, max_model_len=512),
        )


def test_grpo_settings_use_effective_batch_across_gradient_accumulation() -> None:
    settings = GRPOSettings(
        "qwen3.5/grpo-accumulation-test@1",
        TrainingLoop(max_steps=1, per_device_batch_size=1, gradient_accumulation_steps=2),
        num_prompts_per_step=1,
        num_generations=2,
    )

    assert settings.loop.per_device_batch_size == 1
    assert settings.loop.gradient_accumulation_steps == 2


def test_algorithm_settings_reject_embedded_backend_and_update_knobs() -> None:
    with pytest.raises(TypeError):
        SFTSettings(
            "qwen3.5-2b/invalid-v1",
            TrainingLoop(max_steps=1),
            qlora=QLoRAUpdate(),  # type: ignore[call-arg]
        )


def test_composed_grpo_preflight_rejects_renderer_quantization_and_topology_mismatches() -> None:
    model = QWEN_35_2B
    base = _inference(model)
    common = (model, FakeRLBridge(), QWEN35_GRPO_SMOKE, FakeEnvironment())

    with pytest.raises(ValueError, match="inference renderer"):
        GRPORequest(*common, _training(), replace(base, renderer="wrong-renderer"))

    plan = QuantizationPlan(
        "quantization/qwen-awq@1",
        "1",
        "awq",
        "awq-v1",
        "a" * 64,
        "int4-group128",
    )
    with pytest.raises(ValueError, match="quantization mode"):
        GRPORequest(
            *common,
            _training(),
            replace(base, engine={**base.engine, "quantization_plan_id": "quantization/other@1"}),
            plan,
        )

    oversized = replace(
        _training(),
        parallelism=TrainingParallelism(tensor_parallel_size=2),
    )
    with pytest.raises(ValueError, match="does not fit"):
        GRPORequest(*common, oversized, base)


def test_trl_rollout_adapter_preserves_identity_rewards_masks_and_native_traces(monkeypatch, tmp_path: Path) -> None:
    observer = Observer()
    context = _context(tmp_path.resolve(), observer)
    request = GRPORequest(
        (model := QWEN_35_2B),
        FakeRLBridge(),
        QWEN35_GRPO_SMOKE,
        FakeEnvironment(),
        _training(),
        _inference(model),
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
    assert observer.traces[0].attributes["model_variant_id"] == QWEN_35_2B.id
    assert all("reward" not in metric for batch in observer.metrics_seen for metric in batch.values)


def test_preference_contract_rejects_unordered_or_identical_pairs() -> None:
    with pytest.raises(ValueError, match="strictly greater"):
        PreferenceExample(
            "bad",
            ({"role": "user", "content": "p"},),
            ({"role": "assistant", "content": "yes"},),
            ({"role": "assistant", "content": "no"},),
            0.0,
            1.0,
        )
    with pytest.raises(ValueError, match="must differ"):
        PreferenceExample(
            "bad",
            ({"role": "user", "content": "p"},),
            ({"role": "assistant", "content": "same"},),
            ({"role": "assistant", "content": "same"},),
            1.0,
            0.0,
        )
