"""Public observer-neutral training operations."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from posttrain.common import (
    EventObservation,
    JsonValue,
    LocalArtifactRef,
    MetricBatchObservation,
    MetricObservation,
    ModelVariant,
    Observer,
    ProducedArtifact,
    RunContext,
    TraceFactUpdateObservation,
    TraceObservation,
)
from posttrain.data import DatasetDescriptor, PreferenceDataset, SupervisedDataset, SupervisedDataSource

from .backends.common import BackendTrainingResult
from .backends.trl import run_dpo, run_sft
from .bindings import FullParameterUpdate, LoRAUpdate, QLoRAUpdate, parameter_update_digest
from .online_rl import EnvironmentRolloutBridge, EnvironmentRolloutEvidence
from .requests import DPORequest, GRPORequest, OnPolicyDistillationRequest, SAMPORequest, SFTRequest
from .results import TeacherScoringSummary, TrainingResult

type TrainingContext = RunContext
type SFTBackend = Callable[
    [TrainingContext, SFTRequest, SupervisedDataset, SupervisedDataset | None, Path],
    BackendTrainingResult,
]
type DPOBackend = Callable[[TrainingContext, DPORequest, PreferenceDataset, Path], BackendTrainingResult]
type GRPOBackend = Callable[[TrainingContext, GRPORequest, Path], BackendTrainingResult]
type SAMPOBackend = Callable[[TrainingContext, SAMPORequest, Path], BackendTrainingResult]
type DistillationBackend = Callable[[TrainingContext, OnPolicyDistillationRequest, Path], BackendTrainingResult]

_LIVE_ROLLOUT_POPULATION_METRICS = frozenset(
    {
        "train/rl/rollouts_requested",
        "train/rl/rollouts_attempted",
        "train/rl/rollouts_completed",
        "train/rl/rollouts_failed",
        "train/rl/rollouts_truncated",
        "train/rl/rollouts_unscorable",
        "train/rl/rollouts_missing",
    }
)


@dataclass(slots=True)
class _LiveMetricObserver:
    """Forward evidence while remembering which logical step emitted each metric.

    Environment bridges replay their durable trace population when a job
    reaches a terminal state.  A trainer can already have emitted a more
    precise live version of a metric for the same optimizer step (for example,
    the retained OLMo3 population after active sampling).  Remembering that
    narrow fact lets finalization retain trace replay as a failure/fallback
    path without adding a second, semantically different point to the run.
    """

    delegate: Observer
    metric_names_by_step: dict[int, set[str]] = field(default_factory=dict)

    def _record(self, names: Iterable[str], step: int | None) -> None:
        if step is None:
            return
        self.metric_names_by_step.setdefault(step, set()).update(names)

    def snapshot(self) -> dict[int, frozenset[str]]:
        return {step: frozenset(names) for step, names in self.metric_names_by_step.items()}

    def event(self, observation: EventObservation) -> None:
        self.delegate.event(observation)

    def metric(self, observation: MetricObservation) -> None:
        self._record((observation.name,), observation.step)
        self.delegate.metric(observation)

    def metrics(self, observation: MetricBatchObservation) -> None:
        self._record(observation.values, observation.step)
        self.delegate.metrics(observation)

    def trace(self, observation: TraceObservation) -> None:
        self.delegate.trace(observation)

    def trace_fact_update(self, observation: TraceFactUpdateObservation) -> None:
        self.delegate.trace_fact_update(observation)

    def artifact(self, artifact: ProducedArtifact) -> None:
        self.delegate.artifact(artifact)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with child.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _finish(
    context: TrainingContext,
    request: SFTRequest | DPORequest | GRPORequest | SAMPORequest | OnPolicyDistillationRequest,
    technique: Literal["sft", "dpo", "grpo", "dapo", "olmo3", "sampo", "distill"],
    backend: BackendTrainingResult,
    dataset: SupervisedDataset | PreferenceDataset | None = None,
    validation_dataset: SupervisedDataset | None = None,
) -> TrainingResult:
    backend.validate(context.workspace)
    model = _source_model(request)
    resolved_dataset = (
        request.bridge.dataset
        if isinstance(request, GRPORequest | SAMPORequest | OnPolicyDistillationRequest)
        else dataset
    )
    if resolved_dataset is None:
        raise AssertionError("materialized training data is required")
    attributes = {
        "technique": technique,
        "model_variant_id": model.id,
        "source_model_form": model.form,
        "training_settings_id": request.settings.id,
        "training_settings_revision": request.settings.revision,
        "dataset_id": resolved_dataset.id,
        "dataset_revision": resolved_dataset.revision,
        **_seat_attributes(request),
    }
    if validation_dataset is not None:
        attributes.update(
            {
                "validation_dataset_id": validation_dataset.id,
                "validation_dataset_revision": validation_dataset.revision,
                "validation_dataset_examples": len(validation_dataset.examples),
            }
        )
    update = request.training.update
    is_full_update = isinstance(update, FullParameterUpdate)
    model_ref = LocalArtifactRef(backend.model_dir.resolve(), _digest(backend.model_dir))
    output_name = "weights" if is_full_update else "adapter"
    model_artifact = ProducedArtifact(
        name=f"training/{model.id}/{technique}/{update.kind}/{output_name}",
        kind="model-weights" if is_full_update else "model-adapter",
        reference=model_ref,
        metadata=attributes,
        role="model",
    )
    recovery_artifact = None
    if backend.recovery_checkpoint is not None:
        recovery_ref = LocalArtifactRef(
            backend.recovery_checkpoint.resolve(),
            _digest(backend.recovery_checkpoint),
        )
        recovery_artifact = ProducedArtifact(
            name=f"training/{model.id}/{technique}/recovery-checkpoint",
            kind="training-checkpoint",
            reference=recovery_ref,
            metadata={**attributes, "global_step": backend.summary.global_step},
            role="recovery",
        )
    summary_ref = LocalArtifactRef(backend.summary_file.resolve(), _digest(backend.summary_file))
    native_artifact = ProducedArtifact(
        name=f"training/{model.id}/{technique}/summary",
        kind="training-summary",
        reference=summary_ref,
        metadata=attributes,
        role="summary",
    )
    retention_artifact = None
    if backend.retention_manifest is not None:
        retention_ref = LocalArtifactRef(
            backend.retention_manifest.resolve(),
            _digest(backend.retention_manifest),
        )
        retention_artifact = ProducedArtifact(
            name=f"training/{model.id}/{technique}/retention-manifest",
            kind="training-retention-manifest",
            reference=retention_ref,
            metadata=attributes,
            role="retention",
        )
    context.artifact(model_artifact)
    if recovery_artifact is not None:
        context.artifact(recovery_artifact)
    context.artifact(native_artifact)
    if retention_artifact is not None:
        context.artifact(retention_artifact)
    context.metrics(
        {
            "train/global_step": backend.summary.global_step,
            "train/final_loss": backend.summary.train_loss,
            "train/runtime_seconds": backend.summary.runtime_seconds,
            "train/samples_per_second": backend.summary.samples_per_second,
            "train/steps_per_second": backend.summary.steps_per_second,
        },
        attributes=attributes,
    )
    context.event("training_completed", attributes)
    output_model = ModelVariant(
        id=f"{model.id}/{technique}-{update.kind}-{model_ref.digest[-12:]}",
        artifact=model_ref,
        form="full-finetuned" if is_full_update else "adapter",
        weight_precision=model.weight_precision,
        family=model.family,
        parameters=model.parameters,
        instruction_tuned=model.instruction_tuned,
        renderer=model.renderer,
        capabilities=model.capabilities,
        base=model.base,
        tokenizer_fingerprint=model.tokenizer_fingerprint,
        digest=model_ref.digest,
        parent=model.id,
        provenance={
            "operation": technique,
            "settings_id": request.settings.id,
            "training_binding_id": request.training.id,
            "parameter_update_kind": request.training.update.kind,
            "parameter_update_digest": parameter_update_digest(request.training.update),
            "base_model_repo_id": model.base.repo_id,
            "base_model_revision": model.base.revision,
        },
    )
    teacher_scoring = None
    if isinstance(request, OnPolicyDistillationRequest):
        teacher_scoring = TeacherScoringSummary(
            request.teacher,
            "exact-token",
            request.settings.temperature,
            1,
            request.teacher_inference.id,
            request.teacher_inference.revision,
            request.teacher_inference.backend,
        )
    return TrainingResult(
        technique,
        model,
        output_model,
        backend.summary,
        model_artifact,
        recovery_artifact,
        native_artifact,
        teacher_scoring,
    )


def sft(
    context: TrainingContext,
    request: SFTRequest,
    *,
    runner: SFTBackend = run_sft,
) -> TrainingResult:
    dataset = _load_supervised(request)
    validation_dataset = (
        _load_supervised_source(request.validation_data) if request.validation_data is not None else None
    )
    if validation_dataset is not None:
        train_ids = {example.id for example in dataset.examples}
        validation_ids = {example.id for example in validation_dataset.examples}
        overlap = train_ids & validation_ids
        if overlap:
            examples = ", ".join(sorted(overlap)[:3])
            raise ValueError(f"SFT train and validation datasets overlap: {examples}")
    attributes = {
        "technique": "sft",
        "model_variant_id": request.model.id,
        "training_settings_id": request.settings.id,
        "training_settings_revision": request.settings.revision,
        "dataset_id": dataset.id,
        "dataset_revision": dataset.revision,
        "dataset_schema_version": dataset.schema_version,
        "dataset_examples": len(dataset.examples),
        **_seat_attributes(request),
    }
    if validation_dataset is not None:
        assert request.settings.validation is not None
        attributes.update(
            {
                "validation_dataset_id": validation_dataset.id,
                "validation_dataset_revision": validation_dataset.revision,
                "validation_dataset_schema_version": validation_dataset.schema_version,
                "validation_dataset_examples": len(validation_dataset.examples),
                "validation_steps": request.settings.validation.steps,
                "validation_on_start": request.settings.validation.on_start,
                "validation_at_end": request.settings.validation.at_end,
            }
        )
    context.event("training_started", attributes)
    output_dir = context.workspace / "training" / "sft" / "trainer"
    output_dir.mkdir(parents=True, exist_ok=False)
    return _finish(
        context,
        request,
        "sft",
        runner(context, request, dataset, validation_dataset, output_dir),
        dataset,
        validation_dataset,
    )


def dpo(
    context: TrainingContext,
    request: DPORequest,
    *,
    runner: DPOBackend = run_dpo,
) -> TrainingResult:
    dataset = _load_preferences(request)
    attributes = {
        "technique": "dpo",
        "model_variant_id": request.model.id,
        "training_settings_id": request.settings.id,
        "training_settings_revision": request.settings.revision,
        "dataset_id": dataset.id,
        "dataset_revision": dataset.revision,
        "dataset_schema_version": dataset.schema_version,
        "dataset_examples": len(dataset.examples),
        **_seat_attributes(request),
    }
    context.event("training_started", attributes)
    output_dir = context.workspace / "training" / "dpo" / "trainer"
    output_dir.mkdir(parents=True, exist_ok=False)
    return _finish(context, request, "dpo", runner(context, request, dataset, output_dir), dataset)


def grpo(
    context: TrainingContext,
    request: GRPORequest,
    *,
    runner: GRPOBackend | None = None,
) -> TrainingResult:
    selected_runner = runner or _grpo_backend(request.training.backend)
    dataset = request.bridge.dataset
    attributes = {
        "technique": request.settings.algorithm,
        "model_variant_id": request.policy.id,
        "training_settings_id": request.settings.id,
        "training_settings_revision": request.settings.revision,
        "dataset_id": dataset.id,
        **_seat_attributes(request),
    }
    context.event("training_started", attributes)
    output_dir = context.workspace / "training" / request.settings.algorithm / "trainer"
    output_dir.mkdir(parents=True, exist_ok=False)
    backend = _run_environment_backend(
        context,
        request.bridge,
        lambda active_context: selected_runner(active_context, request, output_dir),
        replay_exclusions=_rollout_replay_exclusions(request.training.backend),
    )
    return _finish(context, request, request.settings.algorithm, backend)


def sampo(
    context: TrainingContext,
    request: SAMPORequest,
    *,
    runner: SAMPOBackend | None = None,
) -> TrainingResult:
    selected_runner = runner or _sampo_backend(request.training.backend)
    dataset = request.bridge.dataset
    attributes = {
        "technique": "sampo",
        "model_variant_id": request.policy.id,
        "training_settings_id": request.settings.id,
        "training_settings_revision": request.settings.revision,
        "dataset_id": dataset.id,
        **_seat_attributes(request),
    }
    context.event("training_started", attributes)
    output_dir = context.workspace / "training" / "sampo" / "trainer"
    output_dir.mkdir(parents=True, exist_ok=False)
    backend = _run_environment_backend(
        context,
        request.bridge,
        lambda active_context: selected_runner(active_context, request, output_dir),
        replay_exclusions=_rollout_replay_exclusions(request.training.backend),
    )
    return _finish(context, request, "sampo", backend)


def _rollout_replay_exclusions(training_backend: str) -> frozenset[str]:
    """Avoid duplicate population metrics only when the backend emits them live."""

    product = training_backend.split("@", 1)[0]
    return _LIVE_ROLLOUT_POPULATION_METRICS if product == "trl" else frozenset()


def distill(
    context: TrainingContext,
    request: OnPolicyDistillationRequest,
    *,
    runner: DistillationBackend | None = None,
) -> TrainingResult:
    selected_runner = runner or _distillation_backend(request.training.backend)
    dataset = request.bridge.dataset
    attributes = {
        "technique": "distill",
        "student_model_variant_id": request.student.id,
        "teacher_model_variant_id": request.teacher.id,
        "training_settings_id": request.settings.id,
        "training_settings_revision": request.settings.revision,
        "dataset_id": dataset.id,
        **_seat_attributes(request),
    }
    context.event("training_started", attributes)
    output_dir = context.workspace / "training" / "distill" / "trainer"
    output_dir.mkdir(parents=True, exist_ok=False)
    backend = _run_environment_backend(
        context,
        request.bridge,
        lambda active_context: selected_runner(active_context, request, output_dir),
    )
    context.metrics(
        {
            "train/distill/loss": backend.summary.train_loss,
            "train/distill/reverse_kl": backend.summary.train_loss,
        },
        attributes=attributes,
    )
    return _finish(context, request, "distill", backend)


def _run_environment_backend(
    context: TrainingContext,
    bridge: EnvironmentRolloutBridge,
    run: Callable[[TrainingContext], BackendTrainingResult],
    *,
    replay_exclusions: frozenset[str] = frozenset(),
) -> BackendTrainingResult:
    live_observer = _LiveMetricObserver(context.observer)
    live_context = replace(context, observer=live_observer)
    try:
        result = run(live_context)
    except BaseException as training_error:
        try:
            # A failed backend may have emitted only a partial live population.
            # Replay all terminal evidence so failed/unscorable counts cannot be
            # hidden by the normal success-path de-duplication policy.
            _publish_bridge_artifacts(
                live_context,
                bridge,
                replay_exclusions=frozenset(),
            )
        except BaseException as finalization_error:
            training_error.add_note(
                f"environment trace finalization also failed: {type(finalization_error).__name__}: {finalization_error}"
            )
        raise
    result.validate(live_context.workspace)
    _publish_bridge_artifacts(
        live_context,
        bridge,
        replay_exclusions=replay_exclusions,
        live_metric_names=live_observer.snapshot(),
    )
    return result


def _publish_bridge_artifacts(
    context: TrainingContext,
    bridge: EnvironmentRolloutBridge,
    *,
    replay_exclusions: frozenset[str] = frozenset(),
    live_metric_names: dict[int, frozenset[str]] | None = None,
) -> None:
    evidence_reader = getattr(bridge, "evidence", None)
    if callable(evidence_reader):
        evidence = evidence_reader()
        if not isinstance(evidence, EnvironmentRolloutEvidence):
            raise TypeError("environment bridge evidence must use EnvironmentRolloutEvidence")
        for observation in evidence.metrics:
            observed_at_step = (
                live_metric_names.get(observation.step, frozenset())
                if live_metric_names is not None and observation.step is not None
                else frozenset()
            )
            values = {
                name: value
                for name, value in observation.values.items()
                if name not in replay_exclusions and name not in observed_at_step
            }
            if not values:
                continue
            attributes = dict(observation.attributes)
            if observation.step is not None:
                attributes["source_step"] = observation.step
            context.metrics(
                values,
                # Evidence is replayed after the backend has emitted its live
                # training steps. Keep the source step as metadata and let the
                # tracking provider append it at the current position.
                step=None,
                attributes=attributes,
            )
        for observation in evidence.traces:
            context.trace(observation)
    for artifact in bridge.finalize():
        context.artifact(artifact)


def _grpo_backend(backend: str) -> GRPOBackend:
    product = backend.split("@", 1)[0]
    if product == "trl":
        from .backends.trl import run_grpo

        return run_grpo
    if product == "verl":
        from .backends.verl import run_grpo

        return run_grpo
    raise ValueError(f"unsupported GRPO training backend {backend!r}")


def _distillation_backend(backend: str) -> DistillationBackend:
    product = backend.split("@", 1)[0]
    if product == "trl":
        from .backends.trl import run_distillation

        return run_distillation
    if product == "verl":
        from .backends.verl import run_distillation

        return run_distillation
    raise ValueError(f"unsupported distillation training backend {backend!r}")


def _sampo_backend(backend: str) -> SAMPOBackend:
    product = backend.split("@", 1)[0]
    if product == "trl":
        from .backends.trl import run_sampo

        return run_sampo
    if product == "verl":
        from .backends.verl import run_sampo

        return run_sampo
    raise ValueError(f"unsupported SAMPO training backend {backend!r}")


def _validate_materialization(expected: DatasetDescriptor, actual: DatasetDescriptor) -> None:
    if actual != expected:
        raise ValueError(f"data source materialized {actual!r}, expected its declared descriptor {expected!r}")


def _source_model(
    request: SFTRequest | DPORequest | GRPORequest | SAMPORequest | OnPolicyDistillationRequest,
) -> ModelVariant:
    if isinstance(request, GRPORequest | SAMPORequest):
        return request.policy
    if isinstance(request, OnPolicyDistillationRequest):
        return request.student
    return request.model


def _seat_attributes(
    request: SFTRequest | DPORequest | GRPORequest | SAMPORequest | OnPolicyDistillationRequest,
) -> dict[str, JsonValue]:
    update = request.training.update
    attributes: dict[str, JsonValue] = {
        "training_binding_id": request.training.id,
        "training_binding_revision": request.training.revision,
        "target_id": request.training.target.id,
        "target_revision": request.training.target.revision,
        "parameter_update_kind": request.training.update.kind,
        "parameter_update_digest": parameter_update_digest(request.training.update),
        "base_model_repo_id": _source_model(request).base.repo_id,
        "base_model_revision": _source_model(request).base.revision,
    }
    source_revision = request.training.backend_options.get("source_revision")
    if isinstance(source_revision, str):
        attributes["training_backend_source_revision"] = source_revision
    lock_digest = request.training.backend_options.get("dependency_lock_sha256")
    if isinstance(lock_digest, str):
        attributes["dependency_lock_sha256"] = lock_digest
    if isinstance(update, LoRAUpdate | QLoRAUpdate):
        attributes.update(
            {
                "peft_rank": update.rank,
                "peft_alpha": update.alpha,
                "peft_dropout": update.dropout,
                "peft_target_modules": update.target_modules,
            }
        )
    if isinstance(update, QLoRAUpdate):
        attributes.update(
            {
                "qlora_quant_type": update.quant_type,
                "qlora_compute_dtype": update.compute_dtype,
                "qlora_double_quant": update.double_quant,
                "qlora_storage_bits": 4,
            }
        )
    if isinstance(request, GRPORequest):
        attributes["online_rl_algorithm"] = request.settings.algorithm
        attributes["advantage_scaling"] = request.settings.advantage_scaling
        attributes["clip_epsilon_low"] = request.settings.clip_epsilon_low
        attributes["clip_epsilon_high"] = request.settings.resolved_clip_epsilon_high
        attributes["mask_truncated_completions"] = request.settings.mask_truncated_completions
        attributes["shuffle_prompts"] = request.settings.shuffle_prompts
        attributes["active_sampling"] = request.settings.active_sampling is not None
        if request.settings.active_sampling is not None:
            attributes["active_sampling_max_candidate_batches"] = request.settings.active_sampling.max_candidate_batches
        attributes["overlong_penalty_factor"] = request.settings.overlong_penalty_factor
        if request.settings.overlong_buffer_tokens is not None:
            attributes["overlong_buffer_tokens"] = request.settings.overlong_buffer_tokens
        attributes["environment_id"] = request.environment.id
        attributes["environment_revision"] = request.environment.revision
        attributes["inference_id"] = request.inference.id
        attributes["inference_revision"] = request.inference.revision
        attributes["rollout_target_id"] = request.inference.target.id
        if request.quantization is not None:
            attributes["quantization_plan_id"] = request.quantization.id
            attributes["quantization_recipe_digest"] = request.quantization.recipe_digest
    if isinstance(request, SAMPORequest):
        attributes.update(
            {
                "online_rl_algorithm": "sampo",
                "clip_epsilon_low": request.settings.clip_epsilon_low,
                "clip_epsilon_high": request.settings.clip_epsilon_high,
                "discount_gamma": request.settings.discount_gamma,
                "step_advantage_weight": request.settings.step_advantage_weight,
                "advantage_normalization": request.settings.advantage_normalization,
                "mask_truncated_completions": request.settings.mask_truncated_completions,
                "dynamic_sampling_max_candidate_batches": (request.settings.dynamic_sampling.max_candidate_batches),
                "environment_id": request.environment.id,
                "environment_revision": request.environment.revision,
                "inference_id": request.inference.id,
                "inference_revision": request.inference.revision,
                "rollout_target_id": request.inference.target.id,
            }
        )
        if request.quantization is not None:
            attributes["quantization_plan_id"] = request.quantization.id
            attributes["quantization_recipe_digest"] = request.quantization.recipe_digest
    if isinstance(request, OnPolicyDistillationRequest):
        attributes.update(
            {
                "environment_id": request.environment.id,
                "environment_revision": request.environment.revision,
                "student_model_variant_id": request.student.id,
                "teacher_model_variant_id": request.teacher.id,
                "rollout_inference_id": request.rollout_inference.id,
                "rollout_target_id": request.rollout_inference.target.id,
                "teacher_inference_id": request.teacher_inference.id,
                "teacher_target_id": request.teacher_inference.target.id,
                "tokenizer_fingerprint": request.student.tokenizer_fingerprint or "",
            }
        )
        if request.quantization is not None:
            attributes["quantization_plan_id"] = request.quantization.id
            attributes["quantization_recipe_digest"] = request.quantization.recipe_digest
    return attributes


def _load_supervised(request: SFTRequest) -> SupervisedDataset:
    return _load_supervised_source(request.data)


def _load_supervised_source(source: SupervisedDataSource) -> SupervisedDataset:
    dataset = source.load()
    _validate_materialization(source.descriptor, dataset.descriptor)
    return dataset


def _load_preferences(request: DPORequest) -> PreferenceDataset:
    dataset = request.data.load()
    _validate_materialization(request.data.descriptor, dataset.descriptor)
    return dataset


__all__ = [
    "DPOBackend",
    "DistillationBackend",
    "GRPOBackend",
    "SFTBackend",
    "TrainingContext",
    "distill",
    "dpo",
    "grpo",
    "sft",
]
