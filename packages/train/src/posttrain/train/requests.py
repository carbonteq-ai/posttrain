"""Typed public training operation requests and composed preflight validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from posttrain.common import InferenceBinding, LocalArtifactRef, ModelVariant
from posttrain.data import PreferenceDataSource, SupervisedDataSource

from .bindings import QuantizationAwareUpdate, QuantizationPlan, TrainingBinding, validate_parameter_update
from .online_rl import EnvironmentRolloutBridge
from .profiles import DPOSettings, GRPOSettings, OnPolicyDistillationSettings, SAMPOSettings, SFTSettings, TrainingLoop


class EnvironmentSelection(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def revision(self) -> str: ...


@dataclass(frozen=True, slots=True)
class SFTRequest:
    model: ModelVariant
    data: SupervisedDataSource
    settings: SFTSettings
    training: TrainingBinding
    resume_from: LocalArtifactRef | None = None
    validation_data: SupervisedDataSource | None = None

    def __post_init__(self) -> None:
        _validate_training(self.model, self.settings.loop, self.training, None)
        has_schedule = self.settings.validation is not None
        has_data = self.validation_data is not None
        if has_schedule != has_data:
            raise ValueError("SFT validation data and validation settings must be selected together")


@dataclass(frozen=True, slots=True)
class DPORequest:
    model: ModelVariant
    data: PreferenceDataSource
    settings: DPOSettings
    training: TrainingBinding
    resume_from: LocalArtifactRef | None = None

    def __post_init__(self) -> None:
        _validate_training(self.model, self.settings.loop, self.training, None)


@dataclass(frozen=True, slots=True)
class GRPORequest:
    policy: ModelVariant
    bridge: EnvironmentRolloutBridge
    settings: GRPOSettings
    environment: EnvironmentSelection
    training: TrainingBinding
    inference: InferenceBinding
    quantization: QuantizationPlan | None = None
    reference: ModelVariant | None = None
    resume_from: LocalArtifactRef | None = None

    def __post_init__(self) -> None:
        _validate_online_rl(
            "GRPO",
            self.policy,
            self.bridge,
            self.settings,
            self.training,
            self.inference,
            self.quantization,
        )


@dataclass(frozen=True, slots=True)
class SAMPORequest:
    policy: ModelVariant
    bridge: EnvironmentRolloutBridge
    settings: SAMPOSettings
    environment: EnvironmentSelection
    training: TrainingBinding
    inference: InferenceBinding
    quantization: QuantizationPlan | None = None
    reference: ModelVariant | None = None
    resume_from: LocalArtifactRef | None = None

    def __post_init__(self) -> None:
        _validate_online_rl(
            "SAMPO",
            self.policy,
            self.bridge,
            self.settings,
            self.training,
            self.inference,
            self.quantization,
        )


def _validate_online_rl(
    technique: str,
    policy: ModelVariant,
    bridge: EnvironmentRolloutBridge,
    settings: GRPOSettings | SAMPOSettings,
    training: TrainingBinding,
    inference: InferenceBinding,
    quantization: QuantizationPlan | None,
) -> None:
    if not bridge.dataset.examples:
        raise ValueError(f"{technique} requires at least one rollout example")
    if inference.model.id != policy.id:
        raise ValueError(f"{technique} inference binding must select the policy model variant")
    if inference.renderer != policy.renderer_contract:
        raise ValueError(f"{technique} inference renderer is incompatible with the policy model")
    if "rollout" not in inference.purpose:
        raise ValueError(f"{technique} inference binding must declare the rollout purpose")
    _validate_training(policy, settings.loop, training, quantization)
    sequence_length = settings.max_prompt_length + settings.max_completion_length
    _validate_sequence_length(sequence_length, training)
    engine_limit = inference.engine.get("max_model_len")
    if isinstance(engine_limit, int) and sequence_length > engine_limit:
        raise ValueError("rollout model length must cover prompt and completion limits")
    expected_batch = settings.num_prompts_per_step * settings.num_generations
    global_batch = training.runtime.global_batch_size
    if isinstance(global_batch, int) and global_batch != expected_batch:
        raise ValueError("training global batch must equal prompt groups times generations")
    plan_id = inference.engine.get("quantization_plan_id")
    if plan_id is not None and (quantization is None or plan_id != quantization.id):
        raise ValueError("rollout quantization mode must reference the selected quantization plan")


@dataclass(frozen=True, slots=True)
class OnPolicyDistillationRequest:
    student: ModelVariant
    teacher: ModelVariant
    bridge: EnvironmentRolloutBridge
    settings: OnPolicyDistillationSettings
    environment: EnvironmentSelection
    training: TrainingBinding
    rollout_inference: InferenceBinding
    teacher_inference: InferenceBinding
    quantization: QuantizationPlan | None = None
    resume_from: LocalArtifactRef | None = None

    def __post_init__(self) -> None:
        if not self.bridge.dataset.examples:
            raise ValueError("on-policy distillation requires at least one rollout example")
        if self.student.id == self.teacher.id:
            raise ValueError("on-policy distillation requires distinct student and teacher variants")
        if self.rollout_inference.model.id != self.student.id:
            raise ValueError("distillation rollout inference must select the student model variant")
        if self.teacher_inference.model.id != self.teacher.id:
            raise ValueError("distillation teacher inference must select the teacher model variant")
        if self.rollout_inference.renderer != self.student.renderer_contract:
            raise ValueError("distillation rollout renderer is incompatible with the student model")
        if self.teacher_inference.renderer != self.teacher.renderer_contract:
            raise ValueError("distillation scoring renderer is incompatible with the teacher model")
        if "rollout" not in self.rollout_inference.purpose:
            raise ValueError("distillation rollout inference must declare the rollout purpose")
        if "teacher-score" not in self.teacher_inference.purpose:
            raise ValueError("distillation teacher inference must declare the teacher-score purpose")
        student_fingerprint = self.student.tokenizer_fingerprint
        teacher_fingerprint = self.teacher.tokenizer_fingerprint
        if student_fingerprint is None or teacher_fingerprint is None:
            raise ValueError("distillation requires immutable student and teacher tokenizer fingerprints")
        if student_fingerprint != teacher_fingerprint:
            raise ValueError("distillation requires identical student and teacher token-id mappings")
        _validate_training(self.student, self.settings.loop, self.training, self.quantization)
        sequence_length = self.settings.max_prompt_length + self.settings.max_completion_length
        _validate_sequence_length(sequence_length, self.training)
        for role, binding in (
            ("rollout", self.rollout_inference),
            ("teacher-score", self.teacher_inference),
        ):
            engine_limit = binding.engine.get("max_model_len")
            if isinstance(engine_limit, int) and sequence_length > engine_limit:
                raise ValueError(f"{role} model length must cover distillation prompt and completion limits")
        expected_batch = self.settings.num_prompts_per_step * self.settings.num_generations
        global_batch = self.training.runtime.global_batch_size
        if isinstance(global_batch, int) and global_batch != expected_batch:
            raise ValueError("training global batch must equal distillation prompts times generations")
        plan_id = self.rollout_inference.engine.get("quantization_plan_id")
        if plan_id is not None and (self.quantization is None or plan_id != self.quantization.id):
            raise ValueError("distillation rollout quantization must reference the selected quantization plan")


def _validate_training(
    model: ModelVariant,
    loop: TrainingLoop,
    training: TrainingBinding,
    quantization: QuantizationPlan | None,
) -> None:
    validate_parameter_update(model, training.update)
    if model.family != training.renderer.model_family:
        raise ValueError("training renderer is incompatible with the model family")
    model.conversation.reasoning_mode(training.renderer.reasoning_mode)
    if isinstance(training.update, QuantizationAwareUpdate) and quantization is None:
        raise ValueError("quantization-aware updates require a quantization plan")
    _validate_sequence_length(loop.max_length, training)
    available = training.target.placement.get("world_size", 1)
    if isinstance(available, int) and training.parallelism.required_devices > available:
        raise ValueError("training parallelism does not fit the selected execution target")


def _validate_sequence_length(length: int, training: TrainingBinding) -> None:
    divisor = training.parallelism.sequence_length_divisor
    if divisor is not None and length % divisor:
        raise ValueError("sequence length is not divisible by the training parallelism requirement")


__all__ = [
    "DPORequest",
    "EnvironmentSelection",
    "GRPORequest",
    "OnPolicyDistillationRequest",
    "SAMPORequest",
    "SFTRequest",
]
