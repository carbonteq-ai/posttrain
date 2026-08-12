"""Strict schemas and catalog decoders for train-owned selections."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from posttrain.common import CatalogRef, ContractError, ExecutionTarget, JsonValue, StoredArtifactRef
from posttrain.common.catalog import SelectionDecoder
from posttrain.common.selections import Selection, SelectionFamily
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .bindings import (
    CalibrationSelection,
    FullParameterUpdate,
    LoRAUpdate,
    QLoRAUpdate,
    QuantizationAwareUpdate,
    QuantizationPlan,
    TrainingBinding,
    TrainingCheckpoint,
    TrainingParallelism,
    TrainingRuntime,
)
from .profiles import (
    ActiveGroupSampling,
    DPOSettings,
    DynamicGroupSampling,
    GRPOSettings,
    OnPolicyDistillationSettings,
    SAMPOSettings,
    SFTSettings,
    SFTValidationSettings,
    TrainingLoop,
    TrainingRenderer,
)


class TrainCatalogSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RendererSchema(TrainCatalogSchema):
    id: str
    model_family: str
    implementation: Literal["qwen3.5", "default"]
    reasoning_mode: str


class FullUpdateSchema(TrainCatalogSchema):
    kind: Literal["full"]


class LoRAUpdateSchema(TrainCatalogSchema):
    kind: Literal["lora"]
    rank: int = Field(default=8, gt=0)
    alpha: int = Field(default=16, gt=0)
    dropout: float = Field(default=0.0, ge=0, lt=1)
    target_modules: str = "all-linear"


class QLoRAUpdateSchema(TrainCatalogSchema):
    kind: Literal["qlora"]
    quant_type: Literal["nf4"] = "nf4"
    compute_dtype: Literal["bfloat16"] = "bfloat16"
    double_quant: bool = True
    rank: int = Field(default=8, gt=0)
    alpha: int = Field(default=16, gt=0)
    dropout: float = Field(default=0.0, ge=0, lt=1)
    target_modules: str = "all-linear"


class QATUpdateSchema(TrainCatalogSchema):
    kind: Literal["quantization-aware"]
    fake_quantization: bool = True


type UpdateSchema = Annotated[
    FullUpdateSchema | LoRAUpdateSchema | QLoRAUpdateSchema | QATUpdateSchema,
    Field(discriminator="kind"),
]


class ParallelismSchema(TrainCatalogSchema):
    tensor_parallel_size: int = Field(default=1, gt=0)
    context_parallel_size: int = Field(default=1, gt=0)
    expert_parallel_size: int = Field(default=1, gt=0)
    sequence_length_divisor: int | None = Field(default=None, gt=0)


class TrainingRuntimeSchema(TrainCatalogSchema):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    global_batch_size: int | None = Field(default=None, gt=0)
    nodes: int = Field(default=1, gt=0)
    devices_per_node: int | None = Field(default=None, gt=0)
    parameter_offload: bool = False
    optimizer_offload: bool = False
    timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class TrainingLoopSchema(TrainCatalogSchema):
    max_steps: int = Field(gt=0)
    max_length: int = Field(default=512, gt=0)
    per_device_batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    learning_rate: float = Field(default=2e-4, gt=0)
    warmup_ratio: float = Field(default=0.0, ge=0, lt=1)
    lr_scheduler_type: Literal["linear", "constant", "constant_with_warmup"] = "linear"
    max_grad_norm: float = Field(default=1.0, gt=0)
    logging_steps: int = Field(default=1, gt=0)
    checkpoint_steps: int = Field(default=1, ge=0)
    checkpoint_limit: int = Field(default=1, gt=0)
    seed: int = 42
    gradient_checkpointing: bool = True


class TrainingBindingSchema(TrainCatalogSchema):
    selection_type: Literal["training-binding"]
    id: str
    revision: str
    backend: str
    renderer: RendererSchema
    update: UpdateSchema
    target: str
    parallelism: ParallelismSchema = Field(default_factory=ParallelismSchema)
    runtime: TrainingRuntimeSchema = Field(default_factory=TrainingRuntimeSchema)
    backend_options: dict[str, JsonValue] = Field(default_factory=dict)


class TrainingCheckpointArtifactSchema(TrainCatalogSchema):
    provider: str
    namespace: str
    name: str
    version: str
    content_digest: str = Field(pattern=r"^(?:sha256:)?[0-9a-f]{64}$")


class TrainingCheckpointSchema(TrainCatalogSchema):
    selection_type: Literal["training-checkpoint"]
    id: str
    revision: str
    artifact: TrainingCheckpointArtifactSchema


class SFTValidationSettingsSchema(TrainCatalogSchema):
    steps: int = Field(gt=0)
    per_device_batch_size: int | None = Field(default=None, gt=0)
    on_start: bool = False
    at_end: bool = True


class SFTSettingsSchema(TrainCatalogSchema):
    selection_type: Literal["sft-settings"]
    id: str
    revision: str = "1"
    loop: TrainingLoopSchema
    validation: SFTValidationSettingsSchema | None = None


class DPOSettingsSchema(TrainCatalogSchema):
    selection_type: Literal["dpo-settings"]
    id: str
    revision: str = "1"
    loop: TrainingLoopSchema
    beta: float = Field(default=0.1, gt=0)
    loss_kernel: Literal["liger", "torch"] = "torch"


class DynamicGroupSamplingSchema(TrainCatalogSchema):
    max_candidate_batches: int = Field(default=10, gt=0)


class ActiveGroupSamplingSchema(TrainCatalogSchema):
    max_candidate_batches: int = Field(default=10, gt=0)


class GRPOSettingsSchema(TrainCatalogSchema):
    selection_type: Literal["grpo-settings"]
    id: str
    revision: str = "1"
    loop: TrainingLoopSchema
    num_prompts_per_step: int = Field(default=1, gt=0)
    num_generations: int = Field(default=2, ge=2)
    max_prompt_length: int = Field(default=256, gt=0)
    max_completion_length: int = Field(default=128, gt=0)
    beta: float = Field(default=0.0, ge=0)
    importance_sampling_mode: Literal["token_truncate", "token_mask", "sequence_truncate", "sequence_mask"] = (
        "sequence_truncate"
    )
    importance_sampling_clip_min: float | None = Field(default=0.1, gt=0)
    importance_sampling_clip_max: float | None = Field(default=3.0, gt=0)
    algorithm: Literal["grpo", "dapo", "olmo3"] = "grpo"
    advantage_scaling: Literal["group", "batch", "none"] = "group"
    clip_epsilon_low: float = Field(default=0.2, gt=0, allow_inf_nan=False)
    clip_epsilon_high: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    dynamic_sampling: DynamicGroupSamplingSchema | None = None
    active_sampling: ActiveGroupSamplingSchema | None = None
    mask_truncated_completions: bool = False
    overlong_buffer_tokens: int | None = Field(default=None, gt=0)
    overlong_penalty_factor: float = Field(default=1.0, gt=0, allow_inf_nan=False)


class OnPolicyDistillationSettingsSchema(TrainCatalogSchema):
    selection_type: Literal["on-policy-distillation-settings"]
    id: str
    revision: str = "1"
    loop: TrainingLoopSchema
    temperature: float = Field(default=1.0, gt=0)
    num_prompts_per_step: int = Field(default=1, gt=0)
    num_generations: int = Field(default=1, gt=0)
    max_prompt_length: int = Field(default=256, gt=0)
    max_completion_length: int = Field(default=128, gt=0)
    teacher_prompt_alignment: Literal["exact_full_sequence", "model_native_prefix_exact_completion"] = (
        "exact_full_sequence"
    )
    probability_space: Literal["raw_full_vocab", "generation_constrained"] = "raw_full_vocab"


class SAMPOSettingsSchema(TrainCatalogSchema):
    selection_type: Literal["sampo-settings"]
    id: str
    revision: str = "1"
    loop: TrainingLoopSchema
    num_prompts_per_step: int = Field(default=1, gt=0)
    num_generations: int = Field(default=2, ge=2)
    max_prompt_length: int = Field(default=256, gt=0)
    max_completion_length: int = Field(default=128, gt=0)
    beta: float = Field(default=0.0, ge=0)
    discount_gamma: float = Field(default=0.95, gt=0, le=1, allow_inf_nan=False)
    step_advantage_weight: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    advantage_normalization: Literal["mean", "mean_std"] = "mean"
    clip_epsilon_low: float = Field(default=0.003, gt=0, allow_inf_nan=False)
    clip_epsilon_high: float = Field(default=0.004, gt=0, allow_inf_nan=False)
    dynamic_sampling: DynamicGroupSamplingSchema = Field(
        default_factory=lambda: DynamicGroupSamplingSchema(max_candidate_batches=3)
    )
    mask_truncated_completions: bool = False


type TrainingSelectionSchema = Annotated[
    TrainingBindingSchema
    | TrainingCheckpointSchema
    | SFTSettingsSchema
    | DPOSettingsSchema
    | GRPOSettingsSchema
    | SAMPOSettingsSchema
    | OnPolicyDistillationSettingsSchema,
    Field(discriminator="selection_type"),
]


class CalibrationSchema(TrainCatalogSchema):
    dataset_id: str
    dataset_revision: str
    sample_count: int = Field(gt=0)
    sequence_length: int = Field(gt=0)
    batch_size: int = Field(default=1, gt=0)
    dataset_config: str | None = None
    split: str = "train"
    text_column: str = "text"


class QuantizationPlanSchema(TrainCatalogSchema):
    id: str
    revision: str
    method: Literal["awq", "rtn", "gptq", "gguf", "qat"]
    recipe: str
    recipe_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    weight_format: str
    backend: str
    dependency_lock_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    activation_format: str | None = None
    excluded_modules: tuple[str, ...] = ()
    calibration: CalibrationSchema | None = None
    output_weight_precision: str = "int4"
    output_quantization: dict[str, JsonValue] = Field(default_factory=dict)


def decode_training_selection(
    ref: CatalogRef,
    data: Mapping[str, object],
    known: Mapping[CatalogRef, Selection],
) -> Selection:
    payload = TypeAdapter(TrainingSelectionSchema).validate_python(data)
    if isinstance(payload, TrainingCheckpointSchema):
        artifact = payload.artifact
        return TrainingCheckpoint(
            id=payload.id,
            revision=payload.revision,
            artifact=StoredArtifactRef(
                provider=artifact.provider,
                namespace=artifact.namespace,
                name=artifact.name,
                version=artifact.version,
                provider_metadata={
                    "posttrain_content_digest": artifact.content_digest.removeprefix("sha256:"),
                    "posttrain_content_digest_kind": "tree",
                },
            ),
        )
    if isinstance(payload, SFTSettingsSchema):
        validation = (
            SFTValidationSettings(**payload.validation.model_dump()) if payload.validation is not None else None
        )
        return SFTSettings(
            payload.id,
            TrainingLoop(**payload.loop.model_dump()),
            payload.revision,
            validation,
        )
    if isinstance(payload, DPOSettingsSchema):
        return DPOSettings(
            payload.id,
            TrainingLoop(**payload.loop.model_dump()),
            beta=payload.beta,
            loss_kernel=payload.loss_kernel,
            revision=payload.revision,
        )
    if isinstance(payload, GRPOSettingsSchema):
        values = payload.model_dump(exclude={"selection_type", "id", "revision", "loop"})
        dynamic_sampling = values.pop("dynamic_sampling")
        if dynamic_sampling is not None:
            values["dynamic_sampling"] = DynamicGroupSampling(**dynamic_sampling)
        active_sampling = values.pop("active_sampling")
        if active_sampling is not None:
            values["active_sampling"] = ActiveGroupSampling(**active_sampling)
        return GRPOSettings(
            payload.id,
            TrainingLoop(**payload.loop.model_dump()),
            revision=payload.revision,
            **values,
        )
    if isinstance(payload, OnPolicyDistillationSettingsSchema):
        values = payload.model_dump(exclude={"selection_type", "id", "revision", "loop"})
        return OnPolicyDistillationSettings(
            payload.id,
            TrainingLoop(**payload.loop.model_dump()),
            revision=payload.revision,
            **values,
        )
    if isinstance(payload, SAMPOSettingsSchema):
        values = payload.model_dump(exclude={"selection_type", "id", "revision", "loop"})
        values["dynamic_sampling"] = DynamicGroupSampling(**values["dynamic_sampling"])
        return SAMPOSettings(
            payload.id,
            TrainingLoop(**payload.loop.model_dump()),
            revision=payload.revision,
            **values,
        )
    target_ref = CatalogRef("target", payload.target)
    target = known.get(target_ref)
    if not isinstance(target, ExecutionTarget):
        raise ContractError(f"unresolved catalog link: target/{payload.target}")
    renderer = TrainingRenderer(**payload.renderer.model_dump())
    update_data = payload.update.model_dump()
    kind = update_data.pop("kind")
    update = {
        "full": FullParameterUpdate,
        "lora": LoRAUpdate,
        "qlora": QLoRAUpdate,
        "quantization-aware": QuantizationAwareUpdate,
    }[kind](**update_data)
    return TrainingBinding(
        id=payload.id,
        revision=payload.revision,
        backend=payload.backend,
        renderer=renderer,
        update=update,
        target=target,
        parallelism=TrainingParallelism(**payload.parallelism.model_dump()),
        runtime=TrainingRuntime(**payload.runtime.model_dump()),
        backend_options=payload.backend_options,
    )


def decode_quantization_selection(
    ref: CatalogRef,
    data: Mapping[str, object],
    known: Mapping[CatalogRef, Selection],
) -> Selection:
    del ref, known
    payload = QuantizationPlanSchema.model_validate(data)
    values = payload.model_dump()
    calibration = values.pop("calibration")
    return QuantizationPlan(
        **values,
        calibration=CalibrationSelection(**calibration) if calibration is not None else None,
    )


TRAIN_CATALOG_DECODERS: Mapping[SelectionFamily, SelectionDecoder] = {
    "training": decode_training_selection,
    "quantization": decode_quantization_selection,
}

__all__ = [
    "DPOSettingsSchema",
    "GRPOSettingsSchema",
    "OnPolicyDistillationSettingsSchema",
    "QuantizationPlanSchema",
    "SAMPOSettingsSchema",
    "SFTSettingsSchema",
    "TRAIN_CATALOG_DECODERS",
    "TrainingBindingSchema",
    "TrainingCheckpointSchema",
    "TrainingRuntimeSchema",
    "TrainingLoopSchema",
    "decode_quantization_selection",
    "decode_training_selection",
]
