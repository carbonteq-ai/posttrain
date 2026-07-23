"""Strict schemas and catalog decoders for train-owned selections."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from posttrain.common import CatalogRef, ContractError, ExecutionTarget, JsonValue
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
    TrainingParallelism,
)
from .profiles import (
    DPOSettings,
    GRPOSettings,
    OnPolicyDistillationSettings,
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


class TrainingLoopSchema(TrainCatalogSchema):
    max_steps: int = Field(gt=0)
    max_length: int = Field(default=512, gt=0)
    per_device_batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    learning_rate: float = Field(default=2e-4, gt=0)
    warmup_ratio: float = Field(default=0.0, ge=0, lt=1)
    max_grad_norm: float = Field(default=1.0, gt=0)
    logging_steps: int = Field(default=1, gt=0)
    checkpoint_steps: int = Field(default=1, gt=0)
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
    runtime: dict[str, JsonValue] = Field(default_factory=dict)
    backend_options: dict[str, JsonValue] = Field(default_factory=dict)


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


type TrainingSelectionSchema = Annotated[
    TrainingBindingSchema
    | SFTSettingsSchema
    | DPOSettingsSchema
    | GRPOSettingsSchema
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
        runtime=payload.runtime,
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
    "SFTSettingsSchema",
    "TRAIN_CATALOG_DECODERS",
    "TrainingBindingSchema",
    "TrainingLoopSchema",
    "decode_quantization_selection",
    "decode_training_selection",
]
