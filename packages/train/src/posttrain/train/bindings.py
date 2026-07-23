"""Training, parameter-update, and quantization selections."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Literal

from posttrain.common import ExecutionTarget, JsonValue, ModelVariant
from posttrain.common.selections import validate_revision, validate_selection_id

from .profiles import TrainingRenderer


@dataclass(frozen=True, slots=True)
class FullParameterUpdate:
    kind: Literal["full"] = "full"


@dataclass(frozen=True, slots=True)
class LoRAUpdate:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: str = "all-linear"
    kind: Literal["lora"] = "lora"

    def __post_init__(self) -> None:
        _validate_adapter(self.rank, self.alpha, self.dropout, self.target_modules)


@dataclass(frozen=True, slots=True)
class QLoRAUpdate:
    quant_type: Literal["nf4"] = "nf4"
    compute_dtype: Literal["bfloat16"] = "bfloat16"
    double_quant: bool = True
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: str = "all-linear"
    kind: Literal["qlora"] = "qlora"

    def __post_init__(self) -> None:
        _validate_adapter(self.rank, self.alpha, self.dropout, self.target_modules)


@dataclass(frozen=True, slots=True)
class QuantizationAwareUpdate:
    fake_quantization: bool = True
    kind: Literal["quantization-aware"] = "quantization-aware"


type ParameterUpdatePlan = FullParameterUpdate | LoRAUpdate | QLoRAUpdate | QuantizationAwareUpdate


@dataclass(frozen=True, slots=True)
class TrainingParallelism:
    tensor_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_parallel_size: int = 1
    sequence_length_divisor: int | None = None

    def __post_init__(self) -> None:
        sizes = (self.tensor_parallel_size, self.context_parallel_size, self.expert_parallel_size)
        if any(value < 1 for value in sizes):
            raise ValueError("training parallelism sizes must be positive")
        if self.sequence_length_divisor is not None and self.sequence_length_divisor < 1:
            raise ValueError("sequence length divisor must be positive")

    @property
    def required_devices(self) -> int:
        return self.tensor_parallel_size * self.context_parallel_size * self.expert_parallel_size


@dataclass(frozen=True, slots=True)
class TrainingBinding:
    id: str
    revision: str
    backend: str
    renderer: TrainingRenderer
    update: ParameterUpdatePlan
    target: ExecutionTarget
    parallelism: TrainingParallelism = field(default_factory=TrainingParallelism)
    runtime: Mapping[str, JsonValue] = field(default_factory=dict)
    backend_options: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_selection_id(self.id, "training binding id")
        validate_revision(self.revision, "training binding revision")
        if "@" not in self.backend:
            raise ValueError("training backend must include a product and version")
        object.__setattr__(self, "runtime", MappingProxyType(dict(self.runtime)))
        object.__setattr__(self, "backend_options", MappingProxyType(dict(self.backend_options)))


@dataclass(frozen=True, slots=True)
class CalibrationSelection:
    dataset_id: str
    dataset_revision: str
    sample_count: int
    sequence_length: int
    batch_size: int = 1
    dataset_config: str | None = None
    split: str = "train"
    text_column: str = "text"

    def __post_init__(self) -> None:
        validate_selection_id(self.dataset_id, "calibration dataset id")
        validate_revision(self.dataset_revision, "calibration dataset revision")
        if min(self.sample_count, self.sequence_length, self.batch_size) < 1:
            raise ValueError("calibration budget values must be positive")
        if self.dataset_config is not None and not self.dataset_config.strip():
            raise ValueError("calibration dataset config cannot be empty")
        if not self.split.strip() or not self.text_column.strip():
            raise ValueError("calibration split and text column cannot be empty")


@dataclass(frozen=True, slots=True)
class QuantizationPlan:
    id: str
    revision: str
    method: Literal["awq", "rtn", "gptq", "gguf", "qat"]
    recipe: str
    recipe_digest: str
    weight_format: str
    backend: str = "llmcompressor@unspecified"
    dependency_lock_digest: str | None = None
    activation_format: str | None = None
    excluded_modules: tuple[str, ...] = ()
    calibration: CalibrationSelection | None = None
    output_weight_precision: str = "int4"
    output_quantization: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_selection_id(self.id, "quantization plan id")
        validate_revision(self.revision, "quantization plan revision")
        if not self.recipe.strip() or re.fullmatch(r"[0-9a-f]{64}", self.recipe_digest) is None:
            raise ValueError("quantization plans require a recipe and sha256 recipe digest")
        if "@" not in self.backend:
            raise ValueError("quantization plans require a versioned backend")
        if (
            self.dependency_lock_digest is not None
            and re.fullmatch(r"[0-9a-f]{64}", self.dependency_lock_digest) is None
        ):
            raise ValueError("quantization dependency lock digest must be a sha256")
        if not self.weight_format.strip() or not self.output_weight_precision.strip():
            raise ValueError("quantization formats cannot be empty")
        if any(not item.strip() for item in self.excluded_modules):
            raise ValueError("excluded module names cannot be empty")
        object.__setattr__(self, "output_quantization", MappingProxyType(dict(self.output_quantization)))


def _validate_adapter(rank: int, alpha: int, dropout: float, target_modules: str) -> None:
    if rank < 1 or alpha < 1 or not 0 <= dropout < 1 or not target_modules.strip():
        raise ValueError("invalid adapter update plan")


def parameter_update_digest(update: ParameterUpdatePlan) -> str:
    """Stable digest for resolved-run snapshots and lineage metadata."""

    payload = json.dumps(asdict(update), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_parameter_update(model: ModelVariant, update: ParameterUpdatePlan) -> None:
    """Reject ambiguous or backend-unsafe model/update combinations before loading weights."""

    if model.form == "weight-quantized":
        raise ValueError(
            "training a persistent weight-quantized model is not supported in this slice; "
            "use QLoRA with the pinned unquantized base instead"
        )
    if model.form in {"adapter", "peft-adapter"}:
        if model.parent is None:
            raise ValueError("adapter inputs require an explicit parent model variant")
        if isinstance(update, FullParameterUpdate):
            raise ValueError("full-parameter updates cannot continue from an unmerged PEFT adapter")
        produced_with = model.provenance.get("parameter_update_kind")
        if isinstance(produced_with, str) and produced_with != update.kind:
            raise ValueError(
                f"adapter was produced with {produced_with!r} but the selected continuation uses {update.kind!r}"
            )
    elif model.form != "foundation":
        raise ValueError(f"unsupported training model form: {model.form!r}")
    if isinstance(update, QLoRAUpdate) and model.quantization:
        raise ValueError(
            "QLoRA requires an unquantized source variant; refusing to quantize an already quantized model"
        )


__all__ = [
    "CalibrationSelection",
    "FullParameterUpdate",
    "LoRAUpdate",
    "ParameterUpdatePlan",
    "QLoRAUpdate",
    "QuantizationAwareUpdate",
    "QuantizationPlan",
    "TrainingBinding",
    "TrainingParallelism",
    "parameter_update_digest",
    "validate_parameter_update",
]
