"""Typed, reusable training defaults with no framework imports."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

_ID = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")


@dataclass(frozen=True, slots=True)
class RendererProfile:
    id: str
    model_family: str
    implementation: Literal["qwen3.5", "default"]
    reasoning_mode: str

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.id) or not self.model_family or not self.reasoning_mode:
            raise ValueError("renderer profile identity, family, and reasoning mode are required")


@dataclass(frozen=True, slots=True)
class QLoRAProfile:
    quant_type: Literal["nf4"] = "nf4"
    compute_dtype: Literal["bfloat16"] = "bfloat16"
    double_quant: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: Literal["all-linear"] = "all-linear"

    def __post_init__(self) -> None:
        if self.lora_rank < 1 or self.lora_alpha < 1 or not 0 <= self.lora_dropout < 1:
            raise ValueError("invalid QLoRA adapter profile")


@dataclass(frozen=True, slots=True)
class TrainingLoop:
    max_steps: int
    max_length: int = 512
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.0
    max_grad_norm: float = 1.0
    logging_steps: int = 1
    checkpoint_steps: int = 1
    checkpoint_limit: int = 1
    seed: int = 42
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        counts = (
            self.max_steps,
            self.max_length,
            self.per_device_batch_size,
            self.gradient_accumulation_steps,
            self.logging_steps,
            self.checkpoint_steps,
            self.checkpoint_limit,
        )
        if any(value < 1 for value in counts):
            raise ValueError("training loop counts must be positive")
        if self.learning_rate <= 0 or not 0 <= self.warmup_ratio < 1 or self.max_grad_norm <= 0:
            raise ValueError("invalid training optimization values")


@dataclass(frozen=True, slots=True)
class SFTProfile:
    id: str
    model_family: str
    renderer: RendererProfile
    loop: TrainingLoop
    qlora: QLoRAProfile = field(default_factory=QLoRAProfile)

    def __post_init__(self) -> None:
        _validate_profile(self.id, self.model_family, self.renderer)


@dataclass(frozen=True, slots=True)
class DPOProfile:
    id: str
    model_family: str
    renderer: RendererProfile
    loop: TrainingLoop
    beta: float = 0.1
    qlora: QLoRAProfile = field(default_factory=QLoRAProfile)

    def __post_init__(self) -> None:
        _validate_profile(self.id, self.model_family, self.renderer)
        if self.beta <= 0:
            raise ValueError("DPO beta must be positive")


def _validate_profile(identifier: str, family: str, renderer: RendererProfile) -> None:
    if not _ID.fullmatch(identifier) or not family:
        raise ValueError("training profile identity and family are required")
    if renderer.model_family != family:
        raise ValueError("renderer and training profile model families must match")


QWEN35_RENDERER = RendererProfile("qwen3.5-off-v1", "qwen3.5", "qwen3.5", "off")
LFM25_RENDERER = RendererProfile(
    "lfm2.5-native-v1",
    "lfm2.5",
    "default",
    "native",
)

QWEN35_SFT_SMOKE = SFTProfile(
    "qwen3.5-2b/sft-qlora-smoke-v1",
    "qwen3.5",
    QWEN35_RENDERER,
    TrainingLoop(max_steps=2),
)
QWEN35_DPO_SMOKE = DPOProfile(
    "qwen3.5-2b/dpo-qlora-smoke-v1",
    "qwen3.5",
    QWEN35_RENDERER,
    TrainingLoop(max_steps=2, learning_rate=1e-4),
)
LFM25_SFT_SMOKE = SFTProfile(
    "lfm2.5-1.2b/sft-qlora-smoke-v1",
    "lfm2.5",
    LFM25_RENDERER,
    TrainingLoop(max_steps=2),
)
LFM25_DPO_SMOKE = DPOProfile(
    "lfm2.5-1.2b/dpo-qlora-smoke-v1",
    "lfm2.5",
    LFM25_RENDERER,
    TrainingLoop(max_steps=2, learning_rate=1e-4),
)

__all__ = [
    "DPOProfile",
    "LFM25_DPO_SMOKE",
    "LFM25_RENDERER",
    "LFM25_SFT_SMOKE",
    "QLoRAProfile",
    "QWEN35_DPO_SMOKE",
    "QWEN35_RENDERER",
    "QWEN35_SFT_SMOKE",
    "RendererProfile",
    "SFTProfile",
    "TrainingLoop",
]
