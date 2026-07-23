"""Algorithm settings: technique and optimization schedule only."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

_ID = re.compile(r"^[a-z0-9][a-z0-9._/@:-]*$")


@dataclass(frozen=True, slots=True)
class TrainingRenderer:
    """Renderer selection used by a training binding, not an algorithm."""

    id: str
    model_family: str
    implementation: Literal["qwen3.5", "default"]
    reasoning_mode: str

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.id) or not self.model_family or not self.reasoning_mode:
            raise ValueError("training renderer identity, family, and reasoning mode are required")


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
class SFTValidationSettings:
    """Bounded teacher-forced validation performed by the SFT trainer."""

    steps: int
    per_device_batch_size: int | None = None
    on_start: bool = False
    at_end: bool = True

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("SFT validation steps must be positive")
        if self.per_device_batch_size is not None and self.per_device_batch_size < 1:
            raise ValueError("SFT validation batch size must be positive")


@dataclass(frozen=True, slots=True)
class SFTSettings:
    id: str
    loop: TrainingLoop
    revision: str = "1"
    validation: SFTValidationSettings | None = None

    def __post_init__(self) -> None:
        _validate_settings(self.id, self.revision)


@dataclass(frozen=True, slots=True)
class DPOSettings:
    id: str
    loop: TrainingLoop
    beta: float = 0.1
    loss_kernel: Literal["liger", "torch"] = "torch"
    revision: str = "1"

    def __post_init__(self) -> None:
        _validate_settings(self.id, self.revision)
        if self.beta <= 0:
            raise ValueError("DPO beta must be positive")


@dataclass(frozen=True, slots=True)
class GRPOSettings:
    id: str
    loop: TrainingLoop
    num_prompts_per_step: int = 1
    num_generations: int = 2
    max_prompt_length: int = 256
    max_completion_length: int = 128
    beta: float = 0.0
    importance_sampling_mode: Literal["token_truncate", "token_mask", "sequence_truncate", "sequence_mask"] = (
        "sequence_truncate"
    )
    importance_sampling_clip_min: float | None = 0.1
    importance_sampling_clip_max: float | None = 3.0
    revision: str = "1"

    def __post_init__(self) -> None:
        _validate_settings(self.id, self.revision)
        if self.num_prompts_per_step < 1 or self.num_generations < 2:
            raise ValueError("GRPO requires positive prompt groups and at least two generations")
        expected_batch = self.num_prompts_per_step * self.num_generations
        effective_batch = self.loop.per_device_batch_size * self.loop.gradient_accumulation_steps
        if effective_batch != expected_batch:
            raise ValueError("GRPO effective batch must equal prompts per step times generations")
        if self.max_prompt_length < 1 or self.max_completion_length < 1 or self.beta < 0:
            raise ValueError("invalid GRPO length or KL settings")
        bounds = (self.importance_sampling_clip_min, self.importance_sampling_clip_max)
        if any(value is not None and value <= 0 for value in bounds):
            raise ValueError("importance-sampling bounds must be positive")
        if (
            self.importance_sampling_clip_min is not None
            and self.importance_sampling_clip_max is not None
            and self.importance_sampling_clip_min >= self.importance_sampling_clip_max
        ):
            raise ValueError("importance-sampling minimum must be smaller than maximum")


@dataclass(frozen=True, slots=True)
class OnPolicyDistillationSettings:
    """Fully on-policy sampled-token reverse-KL settings."""

    id: str
    loop: TrainingLoop
    temperature: float = 1.0
    num_prompts_per_step: int = 1
    num_generations: int = 1
    max_prompt_length: int = 256
    max_completion_length: int = 128
    revision: str = "1"

    def __post_init__(self) -> None:
        _validate_settings(self.id, self.revision)
        counts = (
            self.num_prompts_per_step,
            self.num_generations,
            self.max_prompt_length,
            self.max_completion_length,
        )
        if any(value < 1 for value in counts) or self.temperature <= 0:
            raise ValueError("on-policy distillation counts and temperature must be positive")
        expected_batch = self.num_prompts_per_step * self.num_generations
        effective_batch = self.loop.per_device_batch_size * self.loop.gradient_accumulation_steps
        if effective_batch != expected_batch:
            raise ValueError("distillation effective batch must equal prompts per step times generations")
        if self.max_prompt_length + self.max_completion_length > self.loop.max_length:
            raise ValueError("distillation loop max_length must cover prompt and completion limits")


def _validate_settings(identifier: str, revision: str) -> None:
    if not _ID.fullmatch(identifier) or not revision:
        raise ValueError("algorithm settings require a stable identity and revision")


QWEN35_RENDERER = TrainingRenderer("qwen3.5-off-v1", "qwen3.5", "qwen3.5", "off")
QWEN35_THINKING_RENDERER = TrainingRenderer("qwen3.5-thinking-v1", "qwen3.5", "qwen3.5", "thinking")
LFM25_RENDERER = TrainingRenderer("lfm2.5-native-v1", "lfm2.5", "default", "native")

QWEN35_SFT_SMOKE = SFTSettings("qwen3.5-2b/sft-smoke-v2", TrainingLoop(max_steps=2))
QWEN35_DPO_SMOKE = DPOSettings(
    "qwen3.5-2b/dpo-smoke-v2",
    TrainingLoop(max_steps=2, max_length=448, learning_rate=1e-4),
)
LFM25_SFT_SMOKE = SFTSettings("lfm2.5-1.2b/sft-smoke-v2", TrainingLoop(max_steps=2))
LFM25_DPO_SMOKE = DPOSettings(
    "lfm2.5-1.2b/dpo-smoke-v2",
    TrainingLoop(max_steps=2, learning_rate=1e-4),
    loss_kernel="liger",
)
QWEN35_GRPO_SMOKE = GRPOSettings(
    "qwen3.5-2b/grpo-smoke-v3",
    TrainingLoop(max_steps=1, per_device_batch_size=2, learning_rate=1e-5),
    max_completion_length=384,
)
QWEN35_GRPO_MTP_SMOKE = GRPOSettings(
    "qwen3.5-2b/grpo-mtp-smoke-v2",
    TrainingLoop(max_steps=1, per_device_batch_size=2, learning_rate=1e-5),
    max_completion_length=512,
)

__all__ = [
    "DPOSettings",
    "GRPOSettings",
    "OnPolicyDistillationSettings",
    "LFM25_DPO_SMOKE",
    "LFM25_RENDERER",
    "LFM25_SFT_SMOKE",
    "QWEN35_DPO_SMOKE",
    "QWEN35_GRPO_MTP_SMOKE",
    "QWEN35_GRPO_SMOKE",
    "QWEN35_RENDERER",
    "QWEN35_THINKING_RENDERER",
    "QWEN35_SFT_SMOKE",
    "TrainingRenderer",
    "SFTSettings",
    "SFTValidationSettings",
    "TrainingLoop",
]
