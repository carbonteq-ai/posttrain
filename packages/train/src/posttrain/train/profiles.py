"""Algorithm settings: technique and optimization schedule only."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
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
    lr_scheduler_type: Literal["linear", "constant", "constant_with_warmup"] = "linear"
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
            self.checkpoint_limit,
        )
        if any(value < 1 for value in counts):
            raise ValueError("training loop counts must be positive")
        if self.checkpoint_steps < 0:
            raise ValueError("checkpoint steps must be non-negative")
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
class DynamicGroupSampling:
    """Bounded DAPO replacement sampling for reward-constant prompt groups."""

    max_candidate_batches: int = 10

    def __post_init__(self) -> None:
        if self.max_candidate_batches < 1:
            raise ValueError("dynamic sampling max candidate batches must be positive")


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
    algorithm: Literal["grpo", "dapo"] = "grpo"
    advantage_scaling: Literal["group", "batch", "none"] = "group"
    clip_epsilon_low: float = 0.2
    clip_epsilon_high: float | None = None
    dynamic_sampling: DynamicGroupSampling | None = None
    mask_truncated_completions: bool = False
    overlong_buffer_tokens: int | None = None
    overlong_penalty_factor: float = 1.0

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
        clip_high = self.resolved_clip_epsilon_high
        if (
            not math.isfinite(self.clip_epsilon_low)
            or not math.isfinite(clip_high)
            or self.clip_epsilon_low <= 0
            or clip_high <= 0
        ):
            raise ValueError("policy clip epsilons must be finite positive numbers")
        if self.algorithm == "grpo" and clip_high != self.clip_epsilon_low:
            raise ValueError("GRPO requires symmetric policy clipping")
        if self.algorithm == "dapo" and clip_high < self.clip_epsilon_low:
            raise ValueError("DAPO upper clipping epsilon cannot be smaller than its lower epsilon")
        if self.advantage_scaling not in {"group", "batch", "none"}:
            raise ValueError("unsupported GRPO advantage scaling")
        if self.dynamic_sampling is not None and self.algorithm != "dapo":
            raise ValueError("dynamic group sampling requires the DAPO algorithm")
        if self.overlong_buffer_tokens is not None:
            if self.algorithm != "dapo":
                raise ValueError("soft overlong punishment requires the DAPO algorithm")
            if self.overlong_buffer_tokens < 1 or self.overlong_buffer_tokens >= self.max_completion_length:
                raise ValueError("DAPO overlong buffer must be positive and smaller than the completion limit")
        if not math.isfinite(self.overlong_penalty_factor) or self.overlong_penalty_factor <= 0:
            raise ValueError("DAPO overlong penalty factor must be a finite positive number")

    @property
    def resolved_clip_epsilon_high(self) -> float:
        if self.clip_epsilon_high is not None:
            return self.clip_epsilon_high
        return 0.28 if self.algorithm == "dapo" else self.clip_epsilon_low


@dataclass(frozen=True, slots=True)
class SAMPOSettings:
    """Hierarchical multi-turn policy-optimization settings."""

    id: str
    loop: TrainingLoop
    num_prompts_per_step: int = 1
    num_generations: int = 2
    max_prompt_length: int = 256
    max_completion_length: int = 128
    beta: float = 0.0
    discount_gamma: float = 0.95
    step_advantage_weight: float = 1.0
    advantage_normalization: Literal["mean", "mean_std"] = "mean"
    clip_epsilon_low: float = 0.003
    clip_epsilon_high: float = 0.004
    dynamic_sampling: DynamicGroupSampling = field(default_factory=lambda: DynamicGroupSampling(3))
    mask_truncated_completions: bool = False
    revision: str = "1"

    def __post_init__(self) -> None:
        _validate_settings(self.id, self.revision)
        if self.num_prompts_per_step < 1 or self.num_generations < 2:
            raise ValueError("SAMPO requires positive prompt groups and at least two generations")
        expected_batch = self.num_prompts_per_step * self.num_generations
        effective_batch = self.loop.per_device_batch_size * self.loop.gradient_accumulation_steps
        if effective_batch != expected_batch:
            raise ValueError("SAMPO effective batch must equal prompts per step times generations")
        if self.max_prompt_length < 1 or self.max_completion_length < 1 or self.beta < 0:
            raise ValueError("invalid SAMPO length or KL settings")
        if self.max_prompt_length + self.max_completion_length > self.loop.max_length:
            raise ValueError("SAMPO loop max_length must cover prompt and completion limits")
        values = (
            self.discount_gamma,
            self.step_advantage_weight,
            self.clip_epsilon_low,
            self.clip_epsilon_high,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("SAMPO numeric settings must be finite")
        if not 0 < self.discount_gamma <= 1:
            raise ValueError("SAMPO discount gamma must be in (0, 1]")
        if self.step_advantage_weight < 0:
            raise ValueError("SAMPO step-advantage weight cannot be negative")
        if self.clip_epsilon_low <= 0 or self.clip_epsilon_high <= 0:
            raise ValueError("SAMPO clip epsilons must be positive")


def shape_online_reward(settings: GRPOSettings, reward: float, completion_tokens: int) -> float:
    """Apply the selected portable DAPO soft overlong punishment."""

    buffer = settings.overlong_buffer_tokens
    if settings.algorithm != "dapo" or buffer is None:
        return reward
    return shape_soft_overlong_reward(
        reward,
        completion_tokens,
        max_completion_tokens=settings.max_completion_length,
        buffer_tokens=buffer,
        penalty_factor=settings.overlong_penalty_factor,
    )


def shape_soft_overlong_reward(
    reward: float,
    completion_tokens: int,
    *,
    max_completion_tokens: int,
    buffer_tokens: int,
    penalty_factor: float,
) -> float:
    """Apply a bounded linear penalty over the final completion-token buffer."""

    threshold = max_completion_tokens - buffer_tokens
    excess = min(max(completion_tokens - threshold, 0), buffer_tokens)
    return reward - (excess / buffer_tokens) * penalty_factor


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
GEMMA4_RENDERER = TrainingRenderer("gemma4-off-v1", "gemma4", "default", "off")

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
    "DynamicGroupSampling",
    "GRPOSettings",
    "shape_online_reward",
    "shape_soft_overlong_reward",
    "OnPolicyDistillationSettings",
    "SAMPOSettings",
    "LFM25_DPO_SMOKE",
    "LFM25_RENDERER",
    "GEMMA4_RENDERER",
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
