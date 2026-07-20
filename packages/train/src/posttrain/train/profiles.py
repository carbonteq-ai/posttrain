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
    loss_kernel: Literal["liger", "torch"] = "torch"
    qlora: QLoRAProfile = field(default_factory=QLoRAProfile)

    def __post_init__(self) -> None:
        _validate_profile(self.id, self.model_family, self.renderer)
        if self.beta <= 0:
            raise ValueError("DPO beta must be positive")


@dataclass(frozen=True, slots=True)
class GRPORolloutProfile:
    id: str
    engine: Literal["transformers", "vllm"]
    vllm_mode: Literal["colocate"] | None = None
    sleep_during_optimization: bool = False
    gpu_memory_utilization: float | None = None
    tensor_parallel_size: int = 1
    max_model_length: int | None = None
    text_only: bool = False
    skip_multimodal_profiling: bool = False
    kv_cache_memory_bytes: int | None = None
    speculative_method: str | None = None
    num_speculative_tokens: int | None = None

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.id):
            raise ValueError("rollout profile id is invalid")
        if self.tensor_parallel_size < 1:
            raise ValueError("rollout tensor parallel size must be positive")
        if self.engine == "transformers":
            if any(
                value is not None
                for value in (
                    self.vllm_mode,
                    self.gpu_memory_utilization,
                    self.max_model_length,
                    self.speculative_method,
                    self.num_speculative_tokens,
                    self.kv_cache_memory_bytes,
                )
            ) or self.sleep_during_optimization or self.text_only or self.skip_multimodal_profiling:
                raise ValueError("Transformers rollouts cannot declare vLLM settings")
            return
        if self.vllm_mode != "colocate" or self.gpu_memory_utilization is None:
            raise ValueError("vLLM rollouts require colocate mode and a memory budget")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("vLLM GPU memory utilization must be between zero and one")
        if self.skip_multimodal_profiling and not self.text_only:
            raise ValueError("skipping multimodal profiling requires an explicit text-only rollout")
        if (self.speculative_method is None) != (self.num_speculative_tokens is None):
            raise ValueError("speculative method and token count must be configured together")
        if self.num_speculative_tokens is not None and self.num_speculative_tokens < 1:
            raise ValueError("speculative token count must be positive")
        if self.kv_cache_memory_bytes is not None and self.kv_cache_memory_bytes < 1:
            raise ValueError("vLLM KV cache memory must be positive")

    def speculative_config(self) -> dict[str, str | int] | None:
        if self.speculative_method is None:
            return None
        assert self.num_speculative_tokens is not None
        return {
            "method": self.speculative_method,
            "num_speculative_tokens": self.num_speculative_tokens,
        }


TRANSFORMERS_GRPO_ROLLOUT = GRPORolloutProfile("transformers-generate-v1", "transformers")


@dataclass(frozen=True, slots=True)
class GRPOProfile:
    id: str
    model_family: str
    renderer: RendererProfile
    loop: TrainingLoop
    num_generations: int = 2
    max_prompt_length: int = 256
    max_completion_length: int = 128
    beta: float = 0.0
    rollout: GRPORolloutProfile = TRANSFORMERS_GRPO_ROLLOUT
    qlora: QLoRAProfile = field(default_factory=QLoRAProfile)

    def __post_init__(self) -> None:
        _validate_profile(self.id, self.model_family, self.renderer)
        if self.num_generations < 2:
            raise ValueError("GRPO requires at least two generations per prompt")
        if self.loop.per_device_batch_size % self.num_generations != 0:
            raise ValueError("GRPO batch size must be divisible by num_generations")
        if self.max_prompt_length < 1 or self.max_completion_length < 1 or self.beta < 0:
            raise ValueError("invalid GRPO generation or KL settings")


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
    loss_kernel="liger",
)
QWEN35_GRPO_SMOKE = GRPOProfile(
    "qwen3.5-2b/grpo-qlora-vllm-smoke-v1",
    "qwen3.5",
    QWEN35_RENDERER,
    TrainingLoop(max_steps=1, per_device_batch_size=2, learning_rate=1e-5),
    max_completion_length=256,
    rollout=GRPORolloutProfile(
        "qwen3.5-2b/vllm-colocate-v1",
        "vllm",
        vllm_mode="colocate",
        sleep_during_optimization=True,
        gpu_memory_utilization=0.2,
        max_model_length=512,
        text_only=True,
        skip_multimodal_profiling=True,
        kv_cache_memory_bytes=64 * 1024 * 1024,
    ),
)
QWEN35_GRPO_MTP_SMOKE = GRPOProfile(
    "qwen3.5-2b/grpo-qlora-vllm-mtp-smoke-v1",
    "qwen3.5",
    QWEN35_RENDERER,
    TrainingLoop(max_steps=1, per_device_batch_size=2, learning_rate=1e-5),
    max_completion_length=512,
    rollout=GRPORolloutProfile(
        "qwen3.5-2b/vllm-colocate-mtp-v1",
        "vllm",
        vllm_mode="colocate",
        sleep_during_optimization=True,
        gpu_memory_utilization=0.2,
        max_model_length=1_024,
        text_only=True,
        skip_multimodal_profiling=True,
        speculative_method="qwen3_next_mtp",
        num_speculative_tokens=2,
    ),
)

__all__ = [
    "DPOProfile",
    "GRPOProfile",
    "GRPORolloutProfile",
    "LFM25_DPO_SMOKE",
    "LFM25_RENDERER",
    "LFM25_SFT_SMOKE",
    "QLoRAProfile",
    "QWEN35_DPO_SMOKE",
    "QWEN35_GRPO_MTP_SMOKE",
    "QWEN35_GRPO_SMOKE",
    "QWEN35_RENDERER",
    "QWEN35_SFT_SMOKE",
    "RendererProfile",
    "SFTProfile",
    "TrainingLoop",
    "TRANSFORMERS_GRPO_ROLLOUT",
]
