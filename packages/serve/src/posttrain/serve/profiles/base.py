"""Typed, vLLM-native serving definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from posttrain.common import ModelProfile

type KvCacheDtype = Literal["auto", "turboquant_k8v4"]


@dataclass(frozen=True, slots=True)
class VllmSpeculativeConfig:
    method: str
    num_speculative_tokens: int

    def __post_init__(self) -> None:
        if not self.method:
            raise ValueError("speculative method cannot be empty")
        if self.num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be positive")

    def as_vllm(self) -> dict[str, str | int]:
        return {
            "method": self.method,
            "num_speculative_tokens": self.num_speculative_tokens,
        }


@dataclass(frozen=True, slots=True)
class VllmEngineConfig:
    max_model_len: int
    gpu_memory_utilization: float
    dtype: str = "float16"
    load_format: str = "auto"
    enforce_eager: bool = False
    enable_chunked_prefill: bool = True
    disable_log_stats: bool = False
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    kv_cache_dtype: KvCacheDtype = "auto"
    text_only: bool = False
    skip_mm_profiling: bool = False
    flash_attn_version: int | None = None
    speculative: VllmSpeculativeConfig | None = None

    def __post_init__(self) -> None:
        if self.max_model_len < 1:
            raise ValueError("max_model_len must be positive")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.max_num_seqs is not None and self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be positive")
        if self.max_num_batched_tokens is not None and self.max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be positive")
        if self.skip_mm_profiling and not self.text_only:
            raise ValueError("skip_mm_profiling is only safe for an explicit text-only profile")

    def as_vllm_kwargs(self, model: ModelProfile | None = None) -> dict[str, object]:
        values: dict[str, object] = {
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": self.dtype,
            "load_format": self.load_format,
            "enforce_eager": self.enforce_eager,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "disable_log_stats": self.disable_log_stats,
            "kv_cache_dtype": self.kv_cache_dtype,
        }
        if self.max_num_seqs is not None:
            values["max_num_seqs"] = self.max_num_seqs
        if self.max_num_batched_tokens is not None:
            values["max_num_batched_tokens"] = self.max_num_batched_tokens
        if self.text_only:
            values["limit_mm_per_prompt"] = {"image": 0, "video": 0}
            values["language_model_only"] = True
            if model is not None and model.hf_text_generation_architecture is not None:
                values["hf_overrides"] = {
                    "architectures": [model.hf_text_generation_architecture]
                }
        if self.skip_mm_profiling:
            values["skip_mm_profiling"] = True
        if self.flash_attn_version is not None:
            values["attention_config"] = {"flash_attn_version": self.flash_attn_version}
        if self.speculative is not None:
            values["speculative_config"] = self.speculative.as_vllm()
        return values

    def as_cli_args(self, model: ModelProfile | None = None) -> tuple[str, ...]:
        values: list[str] = [
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--dtype",
            self.dtype,
            "--load-format",
            self.load_format,
            "--kv-cache-dtype",
            self.kv_cache_dtype,
        ]
        if self.enforce_eager:
            values.append("--enforce-eager")
        if self.enable_chunked_prefill:
            values.append("--enable-chunked-prefill")
        if self.disable_log_stats:
            values.append("--disable-log-stats")
        if self.max_num_seqs is not None:
            values.extend(("--max-num-seqs", str(self.max_num_seqs)))
        if self.max_num_batched_tokens is not None:
            values.extend(("--max-num-batched-tokens", str(self.max_num_batched_tokens)))
        if self.text_only:
            values.append("--language-model-only")
            values.extend(("--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0})))
            if model is not None and model.hf_text_generation_architecture is not None:
                values.extend(
                    (
                        "--hf-overrides",
                        json.dumps({"architectures": [model.hf_text_generation_architecture]}),
                    )
                )
        if self.skip_mm_profiling:
            values.append("--skip-mm-profiling")
        if self.flash_attn_version is not None:
            values.extend(("--attention-config", json.dumps({"flash_attn_version": self.flash_attn_version})))
        if self.speculative is not None:
            values.extend(("--speculative-config", json.dumps(self.speculative.as_vllm())))
        return tuple(values)


@dataclass(frozen=True, slots=True)
class VllmSamplingConfig:
    max_tokens: int
    temperature: float = 0.0
    top_k: int | None = None
    repetition_penalty: float | None = None
    ignore_eos: bool = False
    min_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if self.min_tokens is not None and not 0 <= self.min_tokens <= self.max_tokens:
            raise ValueError("min_tokens must be between zero and max_tokens")

    def as_vllm_kwargs(self) -> dict[str, object]:
        values: dict[str, object] = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "ignore_eos": self.ignore_eos,
        }
        if self.top_k is not None:
            values["top_k"] = self.top_k
        if self.repetition_penalty is not None:
            values["repetition_penalty"] = self.repetition_penalty
        if self.min_tokens is not None:
            values["min_tokens"] = self.min_tokens
        return values


@dataclass(frozen=True, slots=True)
class VllmServeProfile:
    id: str
    model_family: str
    engine: VllmEngineConfig
    sampling: VllmSamplingConfig
    variant: Literal["standard", "turboquant", "mtp"] = "standard"
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None

    def validate_model(self, model: ModelProfile) -> None:
        if model.family != self.model_family:
            raise ValueError(f"serve profile {self.id!r} does not support model family {model.family!r}")
        if self.engine.max_model_len > model.capabilities.native_context_window:
            raise ValueError("serve context exceeds the model's native context window")
        if self.variant == "mtp" and not model.capabilities.mtp:
            raise ValueError("MTP profile requires a model trained with MTP")
        if self.tool_call_parser is not None and model.conversation.tool_calls is None:
            raise ValueError("tool-call parser requires a model tool-call protocol")

    def frontend_args(self) -> tuple[str, ...]:
        values: list[str] = []
        if self.tool_call_parser is not None:
            values.extend(("--enable-auto-tool-choice", "--tool-call-parser", self.tool_call_parser))
        if self.reasoning_parser is not None:
            values.extend(("--reasoning-parser", self.reasoning_parser))
        return tuple(values)
