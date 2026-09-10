"""Typed, vLLM-native serving definitions."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

type KvCacheDtype = Literal["auto", "turboquant_k8v4"]


@dataclass(frozen=True, slots=True)
class VllmDraftModel:
    """An immutable draft checkpoint, optionally pre-materialized by the host."""

    repo_id: str
    revision: str
    path: str | None = None

    def __post_init__(self) -> None:
        if not self.repo_id.strip() or "/" not in self.repo_id:
            raise ValueError("draft model repo_id must be a non-empty Hub repository id")
        if re.fullmatch(r"[0-9a-f]{40}", self.revision) is None:
            raise ValueError("draft model revision must be an immutable 40-character commit SHA")
        if self.path is not None and not Path(self.path).is_absolute():
            raise ValueError("draft model path must be absolute inside the serving runtime")

    def as_vllm(self) -> dict[str, str]:
        if self.path is not None:
            return {"model": self.path}
        return {"model": self.repo_id, "revision": self.revision}


@dataclass(frozen=True, slots=True)
class VllmSpeculativeConfig:
    method: str
    num_speculative_tokens: int
    draft_model: VllmDraftModel | None = None

    def __post_init__(self) -> None:
        if not self.method or re.fullmatch(r"[a-z0-9][a-z0-9._-]*", self.method) is None:
            raise ValueError("speculative method must be a lowercase stable identifier")
        if self.num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be positive")
        if self.method == "dspark" and self.draft_model is None:
            raise ValueError("DSpark speculative decoding requires an immutable draft model")

    def as_vllm(self) -> dict[str, str | int]:
        values: dict[str, str | int] = {
            "method": self.method,
            "num_speculative_tokens": self.num_speculative_tokens,
        }
        if self.draft_model is not None:
            values.update(self.draft_model.as_vllm())
        return values


@dataclass(frozen=True, slots=True)
class VllmEngineConfig:
    max_model_len: int
    gpu_memory_utilization: float
    tensor_parallel_size: int = 1
    dtype: str = "float16"
    load_format: str = "auto"
    enforce_eager: bool = False
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = False
    trust_remote_code: bool = False
    disable_log_stats: bool = False
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    kv_cache_dtype: KvCacheDtype = "auto"
    text_only: bool = False
    skip_mm_profiling: bool = False
    flash_attn_version: int | None = None
    speculative: VllmSpeculativeConfig | None = None

    def __post_init__(self) -> None:
        if isinstance(self.max_model_len, bool) or not isinstance(self.max_model_len, int) or self.max_model_len < 1:
            raise ValueError("max_model_len must be positive")
        if (
            isinstance(self.gpu_memory_utilization, bool)
            or not isinstance(self.gpu_memory_utilization, (int, float))
            or not math.isfinite(float(self.gpu_memory_utilization))
            or not 0 < self.gpu_memory_utilization <= 1
        ):
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if (
            isinstance(self.tensor_parallel_size, bool)
            or not isinstance(self.tensor_parallel_size, int)
            or self.tensor_parallel_size < 1
        ):
            raise ValueError("tensor_parallel_size must be positive")
        if self.max_num_seqs is not None and (
            isinstance(self.max_num_seqs, bool) or not isinstance(self.max_num_seqs, int) or self.max_num_seqs < 1
        ):
            raise ValueError("max_num_seqs must be positive")
        if self.max_num_batched_tokens is not None and (
            isinstance(self.max_num_batched_tokens, bool)
            or not isinstance(self.max_num_batched_tokens, int)
            or self.max_num_batched_tokens < 1
        ):
            raise ValueError("max_num_batched_tokens must be positive")
        if self.skip_mm_profiling and not self.text_only:
            raise ValueError("skip_mm_profiling is only safe for an explicit text-only profile")

    def as_vllm_kwargs(self) -> dict[str, object]:
        values: dict[str, object] = {
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "dtype": self.dtype,
            "load_format": self.load_format,
            "enforce_eager": self.enforce_eager,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "enable_prefix_caching": self.enable_prefix_caching,
            "trust_remote_code": self.trust_remote_code,
            "disable_log_stats": self.disable_log_stats,
            "kv_cache_dtype": self.kv_cache_dtype,
        }
        if self.max_num_seqs is not None:
            values["max_num_seqs"] = self.max_num_seqs
        if self.max_num_batched_tokens is not None:
            values["max_num_batched_tokens"] = self.max_num_batched_tokens
        if self.text_only:
            values["limit_mm_per_prompt"] = {"image": 0, "video": 0, "audio": 0}
        if self.skip_mm_profiling:
            values["skip_mm_profiling"] = True
        if self.flash_attn_version is not None:
            values["attention_config"] = {"flash_attn_version": self.flash_attn_version}
        if self.speculative is not None:
            values["speculative_config"] = self.speculative.as_vllm()
        return values

    def as_cli_args(self) -> tuple[str, ...]:
        values: list[str] = [
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
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
        if self.enable_prefix_caching:
            values.append("--enable-prefix-caching")
        if self.trust_remote_code:
            values.append("--trust-remote-code")
        if self.disable_log_stats:
            values.append("--disable-log-stats")
        if self.max_num_seqs is not None:
            values.extend(("--max-num-seqs", str(self.max_num_seqs)))
        if self.max_num_batched_tokens is not None:
            values.extend(("--max-num-batched-tokens", str(self.max_num_batched_tokens)))
        if self.text_only:
            values.extend(("--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0, "audio": 0})))
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
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    ignore_eos: bool = False
    min_tokens: int | None = None
    extra_body: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or self.temperature < 0
        ):
            raise ValueError("temperature cannot be negative")
        for name, value in (("top_p", self.top_p), ("min_p", self.min_p)):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be in [0, 1]")
        if self.presence_penalty is not None and (
            isinstance(self.presence_penalty, bool)
            or not isinstance(self.presence_penalty, (int, float))
            or not math.isfinite(float(self.presence_penalty))
        ):
            raise ValueError("presence_penalty must be finite")
        if self.extra_body is not None and (
            not isinstance(self.extra_body, Mapping)
            or any(not isinstance(key, str) for key in self.extra_body)
        ):
            raise ValueError("extra_body must be an object with string keys")
        if self.min_tokens is not None and (
            isinstance(self.min_tokens, bool)
            or not isinstance(self.min_tokens, int)
            or not 0 <= self.min_tokens <= self.max_tokens
        ):
            raise ValueError("min_tokens must be between zero and max_tokens")

    def as_vllm_kwargs(self) -> dict[str, object]:
        values: dict[str, object] = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "ignore_eos": self.ignore_eos,
        }
        if self.top_p is not None:
            values["top_p"] = self.top_p
        if self.top_k is not None:
            values["top_k"] = self.top_k
        if self.min_p is not None:
            values["min_p"] = self.min_p
        if self.repetition_penalty is not None:
            values["repetition_penalty"] = self.repetition_penalty
        if self.presence_penalty is not None:
            values["presence_penalty"] = self.presence_penalty
        if self.min_tokens is not None:
            values["min_tokens"] = self.min_tokens
        return values
