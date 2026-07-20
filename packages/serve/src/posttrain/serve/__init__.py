"""Reusable vLLM serving operations and typed profile definitions."""

from .benchmarks import CORE_INFERENCE_V1, BenchmarkCell, BenchmarkSuite, WorkloadShape
from .profiles import (
    LFM25_VLLM,
    LFM25_VLLM_TURBOQUANT_K8,
    QWEN35_VLLM_MTP,
    QWEN35_VLLM_TEXT,
    QWEN35_VLLM_TURBOQUANT_K8,
    SERVE_PROFILES,
    VllmEngineConfig,
    VllmSamplingConfig,
    VllmServeProfile,
    VllmSpeculativeConfig,
)

__all__ = [
    "BenchmarkCell",
    "BenchmarkSuite",
    "CORE_INFERENCE_V1",
    "LFM25_VLLM",
    "LFM25_VLLM_TURBOQUANT_K8",
    "QWEN35_VLLM_MTP",
    "QWEN35_VLLM_TEXT",
    "QWEN35_VLLM_TURBOQUANT_K8",
    "SERVE_PROFILES",
    "VllmEngineConfig",
    "VllmSamplingConfig",
    "VllmServeProfile",
    "VllmSpeculativeConfig",
    "WorkloadShape",
]
