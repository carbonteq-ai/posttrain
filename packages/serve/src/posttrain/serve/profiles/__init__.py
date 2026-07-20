"""Published serving profiles owned by the serve package."""

from .base import VllmEngineConfig, VllmSamplingConfig, VllmServeProfile, VllmSpeculativeConfig
from .lfm25 import LFM25_VLLM, LFM25_VLLM_TURBOQUANT_K8
from .qwen35 import QWEN35_VLLM_MTP, QWEN35_VLLM_TEXT, QWEN35_VLLM_TURBOQUANT_K8

SERVE_PROFILES = {
    profile.id: profile
    for profile in (
        QWEN35_VLLM_TEXT,
        QWEN35_VLLM_TURBOQUANT_K8,
        QWEN35_VLLM_MTP,
        LFM25_VLLM,
        LFM25_VLLM_TURBOQUANT_K8,
    )
}

__all__ = [
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
]
