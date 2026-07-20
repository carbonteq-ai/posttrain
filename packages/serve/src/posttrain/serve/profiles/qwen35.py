"""Qwen3.5 vLLM profiles."""

from .base import VllmEngineConfig, VllmSamplingConfig, VllmServeProfile, VllmSpeculativeConfig

_TEXT_ENGINE = VllmEngineConfig(
    max_model_len=4_096,
    gpu_memory_utilization=0.75,
    enforce_eager=True,
    max_num_seqs=4,
    max_num_batched_tokens=4_096,
    text_only=True,
    skip_mm_profiling=True,
)

QWEN35_VLLM_TEXT = VllmServeProfile(
    id="qwen3.5-2b/vllm-text",
    model_family="qwen3.5",
    engine=_TEXT_ENGINE,
    sampling=VllmSamplingConfig(max_tokens=128),
    tool_call_parser="qwen3_xml",
    reasoning_parser="qwen3",
)

QWEN35_VLLM_TURBOQUANT_K8 = VllmServeProfile(
    id="qwen3.5-2b/vllm-turboquant-k8",
    model_family="qwen3.5",
    engine=VllmEngineConfig(
        max_model_len=32_768,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
        max_num_seqs=4,
        max_num_batched_tokens=4_096,
        kv_cache_dtype="turboquant_k8v4",
        text_only=True,
        skip_mm_profiling=True,
        flash_attn_version=2,
    ),
    sampling=VllmSamplingConfig(max_tokens=128),
    variant="turboquant",
    tool_call_parser="qwen3_xml",
    reasoning_parser="qwen3",
)

QWEN35_VLLM_MTP = VllmServeProfile(
    id="qwen3.5-2b/vllm-mtp",
    model_family="qwen3.5",
    engine=VllmEngineConfig(
        max_model_len=4_096,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
        max_num_seqs=4,
        max_num_batched_tokens=4_096,
        text_only=True,
        skip_mm_profiling=True,
        speculative=VllmSpeculativeConfig(method="qwen3_next_mtp", num_speculative_tokens=2),
    ),
    sampling=VllmSamplingConfig(max_tokens=128),
    variant="mtp",
    tool_call_parser="qwen3_xml",
    reasoning_parser="qwen3",
)
