"""LFM2.5 vLLM profiles."""

from .base import VllmEngineConfig, VllmSamplingConfig, VllmServeProfile

_SAMPLING = VllmSamplingConfig(
    max_tokens=512,
    temperature=0.05,
    top_k=50,
    repetition_penalty=1.05,
)

LFM25_VLLM = VllmServeProfile(
    id="lfm2.5-1.2b-thinking/vllm",
    model_family="lfm2.5",
    engine=VllmEngineConfig(max_model_len=4_096, gpu_memory_utilization=0.82),
    sampling=_SAMPLING,
    tool_call_parser="lfm2",
    reasoning_parser="deepseek_r1",
)

LFM25_VLLM_TURBOQUANT_K8 = VllmServeProfile(
    id="lfm2.5-1.2b-thinking/vllm-turboquant-k8",
    model_family="lfm2.5",
    engine=VllmEngineConfig(
        max_model_len=32_768,
        gpu_memory_utilization=0.82,
        kv_cache_dtype="turboquant_k8v4",
        flash_attn_version=2,
    ),
    sampling=_SAMPLING,
    variant="turboquant",
    tool_call_parser="lfm2",
    reasoning_parser="deepseek_r1",
)
