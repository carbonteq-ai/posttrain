"""Package-local serving selections used by tests."""

from __future__ import annotations

import pytest
from posttrain.common import ExecutionTarget, InferenceBinding, JsonValue, Workload
from posttrain.common.variants import LFM_25_12B_THINKING, QWEN_35_2B
from posttrain.serve.prompts import load_prompt_corpus


@pytest.fixture
def local_cuda_target() -> ExecutionTarget:
    return ExecutionTarget(
        id="targets/local-cuda-8gb",
        revision="1",
        device_class="nvidia-cuda",
        memory_gb=8,
        placement={"world_size": 1},
    )


def _screen_engine() -> dict[str, JsonValue]:
    return {
        "max_model_len": 4_096,
        "gpu_memory_utilization": 0.75,
        "dtype": "float16",
        "load_format": "auto",
        "enforce_eager": True,
        "enable_chunked_prefill": True,
        "disable_log_stats": False,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 4_096,
        "kv_cache_dtype": "auto",
        "text_only": True,
        "skip_mm_profiling": True,
    }


@pytest.fixture
def qwen_screen_binding(local_cuda_target: ExecutionTarget) -> InferenceBinding:
    return InferenceBinding(
        id="inference/qwen3.5-2b-vllm-screen@1",
        revision="1",
        model=QWEN_35_2B,
        backend="vllm@0.25.1",
        renderer=QWEN_35_2B.renderer.id,
        engine=_screen_engine(),
        sampling={"max_tokens": 128, "temperature": 0.0},
        target=local_cuda_target,
        purpose=("screen", "smoke"),
    )


@pytest.fixture
def lfm_screen_binding(local_cuda_target: ExecutionTarget) -> InferenceBinding:
    engine = _screen_engine()
    engine.update({"tool_call_parser": "lfm2", "reasoning_parser": "lfm2"})
    return InferenceBinding(
        id="inference/lfm2.5-1.2b-vllm-screen@1",
        revision="1",
        model=LFM_25_12B_THINKING,
        backend="vllm@0.25.1",
        renderer=LFM_25_12B_THINKING.renderer.id,
        engine=engine,
        sampling={"max_tokens": 128, "temperature": 0.0},
        target=local_cuda_target,
        purpose=("screen", "smoke"),
    )


@pytest.fixture
def foundation_smoke_workload() -> Workload:
    return Workload(
        id="workloads/foundation-smoke-v1@1",
        revision="1",
        requests={
            "suite_id": "foundation-smoke-v1",
            "shape_id": "short-interactive",
            "context_window": 1_024,
            "input_tokens": 128,
            "output_tokens": 32,
        },
        concurrency=(1,),
        warmup_repetitions=1,
        measured_repetitions=1,
        required_measures=(
            "serve/output_token_throughput",
            "serve/p95_ttft",
            "serve/peak_gpu_memory_gib",
        ),
    )


@pytest.fixture
def representative_workload() -> Workload:
    corpus = load_prompt_corpus("general-serving-v1")
    return Workload(
        id="workloads/general-serving-v1@1",
        revision="1",
        requests={
            "suite_id": "representative-capacity-v1",
            "shape_id": "general-serving-v1-128out",
            "context_window": 32_768,
            "output_tokens": 128,
            "cohort": "representative",
            "corpus": {
                "id": corpus.manifest.id,
                "revision": corpus.manifest.revision,
                "digest": corpus.manifest.digest,
            },
            "selection_seed": 17,
            "record_count": 128,
        },
        concurrency=(4,),
        warmup_repetitions=1,
        measured_repetitions=32,
    )
