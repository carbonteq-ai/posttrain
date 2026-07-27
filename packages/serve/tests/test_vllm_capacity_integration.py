"""Opt-in real-GPU release gate for the canonical serving-capacity sweep."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from posttrain.common import (
    ExecutionTarget,
    InferenceBinding,
    JsonValue,
    NullObserver,
    RunContext,
    Workload,
)
from posttrain.common.variants import QWEN_35_08B
from posttrain.serve import ServeBenchmarkRequest, benchmark
from posttrain.serve.prompts import load_prompt_corpus


@pytest.mark.gpu
@pytest.mark.network
def test_qwen08b_representative_capacity_sweep_on_real_vllm(tmp_path: Path) -> None:
    """Load one real engine and retain two representative operating points.

    Set ``POSTTRAIN_RUN_SERVE_GPU_INTEGRATION=1`` to run this release gate.
    ``POSTTRAIN_SERVE_GPU_VARIANT`` may be ``standard``, ``mtp``,
    ``turboquant``, or ``mtp-turboquant``.
    """

    if os.environ.get("POSTTRAIN_RUN_SERVE_GPU_INTEGRATION") != "1":
        pytest.skip("set POSTTRAIN_RUN_SERVE_GPU_INTEGRATION=1 for the real vLLM release gate")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("a CUDA GPU is required for the real vLLM release gate")

    variant = os.environ.get("POSTTRAIN_SERVE_GPU_VARIANT", "standard")
    if variant not in {"standard", "mtp", "turboquant", "mtp-turboquant"}:
        pytest.fail(f"unsupported POSTTRAIN_SERVE_GPU_VARIANT: {variant}")
    engine: dict[str, JsonValue] = {
        "max_model_len": 4_096,
        "gpu_memory_utilization": 0.75,
        "dtype": "float16",
        "load_format": "auto",
        "enforce_eager": True,
        "enable_chunked_prefill": True,
        "disable_log_stats": False,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 4_096,
        "kv_cache_dtype": "turboquant_k8v4" if "turboquant" in variant else "auto",
        "text_only": True,
        "skip_mm_profiling": True,
    }
    if "turboquant" in variant:
        engine["flash_attn_version"] = 2
    if "mtp" in variant:
        engine["speculative_config"] = {"method": "mtp", "num_speculative_tokens": 1}
    target = ExecutionTarget(
        id="targets/integration-cuda",
        revision="1",
        device_class="nvidia-cuda",
        memory_gb=8,
        placement={"world_size": 1},
    )
    binding = InferenceBinding(
        id=f"inference/qwen3.5-0.8b-vllm-integration-{variant}@1",
        revision="1",
        model=QWEN_35_08B,
        backend="vllm@0.25.1",
        renderer=QWEN_35_08B.renderer.id,
        engine=engine,
        sampling={"max_tokens": 8, "temperature": 0.0},
        target=target,
        purpose=("screen",),
    )
    corpus = load_prompt_corpus("general-serving-v1")
    workload = Workload(
        id="workloads/general-serving-gpu-integration@1",
        revision="1",
        requests={
            "suite_id": "general-serving-gpu-integration",
            "shape_id": "representative-8out",
            "context_window": 4_096,
            "output_tokens": 8,
            "cohort": "representative",
            "corpus": {
                "id": corpus.manifest.id,
                "revision": corpus.manifest.revision,
                "digest": corpus.manifest.digest,
            },
            "selection_seed": 17,
            "record_count": 7,
        },
        concurrency=(1, 2, 4),
        warmup_repetitions=1,
        measured_repetitions=1,
    )

    context = RunContext(
        project_id="serve-integration",
        work_package_id="screen/qwen08b-vllm",
        run_id="serve-integration-run",
        job_kind="serve.benchmark",
        job_definition_version="serve/vllm-benchmark@1",
        workspace=tmp_path,
        observer=NullObserver(),
    )
    sweep = benchmark(context, ServeBenchmarkRequest(binding, workload, target))

    assert sweep.configured_concurrencies == (1, 2, 4)
    assert sweep.completed_concurrencies == (1, 2, 4)
    assert sweep.point_failures == ()
    assert sweep.termination_reason == "configured_sweep_complete"
    assert all(point.cohort == "representative" for point in sweep.points)
    assert all(point.corpus_digest == corpus.manifest.digest for point in sweep.points)
    assert all(point.request_results for point in sweep.points)
    assert all(point.metrics()["serve/run/measurement_duration_s"] > 0 for point in sweep.points)
    artifact = json.loads((tmp_path / "serving-result.json").read_text())
    assert artifact["schema_version"] == 2
    assert artifact["configured_concurrencies"] == [1, 2, 4]
    assert artifact["model_variant_id"] == QWEN_35_08B.id
    assert len(artifact["points"]) == 3
