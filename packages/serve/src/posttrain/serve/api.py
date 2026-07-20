"""Public serving operations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable

from posttrain.common import ExecutionContext, LocalArtifactRef, ProducedArtifact, TraceObservation

from .backends.vllm import run_offline_benchmark
from .requests import BenchmarkRequest
from .results import BenchmarkResult

type BenchmarkRunner = Callable[[BenchmarkRequest], BenchmarkResult]


def benchmark(
    context: ExecutionContext,
    request: BenchmarkRequest,
    *,
    runner: BenchmarkRunner = run_offline_benchmark,
) -> BenchmarkResult:
    """Run one benchmark cell and emit only direct observations."""

    context.event(
        "serve_benchmark_started",
        {
            "model_profile_id": request.model.id,
            "serve_profile_id": request.profile.id,
            "suite_id": request.cell.suite_id,
            "cell_id": request.cell.id,
        },
    )
    result = runner(request)
    metric_attributes = {
        "model_profile_id": request.model.id,
        "serve_profile_id": request.profile.id,
        "suite_id": request.cell.suite_id,
        "cell_id": request.cell.id,
    }
    for name, value in result.metrics().items():
        context.metric(name, value, attributes=metric_attributes)
    for index, sample in enumerate(result.samples):
        context.trace(
            TraceObservation(
                trace_type="inference",
                external_id=f"{context.attempt.id}:{request.cell.id}:{index}",
                payload={
                    "messages": [{"role": "assistant", "content": sample}],
                    "model": result.model,
                    "revision": result.revision,
                    "target_input_tokens": result.target_input_tokens,
                    "target_output_tokens": result.target_output_tokens,
                },
                attributes=metric_attributes,
            )
        )
    encoded = (json.dumps(result.as_json(), indent=2, sort_keys=True) + "\n").encode()
    output = context.workspace / "benchmark.json"
    output.write_bytes(encoded)
    context.artifact(
        ProducedArtifact(
            name=f"serving/{request.model.id}/{request.cell.id}",
            kind="serving-benchmark",
            reference=LocalArtifactRef(output, hashlib.sha256(encoded).hexdigest()),
            metadata=metric_attributes,
        )
    )
    context.event("serve_benchmark_completed", {"cell_id": request.cell.id})
    return result
