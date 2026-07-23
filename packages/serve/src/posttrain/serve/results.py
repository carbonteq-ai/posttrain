"""Typed results returned by serving operations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    backend: str
    model: str
    revision: str
    inference_binding_id: str
    suite_id: str
    cell_id: str
    shape_id: str
    context_window: int
    concurrency: int
    target_input_tokens: int
    target_output_tokens: int
    requests: int
    iterations: int
    input_tokens: int
    output_tokens: int
    engine_start_seconds: float
    elapsed_seconds: float
    request_throughput: float
    input_token_throughput: float
    output_token_throughput: float
    total_token_throughput: float
    mean_batch_latency: float
    mean_ttft: float | None
    p50_ttft: float | None
    p95_ttft: float | None
    mean_tpot: float | None
    p50_tpot: float | None
    p95_tpot: float | None
    baseline_gpu_memory_gib: float | None
    peak_gpu_memory_gib: float | None
    peak_gpu_memory_delta_gib: float | None
    samples: tuple[str, ...]

    def metrics(self) -> dict[str, int | float]:
        values: dict[str, int | float] = {
            "serve/requests": self.requests,
            "serve/iterations": self.iterations,
            "serve/input_tokens": self.input_tokens,
            "serve/output_tokens": self.output_tokens,
            "serve/engine_start_seconds": self.engine_start_seconds,
            "serve/elapsed_seconds": self.elapsed_seconds,
            "serve/request_throughput": self.request_throughput,
            "serve/input_token_throughput": self.input_token_throughput,
            "serve/output_token_throughput": self.output_token_throughput,
            "serve/total_token_throughput": self.total_token_throughput,
            "serve/mean_batch_latency": self.mean_batch_latency,
            "serve/context_window": self.context_window,
            "serve/concurrency": self.concurrency,
        }
        optional = {
            "serve/mean_ttft": self.mean_ttft,
            "serve/p50_ttft": self.p50_ttft,
            "serve/p95_ttft": self.p95_ttft,
            "serve/mean_tpot": self.mean_tpot,
            "serve/p50_tpot": self.p50_tpot,
            "serve/p95_tpot": self.p95_tpot,
            "serve/baseline_gpu_memory_gib": self.baseline_gpu_memory_gib,
            "serve/peak_gpu_memory_gib": self.peak_gpu_memory_gib,
            "serve/peak_gpu_memory_delta_gib": self.peak_gpu_memory_delta_gib,
        }
        values.update({name: value for name, value in optional.items() if value is not None})
        return values

    def as_json(self) -> dict[str, Any]:
        return asdict(self)
