"""Typed results returned by serving operations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class InferenceRequestResult:
    request_id: str
    record_id: str | None
    input_tokens: int
    output_tokens: int
    queue_seconds: float | None
    prefill_seconds: float | None
    decode_seconds: float | None
    engine_e2e_seconds: float | None
    ttft_seconds: float | None
    tpot_seconds: float | None


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
    target_input_tokens: int | None
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
    cohort: str = "controlled"
    corpus_id: str | None = None
    corpus_revision: str | None = None
    corpus_digest: str | None = None
    corpus_records_measured: int | None = None
    input_tokens_mean: float | None = None
    input_tokens_p95: float | None = None
    request_results: tuple[InferenceRequestResult, ...] = ()
    kv_cache_capacity_tokens: int | None = None
    kv_cache_peak_usage_ratio: float | None = None

    def metrics(self) -> dict[str, int | float]:
        """Return irreducible point evidence.

        Rates and latency percentiles remain available on the direct Python
        result for an immediate caller, but they are not persisted as a second
        source of truth. Observatory rebuilds them from these counters and the
        complete request-trace population.
        """

        values: dict[str, int | float] = {
            "serve/run/concurrency": self.concurrency,
            "serve/run/context_tokens": self.context_window,
            "serve/run/requests_attempted": self.requests,
            "serve/run/requests_measured": len(self.request_results) or self.requests,
            "serve/run/requests_failed": 0,
            "serve/run/requests_unsupported": 0,
            "serve/run/point_resource_exhausted": 0,
            "serve/run/point_unsupported": 0,
            "serve/run/point_failed": 0,
            "serve/run/input_tokens_measured": self.input_tokens,
            "serve/run/output_tokens_measured": self.output_tokens,
            "serve/run/measurement_duration_s": self.elapsed_seconds,
            "serve/backend/engine_start_duration_s": self.engine_start_seconds,
        }
        if self.corpus_records_measured is not None:
            values["serve/run/corpus_records_measured"] = self.corpus_records_measured
        optional = {
            "serve/backend/peak_vram_bytes": (
                self.peak_gpu_memory_gib * 1024**3 if self.peak_gpu_memory_gib is not None else None
            ),
            "serve/backend/kv_cache_capacity_tokens": self.kv_cache_capacity_tokens,
            "serve/backend/kv_cache_peak_usage_ratio": self.kv_cache_peak_usage_ratio,
        }
        values.update({name: value for name, value in optional.items() if value is not None})
        return values

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkPointFailure:
    """Safe evidence for a concurrency point that could not be measured."""

    sweep_index: int
    concurrency: int
    status: Literal["resource_exhausted", "unsupported", "failed"]
    error_class: str
    message: str
    recoverable: bool

    def __post_init__(self) -> None:
        if self.sweep_index < 0:
            raise ValueError("benchmark point failure sweep_index must be non-negative")
        if self.concurrency < 1:
            raise ValueError("benchmark point failure concurrency must be positive")
        if not self.error_class or not self.message:
            raise ValueError("benchmark point failure requires a safe error class and message")


@dataclass(frozen=True, slots=True)
class BenchmarkSweepResult:
    """One engine lifecycle containing an ordered concurrency sweep."""

    points: tuple[BenchmarkResult, ...]
    configured_concurrencies: tuple[int, ...]
    point_failures: tuple[BenchmarkPointFailure, ...] = ()
    termination_reason: Literal[
        "configured_sweep_complete",
        "resource_exhausted_boundary",
        "unsupported_boundary",
        "failed_boundary",
    ] = "configured_sweep_complete"
    schema_version: Literal[2] = 2

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("benchmark sweep must contain at least one measured point")
        measured = tuple(point.concurrency for point in self.points)
        if measured != self.configured_concurrencies[: len(measured)]:
            raise ValueError("benchmark sweep points must preserve configured concurrency order")
        expected_failure_indices = tuple(range(len(measured), len(measured) + len(self.point_failures)))
        if tuple(failure.sweep_index for failure in self.point_failures) != expected_failure_indices:
            raise ValueError("benchmark point failures must follow measured points in configured order")
        if any(
            failure.concurrency != self.configured_concurrencies[failure.sweep_index] for failure in self.point_failures
        ):
            raise ValueError("benchmark point failures must preserve configured concurrency order")

    @property
    def completed_concurrencies(self) -> tuple[int, ...]:
        return tuple(point.concurrency for point in self.points)

    def as_json(self, *, runtime_configuration: dict[str, Any] | None = None) -> dict[str, Any]:
        first = self.points[0]
        return {
            "schema_version": self.schema_version,
            "methodology": "single_run_concurrency_sweep",
            "backend": first.backend,
            "model_variant": {
                "repository": first.model,
                "revision": first.revision,
            },
            "inference_binding_id": first.inference_binding_id,
            "suite_id": first.suite_id,
            "context_tokens": first.context_window,
            "configured_concurrencies": list(self.configured_concurrencies),
            "completed_concurrencies": list(self.completed_concurrencies),
            "termination_reason": self.termination_reason,
            "points": [point.as_json() for point in self.points],
            "point_failures": [asdict(failure) for failure in self.point_failures],
            "runtime_configuration": runtime_configuration or {},
        }
