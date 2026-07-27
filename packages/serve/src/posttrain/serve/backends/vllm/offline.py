"""Offline vLLM benchmark adapter with no job or observability dependency."""

from __future__ import annotations

import statistics
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from random import Random
from typing import Any, Literal, cast

from posttrain.common.cuda import TorchModule, activate_cuda_toolkit

from ...benchmarks import BenchmarkCell
from ...prompts import render_prompt
from ...results import (
    BenchmarkPointFailure,
    BenchmarkResult,
    BenchmarkSweepResult,
    InferenceRequestResult,
)
from .bindings import VllmBenchmarkConfig

type BenchmarkPhase = Callable[[str, dict[str, str | int]], AbstractContextManager[Any]]
type BenchmarkMetric = Callable[[dict[str, float], dict[str, str | int]], None]


@contextmanager
def _unobserved_phase(_name: str, _attributes: dict[str, str | int]) -> Iterator[None]:
    yield


def _unobserved_metric(_values: dict[str, float], _attributes: dict[str, str | int]) -> None:
    pass


class _KvCacheTracker:
    """Retain measured scheduler pressure and emit a bounded live series."""

    def __init__(self, metric: BenchmarkMetric, interval_seconds: float = 0.5) -> None:
        self.metric = metric
        self.interval_seconds = interval_seconds
        self.peak_usage_ratio = 0.0
        self.samples = 0
        self._active = False
        self._last_emitted_at = 0.0
        self._attributes: dict[str, str | int] = {}

    def start(self, attributes: dict[str, str | int] | None = None) -> None:
        self.peak_usage_ratio = 0.0
        self.samples = 0
        self._last_emitted_at = 0.0
        self._attributes = dict(attributes or {})
        self._active = True

    def stop(self) -> None:
        self._active = False

    def record(
        self,
        scheduler_stats: Any,
        iteration_stats: Any,
        mm_cache_stats: Any = None,
        engine_idx: int = 0,
    ) -> None:
        del iteration_stats, mm_cache_stats
        if not self._active or scheduler_stats is None:
            return
        usage = max(0.0, min(float(scheduler_stats.kv_cache_usage), 1.0))
        self.samples += 1
        self.peak_usage_ratio = max(self.peak_usage_ratio, usage)
        observed_at = time.monotonic()
        if self._last_emitted_at and observed_at - self._last_emitted_at < self.interval_seconds:
            return
        self._last_emitted_at = observed_at
        self.metric(
            {
                "serve/backend/kv_cache_usage_ratio": usage,
                "serve/backend/running_requests": float(scheduler_stats.num_running_reqs),
                "serve/backend/waiting_requests": float(scheduler_stats.num_waiting_reqs),
            },
            {
                "backend": "vllm",
                "engine_index": engine_idx,
                "phase": "measurement",
                **self._attributes,
            },
        )

    def log(self) -> None:
        pass

    def log_engine_initialized(self) -> None:
        pass

    def record_sleep_state(self, sleep: int = 0, level: int = 0) -> None:
        del sleep, level


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile + 0.5)))
    return ordered[index]


def _controlled_prompt_ids(tokenizer: Any, target_tokens: int, concurrency: int) -> list[dict[str, list[int]]]:
    seeds = (
        "Inference benchmark text covering language, numbers, and punctuation. ",
        "A deterministic workload should be reproducible across repeated runs. ",
        "Prefill and decode performance depend on token and batch dimensions. ",
        "Systems measurements remain separate from model capability evaluation. ",
    )
    seed_ids = [token for seed in seeds for token in tokenizer.encode(seed, add_special_tokens=False)]
    if not seed_ids:
        raise RuntimeError("tokenizer produced no IDs for the controlled prompt seed")
    prompts: list[dict[str, list[int]]] = []
    for request_index in range(concurrency):
        offset = request_index % len(seed_ids)
        rotated = seed_ids[offset:] + seed_ids[:offset]
        repeats = (target_tokens + len(rotated) - 1) // len(rotated)
        prompts.append({"prompt_token_ids": (rotated * repeats)[:target_tokens]})
    return prompts


def _representative_prompt_batches(
    tokenizer: Any,
    request: VllmBenchmarkConfig,
    cell: BenchmarkCell,
) -> tuple[list[list[dict[str, list[int]]]], tuple[str, ...]]:
    corpus = request.corpus
    if corpus is None:
        raise RuntimeError("representative benchmark requires a resolved prompt corpus")
    records = list(corpus.records)
    Random(request.selection_seed).shuffle(records)
    batches: list[list[dict[str, list[int]]]] = []
    selected_ids: list[str] = []
    for iteration in range(cell.iterations):
        batch: list[dict[str, list[int]]] = []
        for lane in range(cell.concurrency):
            record = records[(iteration * cell.concurrency + lane) % len(records)]
            prompt_ids = list(render_prompt(tokenizer, record, request.model))
            if len(prompt_ids) + cell.output_tokens > cell.context_window:
                raise RuntimeError(
                    f"rendered prompt {record.id!r} exceeds the selected context window with the output budget"
                )
            batch.append({"prompt_token_ids": prompt_ids})
            selected_ids.append(record.id)
        batches.append(batch)
    return batches, tuple(selected_ids)


class _GpuMemoryMonitor:
    def __init__(self, device_index: int = 0, interval_seconds: float = 0.02) -> None:
        self.device_index = device_index
        self.interval_seconds = interval_seconds
        self.baseline_bytes: int | None = None
        self.peak_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml: Any = None
        self._handle: Any = None
        self._window_peak_bytes: int | None = None

    def _sample(self) -> int:
        return int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used)

    def start(self) -> None:
        try:
            import pynvml  # pyright: ignore[reportMissingImports]
        except ImportError:
            self._nvml = None
            return
        try:
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self.baseline_bytes = self._sample()
            self.peak_bytes = self.baseline_bytes
        except pynvml.NVMLError:
            self._nvml = None
            return

        def sample_until_stopped() -> None:
            while not self._stop.wait(self.interval_seconds):
                try:
                    sample = self._sample()
                    self.peak_bytes = max(self.peak_bytes or 0, sample)
                    if self._window_peak_bytes is not None:
                        self._window_peak_bytes = max(self._window_peak_bytes, sample)
                except self._nvml.NVMLError:
                    return

        self._thread = threading.Thread(target=sample_until_stopped, daemon=True)
        self._thread.start()

    def start_window(self) -> None:
        self._window_peak_bytes = self._sample() if self._nvml is not None else None

    def stop_window(self) -> float | None:
        if self._nvml is None or self._window_peak_bytes is None:
            return None
        try:
            self._window_peak_bytes = max(self._window_peak_bytes, self._sample())
        except self._nvml.NVMLError:
            pass
        value = self._window_peak_bytes / 1024**3
        self._window_peak_bytes = None
        return value

    def stop(self) -> tuple[float | None, float | None, float | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._nvml is None:
            return None, None, None
        try:
            self.peak_bytes = max(self.peak_bytes or 0, self._sample())
            self._nvml.nvmlShutdown()
        except self._nvml.NVMLError:
            pass
        baseline = (self.baseline_bytes or 0) / 1024**3
        peak = (self.peak_bytes or 0) / 1024**3
        return baseline, peak, max(0.0, peak - baseline)


def _load_vllm() -> tuple[Any, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is not installed; install posttrain-serve[vllm]") from error
    activate_cuda_toolkit(cast(TorchModule, torch))
    try:
        from vllm import LLM, SamplingParams  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("vLLM is unavailable or not aligned with the active PyTorch CUDA build") from error
    from ...vllm_compat import apply_vllm_compatibility_patches

    apply_vllm_compatibility_patches()
    return LLM, SamplingParams


def _shutdown_llm(llm: Any) -> None:
    engine = getattr(llm, "llm_engine", None)
    client = getattr(engine, "engine_core", None)
    shutdown = getattr(client, "shutdown", None)
    if callable(shutdown):
        shutdown()


def _duration(start: float, finish: float) -> float | None:
    if start <= 0 or finish < start:
        return None
    return float(finish - start)


def _point_failure(
    error: Exception,
    *,
    sweep_index: int,
    concurrency: int,
) -> BenchmarkPointFailure:
    error_class = type(error).__name__
    message = " ".join(str(error).split())[:500] or error_class
    normalized = f"{error_class} {message}".lower()
    if any(marker in normalized for marker in ("out of memory", "oom", "resource exhausted")):
        status = "resource_exhausted"
    elif any(marker in normalized for marker in ("unsupported", "not implemented", "not supported")):
        status = "unsupported"
    else:
        status = "failed"
    return BenchmarkPointFailure(
        sweep_index=sweep_index,
        concurrency=concurrency,
        status=status,
        error_class=error_class,
        message=message,
        recoverable=True,
    )


def run_offline_benchmark(
    request: VllmBenchmarkConfig,
    *,
    phase: BenchmarkPhase = _unobserved_phase,
    metric: BenchmarkMetric = _unobserved_metric,
) -> BenchmarkSweepResult:
    """Measure an ordered concurrency sweep with one vLLM engine lifecycle."""

    context_windows = {cell.context_window for cell in request.cells}
    if len(context_windows) != 1:
        raise ValueError("one benchmark sweep requires one context allocation")
    engine = replace(request.engine, max_model_len=request.cells[0].context_window)
    memory = _GpuMemoryMonitor()
    llm: Any | None = None
    kv_cache_tracker = _KvCacheTracker(metric)
    kv_cache_capacity_tokens: int | None = None
    results: list[BenchmarkResult] = []
    point_failures: list[BenchmarkPointFailure] = []
    termination_reason: Literal[
        "configured_sweep_complete",
        "resource_exhausted_boundary",
        "unsupported_boundary",
        "failed_boundary",
    ] = "configured_sweep_complete"
    memory.start()
    try:
        with phase("model_loading", {"backend": "vllm"}):
            LLM, SamplingParams = _load_vllm()
            engine_started = time.perf_counter()
            active_llm = LLM(
                model=request.model.base.repo_id,
                revision=request.model.base.revision,
                **engine.as_vllm_kwargs(),
            )
            llm = active_llm
            engine_start_seconds = time.perf_counter() - engine_started
            cache_config = active_llm.llm_engine.vllm_config.cache_config
            capacity = getattr(cache_config, "kv_cache_size_tokens", None)
            if isinstance(capacity, int) and capacity > 0:
                kv_cache_capacity_tokens = capacity
            logger_manager = getattr(active_llm.llm_engine, "logger_manager", None)
            if logger_manager is not None:
                logger_manager.stat_loggers.append(kv_cache_tracker)

        with phase("runtime_initialization", {"backend": "vllm"}):
            tokenizer = active_llm.get_tokenizer()
        for sweep_index, cell in enumerate(request.cells):
            try:
                results.append(
                    _measure_cell(
                        active_llm,
                        SamplingParams,
                        tokenizer,
                        request,
                        cell,
                        sweep_index=sweep_index,
                        engine_start_seconds=engine_start_seconds,
                        kv_cache_capacity_tokens=kv_cache_capacity_tokens,
                        kv_cache_tracker=kv_cache_tracker,
                        memory=memory,
                        phase=phase,
                    )
                )
            except Exception as error:
                if not results:
                    raise
                failure = _point_failure(
                    error,
                    sweep_index=sweep_index,
                    concurrency=cell.concurrency,
                )
                point_failures.append(failure)
                if failure.status == "resource_exhausted":
                    termination_reason = "resource_exhausted_boundary"
                elif failure.status == "unsupported":
                    termination_reason = "unsupported_boundary"
                else:
                    termination_reason = "failed_boundary"
                break
    finally:
        with phase("runtime_cleanup", {"backend": "vllm"}):
            if llm is not None:
                _shutdown_llm(llm)
            memory.stop()
    return BenchmarkSweepResult(
        points=tuple(results),
        configured_concurrencies=tuple(cell.concurrency for cell in request.cells),
        point_failures=tuple(point_failures),
        termination_reason=termination_reason,
    )


def _measure_cell(
    active_llm: Any,
    sampling_type: Any,
    tokenizer: Any,
    request: VllmBenchmarkConfig,
    cell: BenchmarkCell,
    *,
    sweep_index: int,
    engine_start_seconds: float,
    kv_cache_capacity_tokens: int | None,
    kv_cache_tracker: _KvCacheTracker,
    memory: _GpuMemoryMonitor,
    phase: BenchmarkPhase,
) -> BenchmarkResult:
    sampling = replace(
        request.sampling,
        max_tokens=cell.output_tokens,
        min_tokens=cell.output_tokens,
        ignore_eos=True,
    )
    point_attributes = {
        "backend": "vllm",
        "sweep_index": sweep_index,
        "cell_id": cell.id,
        "concurrency": cell.concurrency,
    }
    with phase("benchmark_point_setup", point_attributes):
        params = sampling_type(**sampling.as_vllm_kwargs())
        if request.cohort == "representative":
            prompt_batches, measured_record_ids = _representative_prompt_batches(tokenizer, request, cell)
        else:
            if cell.input_tokens is None:
                raise RuntimeError("controlled benchmark requires an exact input token count")
            prompts = _controlled_prompt_ids(tokenizer, cell.input_tokens, cell.concurrency)
            prompt_batches = [prompts] * cell.iterations
            measured_record_ids = ()

    def generate(prompts: list[dict[str, list[int]]]) -> Any:
        return active_llm.generate(prompts, sampling_params=params, use_tqdm=False)

    with phase(
        "benchmark_warmup",
        {**point_attributes, "iterations": cell.warmup_iterations},
    ):
        for _ in range(cell.warmup_iterations):
            generate(prompt_batches[0])

    latencies: list[float] = []
    ttfts: list[float] = []
    tpots: list[float] = []
    input_lengths: list[int] = []
    total_input_tokens = 0
    total_output_tokens = 0
    samples: list[str] = []
    request_results: list[InferenceRequestResult] = []
    record_offset = 0
    point_peak_memory_gib: float | None = None
    with phase(
        "benchmark_measurement",
        {
            **point_attributes,
            "iterations": cell.iterations,
            "cohort": request.cohort,
        },
    ):
        memory.start_window()
        kv_cache_tracker.start(
            {
                "sweep_index": sweep_index,
                "cell_id": cell.id,
                "concurrency": cell.concurrency,
            }
        )
        try:
            started = time.perf_counter()
            for iteration, prompts in enumerate(prompt_batches):
                batch_started = time.perf_counter()
                outputs = generate(prompts)
                latencies.append(time.perf_counter() - batch_started)
                batch_input_lengths = [len(output.prompt_token_ids) for output in outputs]
                input_lengths.extend(batch_input_lengths)
                total_input_tokens += sum(batch_input_lengths)
                total_output_tokens += sum(len(output.outputs[0].token_ids) for output in outputs)
                for output_index, output in enumerate(outputs):
                    request_metrics = output.metrics
                    if request_metrics is not None and request_metrics.first_token_latency > 0:
                        ttfts.append(float(request_metrics.first_token_latency))
                    generated = len(output.outputs[0].token_ids)
                    tpot = None
                    if (
                        request_metrics is not None
                        and generated > 1
                        and request_metrics.last_token_ts > request_metrics.first_token_ts
                    ):
                        tpot = float(request_metrics.last_token_ts - request_metrics.first_token_ts) / (generated - 1)
                        tpots.append(tpot)
                    record_id = measured_record_ids[record_offset + output_index] if measured_record_ids else None
                    request_results.append(
                        InferenceRequestResult(
                            request_id=str(getattr(output, "request_id", f"{iteration}:{output_index}")),
                            record_id=record_id,
                            input_tokens=len(output.prompt_token_ids),
                            output_tokens=generated,
                            queue_seconds=(
                                _duration(request_metrics.queued_ts, request_metrics.scheduled_ts)
                                if request_metrics is not None
                                else None
                            ),
                            prefill_seconds=(
                                _duration(request_metrics.scheduled_ts, request_metrics.first_token_ts)
                                if request_metrics is not None
                                else None
                            ),
                            decode_seconds=(
                                _duration(request_metrics.first_token_ts, request_metrics.last_token_ts)
                                if request_metrics is not None
                                else None
                            ),
                            engine_e2e_seconds=(
                                _duration(request_metrics.queued_ts, request_metrics.last_token_ts)
                                if request_metrics is not None
                                else None
                            ),
                            ttft_seconds=(
                                float(request_metrics.first_token_latency)
                                if request_metrics is not None and request_metrics.first_token_latency > 0
                                else None
                            ),
                            tpot_seconds=tpot,
                        )
                    )
                record_offset += len(outputs)
                if request.cohort == "controlled" and iteration == 0:
                    samples.extend(output.outputs[0].text for output in outputs)
            elapsed = time.perf_counter() - started
        finally:
            kv_cache_tracker.stop()
            point_peak_memory_gib = memory.stop_window()
    request_count = cell.concurrency * cell.iterations
    return BenchmarkResult(
        backend="vllm",
        model=request.model.base.repo_id,
        revision=request.model.base.revision,
        inference_binding_id=request.inference_binding_id,
        suite_id=cell.suite_id,
        cell_id=cell.id,
        shape_id=cell.shape_id,
        context_window=cell.context_window,
        concurrency=cell.concurrency,
        target_input_tokens=cell.input_tokens,
        target_output_tokens=cell.output_tokens,
        requests=request_count,
        iterations=cell.iterations,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        engine_start_seconds=engine_start_seconds,
        elapsed_seconds=elapsed,
        request_throughput=request_count / elapsed,
        input_token_throughput=total_input_tokens / elapsed,
        output_token_throughput=total_output_tokens / elapsed,
        total_token_throughput=(total_input_tokens + total_output_tokens) / elapsed,
        mean_batch_latency=statistics.fmean(latencies),
        mean_ttft=statistics.fmean(ttfts) if ttfts else None,
        p50_ttft=_percentile(ttfts, 0.50),
        p95_ttft=_percentile(ttfts, 0.95),
        mean_tpot=statistics.fmean(tpots) if tpots else None,
        p50_tpot=_percentile(tpots, 0.50),
        p95_tpot=_percentile(tpots, 0.95),
        baseline_gpu_memory_gib=(memory.baseline_bytes / 1024**3 if memory.baseline_bytes is not None else None),
        peak_gpu_memory_gib=point_peak_memory_gib,
        peak_gpu_memory_delta_gib=(
            max(0.0, point_peak_memory_gib - memory.baseline_bytes / 1024**3)
            if point_peak_memory_gib is not None and memory.baseline_bytes is not None
            else None
        ),
        samples=tuple(samples),
        cohort=request.cohort,
        corpus_id=request.corpus.manifest.id if request.corpus is not None else None,
        corpus_revision=request.corpus.manifest.revision if request.corpus is not None else None,
        corpus_digest=request.corpus.manifest.digest if request.corpus is not None else None,
        corpus_records_measured=len(set(measured_record_ids)) if measured_record_ids else None,
        input_tokens_mean=statistics.fmean(input_lengths),
        input_tokens_p95=_percentile(input_lengths, 0.95),
        request_results=tuple(request_results),
        kv_cache_capacity_tokens=kv_cache_capacity_tokens,
        kv_cache_peak_usage_ratio=(kv_cache_tracker.peak_usage_ratio if kv_cache_tracker.samples else None),
    )
