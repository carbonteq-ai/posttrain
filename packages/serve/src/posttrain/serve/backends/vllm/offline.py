"""Offline vLLM benchmark adapter with no job or observability dependency."""

from __future__ import annotations

import statistics
import threading
import time
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, cast

from posttrain.common.cuda import TorchModule, activate_cuda_toolkit

from ...requests import BenchmarkRequest
from ...results import BenchmarkResult


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
                    self.peak_bytes = max(self.peak_bytes or 0, self._sample())
                except self._nvml.NVMLError:
                    return

        self._thread = threading.Thread(target=sample_until_stopped, daemon=True)
        self._thread.start()

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


def run_offline_benchmark(request: BenchmarkRequest) -> BenchmarkResult:
    """Measure exact token shapes; engine startup is excluded from steady-state metrics."""

    LLM, SamplingParams = _load_vllm()
    engine = replace(request.profile.engine, max_model_len=request.cell.context_window)
    sampling = replace(
        request.profile.sampling,
        max_tokens=request.cell.output_tokens,
        min_tokens=request.cell.output_tokens,
        ignore_eos=True,
    )
    memory = _GpuMemoryMonitor()
    memory.start()
    try:
        engine_started = time.perf_counter()
        llm = LLM(
            model=request.model.artifact.repo_id,
            revision=request.model.artifact.revision,
            **engine.as_vllm_kwargs(),
        )
        engine_start_seconds = time.perf_counter() - engine_started
        params = SamplingParams(**sampling.as_vllm_kwargs())
        prompts = _controlled_prompt_ids(llm.get_tokenizer(), request.cell.input_tokens, request.cell.concurrency)

        def generate() -> Any:
            return llm.generate(prompts, sampling_params=params, use_tqdm=False)

        for _ in range(request.cell.warmup_iterations):
            generate()
        latencies: list[float] = []
        ttfts: list[float] = []
        tpots: list[float] = []
        total_input_tokens = 0
        total_output_tokens = 0
        samples: list[str] = []
        started = time.perf_counter()
        for iteration in range(request.cell.iterations):
            batch_started = time.perf_counter()
            outputs = generate()
            latencies.append(time.perf_counter() - batch_started)
            total_input_tokens += sum(len(output.prompt_token_ids) for output in outputs)
            total_output_tokens += sum(len(output.outputs[0].token_ids) for output in outputs)
            for output in outputs:
                metrics = output.metrics
                if metrics is not None and metrics.first_token_latency > 0:
                    ttfts.append(float(metrics.first_token_latency))
                generated = len(output.outputs[0].token_ids)
                if metrics is not None and generated > 1 and metrics.last_token_ts > metrics.first_token_ts:
                    tpots.append(float(metrics.last_token_ts - metrics.first_token_ts) / (generated - 1))
            if iteration == 0:
                samples.extend(output.outputs[0].text for output in outputs)
        elapsed = time.perf_counter() - started
    finally:
        baseline_memory, peak_memory, memory_delta = memory.stop()
    request_count = request.cell.concurrency * request.cell.iterations
    return BenchmarkResult(
        backend="vllm",
        model=request.model.artifact.repo_id,
        revision=request.model.artifact.revision,
        profile_id=request.profile.id,
        suite_id=request.cell.suite_id,
        cell_id=request.cell.id,
        shape_id=request.cell.shape_id,
        context_window=request.cell.context_window,
        concurrency=request.cell.concurrency,
        target_input_tokens=request.cell.input_tokens,
        target_output_tokens=request.cell.output_tokens,
        requests=request_count,
        iterations=request.cell.iterations,
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
        baseline_gpu_memory_gib=baseline_memory,
        peak_gpu_memory_gib=peak_memory,
        peak_gpu_memory_delta_gib=memory_delta,
        samples=tuple(samples),
    )
