"""Profile-driven, Trackio-observed offline inference benchmark."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, cast

from common import PROFILES_DIR, ProfileResolver, TrackedRun

from .cuda import TorchModule, resolve_cuda_home


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    backend: str
    model: str
    revision: str
    suite_id: str | None
    suite_execution_id: str | None
    case_id: str | None
    shape_id: str | None
    context_window: int
    concurrency: int
    prompt_source: str
    reasoning_mode: str
    target_input_tokens: int | None
    target_output_tokens: int | None
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
        for name, value in (
            ("serve/mean_ttft", self.mean_ttft),
            ("serve/p50_ttft", self.p50_ttft),
            ("serve/p95_ttft", self.p95_ttft),
            ("serve/mean_tpot", self.mean_tpot),
            ("serve/p50_tpot", self.p50_tpot),
            ("serve/p95_tpot", self.p95_tpot),
        ):
            if value is not None:
                values[name] = value
        if self.baseline_gpu_memory_gib is not None:
            values["serve/baseline_gpu_memory_gib"] = self.baseline_gpu_memory_gib
        if self.peak_gpu_memory_gib is not None:
            values["serve/peak_gpu_memory_gib"] = self.peak_gpu_memory_gib
        if self.peak_gpu_memory_delta_gib is not None:
            values["serve/peak_gpu_memory_delta_gib"] = self.peak_gpu_memory_delta_gib
        return values


def _hf_target(artifact: str) -> tuple[str, str]:
    if not artifact.startswith("hf://"):
        raise ValueError(f"vLLM benchmark requires an hf:// artifact, got {artifact!r}")
    repository, separator, revision = artifact.removeprefix("hf://").rpartition("@")
    if not separator or not repository or not revision:
        raise ValueError(f"Hugging Face artifact must include an immutable revision: {artifact!r}")
    return repository, revision


def _default_prompts() -> list[str]:
    return [
        "Explain why the sky appears blue in two concise sentences.",
        "A shop has 18 apples and sells 7. How many remain? Give only the answer.",
        "Write a Python function signature for adding two integers.",
        "Name the capital of Japan and one landmark there.",
    ]


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile + 0.5)))
    return ordered[index]


def _controlled_prompt_ids(
    tokenizer: Any,
    target_tokens: int,
    concurrency: int,
) -> list[dict[str, list[int]]]:
    """Build deterministic, exact-length token prompts for systems benchmarks."""

    if target_tokens < 1 or concurrency < 1:
        raise ValueError("target_tokens and concurrency must be positive")
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
    """Sample whole-device memory because vLLM allocates in a worker process."""

    def __init__(self, device_index: int, interval_seconds: float = 0.02) -> None:
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
            return
        try:
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self.baseline_bytes = self._sample()
            self.peak_bytes = self.baseline_bytes
        except pynvml.NVMLError:
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


def _load_vllm() -> tuple[Any, Any, Any]:
    """Load the CUDA-aligned vLLM runtime with a useful setup error."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is not installed; sync the serve package with its vLLM extra") from error

    cuda_home = resolve_cuda_home(cast(TorchModule, torch))
    os.environ["CUDA_HOME"] = str(cuda_home)
    path_entries = [entry for entry in os.environ.get("PATH", "").split(":") if entry]
    toolkit_bin = str(cuda_home / "bin")
    if toolkit_bin not in path_entries:
        os.environ["PATH"] = ":".join([toolkit_bin, *path_entries])

    try:
        from vllm import LLM, SamplingParams  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError(
            "vLLM is not installed or its CUDA wheel is not aligned with PyTorch; "
            "run `uv sync --package serve --extra vllm --python 3.12`"
        ) from error
    from .vllm_compat import apply_vllm_compatibility_patches

    apply_vllm_compatibility_patches()
    return torch, LLM, SamplingParams


def run_vllm_benchmark(
    model: str,
    revision: str,
    *,
    engine_config: dict[str, Any],
    prompts: Sequence[str] | None,
    sampling_config: dict[str, Any],
    controlled_input_tokens: int | None = None,
    concurrency: int | None = None,
    suite_id: str | None = None,
    suite_execution_id: str | None = None,
    case_id: str | None = None,
    shape_id: str | None = None,
    reasoning_mode: str = "native",
    warmup_iterations: int = 1,
    iterations: int = 3,
) -> BenchmarkResult:
    """Run batched offline generation and calculate portable throughput metrics."""

    if controlled_input_tokens is None and not prompts:
        raise ValueError("at least one prompt is required")
    if controlled_input_tokens is not None and concurrency is None:
        raise ValueError("controlled token benchmarks require concurrency")
    if iterations < 1 or warmup_iterations < 0:
        raise ValueError("iterations must be >= 1 and warmup_iterations must be >= 0")

    torch, LLM, SamplingParams = _load_vllm()

    memory = _GpuMemoryMonitor(torch.cuda.current_device())
    memory.start()
    try:
        engine_started = time.perf_counter()
        llm = LLM(model=model, revision=revision, **engine_config)
        engine_start_seconds = time.perf_counter() - engine_started
        sampling = SamplingParams(**sampling_config)
        if controlled_input_tokens is not None:
            requests: Sequence[Any] = _controlled_prompt_ids(
                llm.get_tokenizer(), controlled_input_tokens, concurrency or 1
            )
            generate = lambda: llm.generate(  # noqa: E731
                requests, sampling_params=sampling, use_tqdm=False
            )
            prompt_source = "controlled_tokens"
        else:
            conversations = [[{"role": "user", "content": prompt}] for prompt in prompts or ()]
            requests = conversations
            generate = lambda: llm.chat(  # noqa: E731
                conversations, sampling_params=sampling, use_tqdm=False
            )
            prompt_source = "representative_messages"

        for _ in range(warmup_iterations):
            generate()

        latencies: list[float] = []
        total_input_tokens = 0
        total_output_tokens = 0
        samples: list[str] = []
        ttfts: list[float] = []
        tpots: list[float] = []
        started = time.perf_counter()
        for iteration in range(iterations):
            batch_started = time.perf_counter()
            outputs = generate()
            latencies.append(time.perf_counter() - batch_started)
            total_input_tokens += sum(len(output.prompt_token_ids) for output in outputs)
            total_output_tokens += sum(len(output.outputs[0].token_ids) for output in outputs)
            for output in outputs:
                metrics = output.metrics
                if metrics is None:
                    continue
                if metrics.first_token_latency > 0:
                    ttfts.append(float(metrics.first_token_latency))
                generated = len(output.outputs[0].token_ids)
                if generated > 1 and metrics.last_token_ts > metrics.first_token_ts:
                    tpots.append(float(metrics.last_token_ts - metrics.first_token_ts) / (generated - 1))
            if iteration == 0:
                samples.extend(output.outputs[0].text for output in outputs)
        elapsed = time.perf_counter() - started
    finally:
        baseline_memory, peak_memory, memory_delta = memory.stop()
    request_count = len(requests) * iterations
    return BenchmarkResult(
        backend="vllm",
        model=model,
        revision=revision,
        suite_id=suite_id,
        suite_execution_id=suite_execution_id,
        case_id=case_id,
        shape_id=shape_id,
        context_window=int(engine_config["max_model_len"]),
        concurrency=len(requests),
        prompt_source=prompt_source,
        reasoning_mode=reasoning_mode,
        target_input_tokens=controlled_input_tokens,
        target_output_tokens=sampling_config.get("max_tokens"),
        requests=request_count,
        iterations=iterations,
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a model profile with vLLM.")
    parser.add_argument("model_profile", help="Reference under profiles/models")
    parser.add_argument("--serve-profile", help="Override the model's default vLLM profile")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--warmup-iterations", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--input-tokens", type=int)
    parser.add_argument("--output-tokens", type=int)
    parser.add_argument("--context-window", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--suite-id")
    parser.add_argument("--suite-execution-id")
    parser.add_argument("--case-id")
    parser.add_argument("--shape-id")
    parser.add_argument("--reasoning-mode", default="native")
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--name", help="Optional Trackio run name")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    resolver = ProfileResolver(PROFILES_DIR)
    model_profile = resolver.resolve("models", args.model_profile)
    serve_reference = args.serve_profile or model_profile.data["defaults"]["serve"]["vllm"]
    serve_profile = resolver.resolve("serve", serve_reference)
    config = serve_profile.data
    if config.get("backend") != "vllm":
        raise ValueError(f"serve profile {serve_reference!r} does not select vLLM")

    model, revision = _hf_target(model_profile.data["model"]["artifact"])
    engine_config = dict(config.get("engine", {}))
    engine_config.setdefault("disable_log_stats", False)
    workload = dict(config.get("workload", {}))
    sampling = dict(config.get("sampling", {}))
    iterations = args.iterations if args.iterations is not None else workload.get("iterations", 3)
    warmup = args.warmup_iterations if args.warmup_iterations is not None else workload.get("warmup_iterations", 1)
    if args.max_tokens is not None:
        sampling["max_tokens"] = args.max_tokens
    if args.output_tokens is not None:
        sampling.update(
            max_tokens=args.output_tokens,
            min_tokens=args.output_tokens,
            ignore_eos=True,
        )
    if args.context_window is not None:
        engine_config["max_model_len"] = args.context_window
    native_context_window = model_profile.data["model"]["capabilities"]["context_window"]
    configured_context_window = engine_config.get("max_model_len")
    if (
        not isinstance(configured_context_window, int)
        or isinstance(configured_context_window, bool)
        or configured_context_window < 1
    ):
        raise ValueError("serve profile must define a positive engine.max_model_len")
    if configured_context_window > native_context_window:
        raise ValueError(
            f"configured context ({configured_context_window}) exceeds the model's "
            f"native context ({native_context_window})"
        )
    concurrency = args.concurrency or workload.get("concurrency", 4)
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if args.input_tokens is not None:
        output_tokens = sampling.get("max_tokens")
        if not isinstance(output_tokens, int):
            raise ValueError("controlled benchmark requires --output-tokens")
        if args.input_tokens + output_tokens > engine_config["max_model_len"]:
            raise ValueError("input and output tokens exceed the configured context window")
        prompts = None
    else:
        source_prompts = args.prompts or _default_prompts()
        prompts = [source_prompts[index % len(source_prompts)] for index in range(concurrency)]
    resolved_config = {
        "model_profile": model_profile.reference,
        "serve_profile": serve_profile.reference,
        "backend": "vllm",
        "engine": engine_config,
        "sampling": sampling,
        "workload": {
            **workload,
            "iterations": iterations,
            "warmup_iterations": warmup,
            "prompt_count": concurrency,
            "concurrency": concurrency,
            "input_tokens": args.input_tokens,
            "output_tokens": sampling.get("max_tokens"),
        },
        "suite_id": args.suite_id,
        "suite_execution_id": args.suite_execution_id,
        "case_id": args.case_id,
        "shape_id": args.shape_id,
        "reasoning_mode": args.reasoning_mode,
    }

    with TrackedRun.start(
        "serving-benchmark",
        resolved_config,
        resolved_profile=model_profile,
        name=args.name,
    ) as run:
        result = run_vllm_benchmark(
            model,
            revision,
            engine_config=engine_config,
            prompts=prompts,
            sampling_config=sampling,
            controlled_input_tokens=args.input_tokens,
            concurrency=concurrency,
            suite_id=args.suite_id,
            suite_execution_id=args.suite_execution_id,
            case_id=args.case_id,
            shape_id=args.shape_id,
            reasoning_mode=args.reasoning_mode,
            warmup_iterations=warmup,
            iterations=iterations,
        )
        run.log(result.metrics())
        result_path = run.context.output_dir / "benchmark.json"
        result_path.write_text(
            json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        run.log_artifact(
            result_path,
            name=f"{run.name}-serving-benchmark",
            artifact_type="serving-benchmark",
            aliases=("latest",),
            metadata={"model": model, "revision": revision, "backend": "vllm"},
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
