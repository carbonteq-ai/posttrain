"""Internal translation from framework selections to vLLM adapter config."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from posttrain.common import InferenceBinding, JsonValue, ModelVariant

from ...benchmarks import BenchmarkCell
from ...profiles import VllmEngineConfig, VllmSamplingConfig, VllmSpeculativeConfig
from ...requests import ServeBenchmarkRequest


@dataclass(frozen=True, slots=True)
class VllmBenchmarkConfig:
    model: ModelVariant
    inference_binding_id: str
    engine: VllmEngineConfig
    sampling: VllmSamplingConfig
    cell: BenchmarkCell


def benchmark_config(request: ServeBenchmarkRequest) -> VllmBenchmarkConfig:
    binding = request.inference
    if not binding.backend.startswith("vllm@"):
        raise ValueError(f"unsupported serve.benchmark backend: {binding.backend!r}")
    engine = engine_config(binding)
    sampling = sampling_config(binding)
    variant = "standard"
    if engine.speculative is not None:
        variant = "mtp"
    elif engine.kv_cache_dtype != "auto":
        variant = "turboquant"
    workload = request.workload
    values: Mapping[str, JsonValue] = workload.requests
    cell = BenchmarkCell(
        suite_id=_string(values, "suite_id"),
        shape_id=_string(values, "shape_id"),
        context_window=_integer(values, "context_window"),
        concurrency=workload.concurrency[0],
        input_tokens=_integer(values, "input_tokens"),
        output_tokens=_integer(values, "output_tokens"),
        warmup_iterations=workload.warmup_repetitions,
        iterations=workload.measured_repetitions,
        required_variant=variant if variant != "standard" else None,
    )
    if cell.concurrency > 4 and engine.max_num_seqs == 4:
        raise ValueError("this inference binding supports at most four concurrent sequences")
    return VllmBenchmarkConfig(binding.model, binding.id, engine, sampling, cell)


def engine_config(binding: InferenceBinding) -> VllmEngineConfig:
    values: dict[str, Any] = dict(binding.engine)
    speculative = values.pop("speculative_config", values.pop("speculative", None))
    values.pop("tool_call_parser", None)
    values.pop("reasoning_parser", None)
    if isinstance(speculative, Mapping):
        values["speculative"] = VllmSpeculativeConfig(**dict(speculative))
    return VllmEngineConfig(**values)


def sampling_config(binding: InferenceBinding) -> VllmSamplingConfig:
    return VllmSamplingConfig(**dict[str, Any](binding.sampling))


def frontend_args(binding: InferenceBinding) -> tuple[str, ...]:
    values: list[str] = []
    tool_parser = binding.engine.get("tool_call_parser")
    reasoning_parser = binding.engine.get("reasoning_parser")
    if isinstance(tool_parser, str):
        values.extend(("--enable-auto-tool-choice", "--tool-call-parser", tool_parser))
    if isinstance(reasoning_parser, str):
        values.extend(("--reasoning-parser", reasoning_parser))
    return tuple(values)


def _integer(values: Mapping[str, JsonValue], name: str) -> int:
    value = values[name]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"benchmark workload {name} must be an integer")
    return value


def _string(values: Mapping[str, JsonValue], name: str) -> str:
    value = values[name]
    if not isinstance(value, str) or not value:
        raise ValueError(f"benchmark workload {name} must be a non-empty string")
    return value


__all__ = [
    "VllmBenchmarkConfig",
    "benchmark_config",
    "engine_config",
    "frontend_args",
    "sampling_config",
]
