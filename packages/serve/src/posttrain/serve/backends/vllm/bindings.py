"""Internal translation from framework selections to vLLM adapter config."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil
from typing import Any

from posttrain.common import InferenceBinding, JsonValue, ModelVariant

from ...benchmarks import BenchmarkCell
from ...profiles import VllmEngineConfig, VllmSamplingConfig, VllmSpeculativeConfig
from ...prompts import PromptCorpus, load_prompt_corpus
from ...requests import ServeBenchmarkRequest


@dataclass(frozen=True, slots=True)
class VllmBenchmarkConfig:
    model: ModelVariant
    inference_binding_id: str
    engine: VllmEngineConfig
    sampling: VllmSamplingConfig
    cells: tuple[BenchmarkCell, ...]
    cohort: str
    corpus: PromptCorpus | None
    selection_seed: int


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
    cohort = _optional_string(values, "cohort", "controlled")
    corpus: PromptCorpus | None = None
    selection_seed = 0
    if cohort == "representative":
        selection = values.get("corpus")
        if not isinstance(selection, Mapping):
            raise ValueError("representative benchmark workload requires a corpus selection")
        corpus_id = _mapping_string(selection, "id")
        corpus = load_prompt_corpus(corpus_id)
        if corpus.manifest.revision != _mapping_string(selection, "revision"):
            raise ValueError(f"prompt corpus revision mismatch for {corpus_id}")
        if corpus.manifest.digest != _mapping_string(selection, "digest"):
            raise ValueError(f"prompt corpus digest mismatch for {corpus_id}")
        selection_seed = _integer(values, "selection_seed")
    record_count = _optional_integer(values, "record_count")
    cells = tuple(
        BenchmarkCell(
            suite_id=_string(values, "suite_id"),
            shape_id=_string(values, "shape_id"),
            context_window=_integer(values, "context_window"),
            concurrency=concurrency,
            input_tokens=_integer(values, "input_tokens") if cohort == "controlled" else None,
            output_tokens=_integer(values, "output_tokens"),
            warmup_iterations=workload.warmup_repetitions,
            iterations=(
                ceil(record_count / concurrency) if record_count is not None else workload.measured_repetitions
            ),
            required_variant=variant if variant != "standard" else None,
        )
        for concurrency in workload.concurrency
    )
    if engine.max_num_seqs is not None and cells[-1].concurrency > engine.max_num_seqs:
        raise ValueError(f"this inference binding supports at most {engine.max_num_seqs} concurrent sequences")
    return VllmBenchmarkConfig(
        binding.model,
        binding.id,
        engine,
        sampling,
        cells,
        cohort,
        corpus,
        selection_seed,
    )


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


def _optional_integer(values: Mapping[str, JsonValue], name: str) -> int | None:
    value = values.get(name)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"benchmark workload {name} must be a positive integer")
    return value


def _string(values: Mapping[str, JsonValue], name: str) -> str:
    value = values[name]
    if not isinstance(value, str) or not value:
        raise ValueError(f"benchmark workload {name} must be a non-empty string")
    return value


def _optional_string(values: Mapping[str, JsonValue], name: str, default: str) -> str:
    value = values.get(name, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"benchmark workload {name} must be a non-empty string")
    return value


def _mapping_string(values: Mapping[str, object], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"benchmark corpus {name} must be a non-empty string")
    return value


__all__ = [
    "VllmBenchmarkConfig",
    "benchmark_config",
    "engine_config",
    "frontend_args",
    "sampling_config",
]
