"""Tests for vLLM inference-binding translation."""

import pytest
from posttrain.common import InferenceBinding, Workload
from posttrain.serve import ServeBenchmarkRequest
from posttrain.serve.backends.vllm.bindings import benchmark_config, engine_config, frontend_args
from posttrain.serve.benchmarks import CORE_INFERENCE_V1
from posttrain.serve.profiles import VllmEngineConfig


def test_qwen_screen_binding_captures_tested_8gb_constraints(qwen_screen_binding: InferenceBinding) -> None:
    binding = qwen_screen_binding
    kwargs = engine_config(binding).as_vllm_kwargs()
    assert kwargs["enforce_eager"] is True
    assert kwargs["gpu_memory_utilization"] == 0.75
    assert kwargs["limit_mm_per_prompt"] == {"image": 0, "video": 0, "audio": 0}
    assert kwargs["skip_mm_profiling"] is True
    assert kwargs["max_num_seqs"] == 4


def test_lfm_binding_uses_model_renderer_and_frontend_parsers(lfm_screen_binding: InferenceBinding) -> None:
    binding = lfm_screen_binding
    assert binding.renderer == binding.model.renderer.id
    assert frontend_args(binding) == (
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "lfm2",
        "--reasoning-parser",
        "lfm2",
    )


def test_skip_mm_profiling_requires_text_only_mode() -> None:
    with pytest.raises(ValueError, match="text-only"):
        VllmEngineConfig(max_model_len=1_024, gpu_memory_utilization=0.75, skip_mm_profiling=True)


@pytest.mark.parametrize("max_lora_rank", [0, -1])
def test_max_lora_rank_must_be_positive(max_lora_rank: int) -> None:
    with pytest.raises(ValueError, match="max_lora_rank must be positive"):
        VllmEngineConfig(
            max_model_len=1_024,
            gpu_memory_utilization=0.75,
            max_lora_rank=max_lora_rank,
        )


def test_max_lora_rank_translates_to_vllm_kwargs_and_cli() -> None:
    engine = VllmEngineConfig(
        max_model_len=1_024,
        gpu_memory_utilization=0.75,
        max_lora_rank=32,
    )

    assert engine.as_vllm_kwargs()["max_lora_rank"] == 32
    command = engine.as_cli_args()
    assert command[command.index("--max-lora-rank") + 1] == "32"


def test_local_matrix_stops_at_concurrency_four_and_requires_turboquant_at_32k() -> None:
    cells = CORE_INFERENCE_V1.cells(max_concurrency=4)
    assert {cell.concurrency for cell in cells} == {1, 2, 4}
    assert all(cell.required_variant == "turboquant" for cell in cells if cell.context_window == 32_768)


def test_representative_workload_resolves_and_verifies_packaged_corpus(
    qwen_screen_binding: InferenceBinding,
    representative_workload: Workload,
) -> None:
    config = benchmark_config(ServeBenchmarkRequest(qwen_screen_binding, representative_workload))

    assert config.cohort == "representative"
    assert config.cells[0].input_tokens is None
    assert config.corpus is not None
    assert config.corpus.manifest.id == "general-serving-v1"
    assert config.corpus.manifest.record_count == 128
    assert config.selection_seed == 17


def test_workload_concurrency_becomes_one_ordered_sweep(
    qwen_screen_binding: InferenceBinding,
    representative_workload: Workload,
) -> None:
    workload = Workload(
        id=representative_workload.id,
        revision=representative_workload.revision,
        requests=representative_workload.requests,
        concurrency=(1, 2, 4),
        warmup_repetitions=representative_workload.warmup_repetitions,
        measured_repetitions=representative_workload.measured_repetitions,
    )

    config = benchmark_config(ServeBenchmarkRequest(qwen_screen_binding, workload))

    assert tuple(cell.concurrency for cell in config.cells) == (1, 2, 4)
