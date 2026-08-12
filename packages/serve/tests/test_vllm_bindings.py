"""Tests for vLLM inference-binding translation."""

from dataclasses import replace

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


def test_qwen_tool_capability_uses_its_declared_xml_protocol(qwen_screen_binding: InferenceBinding) -> None:
    binding = replace(
        qwen_screen_binding,
        capabilities=("tool-calling",),
        engine={**qwen_screen_binding.engine, "reasoning_parser": "qwen3"},
    )

    assert binding.model.conversation.tool_calls is not None
    assert binding.model.conversation.tool_calls.id == "qwen3_xml"
    assert frontend_args(binding) == (
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_xml",
        "--reasoning-parser",
        "qwen3",
    )


def test_vllm_rejects_parser_override_that_conflicts_with_model_protocol(
    qwen_screen_binding: InferenceBinding,
) -> None:
    binding = replace(
        qwen_screen_binding,
        capabilities=("tool-calling",),
        engine={**qwen_screen_binding.engine, "tool_call_parser": "hermes"},
    )

    with pytest.raises(ValueError, match="conflicts with the selected model tool-call protocol"):
        frontend_args(binding)


def test_skip_mm_profiling_requires_text_only_mode() -> None:
    with pytest.raises(ValueError, match="text-only"):
        VllmEngineConfig(max_model_len=1_024, gpu_memory_utilization=0.75, skip_mm_profiling=True)


def test_request_level_structured_output_whitespace_is_not_a_server_cli_flag() -> None:
    config = VllmEngineConfig(
        max_model_len=1_024,
        gpu_memory_utilization=0.75,
        structured_outputs_whitespace_pattern=r" ?",
    )

    assert config.structured_outputs_whitespace_pattern == r" ?"
    assert "structured_outputs_whitespace_pattern" not in config.as_vllm_kwargs()
    assert "--structured-outputs-whitespace-pattern" not in config.as_cli_args()


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
