"""Tests for vLLM inference-binding translation."""

import pytest
from posttrain.common import InferenceBinding
from posttrain.serve.backends.vllm.bindings import engine_config, frontend_args
from posttrain.serve.benchmarks import CORE_INFERENCE_V1
from posttrain.serve.profiles import VllmEngineConfig


def test_qwen_screen_binding_captures_tested_8gb_constraints(qwen_screen_binding: InferenceBinding) -> None:
    binding = qwen_screen_binding
    kwargs = engine_config(binding).as_vllm_kwargs()
    assert kwargs["enforce_eager"] is True
    assert kwargs["gpu_memory_utilization"] == 0.75
    assert kwargs["limit_mm_per_prompt"] == {"image": 0, "video": 0}
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


def test_local_matrix_stops_at_concurrency_four_and_requires_turboquant_at_32k() -> None:
    cells = CORE_INFERENCE_V1.cells(max_concurrency=4)
    assert {cell.concurrency for cell in cells} == {1, 2, 4}
    assert all(cell.required_variant == "turboquant" for cell in cells if cell.context_window == 32_768)
