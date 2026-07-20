from __future__ import annotations

import pytest
from posttrain.common import ModelProfile
from posttrain.common.profiles import LFM_25_12B_THINKING, QWEN_35_2B
from posttrain.serve.benchmarks import CORE_INFERENCE_V1
from posttrain.serve.profiles import (
    LFM25_VLLM,
    LFM25_VLLM_TURBOQUANT_K8,
    QWEN35_VLLM_MTP,
    QWEN35_VLLM_TEXT,
    QWEN35_VLLM_TURBOQUANT_K8,
    VllmEngineConfig,
    VllmServeProfile,
)


def test_qwen_text_profile_captures_tested_8gb_constraints() -> None:
    profile = QWEN35_VLLM_TEXT
    profile.validate_model(QWEN_35_2B)
    kwargs = profile.engine.as_vllm_kwargs()
    assert kwargs["enforce_eager"] is True
    assert kwargs["gpu_memory_utilization"] == 0.75
    assert kwargs["limit_mm_per_prompt"] == {"image": 0, "video": 0}
    assert kwargs["skip_mm_profiling"] is True
    assert kwargs["max_num_seqs"] == 4
    assert profile.frontend_args() == (
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_xml",
        "--reasoning-parser",
        "qwen3",
    )


def test_lfm_profile_uses_native_tool_parser_and_tag_compatible_reasoning_parser() -> None:
    assert LFM_25_12B_THINKING.conversation.tool_calls is not None
    assert LFM_25_12B_THINKING.conversation.tool_calls.id == "lfm2_pythonic"
    assert LFM25_VLLM.frontend_args() == (
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "lfm2",
        "--reasoning-parser",
        "deepseek_r1",
    )


@pytest.mark.parametrize(
    ("profile", "model"),
    [
        (QWEN35_VLLM_TURBOQUANT_K8, QWEN_35_2B),
        (QWEN35_VLLM_MTP, QWEN_35_2B),
        (LFM25_VLLM, LFM_25_12B_THINKING),
        (LFM25_VLLM_TURBOQUANT_K8, LFM_25_12B_THINKING),
    ],
)
def test_published_profiles_match_their_models(profile: VllmServeProfile, model: ModelProfile) -> None:
    profile.validate_model(model)


def test_skip_mm_profiling_requires_text_only_mode() -> None:
    with pytest.raises(ValueError, match="text-only"):
        VllmEngineConfig(
            max_model_len=1_024,
            gpu_memory_utilization=0.75,
            skip_mm_profiling=True,
        )


def test_local_matrix_stops_at_concurrency_four_and_requires_turboquant_at_32k() -> None:
    cells = CORE_INFERENCE_V1.cells(max_concurrency=4)
    assert {cell.concurrency for cell in cells} == {1, 2, 4}
    assert {cell.context_window for cell in cells} == {1_024, 2_048, 4_096, 8_192, 16_384, 32_768}
    assert all(cell.required_variant == "turboquant" for cell in cells if cell.context_window == 32_768)
    assert all(cell.required_variant is None for cell in cells if cell.context_window < 32_768)
