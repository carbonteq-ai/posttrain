"""Tests for pinned foundation model variants."""

from posttrain.common.variants import (
    GEMMA_4_12B_IT,
    GEMMA_4_31B_IT,
    GEMMA_4_E2B_IT,
    GEMMA_4_E4B_IT,
    LFM_25_12B_THINKING,
    QWEN35_THINKING_RENDERER_CONTRACT,
    QWEN_35_08B,
    QWEN_35_2B,
)


def test_foundation_variants_publish_explicit_model_and_renderer_contracts() -> None:
    assert LFM_25_12B_THINKING.family == "lfm2.5"
    assert LFM_25_12B_THINKING.capabilities.native_context_window == 32_768
    assert LFM_25_12B_THINKING.renderer.id == "lfm2.5-tools@1"
    assert QWEN_35_2B.capabilities.native_context_window == 262_144
    assert QWEN_35_2B.renderer.model_family == QWEN_35_2B.family
    assert QWEN_35_2B.renderer.id == "qwen3.5-tools@1"
    assert QWEN_35_08B.parameters == 800_000_000
    assert QWEN_35_08B.base.revision == "2fc06364715b967f1860aea9cf38778875588b17"
    assert QWEN_35_08B.capabilities.mtp is True
    assert GEMMA_4_12B_IT.family == "gemma4"
    assert GEMMA_4_12B_IT.parameters == 11_959_730_224
    assert GEMMA_4_12B_IT.base.revision == "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
    assert GEMMA_4_12B_IT.capabilities.modalities == ("text", "image", "audio")
    assert GEMMA_4_12B_IT.capabilities.mtp is True
    assert GEMMA_4_12B_IT.provenance["mtp_mode"] == "paired-assistant"
    assert GEMMA_4_12B_IT.provenance["mtp_assistant_revision"] == "364bd03c9952e5b7da73665ee30c9eccfc408345"
    assert GEMMA_4_12B_IT.renderer.id == "gemma4-tools@1"
    assert GEMMA_4_12B_IT.provenance["upstream_model_type"] == "gemma4_unified"
    assert GEMMA_4_12B_IT.provenance["upstream_architecture"] == "Gemma4UnifiedForConditionalGeneration"


def test_gemma4_dense_matrix_is_pinned_and_mtp_ready() -> None:
    expected = (
        (GEMMA_4_E2B_IT, "google/gemma-4-E2B-it", 5_123_178_051, 131_072),
        (GEMMA_4_E4B_IT, "google/gemma-4-E4B-it", 7_996_156_490, 131_072),
        (GEMMA_4_12B_IT, "google/gemma-4-12B-it", 11_959_730_224, 262_144),
        (GEMMA_4_31B_IT, "google/gemma-4-31B-it", 31_273_088_876, 262_144),
    )
    for variant, repo_id, parameters, context_window in expected:
        assert variant.base.repo_id == repo_id
        assert len(variant.base.revision) == 40
        assert variant.parameters == parameters
        assert variant.capabilities.native_context_window == context_window
        assert variant.capabilities.mtp is True
        assert variant.provenance["mtp_mode"] == "paired-assistant"
        assert variant.provenance["mtp_assistant_repo_id"] == f"{repo_id}-assistant"
        assistant_revision = variant.provenance["mtp_assistant_revision"]
        assert isinstance(assistant_revision, str)
        assert len(assistant_revision) == 40
        assert variant.provenance["upstream_architecture"] in {
            "Gemma4ForConditionalGeneration",
            "Gemma4UnifiedForConditionalGeneration",
        }


def test_qwen_thinking_renderer_contract_is_explicit() -> None:
    assert QWEN35_THINKING_RENDERER_CONTRACT.id == "qwen3.5-tools-thinking@1"
    assert QWEN35_THINKING_RENDERER_CONTRACT.conversation.default_reasoning_mode == "thinking"
    assert QWEN35_THINKING_RENDERER_CONTRACT.conversation.reasoning_mode("thinking").kwargs() == {
        "enable_thinking": True
    }
