"""Tests for pinned foundation model variants."""

from posttrain.common.variants import (
    GEMMA_4_12B_IT,
    GEMMA_4_E4B_IT,
    LFM_25_12B_THINKING,
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
    assert GEMMA_4_12B_IT.capabilities.modalities == ("text", "image", "audio", "video")
    assert GEMMA_4_12B_IT.capabilities.mtp is False
    assert GEMMA_4_12B_IT.renderer.id == "gemma4-tools@1"
    assert GEMMA_4_12B_IT.provenance["upstream_model_type"] == "gemma4_unified"
    assert GEMMA_4_12B_IT.provenance["upstream_architecture"] == "Gemma4UnifiedForConditionalGeneration"
    assert GEMMA_4_E4B_IT.family == "gemma4"
    assert GEMMA_4_E4B_IT.parameters == 7_996_156_490
    assert GEMMA_4_E4B_IT.base.repo_id == "google/gemma-4-E4B-it"
    assert GEMMA_4_E4B_IT.base.revision == "ee0ef6023621cff504d758262d4e04895a5af4a2"
    assert GEMMA_4_E4B_IT.capabilities.modalities == ("text", "image", "audio", "video")
    assert GEMMA_4_E4B_IT.capabilities.native_context_window == 131_072
    assert GEMMA_4_E4B_IT.renderer.id == "gemma4-e4b-tools@1"
    assert GEMMA_4_E4B_IT.renderer is not GEMMA_4_12B_IT.renderer
    assert GEMMA_4_E4B_IT.tokenizer_fingerprint == ("1ab787c816b67a0936e8d1c9ff20e6cf5bd8b77faabfe6ada5905bd2c433b413")
    assert GEMMA_4_E4B_IT.provenance["upstream_model_type"] == "gemma4"
    assert GEMMA_4_E4B_IT.provenance["upstream_architecture"] == "Gemma4ForConditionalGeneration"
