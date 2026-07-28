"""Tests for pinned foundation model variants."""

from posttrain.common.variants import (
    GEMMA4_RENDERER_CONTRACT,
    GEMMA_4_31B_IT,
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


def test_gemma4_variants_use_pinned_native_renderer_and_tokenizer_template() -> None:
    assert GEMMA4_RENDERER_CONTRACT.id == "gemma4-native-v1"
    assert GEMMA4_RENDERER_CONTRACT.conversation.chat_template.source == "tokenizer"
    assert GEMMA4_RENDERER_CONTRACT.conversation.default_reasoning_mode == "native"
    assert GEMMA_4_E4B_IT.renderer is GEMMA4_RENDERER_CONTRACT
    assert GEMMA_4_E4B_IT.base.revision == "ee0ef6023621cff504d758262d4e04895a5af4a2"
    assert GEMMA_4_31B_IT.base.revision == "842da3794eaa0b77d5f08bae87a17459d91ff475"
    assert GEMMA_4_E4B_IT.capabilities.modalities == ("text", "image", "video", "audio")
    assert GEMMA_4_E4B_IT.tokenizer_fingerprint == GEMMA_4_31B_IT.tokenizer_fingerprint
    assert (
        GEMMA_4_E4B_IT.tokenizer_fingerprint
        == "1ab787c816b67a0936e8d1c9ff20e6cf5bd8b77faabfe6ada5905bd2c433b413"
    )
