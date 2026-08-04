"""Tests for pinned foundation model variants and renderer contracts."""

from posttrain.common.variants import (
    GEMMA4_RENDERER_CONTRACT,
    LFM_25_12B_THINKING,
    QWEN_35_08B,
    QWEN_35_2B,
    RENDERER_CONTRACTS,
)


def test_gemma4_renderer_contract_is_registered_without_project_imports() -> None:
    assert RENDERER_CONTRACTS["gemma4-tools@1"] is GEMMA4_RENDERER_CONTRACT
    assert GEMMA4_RENDERER_CONTRACT.model_family == "gemma4"
    assert GEMMA4_RENDERER_CONTRACT.conversation.chat_template.source == "tokenizer"
    assert GEMMA4_RENDERER_CONTRACT.conversation.roles == ("system", "user", "assistant", "tool")
    assert GEMMA4_RENDERER_CONTRACT.conversation.reasoning_mode("off").kwargs() == {"enable_thinking": False}
    assert GEMMA4_RENDERER_CONTRACT.conversation.reasoning_mode("thinking").kwargs() == {"enable_thinking": True}
    assert GEMMA4_RENDERER_CONTRACT.conversation.tool_calls is None
    assert GEMMA4_RENDERER_CONTRACT.conversation.strips_past_reasoning is False


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
