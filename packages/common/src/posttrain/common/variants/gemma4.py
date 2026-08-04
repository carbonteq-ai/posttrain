"""Gemma 4 tokenizer-backed conversation contract."""

from posttrain.common.models import ChatTemplate, ConversationProfile, ReasoningMode, RendererContract

GEMMA4_RENDERER_CONTRACT = RendererContract(
    id="gemma4-tools@1",
    model_family="gemma4",
    conversation=ConversationProfile(
        chat_template=ChatTemplate("tokenizer"),
        roles=("system", "user", "assistant", "tool"),
        reasoning_modes=(
            ReasoningMode("off", (("enable_thinking", False),)),
            ReasoningMode("thinking", (("enable_thinking", True),)),
        ),
        default_reasoning_mode="off",
        # Callers do not need every feature described by a renderer contract.
        # Keep structured tool calls unset until common owns a Gemma-native
        # ToolCallProtocol identifier.
        tool_calls=None,
        strips_past_reasoning=False,
    ),
)

__all__ = ["GEMMA4_RENDERER_CONTRACT"]
