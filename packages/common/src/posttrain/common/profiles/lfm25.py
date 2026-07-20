"""Pinned LFM2.5 Thinking foundation profile."""

from posttrain.common.artifacts import HubModelRef
from posttrain.common.models import (
    ChatTemplate,
    ConversationProfile,
    ModelCapabilities,
    ModelProfile,
    ReasoningMode,
    ToolCallProtocol,
)

LFM_25_12B_THINKING = ModelProfile(
    id="lfm2.5-1.2b-thinking",
    artifact=HubModelRef(
        repo_id="LiquidAI/LFM2.5-1.2B-Thinking",
        revision="95053d21d8e0b7ca99421a2127ae39c64f685ff3",
    ),
    family="lfm2.5",
    parameters=1_170_000_000,
    instruction_tuned=True,
    capabilities=ModelCapabilities(
        modalities=("text",),
        native_context_window=32_768,
    ),
    conversation=ConversationProfile(
        chat_template=ChatTemplate("package", "lfm25_tool_chat.jinja"),
        roles=("system", "user", "assistant", "tool"),
        reasoning_modes=(ReasoningMode("native"),),
        default_reasoning_mode="native",
        tool_calls=ToolCallProtocol(
            id="lfm2_pythonic",
            assistant_format="Python call list",
            start_token="<|tool_call_start|>",
            end_token="<|tool_call_end|>",
        ),
        strips_past_reasoning=True,
    ),
)
