"""Pinned Qwen3.5 foundation profile."""

from posttrain.common.artifacts import HubModelRef
from posttrain.common.models import (
    ChatTemplate,
    ConversationProfile,
    ModelCapabilities,
    ModelProfile,
    ReasoningMode,
    ToolCallProtocol,
)

QWEN_35_2B = ModelProfile(
    id="qwen3.5-2b",
    artifact=HubModelRef(
        repo_id="Qwen/Qwen3.5-2B",
        revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
    ),
    family="qwen3.5",
    parameters=2_000_000_000,
    instruction_tuned=True,
    capabilities=ModelCapabilities(
        modalities=("text", "image"),
        native_context_window=262_144,
        mtp=True,
    ),
    conversation=ConversationProfile(
        chat_template=ChatTemplate("tokenizer"),
        roles=("system", "user", "assistant", "tool"),
        reasoning_modes=(
            ReasoningMode("native"),
            ReasoningMode("off", (("enable_thinking", False),)),
            ReasoningMode("thinking", (("enable_thinking", True),)),
        ),
        default_reasoning_mode="off",
        tool_calls=ToolCallProtocol(
            id="qwen3_xml",
            assistant_format="XML function and parameter elements",
            start_token="<tool_call>",
            end_token="</tool_call>",
        ),
        strips_past_reasoning=True,
    ),
    hf_text_generation_architecture="Qwen3_5ForCausalLM",
    vllm_text_generation_model_class="vllm.model_executor.models.qwen3_5:Qwen3_5ForCausalLM",
)
