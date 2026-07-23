"""Pinned Qwen3.5 foundation variant."""

from posttrain.common.artifacts import HubModelRef
from posttrain.common.models import (
    ChatTemplate,
    ConversationProfile,
    ModelCapabilities,
    ModelVariant,
    ReasoningMode,
    RendererContract,
    ToolCallProtocol,
)

QWEN35_RENDERER_CONTRACT = RendererContract(
    id="qwen3.5-tools@1",
    model_family="qwen3.5",
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
)

QWEN_35_08B = ModelVariant(
    id="qwen3.5-0.8b",
    artifact=HubModelRef(
        repo_id="Qwen/Qwen3.5-0.8B",
        revision="2fc06364715b967f1860aea9cf38778875588b17",
    ),
    form="foundation",
    weight_precision="bf16",
    family="qwen3.5",
    parameters=800_000_000,
    instruction_tuned=True,
    capabilities=ModelCapabilities(
        modalities=("text", "image"),
        native_context_window=262_144,
        mtp=True,
    ),
    renderer=QWEN35_RENDERER_CONTRACT,
    base=HubModelRef(
        repo_id="Qwen/Qwen3.5-0.8B",
        revision="2fc06364715b967f1860aea9cf38778875588b17",
    ),
    tokenizer_fingerprint="544bc020ecb01661a305ed3ba1fffe49011d65eed195b059457edb69db4ded0c",
)

QWEN_35_2B = ModelVariant(
    id="qwen3.5-2b",
    artifact=HubModelRef(
        repo_id="Qwen/Qwen3.5-2B",
        revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
    ),
    form="foundation",
    weight_precision="bf16",
    family="qwen3.5",
    parameters=2_000_000_000,
    instruction_tuned=True,
    capabilities=ModelCapabilities(
        modalities=("text", "image"),
        native_context_window=262_144,
        mtp=True,
    ),
    renderer=QWEN35_RENDERER_CONTRACT,
    base=HubModelRef(
        repo_id="Qwen/Qwen3.5-2B",
        revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
    ),
)
