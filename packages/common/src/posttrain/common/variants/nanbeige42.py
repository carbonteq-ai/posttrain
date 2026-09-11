"""Pinned Nanbeige 4.2 foundation variant and renderer contract."""

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

_NANBEIGE_42_3B_REVISION = "3384e426066d1a49c3aea90a7190b81260a6533f"

NANBEIGE42_RENDERER_CONTRACT = RendererContract(
    id="nanbeige4.2-tools-thinking@1",
    model_family="nanbeige4.2",
    conversation=ConversationProfile(
        chat_template=ChatTemplate("tokenizer"),
        roles=("system", "user", "assistant", "tool"),
        reasoning_modes=(
            ReasoningMode("off", (("enable_thinking", False), ("preserve_thinking", False))),
            ReasoningMode("thinking", (("enable_thinking", True), ("preserve_thinking", False))),
            ReasoningMode(
                "thinking-preserved",
                (("enable_thinking", True), ("preserve_thinking", True)),
            ),
        ),
        default_reasoning_mode="thinking",
        tool_calls=ToolCallProtocol(
            id="nanbeige_xml",
            assistant_format="XML function and parameter elements",
            start_token="<tool_call>",
            end_token="</tool_call>",
        ),
        strips_past_reasoning=True,
    ),
)

NANBEIGE_42_3B = ModelVariant(
    id="nanbeige4.2-3b",
    artifact=HubModelRef(
        repo_id="Nanbeige/Nanbeige4.2-3B",
        revision=_NANBEIGE_42_3B_REVISION,
    ),
    form="foundation",
    weight_precision="bf16",
    family="nanbeige4.2",
    parameters=4_000_000_000,
    instruction_tuned=True,
    capabilities=ModelCapabilities(
        modalities=("text",),
        native_context_window=262_144,
        mtp=False,
    ),
    renderer=NANBEIGE42_RENDERER_CONTRACT,
    base=HubModelRef(
        repo_id="Nanbeige/Nanbeige4.2-3B",
        revision=_NANBEIGE_42_3B_REVISION,
    ),
    tokenizer_fingerprint="1d858a0fc007f22af6ae18bfa1ae52d30e398aa9cd1ea06e7777176869346a3f",
    provenance={
        "source": "huggingface",
        "license": "apache-2.0",
        "upstream_model_type": "nanbeige",
        "upstream_architecture": "NanbeigeForCausalLM",
        "parameter_count_basis": "model-card-total-rounded",
    },
)

__all__ = ["NANBEIGE42_RENDERER_CONTRACT", "NANBEIGE_42_3B"]
