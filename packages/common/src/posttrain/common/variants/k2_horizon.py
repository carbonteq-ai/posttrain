"""Pinned IFM K2-Horizon foundation variant and tokenizer-owned protocol."""

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

_K2_HORIZON_7B_REVISION = "586b03f0fd1fbbf2f13eeafc33749e95ae34dd10"

K2_HORIZON_RENDERER_CONTRACT = RendererContract(
    id="k2-horizon-tools-thinking@1",
    model_family="k2-horizon",
    conversation=ConversationProfile(
        chat_template=ChatTemplate("tokenizer"),
        roles=("system", "user", "assistant", "tool"),
        reasoning_modes=(
            ReasoningMode(
                "high",
                (
                    ("reasoning_effort", "high"),
                    ("tool_presentation_format", "markdown"),
                    ("tool_call_format", "xml"),
                ),
            ),
        ),
        default_reasoning_mode="high",
        tool_calls=ToolCallProtocol(
            id="k2_ifm_xml",
            assistant_format="IFM XML tool-call block with key/value arguments",
            start_token="<ifm|tool_calls>",
            end_token="</ifm|tool_calls>",
        ),
        strips_past_reasoning=True,
    ),
)

K2_HORIZON_7B = ModelVariant(
    id="k2-horizon-7b",
    artifact=HubModelRef(repo_id="IFM/K2-Horizon-7B", revision=_K2_HORIZON_7B_REVISION),
    form="foundation",
    weight_precision="bf16",
    family="k2-horizon",
    parameters=7_000_000_000,
    instruction_tuned=True,
    renderer=K2_HORIZON_RENDERER_CONTRACT,
    capabilities=ModelCapabilities(modalities=("text",), native_context_window=524_288),
    base=HubModelRef(repo_id="IFM/K2-Horizon-7B", revision=_K2_HORIZON_7B_REVISION),
    provenance={
        "source": "huggingface",
        "license": "apache-2.0",
        "upstream_model_type": "k2_horizon",
        "upstream_architecture": "K2HorizonForCausalLM",
        # The executable model code is bound to the immutable Hub revision
        # above. Loaders must never infer this permission for arbitrary models.
        "trust_remote_code": True,
    },
)

__all__ = ["K2_HORIZON_7B", "K2_HORIZON_RENDERER_CONTRACT"]
