"""Pinned Gemma 4 foundation variants."""

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

_GEMMA_4_12B_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
_GEMMA_4_E2B_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
_GEMMA_4_E4B_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"

_GEMMA4_CONVERSATION = ConversationProfile(
    chat_template=ChatTemplate("tokenizer"),
    roles=("system", "user", "assistant", "tool"),
    reasoning_modes=(
        ReasoningMode("off", (("enable_thinking", False),)),
        ReasoningMode("thinking", (("enable_thinking", True),)),
    ),
    default_reasoning_mode="off",
    tool_calls=ToolCallProtocol(
        id="gemma4_structured",
        assistant_format="Gemma structured call syntax with unquoted keys",
        start_token="<|tool_call>",
        end_token="<tool_call|>",
    ),
    strips_past_reasoning=True,
)

GEMMA4_RENDERER_CONTRACT = RendererContract(
    id="gemma4-tools@1",
    model_family="gemma4",
    conversation=_GEMMA4_CONVERSATION,
)

GEMMA4_E4B_RENDERER_CONTRACT = RendererContract(
    id="gemma4-e4b-tools@1",
    model_family="gemma4",
    conversation=_GEMMA4_CONVERSATION,
)

# E2B and E4B have byte-identical pinned tokenizer and chat-template assets.
# Keep the published E4B contract identity stable while exposing its actual
# family scope to new small-Gemma variants.
GEMMA4_SMALL_RENDERER_CONTRACT = GEMMA4_E4B_RENDERER_CONTRACT

GEMMA_4_12B_IT = ModelVariant(
    id="gemma4-12b-it",
    artifact=HubModelRef(
        repo_id="google/gemma-4-12B-it",
        revision=_GEMMA_4_12B_REVISION,
    ),
    form="foundation",
    weight_precision="bf16",
    family="gemma4",
    parameters=11_959_730_224,
    instruction_tuned=True,
    capabilities=ModelCapabilities(
        modalities=("text", "image", "audio", "video"),
        native_context_window=262_144,
        mtp=False,
    ),
    renderer=GEMMA4_RENDERER_CONTRACT,
    base=HubModelRef(
        repo_id="google/gemma-4-12B-it",
        revision=_GEMMA_4_12B_REVISION,
    ),
    tokenizer_fingerprint="059d0f7dd1efb018ec9801f316c99ab31a7c39e712de08626ac90c1898b42416",
    provenance={
        "source": "huggingface",
        "license": "apache-2.0",
        "upstream_model_type": "gemma4_unified",
        "upstream_architecture": "Gemma4UnifiedForConditionalGeneration",
    },
)

GEMMA_4_E2B_IT = ModelVariant(
    id="gemma4-e2b-it",
    artifact=HubModelRef(
        repo_id="google/gemma-4-E2B-it",
        revision=_GEMMA_4_E2B_REVISION,
    ),
    form="foundation",
    weight_precision="bf16",
    family="gemma4",
    parameters=5_123_178_051,
    instruction_tuned=True,
    capabilities=ModelCapabilities(
        modalities=("text", "image", "audio", "video"),
        native_context_window=131_072,
        mtp=False,
    ),
    renderer=GEMMA4_SMALL_RENDERER_CONTRACT,
    base=HubModelRef(
        repo_id="google/gemma-4-E2B-it",
        revision=_GEMMA_4_E2B_REVISION,
    ),
    tokenizer_fingerprint="1ab787c816b67a0936e8d1c9ff20e6cf5bd8b77faabfe6ada5905bd2c433b413",
    provenance={
        "source": "huggingface",
        "license": "apache-2.0",
        "upstream_model_type": "gemma4",
        "upstream_architecture": "Gemma4ForConditionalGeneration",
    },
)

GEMMA_4_E4B_IT = ModelVariant(
    id="gemma4-e4b-it",
    artifact=HubModelRef(
        repo_id="google/gemma-4-E4B-it",
        revision=_GEMMA_4_E4B_REVISION,
    ),
    form="foundation",
    weight_precision="bf16",
    family="gemma4",
    parameters=7_996_156_490,
    instruction_tuned=True,
    capabilities=ModelCapabilities(
        modalities=("text", "image", "audio", "video"),
        native_context_window=131_072,
        mtp=False,
    ),
    renderer=GEMMA4_E4B_RENDERER_CONTRACT,
    base=HubModelRef(
        repo_id="google/gemma-4-E4B-it",
        revision=_GEMMA_4_E4B_REVISION,
    ),
    tokenizer_fingerprint="1ab787c816b67a0936e8d1c9ff20e6cf5bd8b77faabfe6ada5905bd2c433b413",
    provenance={
        "source": "huggingface",
        "upstream_model_type": "gemma4",
        "upstream_architecture": "Gemma4ForConditionalGeneration",
    },
)

__all__ = [
    "GEMMA4_E4B_RENDERER_CONTRACT",
    "GEMMA4_RENDERER_CONTRACT",
    "GEMMA4_SMALL_RENDERER_CONTRACT",
    "GEMMA_4_12B_IT",
    "GEMMA_4_E2B_IT",
    "GEMMA_4_E4B_IT",
]
