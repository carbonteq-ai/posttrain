"""Pinned LFM2.5 foundation variants and conversation contracts."""

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

LFM25_RENDERER_CONTRACT = RendererContract(
    id="lfm2.5-tools@1",
    model_family="lfm2.5",
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

LFM25_INSTRUCT_RENDERER_CONTRACT = RendererContract(
    id="lfm2.5-instruct-tools@1",
    model_family="lfm2.5",
    conversation=ConversationProfile(
        chat_template=ChatTemplate("package", "lfm25_instruct_chat.jinja"),
        roles=("system", "user", "assistant", "tool"),
        reasoning_modes=(ReasoningMode("off", (("preserve_thinking", False),)),),
        default_reasoning_mode="off",
        tool_calls=ToolCallProtocol(
            id="lfm2_pythonic",
            assistant_format="Python call list",
            start_token="<|tool_call_start|>",
            end_token="<|tool_call_end|>",
        ),
        strips_past_reasoning=True,
    ),
)

_LFM25_TOKENIZER_FINGERPRINT = "df1d8d5ec5d091b460562ffd545e4a5e91d17d4a0db7ebe733be34ed374377bd"

LFM_25_350M = ModelVariant(
    id="lfm2.5-350m",
    artifact=HubModelRef(
        repo_id="LiquidAI/LFM2.5-350M",
        revision="9e6c6ccf47cd318696e137d381a7ded8fe4df09f",
    ),
    form="foundation",
    weight_precision="bf16",
    family="lfm2.5",
    parameters=354_483_968,
    instruction_tuned=True,
    capabilities=ModelCapabilities(modalities=("text",), native_context_window=32_768),
    renderer=LFM25_INSTRUCT_RENDERER_CONTRACT,
    base=HubModelRef(
        repo_id="LiquidAI/LFM2.5-350M",
        revision="9e6c6ccf47cd318696e137d381a7ded8fe4df09f",
    ),
    tokenizer_fingerprint=_LFM25_TOKENIZER_FINGERPRINT,
    provenance={
        "source": "huggingface",
        "license": "lfm1.0",
        "upstream_architecture": "Lfm2ForCausalLM",
        "chat_template_sha256": "ba551d58630afa3190b1be3602e28301f3d2e9bbac978dfc49d6d825171648b6",
    },
)

LFM_25_12B_INSTRUCT = ModelVariant(
    id="lfm2.5-1.2b-instruct",
    artifact=HubModelRef(
        repo_id="LiquidAI/LFM2.5-1.2B-Instruct",
        revision="df58c174f05ff733f83f8cae10ea9298224c8006",
    ),
    form="foundation",
    weight_precision="bf16",
    family="lfm2.5",
    parameters=1_170_340_608,
    instruction_tuned=True,
    capabilities=ModelCapabilities(modalities=("text",), native_context_window=32_768),
    renderer=LFM25_INSTRUCT_RENDERER_CONTRACT,
    base=HubModelRef(
        repo_id="LiquidAI/LFM2.5-1.2B-Instruct",
        revision="df58c174f05ff733f83f8cae10ea9298224c8006",
    ),
    tokenizer_fingerprint=_LFM25_TOKENIZER_FINGERPRINT,
    provenance={
        "source": "huggingface",
        "license": "lfm1.0",
        "upstream_architecture": "Lfm2ForCausalLM",
        "chat_template_sha256": "ba551d58630afa3190b1be3602e28301f3d2e9bbac978dfc49d6d825171648b6",
    },
)

LFM_25_12B_THINKING = ModelVariant(
    id="lfm2.5-1.2b-thinking",
    artifact=HubModelRef(
        repo_id="LiquidAI/LFM2.5-1.2B-Thinking",
        revision="95053d21d8e0b7ca99421a2127ae39c64f685ff3",
    ),
    form="foundation",
    weight_precision="bf16",
    family="lfm2.5",
    parameters=1_170_000_000,
    instruction_tuned=True,
    capabilities=ModelCapabilities(modalities=("text",), native_context_window=32_768),
    renderer=LFM25_RENDERER_CONTRACT,
    base=HubModelRef(
        repo_id="LiquidAI/LFM2.5-1.2B-Thinking",
        revision="95053d21d8e0b7ca99421a2127ae39c64f685ff3",
    ),
)
