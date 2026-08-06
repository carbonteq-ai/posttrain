"""Pinned Gemma 4 foundation variants and their shared renderer contract."""

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

_GEMMA_4_E2B_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
_GEMMA_4_E4B_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
_GEMMA_4_12B_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
_GEMMA_4_31B_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
_GEMMA_4_TOKEN_ID_MAPPING_FINGERPRINT = (
    "059d0f7dd1efb018ec9801f316c99ab31a7c39e712de08626ac90c1898b42416"
)

_GEMMA_4_E2B_MTP_ASSISTANT_REPO = "google/gemma-4-E2B-it-assistant"
_GEMMA_4_E2B_MTP_ASSISTANT_REVISION = "2d874ef7d29f9a30599a1e4b3c1cbc9595f005df"
_GEMMA_4_E4B_MTP_ASSISTANT_REPO = "google/gemma-4-E4B-it-assistant"
_GEMMA_4_E4B_MTP_ASSISTANT_REVISION = "8d0031ea8c2109e2b1e86bb9368a4539b537f80a"
_GEMMA_4_12B_MTP_ASSISTANT_REPO = "google/gemma-4-12B-it-assistant"
_GEMMA_4_12B_MTP_ASSISTANT_REVISION = "364bd03c9952e5b7da73665ee30c9eccfc408345"
_GEMMA_4_31B_MTP_ASSISTANT_REPO = "google/gemma-4-31B-it-assistant"
_GEMMA_4_31B_MTP_ASSISTANT_REVISION = "627c5ec1458b9086b841a91e0512fd31fd2fbbf1"

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
        tool_calls=ToolCallProtocol(
            id="gemma4_structured",
            assistant_format="Gemma structured call syntax with unquoted keys",
            start_token="<|tool_call>",
            end_token="<tool_call|>",
        ),
        strips_past_reasoning=True,
    ),
)


def _gemma4_variant(
    *,
    id: str,
    repo_id: str,
    revision: str,
    parameters: int,
    modalities: tuple[str, ...],
    native_context_window: int,
    upstream_model_type: str,
    upstream_architecture: str,
    assistant_repo: str,
    assistant_revision: str,
    tokenizer_fingerprint: str | None = None,
) -> ModelVariant:
    artifact = HubModelRef(repo_id=repo_id, revision=revision)
    return ModelVariant(
        id=id,
        artifact=artifact,
        form="foundation",
        weight_precision="bf16",
        family="gemma4",
        parameters=parameters,
        instruction_tuned=True,
        capabilities=ModelCapabilities(
            modalities=modalities,
            native_context_window=native_context_window,
            mtp=True,
        ),
        renderer=GEMMA4_RENDERER_CONTRACT,
        base=artifact,
        tokenizer_fingerprint=tokenizer_fingerprint,
        provenance={
            "source": "huggingface",
            "license": "apache-2.0",
            "upstream_model_type": upstream_model_type,
            "upstream_architecture": upstream_architecture,
            "mtp_mode": "paired-assistant",
            "mtp_assistant_repo_id": assistant_repo,
            "mtp_assistant_revision": assistant_revision,
        },
    )


GEMMA_4_E2B_IT = _gemma4_variant(
    id="gemma4-e2b-it",
    repo_id="google/gemma-4-E2B-it",
    revision=_GEMMA_4_E2B_REVISION,
    parameters=5_123_178_051,
    modalities=("text", "image", "audio"),
    native_context_window=131_072,
    upstream_model_type="gemma4",
    upstream_architecture="Gemma4ForConditionalGeneration",
    assistant_repo=_GEMMA_4_E2B_MTP_ASSISTANT_REPO,
    assistant_revision=_GEMMA_4_E2B_MTP_ASSISTANT_REVISION,
    tokenizer_fingerprint=_GEMMA_4_TOKEN_ID_MAPPING_FINGERPRINT,
)

GEMMA_4_E4B_IT = _gemma4_variant(
    id="gemma4-e4b-it",
    repo_id="google/gemma-4-E4B-it",
    revision=_GEMMA_4_E4B_REVISION,
    parameters=7_996_156_490,
    modalities=("text", "image", "audio"),
    native_context_window=131_072,
    upstream_model_type="gemma4",
    upstream_architecture="Gemma4ForConditionalGeneration",
    assistant_repo=_GEMMA_4_E4B_MTP_ASSISTANT_REPO,
    assistant_revision=_GEMMA_4_E4B_MTP_ASSISTANT_REVISION,
    tokenizer_fingerprint=_GEMMA_4_TOKEN_ID_MAPPING_FINGERPRINT,
)

GEMMA_4_12B_IT = _gemma4_variant(
    id="gemma4-12b-it",
    repo_id="google/gemma-4-12B-it",
    revision=_GEMMA_4_12B_REVISION,
    parameters=11_959_730_224,
    modalities=("text", "image", "audio"),
    native_context_window=262_144,
    upstream_model_type="gemma4_unified",
    upstream_architecture="Gemma4UnifiedForConditionalGeneration",
    assistant_repo=_GEMMA_4_12B_MTP_ASSISTANT_REPO,
    assistant_revision=_GEMMA_4_12B_MTP_ASSISTANT_REVISION,
    tokenizer_fingerprint=_GEMMA_4_TOKEN_ID_MAPPING_FINGERPRINT,
)

GEMMA_4_31B_IT = _gemma4_variant(
    id="gemma4-31b-it",
    repo_id="google/gemma-4-31B-it",
    revision=_GEMMA_4_31B_REVISION,
    parameters=31_273_088_876,
    modalities=("text", "image"),
    native_context_window=262_144,
    upstream_model_type="gemma4",
    upstream_architecture="Gemma4ForConditionalGeneration",
    assistant_repo=_GEMMA_4_31B_MTP_ASSISTANT_REPO,
    assistant_revision=_GEMMA_4_31B_MTP_ASSISTANT_REVISION,
    tokenizer_fingerprint=_GEMMA_4_TOKEN_ID_MAPPING_FINGERPRINT,
)

__all__ = [
    "GEMMA4_RENDERER_CONTRACT",
    "GEMMA_4_E2B_IT",
    "GEMMA_4_E4B_IT",
    "GEMMA_4_12B_IT",
    "GEMMA_4_31B_IT",
]
