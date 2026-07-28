"""Pinned Gemma 4 foundation variants used by the Policy Prism experiment."""

from posttrain.common.artifacts import HubModelRef
from posttrain.common.models import (
    ChatTemplate,
    ConversationProfile,
    ModelCapabilities,
    ModelVariant,
    ReasoningMode,
    RendererContract,
)

GEMMA4_RENDERER_CONTRACT = RendererContract(
    id="gemma4-native-v1",
    model_family="gemma4",
    conversation=ConversationProfile(
        chat_template=ChatTemplate("tokenizer"),
        roles=("system", "user", "assistant", "tool"),
        reasoning_modes=(ReasoningMode("native"),),
        default_reasoning_mode="native",
    ),
)

_GEMMA4_TOKENIZER_FINGERPRINT = "1ab787c816b67a0936e8d1c9ff20e6cf5bd8b77faabfe6ada5905bd2c433b413"

GEMMA_4_E4B_IT = ModelVariant(
    id="gemma4-e4b-it",
    artifact=HubModelRef(
        repo_id="google/gemma-4-E4B-it",
        revision="ee0ef6023621cff504d758262d4e04895a5af4a2",
    ),
    form="foundation",
    weight_precision="bf16",
    family="gemma4",
    parameters=4_000_000_000,
    instruction_tuned=True,
    tokenizer_fingerprint=_GEMMA4_TOKENIZER_FINGERPRINT,
    capabilities=ModelCapabilities(
        modalities=("text", "image", "video", "audio"),
        native_context_window=131_072,
    ),
    renderer=GEMMA4_RENDERER_CONTRACT,
    base=HubModelRef(
        repo_id="google/gemma-4-E4B-it",
        revision="ee0ef6023621cff504d758262d4e04895a5af4a2",
    ),
)

GEMMA_4_31B_IT = ModelVariant(
    id="gemma4-31b-it",
    artifact=HubModelRef(
        repo_id="google/gemma-4-31B-it",
        revision="842da3794eaa0b77d5f08bae87a17459d91ff475",
    ),
    form="foundation",
    weight_precision="bf16",
    family="gemma4",
    parameters=31_000_000_000,
    instruction_tuned=True,
    tokenizer_fingerprint=_GEMMA4_TOKENIZER_FINGERPRINT,
    capabilities=ModelCapabilities(
        modalities=("text", "image"),
        native_context_window=262_144,
    ),
    renderer=GEMMA4_RENDERER_CONTRACT,
    base=HubModelRef(
        repo_id="google/gemma-4-31B-it",
        revision="842da3794eaa0b77d5f08bae87a17459d91ff475",
    ),
)

__all__ = ["GEMMA4_RENDERER_CONTRACT", "GEMMA_4_31B_IT", "GEMMA_4_E4B_IT"]
