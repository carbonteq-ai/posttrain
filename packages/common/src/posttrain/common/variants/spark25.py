"""Pinned Spark-X2.5 foundation variant and renderer contract."""

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

_SPARK_X25_4B_REVISION = "5e10fcc0286756aebf7c41dc52c1e42d95c70281"

SPARK25_RENDERER_CONTRACT = RendererContract(
    id="spark2.5-tools-thinking@1",
    model_family="spark2.5",
    conversation=ConversationProfile(
        chat_template=ChatTemplate("tokenizer"),
        roles=("system", "user", "assistant", "tool"),
        reasoning_modes=(
            ReasoningMode("off", (("enable_thinking", False),)),
            ReasoningMode("thinking", (("enable_thinking", True),)),
        ),
        default_reasoning_mode="thinking",
        tool_calls=ToolCallProtocol(
            id="spark25_xml",
            assistant_format="Spark XML function with XML key/value arguments",
            start_token="<tool_call>",
            end_token="</tool_call>",
        ),
        strips_past_reasoning=True,
    ),
)

SPARK_X25_4B = ModelVariant(
    id="spark-x2.5-4b",
    artifact=HubModelRef(
        repo_id="XHToken/Spark-X2.5-4B",
        revision=_SPARK_X25_4B_REVISION,
    ),
    form="foundation",
    weight_precision="bf16",
    family="spark2.5",
    parameters=4_000_000_000,
    instruction_tuned=True,
    capabilities=ModelCapabilities(
        modalities=("text",),
        native_context_window=1_048_576,
        mtp=False,
    ),
    renderer=SPARK25_RENDERER_CONTRACT,
    base=HubModelRef(
        repo_id="XHToken/Spark-X2.5-4B",
        revision=_SPARK_X25_4B_REVISION,
    ),
    provenance={
        "source": "huggingface",
        "license": "apache-2.0",
        "upstream_model_type": "spark2_5",
        "upstream_architecture": "Spark2_5ForCausalLM",
        "parameter_count_basis": "model-card-name-rounded",
    },
)

__all__ = ["SPARK25_RENDERER_CONTRACT", "SPARK_X25_4B"]
