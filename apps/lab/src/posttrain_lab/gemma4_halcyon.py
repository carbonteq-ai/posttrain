"""Lab-local selections for Gemma 4 Halcyon GraphQL SFT runs."""

from dataclasses import replace

from posttrain.common import (
    ChatTemplate,
    ConversationProfile,
    ExecutionTarget,
    HubModelRef,
    ModelCapabilities,
    ModelVariant,
    ReasoningMode,
    RendererContract,
    ToolCallProtocol,
)
from posttrain.train import (
    LoRAUpdate,
    SFTSettings,
    SFTValidationSettings,
    TrainingBinding,
    TrainingLoop,
    TrainingRenderer,
    TrainingRuntime,
)

GEMMA4_MODEL_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
GEMMA4_TARGET_MODULES = (
    r".*[.]language_model[.].*[.](q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)

GEMMA4_RENDERER_CONTRACT = RendererContract(
    id="gemma4-tools@lab-v1",
    model_family="gemma4",
    conversation=ConversationProfile(
        chat_template=ChatTemplate("tokenizer"),
        roles=("system", "user", "assistant", "tool"),
        reasoning_modes=(
            ReasoningMode("thinking", (("enable_thinking", True),)),
            ReasoningMode("off", (("enable_thinking", False),)),
        ),
        default_reasoning_mode="thinking",
        tool_calls=ToolCallProtocol(
            id="gemma4_native",
            assistant_format="Gemma native call:<name>{arguments} form",
            start_token="<|tool_call>",
            end_token="<tool_call|>",
        ),
        strips_past_reasoning=True,
    ),
)

_GEMMA4_BASE = HubModelRef("google/gemma-4-12B-it", GEMMA4_MODEL_REVISION)

GEMMA_4_12B_IT = ModelVariant(
    id="gemma4-12b-it",
    artifact=_GEMMA4_BASE,
    form="foundation",
    weight_precision="bf16",
    family="gemma4",
    parameters=11_959_730_176,
    instruction_tuned=True,
    renderer=GEMMA4_RENDERER_CONTRACT,
    capabilities=ModelCapabilities(
        modalities=("text", "image", "audio"),
        native_context_window=262_144,
        mtp=False,
    ),
    base=_GEMMA4_BASE,
)

RUNPOD_RTX_PRO_6000_96GB = ExecutionTarget(
    id="targets/runpod-rtx-pro-6000-96gb",
    revision="1",
    device_class="nvidia-cuda",
    memory_gb=96,
    placement={"world_size": 1},
)

GEMMA4_HALCYON_LORA = TrainingBinding(
    id="training/gemma4-12b-trl-halcyon-graphql-lora@lab-v1",
    revision="1",
    backend="trl@1.8.0",
    renderer=TrainingRenderer(
        id="gemma4-thinking-lab-v1",
        model_family="gemma4",
        implementation="default",
        reasoning_mode="thinking",
    ),
    update=LoRAUpdate(
        rank=8,
        alpha=16,
        dropout=0.0,
        target_modules=GEMMA4_TARGET_MODULES,
    ),
    target=RUNPOD_RTX_PRO_6000_96GB,
    runtime=TrainingRuntime(global_batch_size=1, nodes=1, devices_per_node=1),
    backend_options={
        "use_liger_kernel": False,
        "source_revision": "b43a0a3d622ab1547f4d2abbd1b25eab3c52a0b9",
        "dependency_lock_sha256": "1a24280c5e1db1536226f2958c1bd91a21e248f1c940d5598b1f64f27a6482cc",
    },
)

GEMMA4_HALCYON_CANARY = SFTSettings(
    id="gemma4-12b/graphql-stage1-sft-canary-lab-v1",
    revision="1",
    loop=TrainingLoop(
        max_steps=1,
        max_length=2_048,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=0.0001,
        warmup_ratio=0.0,
        max_grad_norm=1.0,
        logging_steps=1,
        checkpoint_steps=1,
        checkpoint_limit=1,
        seed=42,
        gradient_checkpointing=True,
    ),
    validation=SFTValidationSettings(
        steps=1,
        per_device_batch_size=1,
        on_start=True,
        at_end=True,
    ),
)

GEMMA4_HALCYON_LORA_FULL = replace(
    GEMMA4_HALCYON_LORA,
    id="training/gemma4-12b-trl-halcyon-graphql-lora-full@lab-v1",
    runtime=TrainingRuntime(global_batch_size=8, nodes=1, devices_per_node=1),
)

GEMMA4_HALCYON_SFT = SFTSettings(
    id="gemma4-12b/graphql-stage1-sft-lab-v1",
    revision="1",
    loop=TrainingLoop(
        max_steps=98,
        max_length=2_048,
        per_device_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=0.0001,
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        logging_steps=1,
        checkpoint_steps=49,
        checkpoint_limit=2,
        seed=42,
        gradient_checkpointing=True,
    ),
    validation=SFTValidationSettings(
        steps=49,
        per_device_batch_size=1,
        on_start=True,
        at_end=True,
    ),
)

__all__ = [
    "GEMMA4_HALCYON_CANARY",
    "GEMMA4_HALCYON_LORA",
    "GEMMA4_HALCYON_LORA_FULL",
    "GEMMA4_HALCYON_SFT",
    "GEMMA4_MODEL_REVISION",
    "GEMMA4_RENDERER_CONTRACT",
    "GEMMA4_TARGET_MODULES",
    "GEMMA_4_12B_IT",
    "RUNPOD_RTX_PRO_6000_96GB",
]
