"""Pinned Qwen3.5 foundation profile."""

from posttrain.common.artifacts import HubModelRef
from posttrain.common.models import ModelCapabilities, ModelProfile

QWEN_35_2B = ModelProfile(
    id="qwen3.5-2b",
    artifact=HubModelRef(
        repo_id="Qwen/Qwen3.5-2B",
        revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
    ),
    family="qwen3.5",
    parameters=2_000_000_000,
    instruction_tuned=True,
    renderer="qwen3.5",
    default_reasoning_mode="off",
    capabilities=ModelCapabilities(
        modalities=("text", "image"),
        reasoning_modes=("native", "off", "thinking"),
        native_context_window=262_144,
        mtp=True,
    ),
)
