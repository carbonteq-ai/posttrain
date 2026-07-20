"""Pinned LFM2.5 Thinking foundation profile."""

from posttrain.common.artifacts import HubModelRef
from posttrain.common.models import ModelCapabilities, ModelProfile

LFM_25_12B_THINKING = ModelProfile(
    id="lfm2.5-1.2b-thinking",
    artifact=HubModelRef(
        repo_id="LiquidAI/LFM2.5-1.2B-Thinking",
        revision="95053d21d8e0b7ca99421a2127ae39c64f685ff3",
    ),
    family="lfm2.5",
    parameters=1_170_000_000,
    instruction_tuned=True,
    renderer="default",
    default_reasoning_mode="native",
    capabilities=ModelCapabilities(
        modalities=("text",),
        reasoning_modes=("native",),
        native_context_window=32_768,
    ),
)
