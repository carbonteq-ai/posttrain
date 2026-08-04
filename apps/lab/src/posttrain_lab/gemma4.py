"""Lab-local Gemma 4 conversation contract for SQL protocol experiments."""

from __future__ import annotations

from posttrain.common.models import ChatTemplate, ConversationProfile, ReasoningMode, RendererContract
from posttrain.common.variants import RENDERER_CONTRACTS

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
        # SkyRL-SQL is a visible text protocol, not a structured tool-call
        # environment. Keep this unset instead of assigning Gemma delimiters to
        # one of common's unrelated Qwen/LFM protocol identifiers.
        tool_calls=None,
        strips_past_reasoning=False,
    ),
)


def register_gemma4_renderer() -> None:
    """Register the Lab contract before its project catalog is decoded."""

    existing = RENDERER_CONTRACTS.get(GEMMA4_RENDERER_CONTRACT.id)
    if existing is not None and existing != GEMMA4_RENDERER_CONTRACT:
        raise RuntimeError(f"renderer contract {GEMMA4_RENDERER_CONTRACT.id!r} is already registered differently")
    RENDERER_CONTRACTS[GEMMA4_RENDERER_CONTRACT.id] = GEMMA4_RENDERER_CONTRACT


__all__ = ["GEMMA4_RENDERER_CONTRACT", "register_gemma4_renderer"]
