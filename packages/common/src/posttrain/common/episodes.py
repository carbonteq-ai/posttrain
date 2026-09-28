"""How a rollout episode ended: one label per episode, shared by every reader.

An episode is one agent rollout in an environment (one Verifiers trace). The
integration that understands the native trace decides the label; tracking
stores it as the trace attribute ``episode_ending``; training and Observatory
count and show it. The names and meanings live here so that every producer and
reader agrees on them.

Every ending other than ``completed`` and ``error`` is a *truncation*: the
episode stopped at a limit rather than on its own, and it keeps the reward the
environment scored for the state it reached. ``truncated`` flags recorded
before these labels existed keep exactly that meaning.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, cast

type EpisodeEnding = Literal[
    "completed",
    "turn_limit",
    "token_budget",
    "time_limit",
    "reply_token_limit",
    "context_limit_reply_cut",
    "context_rejected",
    "error",
]

EPISODE_ENDING_ATTRIBUTE = "episode_ending"
"""Trace attribute (and tracking trace metadata key) that carries the label."""

EPISODE_ENDING_DESCRIPTIONS: Mapping[EpisodeEnding, str] = MappingProxyType(
    {
        "completed": "The agent finished on its own: no limit stopped it and nothing failed.",
        "turn_limit": "The episode used every turn it was allowed (Verifiers stop condition max_turns).",
        "token_budget": (
            "A rollout-wide token budget stopped the episode (Verifiers stop condition max_output_tokens, "
            "max_input_tokens or max_total_tokens)."
        ),
        "time_limit": "The harness stopped the episode at its time limit (legacy stop condition harness_timeout).",
        "reply_token_limit": (
            "The last model reply reached the per-call max_tokens and was cut; a cut reply has no tool calls, "
            "so the agent treated it as its final answer."
        ),
        "context_limit_reply_cut": (
            "The last model reply was cut because prompt plus reply reached the model's context length "
            "(finish_reason length before the per-call max_tokens)."
        ),
        "context_rejected": (
            "The next model request was refused because its prompt exceeded the model's context length "
            "(HTTP 400 on the final call, or the legacy stop condition context_length)."
        ),
        "error": "The episode failed execution: a harness, environment or other model-call error.",
    }
)

EPISODE_ENDINGS: tuple[EpisodeEnding, ...] = tuple(EPISODE_ENDING_DESCRIPTIONS)

TRUNCATED_EPISODE_ENDINGS: frozenset[EpisodeEnding] = frozenset(
    ending for ending in EPISODE_ENDINGS if ending not in {"completed", "error"}
)


def episode_ending(value: object) -> EpisodeEnding | None:
    """The label when ``value`` is a known ending, else None (absent or unknown)."""

    return cast(EpisodeEnding, value) if isinstance(value, str) and value in EPISODE_ENDING_DESCRIPTIONS else None


def episode_ending_is_truncated(ending: EpisodeEnding) -> bool:
    """Whether the ending is a truncation (the flag readers call ``truncated``)."""

    return ending in TRUNCATED_EPISODE_ENDINGS


__all__ = [
    "EPISODE_ENDINGS",
    "EPISODE_ENDING_ATTRIBUTE",
    "EPISODE_ENDING_DESCRIPTIONS",
    "TRUNCATED_EPISODE_ENDINGS",
    "EpisodeEnding",
    "episode_ending",
    "episode_ending_is_truncated",
]
