"""Strict parser for the visible SkyRL-SQL conversation protocol."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

_TURN = re.compile(
    r"\A\s*<think>(?P<think>.*?)</think>\s*"
    r"(?:(?:<sql>(?P<sql>.*?)</sql>)|(?:<solution>(?P<solution>.*?)</solution>))\s*\Z",
    re.DOTALL,
)
_RESERVED = re.compile(r"</?(?:think|sql|solution|observation)>", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ParsedTurn:
    reasoning: str
    query: str
    kind: Literal["sql", "solution"]


class ProtocolError(ValueError):
    """The assistant turn does not conform to the SkyRL-SQL protocol."""


def parse_turn(message: str) -> ParsedTurn:
    """Parse one complete assistant turn and reject nested or extra protocol tags."""

    match = _TURN.fullmatch(message)
    if match is None:
        raise ProtocolError(
            "respond with one non-empty <think> block followed by exactly one "
            "<sql> or <solution> block"
        )
    reasoning = match.group("think").strip()
    sql = match.group("sql")
    solution = match.group("solution")
    query = (sql if sql is not None else solution or "").strip()
    if not reasoning:
        raise ProtocolError("the <think> block cannot be empty")
    if not query:
        raise ProtocolError("the SQL block cannot be empty")
    if _RESERVED.search(reasoning) or _RESERVED.search(query):
        raise ProtocolError("nested or model-authored protocol tags are not allowed")
    return ParsedTurn(reasoning, query, "sql" if sql is not None else "solution")


def corrective_observation(error: str) -> str:
    return (
        "<observation>Protocol error: "
        + error
        + ". Try again using <think>...</think><sql>...</sql>, or finish with "
        "<think>...</think><solution>...</solution>.</observation>"
    )


__all__ = ["ParsedTurn", "ProtocolError", "corrective_observation", "parse_turn"]
