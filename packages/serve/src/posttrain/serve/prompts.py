"""Canonical representative prompts and model-declared reasoning controls."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from posttrain.common import ModelVariant


class PromptError(ValueError):
    """Raised when prompt data or a reasoning mode is invalid."""


@dataclass(frozen=True, slots=True)
class PromptRecord:
    id: str
    messages: tuple[dict[str, Any], ...]
    tags: tuple[str, ...]
    reasoning_mode: str = "native"


def _parse_prompt_records(text: str, source: str) -> tuple[PromptRecord, ...]:
    records: list[PromptRecord] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise PromptError(f"invalid JSON at {source}:{line_number}: {error}") from error
        prompt_id = raw.get("id") if isinstance(raw, dict) else None
        messages = raw.get("messages") if isinstance(raw, dict) else None
        if not isinstance(prompt_id, str) or not prompt_id or prompt_id in seen:
            raise PromptError(f"prompt id must be non-empty and unique at {source}:{line_number}")
        if not isinstance(messages, list) or not messages:
            raise PromptError(f"prompt {prompt_id!r} requires non-empty messages")
        for message in messages:
            if (
                not isinstance(message, dict)
                or message.get("role")
                not in {
                    "system",
                    "user",
                    "assistant",
                    "tool",
                }
                or "content" not in message
            ):
                raise PromptError(f"prompt {prompt_id!r} contains an invalid message")
        tags = raw.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise PromptError(f"prompt {prompt_id!r} tags must be strings")
        seen.add(prompt_id)
        records.append(
            PromptRecord(
                id=prompt_id,
                messages=tuple(messages),
                tags=tuple(tags),
                reasoning_mode=str(raw.get("reasoning_mode", "native")),
            )
        )
    if not records:
        raise PromptError(f"prompt corpus is empty: {source}")
    return tuple(records)


def load_prompt_records(path: Path) -> tuple[PromptRecord, ...]:
    return _parse_prompt_records(path.read_text(encoding="utf-8"), str(path))


def representative_prompt_records() -> tuple[PromptRecord, ...]:
    resource = files("posttrain.serve.benchmarks.resources").joinpath("corpora/representative-v1.jsonl")
    return _parse_prompt_records(resource.read_text(encoding="utf-8"), str(resource))


def reasoning_template_kwargs(
    model: ModelVariant,
    requested_mode: str,
) -> dict[str, Any]:
    """Resolve a requested mode without pretending unsupported levels exist."""

    try:
        return model.conversation.reasoning_mode(requested_mode).kwargs()
    except ValueError as error:
        raise PromptError(str(error)) from error


def render_prompt(
    tokenizer: Any,
    record: PromptRecord,
    model: ModelVariant,
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> Sequence[int]:
    """Render canonical messages through the model tokenizer's native template."""

    kwargs = reasoning_template_kwargs(model, record.reasoning_mode)
    chat_template = model.conversation.chat_template.text()
    if chat_template is not None:
        kwargs["chat_template"] = chat_template
    unsupported_roles = {str(message["role"]) for message in record.messages} - set(model.conversation.roles)
    if unsupported_roles:
        raise PromptError(f"model does not support message roles: {', '.join(sorted(unsupported_roles))}")
    return tokenizer.apply_chat_template(
        list(record.messages),
        tokenize=True,
        add_generation_prompt=True,
        tools=list(tools) if tools is not None else None,
        **kwargs,
    )


__all__ = [
    "PromptError",
    "PromptRecord",
    "load_prompt_records",
    "reasoning_template_kwargs",
    "representative_prompt_records",
    "render_prompt",
]
