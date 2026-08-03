"""Canonical representative prompts and model-declared reasoning controls."""

from __future__ import annotations

import hashlib
import json
import re
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
    source_id: str | None = None
    source_revision: str | None = None
    source_record_key: str | None = None
    license_id: str | None = None
    tools: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class PromptCorpusSource:
    id: str
    revision: str
    split: str
    license_id: str
    config: str | None = None


@dataclass(frozen=True, slots=True)
class PromptCorpusManifest:
    schema_version: int
    id: str
    revision: str
    digest: str
    record_count: int
    category_counts: Mapping[str, int]
    sources: tuple[PromptCorpusSource, ...]
    selection_algorithm: str
    license_notices: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PromptCorpus:
    manifest: PromptCorpusManifest
    records: tuple[PromptRecord, ...]


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
        tools = raw.get("tools", [])
        if not isinstance(tools, list) or any(not isinstance(tool, dict) for tool in tools):
            raise PromptError(f"prompt {prompt_id!r} tools must be objects")
        seen.add(prompt_id)
        records.append(
            PromptRecord(
                id=prompt_id,
                messages=tuple(messages),
                tags=tuple(tags),
                reasoning_mode=str(raw.get("reasoning_mode", "native")),
                source_id=_optional_string(raw, "source_id", prompt_id),
                source_revision=_optional_string(raw, "source_revision", prompt_id),
                source_record_key=_optional_string(raw, "source_record_key", prompt_id),
                license_id=_optional_string(raw, "license_id", prompt_id),
                tools=tuple(tools),
            )
        )
    if not records:
        raise PromptError(f"prompt corpus is empty: {source}")
    return tuple(records)


def load_prompt_records(path: Path) -> tuple[PromptRecord, ...]:
    return _parse_prompt_records(path.read_text(encoding="utf-8"), str(path))


def representative_prompt_records() -> tuple[PromptRecord, ...]:
    return load_prompt_corpus("general-serving-v1").records


def load_prompt_corpus(corpus_id: str) -> PromptCorpus:
    """Load and verify one packaged, provenance-aware prompt population."""

    if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", corpus_id) is None:
        raise PromptError(f"invalid prompt corpus id: {corpus_id!r}")
    root = files("posttrain.serve.benchmarks.general_serving.resources")
    records_resource = root.joinpath(f"{corpus_id}.jsonl")
    manifest_resource = root.joinpath(f"{corpus_id}.manifest.json")
    if not records_resource.is_file() or not manifest_resource.is_file():
        # Keep a narrow read-only fallback for already-built package images
        # during the resource relocation window.  New builds write beside the
        # owning corpus definition under ``general_serving/resources``.
        root = files("posttrain.serve.benchmarks.resources").joinpath("corpora")
        records_resource = root.joinpath(f"{corpus_id}.jsonl")
        manifest_resource = root.joinpath(f"{corpus_id}.manifest.json")
    records_text = records_resource.read_text(encoding="utf-8")
    try:
        raw = json.loads(manifest_resource.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PromptError(f"invalid prompt corpus manifest {manifest_resource}: {error}") from error
    if not isinstance(raw, dict):
        raise PromptError(f"prompt corpus manifest must be an object: {manifest_resource}")
    records = _parse_prompt_records(records_text, str(records_resource))
    manifest = _parse_manifest(raw, str(manifest_resource))
    digest = hashlib.sha256(records_text.encode()).hexdigest()
    if digest != manifest.digest:
        raise PromptError(f"prompt corpus digest mismatch for {corpus_id}")
    if len(records) != manifest.record_count:
        raise PromptError(f"prompt corpus record count mismatch for {corpus_id}")
    counts: dict[str, int] = {}
    for record in records:
        if not all(
            (
                record.source_id,
                record.source_revision,
                record.source_record_key,
                record.license_id,
            )
        ):
            raise PromptError(f"prompt {record.id!r} is missing provenance")
        for tag in record.tags:
            counts[tag] = counts.get(tag, 0) + 1
    if counts != dict(manifest.category_counts):
        raise PromptError(f"prompt corpus category counts mismatch for {corpus_id}")
    return PromptCorpus(manifest, records)


def _optional_string(raw: dict[str, Any], field: str, prompt_id: str) -> str | None:
    value = raw.get(field)
    if value is not None and (not isinstance(value, str) or not value):
        raise PromptError(f"prompt {prompt_id!r} {field} must be a non-empty string")
    return value


def _parse_manifest(raw: dict[str, Any], source: str) -> PromptCorpusManifest:
    required = {
        "schema_version",
        "id",
        "revision",
        "digest",
        "record_count",
        "category_counts",
        "sources",
        "selection_algorithm",
        "license_notices",
    }
    if set(raw) != required:
        raise PromptError(f"prompt corpus manifest fields are invalid: {source}")
    if raw["schema_version"] != 1:
        raise PromptError(f"unsupported prompt corpus schema_version: {raw['schema_version']!r}")
    if (
        not isinstance(raw["id"], str)
        or not isinstance(raw["revision"], str)
        or not isinstance(raw["digest"], str)
        or re.fullmatch(r"[0-9a-f]{64}", raw["digest"]) is None
        or not isinstance(raw["record_count"], int)
        or raw["record_count"] < 1
        or not isinstance(raw["selection_algorithm"], str)
    ):
        raise PromptError(f"prompt corpus manifest scalar fields are invalid: {source}")
    category_counts = raw["category_counts"]
    if not isinstance(category_counts, dict) or any(
        not isinstance(key, str) or not isinstance(value, int) or value < 1 for key, value in category_counts.items()
    ):
        raise PromptError(f"prompt corpus manifest category_counts are invalid: {source}")
    raw_sources = raw["sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise PromptError(f"prompt corpus manifest sources are invalid: {source}")
    parsed_sources: list[PromptCorpusSource] = []
    for item in raw_sources:
        if not isinstance(item, dict) or set(item) != {"id", "revision", "split", "license_id", "config"}:
            raise PromptError(f"prompt corpus manifest source is invalid: {source}")
        if any(not isinstance(item[key], str) or not item[key] for key in ("id", "revision", "split", "license_id")):
            raise PromptError(f"prompt corpus manifest source fields are invalid: {source}")
        config = item["config"]
        if config is not None and (not isinstance(config, str) or not config):
            raise PromptError(f"prompt corpus manifest source config is invalid: {source}")
        parsed_sources.append(PromptCorpusSource(**item))
    notices = raw["license_notices"]
    if not isinstance(notices, list) or any(not isinstance(value, str) or not value for value in notices):
        raise PromptError(f"prompt corpus manifest license_notices are invalid: {source}")
    return PromptCorpusManifest(
        schema_version=1,
        id=raw["id"],
        revision=raw["revision"],
        digest=raw["digest"],
        record_count=raw["record_count"],
        category_counts=dict(category_counts),
        sources=tuple(parsed_sources),
        selection_algorithm=raw["selection_algorithm"],
        license_notices=tuple(notices),
    )


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
) -> tuple[int, ...]:
    """Render canonical messages through the model tokenizer's native template."""

    kwargs = reasoning_template_kwargs(model, record.reasoning_mode)
    chat_template = model.conversation.chat_template.text()
    if chat_template is not None:
        kwargs["chat_template"] = chat_template
    unsupported_roles = {str(message["role"]) for message in record.messages} - set(model.conversation.roles)
    if unsupported_roles:
        raise PromptError(f"model does not support message roles: {', '.join(sorted(unsupported_roles))}")
    rendered = tokenizer.apply_chat_template(
        list(record.messages),
        tokenize=True,
        add_generation_prompt=True,
        tools=list(tools if tools is not None else record.tools) or None,
        **kwargs,
    )
    if isinstance(rendered, Mapping):
        rendered = rendered.get("input_ids")
    if (
        isinstance(rendered, Sequence)
        and not isinstance(rendered, (str, bytes))
        and len(rendered) == 1
        and isinstance(rendered[0], Sequence)
        and not isinstance(rendered[0], (str, bytes))
    ):
        rendered = rendered[0]
    if (
        not isinstance(rendered, Sequence)
        or isinstance(rendered, (str, bytes))
        or not rendered
        or any(not isinstance(token, int) or isinstance(token, bool) for token in rendered)
    ):
        raise PromptError("model chat template did not return one integer token sequence")
    return tuple(rendered)


__all__ = [
    "PromptError",
    "PromptCorpus",
    "PromptCorpusManifest",
    "PromptCorpusSource",
    "PromptRecord",
    "load_prompt_corpus",
    "load_prompt_records",
    "reasoning_template_kwargs",
    "representative_prompt_records",
    "render_prompt",
]
