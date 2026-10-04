"""Optional native parser evidence transport for in-process policy generators."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

from posttrain.common import JsonValue


@lru_cache(maxsize=1)
def _renderer_distributions() -> tuple[tuple[str, str], ...]:
    return tuple(
        (name, importlib.metadata.version(name))
        for name in sorted(importlib.metadata.packages_distributions().get("renderers", ()))
    )


def prepare_parser_evidence() -> None:
    """Resolve stable package provenance during setup, before concurrent turns."""
    _renderer_distributions()


def encode_parser_evidence(
    parsed: Any,
    completion_ids: Sequence[int],
    message: Mapping[str, Any],
    renderer: Any,
    *,
    configuration: Mapping[str, Any],
) -> Mapping[str, JsonValue] | None:
    """Use the env-owned schema without making generic generation require it.

    Older/absent native runtimes retain their existing unsupported call-alignment
    behavior. A present sidecar is validated strictly by the receiving adapter.
    """
    if not parsed.tool_calls:
        return None
    try:
        module = importlib.import_module("verifiers.v1.clients.train")
    except ModuleNotFoundError as error:
        if error.name in {"verifiers", "verifiers.v1", "verifiers.v1.clients", "verifiers.v1.clients.train"}:
            return None
        raise
    capture = getattr(module, "capture_generated_calls", None)
    if capture is None:
        return None
    types = importlib.import_module("verifiers.v1.types")
    calls = tuple(types.ToolCall.model_validate(item) for item in message.get("tool_calls") or ())
    parser = getattr(renderer, "_tool_parser", None)
    descriptor = {
        "renderers_distributions": dict(_renderer_distributions()),
        "renderer": f"{type(renderer).__module__}.{type(renderer).__qualname__}",
        "parser": f"{type(parser).__module__}.{type(parser).__qualname__}" if parser is not None else None,
        "configuration": dict(configuration),
    }
    digest = hashlib.sha256(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    revision = f"{module.GENERATED_CALL_PARSER_REVISION}:posttrain:{digest}"
    attempts = capture(parsed.tool_calls, completion_ids, parser_revision=revision, emitted_calls=calls)
    # JSON copies own their bytes independently of the mutable parsed response.
    return cast(
        Mapping[str, JsonValue],
        json.loads(
            json.dumps(
                {
                    "kind": "verifiers.generated-calls",
                    "schema_version": 1,
                    "producer": descriptor,
                    "attempts": [attempt.model_dump(mode="json") for attempt in attempts],
                },
                allow_nan=False,
            )
        ),
    )


@dataclass(frozen=True)
class DecodedParserEvidence:
    attempts: tuple[Any, ...]
    producer: Any


def decode_parser_evidence(evidence: Mapping[str, JsonValue]) -> DecodedParserEvidence:
    """Present malformed or unsupported evidence is never silently dropped."""
    if evidence.get("kind") != "verifiers.generated-calls":
        raise ValueError("unsupported native parser evidence envelope")
    if type(evidence.get("schema_version")) is not int or evidence.get("schema_version") != 1:
        raise ValueError("unsupported native parser evidence version")
    if set(evidence) != {"kind", "schema_version", "producer", "attempts"}:
        raise ValueError("unsupported native parser evidence envelope")
    producer = evidence.get("producer")
    if not isinstance(producer, dict):
        raise ValueError("native parser evidence requires producer provenance")
    digest = hashlib.sha256(
        json.dumps(producer, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    values = evidence.get("attempts")
    if not isinstance(values, list):
        raise ValueError("native parser evidence requires an ordered attempt list")
    types = importlib.import_module("verifiers.v1.types")
    record_type = getattr(types, "GeneratedCallAttempt", None)
    producer_type = getattr(types, "GeneratedCallProducer", None)
    if record_type is None or producer_type is None:
        raise ValueError("selected native runtime cannot consume generated-call evidence")
    records = tuple(record_type.model_validate(value) for value in values)
    module = importlib.import_module("verifiers.v1.clients.train")
    revision = f"{module.GENERATED_CALL_PARSER_REVISION}:posttrain:{digest}"
    if any(record.parser_revision != revision for record in records):
        raise ValueError("native parser evidence producer digest does not match attempts")
    return DecodedParserEvidence(records, producer_type.capture(revision, producer))
