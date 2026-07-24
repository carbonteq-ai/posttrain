"""Human and JSON output helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from .context import CliState


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def emit(state: CliState, payload: object, human: str) -> None:
    if state.json_output:
        print(
            json.dumps(json_value(payload), indent=2, sort_keys=True),
            file=state.json_stream,
            flush=True,
        )
    else:
        print(human, flush=True)
