"""Validation for veRL's provider-neutral structured file logger output."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class VerlMetricRecord:
    """One immutable native veRL metric record."""

    step: int
    data: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("veRL metric step cannot be negative")
        object.__setattr__(self, "data", MappingProxyType(dict(self.data)))


def read_verl_metric_records(path: Path) -> tuple[VerlMetricRecord, ...]:
    """Read a complete, monotonic JSONL sidecar without importing veRL."""

    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[VerlMetricRecord] = []
    previous_step: int | None = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"veRL metric sidecar contains a blank line at {line_number}")
        try:
            payload = json.loads(line, parse_constant=_reject_non_finite_constant)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid veRL metric JSON at line {line_number}") from error
        if not isinstance(payload, dict):
            raise TypeError(f"veRL metric line {line_number} must be a JSON object")
        step = payload.get("step")
        data = payload.get("data")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(f"veRL metric line {line_number} requires a non-negative integer step")
        if not isinstance(data, dict):
            raise TypeError(f"veRL metric line {line_number} requires a data object")
        if previous_step is not None and step < previous_step:
            raise ValueError(
                f"veRL metric steps must be monotonic; line {line_number} moved from {previous_step} to {step}"
            )
        _validate_finite(data, line_number=line_number)
        records.append(VerlMetricRecord(step=step, data=data))
        previous_step = step
    if not records:
        raise ValueError("veRL metric sidecar is empty")
    return tuple(records)


def _reject_non_finite_constant(value: str) -> None:
    raise ValueError(f"veRL metric JSON contains non-finite constant {value}")


def _validate_finite(value: object, *, line_number: int) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"veRL metric line {line_number} contains a non-finite number")
    if isinstance(value, dict):
        for child in value.values():
            _validate_finite(child, line_number=line_number)
    elif isinstance(value, list):
        for child in value:
            _validate_finite(child, line_number=line_number)


__all__ = ["VerlMetricRecord", "read_verl_metric_records"]
