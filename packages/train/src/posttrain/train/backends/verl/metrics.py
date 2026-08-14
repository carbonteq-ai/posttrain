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


@dataclass(frozen=True, slots=True)
class VerlRolloutRewardRecord:
    """Trace-keyed post-shaping reward emitted by the isolated agent loop."""

    trace_id: str
    step: int
    task_reward: float
    algorithm_reward: float

    def __post_init__(self) -> None:
        if not self.trace_id.strip() or self.step < 0:
            raise ValueError("veRL rollout reward record requires a trace id and non-negative step")
        if not math.isfinite(self.task_reward) or not math.isfinite(self.algorithm_reward):
            raise ValueError("veRL rollout reward record values must be finite")


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


def read_verl_rollout_reward_records(path: Path) -> tuple[VerlRolloutRewardRecord, ...]:
    """Read the append-only, parent-replayed post-shaping reward journal."""

    if not path.is_file():
        return ()
    records: list[VerlRolloutRewardRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line, parse_constant=_reject_non_finite_constant)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid veRL rollout reward JSON at line {line_number}") from error
        if not isinstance(payload, dict):
            raise TypeError(f"veRL rollout reward line {line_number} must be a JSON object")
        trace_id = payload.get("trace_id")
        step = payload.get("step")
        task_reward = payload.get("task_reward")
        algorithm_reward = payload.get("algorithm_reward")
        if (
            not isinstance(trace_id, str)
            or isinstance(step, bool)
            or not isinstance(step, int)
            or isinstance(task_reward, bool)
            or not isinstance(task_reward, int | float)
            or isinstance(algorithm_reward, bool)
            or not isinstance(algorithm_reward, int | float)
        ):
            raise TypeError(f"veRL rollout reward line {line_number} has an invalid record")
        records.append(VerlRolloutRewardRecord(trace_id, step, float(task_reward), float(algorithm_reward)))
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


__all__ = [
    "VerlMetricRecord",
    "VerlRolloutRewardRecord",
    "read_verl_metric_records",
    "read_verl_rollout_reward_records",
]
