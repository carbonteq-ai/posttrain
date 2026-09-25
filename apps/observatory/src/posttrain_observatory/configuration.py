"""Configuration review for a run, computed from its recorded selections.

Observatory runs the shared advisor (``posttrain.advisor``) on each run's
recorded resolved-input snapshot: the performance and LoRA/training rules, and
the settings calculator with its engine recommendations and step options. It
does not read anything the launcher computed, so runs recorded before the rules
existed are reviewed the same way as new ones. The calculator needs the
checkpoint's ``config.json``; without an architecture loader the review holds
rule findings only.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from posttrain.advisor import ArchitectureLoader, review
from posttrain.common import ConfigurationIssue

from .models import (
    ConfigurationFinding,
    ConfigurationReview,
    MetricSeries,
    RecommendedSetting,
    SettingsRecommendation,
    StepCalibration,
    StepCapacityView,
    StepOptionView,
)

ROLLOUT_SECONDS = "train/rl/time/rollout_seconds"
STEP_SECONDS = "train/step_time_seconds"
COMPLETION_TOKENS = "train/rl/completion_tokens_mean"


def _recorded_value(snapshot: Mapping[str, Any], path: str) -> Any:
    role, *fields = path.split(".")
    entry = snapshot.get(role)
    value: Any = entry.get("resolved", entry) if isinstance(entry, Mapping) else None
    for name in fields:
        if not isinstance(value, Mapping):
            return None
        value = value.get(name)
    return value


def _finding(issue: ConfigurationIssue, snapshot: Mapping[str, Any]) -> ConfigurationFinding:
    return ConfigurationFinding(
        code=issue.code,
        severity=issue.severity,
        source="calculator" if issue.code.startswith("CALCULATOR_") else "rule",
        role=issue.role,
        path=issue.path,
        value=_recorded_value(snapshot, issue.path),
        message=issue.message,
        hint=issue.hint,
        related_paths=issue.related_paths,
    )


def _mean(series: MetricSeries | None) -> tuple[float, int] | None:
    values = [point.value for point in series.points if point.value is not None] if series is not None else []
    return (sum(values) / len(values), len(values)) if values else None


def step_calibration(series: Mapping[str, MetricSeries]) -> StepCalibration | None:
    rollout = _mean(series.get(ROLLOUT_SECONDS))
    if rollout is None or rollout[0] <= 0:
        return None
    step = _mean(series.get(STEP_SECONDS))
    tokens = _mean(series.get(COMPLETION_TOKENS))
    return StepCalibration(
        rollout_seconds=round(rollout[0], 1),
        # Rollout time is recorded once per round; steps are recorded once each.
        rounds_per_step=round(max(rollout[1] / step[1], 1.0), 2) if step is not None else 1.0,
        step_seconds=round(step[0], 1) if step is not None and step[0] > 0 else None,
        completion_tokens=round(tokens[0], 1) if tokens is not None and tokens[0] > 0 else None,
        steps=step[1] if step is not None else rollout[1],
    )


def _decoding_seconds(calibration: StepCalibration, current_rows: int, decode_bound: float | None) -> float:
    """Seconds of a measured round the engine spends decoding at the memory-bandwidth bound.

    The remainder of the round is tool calls, prefill and waiting for the slowest
    episode; it repeats per wave but does not grow with more parallel episodes.
    Without a bound the whole round is treated as decoding (the pessimistic case).
    """

    if decode_bound is None or decode_bound <= 0 or calibration.completion_tokens is None:
        return calibration.rollout_seconds
    per_sequence_rate = decode_bound / max(current_rows, 1)
    return min(calibration.completion_tokens / per_sequence_rate, calibration.rollout_seconds)


def _step(raw: Any, calibration: StepCalibration | None, decode_bound: float | None = None) -> StepCapacityView | None:
    if not isinstance(raw, Mapping):
        return None
    current_rows = raw.get("step_sequences") or 1
    options = []
    for option in raw.get("options") or ():
        rollout_seconds = step_seconds = throughput = None
        if calibration is not None:
            decoding = _decoding_seconds(calibration, current_rows, decode_bound)
            waiting = calibration.rollout_seconds - decoding
            per_round = waiting * option["waves"] + decoding * option["relative_step_time"]
            rollout_seconds = round(per_round * calibration.rounds_per_step, 1)
            if calibration.step_seconds is not None:
                # The LoRA update, reward scoring and weight sync scale with rows.
                measured_rollout = calibration.rollout_seconds * calibration.rounds_per_step
                other = max(calibration.step_seconds - measured_rollout, 0.0)
                step_seconds = round(rollout_seconds + other * option["rows"] / current_rows, 1)
                throughput = round((option["rows"] / current_rows) / (step_seconds / calibration.step_seconds), 2)
        options.append(
            StepOptionView(
                **option,
                estimated_rollout_seconds=rollout_seconds,
                estimated_step_seconds=step_seconds,
                estimated_relative_rows_per_second=throughput,
            )
        )
    return StepCapacityView(**{**raw, "options": tuple(options)})


def _recommendation(role: str, raw: Mapping[str, Any], calibration: StepCalibration | None) -> SettingsRecommendation:
    if "unavailable" in raw:
        return SettingsRecommendation(role=role, state="unavailable", unavailable_reason=str(raw["unavailable"]))
    return SettingsRecommendation(
        role=role,
        state="available",
        binding_id=raw.get("binding_id"),
        model=raw.get("model"),
        hardware=dict(raw.get("hardware") or {}),
        task=dict(raw.get("task") or {}),
        settings=tuple(RecommendedSetting(**row) for row in raw.get("settings") or ()),
        environment={str(key): str(value) for key, value in (raw.get("environment") or {}).items()},
        memory_gb={str(key): float(value) for key, value in (raw.get("memory_gb") or {}).items()},
        max_concurrency=raw.get("max_concurrency"),
        decode_tokens_per_s_upper_bound=raw.get("decode_tokens_per_s_upper_bound"),
        notes=tuple(raw.get("notes") or ()),
        step=_step(raw.get("step"), calibration, raw.get("decode_tokens_per_s_upper_bound")),
    )


def configuration_review_sync(
    snapshot: Mapping[str, Any],
    loader: ArchitectureLoader | None,
    calibration: StepCalibration | None = None,
) -> ConfigurationReview:
    result = review(snapshot, loader)
    return ConfigurationReview(
        findings=tuple(_finding(issue, snapshot) for issue in result.findings),
        recommendations=tuple(
            _recommendation(role, payload, calibration) for role, payload in sorted(result.recommendations.items())
        ),
        calculator="available" if loader is not None else "disabled",
        calibration=calibration,
    )


async def configuration_review(
    snapshot: Mapping[str, Any],
    loader: ArchitectureLoader | None,
    series: Sequence[MetricSeries] = (),
) -> ConfigurationReview:
    """Review off the event loop: the loader may read a model config over the network."""

    calibration = step_calibration({item.name: item for item in series})
    return await asyncio.to_thread(configuration_review_sync, snapshot, loader, calibration)


__all__ = ["ROLLOUT_SECONDS", "STEP_SECONDS", "configuration_review", "configuration_review_sync", "step_calibration"]
