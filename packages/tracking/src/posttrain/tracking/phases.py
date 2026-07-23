"""Runtime-phase projection over provider-neutral event evidence."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from .models import EventRecord, TrackingModel

type RuntimePhaseStatus = Literal["running", "completed", "failed", "incomplete"]


class RuntimePhaseInterval(TrackingModel):
    phase: str = Field(min_length=1)
    phase_id: str = Field(min_length=1)
    started_at: datetime
    finished_at: datetime | None = None
    status: RuntimePhaseStatus

    @model_validator(mode="after")
    def validate_interval(self) -> RuntimePhaseInterval:
        if self.started_at.tzinfo is None or (self.finished_at is not None and self.finished_at.tzinfo is None):
            raise ValueError("runtime phase timestamps must be timezone-aware")
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("runtime phase cannot finish before it starts")
        return self

    def contains(self, observed_at: datetime) -> bool:
        return self.started_at <= observed_at and (self.finished_at is None or observed_at <= self.finished_at)


class RuntimePhaseIntervalSet(TrackingModel):
    intervals: tuple[RuntimePhaseInterval, ...] = ()
    issues: tuple[str, ...] = ()


def runtime_phase_intervals(
    events: tuple[EventRecord, ...],
    *,
    window_finished_at: datetime | None,
) -> RuntimePhaseIntervalSet:
    """Pair phase boundary events while preserving malformed-evidence issues."""

    starts: dict[str, EventRecord] = {}
    intervals: list[RuntimePhaseInterval] = []
    issues: list[str] = []
    for event in sorted(events, key=lambda item: item.occurred_at):
        if event.name not in {
            "runtime_phase_started",
            "runtime_phase_completed",
            "runtime_phase_failed",
        }:
            continue
        phase = event.attributes.get("phase")
        phase_id = event.attributes.get("phase_id")
        if not isinstance(phase, str) or not phase or not isinstance(phase_id, str) or not phase_id:
            issues.append(f"{event.name} is missing a string phase and phase_id")
            continue
        if event.name == "runtime_phase_started":
            if phase_id in starts:
                issues.append(f"duplicate runtime phase start for {phase_id}")
                continue
            starts[phase_id] = event
            continue
        started = starts.pop(phase_id, None)
        if started is None:
            issues.append(f"{event.name} has no start for {phase_id}")
            continue
        started_phase = started.attributes["phase"]
        if phase != started_phase:
            issues.append(f"runtime phase {phase_id} changed from {started_phase!r} to {phase!r}")
        intervals.append(
            RuntimePhaseInterval(
                phase=str(started_phase),
                phase_id=phase_id,
                started_at=started.occurred_at,
                finished_at=event.occurred_at,
                status="completed" if event.name == "runtime_phase_completed" else "failed",
            )
        )
    for phase_id, started in starts.items():
        intervals.append(
            RuntimePhaseInterval(
                phase=str(started.attributes["phase"]),
                phase_id=phase_id,
                started_at=started.occurred_at,
                finished_at=window_finished_at,
                status="incomplete" if window_finished_at is not None else "running",
            )
        )
        if window_finished_at is not None:
            issues.append(f"runtime phase {phase_id} has no terminal event")
    intervals.sort(key=lambda item: (item.started_at, item.finished_at or item.started_at, item.phase_id))
    return RuntimePhaseIntervalSet(intervals=tuple(intervals), issues=tuple(issues))


def phase_at(
    intervals: tuple[RuntimePhaseInterval, ...],
    observed_at: datetime | None,
) -> RuntimePhaseInterval | None:
    """Return the most specific active phase for a timestamped sample."""

    if observed_at is None:
        return None
    candidates = tuple(interval for interval in intervals if interval.contains(observed_at))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda interval: (
            interval.started_at,
            -(interval.finished_at - interval.started_at).total_seconds()
            if interval.finished_at is not None
            else float("-inf"),
        ),
    )


__all__ = [
    "RuntimePhaseInterval",
    "RuntimePhaseIntervalSet",
    "RuntimePhaseStatus",
    "phase_at",
    "runtime_phase_intervals",
]
