"""Project runtime phase events onto existing provider system telemetry."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from statistics import fmean
from typing import Literal

from posttrain.tracking import (
    MetricPoint,
    MetricSeries,
    RunDetail,
    RuntimePhaseInterval,
    phase_at,
    runtime_phase_intervals,
)

from .models import (
    PhaseMetricAggregate,
    RuntimePhaseIntervalView,
    RuntimePhaseSegment,
    RuntimePhaseSummary,
)

_PHASE_LABELS = {
    "operation": "Operation",
    "model_loading": "Model loading",
    "model_offloading": "Model offloading",
    "data_preparation": "Data preparation",
    "runtime_initialization": "Runtime initialization",
    "benchmark_warmup": "Warmup",
    "benchmark_measurement": "Measured inference",
    "runtime_cleanup": "Runtime cleanup",
    "rollout": "Rollout generation",
    "reward_scoring": "Reward scoring",
    "teacher_scoring": "Teacher scoring",
    "actor_update": "Actor update",
    "checkpointing": "Checkpointing",
    "evaluation": "Validation",
    "artifact_export": "Artifact export",
    "backend_execution": "Backend execution",
    "unclassified": "Unclassified",
}

_PHASE_GROUPS = {
    "operation": ("run", "Run"),
    "model_loading": ("startup", "Startup"),
    "model_offloading": ("finalization", "Finalization"),
    "data_preparation": ("startup", "Startup"),
    "runtime_initialization": ("startup", "Startup"),
    "benchmark_warmup": ("inference", "Inference"),
    "benchmark_measurement": ("inference", "Inference"),
    "runtime_cleanup": ("finalization", "Finalization"),
    "rollout": ("training", "Training"),
    "reward_scoring": ("training", "Training"),
    "teacher_scoring": ("training", "Training"),
    "actor_update": ("training", "Training"),
    "checkpointing": ("training", "Training"),
    "evaluation": ("evaluation", "Evaluation"),
    "artifact_export": ("finalization", "Finalization"),
    "backend_execution": ("execution", "Execution"),
    "unclassified": ("unclassified", "Unclassified"),
}

_PHASE_METRICS = {
    "system/gpu_vram_used_bytes": ("GPU memory", "bytes"),
    "system/gpu_utilization": ("GPU utilization", "%"),
    "system/cpu_percent": ("CPU utilization", "%"),
    "system/process_rss_bytes": ("Process memory", "bytes"),
}

type PhaseProjectionState = Literal["available", "partial", "unavailable"]


@dataclass(frozen=True, slots=True)
class RuntimePhaseProjection:
    state: PhaseProjectionState
    intervals: tuple[RuntimePhaseIntervalView, ...] = ()
    segments: tuple[RuntimePhaseSegment, ...] = ()
    summary: tuple[RuntimePhaseSummary, ...] = ()
    issues: tuple[str, ...] = ()
    unclassified_sample_count: int = 0
    vram_observed_peak_bytes: float | None = None


@dataclass(frozen=True, slots=True)
class _Segment:
    phase: str
    phase_id: str
    status: str
    started_at: datetime
    finished_at: datetime


def _phase_label(phase: str) -> str:
    return _PHASE_LABELS.get(phase, phase.replace("_", " ").title())


def _phase_group(phase: str) -> tuple[str, str]:
    return _PHASE_GROUPS.get(phase, ("other", "Other"))


def _effective_segments(
    intervals: tuple[RuntimePhaseInterval, ...],
    *,
    started_at: datetime,
    finished_at: datetime,
) -> tuple[_Segment, ...]:
    boundaries = {started_at, finished_at}
    for interval in intervals:
        interval_end = interval.finished_at or finished_at
        clipped_start = max(started_at, interval.started_at)
        clipped_end = min(finished_at, interval_end)
        if clipped_start < clipped_end:
            boundaries.update((clipped_start, clipped_end))

    ordered = sorted(boundaries)
    segments: list[_Segment] = []
    for index, boundary in enumerate(ordered[:-1]):
        next_boundary = ordered[index + 1]
        if boundary >= next_boundary:
            continue
        midpoint = boundary + (next_boundary - boundary) / 2
        interval = phase_at(intervals, midpoint)
        segment = _Segment(
            phase=interval.phase if interval is not None else "unclassified",
            phase_id=interval.phase_id if interval is not None else "unclassified",
            status=interval.status if interval is not None else "unclassified",
            started_at=boundary,
            finished_at=next_boundary,
        )
        if (
            segments
            and segments[-1].phase_id == segment.phase_id
            and segments[-1].status == segment.status
            and segments[-1].finished_at == segment.started_at
        ):
            previous = segments[-1]
            segments[-1] = _Segment(
                phase=previous.phase,
                phase_id=previous.phase_id,
                status=previous.status,
                started_at=previous.started_at,
                finished_at=segment.finished_at,
            )
        else:
            segments.append(segment)
    return tuple(segments)


def _segment_index(segments: tuple[_Segment, ...], observed_at: datetime) -> int | None:
    for index, segment in enumerate(segments):
        is_last = index == len(segments) - 1
        if segment.started_at <= observed_at < segment.finished_at or (is_last and observed_at == segment.finished_at):
            return index
    return None


def _aggregate(metric: str, points: list[MetricPoint]) -> PhaseMetricAggregate:
    values = [point.value for point in points]
    label, unit = _PHASE_METRICS[metric]
    return PhaseMetricAggregate(
        metric=metric,
        label=label,
        unit=unit,
        mean=float(fmean(values)),
        peak=float(max(values)),
        minimum=float(min(values)),
        samples=len(values),
    )


def project_runtime_phases(
    detail: RunDetail,
    series: dict[str, MetricSeries],
) -> RuntimePhaseProjection:
    """Compute non-overlapping phase windows and per-phase host aggregates."""

    timestamped_points = tuple(
        point
        for name, values in series.items()
        if name.startswith("system/")
        for point in values.points
        if point.observed_at is not None
    )
    latest_sample = max(
        (point.observed_at for point in timestamped_points if point.observed_at is not None),
        default=None,
    )
    finished_at = detail.summary.finished_at or latest_sample
    interval_set = runtime_phase_intervals(
        detail.events,
        window_finished_at=detail.summary.finished_at,
    )
    if not interval_set.intervals or finished_at is None or finished_at <= detail.summary.started_at:
        return RuntimePhaseProjection(
            state="unavailable",
            issues=interval_set.issues,
            vram_observed_peak_bytes=_vram_observed_peak(series),
        )

    segments = _effective_segments(
        interval_set.intervals,
        started_at=detail.summary.started_at,
        finished_at=finished_at,
    )
    if not segments:
        return RuntimePhaseProjection(
            state="unavailable",
            issues=interval_set.issues,
            vram_observed_peak_bytes=_vram_observed_peak(series),
        )

    projected_intervals = tuple(
        RuntimePhaseIntervalView(
            phase=interval.phase,
            phase_id=interval.phase_id,
            label=_phase_label(interval.phase),
            group=_phase_group(interval.phase)[0],
            group_label=_phase_group(interval.phase)[1],
            status=interval.status,
            started_at=max(interval.started_at, detail.summary.started_at),
            finished_at=min(interval.finished_at or finished_at, finished_at),
            start_offset_s=float(
                (max(interval.started_at, detail.summary.started_at) - detail.summary.started_at).total_seconds()
            ),
            end_offset_s=float(
                (min(interval.finished_at or finished_at, finished_at) - detail.summary.started_at).total_seconds()
            ),
            duration_s=float(
                (
                    min(interval.finished_at or finished_at, finished_at)
                    - max(interval.started_at, detail.summary.started_at)
                ).total_seconds()
            ),
        )
        for interval in interval_set.intervals
        if max(interval.started_at, detail.summary.started_at) <= min(interval.finished_at or finished_at, finished_at)
    )

    points_by_segment: dict[int, dict[str, list[MetricPoint]]] = defaultdict(lambda: defaultdict(list))
    sample_times_by_segment: dict[int, set[datetime]] = defaultdict(set)
    unassigned_times: set[datetime] = set()
    untimestamped_count = 0
    for metric, values in series.items():
        if metric not in _PHASE_METRICS:
            continue
        untimestamped_count = max(
            untimestamped_count,
            sum(point.observed_at is None for point in values.points),
        )
        for point in values.points:
            if point.observed_at is None:
                continue
            index = _segment_index(segments, point.observed_at)
            if index is None:
                unassigned_times.add(point.observed_at)
                continue
            points_by_segment[index][metric].append(point)
            sample_times_by_segment[index].add(point.observed_at)

    projected = tuple(
        RuntimePhaseSegment(
            phase=segment.phase,
            phase_id=segment.phase_id,
            label=_phase_label(segment.phase),
            group=_phase_group(segment.phase)[0],
            group_label=_phase_group(segment.phase)[1],
            status=segment.status,  # type: ignore[arg-type]
            started_at=segment.started_at,
            finished_at=segment.finished_at,
            start_offset_s=float((segment.started_at - detail.summary.started_at).total_seconds()),
            end_offset_s=float((segment.finished_at - detail.summary.started_at).total_seconds()),
            duration_s=float((segment.finished_at - segment.started_at).total_seconds()),
            sample_count=len(sample_times_by_segment[index]),
            metrics=tuple(
                _aggregate(metric, metric_points)
                for metric, metric_points in sorted(points_by_segment[index].items())
                if metric_points
            ),
        )
        for index, segment in enumerate(segments)
    )

    phase_indices: dict[str, list[int]] = defaultdict(list)
    for index, segment in enumerate(segments):
        phase_indices[segment.phase].append(index)
    summaries = []
    for phase, indices in sorted(phase_indices.items(), key=lambda item: item[1][0]):
        metric_points: dict[str, list[MetricPoint]] = defaultdict(list)
        sample_times: set[datetime] = set()
        for index in indices:
            sample_times.update(sample_times_by_segment[index])
            for metric, values in points_by_segment[index].items():
                metric_points[metric].extend(values)
        summaries.append(
            RuntimePhaseSummary(
                phase=phase,
                label=_phase_label(phase),
                group=_phase_group(phase)[0],
                group_label=_phase_group(phase)[1],
                duration_s=float(sum(projected[index].duration_s for index in indices)),
                occurrences=len({segments[index].phase_id for index in indices}),
                sample_count=len(sample_times),
                metrics=tuple(_aggregate(metric, values) for metric, values in sorted(metric_points.items()) if values),
            )
        )

    unclassified = (
        untimestamped_count
        + len(unassigned_times)
        + sum(segment.sample_count for segment in projected if segment.phase == "unclassified")
    )
    issues = list(interval_set.issues)
    if untimestamped_count:
        issues.append(f"{untimestamped_count} system samples have no provider timestamp")
    if unassigned_times:
        issues.append(f"{len(unassigned_times)} timestamped system samples fall outside the run window")
    if any(segment.phase == "unclassified" for segment in projected):
        issues.append("runtime phase events do not cover the complete run window")
    state: PhaseProjectionState = "partial" if issues or unclassified else "available"
    return RuntimePhaseProjection(
        state=state,
        intervals=projected_intervals,
        segments=projected,
        summary=tuple(summaries),
        issues=tuple(issues),
        unclassified_sample_count=unclassified,
        vram_observed_peak_bytes=_vram_observed_peak(series),
    )


def _vram_observed_peak(series: dict[str, MetricSeries]) -> float | None:
    values = series.get("system/gpu_vram_used_bytes")
    if values is None or not values.points:
        return None
    return float(max(point.value for point in values.points))


__all__ = ["RuntimePhaseProjection", "project_runtime_phases"]
