"""Generic and Verifiers trace projections used by evaluation and GRPO views."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from statistics import fmean

from posttrain.common import JsonValue
from posttrain.tracking import RunDataSource, TraceQuery, TraceRecord

from .models import (
    EvaluationSlice,
    RewardComponent,
    TraceDetail,
    TraceEvaluationView,
    TraceSummary,
)
from .redaction import RedactionPolicy


def _number(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _integer(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _summary(record: TraceRecord) -> TraceSummary:
    payload = record.payload
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    reward = _number(payload.get("reward"))
    if reward is None:
        reward = _number(payload.get("score"))
    success = payload.get("success")
    if not isinstance(success, bool):
        success = None
    error = payload.get("error")
    error_text = str(error) if error not in (None, False, "") else None
    task = payload.get("task") or payload.get("task_id") or metadata.get("task")
    tool_calls = _integer(payload.get("num_tool_calls"))
    if tool_calls is None:
        calls = payload.get("tool_calls")
        tool_calls = len(calls) if isinstance(calls, list) else None
    return TraceSummary(
        external_id=record.external_id,
        trace_type=record.trace_type,
        task=str(task) if task is not None else None,
        reward=reward,
        success=success,
        truncated=bool(payload.get("truncated", False)),
        error=error_text,
        tool_calls=tool_calls,
        latency_ms=_number(payload.get("latency_ms")),
        tokens=_integer(payload.get("tokens")),
    )


def project_trace(record: TraceRecord, redaction: RedactionPolicy) -> TraceDetail:
    payload = redaction.mapping(record.payload)
    warning = None
    if record.trace_type != "verifiers":
        warning = f"No specialized projector is registered for trace type {record.trace_type!r}."
    components: list[RewardComponent] = []
    raw_components = payload.get("reward_components") or payload.get("rewards")
    if isinstance(raw_components, Mapping):
        for name, value in sorted(raw_components.items()):
            number = _number(value)
            if number is not None:
                components.append(RewardComponent(name=str(name), value=number))
    transcript_value = payload.get("messages") or payload.get("transcript") or payload.get("nodes")
    transcript: list[dict[str, JsonValue]] = []
    if isinstance(transcript_value, list):
        transcript = [dict(item) for item in transcript_value if isinstance(item, dict)]
    return TraceDetail(
        summary=_summary(record),
        reward_components=tuple(components),
        transcript=tuple(transcript),
        attributes=redaction.mapping(record.attributes),
        raw=payload,
        projection_warning=warning,
    )


async def trace_evaluation_view(
    source: RunDataSource,
    run_id: str,
    *,
    expected: int | None = None,
    trace_type: str = "verifiers",
    safety_limit: int = 5000,
) -> TraceEvaluationView:
    cursor: str | None = None
    records: list[TraceRecord] = []
    live = False
    while len(records) < safety_limit:
        page = await source.traces(
            run_id,
            TraceQuery(trace_type=trace_type, cursor=cursor, limit=min(1000, safety_limit - len(records))),
        )
        live = page.live
        records.extend(page.items)
        if page.next_cursor is None:
            cursor = None
            break
        cursor = page.next_cursor
    summaries = tuple(_summary(record) for record in records)
    rewards = [item.reward for item in summaries if item.reward is not None]
    successes = [item.success for item in summaries if item.success is not None]
    grouped: dict[str, list[TraceSummary]] = defaultdict(list)
    for item in summaries:
        grouped[item.task or "unspecified"].append(item)
    slices = []
    for key, values in sorted(grouped.items()):
        slice_rewards = [item.reward for item in values if item.reward is not None]
        slice_successes = [item.success for item in values if item.success is not None]
        slices.append(
            EvaluationSlice(
                key=key,
                count=len(values),
                mean_reward=fmean(slice_rewards) if slice_rewards else None,
                success_rate=(
                    sum(1 for value in slice_successes if value) / len(slice_successes) if slice_successes else None
                ),
            )
        )
    complete = cursor is None and (expected is None or len(records) >= expected)
    state = "complete" if complete else "partial"
    if not records and expected in (None, 0):
        state = "unavailable"
    return TraceEvaluationView(
        state=state,
        scanned=len(records),
        expected=expected,
        included=len(records),
        mean_reward=fmean(rewards) if rewards else None,
        success_rate=(sum(1 for value in successes if value) / len(successes) if successes else None),
        failures=sum(1 for item in summaries if item.error is not None),
        truncated=sum(1 for item in summaries if item.truncated),
        slices=tuple(slices),
        traces=summaries,
        next_cursor=cursor,
        live=live,
    )


async def get_trace_detail(
    source: RunDataSource,
    run_id: str,
    external_id: str,
    redaction: RedactionPolicy,
) -> TraceDetail:
    cursor: str | None = None
    while True:
        page = await source.traces(run_id, TraceQuery(cursor=cursor, limit=1000))
        for record in page.items:
            if record.external_id == external_id:
                return project_trace(record, redaction)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    raise LookupError(f"trace {external_id!r} was not found in run {run_id!r}")


__all__ = ["get_trace_detail", "project_trace", "trace_evaluation_view"]
