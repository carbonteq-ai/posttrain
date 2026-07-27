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

_TRUNCATED_STOP_CONDITIONS = frozenset(
    {
        "max_turns",
        "max_input_tokens",
        "max_output_tokens",
        "max_total_tokens",
        "context_length",
        "harness_timeout",
    }
)


def _number(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _integer(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _wire_reward(payload: Mapping[str, JsonValue]) -> float | None:
    reward = _number(payload.get("reward"))
    if reward is None:
        reward = _number(payload.get("score"))
    if reward is not None:
        return reward
    components = payload.get("rewards")
    if not isinstance(components, Mapping):
        return None
    values = [_number(value) for value in components.values()]
    numbers = [value for value in values if value is not None]
    return sum(numbers) if numbers else None


def _wire_success(payload: Mapping[str, JsonValue]) -> bool | None:
    success = payload.get("success")
    if isinstance(success, bool):
        return success
    for container_name in ("metrics", "rewards"):
        container = payload.get(container_name)
        if not isinstance(container, Mapping):
            continue
        for field in ("success", "correct"):
            value = container.get(field)
            if isinstance(value, bool):
                return value
            number = _number(value)
            if number is not None:
                return number > 0
    return None


def _wire_error(payload: Mapping[str, JsonValue]) -> str | None:
    error = payload.get("error")
    if error not in (None, False, ""):
        return str(error)
    errors = payload.get("errors")
    if not isinstance(errors, list) or not errors:
        return None
    latest = errors[-1]
    if isinstance(latest, Mapping):
        error_type = latest.get("type")
        return str(error_type) if error_type else "trace reported an error"
    return "trace reported an error"


def _wire_truncated(payload: Mapping[str, JsonValue]) -> bool:
    explicit = payload.get("truncated")
    if isinstance(explicit, bool):
        return explicit
    explicit = payload.get("is_truncated")
    if isinstance(explicit, bool):
        return explicit
    if payload.get("stop_condition") in _TRUNCATED_STOP_CONDITIONS:
        return True
    calls = payload.get("calls")
    if not isinstance(calls, list):
        return False
    for call in reversed(calls):
        if not isinstance(call, Mapping) or call.get("error") not in (None, False, ""):
            continue
        return call.get("finish_reason") == "length"
    return False


def _wire_tool_calls(payload: Mapping[str, JsonValue]) -> int | None:
    direct = _integer(payload.get("num_tool_calls"))
    if direct is not None:
        return direct
    calls = payload.get("tool_calls")
    if isinstance(calls, list):
        return len(calls)
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        return None
    count = 0
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        message = node.get("message")
        if not isinstance(message, Mapping):
            continue
        nested_calls = message.get("tool_calls")
        if isinstance(nested_calls, list):
            count += len(nested_calls)
    return count


def _task_scalar(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None


def _task_from_record(value: object) -> str | None:
    scalar = _task_scalar(value)
    if scalar is not None:
        return scalar
    if not isinstance(value, Mapping):
        return None
    task_type = _task_scalar(value.get("type"))
    data = value.get("data")
    data = data if isinstance(data, Mapping) else value
    identity = next(
        (
            scalar
            for field in ("id", "task_id", "example_id", "name", "idx")
            if (scalar := _task_scalar(data.get(field))) is not None
        ),
        None,
    )
    if task_type is not None and identity is not None:
        return f"{task_type}:{identity}"
    return identity or task_type


def _wire_task(
    payload: Mapping[str, JsonValue],
    metadata: Mapping[str, JsonValue],
    info: Mapping[str, JsonValue],
) -> str | None:
    for value in (
        payload.get("task_id"),
        info.get("example_id"),
        info.get("task_id"),
        metadata.get("task_id"),
        metadata.get("task"),
        payload.get("task"),
        info.get("task"),
    ):
        if (task := _task_from_record(value)) is not None:
            return task
    return None


def _summary(record: TraceRecord) -> TraceSummary:
    payload = record.payload
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    info = payload.get("info")
    info = info if isinstance(info, Mapping) else {}
    return TraceSummary(
        external_id=record.external_id,
        trace_type=record.trace_type,
        task=_wire_task(payload, metadata, info),
        reward=_wire_reward(payload),
        success=_wire_success(payload),
        truncated=_wire_truncated(payload),
        error=_wire_error(payload),
        tool_calls=_wire_tool_calls(payload),
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
