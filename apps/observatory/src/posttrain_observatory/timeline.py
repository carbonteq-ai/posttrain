"""Where one rollout spent its wall-clock time, from native Verifiers timing.

A Verifiers trace records phase spans (``timing``: setup, agent, finalize,
scoring) and one span per model call (``calls[].time``). Model calls run on the
inference server's GPU; the time between them is the harness executing the
tool calls the previous reply requested, on CPU. Tool-result nodes are
timestamped when the next call commits, so gaps between call spans, not node
timestamps, measure tool time. A call span includes any queueing inside the
inference server, which the trace cannot separate from compute.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from .models import TraceTimeline, TraceTimelineSegment

_MIN_SEGMENT_MS = 1.0


def trace_timeline(payload: Mapping[str, Any]) -> TraceTimeline | None:
    """Return the rollout's phase and call timeline, or None without timing."""

    timing = payload.get("timing")
    if not isinstance(timing, Mapping):
        return None
    origin = _number(timing.get("start"))
    if origin is None:
        return None
    segments: list[TraceTimelineSegment] = []

    def add(
        kind: Literal["setup", "inference", "tools", "harness", "scoring"],
        start: float | None,
        end: float | None,
        **extra: Any,
    ) -> None:
        if start is None or end is None or end <= start:
            return
        duration = (end - start) * 1000
        if duration < _MIN_SEGMENT_MS and kind not in {"inference"}:
            return
        segments.append(
            TraceTimelineSegment(kind=kind, start_ms=(start - origin) * 1000, duration_ms=duration, **extra)
        )

    setup = _span(timing.get("setup"))
    agent = _span(timing.get("agent"))
    add("setup", *setup)

    calls = _call_spans(payload)
    raw_nodes = payload.get("nodes")
    nodes: list[Any] = raw_nodes if isinstance(raw_nodes, list) else []
    cursor = agent[0]
    for index, (call, start, end) in enumerate(calls):
        if cursor is not None:
            previous = calls[index - 1][0] if index else None
            tools = _requested_tools(previous, nodes)
            add("tools" if tools else "harness", cursor, start, tools=tools)
        raw_usage = call.get("usage")
        usage: Mapping[str, Any] = raw_usage if isinstance(raw_usage, Mapping) else {}
        add(
            "inference",
            start,
            end,
            call_index=index,
            node=_count(call.get("node")),
            prompt_tokens=_count(usage.get("prompt_tokens")),
            completion_tokens=_count(usage.get("completion_tokens")),
            thinking_tokens=_count(usage.get("reasoning_tokens")),
            finish_reason=call.get("finish_reason") if isinstance(call.get("finish_reason"), str) else None,
        )
        cursor = end
    if calls and cursor is not None:
        tools = _requested_tools(calls[-1][0], nodes)
        add("tools" if tools else "harness", cursor, agent[1], tools=tools)
    elif not calls:
        add("harness", *agent)

    finalize = _span(timing.get("finalize"))
    scoring = _span(timing.get("scoring"))
    add("scoring", finalize[0] if finalize[0] is not None else scoring[0], scoring[1] or finalize[1])

    ends = [value for value in (scoring[1], finalize[1], agent[1], setup[1]) if value is not None]
    ends.extend(end for _, _, end in calls)
    if not ends and not segments:
        return None
    total_ms = (max(ends) - origin) * 1000 if ends else max(s.start_ms + s.duration_ms for s in segments)

    def total(*kinds: str) -> float:
        return sum(segment.duration_ms for segment in segments if segment.kind in kinds)

    return TraceTimeline(
        total_ms=total_ms,
        inference_ms=total("inference"),
        tools_ms=total("tools", "harness"),
        setup_ms=total("setup"),
        scoring_ms=total("scoring"),
        model_calls=len(calls),
        segments=tuple(segments),
    )


def _call_spans(payload: Mapping[str, Any]) -> list[tuple[Mapping[str, Any], float, float]]:
    calls = payload.get("calls")
    if not isinstance(calls, list):
        return []
    spans: list[tuple[Mapping[str, Any], float, float]] = []
    for call in calls:
        if not isinstance(call, Mapping):
            continue
        start, end = _span(call.get("time"))
        if start is not None and end is not None and end >= start:
            spans.append((call, start, end))
    return sorted(spans, key=lambda item: item[1])


def _requested_tools(call: Mapping[str, Any] | None, nodes: Sequence[Any]) -> tuple[str, ...]:
    """Tool names the reply committed by ``call`` asked the harness to run."""

    if call is None:
        return ()
    index = call.get("node")
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(nodes):
        return ()
    node = nodes[index]
    message = node.get("message") if isinstance(node, Mapping) else None
    tool_calls = message.get("tool_calls") if isinstance(message, Mapping) else None
    if not isinstance(tool_calls, list):
        return ()
    names: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, Mapping):
            continue
        function = tool_call.get("function")
        name = function.get("name") if isinstance(function, Mapping) else tool_call.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return tuple(names)


def _span(value: Any) -> tuple[float | None, float | None]:
    if not isinstance(value, Mapping):
        return None, None
    return _number(value.get("start")), _number(value.get("end"))


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


__all__ = ["trace_timeline"]
