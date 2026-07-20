"""Small observation summaries computed from authoritative Verifiers traces."""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any


def _trace_reward(trace: dict[str, Any]) -> float:
    rewards = trace.get("rewards", {})
    return float(sum(rewards.values())) if isinstance(rewards, dict) else 0.0


def _duration(span: Any) -> float:
    if not isinstance(span, dict):
        return 0.0
    start = float(span.get("start", 0.0) or 0.0)
    end = float(span.get("end", 0.0) or 0.0)
    return max(0.0, end - start) if end else 0.0


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _metric_name(value: Any) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "unknown")).strip("_")
    return normalized or "unknown"


def _averages(traces: list[dict[str, Any]], field: str, prefix: str) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for trace in traces:
        observed = trace.get(field, {})
        if not isinstance(observed, dict):
            continue
        for name, value in observed.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.setdefault(_metric_name(name), []).append(float(value))
    return {f"{prefix}/{name}": _mean(samples) for name, samples in values.items()}


def summarize_traces(path: Path) -> dict[str, int | float]:
    traces: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("each Verifiers trace must be a JSON object")
                traces.append(value)
    if not traces:
        raise ValueError(f"Verifiers produced no traces: {path}")

    rewards = [_trace_reward(trace) for trace in traces]
    errors = sum(bool(trace.get("errors")) for trace in traces)
    completed = sum(bool(trace.get("is_completed")) for trace in traces)
    calls = [
        call
        for trace in traces
        for call in trace.get("calls", [])
        if isinstance(call, dict)
    ]
    input_tokens = sum(
        int((call.get("usage") or {}).get("prompt_tokens", 0))
        + int((call.get("usage") or {}).get("cached_input_tokens", 0) or 0)
        for call in calls
    )
    output_tokens = sum(
        int((call.get("usage") or {}).get("completion_tokens", 0))
        for call in calls
    )
    cached_input_tokens = sum(
        int((call.get("usage") or {}).get("cached_input_tokens", 0) or 0) for call in calls
    )
    reasoning_tokens = sum(
        int((call.get("usage") or {}).get("reasoning_tokens", 0) or 0) for call in calls
    )
    agent_cost = sum(float((call.get("usage") or {}).get("cost", 0.0) or 0.0) for call in calls)
    extra_usage = [
        usage
        for trace in traces
        for usage in trace.get("extra_usage", [])
        if isinstance(usage, dict)
    ]
    extra_input_tokens = sum(
        int(usage.get("prompt_tokens", 0) or 0)
        + int(usage.get("cached_input_tokens", 0) or 0)
        for usage in extra_usage
    )
    extra_output_tokens = sum(int(usage.get("completion_tokens", 0) or 0) for usage in extra_usage)
    extra_reasoning_tokens = sum(int(usage.get("reasoning_tokens", 0) or 0) for usage in extra_usage)
    extra_cost = sum(float(usage.get("cost", 0.0) or 0.0) for usage in extra_usage)
    call_errors = sum(bool(call.get("error")) for call in calls)
    call_latencies = [_duration(call.get("time")) for call in calls]
    length_finished_calls = sum(call.get("finish_reason") == "length" for call in calls)
    truncating_stops = {
        "max_turns",
        "max_input_tokens",
        "max_output_tokens",
        "max_total_tokens",
        "context_length",
        "harness_timeout",
    }

    def is_truncated(trace: dict[str, Any]) -> bool:
        if trace.get("stop_condition") in truncating_stops:
            return True
        successful = [
            call
            for call in trace.get("calls", [])
            if isinstance(call, dict) and not call.get("error")
        ]
        return bool(successful and successful[-1].get("finish_reason") == "length")

    truncated_rollouts = sum(is_truncated(trace) for trace in traces)
    phase_durations = {
        phase: [_duration((trace.get("timing") or {}).get(phase)) for trace in traces]
        for phase in ("boot", "setup", "generation", "finalize", "scoring")
    }
    generation_model = [
        float((((trace.get("timing") or {}).get("generation") or {}).get("model") or {}).get("duration", 0.0) or 0.0)
        for trace in traces
    ]
    generation_harness = [
        float((((trace.get("timing") or {}).get("generation") or {}).get("harness") or {}).get("duration", 0.0) or 0.0)
        for trace in traces
    ]
    summary: dict[str, int | float] = {
        "eval/rollouts": len(traces),
        "eval/completed": completed,
        "eval/errors": errors,
        "eval/error_rate": errors / len(traces),
        "eval/mean_reward": statistics.fmean(rewards),
        "eval/min_reward": min(rewards),
        "eval/max_reward": max(rewards),
        "eval/input_tokens": input_tokens,
        "eval/output_tokens": output_tokens,
        "eval/cached_input_tokens": cached_input_tokens,
        "eval/reasoning_tokens": reasoning_tokens,
        "eval/provider_cost": agent_cost + extra_cost,
        "eval/agent_provider_cost": agent_cost,
        "eval/extra_usage_calls": len(extra_usage),
        "eval/extra_input_tokens": extra_input_tokens,
        "eval/extra_output_tokens": extra_output_tokens,
        "eval/extra_reasoning_tokens": extra_reasoning_tokens,
        "eval/extra_provider_cost": extra_cost,
        "eval/model_calls": len(calls),
        "eval/model_calls_per_rollout": len(calls) / len(traces),
        "eval/model_call_errors": call_errors,
        "eval/model_call_error_rate": call_errors / len(calls) if calls else 0.0,
        "eval/model_call_latency_mean_seconds": _mean(call_latencies),
        "eval/model_call_latency_p95_seconds": _percentile(call_latencies, 0.95),
        "eval/length_finished_calls": length_finished_calls,
        "eval/length_finish_rate": length_finished_calls / len(calls) if calls else 0.0,
        "eval/truncated_rollouts": truncated_rollouts,
        "eval/truncated_rollout_rate": truncated_rollouts / len(traces),
    }
    for phase, durations in phase_durations.items():
        summary[f"eval/timing/{phase}_mean_seconds"] = _mean(durations)
    summary["eval/timing/model_mean_seconds"] = _mean(generation_model)
    summary["eval/timing/harness_mean_seconds"] = _mean(generation_harness)

    for trace in traces:
        stop = _metric_name(trace.get("stop_condition"))
        key = f"eval/stop/{stop}"
        summary[key] = summary.get(key, 0) + 1
        for error in trace.get("errors", []):
            if isinstance(error, dict):
                key = f"eval/error_type/{_metric_name(error.get('type'))}"
                summary[key] = summary.get(key, 0) + 1

    summary.update(_averages(traces, "rewards", "eval/reward"))
    summary.update(_averages(traces, "metrics", "eval/environment_metric"))
    return summary


__all__ = ["summarize_traces"]
