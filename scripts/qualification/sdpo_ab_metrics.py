"""Evaluation metrics for the SDPO teacher-context study (plan D11, D20).

The unit is the task: per-task success is the mean of `task_completed_correctly`
over its k sampled episodes, and comparisons are paired over the same tasks.
Truncated episodes and tool-error endings count as failures; episodes without a
native score are infrastructure failures and are reported, never scored.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20261027
DRAWS = 10000
COMPLETED_STOPS = {"agent_completed"}


def read_episodes(path):
    """Native episode records from a Verifiers `traces.jsonl`."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def trace_of(record):
    traces = record.get("traces")
    if traces is None:
        return record
    if len(traces) != 1:
        raise ValueError("expected exactly one native trace per episode")
    return traces[0]


def task_name(record):
    return trace_of(record)["task"]["data"]["name"]


NEGATIVE_MARKERS = ("_not_", "not_exists", "not_sent", "not_updated", "not_changed", "not_voided",
                    "not_has", "not_in_channel", "not_contains", "not_equals")


def negative_family(assertion_type):
    return any(marker in assertion_type for marker in NEGATIVE_MARKERS)


def manifest_findings(record):
    """Completed manifest assessments, flattened (producer, signal, subject, value, reason)."""
    trace = trace_of(record)
    rows = []
    for batch in trace.get("assessment_batches") or []:
        if batch.get("run", {}).get("status") != "complete":
            continue
        producer = batch["run"]["producer_id"]
        try:
            configuration = json.loads(batch["run"].get("configuration_json") or "{}")
        except ValueError:
            configuration = {}
        for assessment in batch.get("assessments") or []:
            subject = assessment.get("subject") or {}
            execution = subject.get("execution") or {}
            rows.append({"producer": producer, "signal": assessment["signal"]["signal_id"],
                         "semantics": assessment["signal"].get("semantics"),
                         "check_id": configuration.get("check_id"),
                         "candidate": configuration.get("candidate_identity"),
                         "direction": assessment["signal"].get("direction"),
                         "kind": subject.get("kind"), "invocation": execution.get("invocation_id"),
                         "value": assessment.get("value"), "status": assessment.get("status"),
                         "reason": assessment.get("reason")})
    return rows


def manifest_harms(findings):
    """Violated prohibited-effect checks: one per guard, from the trace-level compliance verdicts.

    A guard's compliance verdict is 1 when respected and 0 when violated. Retained-row verdicts
    also encode required final values, so they are not counted as harms."""
    return sorted({f["signal"] for f in findings
                   if f["kind"] == "trace" and f["status"] == "valid" and f["value"] == 0
                   and "guards" in f["producer"] and f.get("semantics") != "other"
                   and not f["signal"].endswith(".coverage")})


def manifest_goals(findings):
    goals = [f for f in findings if f["kind"] == "trace" and f["producer"].endswith("obligations")
             and not f["signal"].endswith(".coverage") and isinstance(f["value"], (int, float))]
    return sum(f["value"] for f in goals), len(goals)


def tool_error_text(content):
    """The error string of a tool result, or None."""
    text = content if isinstance(content, str) else json.dumps(content)
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            value = json.loads(stripped)
        except ValueError:
            value = None
        if isinstance(value, dict) and (value.get("success") is False or
                                        (set(value) <= {"error", "success", "details"} and "error" in value)):
            return str(value.get("error") or value.get("details") or stripped)
    lowered = stripped[:80].lower()
    if lowered.startswith(("error", "tool error", "invalid", "unknown tool")):
        return stripped
    return None


def messages_of(record):
    return [node["message"] for node in trace_of(record)["nodes"] if node.get("message")]


def episode_outcome(record, calls=None):
    """One episode's scored facts. `calls` are this episode's generation records, if kept."""
    trace = trace_of(record)
    metrics = trace.get("metrics") or {}
    success = metrics.get("task_completed_correctly")
    assertions = (trace.get("info") or {}).get("automationbench", {}).get("assertions")
    errors = list(record.get("errors") or []) + list(trace.get("errors") or [])
    messages = messages_of(record)
    tool_errors = sum(tool_error_text(m.get("content")) is not None for m in messages if m["role"] == "tool")
    findings = manifest_findings(record)
    harms = manifest_harms(findings)
    goals_met, goals = manifest_goals(findings)
    broken = [a["type"] for a in assertions or [] if not a.get("excluded") and not a.get("passed")
              and negative_family(a["type"])]
    limit = None
    if calls:
        last = calls[-1]
        if not last.get("complete", True):
            limit = last.get("limit_reason", "reply_limit")
    stop = trace.get("stop_condition")
    if limit is None and stop not in COMPLETED_STOPS:
        limit = stop
    scored = isinstance(success, (int, float)) and assertions is not None
    return {"task": task_name(record), "episode_id": record.get("id"),
            "success": float(success) if scored else None,
            "partial_credit": (trace.get("rewards") or {}).get("partial_credit", {}).get("score"),
            "infrastructure_failure": not scored, "errors": [str(e)[:500] for e in errors],
            "stop_condition": stop, "truncated_by": limit,
            "assistant_turns": sum(m["role"] == "assistant" for m in messages),
            "tool_errors": tool_errors, "manifest_harms": harms, "harm": bool(harms),
            "goals_met": goals_met, "goals": goals,
            "preservation_breaks": broken, "preservation_break": bool(broken)}


def task_means(outcomes, key="success"):
    groups = defaultdict(list)
    for outcome in outcomes:
        if outcome["success"] is None:
            continue
        value = outcome[key]
        groups[outcome["task"]].append(float(value) if not isinstance(value, bool) else float(value))
    return {name: sum(values) / len(values) for name, values in groups.items()}


def summary(outcomes, expected_tasks, attempts):
    """Population summary; infrastructure failures are listed, not scored."""
    counts = Counter(o["task"] for o in outcomes)
    missing = {name: attempts - counts.get(name, 0) for name in expected_tasks if counts.get(name, 0) < attempts}
    infra = [o["episode_id"] for o in outcomes if o["infrastructure_failure"]]
    means = task_means(outcomes)
    scored = [o for o in outcomes if o["success"] is not None]
    truncation = Counter(o["truncated_by"] for o in scored if o["truncated_by"])
    return {"episodes": len(outcomes), "scored_episodes": len(scored), "tasks": len(expected_tasks),
            "attempts": attempts, "missing_attempts": missing, "infrastructure_failures": infra,
            "successes": int(sum(o["success"] for o in scored)),
            "mean_task_success": (sum(means.get(n, 0.0) for n in expected_tasks) / len(expected_tasks)
                                  if expected_tasks else None),
            "task_success": {n: means.get(n) for n in expected_tasks},
            "harm_share": sum(o["harm"] for o in scored) / len(scored) if scored else None,
            "harms_per_episode": sum(len(o["manifest_harms"]) for o in scored) / len(scored) if scored else None,
            "preservation_break_share": sum(o["preservation_break"] for o in scored) / len(scored) if scored else None,
            "tool_errors_per_episode": sum(o["tool_errors"] for o in scored) / len(scored) if scored else None,
            "truncation_share": sum(truncation.values()) / len(scored) if scored else None,
            "truncation_by_limit": dict(truncation),
            "partial_credit": (sum(o["partial_credit"] or 0 for o in scored) / len(scored)) if scored else None,
            "goal_rate": (sum(o["goals_met"] for o in scored) / max(1, sum(o["goals"] for o in scored)))
            if scored else None}


def quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_bootstrap(before, after, draws=DRAWS, seed=SEED):
    """Mean paired difference over tasks with a 95% task-resampling interval."""
    if set(before) != set(after) or not before:
        raise ValueError("paired bootstrap needs identical non-empty task sets")
    names = sorted(before)
    rng = random.Random(seed)
    deltas = []
    for _ in range(draws):
        sample = rng.choices(names, k=len(names))
        deltas.append(sum(after[n] - before[n] for n in sample) / len(sample))
    up = sum(after[n] > before[n] for n in names)
    down = sum(after[n] < before[n] for n in names)
    mean_before = sum(before.values()) / len(names)
    mean_after = sum(after.values()) / len(names)
    return {"tasks": len(names), "before": mean_before, "after": mean_after, "delta": mean_after - mean_before,
            "delta_ci95": [quantile(deltas, 0.025), quantile(deltas, 0.975)], "draws": draws, "seed": seed,
            "tasks_up": up, "tasks_down": down, "tasks_same": len(names) - up - down,
            "up": sorted(n for n in names if after[n] > before[n]),
            "down": sorted(n for n in names if after[n] < before[n])}


def compare(control, treated, expected_tasks):
    """D18 comparison of two summaries on the same task set."""
    before = {n: control["task_success"][n] for n in expected_tasks}
    after = {n: treated["task_success"][n] for n in expected_tasks}
    if any(v is None for v in list(before.values()) + list(after.values())):
        raise ValueError("comparison needs a score for every task")
    result = paired_bootstrap(before, after)
    result["harm_share_change"] = treated["harm_share"] - control["harm_share"]
    result["preservation_break_share_change"] = (treated["preservation_break_share"]
                                                 - control["preservation_break_share"])
    result["truncation_share_change"] = treated["truncation_share"] - control["truncation_share"]
    return result


def successes_from_task_means(task_means_by_name, attempts):
    """Episode success count implied by complete per-task means (evidence validation)."""
    return round(sum(v * attempts for v in task_means_by_name.values()))
