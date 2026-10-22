"""Audit native heldout62 bundles without replacing missing scores with zeros.

Example: --arm 8b=/tmp/results/8b --arm base=/tmp/results/base --output audit.json
Paired intervals use only tasks with five observed values in both arms and
must be labelled a complete-task sensitivity analysis when coverage is missing.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

METRICS = ("success", "partial_credit", "goal_rate", "harms", "harm_free")


def episode_scores(trace: dict[str, Any]) -> tuple[dict[str, float | None], int]:
    metrics = trace.get("metrics") or {}
    # Empty rewards on a truncated episode are absent observations, not zero.
    scored = metrics.get("assertions_scored")
    result: dict[str, float | None] = {
        "success": metrics.get("task_completed_correctly"),
        "partial_credit": metrics.get("assertions_passed", 0) / scored if scored else None,
        "goal_rate": None,
        "harms": None,
        "harm_free": None,
    }
    findings: dict[str, dict[str, Any]] = {}
    goals: dict[str, list[float]] = defaultdict(list)
    harms = duplicates = 0
    guard_observed = False
    for batch in trace.get("assessment_batches") or []:
        producer = (batch.get("run") or {}).get("producer_id", "")
        assessments = batch.get("assessments")
        if not isinstance(assessments, list):
            continue  # Compact lifecycle references; findings live in final batches.
        for assessment in assessments:
            identity = assessment["assessment_id"]
            if identity in findings:
                if findings[identity] != assessment:
                    raise ValueError(f"conflicting assessment identity {identity}")
                duplicates += 1
                continue
            findings[identity] = assessment
            signal = assessment.get("signal") or {}
            if assessment.get("status") != "valid" or signal.get("semantics") != "outcome":
                continue
            value = float(assessment["value"])
            if producer.endswith("manifest_obligations") and signal.get("direction") == "higher":
                goals[signal["signal_id"]].append(value)
            if producer.endswith("manifest_guards"):
                guard_observed = True
            if signal.get("direction") == "lower" and value >= 1:
                harms += 1
    if goals:
        result["goal_rate"] = sum(max(values) >= 1 for values in goals.values()) / len(goals)
    # A reviewed manifest with no guards has zero applicable harm findings;
    # distinguish that declared empty population from absent assessment evidence.
    no_declared_guards = False
    if not guard_observed and result["success"] is not None:
        for view in trace.get("assessment_views") or []:
            contract = json.loads(view.get("input_json", "{}")).get("contract")
            if contract is not None and "checks" in contract:
                no_declared_guards = not any(check.get("role") == "guard" for check in contract["checks"])
                break
    if (guard_observed or no_declared_guards) and trace.get("assessment_finalization_state") == "completed":
        result.update(harms=float(harms), harm_free=float(harms == 0))
    return result, duplicates


def audit(directory: Path, task_names: list[str]) -> dict[str, Any]:
    manifest_path = directory / "evaluation-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest is not None and [entry["task"]["key"] for entry in manifest["tasks"]] != task_names:
        raise ValueError("ordered evaluation manifest differs from frozen task fixture")
    path = directory / "traces.jsonl.gz"
    if not path.exists():
        path = directory / "traces.jsonl"
    opener = gzip.open if path.suffix == ".gz" else open
    by_task: dict[str, list[dict[str, float | None]]] = defaultdict(list)
    identities: set[tuple[str, int]] = set()
    episode_ids: set[str] = set()
    stops: Counter[str] = Counter()
    missing: list[dict[str, Any]] = []
    duplicates = errors = assessment_errors = 0
    error_types: Counter[str] = Counter()
    provider_error_nodes = 0
    finish_reasons: Counter[str] = Counter()
    with opener(path, "rt") as stream:
        for line in stream:
            episode = json.loads(line)
            task = episode["task"]["key"]
            repetition = episode["run"]["repetition_index"]
            identity = (task, repetition)
            if identity in identities or episode["id"] in episode_ids:
                raise ValueError("duplicate episode or task/repetition identity")
            identities.add(identity)
            episode_ids.add(episode["id"])
            if len(episode["traces"]) != 1:
                raise ValueError("expected one native trace per evaluation episode")
            trace = episode["traces"][0]
            stops[trace["stop_condition"]] += 1
            errors += len(episode.get("errors") or []) + len(trace.get("errors") or [])
            error_types.update(error.get("type", "unknown") for error in trace.get("errors") or [])
            assessment_errors += len(trace.get("assessment_errors") or [])
            provider_error_nodes += sum(
                (node.get("message") or {}).get("type") == "ProviderError" for node in trace.get("nodes", [])
            )
            finish_reasons.update(str(call.get("finish_reason")) for call in trace.get("calls", []))
            scores, dedup_count = episode_scores(trace)
            duplicates += dedup_count
            by_task[task].append(scores)
            if scores["success"] is None:
                missing.append({"task": task, "repetition": repetition, "stop": trace["stop_condition"]})
    if set(by_task) != set(task_names) or any(len(by_task[t]) != 5 for t in task_names):
        raise ValueError("task population or five-repetition coverage mismatch")
    summary = {}
    for metric in METRICS:
        observed = [value for episodes in by_task.values() for e in episodes if (value := e[metric]) is not None]
        per_task = {
            t: mean(values)
            for t, episodes in by_task.items()
            if (values := [value for e in episodes if (value := e[metric]) is not None])
        }
        complete = {t: per_task[t] for t in per_task if all(e[metric] is not None for e in by_task[t])}
        summary[metric] = {
            "observed_episodes": len(observed),
            "complete_tasks": complete,
            "strict_task_mean": mean(per_task.values()) if len(observed) == 310 else None,
            "available_repetition_task_mean": mean(per_task.values()) if per_task else None,
            "complete_task_domain_sensitivity": {
                domain: {
                    "total_tasks": sum(t.split(".", 1)[0] == domain for t in task_names),
                    "complete_tasks": len(values),
                    "mean": mean(values) if values else None,
                }
                for domain in sorted({t.split(".", 1)[0] for t in task_names})
                for values in [[value for t, value in complete.items() if t.split(".", 1)[0] == domain]]
            },
        }
        if metric == "goal_rate":
            summary[metric]["coverage_note"] = "Undefined without decided obligations; absence alone is not an error."
        if metric in ("success", "partial_credit", "harm_free"):
            summary[metric]["full_population_identification_bounds"] = [
                sum(observed) / 310,
                (sum(observed) + 310 - len(observed)) / 310,
            ]
    return {
        "path": str(path),
        "episodes": len(episode_ids),
        "tasks": len(by_task),
        "task_names": task_names,
        "ordered_manifest_verified": manifest is not None,
        "stops": dict(stops),
        "call_finish_reasons": dict(finish_reasons),
        "errors": errors,
        "trace_error_types": dict(error_types),
        "assessment_errors": assessment_errors,
        "provider_error_nodes": provider_error_nodes,
        "retry_count": None,
        "retry_count_note": "No authoritative model transport-attempt counter in this trace schema; errors are separate.",
        "deduplicated_assessment_copies": duplicates,
        "missing_success": missing,
        "metrics": summary,
    }


def bootstrap(arms: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, arm in arms.items():
        for reference, baseline in arms.items():
            if name == reference:
                continue
            for metric in METRICS:
                a = arm["metrics"][metric]["complete_tasks"]
                b = baseline["metrics"][metric]["complete_tasks"]
                tasks = sorted(set(a) & set(b))
                if not tasks:
                    continue
                av = np.array([a[t] for t in tasks])
                bv = np.array([b[t] for t in tasks])
                # Shared draws for the same metric/task cohort; adding another
                # arm or reversing a comparison must not change its interval.
                seed_key = "20261016|" + metric + "|" + "\n".join(tasks)
                rng = np.random.default_rng(int(hashlib.sha256(seed_key.encode()).hexdigest()[:16], 16))
                draws = rng.integers(len(tasks), size=(10000, len(tasks)))
                result[f"{name}-vs-{reference}/{metric}"] = {
                    "complete_tasks": len(tasks),
                    "excluded_tasks": sorted(set(arm["task_names"]) - set(tasks)),
                    "label": "complete-task sensitivity; conditional on observed coverage",
                    "arm_mean": float(av.mean()),
                    "reference_mean": float(bv.mean()),
                    "difference": float((av - bv).mean()),
                    "paired_difference_ci95": np.quantile((av - bv)[draws].mean(axis=1), [0.025, 0.975]).tolist(),
                    "absolute_arm_ci95": np.quantile(av[draws].mean(axis=1), [0.025, 0.975]).tolist(),
                    "absolute_reference_ci95": np.quantile(bv[draws].mean(axis=1), [0.025, 0.975]).tolist(),
                }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="label=bundle-directory")
    parser.add_argument(
        "--fixture", type=Path, default=Path("scripts/qualification/fixtures/automationbench_heldout62_v1.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    task_names = fixture["task_names"]
    if hashlib.sha256("\n".join(task_names).encode()).hexdigest() != fixture["selected_names_sha256"]:
        raise ValueError("frozen task fixture digest mismatch")
    arms = {label: audit(Path(directory), task_names) for label, directory in (a.split("=", 1) for a in args.arm)}
    result = {
        "fixture": fixture["id"],
        "selected_names_sha256": fixture["selected_names_sha256"],
        "environment_revision": fixture["environment_revision"],
        "arms": arms,
        "comparisons": bootstrap(arms),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
