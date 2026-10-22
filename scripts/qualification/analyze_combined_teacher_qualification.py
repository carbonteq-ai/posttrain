"""Independent missing-aware audit of the paired teacher-context qualifier."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from qualify_combined_teacher_contexts import FAILURES, RETENTION


def native_summary(folder, expected):
    episodes = [json.loads(line) for line in (folder / "native/traces.jsonl").read_text().splitlines() if line]
    if Counter(e["task"]["data"]["name"] for e in episodes) != Counter({n: 2 for n in expected}):
        raise ValueError("native task/repetition cohort mismatch")
    if len({e["id"] for e in episodes}) != 8:
        raise ValueError("duplicate native IDs")
    values, errors, stops = defaultdict(lambda: defaultdict(list)), [], Counter()
    preservation, states = [], defaultdict(list)
    for episode in episodes:
        errors.extend(episode.get("errors", []))
        errors.extend(episode.get("assessment_errors", []))
        errors.extend(episode.get("credit_errors", []))
        if len(episode["traces"]) != 1:
            raise ValueError("unexpected trace count")
        trace = episode["traces"][0]
        errors.extend(trace.get("errors", []))
        errors.extend(trace.get("assessment_errors", []))
        errors.extend(trace.get("credit_errors", []))
        name = episode["task"]["data"]["name"]
        states[name].append(hashlib.sha256(json.dumps(
            episode["task"]["data"]["initial_state"], sort_keys=True,
            separators=(",", ":")).encode()).hexdigest())
        stops[trace["stop_condition"]] += 1
        metrics = trace.get("metrics", {})
        for metric in ["task_completed_correctly", "tool_mistakes", "tool_unknown_id"]:
            value = metrics.get(metric)
            if isinstance(value, (int, float)) and math.isfinite(value):
                values[metric][name].append(value)
        value = trace.get("rewards", {}).get("partial_credit", {}).get("score")
        if isinstance(value, (int, float)) and math.isfinite(value):
            values["partial_credit"][name].append(value)
        for assertion in trace.get("info", {}).get("automationbench", {}).get("assertions", []):
            kind = assertion.get("type", "")
            if "not_" in kind or "not_changed" in kind:
                preservation.append({"episode_id": episode["id"], "task": name, **assertion})
    calculated = {}
    saved = json.loads((folder / "outcomes.json").read_text())
    for metric in ["task_completed_correctly", "partial_credit", "tool_mistakes", "tool_unknown_id"]:
        groups = values[metric]
        means = {n: sum(v) / 2 for n, v in groups.items() if len(v) == 2}
        mean = sum(means.values()) / 4 if len(means) == 4 else None
        reported = saved["metrics"][metric]
        if reported["complete_task_means"] != means or reported["full_population_task_mean"] != mean:
            raise ValueError("saved summary/native trace disagreement")
        calculated[metric] = {"task_means": means, "task_mean": mean, "score_coverage": sum(map(len, groups.values()))}
    return {"episode_ids": [e["id"] for e in episodes], "metrics": calculated, "errors": errors,
            "stop_conditions": dict(stops), "legacy_preservation_checks": preservation,
            "initial_state_hashes": dict(states),
            "guard_scope": "Legacy assertion diagnostics only; not a full applicable-manifest harm audit."}


def analyze(root, output):
    receipt = json.loads((root / "receipt.json").read_text())
    if not receipt["runtime_completed"] or receipt["total_diagnostic_updates"] != 2:
        raise ValueError("paired runtime receipt incomplete")
    cohorts, all_ids = {}, []
    for step, arm in [(0, "baseline"), (1, "original_teacher"), (2, "combined_teacher")]:
        cohorts[arm] = {}
        for kind, expected in [("failures", FAILURES), ("retention", RETENTION)]:
            summary = native_summary(root / f"step-{step:04d}" / f"{arm}_{kind}", expected)
            all_ids.extend(summary["episode_ids"])
            cohorts[arm][kind] = summary
    if len(all_ids) != 48 or len(set(all_ids)) != 48:
        raise ValueError("expected48globallyunique native IDs")
    checks, table = {}, []
    checks["matched_initial_states"] = all(
        len({h for arm in cohorts for h in cohorts[arm][kind]["initial_state_hashes"][name]}) == 1
        for kind, names in [("failures", FAILURES), ("retention", RETENTION)] for name in names)
    for kind, names in [("failures", FAILURES), ("retention", RETENTION)]:
        for name in names:
            table.append({"cohort": kind, "task": name,
                          **{arm: {m: cohorts[arm][kind]["metrics"][m]["task_means"].get(name)
                                   for m in ["task_completed_correctly", "partial_credit", "tool_mistakes"]}
                             for arm in cohorts}})
        complete = all(cohorts[arm][kind]["metrics"][m]["score_coverage"] == 8
                       for arm in cohorts for m in ["task_completed_correctly", "partial_credit"])
        checks[kind + "_complete_coverage"] = complete
        for reference in ["baseline", "original_teacher"]:
            checks[f"{kind}_no_success_partial_regression_vs_{reference}"] = complete and all(
                cohorts["combined_teacher"][kind]["metrics"][m]["task_means"][name] >=
                cohorts[reference][kind]["metrics"][m]["task_means"][name]
                for name in names for m in ["task_completed_correctly", "partial_credit"])
    for reference in ["baseline", "original_teacher"]:
        for metric in ["task_completed_correctly", "partial_credit"]:
            a = cohorts["combined_teacher"]["failures"]["metrics"][metric]["task_mean"]
            b = cohorts[reference]["failures"]["metrics"][metric]["task_mean"]
            checks[f"failure_{metric}_improved_vs_{reference}"] = a is not None and b is not None and a > b
    checks["no_native_errors"] = not any(cohorts[a][k]["errors"] for a in cohorts for k in cohorts[a])
    updates = {a: json.loads((root / a / "update.json").read_text()) for a in ["original_teacher", "combined_teacher"]}
    checks["one_update_per_arm"] = all(v["optimizer_updates"] == 1 and v["optimizer_moment_steps"] == [1]
                                        for v in updates.values())
    checks["only_four_teacher_rows_changed"] = (updates["original_teacher"]["changed_teacher_rows"] == 0
                                                and updates["combined_teacher"]["changed_teacher_rows"] == 4)
    proposals = json.loads((root / "same-decision-proposals.json").read_text())
    checks["all_twelve_proposals_complete_parseable_replayable"] = (len(proposals) == 12 and all(
        p["complete"] and "native_replay" in p and not p.get("native_error") for p in proposals))
    proposal_summary = [{"mode": p["mode"], "task": p["task"], "complete": p["complete"],
                         "native_error": p.get("native_error"),
                         "decisions": p.get("native_replay", {}).get("decisions"),
                         "after_one_decision": p.get("native_replay", {}).get("scoring_after_one_decision")}
                        for p in proposals]
    result = {"cohorts": cohorts, "task_table": table, "checks": checks,
              "native_behavior_gate": all(checks.values()), "qualified": False,
              "proposal_summary": proposal_summary,
              "remaining_gates": ["independent proposal correctness review", "tool-token gradient/actual Adam audit", "adequate applicable guard audit"],
              "scope": "4failure+4retention training-task diagnostic;2greedy replicas are not independent task evidence. No heldout or training restart."}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"checks": checks, "task_table": table}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root, args.output)
