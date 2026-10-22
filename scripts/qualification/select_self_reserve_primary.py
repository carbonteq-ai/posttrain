"""Select a primary from local repair evidence before opening reserved outcomes."""

import argparse
import json
from pathlib import Path

from analyze_self_repair_run import FAILURES


def eligible(row):
    return (row["complete_scores"] and row["native_errors"] == 0
            and row["unknown_guard_checks"] == 0 and row["preservation_checks"] == 14
            and row["broken_preservation_checks"] == 0 and row["retention_success"] == 1
            and row["retention_partial"] == 1 and row["retention_agent_completed"] == 8
            and row["retention_repeats"] == 0)


def choose(rows):
    acceptable = [r for r in rows if eligible(r)]
    if not acceptable:
        raise ValueError("no eligible primary; do not substitute an unsafe or incomplete arm")
    ordered = sorted(acceptable, key=lambda r: (-r["qualified"], -r["success"], -r["partial"],
                                             r["failure_repeats"], r["step"], r["experiment"]))
    primary = ordered[0]
    return {"primary": primary["name"], "primary_qualified": primary["qualified"],
            "decision_basis": "Local repair/retention/preservation evidence only; reserved outcomes unopened. "
                              "Qualified first, then success, partial, fewer repeats, earlier step, lower experiment number.",
            "scope": "If no strict qualifier exists, the selected primary is exploratory, not promotion-ready.",
            "primary_row": primary, "all_local_rows": rows, "eligible_order": [r["name"] for r in ordered]}


def rows_from(evidence, experiment):
    audit = json.loads((evidence / f"adaptive{experiment:02d}-coverage-audit.json").read_text())
    guards = json.loads((evidence / f"adaptive{experiment:02d}-native-guards.json").read_text())
    margins = json.loads((evidence / f"adaptive{experiment:02d}-actual-update-margins.json").read_text())
    rows = []
    for step in [5, 10]:
        parts = {p["cohort"]: p for p in audit["phases"] if p["step"] == step}
        native = [p for p in guards["phases"] if p["phase"].startswith(f"step-{step:04d}/")]
        if set(parts) != {"failures", "retention"} or len(native) != 2:
            raise ValueError("missing local qualification population")
        complete = all(m["episode_coverage"] == 8 and len(m["complete_task_means"]) == 4
                       for p in parts.values() for k, m in p["summary"]["metrics"].items()
                       if k in ["task_completed_correctly", "partial_credit"])
        failure, retained = parts["failures"], parts["retention"]
        scores = failure["summary"]["metrics"]
        passed = scores["task_completed_correctly"]["complete_task_means"]
        partial = scores["partial_credit"]["complete_task_means"]
        before, after = margins["baseline"][:4], margins["actual_updates"][str(step)][:4]
        if set(x["task"] for x in before) != set(FAILURES) or len(after) != 4:
            raise ValueError("missing first-context margin evidence")
        improved = True
        for b, a in zip(before, after, strict=True):
            if b["task"] != a["task"] or b["history_tool_messages"] != 0 or a["history_tool_messages"] != 0:
                raise ValueError("first-context identities differ")
            improved &= a["correct_minus_bad_logit"] > b["correct_minus_bad_logit"]
        native_targets = (all(passed.get(n) == 1 for n in FAILURES if n != "hr.offboarding_automation")
                          and partial.get("hr.offboarding_automation", -1) >= 2 / 7)
        row = {"name": f"full_span{experiment:02d}_step{step}", "experiment": experiment, "step": step,
               "complete_scores": complete, "native_errors": sum(len(p["summary"]["errors"]) for p in parts.values()),
               "preservation_checks": sum(p["totals"]["preservation_checks"] for p in native),
               "unknown_guard_checks": sum(p["totals"]["unknown_initial_checks"] for p in native),
               "broken_preservation_checks": sum(p["totals"]["broken_preservation_checks"] for p in native),
               "retention_success": retained["summary"]["metrics"]["task_completed_correctly"]["full_population_task_mean"],
               "retention_partial": retained["summary"]["metrics"]["partial_credit"]["full_population_task_mean"],
               "retention_agent_completed": retained["summary"]["stop_conditions"].get("agent_completed", 0),
               "retention_repeats": retained["tool_observations"]["totals"]["repeated_exact_tool_calls"],
               "success": scores["task_completed_correctly"]["full_population_task_mean"],
               "partial": scores["partial_credit"]["full_population_task_mean"],
               "failure_repeats": failure["tool_observations"]["totals"]["repeated_exact_tool_calls"],
               "first_context_margins_improve": bool(improved), "native_targets": native_targets}
        row["qualified"] = bool(eligible(row) and improved and native_targets)
        rows.append(row)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = rows_from(args.evidence, 6) + rows_from(args.evidence, 9)
    with args.output.open("x") as stream:
        stream.write(json.dumps(choose(rows), indent=2) + "\n")
