"""Development-first primary selection without reading research-reserve outcomes."""

import datetime
import json
import math

from sdpo_agent_config import MODEL_REVISION
from sdpo_execution_budget import reserve_profile


def candidate(audit, guards, step, metadata, review):
    if any(p["cohort"] == "reserve" for p in audit["populations"]):
        raise ValueError("reserve outcomes cannot enter primary selection")
    reserve_profile(metadata)
    reasons, before, after = [], {}, {}
    for group in ("train", "development"):
        populations = [p for p in audit["populations"] if p["cohort"] == group]
        first = next((p for p in populations if p["step"] == 0), None)
        last = next((p for p in populations if p["step"] == step), None)
        if first is None or last is None:
            reasons.append(f"{group}:missing_native_population")
            continue
        before[group], after[group] = first["summary"], last["summary"]
        for metric in ("task_completed_correctly", "partial_credit"):
            old, new = first["summary"]["metrics"][metric], last["summary"]["metrics"][metric]
            if old["full_population_task_mean"] is None or new["full_population_task_mean"] is None:
                reasons.append(f"{group}:{metric}:missing_score_coverage")
                continue
            if not all(math.isfinite(v) for v in [old["full_population_task_mean"], new["full_population_task_mean"],
                    *old["complete_task_means"].values(), *new["complete_task_means"].values()]):
                reasons.append(f"{group}:{metric}:nonfinite_score")
                continue
            if set(old["complete_task_means"]) != set(new["complete_task_means"]):
                reasons.append(f"{group}:{metric}:task_identity_mismatch")
                continue
            for task, value in old["complete_task_means"].items():
                current = new["complete_task_means"][task]
                if ((group == "development" and current < value - 1e-12)
                        or (group == "train" and metric == "task_completed_correctly" and value == 1 and current != 1)):
                    reasons.append(f"{group}:{task}:{metric}:regression")
        phases = []
        for population in (first, last):
            phase = next((p for p in guards["phases"] if p["phase"] ==
                          f"step-{population['step']:04d}/{group}/native/traces.jsonl"), None)
            if phase is None or phase["native_sha256"] != population["native_sha256"]:
                reasons.append(f"{group}:missing_or_mismatched_guard_evidence")
                break
            if any(k not in phase["totals"] for k in
                   ("unknown_initial_checks", "preservation_checks", "broken_preservation_checks")):
                reasons.append(f"{group}:missing_preservation_coverage_counts")
                break
            phases.append(phase)
            if phase["totals"]["unknown_initial_checks"]:
                reasons.append(f"{group}:unknown_preservation_checks")
        if len(phases) == 2:
            broken = [{(e["task"], json.dumps(item["spec"], sort_keys=True))
                       for e in phase["episodes"] for item in e["broken"]} for phase in phases]
            if (broken[1] - broken[0] or phases[1]["totals"].get("broken_preservation_checks", 0) >
                    phases[0]["totals"].get("broken_preservation_checks", 0)):
                reasons.append(f"{group}:added_preservation_violation")
    if (not audit["completed"] or not review.get("improved_native_behavior_verified")
            or not review.get("evidence_sha256") or not review.get("native_episode_ids")):
        reasons.append("actual_completed_native_behavior_review_required")
    improved = any(after[g]["metrics"][m]["full_population_task_mean"] is not None and
                   before[g]["metrics"][m]["full_population_task_mean"] is not None and
                   after[g]["metrics"][m]["full_population_task_mean"] >
                   before[g]["metrics"][m]["full_population_task_mean"] + 1e-12
                   for g in after for m in ("task_completed_correctly", "partial_credit"))
    if not improved:
        reasons.append("no_native_task_outcome_gain")
    return {"metadata": metadata, "step": step, "eligible": not reasons, "reasons": reasons,
            "before": before, "after": after, "native_review": review}


def choose(candidates, reserve_sha256):
    eligible = [c for c in candidates if c["eligible"]]
    if not eligible:
        return {"approved_for_reserve": False, "reason": "no_candidate_met_native_and_preservation_gates"}

    def rank(entry):
        metrics = entry["after"]["development"]["metrics"]
        mistakes = metrics["tool_mistakes"]["full_population_task_mean"]
        return (-metrics["task_completed_correctly"]["full_population_task_mean"],
                -metrics["partial_credit"]["full_population_task_mean"],
                float("inf") if mistakes is None else mistakes,
                entry["step"], entry["metadata"]["budget_slot"], entry["metadata"].get("branch_order", 0))

    winner = min(eligible, key=rank)
    metadata = winner["metadata"]
    return {**metadata, "checkpoint_step": winner["step"], "model_revision": MODEL_REVISION,
            "approved_for_reserve": True, "frozen_before_reserve": True,
            "frozen_at_utc": datetime.datetime.now(datetime.UTC).isoformat(), "reserve_sha256": reserve_sha256,
            "selection_rule": "Native outcome/guard eligibility; development success, partial credit, "
                "fewer tool mistakes, earlier checkpoint, earlier experiment, no-Luna branch first on exact tie. "
                "No ranking by raw training reward or reserve results. Operating profiles may differ; "
                "reserve fresh base must match the chosen profile.",
            "native_review": winner["native_review"],
            "candidate_audit": [{"budget_slot": c["metadata"]["budget_slot"], "step": c["step"],
                                 "eligible": c["eligible"], "reasons": c["reasons"]} for c in candidates]}
