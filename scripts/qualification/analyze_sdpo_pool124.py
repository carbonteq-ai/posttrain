"""Read-only full-pool checkpoint audit; native missing scores remain missing."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from sdpo_pool124_config import collection_budget, step_tasks, validate_pool
from self_campaign_config import comparisons, summarize


def read(path):
    return json.loads(path.read_text())


def bounded_scores(episodes, names, replicas):
    """Identification bounds are not confidence intervals or zero-filled estimates."""
    summary = summarize(episodes, names, replicas)
    result = {}
    for metric in ("task_completed_correctly", "partial_credit"):
        values = []
        for episode in episodes:
            trace = episode["traces"][0]
            value = (trace.get("metrics", {}).get(metric) if metric == "task_completed_correctly"
                     else trace.get("rewards", {}).get("partial_credit", {}).get("score"))
            if value is not None:
                if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("native metric violates declared0to1 bounds")
                values.append(float(value))
        total, observed = len(episodes), len(values)
        observed_sum = math.fsum(values)
        result[metric] = {"observed": observed, "expected": total, "missing": total - observed,
                          "identification_bounds": [observed_sum / total,
                                                    (observed_sum + (total - observed)) / total],
                          "strict_task_mean": summary["metrics"][metric]["full_population_task_mean"],
                          "scope": "Bounds under unknown missing outcomes; not confidence intervals"}
    return summary, result


def paired(baseline, current, names, replicas):
    result = comparisons(baseline["summary"], current["summary"])
    for metric, item in result.items():
        a = baseline["summary"]["metrics"][metric]["complete_task_means"]
        b = current["summary"]["metrics"][metric]["complete_task_means"]
        item["excluded_tasks"] = sorted(set(names) - (set(a) & set(b)))
        if item["paired"]:
            item["paired"]["unit"] = f"task; preserve all{replicas}repetitions within each task"
        if metric in current["bounded_metrics"]:
            lower, upper = current["bounded_metrics"][metric]["identification_bounds"]
            base_lower, base_upper = baseline["bounded_metrics"][metric]["identification_bounds"]
            item["full_population_gain_identification_bounds"] = [lower - base_upper, upper - base_lower]
    return result


def comparable_summary(summary):
    """Native persistence omits an unset traceback; preserve every error and score."""
    return summary | {"errors": [{k: v for k, v in error.items()
                                  if not (k == "traceback" and v is None)}
                                 for error in summary["errors"]]}


def generation_evidence(path, expected_version):
    """Check every original call without retaining its large token arrays."""
    incomplete, context, reasons = 0, 0, Counter()
    with path.open() as stream:
        for line in stream:
            line = line.rstrip("\r\n")
            if not line:
                continue
            call = json.loads(line)
            if call["policy_version"] != expected_version:
                raise ValueError("stale native policy")
            incomplete += not call["complete"]
            reasons[call.get("finish_reason", "not_recorded")] += 1
            context += call.get("limit_reason") == "context_limit"
    return {"incomplete_generations": incomplete, "finish_reasons": dict(reasons),
            "context_refusals": context}


def audit(root, cohort, pool, heldout):
    validate_pool(cohort, pool, heldout)
    inputs = read(root / "inputs.json")
    if inputs["run_profile"] not in ("pool124-qualify", "pool124-50", "pool124-bounded"):
        raise ValueError("not a pool124 campaign")
    updates = read(root / "updates.json") if (root / "updates.json").exists() else []
    if [u["step"] for u in updates] != list(range(1, len(updates) + 1)) or len(updates) > inputs["steps"]:
        raise ValueError("actual optimizer update population mismatch")
    for u in updates:
        if u["batch"] != 32 or u["optimizer_moment_steps"] != [u["step"]] or u["ema_updates"] != u["step"]:
            raise ValueError("batch/moment/EMA counter mismatch")
    seen, visited, populations, baselines = set(), Counter(), [], {}
    for path in sorted(root.glob("step-*/*/native/traces.jsonl")):
        folder = path.parents[1]
        if not (folder / "outcomes.json").exists():
            continue
        step, group = int(folder.parent.name.split("-")[1]), folder.name
        names = (heldout["task_names"] if group == "heldout62" else cohort["reference_diagnostic"]
                 if group == "reference_train" else
                 step_tasks(cohort["train"], step if group == "online_train" else 1))
        replicas = collection_budget(group, inputs["run_profile"])["replicas"]
        episodes = [json.loads(s) for s in path.read_text().splitlines() if s]
        ids = {e["id"] for e in episodes}
        if len(ids) != len(episodes) or ids & seen:
            raise ValueError("native episode identity reused")
        seen |= ids
        summary, bounded = bounded_scores(episodes, names, replicas)
        if comparable_summary(summary) != comparable_summary(read(folder / "outcomes.json")):
            raise ValueError("retained/native summary disagreement")
        expected_version = step - 1 if group == "online_train" else step
        entry = {"step": step, "cohort": group, "summary": summary, "bounded_metrics": bounded,
                 "native_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                 **generation_evidence(folder / "generation-calls.jsonl", expected_version)}
        if group == "online_train" and (folder / "training-admission.json").exists():
            admission = read(folder / "training-admission.json")
            selected = admission["selected"]
            episode_tasks = {e["id"]: e["task"]["data"]["name"] for e in episodes}
            if (admission["truncated_training_sequences"] or len(selected) != 32
                    or Counter(r["task"] for r in selected) != Counter(names)
                    or any(episode_tasks.get(r["episode_id"]) != r["task"] for r in selected)):
                raise ValueError("not32distinct admitted tasks; no silent fallback")
            visited.update(r["task"] for r in selected)
            entry["admission"] = admission
        if group != "online_train":
            if step == 0:
                baselines[group] = entry
            else:
                entry["paired_vs_base"] = paired(baselines[group], entry, names, replicas)
        populations.append(entry)
    receipt = read(root / "receipt.json") if (root / "receipt.json").exists() else None
    if receipt and (not receipt["completed"] or len(updates) != inputs["steps"]
                    or receipt["optimizer_updates"] != len(updates)):
        raise ValueError("completion receipt disagrees with actual updates")
    if receipt:
        expected_episodes, expected_populations = {
            "pool124-qualify": (576, 9), "pool124-50": (5444, 62),
            "pool124-bounded": (1024, 10)}[inputs["run_profile"]]
        if len(seen) != expected_episodes or len(populations) != expected_populations or len(visited) != 124:
            raise ValueError("completion lacks the full declared native or training-task population")
    return {"completed": bool(receipt), "receipt": receipt, "updates": updates,
            "unique_native_episodes": len(seen), "distinct_tasks_with_selected_actions": len(visited),
            "selected_task_visit_counts": dict(visited), "populations": populations,
            "retry_count": None, "guard_status": "Separate native initial-state preservation audit required; unknown is not harm-free",
            "scope": "Repeated heldout monitoring/selection, not pristine final validation. "
                     "Absolute intervals, paired complete-task sensitivity and missing-score bounds are distinct."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    for name in ("cohorts", "pool", "heldout", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.root, read(args.cohorts), read(args.pool), read(args.heldout)),
                                      indent=2) + "\n")
