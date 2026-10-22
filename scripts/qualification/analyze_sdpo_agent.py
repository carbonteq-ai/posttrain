"""Reconcile SDPO native populations and actual updates without inventing rewards."""

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from analyze_self_repair_run import native_tool_observations
from sdpo_execution_budget import remaining_runtime, reserve_profile
from self_campaign_config import comparisons, summarize


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(root, cohorts):
    if (root / "paired-plan.json").exists():
        return audit_paired(root, cohorts)
    if (root / "primary-selection.json").exists():
        return audit_reserve(root, cohorts)
    receipt = read(root / "receipt.json") if (root / "receipt.json").exists() else None
    updates = read(root / "updates.json") if (root / "updates.json").exists() else []
    inputs = read(root / "inputs.json")
    from types import SimpleNamespace

    from sdpo_long_campaign_config import campaign_learning_rate, execution_profile

    execution = execution_profile(SimpleNamespace(**inputs))
    if [u["step"] for u in updates] != list(range(1, len(updates) + 1)):
        raise ValueError("noncontiguous optimizer updates")
    if len(updates) > inputs["steps"] or any(u["optimizer_moment_steps"] != [u["step"]] or
                               u["ema_updates"] != u["step"] for u in updates):
        raise ValueError("moment/EMA/budget mismatch")
    if any(u["batch"] != execution["batch"] for u in updates):
        raise ValueError("actual optimizer batch differs from execution profile")
    if inputs.get("run_profile", "ablation") != "ablation":
        for u in updates:
            if not math.isclose(u["learning_rate"], campaign_learning_rate(inputs["lr"], u["step"]), rel_tol=1e-9):
                raise ValueError("actual campaign learning-rate boundary mismatch")
    seen, populations, baseline, latest = set(), [], {}, {}
    for path in sorted(root.glob("step-*/*/native/traces.jsonl")):
        folder = path.parents[1]
        if not (folder / "outcomes.json").exists():
            continue  # live population is not a completed denominator
        step, cohort = int(folder.parent.name.split("-")[1]), folder.name
        names = cohorts["train" if cohort == "online_train" else cohort]
        episodes = lines(path)
        ids = {e["id"] for e in episodes}
        if len(ids) != len(episodes) or ids & seen:
            raise ValueError("episode ID reused across native populations")
        seen |= ids
        replicas = inputs.get("online_rollouts", 2) if cohort == "online_train" else 2
        if (len(episodes) != len(names) * replicas or
                Counter(e["task"]["data"]["name"] for e in episodes) != Counter({name: replicas for name in names})):
            raise ValueError("native task/replica population mismatch")
        summary = summarize(episodes, names, replicas)
        if summary != read(folder / "outcomes.json"):
            raise ValueError("saved/native outcome summary mismatch")
        calls = lines(folder / "generation-calls.jsonl")
        expected_version = step - 1 if cohort == "online_train" else step
        if any(c["policy_version"] != expected_version for c in calls):
            raise ValueError("stale native policy")
        admission_path = folder / "training-admission.json"
        admission = read(admission_path) if admission_path.exists() else None
        if admission and admission["truncated_training_sequences"]:
            raise ValueError("training truncation")
        stop = Counter(t["stop_condition"] for e in episodes for t in e["traces"])
        entry = {"step": step, "cohort": cohort, "summary": summary, "stop_conditions": dict(stop),
                 "native_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                 "generation_calls": len(calls), "incomplete_generations": sum(not c["complete"] for c in calls),
                 "context_rejected_calls": sum(c.get("limit_reason") == "context_limit" for c in calls),
                 "tool_observations": observations(episodes), "training_admission": admission}
        if cohort != "online_train":
            if step == 0:
                baseline[cohort] = summary
            if step > 0:
                entry["paired_vs_fresh_base"] = comparisons(baseline[cohort], summary)
                latest[cohort] = entry["paired_vs_fresh_base"]
        populations.append(entry)
    if receipt:
        if (not receipt["completed"] or receipt["optimizer_updates"] != inputs["steps"] or
                len(updates) != inputs["steps"] or receipt["elapsed_s"] >
                (86400 if inputs.get("run_profile") == "campaign250" else 7200)):
            raise ValueError("receipt update/time budget mismatch")
        cadence = execution["evaluate_every"]
        expected_population_count = 2 + inputs["steps"] + 2 * (inputs["steps"] // cadence + bool(inputs["steps"] % cadence))
        if len(populations) != expected_population_count:
            raise ValueError("missing completed native population")
    return {"completed": bool(receipt), "receipt": receipt, "updates": updates,
            "unique_native_episodes": len(seen), "populations": populations, "latest_paired_comparisons": latest,
            "retry_count": None, "guard_status": "Requires separate initial-state native assertion audit; unknown is not harm-free.",
            "scope": "Small adaptive train/development diagnostics, not heldout evidence or a global optimum. "
                     "Sampled teacher advantages/actual Adam direction do not establish correctness by themselves."}


def audit_paired(root, cohorts):
    plan = read(root / "paired-plan.json")
    receipt = read(root / "receipt.json") if (root / "receipt.json").exists() else None
    if plan["budget_slot"] != 9 or plan["steps"] != 20:
        raise ValueError("paired experiment budget mismatch")
    branches, inputs, seen = {}, {}, set()
    for name in ("control", "luna"):
        folder = root / name
        if not (folder / "inputs.json").exists():
            continue
        branches[name] = audit(folder, cohorts)
        inputs[name] = read(folder / "inputs.json")
        for path in folder.glob("step-*/*/native/traces.jsonl"):
            ids = {e["id"] for e in lines(path)}
            if ids & seen:
                raise ValueError("episode identity reused across paired branches")
            seen |= ids
    paired = {}
    if set(branches) == {"control", "luna"}:
        common = ("model", "steps", "lr", "lr_schedule", "lr_end_ratio", "alpha", "topk", "teacher_rate",
                  "context_limit", "reply_limit", "tool_parser_mode", "online_rollouts", "selection_mode")
        if any(inputs["control"][k] != inputs["luna"][k] for k in common):
            raise ValueError("paired branch optimizer/data/budget mismatch")
        if (inputs["control"]["steps"] != 10 or inputs["control"]["context_limit"] != 32768
                or inputs["control"]["reply_limit"] != 4096
                or inputs["control"]["teacher_feedback"] != "solution-or-feedback"
                or inputs["luna"]["teacher_feedback"] != "luna-or-solution"
                or any(v.get("thinking", False) for v in inputs.values())):
            raise ValueError("paired treatment mismatch")
        for cohort in ("train", "development"):
            populations = {name: [p for p in branch["populations"] if p["cohort"] == cohort]
                           for name, branch in branches.items()}
            if populations["control"] and populations["luna"]:
                if populations["control"][0]["summary"] != populations["luna"][0]["summary"]:
                    raise ValueError("paired fresh-base native outcomes differ")
                if all(p[-1]["step"] == 10 for p in populations.values()):
                    paired[cohort] = comparisons(populations["control"][-1]["summary"],
                                                 populations["luna"][-1]["summary"])
    total_updates = sum(len(branch["updates"]) for branch in branches.values())
    if total_updates > 20:
        raise ValueError("paired total optimizer budget exceeded")
    if receipt:
        if (not receipt["completed"] or receipt["optimizer_updates"] != total_updates or total_updates != 20
                or len(branches) != 2 or not all(b["completed"] for b in branches.values())
                or receipt["elapsed_s"] > plan["gpu_timeout_seconds"]):
            raise ValueError("paired final receipt incomplete")
        gate = read(root / "control/capacity-qualification.json")
        if (not gate["qualified"] or gate["optimizer_updates"] != 0 or gate["context_limit"] != 32768
                or gate["reply_limit"] != 4096 or not gate["weights_unchanged"]):
            raise ValueError("paired capacity gate missing/invalid")
    return {"completed": bool(receipt), "receipt": receipt, "plan": plan, "branches": branches,
            "total_optimizer_updates": total_updates, "unique_native_episodes": len(seen),
            "luna_vs_matched_larger_budget_control": paired,
            "scope": "Separate fresh10-update branches; no pooled populations. Compare each to its own base "
                     "and compare matched final task outcomes. Initial-state guard/file audits still required."}


def observations(episodes):
    normalized = copy.deepcopy(episodes)
    for episode in normalized:
        for trace in episode["traces"]:
            for node in trace["nodes"]:
                message = node.get("message")
                if message is not None and message.get("tool_calls") is None:
                    message["tool_calls"] = []
    return native_tool_observations(normalized)


def audit_reserve(root, cohorts):
    """Audit the explicitly zero-update, preselected two-policy comparison."""
    from sdpo_agent_config import MODEL_REVISION, fingerprint

    selection, inputs = read(root / "primary-selection.json"), read(root / "inputs.json")
    profile = reserve_profile(selection)
    if any(inputs.get(k) != v for k, v in profile.items()):
        raise ValueError("reserve mode/token profile changed after selection")
    if (not selection["frozen_before_reserve"] or not selection["approved_for_reserve"]
            or selection["model_revision"] != MODEL_REVISION
            or selection["reserve_sha256"] != fingerprint(cohorts["reserve"])
            or len(cohorts["reserve"]) != 24
            or inputs["tool_parser_mode"] != selection["tool_parser_mode"]
            or inputs["tool_parser_mode"] not in ("native", "schema-bound-v1")):
        raise ValueError("reserve preselection/transport/cohort mismatch")
    if inputs["tool_parser_mode"] == "schema-bound-v1":
        if read(root / "tool-parser-receipt.json")["mode"] != inputs["tool_parser_mode"]:
            raise ValueError("reserve parser receipt mismatch")
    if (root / "updates.json").exists() and read(root / "updates.json"):
        raise ValueError("optimizer updates in reserved zero-update comparison")
    receipt = read(root / "receipt.json") if (root / "receipt.json").exists() else None
    populations, seen, baseline, paired = [], set(), None, None
    versions = receipt["rollout_versions"] if receipt else None
    for path in sorted(root.glob("step-*/*/native/traces.jsonl")):
        folder = path.parents[1]
        if not (folder / "outcomes.json").exists():
            continue
        step = int(folder.parent.name.split("-")[1])
        if folder.name != "reserve" or step not in (0, 1):
            raise ValueError("unexpected population in reserve comparison")
        episodes = lines(path)
        ids = {e["id"] for e in episodes}
        if seen & ids:
            raise ValueError("reused native reserve episode ID")
        seen |= ids
        summary = summarize(episodes, cohorts["reserve"], 2)
        if summary != read(folder / "outcomes.json"):
            raise ValueError("native reserve summary mismatch")
        calls = lines(folder / "generation-calls.jsonl")
        if any(c["policy_version"] != step for c in calls):
            raise ValueError("stale native reserve policy")
        if versions and any(c["adapter_sha256"] != versions[step]["weights_sha256"] for c in calls):
            raise ValueError("reserve generation adapter digest mismatch")
        if step == 0:
            baseline = summary
        elif baseline:
            paired = comparisons(baseline, summary)
        populations.append({"step": step, "cohort": "reserve", "summary": summary,
            "native_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "generation_calls": len(calls), "incomplete_generations": sum(not c["complete"] for c in calls),
            "tool_observations": observations(episodes)})
    if receipt:
        budget = read(root / "shared-budget-snapshot.json")
        remaining_runtime(budget, 1, now=budget["created_boot_seconds"], boot_id=budget["boot_id"])
        if (receipt["matched_profile"] != profile
                or receipt["shared_deadline_boot_seconds"] != budget["deadline_boot_seconds"]
                or not budget["created_boot_seconds"] <= receipt["finished_boot_seconds"] <= budget["deadline_boot_seconds"] - 30):
            raise ValueError("reserve combined experiment time/profile receipt mismatch")
        loaded = read(root / "candidate-load.json")
        if (not receipt["completed"] or receipt["optimizer_updates"] != 0
                or receipt["trainable_parameters"] != 0 or receipt["elapsed_s"] > 7200
                or not receipt["reserve_evaluated"] or receipt["heldout62_evaluated"]
                or len(seen) != 96 or len(populations) != 2
                or not loaded["all_adapter_tensors_exact"] or loaded["optimizer_updates"] != 0
                or loaded["source_weights_sha256"] != selection["adapter_sha256"]
                or [v["policy_version"] for v in versions] != [0, 1]):
            raise ValueError("incomplete/invalid zero-update reserve receipt")
        if read(root / "fresh-base-gate.json")["max_abs_delta"] != 0:
            raise ValueError("fresh base differs from zero-effect adapter")
        for step in (0, 1):
            parity = read(root / f"parity-v{step}.json")
            if (parity["policy_version"] != step or not parity["actor"]
                    or len(parity["actor"]) != len(parity["rollout"])
                    or not all(math.isfinite(v) for v in parity["actor"] + parity["rollout"])
                    or max(abs(a - b) for a, b in zip(parity["actor"], parity["rollout"], strict=True)) > .15):
                raise ValueError("reserve actor/rollout parity failure")
        invariant = read(root / "batch-invariance-v0.json")
        records = invariant["records"]
        if (not invariant["exact_batch_invariance"] or not records
                or max(r["peak_in_flight"] for r in records) != 32
                or any(r["tokens"] != records[0]["tokens"] or r["logprobs"] != records[0]["logprobs"]
                       for r in records)):
            raise ValueError("reserve batch invariance/admission failure")
    return {"completed": bool(receipt), "receipt": receipt, "selection": selection,
        "optimizer_updates": 0, "unique_native_episodes": len(seen), "populations": populations,
        "paired_vs_matched_fresh_base": paired, "retry_count": None,
        "guard_status": "Separate native initial-state guard audit required; unknown is not harm-free.",
        "scope": "Frozen primary, matched transport/sampling/base on 24-task research reserve; no winner reselection. Historical global novelty unknown. Missing native scores remain missing."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.root, read(args.cohorts)), indent=2) + "\n")
