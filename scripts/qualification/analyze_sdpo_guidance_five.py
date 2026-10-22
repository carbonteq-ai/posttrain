"""Audit frozen ten-update guidance arms; never zero-fill native outcomes."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from analyze_sdpo_pool124 import bounded_scores, comparable_summary, generation_evidence, paired
from analyze_self_pilot import paired_bootstrap
from analyze_self_repair_run import native_tool_observations
from sdpo_long_campaign_config import execution_profile


def read(path):
    return json.loads(path.read_text())


def absolute_uncertainty(summary, names, replicas):
    result = {}
    for key, metric in summary["metrics"].items():
        complete = metric["complete_task_means"]
        interval = paired_bootstrap(complete, complete)["before_absolute_ci95"] if complete else None
        result[key] = {"complete_tasks": len(complete), "expected_tasks": len(names),
            "full_population_absolute_ci95": interval if len(complete) == len(names) else None,
            "complete_task_absolute_sensitivity_ci95": interval,
            "unit": f"task; retain all{replicas}episodes per task",
            "scope": "Absolute task-bootstrap uncertainty, conditional on complete-task coverage when missing. "
                     "Separate from paired gain intervals and missing-outcome identification bounds."}
    return result


def runtime_evidence(folder, declared):
    from sdpo_greedy_repeatability import audit as repeatability_audit

    calls = [json.loads(line) for line in (folder / "generation-calls.jsonl").read_text().splitlines() if line]
    computed = repeatability_audit(calls)
    path = folder / "native-greedy-repeatability.json"
    if path.exists() and read(path) != computed:
        raise ValueError("saved native repeatability contradicts original calls")
    return {"declared_gate": bool(declared), "persisted_gate": path.exists(),
            "qualified": computed["qualified"] if declared and path.exists() else None,
            "retrospective_repeatability_passed": computed["qualified"],
            "compared_pairs": computed["compared_pairs"],
            "different_output_pairs": computed["different_output_pairs"],
            "scope": "Finite native runtime evidence. Missing/undeclared gates are not silently promoted; "
                     "runtime-rejected scores remain raw observations, not qualified benchmark comparisons."}


def audit(root, cohorts, heldout):
    inputs = read(root / "inputs.json")
    if inputs["run_profile"] != "guidance-five":
        raise ValueError("not a guidance-five experiment")
    execution_profile(SimpleNamespace(**inputs))
    train, evaluation = cohorts["train"], heldout["task_names"]
    if (len(train) != len(set(train)) or len(train) != 20 or len(evaluation) != 62
            or len(set(evaluation)) != 62 or set(train) & set(evaluation)):
        raise ValueError("frozen task identities differ")
    updates = read(root / "updates.json") if (root / "updates.json").exists() else []
    if ([u["step"] for u in updates] != list(range(1, len(updates) + 1)) or len(updates) > 10
            or any(u["batch"] != 32 or u["optimizer_moment_steps"] != [u["step"]]
                   or u["ema_updates"] != u["step"] for u in updates)):
        raise ValueError("actual update/moment/EMA/batch count differs")
    seen, populations, baselines = set(), [], {}
    referenced = 0
    reference_record = root / "step-0000/heldout62/shared-base.json"
    if bool(inputs.get("heldout_base_reference")) != reference_record.exists():
        raise ValueError("shared-control declaration/evidence differs")
    if reference_record.exists():
        from sdpo_shared_base import load

        capsule = root.parents[1] if len(root.parents) > 1 else Path(inputs["teacher_guidance"]).parent
        if not (capsule / "driver").exists():
            capsule = Path(inputs["teacher_guidance"]).parent
        train_calls = root / "step-0000/train/generation-calls.jsonl"
        initial_hashes = {c["adapter_sha256"] for line in train_calls.read_text().splitlines()
                          if line and (c := json.loads(line))["action_ids"]}
        if len(initial_hashes) != 1:
            raise ValueError("consumer fresh-base identity missing or mixed")
        shared = load(capsule / "baseline/reference.json", capsule / "driver", inputs,
                      next(iter(initial_hashes)), capsule / "heldout62.json")
        recorded = read(reference_record)
        if shared != {k: v for k, v in recorded.items() if k != "source_reference"}:
            raise ValueError("saved shared-control audit differs from native source")
        entry = {"step": 0, "cohort": "heldout62", "reused_control": True,
            "origin_run": shared["origin_run"], "summary": shared["summary"],
            "bounded_metrics": shared["bounded_metrics"], "native_sha256": shared["native_sha256"],
            "tool_observations": shared["tool_observations"], **shared["generation"],
            "absolute_uncertainty": absolute_uncertainty(shared["summary"], evaluation, 5)}
        entry["native_runtime"] = {"qualified": True, "shared_reference_revalidated": True}
        baselines["heldout62"] = entry
        populations.append(entry)
        seen = set(shared["episode_ids"])
        referenced = len(seen)
    for path in sorted(root.glob("step-*/*/native/traces.jsonl")):
        folder = path.parents[1]
        if not (folder / "outcomes.json").exists():
            continue
        step, group = int(folder.parent.name.split("-")[1]), folder.name
        if (group not in ("train", "online_train", "heldout62")
                or group == "online_train" and not 1 <= step <= 10
                or group != "online_train" and step not in (0, 10)):
            raise ValueError("undeclared native population")
        names, replicas = (evaluation, 5) if group == "heldout62" else (train, 2)
        episodes = [json.loads(line) for line in path.read_text().splitlines() if line]
        ids = {e["id"] for e in episodes}
        if len(ids) != len(episodes) or ids & seen:
            raise ValueError("native episode identity duplicated")
        seen |= ids
        summary, bounded = bounded_scores(episodes, names, replicas)
        if comparable_summary(summary) != comparable_summary(read(folder / "outcomes.json")):
            raise ValueError("saved/native summary differs")
        summary["goals_and_harms"] = "Native manifest assessments require a separate scope/coverage audit; legacy scores do not establish them."
        expected_version = step - 1 if group == "online_train" else step
        entry = {"step": step, "cohort": group, "summary": summary, "bounded_metrics": bounded,
            "absolute_uncertainty": absolute_uncertainty(summary, names, replicas),
            "native_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "tool_observations": native_tool_observations(episodes),
            **generation_evidence(folder / "generation-calls.jsonl", expected_version)}
        admission = folder / "training-admission.json"
        if group == "online_train" and admission.exists():
            admitted = read(admission)
            rows = admitted["selected"]
            mapping = {e["id"]: e["task"]["data"]["name"] for e in episodes}
            if (admitted["truncated_training_sequences"] or len(rows) != 32
                    or len({(r["episode_id"], r["generation_index"]) for r in rows}) != 32
                    or set(r["task"] for r in rows) != set(train)
                    or any(mapping.get(r["episode_id"]) != r["task"] for r in rows)):
                raise ValueError("selected action/task identity or no-truncation contract differs")
            entry["admission"] = admitted
        if group != "online_train":
            entry["native_runtime"] = runtime_evidence(folder, inputs.get("native_repeatability", False))
            if step == 0:
                baselines[group] = entry
            else:
                valid = (entry["native_runtime"]["qualified"] is True
                         and baselines[group]["native_runtime"]["qualified"] is True)
                entry["paired_vs_fresh_base"] = paired(baselines[group], entry, names, replicas) if valid else None
                if not valid:
                    entry["paired_comparison_withheld_reason"] = "Fresh base or readout lacks a qualified native runtime gate."
        populations.append(entry)
    receipt = read(root / "receipt.json") if (root / "receipt.json").exists() else None
    if receipt and (not receipt["completed"] or receipt["optimizer_updates"] != 10
                    or len(updates) != 10 or len(seen) != 1100 or len(populations) != 14):
        raise ValueError("completion receipt lacks all10updates/1100episodes/14populations")
    readouts = [p for p in populations if p["cohort"] != "online_train"]
    unqualified = [{"step": p["step"], "cohort": p["cohort"], "runtime": p["native_runtime"]}
                   for p in readouts if p["native_runtime"]["qualified"] is not True]
    if receipt and inputs.get("native_repeatability") and unqualified:
        raise ValueError("completion receipt contradicts native runtime qualification")
    return {"execution_complete": bool(receipt), "receipt": receipt, "updates": updates,
        "benchmark_readouts_runtime_qualified": bool(receipt) and len(readouts) == 4 and not unqualified,
        "unqualified_native_readouts": unqualified,
        "unique_native_episodes": len(seen), "episode_ids": sorted(seen), "populations": populations,
        "native_episodes_written_by_this_run": len(seen)-referenced,
        "referenced_control_native_episodes": referenced,
        "native_count_scope": "Unique evidence IDs include explicitly referenced common controls. Shared IDs must be deduplicated across studies; they are not new consumer episodes.",
        "retry_count": None, "checkpoint_files_require_separate_audit": True,
        "native_goal_harm_status": "Audit declared manifest signals and closed guard scopes separately; unknown is not harm-free.",
        "scope": "Repeated heldout research monitoring. Paired complete-task sensitivity, absolute intervals "
                 "and missing-score identification bounds are distinct; online losses are not rewards."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.root, read(args.cohorts), read(args.heldout)), indent=2) + "\n")
