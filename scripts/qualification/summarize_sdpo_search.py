"""Reviewed original-ten search snapshot; keep failures and unrun arms explicit."""

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def reference(path):
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def cohort_summary(audit, group):
    populations = sorted((p for p in audit["populations"] if p["cohort"] == group), key=lambda p: p["step"])
    before = next((p for p in populations if p["step"] == 0), None)
    post = [p for p in populations if p["step"] > 0]
    after = post[-1] if post else None

    def metrics(population):
        if population is None:
            return None
        summary = population["summary"]
        result = {"step": population["step"], "episodes": summary["episodes"], "errors": summary["errors"]}
        for name in ("task_completed_correctly", "partial_credit"):
            metric = summary["metrics"][name]
            if metric["full_population_task_mean"] is not None and metric["episode_coverage"] != summary["episodes"]:
                raise ValueError("reported full-population rate has incomplete episode coverage")
            result[name] = metric["full_population_task_mean"]
            result[name + "_coverage"] = metric["episode_coverage"]
        return result

    paired = None
    if after:
        old, broad = after.get("paired_vs_fresh_base"), after.get("paired_vs_base")
        if old is not None and broad is not None and old != broad:
            raise ValueError("conflicting paired comparison sources")
        paired = old if old is not None else broad
    return {"baseline": metrics(before), "final_post_update": metrics(after),
            "paired_final_vs_baseline": paired,
            "checkpoint_curve": [metrics(p) for p in populations]}


def branch_review(audit):
    updates = audit["updates"]
    return {"actual_updates": len(updates), "runtime_completed": audit["completed"],
            "native_episodes": audit["unique_native_episodes"],
            "loss_first": updates[0]["loss"] if updates else None,
            "loss_last": updates[-1]["loss"] if updates else None,
            "train": cohort_summary(audit, "train"), "development": cohort_summary(audit, "development"),
            "reference_training": cohort_summary(audit, "reference_train"),
            "heldout62": cohort_summary(audit, "heldout62")}


def summarize(root):
    ledger_path = root / "experiment-ledger.json"
    ledger = read(ledger_path)
    if ledger["maximum_experiments"] != 10:
        raise ValueError("search identity changed")
    accepted = {e["budget_slot"]: e for e in ledger["experiments"]}
    if len(accepted) != ledger["slots_launched"]:
        raise ValueError("launched attempt ledger mismatch")
    rows, sources = [], [reference(ledger_path)]
    for slot in range(1, 11):
        if slot not in accepted:
            prepared = ledger.get("prepared_experiments", {}).get(str(slot), {})
            pending = {"slot": slot, "status": "not_GPU_submitted", "actual_updates": None,
                         "native_episodes": None, "runtime_completed": False,
                         "prepared_source_sha256": prepared.get("manifest_sha256"),
                         "selection_mode": prepared.get("selection_mode"),
                         "train": None, "development": None}
            config_name = prepared.get("config_receipt")
            if config_name:
                config_path = root / config_name
                pending["prepared_configuration"] = reference(config_path)
                sources.append(reference(config_path))
            rows.append(pending)
            continue
        record = accepted[slot]
        path = root / record.get("audit_file", f"experiment{slot:02d}-audit.json")
        if not path.exists() and record["status"].startswith("accepted_running"):
            # An accepted process is neither an unsubmitted arm nor a completed result.
            receipt_path = root / record["launch_receipt"]
            launch = read(receipt_path)
            if launch.get("run_name") != record["run_name"]:
                raise ValueError("accepted run receipt identity differs")
            sources.append(reference(receipt_path))
            rows.append({"slot": slot, "status": record["status"], "run_name": record["run_name"],
                         "actual_updates": None, "native_episodes": None, "runtime_completed": False,
                         "last_verified_optimizer_updates": record.get("last_verified_optimizer_updates"),
                         "train": None, "development": None, "heldout62": None,
                         "source": reference(receipt_path),
                         "measurement_note": "Accepted; terminal native audit not yet available. No invented zero updates or gains."})
            continue
        audit = read(path)
        sources.append(reference(path))
        paired = "branches" in audit
        if paired:
            branches = {name: branch_review(value) for name, value in audit["branches"].items()}
            actual_updates = sum(b["actual_updates"] for b in branches.values())
            if actual_updates != audit["total_optimizer_updates"]:
                raise ValueError("paired actual update total differs from its branches")
            if audit["completed"] and (set(branches) != {"control", "luna"}
                    or not all(b["runtime_completed"] for b in branches.values())
                    or audit["unique_native_episodes"] != sum(b["native_episodes"] for b in branches.values())):
                raise ValueError("completed paired native populations are incomplete")
            measured = {"branches": branches, "train": None, "development": None,
                        "loss_first": None, "loss_last": None,
                        "luna_vs_matched_control": audit.get("luna_vs_matched_larger_budget_control"),
                        "unstarted_branches": sorted({"control", "luna"} - set(branches)),
                        "closed_native_episodes": audit.get("unique_closed_native_episodes",
                                                            sum(b["native_episodes"] for b in branches.values())),
                        "partial_populations": audit.get("partial_populations", {})}
        else:
            measured = branch_review(audit)
            actual_updates = measured["actual_updates"]
        if actual_updates > record["steps"] or actual_updates > ledger["per_experiment_maximum_updates"]:
            raise ValueError("actual original-search optimizer budget exceeded")
        rows.append({"slot": slot, "status": record["status"], "run_name": record["run_name"],
            "actual_updates": actual_updates, "planned_updates": record["steps"],
            "runtime_completed": audit["completed"], "native_episodes": audit["unique_native_episodes"],
            "receipt_elapsed_seconds": (audit.get("receipt") or {}).get("elapsed_s"),
            "teacher_feedback": record["teacher_feedback"], "peak_lr": record["lr"],
            "schedule": record.get("lr_schedule", "constant"),
            "parser_regime": record.get("tool_parser_mode", "native-pinned"),
            "selection_mode": record.get("selection_mode", "legacy-balanced"),
            **measured, "source": reference(path)})
    validation_path = root / "experiment10-reserve-audit.json"
    validation = None
    if validation_path.exists():
        validation = read(validation_path)
        if validation["optimizer_updates"] != 0:
            raise ValueError("reserve validation unexpectedly changed policy weights")
        sources.append(reference(validation_path))
    return {"scope": "Original ten bounded SDPO attempts only. Separate full-pool50 and earlier batch32/250 campaigns excluded; "
                     "failed qualifications count as attempts, not successful runtime or learning.",
            "attempts_launched": len(accepted), "successful_runtime_receipts": sum(r["runtime_completed"] for r in rows),
            "measurements_through_utc": max((e.get("finished_at_utc", "") for e in accepted.values()), default="") or None,
            "attempts_pending_submission": 10 - len(accepted), "rows": rows, "sources": sources,
            "independent_frozen_primary_validation": validation,
            "metric_definitions": {
                "task_completed_correctly": "Native complete-task success, task-mean across the frozen cohort; fraction0–1.",
                "partial_credit": "Native partial credit including its declared preservation penalties, task-mean; fraction0–1.",
                "paired_final_vs_baseline": "Per-arm task bootstrap from its own fresh base. Parser regimes/modes are separate controls.",
                "missing_post_update": "null when no post-update native population exists, not a zero score or zero gain.",
                "checkpoint_curve": "Observed checkpoints only. Peak values are not a reserve-based winner or durable final performance.",
                "loss": "Batch-averaged SDPO objective on different fresh on-policy actions; changes do not establish reward gain."},
            "reproduction": "PYTHONPATH=scripts/qualification .venv/bin/python scripts/qualification/"
                            "summarize_sdpo_search.py docs/research/evidence/sdpo-vllm-20261018 --output <snapshot.json>"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.root), indent=2) + "\n")
