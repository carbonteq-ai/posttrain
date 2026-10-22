"""Read-only native/data/optimizer gate for the fresh full-pool campaign."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from audit_sdpo_checkpoint_files import audit as checkpoint_audit
from sdpo_pool124_config import step_tasks, validate_pool
from self_campaign_config import summarize


def qualify(capsule):
    root = capsule / "evidence/qualify"
    cohort = json.loads((capsule / "cohorts.json").read_text())
    pool = json.loads((capsule / "train-pool.json").read_text())
    heldout = json.loads((capsule / "heldout62.json").read_text())
    validate_pool(cohort, pool, heldout)
    receipt = json.loads((root / "receipt.json").read_text())
    updates = json.loads((root / "updates.json").read_text())
    if (not receipt["completed"] or receipt["optimizer_updates"] != 4 or len(updates) != 4
            or receipt["elapsed_s"] > 7200):
        raise ValueError("incomplete pool124 qualification")
    files = checkpoint_audit(root)
    if len(files["checkpoints"]) != 4:
        raise ValueError("four actual optimizer checkpoints required")
    seen, visited, populations = set(), set(), []
    for path in sorted(root.glob("step-*/*/native/traces.jsonl")):
        folder = path.parents[1]
        step = int(folder.parent.name.split("-")[1])
        names = step_tasks(cohort["train"], step if folder.name == "online_train" else 1)
        episodes = [json.loads(line) for line in path.read_text().splitlines() if line]
        ids = {e["id"] for e in episodes}
        if len(ids) != 64 or ids & seen:
            raise ValueError("native episode count/identity collision")
        seen |= ids
        if Counter(e["task"]["data"]["name"] for e in episodes) != Counter({n: 2 for n in names}):
            raise ValueError("native32task/two-replica mismatch")
        summary = summarize(episodes, names, 2)
        if summary != json.loads((folder / "outcomes.json").read_text()) or summary["errors"]:
            raise ValueError("native scoring error or retained summary mismatch")
        for metric in ("task_completed_correctly", "partial_credit"):
            if summary["metrics"][metric]["episode_coverage"] != 64:
                raise ValueError("incomplete qualification score coverage")
        populations.append({"path": str(folder.relative_to(root)), "summary": summary,
                            "native_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        if folder.name == "online_train":
            admission = json.loads((folder / "training-admission.json").read_text())
            selected = admission["selected"]
            if (admission["truncated_training_sequences"] or len(selected) != 32
                    or Counter(r["task"] for r in selected) != Counter(names)
                    or len({(r["episode_id"], r["generation_index"]) for r in selected}) != 32):
                raise ValueError("actual one-example-per-distinct-task admission failed")
            visited |= {r["task"] for r in selected}
    if len(seen) != 576 or visited != set(cohort["train"]):
        raise ValueError("did not qualify entire124task training pool")
    for step, row in enumerate(updates, 1):
        if (row["step"] != step or row["batch"] != 32 or row["optimizer_moment_steps"] != [step]
                or row["ema_updates"] != step or not math.isfinite(row["loss"])):
            raise ValueError("actual optimizer/EMA counters or loss invalid")
        metadata = json.loads((root / f"checkpoint-{step}/campaign-checkpoint.json").read_text())
        for relative, digest in metadata["files_sha256"].items():
            if hashlib.sha256((root / f"checkpoint-{step}" / relative).read_bytes()).hexdigest() != digest:
                raise ValueError("checkpoint file changed after receipt")
    for version in range(5):
        parity = json.loads((root / f"parity-v{version}.json").read_text())
        if not math.isfinite(parity["max_abs_delta"]) or parity["max_abs_delta"] > .15:
            raise ValueError("actor/vLLM parity failed")
    long = json.loads((root / "full-evaluation-budget-qualification/receipt.json").read_text())
    invariant = long["batch_invariance"]
    if (not long["qualified"] or not invariant["exact_batch_invariance"]
            or max(r["peak_in_flight"] for r in invariant["records"]) != 32):
        raise ValueError("64K/32slot serving/native gates incomplete")
    gate = {"qualified": True, "actual_updates": 4, "distinct_training_tasks": 124,
            "distinct_tasks_per_update": 32, "native_episodes": 576,
            "source_manifest_sha256": hashlib.sha256((capsule / "source-sha256.txt").read_bytes()).hexdigest(),
            "qualification_receipt_sha256": hashlib.sha256((root / "receipt.json").read_bytes()).hexdigest(),
            "scope": "Runtime/data breadth qualification, not reward improvement or crash-resume proof"}
    for name, value in (("qualification-gate", gate), ("qualification-checkpoints", files),
                        ("qualification-native", populations)):
        (capsule / f"evidence/{name}.json").write_text(json.dumps(value, indent=2) + "\n")
    return gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule", type=Path)
    print(json.dumps(qualify(parser.parse_args().capsule)))
