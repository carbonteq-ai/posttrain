"""Read-only CPU audit required before the separately authorized250-update job."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from analyze_sdpo_agent import audit
from audit_sdpo_checkpoint_files import audit as checkpoint_audit


def qualify(capsule):
    root = capsule / "evidence/qualify"
    cohorts = json.loads((capsule / "cohorts.json").read_text())
    native = audit(root, cohorts)
    files = checkpoint_audit(root)
    updates = native["updates"]
    if (not native["completed"] or len(updates) != 2 or native["unique_native_episodes"] != 136
            or not files["qualified"] or len(files["checkpoints"]) != 2):
        raise ValueError("qualification native/checkpoint population incomplete")
    for population in native["populations"]:
        summary = population["summary"]
        if summary["errors"] or any(summary["metrics"][key]["episode_coverage"] != summary["episodes"]
                                    for key in ("task_completed_correctly", "partial_credit")):
            raise ValueError("incomplete native success/partial scoring coverage")
    counts = []
    for step in (1, 2):
        folder = root / f"step-{step:04d}/online_train"
        selected = json.loads((folder / "selected-transitions.json").read_text())
        count = len({(r["episode_id"], r["generation_index"]) for r in selected})
        if count != 32 or Counter(r["task"] for r in selected) != Counter({t: 4 for t in cohorts["train"]}):
            raise ValueError("actual selected batch32 task/uniqueness mismatch")
        directions = json.loads((folder / "teacher-adam-directions.json").read_text())
        if len(directions) != 32:
            raise ValueError("missing actual full-batch Adam direction diagnostics")
        counts.append(count)
    for version in (0, 1, 2):
        parity = json.loads((root / f"parity-v{version}.json").read_text())
        if (not all(math.isfinite(x) for x in parity["actor"] + parity["rollout"])
                or not math.isfinite(parity["max_abs_delta"]) or parity["max_abs_delta"] > .15):
            raise ValueError("actor/rollout parity failed")
    invariant = json.loads((root / "batch-invariance-v0.json").read_text())
    if not invariant["exact_batch_invariance"] or max(r["peak_in_flight"] for r in invariant["records"]) != 32:
        raise ValueError("actual32 concurrency witness missing")
    gate = json.loads((root / "native-tool-qualification/receipt.json").read_text())
    if not gate["qualified"]:
        raise ValueError("native call/result/final-content gate failed")
    out = {"qualified": True, "actual_updates": 2, "batch": 32, "native_episodes": 136,
           "unique_selected_per_update": counts,
           "source_manifest_sha256": hashlib.sha256((capsule / "source-sha256.txt").read_bytes()).hexdigest(),
           "qualification_receipt_sha256": hashlib.sha256((root / "receipt.json").read_bytes()).hexdigest(),
           "scope": "Runtime/batch32 qualification, not reward improvement or exact crash-resume qualification."}
    (capsule / "evidence/qualification-native-audit.json").write_text(json.dumps(native, indent=2) + "\n")
    (capsule / "evidence/qualification-checkpoint-audit.json").write_text(json.dumps(files, indent=2) + "\n")
    (capsule / "evidence/qualification-gate.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule", type=Path)
    print(json.dumps(qualify(parser.parse_args().capsule)))
