"""Prepare hashed, training-only offline Luna sources for current-runtime CPU review."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare(coverage, cohorts, native, output):
    training = set(cohorts["train"])
    if training & (set(cohorts["development"]) | set(cohorts["reserve"])):
        raise ValueError("task-role overlap")
    current = {e["task"]["data"]["name"]: e["task"]["data"] for e in native}
    output.mkdir(exist_ok=False)
    candidates, missing = [], []
    for name in cohorts["train"]:
        entries = [e for e in coverage["entries"] if e["task_name"] == name]
        if not entries:
            missing.append({"task": name, "reason": "absent_from_saved_reference_index"})
            continue
        if len(entries) != 1:
            raise ValueError("ambiguous offline source")
        entry = entries[0]
        source = Path(entry["source_episode_path"])
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["source_episode_sha256"]:
            raise ValueError("offline source hash mismatch")
        episode = json.loads(raw)
        trace = episode["traces"][0]
        data = trace["task"]["data"]
        reference = current[name]
        if data["initial_state"] != reference["initial_state"] or data["assertions"] != reference["assertions"]:
            raise ValueError("source task state/assertion mismatch")
        source_user = [m["content"] for m in data["prompt"] if m["role"] == "user"]
        current_user = [m["content"] for m in reference["prompt"] if m["role"] == "user"]
        if source_user != current_user:
            raise ValueError("source user instruction mismatch")
        if trace["metrics"].get("task_completed_correctly") != 1 or not episode["ok"]:
            raise ValueError("reference not a recorded successful attempt")
        events = trace["info"]["automationbench_capture"]["events"]
        if not events:
            raise ValueError("reference has no retained action/result evidence")
        (output / f"{name}.json").write_bytes(raw)
        candidates.append({"task": name, "episode_id": episode["id"],
            "source_episode_path": str(source), "source_episode_sha256": entry["source_episode_sha256"],
            "episode_file": f"{name}.json", "initial_state_sha256": digest(data["initial_state"]),
            "assertions_sha256": digest(data["assertions"]), "user_instruction_matches": True,
            "full_prompt_matches": data["prompt"] == reference["prompt"],
            "source_prompt": data["prompt"], "current_prompt": reference["prompt"],
            "recorded_success": True, "recorded_capture_events": len(events),
            "recorded_capture_coverage": trace["info"]["automationbench_capture"].get("coverage"),
            "current_runtime_replay_qualified": False})
    manifest = {"candidates": candidates, "missing": missing, "training_tasks": cohorts["train"],
        "qualified": False, "external_model_calls": 0,
        "scope": "Saved execution observations, not complete token-aligned model trajectories or per-call gold. "
                 "Requires current native final-state/preservation checks and context-length admission."}
    (output / "candidate-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(json.loads(args.coverage.read_text()), json.loads(args.cohorts.read_text()),
        [json.loads(line) for line in args.native.read_text().splitlines() if line], args.output)
    print(json.dumps({"candidates": len(result["candidates"]), "missing": result["missing"], "qualified": False}))
