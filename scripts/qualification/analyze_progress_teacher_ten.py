"""Reconcile all populations/proposals without converting API success to correctness."""

import argparse
import json
from collections import Counter
from pathlib import Path

from progress_teacher_context import digest


def analyze(root, names, native_reference=None):
    populations, identities = {}, set()
    count = len(names)
    for cohort in ["baseline_train10", "online_train"]:
        folder = (native_reference or root) / "step-0000" / cohort
        episodes = [json.loads(s) for s in (folder / "native/traces.jsonl").read_text().splitlines() if s]
        if Counter(e["task"]["data"]["name"] for e in episodes) != Counter({n: 2 for n in names}):
            raise ValueError("native population incomplete")
        ids = {e["id"] for e in episodes}
        if len(ids) != 2 * count or ids & identities:
            raise ValueError("duplicate native episode")
        identities |= ids
        errors = [{"episode_id": e["id"], "field": key, "value": obj[key]}
                  for e in episodes for obj in [e, *e["traces"]]
                  for key in ["errors", "assessment_errors", "credit_errors"] if obj.get(key)]
        populations[cohort] = {"episode_ids": sorted(ids), "errors": errors,
            "initial_state_hashes": {n: sorted({digest(e["task"]["data"]["initial_state"])
                                                for e in episodes if e["task"]["data"]["name"] == n}) for n in names},
            "summary": json.loads((folder / "outcomes.json").read_text()),
            "stop_conditions": dict(Counter(t["stop_condition"] for e in episodes for t in e["traces"]))}
    cases = json.loads((root / "source-cases.json").read_text())["cases"]
    if Counter(c["row"]["task"] for c in cases) != Counter(names):
        raise ValueError("semantic decision population differs")
    proposals = json.loads((root / "same-decision-proposals.json").read_text())
    if Counter((p["task"], p["mode"]) for p in proposals) != Counter(
            (n, m) for n in names for m in ["student", "original_teacher", "revised_teacher"]):
        raise ValueError("three-mode proposal population incomplete")
    return {"populations": populations, "native_episodes": len(identities),
        "native_episodes_referenced": len(identities) if native_reference is not None else 0,
        "native_episodes_new": 0 if native_reference is not None else len(identities),
        "same_decision_generations": len(proposals),
        "proposals": [{"task": p["task"], "mode": p["mode"], "complete": p["complete"],
                       "generation_index": p["generation_index"], "message": p.get("message"),
                       "native_error": p.get("native_error"),
                       "native_replay": p.get("native_replay")} for p in proposals],
        "optimizer_updates": 0, "qualified_for_update": None,
        "review_required": "Independent source/semantic review. API parse/execution and one-decision score do not establish task correctness.",
        "limits": "Training diagnostics only. No missing-score zero fill, safety or generalization claim."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-reference", type=Path)
    args = parser.parse_args()
    result = analyze(args.root, json.loads(args.cohorts.read_text())["train"], args.native_reference)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"native_episodes": result["native_episodes"], "proposals": result["same_decision_generations"],
                      "qualified": result["qualified_for_update"]}))
