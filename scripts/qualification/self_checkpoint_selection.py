"""Freeze a development-only primary checkpoint before heldout evaluation."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def choose(curve, expected_tasks):
    expected = set(expected_tasks)
    if len(expected_tasks) != 24 or len(expected) != 24:
        raise ValueError("exactly24 unique development tasks required")
    rows = [r for r in curve if r["cohort"] == "development"]
    steps = [r["step"] for r in rows]
    if len(steps) != 9 or set(steps) != set(range(0, 257, 32)):
        raise ValueError("all nine unique development checkpoints required")
    candidates = []
    for row in rows:
        summary = row["summary"]
        if summary["episodes"] != 48 or summary["tasks"] != 24 or summary["errors"]:
            raise ValueError("complete error-free development coverage required")
        scores = {}
        for name in ("task_completed_correctly", "partial_credit", "tool_mistakes"):
            metric = summary["metrics"][name]
            values = metric["complete_task_means"]
            if (metric["episode_coverage"] != 48 or metric["total_episodes"] != 48
                    or set(values) != expected):
                raise ValueError("missing or changed development score population")
            if any(not math.isfinite(v) or v < 0 or (name != "tool_mistakes" and v > 1)
                   for v in values.values()):
                raise ValueError("invalid development metric")
            value = metric["full_population_task_mean"]
            if value is None or not math.isfinite(value) or not math.isclose(value, sum(values.values()) / 24):
                raise ValueError("development aggregate disagrees with task means")
            scores[name] = value
        if row["step"]:
            candidates.append({"step": row["step"], **scores})
    winner = min(candidates, key=lambda r: (-r["task_completed_correctly"], -r["partial_credit"],
                                          r["tool_mistakes"], r["step"]))
    arms = sorted({0, 64, 128, 192, 256, winner["step"]})
    return {"primary_checkpoint": winner["step"], "evaluation_steps": arms,
            "development_scores": sorted(candidates, key=lambda r: r["step"]),
            "selection_rule": "success desc, partial credit desc, tool mistakes asc, step asc",
            "heldout_used_for_selection": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.campaign / "campaign-receipt.json").read_text())
    if not receipt["completed"] or receipt["steps"] != 256 or receipt["batch_size"] != 16:
        raise ValueError("complete256-update batch16 campaign required")
    protocol_raw = args.protocol.read_bytes()
    protocol = json.loads(protocol_raw)
    curve = json.loads((args.campaign / "outcome-curve.json").read_text())
    result = choose(curve, protocol["cohorts"]["development"]["task_names"])
    checkpoints = {}
    for step in set(result["evaluation_steps"]) - {0}:
        folder = args.campaign / "checkpoints" / f"checkpoint-{step}"
        manifest = json.loads((folder / "campaign-checkpoint.json").read_text())
        if manifest["step"] != step or manifest["optimizer_moment_steps"] != [step]:
            raise ValueError("checkpoint/update lineage mismatch")
        digest = hashlib.sha256((folder / "adapter_model.safetensors").read_bytes()).hexdigest()
        if digest != manifest["file_sha256"]["adapter_model.safetensors"]:
            raise ValueError("student adapter digest mismatch")
        checkpoints[str(step)] = {"student_adapter_sha256": digest, "path": str(folder)}
    result.update(checkpoints=checkpoints, campaign=str(args.campaign),
                  protocol_sha256=hashlib.sha256(protocol_raw).hexdigest())
    # Refuse to rewrite a selection after benchmark results become available.
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
