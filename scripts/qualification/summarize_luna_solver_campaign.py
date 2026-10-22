"""Reconcile fixed-task diagnostic coverage without zero-filling blocked arms."""

import argparse
import json
from pathlib import Path

from audit_luna_solver_pilot import audit


def summarize(root):
    output = {
        "experiments": [],
        "controls": 0,
        "guided": 0,
        "legacy_successes": 0,
        "scored_episodes": 0,
        "maximum_planned_episodes": 40,
        "optimizer_updates": 0,
        "sft_exported": False,
    }
    scoped_ids = set()
    for number in range(1, 6):
        folder = root / f"experiment{number:02d}"
        value = audit(folder)
        (folder / "audit.json").write_text(json.dumps(value, indent=2) + "\n")
        experiment = {"number": number, "models": {}}
        for size, model in value["models"].items():
            materials = {m["task"]: m for m in json.loads((folder / size / "materials.json").read_text()).values()}
            rows = []
            for episode in model["episodes"]:
                scoped_id = (number, size, episode["episode_id"])
                if scoped_id in scoped_ids:
                    raise ValueError("duplicate native episode inside run population")
                scoped_ids.add(scoped_id)
                checks = materials[episode["task"]]["goal_manifest"]["checks"]
                findings = []
                for check in checks:
                    signal = check["signal_id"]
                    if check["role"] == "harm":
                        signal += ".compliance"
                    matches = [a for a in episode["native_assessments"] if a["signal"] == signal]
                    decided = bool(matches) and all(a["status"] == "valid" and a["value"] is not None for a in matches)
                    findings.append(
                        {
                            "check_id": check["check_id"],
                            "role": check["role"],
                            "decided": decided,
                            "satisfied": all(a["value"] == 1 for a in matches) if decided else None,
                        }
                    )
                score = episode["traces"][0]["metrics"].get("task_completed_correctly")
                output["scored_episodes"] += score is not None
                output["legacy_successes"] += score == 1
                output["controls" if episode["arm"] == "control" else "guided"] += 1
                rows.append(
                    {
                        "task": episode["task"],
                        "arm": episode["arm"],
                        "legacy_success": score,
                        "native_findings": findings,
                        "observation_review_flags": episode["observation_review_flags"],
                        "empty_responses": len(episode["observed_empty_tool_responses"]),
                        "error_responses": len(episode["observed_error_tool_responses"]),
                        "finish_reasons": episode["call_finish_reasons"],
                    }
                )
            state = json.loads((folder / size / "execution-status.json").read_text())
            experiment["models"][size] = {
                "state": state,
                "episodes": rows,
                "guided_executed": sum(e["arm"] == "guided" for e in rows),
            }
        output["experiments"].append(experiment)
    output["native_episodes"] = len(scoped_ids)
    output["unexecuted_due_to_preparation_gates"] = output["maximum_planned_episodes"] - len(scoped_ids)
    output["interpretation"] = (
        "Descriptive reused-training-task diagnostics. Controls already score1; no learning or reward-gain claim. Critic-negative native runs in04/05 are diagnostic only."
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    value = summarize(args.root)
    (args.root / "summary.json").write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({k: v for k, v in value.items() if k != "experiments"}, indent=2))
