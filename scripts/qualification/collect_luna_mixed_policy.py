"""Read-only mixed-role collection; audit policy observation before mutation."""

import json
import subprocess
from pathlib import Path

from audit_luna_solver_pilot import audit, jsonl
from build_luna_solver_viewer import build
from stage_luna_mixed_policy import CAPSULE, ROOT
from stage_luna_solver_experiment import HOST, SSH, remote


def main():
    project = Path(__file__).resolve().parents[2]
    root = project / ROOT
    states = {}
    for phase, directory in (("critic", "critic27b"), ("solver", "4b")):
        launch_path = root / f"{phase}-launch.json"
        if not launch_path.exists():
            continue
        launch = json.loads(launch_path.read_text())
        name = launch["run_id"]
        state = json.loads(remote(f"docker inspect -f '{{{{json .State}}}}' {name}"))
        states[phase] = state
        (root / f"{phase}-state.json").write_text(json.dumps(state, indent=2) + "\n")
        (root / f"{phase}-tail.log").write_text(remote(f"docker logs --tail 20 {name}"))
        subprocess.run(["scp", *SSH, "-r", f"{HOST}:{CAPSULE}/evidence/{directory}", str(root)], check=True)
        (root / directory / "execution-status.json").write_text(json.dumps(state, indent=2) + "\n")
    result = audit(root)
    (root / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    summary = {"states": states, "native_episodes": []}
    if (root / "4b/materials.json").exists():
        materials = json.loads((root / "4b/materials.json").read_text())
        sales = next(m for m in materials.values() if m["task"] == "sales.soft_matching")
        check = next(c for c in sales["goal_manifest"]["checks"] if c["check_id"] == "audit-note-on-target")
        title = next(e["value"]["value"] for e in check["effect_match"]["args"]
                     if e.get("text", {}).get("path", [])[-1:] == ["title"])
        calls = jsonl(root / "4b/solver-calls.jsonl")
        for episode in result["models"]["4b"]["episodes"]:
            row = {k: episode[k] for k in ("task", "arm", "episode_id", "traces", "errors",
                                           "call_finish_reasons", "repeated_exact_tool_calls")}
            row["tool_calls"] = len(episode["tool_calls"])
            row["empty_results"] = len(episode["observed_empty_tool_responses"])
            row["witnessed_prohibited_effects"] = [a for a in episode["native_assessments"]
                                                  if a["reason"] == "witnessed_declared_prohibited_effect"]
            if episode["task"] == sales["task"]:
                selected = [c for c in calls if c["task"] == sales["task"] and c["arm"] == episode["arm"]]
                mutation_checks = []
                for i, call in enumerate(selected):
                    functions = call["response"]["choices"][0]["message"].get("tool_calls") or []
                    if any(c["function"]["name"] in ("salesforce_contact_update", "salesforce_note_create") for c in functions):
                        prior_tools = [m for m in call["clean_student_request"]["messages"] if m["role"] == "tool"]
                        observed = [m for m in prior_tools if title in str(m.get("content", "")) and
                                    "Contact Matching Guidelines" in str(m.get("content", ""))]
                        mutation_checks.append({"actor_turn": i + 1, "policy_in_prior_tool_observations": bool(observed)})
                row["policy_before_mutations"] = mutation_checks
                row["note_goal_assessments"] = [a for a in episode["native_assessments"]
                                                 if a["signal"] == "audit_note_on_target" and a["status"] == "valid"]
                row["initial_actor_policy_literal_absent"] = title not in json.dumps(selected[0]["actual_solver_request"]["messages"])
            summary["native_episodes"].append(row)
        build(root, root.parent / "solver-4b-critic27b-policy.html")
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"states": {k: v["Status"] for k, v in states.items()},
                      "episodes": len(summary["native_episodes"])}))


if __name__ == "__main__":
    main()
