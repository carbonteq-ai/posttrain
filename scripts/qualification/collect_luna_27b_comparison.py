"""Collect the retained 27B run and show saved 4B controls without rerunning them."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from audit_luna_solver_pilot import audit
from build_luna_solver_viewer import build
from stage_luna_solver_experiment import HOST, SSH, remote


def main(suffix="r1"):
    project = Path(__file__).resolve().parents[2]
    evidence = project / "docs/research/evidence/luna-ssr-sft-20261018"
    root = evidence / f"solver-27b-comparison-20261019-{suffix}"
    launch = json.loads((root / "launch.json").read_text())
    name, capsule = launch["run_id"], launch["capsule"]
    state = json.loads(remote(f"docker inspect -f '{{{{json .State}}}}' {name}"))
    (root / "execution-state.json").write_text(json.dumps(state, indent=2) + "\n")
    (root / "container-tail.log").write_text(remote(f"docker logs --tail 30 {name}"))
    subprocess.run(["scp", *SSH, "-r", f"{HOST}:{capsule}/evidence/27b", str(root)], check=True)
    (root / "27b/execution-status.json").write_text(json.dumps(state, indent=2) + "\n")
    if not (root / "4b").exists():
        shutil.copytree(evidence / "adaptive-next-five/experiment05/4b", root / "4b")
    lineage = {
        "4b": "adaptive-next-five/experiment05/4b; saved prior run, not fresh independent replication",
        "27b": {"launch": launch, "execution_state": state},
        "scope": "Three reused training tasks, one rollout per model/task/arm; no learning or generalization claim",
    }
    (root / "comparison-lineage.json").write_text(json.dumps(lineage, indent=2) + "\n")
    result = audit(root)
    (root / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    rows = []
    for size in ("4b", "27b"):
        for episode in result["models"][size]["episodes"]:
            rows.append({"model": size, "task": episode["task"], "arm": episode["arm"],
                         "episode_id": episode["episode_id"], "scores": episode["traces"],
                         "tool_calls": len(episode["tool_calls"]),
                         "tool_call_turns": sum(n > 0 for n in episode["parallel_batches"]),
                         "empty_results": len(episode["observed_empty_tool_responses"]),
                         "tool_error_results": len(episode["observed_error_tool_responses"]),
                         "finish_reasons": episode["call_finish_reasons"],
                         "repeated_exact_calls": episode["repeated_exact_tool_calls"],
                         "witnessed_prohibited_effects": [a for a in episode["native_assessments"]
                                                          if a["reason"] == "witnessed_declared_prohibited_effect"]})
    materials = [json.loads((root / s / "materials.json").read_text()) for s in ("4b", "27b")]
    equal = {materials[0][key]["task"]: materials[0][key] == materials[1][key] for key in materials[0]}
    summary = {"scope": lineage["scope"], "task_material_equality": equal,
               "model_populations": {s: {k: result["models"][s][k] for k in
                                        ("completed_receipt", "expected_episodes", "observed_episodes")}
                                     for s in ("4b", "27b")}, "episodes": rows}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    filename = "solver-27b-vs-4b-comparison.html" if suffix == "r2" else "solver-27b-vs-4b-original.html"
    build(root, evidence / filename)
    for size in ("4b", "27b"):
        for episode in result["models"][size]["episodes"]:
            print(json.dumps({"model": size, "task": episode["task"], "arm": episode["arm"],
                              "traces": episode["traces"], "finishes": episode["call_finish_reasons"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suffix", choices=["r1", "r2"], default="r1")
    main(parser.parse_args().suffix)
