"""Read-only collection and audit of one accepted generation experiment."""

import argparse
import json
import subprocess
from pathlib import Path

from audit_luna_solver_pilot import audit
from stage_luna_solver_experiment import HOST, SSH, remote


def collect(number):
    root = Path(__file__).resolve().parents[2] / "docs/research/evidence/luna-ssr-sft-20261018/adaptive-next-five"
    local = root / f"experiment{number:02d}"
    launch = json.loads((local / "launch.json").read_text())
    config = json.loads((local / "experiment.json").read_text())
    name, capsule, model = launch["run_id"], launch["capsule"], config["model"]
    state = json.loads(remote(f"docker inspect -f '{{{{json .State}}}}' {name}"))
    (local / "execution-state.json").write_text(json.dumps(state, indent=2) + "\n")
    logs = remote(f"docker logs --tail 20 {name}")
    (local / "container-tail.log").write_text(logs)
    subprocess.run(["scp", *SSH, "-r", f"{HOST}:{capsule}/evidence/{model}", str(local)], check=True)
    result = audit(local)
    (local / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    selected = result["models"][model]
    receipt_path = local / model / "receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    summary = {"run_id": name, "state": state["Status"], "exit_code": state["ExitCode"],
               "instruction_errors": receipt.get("instruction_errors"),
               "critic_rejected_diagnostic_tasks": receipt.get("review_rejected_diagnostic_tasks"),
               "episodes": [{"task": e["task"], "arm": e["arm"], "id": e["episode_id"],
                             "scores": [{"success": t["metrics"].get("task_completed_correctly"),
                                         "partial": t["rewards"].get("partial_credit", {}).get("score"),
                                         "stop": t["stop_condition"], "errors": t["errors"]}
                                        for t in e["traces"]],
                             "finishes": e["call_finish_reasons"], "flags": e["observation_review_flags"],
                             "empty_responses": e["observed_empty_tool_responses"],
                             "error_responses": e["observed_error_tool_responses"]}
                            for e in selected["episodes"]]}
    (local / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int, choices=range(1, 6))
    collect(parser.parse_args().number)
