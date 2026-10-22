"""Collect retained experiment evidence and status without launching GPU work."""

import argparse
import json
import subprocess
from pathlib import Path

from audit_luna_solver_pilot import audit
from stage_luna_solver_experiment import HOST, SSH, remote


def collect(number):
    project = Path(__file__).resolve().parents[2]
    root = project / "docs/research/evidence/luna-ssr-sft-20261018/adaptive-five"
    local = root / f"experiment{number:02d}"
    launch = json.loads((local / "launch.json").read_text())
    capsule = launch["capsule"]
    states = {}
    for size, name in zip(("4b", "9b"), launch["run_ids"], strict=True):
        try:
            state = json.loads(remote("docker inspect --format '{{json .State}}' " + name))
        except subprocess.CalledProcessError:
            states[size] = {"status": "not_submitted"}
            continue
        exists = remote(f"if test -d {capsule}/evidence/{size}; then echo present; fi").strip()
        if exists:
            subprocess.run(["scp", "-r", *SSH, f"{HOST}:{capsule}/evidence/{size}", str(local) + "/"], check=True)
        (local / size).mkdir(exist_ok=True)
        status = {
            "name": name,
            "status": state["Status"],
            "exit_code": state["ExitCode"],
            "started_at": state["StartedAt"],
            "finished_at": state["FinishedAt"],
        }
        (local / size / "execution-status.json").write_text(json.dumps(status, indent=2) + "\n")
        states[size] = status
        if state["Status"] == "exited":
            log = remote("docker logs --tail 60 " + name + " 2>&1")
            (local / size / "container-tail.log").write_text(log)
    value = audit(local)
    (local / "audit.json").write_text(json.dumps(value, indent=2) + "\n")
    summary = {}
    for size, model in value["models"].items():
        errors_path = local / size / "instruction-errors.json"
        summary[size] = {
            "state": states[size],
            "stages": [(s["task"], s["stage"]) for s in model["stages"]],
            "instruction_errors": json.loads(errors_path.read_text()) if errors_path.exists() else {},
            "episodes": [
                {"task": e["task"], "arm": e["arm"], "batches": e["parallel_batches"], "traces": e["traces"]}
                for e in model["episodes"]
            ],
        }
    (local / "status-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    ledger = json.loads((root / "ledger.json").read_text())
    entry = next(e for e in ledger["experiments"] if e["number"] == number)
    entry.update(
        status="completed" if all(s["status"] == "exited" for s in states.values()) else "running",
        launch=launch,
        model_states=states,
    )
    (root / "ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int, choices=range(1, 6))
    collect(parser.parse_args().number)
