"""One budgeted experiment, two sequential fresh branches and no pooled updates."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def branch_commands(config, script, root, model, heldout):
    if "training_tasks" in config and config["training_tasks"] != 124:
        raise ValueError("explicit broad-corpus configuration cannot fall back to narrow tasks")
    if config.get("training_tasks") == 124:
        from sdpo_pool124_bounded_config import paired_commands

        return paired_commands(config, script, root, model, heldout, sys.executable)
    if (config["budget_slot"] != 9 or config["steps"] != 20
            or config["context_limit"] != 32768 or config["reply_limit"] != 4096
            or config["wall_time_cap_seconds"] != 7200):
        raise ValueError("paired Luna authorization/profile mismatch")
    common = [sys.executable, str(script), "--model", str(model), "--heldout", str(heldout),
              "--cohorts", str(root / "cohorts.json"), "--context-limit", "32768", "--reply-limit", "4096",
              "--steps", "10", "--lr", str(config["lr"]), "--lr-schedule", config["lr_schedule"],
              "--lr-end-ratio", str(config["lr_end_ratio"]), "--alpha", str(config["alpha"]),
              "--topk", str(config["topk"]), "--teacher-rate", str(config["teacher_rate"]),
              "--selection-mode", config["selection_mode"], "--online-rollouts", "2",
              "--tool-parser-mode", "schema-bound-v1"]
    return [("control", common + ["--teacher-feedback", "solution-or-feedback", "--qualify-capacity"]),
            ("luna", common + ["--teacher-feedback", "luna-or-solution",
                               "--luna-references", str(root / "luna-reviewed.json")])]


def main(args):
    started = time.monotonic()
    config = json.loads((args.capsule / "experiment.json").read_text())
    commands = branch_commands(config, args.capsule / "driver/sdpo_agent_ablation.py",
                               args.capsule, args.model, args.heldout)
    branch_steps = 6 if config.get("training_tasks") == 124 else 10
    args.output.mkdir(exist_ok=False)
    (args.output / "paired-plan.json").write_text(json.dumps(config, indent=2) + "\n")
    phase_limit = int(config["branch_wall_time_seconds"])
    active_cap = int(config["gpu_timeout_seconds"])
    if phase_limit <= 0 or 2 * phase_limit + 60 > active_cap:
        raise ValueError("paired branch caps exceed parent budget")
    completed = []
    for name, command in commands:
        branch = args.output / name
        if name == "luna":
            gate = json.loads((args.output / "control/capacity-qualification.json").read_text())
            if (not gate["qualified"] or gate["optimizer_updates"] != 0
                    or gate["context_limit"] != 32768 or gate["reply_limit"] != 4096):
                raise ValueError("matched larger-context control capacity did not qualify")
        command += ["--output", str(branch), "--run-name", config["run_name"] + "-" + name]
        remaining = active_cap - (time.monotonic() - started) - 30
        if remaining <= 0:
            raise TimeoutError("combined experiment deadline reached")
        print(json.dumps({"stage": "branch_start", "branch": name, "updates": branch_steps}), flush=True)
        with (args.output / f"{name}-process.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=min(phase_limit, remaining))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise TimeoutError(f"{name} exceeded its recorded phase cap") from None
        if code != 0:
            raise RuntimeError(f"{name} exited{code}; preserve evidence, no next branch")
        receipt = json.loads((branch / "receipt.json").read_text())
        updates = json.loads((branch / "updates.json").read_text())
        if (not receipt["completed"] or receipt["optimizer_updates"] != branch_steps
                or len(updates) != branch_steps):
            raise ValueError("branch actual update evidence incomplete")
        completed.append({"name": name, "optimizer_updates": len(updates), "receipt": receipt})
        (args.output / "progress.json").write_text(json.dumps(completed, indent=2) + "\n")
        print(json.dumps({"stage": "branch_complete", "branch": name, "updates": len(updates)}), flush=True)
    result = {"completed": True, "paired_experiment": True, "optimizer_updates": 2 * branch_steps,
              "branches": completed, "elapsed_s": time.monotonic() - started,
              "external_model_calls": 0, "heldout62_evaluated": config.get("training_tasks") == 124,
              "reserve_evaluated": False,
              "scope": f"Two fresh independent{branch_steps}-update branches. "
                       "Compare own matched baselines; never pool."}
    (args.output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stage": "experiment_complete", "optimizer_updates": 2 * branch_steps}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
