"""Seal experiment10 validation without resetting its already-running total budget."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from sdpo_agent_config import MODEL_REVISION, fingerprint
from sdpo_execution_budget import remaining_runtime, reserve_profile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage(fitting, primary_path, adapter, destination):
    if destination.exists():
        raise FileExistsError("preserve existing validation capsule")
    for line in (fitting / "source-sha256.txt").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        source = fitting / relative
        source.resolve().relative_to(fitting.resolve())
        if digest(source) != expected:
            raise ValueError("parent source manifest changed")
    configuration = json.loads((fitting / "experiment.json").read_text())
    if configuration["budget_slot"] != 10 or configuration["steps"] != 10 or not configuration["shared_slot10_budget"]:
        raise ValueError("validation must consume the declared experiment10 fitting budget")
    parent_name = configuration["run_name"]
    state = subprocess.check_output(["docker", "inspect", "--format", "{{.State.Status}}:{{.State.ExitCode}}",
                                     parent_name], text=True, timeout=15).strip()
    if state != "exited:0":
        raise ValueError("parent fitting has not succeeded; no validation submission")
    receipt = json.loads((fitting / "evidence/run/receipt.json").read_text())
    updates = json.loads((fitting / "evidence/run/updates.json").read_text())
    if (not receipt["completed"] or receipt["optimizer_updates"] != 10
            or [u["step"] for u in updates] != list(range(1, 11))
            or any(u["batch"] != 16 or u["optimizer_moment_steps"] != [u["step"]]
                   or u["ema_updates"] != u["step"] for u in updates)):
        raise ValueError("parent actual optimizer/EMA completion evidence incomplete")
    budget_path = fitting / "evidence/shared-budget.json"
    budget = json.loads(budget_path.read_text())
    if budget["fitting_run_name"] != parent_name:
        raise ValueError("shared budget belongs to another fitting run")
    remaining = remaining_runtime(budget, 7100)
    selection = json.loads(primary_path.read_text())
    reserve_profile(selection)
    if (not selection["approved_for_reserve"] or not selection["frozen_before_reserve"]
            or selection["model_revision"] != MODEL_REVISION or not 1 <= selection["budget_slot"] <= 10
            or selection["adapter_sha256"] != digest(adapter / "adapter_model.safetensors")):
        raise ValueError("primary identity/adapter/selection evidence mismatch")
    cohorts = json.loads((fitting / "cohorts.json").read_text())
    if selection["reserve_sha256"] != fingerprint(cohorts["reserve"]):
        raise ValueError("frozen reserve task identity mismatch")
    # Do not create a fresh budget: byte-copy the parent's receipt, including its deadline.
    destination.mkdir(parents=True)
    shutil.copytree(fitting / "driver", destination / "driver")
    (destination / "candidate").mkdir()
    (destination / "evidence").mkdir()
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        shutil.copyfile(adapter / name, destination / "candidate" / name)
    for source, name in ((budget_path, "parent-budget.json"), (primary_path, "primary-selection.json"),
                         (fitting / "cohorts.json", "cohorts.json")):
        shutil.copyfile(source, destination / name)
    config = {"budget_slot": 10, "steps": 0, "wall_time_cap_seconds": 7200,
              "shared_slot10_budget": True, "parent_fitting_run_name": parent_name,
              "run_name": configuration.get("validation_run_name", "sdpo-qwen35-4b-agent10-reserve-20261018-r1"),
              "remaining_seconds_at_staging": remaining,
              "parent_source_manifest_sha256": digest(fitting / "source-sha256.txt"),
              "scope": "Zero-update validation phase of experiment10; no new budget or fitting."}
    (destination / "experiment.json").write_text(json.dumps(config, indent=2) + "\n")
    files = sorted(p for p in destination.rglob("*") if p.is_file())
    (destination / "source-sha256.txt").write_text("".join(
        f"{digest(p)}  {p.relative_to(destination)}\n" for p in files))
    remaining_runtime(budget, 7100)  # staging/copy time is charged against the original deadline
    return {"staged": True, "GPU_submitted": False, "remaining_seconds": remaining,
            "source_manifest_sha256": digest(destination / "source-sha256.txt"),
            "parent_budget_sha256": digest(budget_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fitting", "primary", "adapter", "destination"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.fitting, args.primary, args.adapter, args.destination)))
