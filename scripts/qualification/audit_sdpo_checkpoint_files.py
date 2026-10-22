"""CPU audit of retained SDPO adapter files and actual optimizer moment states."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def audit(root):
    torch.set_num_threads(2)
    from types import SimpleNamespace

    from sdpo_long_campaign_config import execution_profile

    inputs = json.loads((root / "inputs.json").read_text())
    steps = inputs["steps"]
    profile = execution_profile(SimpleNamespace(**inputs))
    rows = []
    for step in range(1, steps + 1):
        if step != steps and step % profile["save_every"]:
            continue
        folder = root / f"checkpoint-{step}"
        files = [folder / "adapter_model.safetensors", folder / "sdpo_teacher/adapter_model.safetensors",
                 folder / "optimizer.pt"]
        if not all(f.is_file() and f.stat().st_size > 0 for f in files):
            raise ValueError(f"missing actual checkpoint files at{step}")
        row = {"step": step, "file_sizes": {str(p.relative_to(folder)): p.stat().st_size for p in files}}
        # Inspect actual saved optimizer tensors, not only their logged counters.
        optimizer = torch.load(folder / "optimizer.pt", map_location="cpu", weights_only=True)
        moments = sorted({int(state["step"]) for state in optimizer["state"].values() if "step" in state})
        if moments != [step] or not optimizer["state"]:
            raise ValueError("saved moment state does not match actual update")
        row["actual_saved_optimizer_moment_steps"] = moments
        del optimizer
        if step % profile["evaluate_every"] == 0 or step == steps:
            row["adapter_sha256"] = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in files[:2]}
            for file in files[:2]:
                values = load_file(file, device="cpu")
                if not all(torch.isfinite(v).all() for v in values.values()):
                    raise ValueError("nonfinite student/teacher adapter")
                if not any(v.any() for k, v in values.items() if "lora_B" in k):
                    raise ValueError("student/teacher has unchanged zero-effect adapter")
                del values
            row["student_and_teacher_finite_nonzero"] = True
        rows.append(row)
    return {"qualified": True, "gpu_used": False, "checkpoints": rows,
            "scope": "Actual saved optimizer moments at every step; finite changed student/teacher tensors and file hashes at saved evaluation checkpoints. EMA recurrence not independently recomputed."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.root), indent=2) + "\n")
