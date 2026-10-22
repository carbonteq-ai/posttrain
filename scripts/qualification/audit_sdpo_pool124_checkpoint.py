"""Read one closed pool124 checkpoint without demanding future checkpoints."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file
from sdpo_agent_config import update_learning_rate
from sdpo_long_campaign_config import campaign_learning_rate


def audit(root, step):
    torch.set_num_threads(2)
    inputs = json.loads((root / "inputs.json").read_text())
    if inputs["run_profile"] == "pool124-bounded":
        from types import SimpleNamespace

        from sdpo_pool124_bounded_config import profile

        profile(SimpleNamespace(**inputs))
        if step not in range(1, inputs["steps"] + 1):
            raise ValueError("not a closed bounded checkpoint")
        expected_lr = update_learning_rate(inputs["lr"], step, inputs["steps"],
                                           inputs["lr_schedule"], inputs["lr_end_ratio"])
    elif inputs["run_profile"] != "pool124-50" or inputs["steps"] != 50 or step not in range(10, 51, 10):
        raise ValueError("not a closed ten-step pool124 checkpoint")
    else:
        expected_lr = campaign_learning_rate(inputs["lr"], step, inputs["lr_end_ratio"])
    folder = root / f"checkpoint-{step}"
    metadata = json.loads((folder / "campaign-checkpoint.json").read_text())
    if (metadata["step"] != step or metadata["ema_updates"] != step or metadata["batch"] != 32
            or metadata["run_profile"] != inputs["run_profile"] or metadata["resume_qualified"]
            or not math.isclose(metadata["learning_rate"], expected_lr, rel_tol=1e-9)):
        raise ValueError("checkpoint counter/profile/LR authority mismatch")
    hashes = {}
    for relative, expected in metadata["files_sha256"].items():
        file = folder / relative
        file.resolve().relative_to(folder.resolve())
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError("checkpoint file hash mismatch")
        hashes[relative] = digest
    required = {"optimizer.pt", "rng_state.pth", "adapter_model.safetensors",
                "sdpo_teacher/adapter_model.safetensors"}
    if not required <= set(hashes):
        raise ValueError("checkpoint receipt lacks required saved files")
    optimizer = torch.load(folder / "optimizer.pt", weights_only=True, map_location="cpu")
    moments = sorted({int(state["step"]) for state in optimizer["state"].values()})
    if not optimizer["state"] or moments != [step]:
        raise ValueError("actual saved moment differs from checkpoint label")
    if (any(not torch.isfinite(v).all() for state in optimizer["state"].values()
            for v in state.values() if isinstance(v, torch.Tensor))
            or any(not math.isclose(g["lr"], expected_lr, rel_tol=1e-9) for g in optimizer["param_groups"])):
        raise ValueError("actual optimizer nonfinite or applied LR mismatch")
    del optimizer
    for relative in ("adapter_model.safetensors", "sdpo_teacher/adapter_model.safetensors"):
        tensors = load_file(folder / relative, device="cpu")
        if not tensors or any(not torch.isfinite(v).all() for v in tensors.values()):
            raise ValueError("nonfinite or empty saved adapter")
        if not any(v.any() for k, v in tensors.items() if "lora_B" in k):
            raise ValueError("saved adapter has unchanged zero-effectB")
    return {"qualified": True, "step": step, "actual_saved_optimizer_moments": moments,
            "learning_rate": expected_lr, "files_sha256": hashes, "GPU_used": False,
            "scope": "Saved file/moment/finite-adapter audit; no independent EMA recurrence or exact crash-resume proof"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("step", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.root, args.step), indent=2) + "\n")
