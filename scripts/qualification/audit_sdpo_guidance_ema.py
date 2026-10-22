"""Recompute the final EMA from retained student exports without allocating a GPU."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file


def compare(initial, students, actual, rate):
    if not 0 < rate <= 1 or not students or not initial or set(initial) != set(actual):
        raise ValueError("EMA authority/keys/update rate differ")
    if any(not value.is_floating_point() or not torch.isfinite(value).all() for value in initial.values()):
        raise ValueError("initial adapter dtype/finite check failed")
    expected = {key: value.clone() for key, value in initial.items()}
    for student in students:
        if set(student) != set(expected):
            raise ValueError("student export keys differ")
        for key, value in expected.items():
            update = student[key]
            if update.shape != value.shape or update.dtype != value.dtype or not torch.isfinite(update).all():
                raise ValueError("student export shape/dtype/finite check failed")
            expected[key] = value.lerp(update, rate)
    details, error_squared, movement_squared = {}, 0., 0.
    for key, value in expected.items():
        retained = actual[key]
        if retained.shape != value.shape or retained.dtype != value.dtype or not torch.isfinite(retained).all():
            raise ValueError("retained teacher shape/dtype/finite check failed")
        error = (retained.double() - value.double())
        scale = max(float(value.abs().max()), float(initial[key].abs().max()))
        # CPU/CUDA lerp can differ by rounding. Bound error by dtype precision
        # and recurrence depth rather than a fixed tolerance that swallows small B updates.
        tolerance = 8 * len(students) * torch.finfo(value.dtype).eps * scale
        maximum = float(error.abs().max())
        if maximum > tolerance:
            raise ValueError(f"teacher differs from retained-student EMA recurrence: {key}")
        error_squared += float(error.square().sum())
        movement_squared += float((value.double() - initial[key].double()).square().sum())
        details[key] = {"dtype": str(value.dtype), "max_abs_error": maximum,
                        "precision_bound": tolerance}
    if not any(value.any() for key, value in expected.items() if "lora_B" in key):
        raise ValueError("expected EMA has no applied B update")
    if not movement_squared or not math.isfinite(error_squared + movement_squared):
        raise ValueError("EMA movement is absent or nonfinite")
    return {"qualified": True, "updates": len(students), "teacher_update_rate": rate,
            "squared_error": error_squared, "expected_squared_movement": movement_squared,
            "error_relative_to_expected_movement": math.sqrt(error_squared / movement_squared),
            "tensors": details,
            "scope": "Independent CPU recurrence from all retained student exports; precision-bounded, not bitwise CUDA equivalence or learning evidence."}


def audit(root):
    torch.set_num_threads(2)
    inputs = json.loads((root / "inputs.json").read_text())
    updates = json.loads((root / "updates.json").read_text())
    if (inputs["run_profile"] != "guidance-five" or inputs["steps"] != 10
            or [row["step"] for row in updates] != list(range(1, 11))
            or any(row["ema_updates"] != row["step"] for row in updates)
            or not (root / "parity-v10.json").is_file()):
        raise ValueError("ten closed optimizer/EMA exports are required")
    paths = [root / "initial-student/adapter_model.safetensors"]
    paths += [root / f"rollout-policy-v{step}/adapter_model.safetensors" for step in range(1, 11)]
    teacher = root / "checkpoint-10/sdpo_teacher/adapter_model.safetensors"
    # Small LoRA exports only; no base-model tensors or model execution.
    result = compare(load_file(paths[0]), [load_file(p) for p in paths[1:]],
                     load_file(teacher), inputs["teacher_rate"])
    result["source_sha256"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in paths + [teacher]}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.root), indent=2) + "\n")
