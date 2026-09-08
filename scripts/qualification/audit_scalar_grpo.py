"""Audit scalar GRPO/OLMo training mechanics without conflating zero signal and failure."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file


def _metric_rows(path: Path):
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["kind"] == "metric":
            yield row["value"]["step"], {row["value"]["name"]: row["value"]["value"]}
        elif row["kind"] == "metrics":
            yield row["value"]["step"], row["value"]["values"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--require-nonzero-update", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    selection = json.loads((root / "selection.json").read_text())
    result = json.loads((root / "result.json").read_text())
    algorithm = selection["settings"]["algorithm"]
    expected_steps = int(selection["settings"]["loop"]["max_steps"])
    assert algorithm in {"grpo", "olmo3"}
    assert result["summary"]["global_step"] == expected_steps
    rows = list(_metric_rows(root / "observations.jsonl"))
    optimizer_steps = {
        int(step)
        for step, values in rows
        if "train/rl/advantage_abs_mean" in values or "train/grad_norm" in values
    }
    assert optimizer_steps == set(range(1, expected_steps + 1))
    advantage_abs = [
        float(values["train/rl/advantage_abs_mean"])
        for _, values in rows
        if "train/rl/advantage_abs_mean" in values
    ]
    grad_norms = [float(values["train/grad_norm"]) for _, values in rows if "train/grad_norm" in values]
    assert advantage_abs and grad_norms
    assert all(math.isfinite(value) and value >= 0 for value in (*advantage_abs, *grad_norms))
    adapter = root / "training" / algorithm / "adapter" / "adapter_model.safetensors"
    tensors = load_file(adapter)
    assert tensors and all(torch.isfinite(value).all() for value in tensors.values())
    b_norm = math.sqrt(
        math.fsum(float(value.float().square().sum()) for name, value in tensors.items() if "lora_B" in name)
    )
    nonzero_steps = sum(value > 0 for value in advantage_abs)
    if args.require_nonzero_update:
        assert nonzero_steps > 0 and b_norm > 0
    runtime = next(
        values
        for _, values in rows
        if values.get("train/global_step") == expected_steps and "train/runtime_seconds" in values
    )
    active_rows = [
        values
        for _, values in rows
        if any(name.startswith("train/rl/active_sampling_") for name in values)
    ]
    if algorithm == "olmo3":
        assert selection["settings"]["active_sampling"] == {"max_candidate_batches": 10}
        assert active_rows
    report = {
        "algorithm": algorithm,
        "global_steps": expected_steps,
        "steps_with_nonzero_advantage": nonzero_steps,
        "zero_advantage_steps": expected_steps - nonzero_steps,
        "max_grad_norm": max(grad_norms),
        "lora_B_norm": b_norm,
        "nonzero_update": b_norm > 0,
        "runtime_seconds": runtime["train/runtime_seconds"],
        "active_sampling_metric_rows": len(active_rows),
    }
    (root / "scalar-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
