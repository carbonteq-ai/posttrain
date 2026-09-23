#!/usr/bin/env python3
"""Write a synthetic LoRA adapter shaped like the rollout policy's adapter.

Production AutomationBench rollouts serve the base model plus a rank-4
all-linear policy LoRA. The adapter's values do not affect replay timing
(output lengths are fixed), only its shapes and target modules do, so this
builds one from the base checkpoint's linear weights without PEFT.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# LFM2.5 linear projections: attention q/k/v/out, feed-forward w1/w2/w3, and
# the short-conv in/out projections.
DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3")


def write_adapter(model_dir: str, out_dir: Path, rank: int, targets: tuple[str, ...]) -> list[str]:
    shapes = {}
    for shard in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(shard, framework="pt") as handle:
            for key in handle.keys():
                parts = key.split(".")
                if key.endswith(".weight") and len(parts) >= 3 and parts[-2] in targets:
                    shape = handle.get_slice(key).get_shape()
                    if len(shape) == 2:
                        shapes[key.removesuffix(".weight")] = shape
    generator = torch.Generator().manual_seed(0)
    tensors = {}
    for module, (out_features, in_features) in sorted(shapes.items()):
        tensors[f"base_model.model.{module}.lora_A.weight"] = (
            torch.randn(rank, in_features, generator=generator) * 0.02
        ).to(torch.bfloat16)
        tensors[f"base_model.model.{module}.lora_B.weight"] = torch.zeros(out_features, rank, dtype=torch.bfloat16)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_dir / "adapter_model.safetensors"))
    found = sorted({m.split(".")[-1] for m in shapes})
    (out_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": rank,
                "lora_alpha": 2 * rank,
                "lora_dropout": 0.0,
                "bias": "none",
                "target_modules": found,
                "base_model_name_or_path": model_dir,
                "fan_in_fan_out": False,
                "use_rslora": False,
                "modules_to_save": None,
            }
        )
    )
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    args = parser.parse_args()
    found = write_adapter(args.model_dir, args.out, args.rank, tuple(args.targets.split(",")))
    print("targets:", found)


if __name__ == "__main__":
    main()
