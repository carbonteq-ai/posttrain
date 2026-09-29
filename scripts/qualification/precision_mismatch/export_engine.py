"""Export a rollout inference binding as the vLLM engine the colocated TRL trainer builds.

Runs on the host with the repository environment (it reads the project catalog and
uses the TRL backend's own translation, ``vllm_rollout_options``), and writes the
JSON that bench_rollout_engine.py replays inside the runtime image:

  uv run python scripts/qualification/precision_mismatch/export_engine.py \
      --project apps/lab \
      --inference inference/lfm2.5-2.6b-vllm-automationbench-rollout-local-c144-4k-t05@1 \
      --training training/lfm2.5-2.6b-trl-lora-automationbench-local-g144-w12@1 \
      --out $OUT/engine_c144.json

The LLM arguments mirror the TRL fork's colocated construction (processed log-probs,
LoRA slots sized from the training binding's rank, the binding's scheduler, KV cache,
attention and speculative settings); sleep mode is left out because the benchmark never
sleeps. ``dtype`` comes from the binding (default: the checkpoint) and can be overridden
per run with bench_rollout_engine.py --dtype.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from posttrain.catalog import discover_project, open_catalog
from posttrain.common import CatalogRef
from posttrain.train.backends.trl.common import vllm_rollout_options

_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


def export(project: Path, inference_id: str, training_id: str | None) -> dict[str, Any]:
    layout = discover_project(project, explicit_root=project)
    catalog = open_catalog(
        scope=layout.project_id,
        overlays=layout.catalog_overlays,
        catalog_root=layout.base_catalog,
        required_plugin_distributions=layout.catalog_plugin_requirements,
    )
    binding = catalog.resolve(CatalogRef("inference", inference_id)).value
    engine = binding.engine
    speculative, engine_kwargs = vllm_rollout_options(binding.model, engine)
    llm: dict[str, Any] = {
        "model": binding.model.base.repo_id,
        "revision": binding.model.base.revision,
        "tensor_parallel_size": engine.get("tensor_parallel_size", 1),
        "gpu_memory_utilization": engine.get("gpu_memory_utilization"),
        "max_model_len": engine.get("max_model_len"),
        "seed": 0,
        "logprobs_mode": "processed_logprobs",
        "disable_log_stats": False,
        "speculative_config": speculative,
        **(engine_kwargs or {}),
    }
    env: dict[str, str] = {}
    if engine.get("batch_invariant") is True:
        env["VLLM_BATCH_INVARIANT"] = "1"
    if engine.get("disable_torch_compile") is True:
        env["TORCH_COMPILE_DISABLE"] = "1"
    training = None
    if engine.get("weight_sync_mode") == "lora":
        rank = 8
        if training_id is not None:
            training = catalog.resolve(CatalogRef("training", training_id)).value
            rank = getattr(training.update, "rank", rank)
        llm.update(enable_lora=True, max_loras=1, max_lora_rank=next(r for r in _LORA_RANKS if r >= rank))
    return {
        "inference_binding": inference_id,
        "training_binding": training_id,
        "training_precision": (
            training.backend_options.get("training_precision", "bf16") if training is not None else None
        ),
        "llm": {key: value for key, value in llm.items() if value is not None},
        "env": env,
        "sampling": dict(binding.sampling),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=Path, default=Path("apps/lab"))
    parser.add_argument("--inference", required=True)
    parser.add_argument("--training")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = export(args.project.resolve(), args.inference, args.training)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
