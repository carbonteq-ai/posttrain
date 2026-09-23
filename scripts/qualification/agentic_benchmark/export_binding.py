#!/usr/bin/env python3
"""Export a catalog inference binding as the vLLM engine arguments it runs with.

Resolves the binding through ``posttrain catalog show`` so the benchmark always
measures the configuration the framework would launch, then applies the same
field mapping as ``VllmEngineConfig.as_vllm_kwargs`` in
``packages/serve/src/posttrain/serve/profiles/base.py``. Colocated-trainer
fields that do not exist on a standalone engine (sleep, weight sync, trainer
memory share) are recorded but not passed to vLLM.

Run from ``apps/lab`` (the reference project) so its catalog overlays load:

    uv run python ../../scripts/qualification/agentic_benchmark/export_binding.py \
        inference/lfm2.5-2.6b-vllm-automationbench-rollout-local-c32-4k@2 \
        --lora-rank 4 --output ../../scripts/qualification/agentic_benchmark/bindings/lfm25-rollout-c32-4k-v2.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

# vLLM's supported LoRA ranks; the framework rounds a trained rank up to these.
_SUPPORTED_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)
_COLOCATION_ONLY = (
    "mode",
    "request_mode",
    "sleep_during_optimization",
    "weight_sync_mode",
    "free_cache_engine",
    "gpu_memory_utilization",
)


def engine_args_from_binding(engine: dict, lora_rank: int | None) -> dict:
    args: dict[str, object] = {"dtype": engine.get("dtype", "bfloat16")}
    for name in (
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "kv_cache_memory_bytes",
        "enforce_eager",
        "enable_chunked_prefill",
        "enable_prefix_caching",
        "tensor_parallel_size",
        "kv_cache_dtype",
    ):
        if name in engine:
            args[name] = engine[name]
    if engine.get("text_only"):
        args["limit_mm_per_prompt"] = {"image": 0, "video": 0, "audio": 0}
    if engine.get("skip_mm_profiling"):
        args["skip_mm_profiling"] = True
    attention: dict[str, object] = {}
    if engine.get("flash_attn_version") is not None:
        attention["flash_attn_version"] = engine["flash_attn_version"]
    if engine.get("attention_backend_priority"):
        attention["backend"] = engine["attention_backend_priority"][0]
    if attention:
        args["attention_config"] = attention
    if engine.get("speculative"):
        args["speculative_config"] = engine["speculative"]
    if lora_rank:
        args["enable_lora"] = True
        args["max_loras"] = 1
        args["max_lora_rank"] = next(r for r in _SUPPORTED_LORA_RANKS if r >= lora_rank)
    # A standalone engine has the whole GPU; KV capacity stays pinned by
    # kv_cache_memory_bytes, so the utilization bound only needs to admit it.
    args["gpu_memory_utilization"] = 0.85
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("binding_id")
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shown = subprocess.run(
        ["uv", "run", "posttrain", "catalog", "show", "inference", args.binding_id],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    selection = json.loads(shown)["selection"]
    engine = selection["engine"]
    exported = {
        "binding_id": args.binding_id,
        "backend": selection.get("backend"),
        "model": selection["model"].get("artifact") or selection["model"].get("base"),
        "engine_args": engine_args_from_binding(engine, args.lora_rank),
        "lora_rank": args.lora_rank,
        "not_applied": {k: engine[k] for k in _COLOCATION_ONLY if k in engine},
        "binding_engine": engine,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(exported, indent=1) + "\n")
    print(json.dumps(exported["engine_args"], indent=1))


if __name__ == "__main__":
    main()
