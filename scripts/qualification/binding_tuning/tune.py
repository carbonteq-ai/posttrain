#!/usr/bin/env python3
"""Measure one inference binding's throughput under engine-setting variants.

Loads a binding exported by ``scripts/qualification/agentic_benchmark/export_binding.py``
(the same field mapping the framework applies), applies each named variant, and
runs a fixed synthetic workload with vLLM's offline engine: ``requests`` prompts of
``input_tokens`` random tokens, each forced to ``output_tokens`` outputs, submitted at
once so the engine runs at its ``max_num_seqs`` concurrency. Each variant starts a
fresh engine in a subprocess, so a variant that cannot start (for example CUDA graphs
exceeding the memory budget) is recorded as a failure instead of stopping the sweep.

The binding's target memory is emulated on a larger GPU: ``gpu_memory_utilization``
is scaled by ``target_gb / device_gb``, since vLLM treats it as a fraction of the
whole device.

    python tune.py --binding bindings/qwen35-2b-eval.json --target-gb 8 \\
        --variants '{"binding": {}, "graphs": {"enforce_eager": false}}' \\
        --input-tokens 2048 --output-tokens 512 --output results/qwen35-2b-eval.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path


def _run_variant(model: str, revision: str | None, engine_args: dict, shape: dict, queue: mp.Queue) -> None:
    try:
        import torch
        from vllm import LLM, SamplingParams, TokensPrompt

        started = time.perf_counter()
        llm = LLM(model=model, revision=revision, tokenizer_revision=revision, **engine_args)
        startup = time.perf_counter() - started
        vocab = llm.get_tokenizer().vocab_size
        generator = torch.Generator().manual_seed(0)
        prompts = [
            TokensPrompt(prompt_token_ids=torch.randint(100, vocab - 100, (shape["input_tokens"],), generator=generator).tolist())
            for _ in range(shape["requests"])
        ]
        params = SamplingParams(
            max_tokens=shape["output_tokens"], min_tokens=shape["output_tokens"], ignore_eos=True, temperature=0.8, top_p=0.95
        )
        llm.generate(prompts[: min(4, len(prompts))], params, use_tqdm=False)  # warm-up
        started = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        wall = time.perf_counter() - started
        produced = sum(len(o.outputs[0].token_ids) for o in outputs)
        queue.put(
            {
                "ok": True,
                "startup_s": startup,
                "wall_s": wall,
                "output_tokens_per_s": produced / wall,
                "requests_per_s": len(outputs) / wall,
                "peak_memory_gb": torch.cuda.max_memory_reserved() / 2**30,
            }
        )
    except BaseException as error:  # noqa: BLE001 - report every startup or runtime failure
        queue.put({"ok": False, "error": f"{type(error).__name__}: {str(error)[-600:]}"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--model", default=None, help="override the binding's model path")
    parser.add_argument("--target-gb", type=float, required=True)
    parser.add_argument("--device-gb", type=float, default=None)
    parser.add_argument("--variants", required=True, help="JSON object: name -> engine-arg overrides")
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--requests", type=int, default=None, help="default: 3x max_num_seqs, at least 8")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    exported = json.loads(args.binding.read_text())
    base = dict(exported["engine_args"])
    base.pop("enable_lora", None), base.pop("max_loras", None), base.pop("max_lora_rank", None)
    source = exported["model"]
    # Catalog models are pinned {repo_id, revision}; local paths are plain strings.
    model = args.model or (source["repo_id"] if isinstance(source, dict) else source)
    revision = None if args.model else (source.get("revision") if isinstance(source, dict) else None)
    if args.device_gb is None:
        import torch

        args.device_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    results = {"binding": exported.get("binding_id"), "target_gb": args.target_gb, "variants": {}}
    for name, overrides in json.loads(args.variants).items():
        # The export pins a replay-friendly fraction; tuning starts from the binding's own.
        binding_fraction = (exported.get("binding_engine") or {}).get("gpu_memory_utilization")
        engine = {**base, **({"gpu_memory_utilization": binding_fraction} if binding_fraction else {}), **overrides}
        fraction = engine.get("gpu_memory_utilization", 0.9)
        engine["gpu_memory_utilization"] = round(fraction * args.target_gb / args.device_gb, 4)
        seqs = int(engine.get("max_num_seqs") or 8)
        shape = {
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "requests": args.requests or max(8, 3 * seqs),
        }
        queue: mp.Queue = mp.get_context("spawn").Queue()
        process = mp.get_context("spawn").Process(target=_run_variant, args=(model, revision, engine, shape, queue))
        process.start()
        process.join(args.timeout)
        if process.is_alive():
            process.kill()
            outcome = {"ok": False, "error": "timeout"}
        else:
            outcome = queue.get() if not queue.empty() else {"ok": False, "error": f"exit {process.exitcode}"}
        outcome.update({"engine_overrides": overrides, "shape": shape, "effective_gpu_memory_utilization": engine["gpu_memory_utilization"]})
        results["variants"][name] = outcome
        status = (
            f"{outcome['output_tokens_per_s']:.1f} out tok/s, startup {outcome['startup_s']:.0f}s, peak {outcome['peak_memory_gb']:.1f} GB"
            if outcome["ok"]
            else outcome["error"][:200]
        )
        print(f"[{results['binding']}] {name}: {status}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
