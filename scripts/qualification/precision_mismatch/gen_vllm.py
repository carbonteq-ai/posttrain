"""Sample the prompt set with vLLM at one dtype and keep the sampler log-probabilities.

Writes <out-dir>/samples_<tag>.json with each completion's token ids and vLLM's
processed log-probability of every sampled token (after temperature and top-k/top-p,
as TRL requests), plus a summary with generation throughput.

--hooks   eager mode with forward hooks on every vLLM module (largest finite |output|,
          inf and NaN counts) while decoding, then a prefill pass over another sample
          set's prompt+completion (--prefill-samples) with prompt log-probabilities;
          writes <out-dir>/vllm_hooks_<tag>.json instead of a sample set.
--head32  hf_overrides head_dtype=float32 (float32-accumulated LM head and logits).

A dtype the model's kernels reject (for example float32 with Gated-DeltaNet layers)
exits non-zero after writing <out-dir>/failed_<tag>.json with the error.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

STATS: dict[str, dict[str, float]] = defaultdict(lambda: {"max": 0.0, "inf": 0, "nan": 0, "calls": 0})


def _install_hooks(model: Any) -> int:
    def hook(name: str) -> Any:
        def record(_module: Any, _inputs: Any, output: Any) -> None:
            outputs = output if isinstance(output, tuple | list) else (output,)
            for index, tensor in enumerate(outputs):
                if not torch.is_tensor(tensor) or not tensor.is_floating_point() or tensor.numel() == 0:
                    continue
                stats = STATS[f"{name}#{index}" if index else name]
                stats["calls"] += 1
                nan, inf = torch.isnan(tensor), torch.isinf(tensor)
                stats["nan"] += int(nan.sum())
                stats["inf"] += int(inf.sum())
                finite = tensor.detach().float().abs().masked_fill(nan | inf, 0)
                stats["max"] = max(stats["max"], float(finite.max()))

        return record

    handles = [module.register_forward_hook(hook(name)) for name, module in model.named_modules() if name]
    return len(handles)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dtype", required=True, choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--prompts", default="/work/prompts.json")
    parser.add_argument("--out-dir", default="/work")
    parser.add_argument("--tag", help="output name suffix (default: the dtype)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.62)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--language-model-only", action="store_true", help="multimodal checkpoints (Qwen3.5)")
    parser.add_argument("--head32", action="store_true")
    parser.add_argument("--hooks", action="store_true")
    parser.add_argument("--prefill-samples", help="sample set replayed as prefill in --hooks mode")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    tag = args.tag or args.dtype
    out = Path(args.out_dir)
    prompts = json.loads(Path(args.prompts).read_text())
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    extra: dict[str, Any] = {"hf_overrides": {"head_dtype": "float32"}} if args.head32 else {}
    if args.language_model_only:
        extra.update(language_model_only=True, skip_mm_profiling=True)
    try:
        llm = LLM(
            model=args.model,
            revision=args.revision,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            seed=args.seed,
            enforce_eager=args.hooks,
            logprobs_mode="processed_logprobs",
            **extra,
        )
        if args.hooks:
            print("hooked modules", llm.apply_model(_install_hooks), flush=True)
        params = [
            SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=prompt["max_tokens"],
                logprobs=0,
                seed=1000 + i,
            )
            for i, prompt in enumerate(prompts)
        ]
        started = time.perf_counter()
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=p["prompt_ids"]) for p in prompts], params, use_tqdm=False
        )
        elapsed = time.perf_counter() - started
    except Exception as error:  # noqa: BLE001 - record which dtype this model cannot run
        (out / f"failed_{tag}.json").write_text(
            json.dumps({"dtype": args.dtype, "error": repr(error), "traceback": traceback.format_exc()[-4000:]})
        )
        raise
    records, generated, nonfinite = [], 0, 0
    for prompt, output in zip(prompts, outputs, strict=True):
        completion = output.outputs[0]
        logps = [
            step[token].logprob for token, step in zip(completion.token_ids, completion.logprobs or [], strict=True)
        ]
        nonfinite += sum(1 for value in logps if not math.isfinite(value))
        generated += len(completion.token_ids)
        records.append(
            {
                "id": prompt["id"],
                "set": prompt["set"],
                "prompt_ids": prompt["prompt_ids"],
                "completion_ids": list(completion.token_ids),
                "sampler_logps": logps,
                "finish_reason": completion.finish_reason,
            }
        )
    summary = {
        "model": args.model,
        "dtype": args.dtype,
        "sampler": "vllm",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "head32": args.head32,
        "hooks": args.hooks,
        "requests": len(prompts),
        "seconds": elapsed,
        "generated_tokens": generated,
        "tokens_per_second": generated / elapsed,
        "nonfinite_sampler_logps": nonfinite,
        "peak_torch_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    print(json.dumps(summary), flush=True)
    if not args.hooks:
        (out / f"samples_{tag}.json").write_text(json.dumps({"summary": summary, "records": records}))
        return
    decode = {name: dict(values) for name, values in STATS.items()}
    STATS.clear()
    prefill_logps = []
    if args.prefill_samples:
        base = json.loads(Path(args.prefill_samples).read_text())["records"]
        sequences = [TokensPrompt(prompt_token_ids=r["prompt_ids"] + r["completion_ids"]) for r in base]
        prefill_outputs = llm.generate(
            sequences, SamplingParams(temperature=1.0, max_tokens=1, prompt_logprobs=0), use_tqdm=False
        )
        for record, output in zip(base, prefill_outputs, strict=True):
            start = len(record["prompt_ids"])
            ids = record["prompt_ids"] + record["completion_ids"]
            steps = output.prompt_logprobs or []
            prefill_logps.append(
                [steps[start + j][ids[start + j]].logprob for j in range(len(record["completion_ids"]))]
            )
    (out / f"vllm_hooks_{tag}.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "decode": decode,
                "prefill": {name: dict(values) for name, values in STATS.items()},
                "prefill_logps": prefill_logps,
            }
        )
    )


if __name__ == "__main__":
    main()
