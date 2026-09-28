"""Sample the prompt set with vLLM at one dtype and keep the sampler log-probabilities.

usage: gen_vllm.py DTYPE [--temp T] [--head32] [--hooks]

Writes /work/samples_<DTYPE>[h32][t<T>].json with each completion's token ids
and vLLM's processed log-probability of every sampled token (as TRL requests).

--head32  vLLM hf_overrides head_dtype=float32 (fp32-accumulated LM head, fp32 logits)
--hooks   eager mode with forward hooks on every vLLM module (max finite |output|,
          inf and NaN counts) during decode, then a prefill pass over the bf16
          sample set's prompt+completion with prompt log-probabilities; writes
          /work/vllm_hooks_<DTYPE>.json. Run after the bfloat16 sample set exists.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

MODEL, REVISION = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
DTYPE = sys.argv[1]
HOOKS = "--hooks" in sys.argv
HEAD32 = "--head32" in sys.argv
TEMPERATURE = float(sys.argv[sys.argv.index("--temp") + 1]) if "--temp" in sys.argv else 1.0
SUFFIX = ("h32" if HEAD32 else "") + (f"t{str(TEMPERATURE).replace('.', '')}" if TEMPERATURE != 1.0 else "")
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


def main() -> None:
    prompts = json.loads(Path("/work/prompts.json").read_text())
    extra = {"hf_overrides": {"head_dtype": "float32"}} if HEAD32 else {}
    llm = LLM(
        model=MODEL,
        revision=REVISION,
        dtype=DTYPE,
        language_model_only=True,
        skip_mm_profiling=True,
        gpu_memory_utilization=0.45 if HOOKS else 0.62,
        max_model_len=4096,
        max_num_seqs=16,
        max_num_batched_tokens=512 if HOOKS else 4096,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        seed=0,
        enforce_eager=HOOKS,
        logprobs_mode="processed_logprobs",
        **extra,
    )
    if HOOKS:
        print("hooked modules", llm.apply_model(_install_hooks), flush=True)
    params = [
        SamplingParams(temperature=TEMPERATURE, top_p=1.0, max_tokens=prompt["max_tokens"], logprobs=0, seed=1000 + i)
        for i, prompt in enumerate(prompts)
    ]
    started = time.perf_counter()
    outputs = llm.generate([TokensPrompt(prompt_token_ids=p["prompt_ids"]) for p in prompts], params, use_tqdm=False)
    elapsed = time.perf_counter() - started
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
        "dtype": DTYPE,
        "temperature": TEMPERATURE,
        "head32": HEAD32,
        "hooks": HOOKS,
        "seconds": elapsed,
        "generated_tokens": generated,
        "tokens_per_second": generated / elapsed,
        "nonfinite_sampler_logps": nonfinite,
        "peak_torch_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    print(json.dumps(summary), flush=True)
    if not HOOKS:
        Path(f"/work/samples_{DTYPE}{SUFFIX}.json").write_text(json.dumps({"summary": summary, "records": records}))
        return
    decode = {name: dict(values) for name, values in STATS.items()}
    STATS.clear()
    base = json.loads(Path("/work/samples_bfloat16.json").read_text())["records"]
    sequences = [TokensPrompt(prompt_token_ids=r["prompt_ids"] + r["completion_ids"]) for r in base]
    prefill_outputs = llm.generate(
        sequences, SamplingParams(temperature=1.0, max_tokens=1, prompt_logprobs=0), use_tqdm=False
    )
    prefill_logps = []
    for record, output in zip(base, prefill_outputs, strict=True):
        start = len(record["prompt_ids"])
        ids = record["prompt_ids"] + record["completion_ids"]
        steps = output.prompt_logprobs or []
        prefill_logps.append([steps[start + j][ids[start + j]].logprob for j in range(len(record["completion_ids"]))])
    Path(f"/work/vllm_hooks_{DTYPE}.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "decode": decode,
                "prefill": {name: dict(values) for name, values in STATS.items()},
                "prefill_logps_on_bf16_samples": prefill_logps,
            }
        )
    )


if __name__ == "__main__":
    main()
