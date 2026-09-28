"""Measure rollout-engine throughput at one dtype with the exact engine of a rollout binding.

usage (inside the runtime image, see run.sh; engine JSON from export_engine.py):
  bench_rollout_engine.py --engine /work/engine_c144.json --dtype float16 \
      --prompts /work/prompts.json --num-prompts 24 --samples-per-prompt 6 \
      [--lora /work/adapter] --out /work/bench_float16.json

Every prompt is sampled --samples-per-prompt times (a GRPO/SAMPO group) with the
binding's sampling, all requests submitted at once so the scheduler runs at the
binding's max_num_seqs, as a rollout collection does. With --lora the requests use
that adapter through the Punica LoRA path, as TRL's LoRA weight sync does. Reports
wall time, generated tokens per second, prefix-cache and speculative-decoding
acceptance, and non-finite sampler log-probabilities.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any


def _counters(llm: Any) -> dict[str, float]:
    values: dict[str, float] = {}
    for metric in llm.get_metrics():
        value = getattr(metric, "value", None)
        if isinstance(value, int | float):
            values[metric.name] = values.get(metric.name, 0.0) + float(value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--dtype", required=True, choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--prompts", default="/work/prompts.json")
    parser.add_argument("--num-prompts", type=int, default=24)
    parser.add_argument("--samples-per-prompt", type=int, default=6)
    parser.add_argument("--lora", help="PEFT adapter directory used for every request")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.engine).read_text())
    os.environ.update(config["env"])
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    kwargs = dict(config["llm"])
    kwargs["dtype"] = args.dtype
    lora = LoRARequest("policy", 1, args.lora) if args.lora else None
    if lora is None:
        kwargs.pop("enable_lora", None)
        kwargs.pop("max_loras", None)
        kwargs.pop("max_lora_rank", None)
    llm = LLM(**kwargs)
    sampling = config["sampling"]
    prompts = json.loads(Path(args.prompts).read_text())[: args.num_prompts]
    requests, params = [], []
    for index, prompt in enumerate(prompts):
        for sample in range(args.samples_per_prompt):
            requests.append(TokensPrompt(prompt_token_ids=prompt["prompt_ids"]))
            params.append(
                SamplingParams(
                    temperature=sampling.get("temperature", 1.0),
                    top_p=sampling.get("top_p", 1.0),
                    top_k=sampling.get("top_k", 0) or 0,
                    repetition_penalty=sampling.get("repetition_penalty", 1.0),
                    presence_penalty=sampling.get("presence_penalty", 0.0),
                    max_tokens=sampling.get("max_tokens", prompt["max_tokens"]),
                    logprobs=0,
                    seed=1000 * index + sample,
                )
            )
    warm = [TokensPrompt(prompt_token_ids=p["prompt_ids"][:512]) for p in prompts[: args.warmup]]
    llm.generate(warm, SamplingParams(max_tokens=64), use_tqdm=False, lora_request=lora)
    before = _counters(llm)
    started = time.perf_counter()
    outputs = llm.generate(requests, params, use_tqdm=False, lora_request=lora)
    elapsed = time.perf_counter() - started
    after = _counters(llm)
    delta = {name: after.get(name, 0.0) - before.get(name, 0.0) for name in after}
    generated = sum(len(output.outputs[0].token_ids) for output in outputs)
    prompt_tokens = sum(len(output.prompt_token_ids or []) for output in outputs)
    nonfinite = sum(
        1
        for output in outputs
        for token, step in zip(output.outputs[0].token_ids, output.outputs[0].logprobs or [], strict=True)
        if not math.isfinite(step[token].logprob)
    )
    drafts = delta.get("vllm:spec_decode_num_drafts", 0.0)
    draft_tokens = delta.get("vllm:spec_decode_num_draft_tokens", 0.0)
    accepted = delta.get("vllm:spec_decode_num_accepted_tokens", 0.0)
    queries = delta.get("vllm:prefix_cache_queries", 0.0)
    summary = {
        "engine": config["inference_binding"],
        "dtype": args.dtype,
        "lora": bool(lora),
        "requests": len(requests),
        "seconds": elapsed,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated,
        "generated_tokens_per_second": generated / elapsed,
        "mean_completion_tokens": generated / len(requests),
        "truncated_fraction": sum(o.outputs[0].finish_reason == "length" for o in outputs) / len(requests),
        "spec_acceptance_rate": accepted / draft_tokens if draft_tokens else None,
        "spec_mean_accepted_length": 1 + accepted / drafts if drafts else None,
        "prefix_cache_hit_rate": delta.get("vllm:prefix_cache_hits", 0.0) / queries if queries else None,
        "nonfinite_sampler_logps": nonfinite,
        "env": config["env"],
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
