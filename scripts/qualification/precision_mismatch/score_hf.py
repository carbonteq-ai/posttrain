"""Trainer-side scoring with the Transformers model, loaded as the TRL trainer loads it.

Scoring (default) reads every <out-dir>/samples_*.json (or --only tags) and writes
<out-dir>/scores_<dtype>[_<tags>].json with, for every sampled token, the trainer's
log-probability computed four ways:

* native_logps: log-softmax in the logits dtype (TRL today for bf16/fp16 logits)
* fp32_logps: model-dtype logits upcast to float32 first (backend_options.logits_float32)
* head32_logps: an fp32 LM head (model-dtype hidden state and weight, float32 products)
* nucleus_logps: fp32_logps after the sampler's top-k/top-p mask and renormalization,
  which is what vLLM's processed log-probabilities are and TRL's trainer does not do

and, for every hooked module output, the largest finite |value|, its token position,
and inf/NaN counts with the first non-finite position.

--generate samples the prompt set with Transformers generate instead (the float32
sampler reference when vLLM cannot run float32) and writes samples_<tag>.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def nucleus_log_softmax(logits: torch.Tensor, top_p: float, top_k: int) -> torch.Tensor:
    """vLLM's top-k/top-p mask (``apply_top_k_top_p_pytorch``) followed by a float32 log-softmax."""

    logits = logits.float()
    if top_k > 0:
        kth = logits.topk(top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, order = logits.sort(dim=-1, descending=False)
        cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative <= 1 - top_p
        remove[..., -1] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.empty_like(sorted_logits).scatter_(-1, order, sorted_logits)
    return torch.log_softmax(logits, dim=-1)


def _generate(model: Any, args: argparse.Namespace) -> None:
    prompts = json.loads(Path(args.prompts).read_text())
    records, started, generated = [], time.perf_counter(), 0
    for index, prompt in enumerate(prompts):
        torch.manual_seed(1000 + index)
        ids = torch.tensor([prompt["prompt_ids"]], device="cuda")
        with torch.no_grad():
            output = model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=1.0,  # override checkpoint generation defaults (LFM2.5: 1.1)
                max_new_tokens=prompt["max_tokens"],
                output_scores=True,
                return_dict_in_generate=True,
            )
        completion = output.sequences[0, ids.shape[1] :].tolist()
        # Scores are the processed logits (temperature and top-k/top-p applied), as vLLM's processed logprobs.
        logps = [
            float(torch.log_softmax(step[0].float(), -1)[token])
            for step, token in zip(output.scores, completion, strict=True)
        ]
        generated += len(completion)
        records.append(
            {
                "id": prompt["id"],
                "set": prompt["set"],
                "prompt_ids": prompt["prompt_ids"],
                "completion_ids": completion,
                "sampler_logps": logps,
                "finish_reason": "hf",
            }
        )
    summary = {
        "model": args.model,
        "dtype": args.dtype,
        "sampler": "hf-generate",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "requests": len(prompts),
        "seconds": time.perf_counter() - started,
        "generated_tokens": generated,
    }
    print(json.dumps(summary))
    tag = args.tag or args.dtype
    (Path(args.out_dir) / f"samples_{tag}.json").write_text(json.dumps({"summary": summary, "records": records}))


def _install_hooks(model: Any, current: dict[str, str | None]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"max": 0.0, "max_pos": -1, "inf": 0, "nan": 0, "first_bad_pos": None}
    )

    def hook(name: str) -> Any:
        def record(_module: Any, _inputs: Any, output: Any) -> None:
            outputs = output if isinstance(output, tuple | list) else (output,)
            for index, tensor in enumerate(outputs):
                if not torch.is_tensor(tensor) or not tensor.is_floating_point() or tensor.dim() < 2:
                    continue
                if tensor.numel() == 0:
                    continue
                entry = stats[f"{name}#{index}" if index else name]
                bad = ~torch.isfinite(tensor)
                if bool(bad.any()):
                    entry["nan"] += int(torch.isnan(tensor).sum())
                    entry["inf"] += int(torch.isinf(tensor).sum())
                    if entry["first_bad_pos"] is None:
                        entry["first_bad_pos"] = {
                            "sample": current["sample"],
                            "index": bad.nonzero()[0].tolist(),
                            "shape": list(tensor.shape),
                        }
                magnitude = tensor.detach().float().abs().masked_fill(bad, 0)
                largest = float(magnitude.max())
                if largest > entry["max"]:
                    flat = int(magnitude.argmax())
                    inner = int(torch.tensor(tensor.shape[2:]).prod()) if tensor.dim() >= 3 else 1
                    entry["max"] = largest
                    entry["max_pos"] = (flat // inner) % tensor.shape[1] if tensor.dim() >= 3 else flat // inner
                    entry["max_sample"] = current["sample"]

        return record

    for name, module in model.named_modules():
        if name:
            module.register_forward_hook(hook(name))
    return stats


def _score(model: Any, args: argparse.Namespace) -> None:
    head = model.get_output_embeddings()
    head_weight32 = head.weight.detach().float()
    backbone = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    current: dict[str, str | None] = {"sample": None}
    stats = _install_hooks(model, current)
    results = {}
    chunk = args.chunk
    for path in sorted(glob.glob(str(Path(args.out_dir) / "samples_*.json"))):
        sampler = Path(path).stem.removeprefix("samples_")
        if args.only and sampler not in args.only:
            continue
        loaded = json.loads(Path(path).read_text())
        summary = loaded["summary"]
        temperature = summary.get("temperature", 1.0)
        top_p, top_k = summary.get("top_p", 1.0), summary.get("top_k", 0)
        scored = []
        for record in loaded["records"]:
            current["sample"] = f"{sampler}:{record['id']}"
            start = len(record["prompt_ids"])
            ids = torch.tensor([record["prompt_ids"] + record["completion_ids"]], device="cuda")
            completion = ids[0, start:]
            native, upcast, head32, nucleus = [], [], [], []
            with torch.no_grad():
                hidden = backbone(input_ids=ids, use_cache=False).last_hidden_state[0, start - 1 : -1]
                for offset in range(0, hidden.shape[0], chunk):
                    target = completion[offset : offset + chunk, None]
                    logits = head(hidden[offset : offset + chunk]) / temperature  # model dtype, as the trainer
                    native.append(torch.log_softmax(logits, -1).gather(-1, target)[:, 0].float())
                    upcast.append(torch.log_softmax(logits.float(), -1).gather(-1, target)[:, 0])
                    full = torch.nn.functional.linear(hidden[offset : offset + chunk].float(), head_weight32)
                    head32.append(torch.log_softmax(full / temperature, -1).gather(-1, target)[:, 0])
                    nucleus.append(nucleus_log_softmax(logits, top_p, top_k).gather(-1, target)[:, 0])
            scored.append(
                {
                    "id": record["id"],
                    "set": record["set"],
                    "native_logps": torch.cat(native).tolist(),
                    "fp32_logps": torch.cat(upcast).tolist(),
                    "head32_logps": torch.cat(head32).tolist(),
                    "nucleus_logps": torch.cat(nucleus).tolist(),
                }
            )
        results[sampler] = scored
        print("scored", sampler, len(scored), flush=True)
    suffix = "_" + "_".join(args.only) if args.only else ""
    (Path(args.out_dir) / f"scores_{args.dtype}{suffix}.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "dtype": args.dtype,
                "scores": results,
                "hooks": {name: dict(values) for name, values in stats.items()},
                "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dtype", required=True, choices=tuple(DTYPES))
    parser.add_argument("--out-dir", default="/work")
    parser.add_argument("--only", type=lambda value: value.split(","), help="comma-separated sample tags")
    parser.add_argument("--chunk", type=int, default=128, help="completion positions per LM-head chunk")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--prompts", default="/work/prompts.json")
    parser.add_argument("--tag")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=DTYPES[args.dtype], attn_implementation="sdpa"
    )
    model = model.cuda().eval()
    print(type(model).__name__, next(model.parameters()).dtype, flush=True)
    if args.generate:
        _generate(model, args)
    else:
        _score(model, args)


if __name__ == "__main__":
    main()
