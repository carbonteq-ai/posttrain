"""Adapter effect check: the adapter must change the model's log-probs on a fixed prompt, loaded the runner's way.

Why: the evaluation runner (wt-base3 fd207334 sdpo_ab_runner.py --adapter) builds the base model plus a fresh LoRA
(r16/a32, dropout 0, all-linear, exclude .*visual.*) and loads the adapter with set_peft_model_state_dict, which
silently ignores keys that do not match. A mismatched or zero adapter then evaluates the untrained base model. This
check loads the adapter on that same path and fails when:
  * any adapter key is unexpected (did not match a module) or any LoRA parameter of the model got no value;
  * any loaded lora_B tensor is all zero (an untrained adapter is the base model) or any tensor is non-finite;
  * the per-token log-probs of a fixed prompt are identical with and without the adapter (max |delta| <= --tolerance).

Intended for a GPU staging job (after training, before any evaluation reading); unit-tested on CPU with a tiny random
model (test_adapter_logprob_check.py).

  python adapter_logprob_check.py --model BASE_DIR --adapter ADAPTER_DIR [--model-class image-text|causal-lm]
         [--lora-rank 16 --lora-alpha 32] [--device cuda] [--dtype bfloat16] [--tokenizer DIR] [--out result.json]
Exit 0 only if every condition holds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FIXED_MESSAGES = [
    {"role": "system", "content": "You are a workflow automation agent. Execute the requested tasks using the "
                                  "available tools."},
    {"role": "user", "content": "Email the January variance summary to controller@company.example.com."},
    {"role": "assistant", "content": "I need the variance workbook and the thresholds first, so I will look for the "
                                     "workbook and read the inbox together."},
]
FIXED_TEXT = ("user: Email the January variance summary to controller@company.example.com.\nassistant: I need the "
              "variance workbook and the thresholds first.")


def runner_lora_config(rank: int, alpha: int):
    from peft import LoraConfig

    return LoraConfig(task_type="CAUSAL_LM", r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules="all-linear",
                      exclude_modules=r".*visual.*")


def load_base(model_dir: str, model_class: str, dtype, device: str):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    cls = AutoModelForImageTextToText if model_class == "image-text" else AutoModelForCausalLM
    return cls.from_pretrained(model_dir, dtype=dtype).to(device).eval()


def attach_adapter(base, adapter_dir: str, rank: int, alpha: int) -> tuple[object, dict]:
    """The runner's path: fresh LoRA on the base, then set_peft_model_state_dict(load_peft_weights(adapter))."""
    import torch
    from peft import get_peft_model, get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict

    model = get_peft_model(base, runner_lora_config(rank, alpha))
    weights = load_peft_weights(adapter_dir, device="cpu")
    # the model's own LoRA state-dict names, in the adapter file's naming (peft strips the adapter name)
    expected = set(get_peft_model_state_dict(model, adapter_name="default"))
    set_peft_model_state_dict(model, weights, adapter_name="default")  # strict=False, exactly as the runner
    unexpected = sorted(set(weights) - expected)
    missing = sorted(expected - set(weights))
    zero_b = sorted(k for k, v in weights.items() if "lora_B" in k and not torch.count_nonzero(v))
    nonfinite = sorted(k for k, v in weights.items() if not torch.isfinite(v.float()).all())
    model.set_adapter("default")
    model.eval()
    return model, {"adapter_tensors": len(weights), "model_lora_tensors": len(expected),
                   "unexpected_keys": unexpected[:20], "unexpected_count": len(unexpected),
                   "missing_keys": missing[:20], "missing_count": len(missing), "zero_lora_B": zero_b[:20],
                   "zero_lora_B_count": len(zero_b), "nonfinite": nonfinite[:20]}


def token_logprobs(model, ids: list[int], device: str) -> list[float]:
    """log p(token_t | tokens_<t) for t = 1..n-1 (teacher forcing over the whole fixed sequence)."""
    import torch

    with torch.no_grad():
        input_ids = torch.tensor([ids], device=device)
        logits = model(input_ids=input_ids).logits[0, :-1].float()
        return logits.log_softmax(-1).gather(-1, input_ids[0, 1:, None]).flatten().tolist()


def compare(model, ids: list[int], device: str, tolerance: float) -> dict:
    with_adapter = token_logprobs(model, ids, device)
    with model.disable_adapter():
        base = token_logprobs(model, ids, device)
    deltas = [abs(a - b) for a, b in zip(with_adapter, base)]
    return {"tokens": len(ids), "max_abs_delta": max(deltas), "mean_abs_delta": sum(deltas) / len(deltas),
            "sum_logprob_base": sum(base), "sum_logprob_adapter": sum(with_adapter),
            "identical": max(deltas) <= tolerance}


def fixed_ids(tokenizer) -> list[int]:
    if getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(FIXED_MESSAGES, tokenize=True, enable_thinking=False,
                                                 return_dict=True)
        return list(rendered["input_ids"])
    return list(tokenizer(FIXED_TEXT, add_special_tokens=False)["input_ids"])


def check(model, ids: list[int], load_report: dict, device: str, tolerance: float) -> dict:
    effect = compare(model, ids, device, tolerance)
    reasons = []
    if load_report["unexpected_count"]:
        reasons.append(f"{load_report['unexpected_count']} adapter keys matched no module (silently ignored)")
    if load_report["missing_count"]:
        reasons.append(f"{load_report['missing_count']} LoRA parameters of the model got no adapter value")
    if load_report["zero_lora_B_count"]:
        reasons.append(f"{load_report['zero_lora_B_count']} lora_B tensors are all zero")
    if load_report["nonfinite"]:
        reasons.append("non-finite adapter tensors")
    if effect["identical"]:
        reasons.append(f"log-probs identical with and without the adapter (max |delta| {effect['max_abs_delta']:.3g})")
    return {"passed": not reasons, "reasons": reasons, "load": load_report, "effect": effect}


def main(argv=None) -> int:
    import torch
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--model-class", choices=("image-text", "causal-lm"), default="image-text")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--tolerance", type=float, default=0.0, help="max |delta| treated as identical")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    base = load_base(args.model, args.model_class, getattr(torch, args.dtype), args.device)
    model, report = attach_adapter(base, args.adapter, args.lora_rank, args.lora_alpha)
    model.to(args.device)
    result = check(model, fixed_ids(tokenizer), report, args.device, args.tolerance)
    result.update(model=args.model, adapter=args.adapter, model_class=args.model_class, rank=args.lora_rank,
                  alpha=args.lora_alpha, dtype=args.dtype)
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
