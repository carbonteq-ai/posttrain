"""Zero-update long-prefix scoring diagnostic; never a learning qualification."""

import argparse
import asyncio
import json
from pathlib import Path


async def run(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from sdpo_agent_ablation import SamplingRollout
    from sdpo_speculation import speculative_config
    from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed

    args.output.mkdir(parents=True, exist_ok=False)
    set_seed(18)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer._sdpo_model_family = "qwen3.5"
    base = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="sdpa", local_files_only=True)
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.,
        target_modules="all-linear", exclude_modules=r".*visual.*", task_type="CAUSAL_LM"))
    initial = args.output / "initial-student"
    model.save_pretrained(initial, selected_adapters=["default"])
    model.load_adapter(initial, adapter_name="sdpo_teacher", is_trainable=False)
    model.set_adapter("default")
    rollout = await SamplingRollout.create(args.model, model, tokenizer, args.output,
        65536, 40, 16, speculative_config=speculative_config("mtp", args.mtp_tokens),
        prefix_caching=False, chunked_prefill=False, enforce_eager=True)
    try:
        seed = tokenizer.encode("A local capacity probe uses neutral repeated text. ", add_special_tokens=False)
        prefix = (seed * (28672 // len(seed) + 1))[:28672]
        target = (seed * (16 // len(seed) + 1))[:16]
        parity = await rollout.score_probe(model, prefix, target)
        (args.output / "diagnostic.json").write_text(json.dumps({
            "parity": parity, "mtp_tokens": args.mtp_tokens, "optimizer_updates": 0,
            "learning_qualified": False, "scope": "Single long-prefix score comparison only."}, indent=2))
    finally:
        await rollout.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mtp-tokens", type=int, choices=(1, 2), required=True)
    parser.add_argument("--method")
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    asyncio.run(run(parser.parse_args()))
