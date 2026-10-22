"""Independent zero-update plain capacity component; native evidence stays separate."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


async def run(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from sdpo_agent_ablation import SamplingRollout, loss_for, score_sampled, teacher_logits
    from sdpo_capacity import qualify_actor, qualify_rollout
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
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    requested = {"method": "none", "speculative_config": None, "concurrency": 40,
        "context": 65536, "reply": 4096, "prefix_caching": False,
        "chunked_prefill": False, "enforce_eager": False, "optimizer_updates": 0}
    (args.output / "requested.json").write_text(json.dumps(requested, indent=2))
    rollout = await SamplingRollout.create(args.model, model, tokenizer, args.output,
        65536, 40, 16, speculative_config=None, prefix_caching=False,
        chunked_prefill=False, enforce_eager=False)
    rollout.reply_limit = 4096
    config = SimpleNamespace(output=args.output, context_limit=32768,
                             reply_limit=4096, concurrency=40, alpha=1., topk=20)
    try:
        print(json.dumps({"stage": "capacity_rollout_start", "optimizer_updates": 0}), flush=True)
        capacity = await qualify_rollout(config, model, tokenizer, rollout)
        print(json.dumps({"stage": "capacity_backward_start", "optimizer_updates": 0}), flush=True)
        await rollout.suspend_for_update()
        actor = qualify_actor(config, model, tokenizer, capacity, teacher_logits, loss_for, score_sampled)
        receipt = {"qualified": True, "component": "plain-capacity-only-v1", "method": "none",
            "optimizer_updates": 0, "native_populations_collected": 0,
            "initial_adapter_sha256": hashlib.sha256((initial / "adapter_model.safetensors").read_bytes()).hexdigest(),
            "actor_capacity": actor, "rollout_capacity": capacity[2],
            "scope": "Capacity component only. Requires separately audited matched native populations; not learning qualification alone."}
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    finally:
        await rollout.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("none",), required=True)
    parser.add_argument("--mtp-tokens", type=int, choices=(2,), required=True)
    parser.add_argument("--disable-prefix-caching", action="store_true", required=True)
    parser.add_argument("--disable-chunked-prefill", action="store_true", required=True)
    asyncio.run(run(parser.parse_args()))
