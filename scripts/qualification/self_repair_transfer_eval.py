"""Zero-update comparison on tasks excluded from the repair adaptation examples."""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from aiohttp import web
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file
from self_campaign import Campaign, write
from self_corrected_call_ablation import RETENTION
from self_failure_diagnosis import TASKS
from self_feedback_vllm import LocalVLLMRollout
from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed
from verifiers_preinstalled import configure_preinstalled_runtime


def transfer_tasks(protocol, heldout):
    pilot = protocol["cohorts"]["pilot"]["task_names"]
    development = set(protocol["cohorts"]["development"]["task_names"])
    excluded = set(TASKS) | set(RETENTION)
    names = sorted(set(pilot) - excluded)
    if len(pilot) != 32 or len(set(pilot)) != 32 or len(excluded) != 8 or len(names) != 24:
        raise ValueError("expected32-task pilot minus8 repair/control tasks")
    if set(names) & (development | set(heldout)):
        raise ValueError("development/heldout contamination")
    return names


def matching_export(candidate, exported):
    expected = load_file(str(candidate / "adapter_model.safetensors"))
    observed = load_file(str(exported / "adapter_model.safetensors"))
    if set(expected) != set(observed) or any(not torch.equal(expected[k], observed[k]) for k in expected):
        raise ValueError("rollout student export differs from selected candidate")
    return {"candidate_sha256": hashlib.sha256((candidate / "adapter_model.safetensors").read_bytes()).hexdigest(),
            "exported_sha256": hashlib.sha256((exported / "adapter_model.safetensors").read_bytes()).hexdigest(),
            "tensor_equality_verified": True}


async def main(args):
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "inputs.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    protocol = json.loads(args.protocol.read_text())
    heldout = json.loads(args.heldout.read_text())["task_names"]
    names = transfer_tasks(protocol, heldout)
    candidates = [("checkpoint64", args.campaign / "checkpoints/checkpoint-64"),
                  ("argument_sft", args.control / "checkpoint-10"),
                  ("argument_self_hybrid", args.hybrid / "checkpoint-10")]
    manifest = json.loads((candidates[0][1] / "campaign-checkpoint.json").read_text())
    if manifest["step"] != 64:
        raise ValueError("reference checkpoint mismatch")
    for root in [args.control, args.hybrid]:
        receipt = json.loads((root / "receipt.json").read_text())
        if not receipt["completed"] or receipt["optimizer_updates"] != 10:
            raise ValueError("comparison arm incomplete")
    write(args.output / "population.json", {"tasks": names, "rollouts_per_task": 2,
          "sorted_names_sha256": hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest(),
          "scope": "excluded from adaptive repair training; previously in original32-task SELF training cohort"})
    os.environ["POSTTRAIN_VERIFIERS_PREINSTALLED"] = "1"
    os.environ["SELF_LOCAL_KEY"] = "EMPTY"
    configure_preinstalled_runtime(interpreter=sys.executable)
    set_seed(17)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa", local_files_only=True)
    model = PeftModel.from_pretrained(base, candidates[0][1], is_trainable=True)
    model.eval()
    args.steps, args.updates_per_round, args.batch_size = 4, 4, 16
    args.context_limit, args.reply_limit, args.feedback_mode = 16384, 2048, "minimal"
    campaign = Campaign(args)
    campaign.model, campaign.tokenizer = model, tokenizer
    rollout = await LocalVLLMRollout.create(args.model, model, tokenizer, args.output, 16384, 32, 16)
    campaign.rollout = rollout
    app = web.Application(client_max_size=16 * 1024 * 1024)
    app.router.add_post("/v1/chat/completions", campaign.completions)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 18774).start()
    provenance = []
    try:
        for step, (name, candidate) in enumerate(candidates):
            if step:
                state = load_file(str(candidate / "adapter_model.safetensors"))
                result = set_peft_model_state_dict(model, state, adapter_name="default")
                if result.unexpected_keys:
                    raise ValueError("unexpected adapter keys")
                model.set_adapter("default")
                model.eval()
                await rollout.refresh()
            exported = Path(rollout.refreshes[-1]["path"])
            provenance.append({"name": name, "checkpoint": str(candidate),
                               **matching_export(candidate, exported)})
            write(args.output / "candidate-provenance.json", provenance)
            await campaign.collect(step, "repair_transfer", names)
            await rollout.suspend_for_update()
            print(json.dumps({"stage": "candidate_complete", "candidate": name}), flush=True)
        write(args.output / "receipt.json", {"completed": True, "candidates": 3, "tasks_per_candidate": 24,
              "episodes_per_candidate": 48, "optimizer_updates": 0, "external_calls": 0,
              "elapsed_s": time.monotonic() - started,
              "scope": "repair-unseen task transfer within original training pool, not heldout test generalization"})
    finally:
        await runner.cleanup()
        await rollout.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["campaign", "control", "hybrid", "protocol", "heldout", "output"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-name", default="self-adaptive03-transfer-eval")
    asyncio.run(main(parser.parse_args()))
