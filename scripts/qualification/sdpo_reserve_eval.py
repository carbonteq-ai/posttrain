"""Zero-update comparison of a preselected SDPO adapter against fresh base."""

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
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from peft.utils.save_and_load import load_peft_weights
from sdpo_agent_ablation import Agent, SamplingRollout
from sdpo_agent_config import MODEL_REVISION, fingerprint
from sdpo_execution_budget import boot_seconds, install_deadline_timer, reserve_profile
from self_campaign import write
from self_failure_diagnosis import logits
from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed
from verifiers_preinstalled import configure_preinstalled_runtime


async def main(args):
    started = time.monotonic()
    budget = json.loads(args.shared_budget.read_text())
    install_deadline_timer(budget)
    selection = json.loads(args.primary.read_text())
    cohorts = json.loads(args.cohorts.read_text())
    heldout = set(json.loads(args.heldout.read_text())["task_names"])
    groups = [set(cohorts[key]) for key in ("train", "development", "reserve")]
    # Transport is part of the preselected operating point. A parser repair must
    # apply identically to fresh base and candidate, and cannot be picked after
    # reserve results. No selection exists yet; require an explicit declaration.
    args.tool_parser_mode = selection["tool_parser_mode"]
    profile = reserve_profile(selection)
    args.context_limit, args.reply_limit, args.thinking = (
        profile["context_limit"], profile["reply_limit"], profile["thinking"])
    args.feedback_mode, args.teacher_feedback = "minimal", "native-feedback"
    if args.tool_parser_mode not in ("native", "schema-bound-v1"):
        raise ValueError("unqualified preselected reserve parser")
    weights = args.candidate / "adapter_model.safetensors"
    config = json.loads((args.candidate / "adapter_config.json").read_text())
    if (Path(args.model).name != MODEL_REVISION or selection["model_revision"] != MODEL_REVISION
            or not selection["frozen_before_reserve"] or not selection["approved_for_reserve"]
            or selection["adapter_sha256"] != hashlib.sha256(weights.read_bytes()).hexdigest()
            or selection["reserve_sha256"] != fingerprint(cohorts["reserve"])
            or len(cohorts["reserve"]) != 24 or any(group & heldout for group in groups)
            or any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3))
            or MODEL_REVISION not in config.get("base_model_name_or_path", "")
            or config["r"] != 16 or config["lora_alpha"] != 32 or config["lora_dropout"] != 0):
        raise ValueError("reserve preselection/model/adapter/cohort authority mismatch")
    expected = load_peft_weights(args.candidate, device="cpu")
    if (not all(torch.isfinite(value).all() for value in expected.values())
            or not any(value.any() for key, value in expected.items() if "lora_B" in key)):
        raise ValueError("candidate has nonfinite or unchanged zero-effect adapter")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "primary-selection.json", selection)
    write(args.output / "shared-budget-snapshot.json", budget)
    write(args.output / "inputs.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    os.environ.update(POSTTRAIN_VERIFIERS_PREINSTALLED="1", SELF_LOCAL_KEY="EMPTY")
    configure_preinstalled_runtime(interpreter=sys.executable)
    set_seed(18)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa", local_files_only=True)
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.,
        target_modules="all-linear", exclude_modules=r".*visual.*", task_type="CAUSAL_LM"))
    initial = get_peft_model_state_dict(model, adapter_name="default")
    if set(expected) != set(initial) or any(expected[key].shape != initial[key].shape for key in expected):
        raise ValueError("candidate adapter keys/shapes differ from qualified model")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_parameters:
        raise ValueError("zero-update reserve comparison must freeze every parameter")
    probe = tokenizer.encode("Local readiness", add_special_tokens=False)
    target = tokenizer.encode(" verified.", add_special_tokens=False)
    with torch.no_grad():
        active = logits(model, probe, target)
        with model.disable_adapter():
            bare = logits(model, probe, target)
        torch.testing.assert_close(active, bare, atol=1e-5, rtol=1e-5)
    write(args.output / "fresh-base-gate.json", {"max_abs_delta": float((active - bare).abs().max())})
    agent = Agent(args)
    agent.model, agent.tokenizer = model, tokenizer
    if args.tool_parser_mode == "schema-bound-v1":
        from sdpo_schema_parser import parser_class

        agent.schema_parser, parser_receipt = parser_class()
        write(args.output / "tool-parser-receipt.json", parser_receipt)
    rollout = await SamplingRollout.create(args.model, model, tokenizer, args.output, args.context_limit, 32, 16)
    rollout.reply_limit = args.reply_limit
    agent.rollout = rollout
    app = web.Application(client_max_size=16 * 1024 * 1024)
    app.router.add_post("/v1/chat/completions", agent.completions)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 18774).start()
    try:
        await rollout.score_probe(model, probe, target)
        await rollout.batch_probe(probe, maximum=32)
        await agent.collect(0, "reserve", cohorts["reserve"])
        await rollout.suspend_for_update()  # serving lifecycle only; no optimizer exists
        set_peft_model_state_dict(model, expected, adapter_name="default")
        actual = get_peft_model_state_dict(model, adapter_name="default")
        for key, value in expected.items():
            torch.testing.assert_close(actual[key].cpu(), value, atol=0, rtol=0, check_dtype=False)
        write(args.output / "candidate-load.json", {"all_adapter_tensors_exact": True,
              "source_weights_sha256": selection["adapter_sha256"], "optimizer_updates": 0})
        await rollout.refresh()
        await rollout.score_probe(model, probe, target)
        await agent.collect(1, "reserve", cohorts["reserve"])
        write(args.output / "receipt.json", {"completed": True, "optimizer_updates": 0,
              "elapsed_s": time.monotonic() - started, "external_calls": 0,
              "trainable_parameters": trainable_parameters,
              "reserve_evaluated": True, "heldout62_evaluated": False,
              "expected_native_episodes": 96, "rollout_versions": rollout.refreshes,
              "matched_profile": profile,
              "finished_boot_seconds": boot_seconds(),
              "shared_deadline_boot_seconds": budget["deadline_boot_seconds"],
              "scope": "Frozen primary against matched fresh base on research reserve; no selection by reserve"})
    finally:
        await runner.cleanup()
        await rollout.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohorts", "heldout", "output", "primary", "candidate", "shared-budget"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-name", required=True)
    asyncio.run(main(parser.parse_args()))
