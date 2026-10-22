"""Zero-update reserved transfer evaluation with a verified fresh-base control."""

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
from live import environment_config
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
from self_campaign import Campaign, write
from self_campaign_config import comparisons, summarize
from self_failure_diagnosis import logits
from self_feedback_vllm import LocalVLLMRollout
from self_repair_transfer_eval import matching_export
from self_reserve_config import (
    MODEL_REVISION,
    candidates,
    check_adapter_config,
    population,
    verify_loaded_state,
    zero_effect_state,
)
from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed
from verifiers.v1.cli.eval.runner import run_eval
from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers_preinstalled import configure_preinstalled_runtime


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply(model, state):
    result = set_peft_model_state_dict(model, state, adapter_name="default")
    if result.unexpected_keys:
        raise ValueError("unexpected candidate adapter keys")
    model.set_adapter("default")
    model.eval()
    verify_loaded_state(state, get_peft_model_state_dict(model, adapter_name="default"))


class ReservedEvaluation(Campaign):
    """Reuse the native HTTP interface; do not impose training-transition admission."""

    def __init__(self, args):
        self.args = args
        self.calls, self.curve, self.baselines = [], [], {}
        self.folder = None

    async def collect(self, step, names):
        self.folder = self.args.output / f"step-{step:04d}" / "reserve_transfer"
        self.folder.mkdir(parents=True, exist_ok=False)
        self.calls = []
        env = environment_config(8, "limited_zapier")
        env["agent"]["max_output_tokens"] = 12288
        env["taskset"]["task"].update(capture_actions=True, deterministic_world=True)
        config = EvalConfig.model_validate({"env": env, "model": "local-self",
            "client": {"type": "eval", "base_url": "http://127.0.0.1:18774/v1", "api_key_var": "SELF_LOCAL_KEY"},
            "sampling": {"temperature": 0., "max_tokens": 2048}, "num_tasks": len(names),
            "task_keys": names, "num_rollouts": 2, "max_concurrent": 32, "shuffle": False,
            "output_dir": self.folder / "native", "run": {"name": f"{self.args.run_name}-v{step}", "dir": "."},
            "push": False, "rich": None, "serve": None})
        episodes = await run_eval(config)
        summary = summarize(episodes, names, 2)
        write(self.folder / "outcomes.json", summary)
        if summary["errors"]:
            raise ValueError("native errors retained; no score substitution")
        if any(c["policy_version"] != self.rollout.version for c in self.calls):
            raise ValueError("stale native rollout policy")
        write(self.folder / "evaluation-admission.json", {"training_allowed": False, "optimizer_updates": 0,
              "context_rejected_calls": sum(c.get("limit_reason") == "context_limit" for c in self.calls),
              "incomplete_generations": sum(not c["complete"] for c in self.calls),
              "scope": "All native episodes retained, including failures with no tool calls; no training data admission."})
        if step == 0:
            self.baselines["reserve"] = summary
        self.curve.append({"step": step, "cohort": "reserve_transfer", "summary": summary,
                           "compared_to_fresh_base": comparisons(self.baselines["reserve"], summary)})
        write(self.args.output / "outcome-curve.json", self.curve)


async def main(args):
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    selection, reserve = read(args.selection), read(args.reserve)
    if Path(args.model).name != MODEL_REVISION:
        raise ValueError("model cache revision differs from authorized pin")
    items = candidates(selection)
    names = population(reserve, read(args.protocol), read(args.pool), read(args.heldout))
    for name in ["protocol", "pool", "heldout"]:
        if digest(getattr(args, name)) != reserve["source_sha256"][name]:
            raise ValueError("reserved cohort source file changed")
    (args.output / "selection-before-outcomes.json").write_bytes(args.selection.read_bytes())
    write(args.output / "selection-sha256.json", {"sha256": digest(args.selection)})
    write(args.output / "inputs.json", {"model": args.model, "run_name": args.run_name,
          "context_limit": 16384, "reply_limit": 2048, "episode_output_limit": 12288,
          "turn_limit": 8, "server_sequences": 32, "episode_concurrency": 32,
          "kv_cache_gib": 16, "temperature": 0., "thinking": False, "optimizer_updates": 0})
    write(args.output / "population.json", {"tasks": names, "rollouts_per_task": 2,
          "sorted_names_sha256": reserve["sorted_names_sha256"], "scope": reserve["scope"]})
    for item in items[1:]:
        path = Path(item["path"])
        if digest(path / "adapter_model.safetensors") != item["adapter_sha256"]:
            raise ValueError("selected adapter changed")
        check_adapter_config(read(path / "adapter_config.json"))
    for item in items[2:]:
        receipt = read(Path(item["receipt_path"]))
        if not receipt["completed"] or receipt["optimizer_updates"] != 10:
            raise ValueError("ablation source is incomplete")
        if Path(item["path"]).name != f"checkpoint-{item['step']}":
            raise ValueError("selected saved step/path mismatch")
    os.environ["POSTTRAIN_VERIFIERS_PREINSTALLED"] = "1"
    os.environ["SELF_LOCAL_KEY"] = "EMPTY"
    configure_preinstalled_runtime(interpreter=sys.executable)
    set_seed(17)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa", local_files_only=True)
    reference = Path(items[1]["path"])
    model = PeftModel.from_pretrained(base, reference, is_trainable=True)
    original = load_file(str(reference / "adapter_model.safetensors"))
    zero = zero_effect_state(original)
    apply(model, zero)
    probe_prefix = tokenizer.encode("A local readiness check", add_special_tokens=False)
    probe_targets = tokenizer.encode(" is ready.", add_special_tokens=False)
    with torch.no_grad():
        active = logits(model, probe_prefix, probe_targets).float()
        with model.disable_adapter():
            bare = logits(model, probe_prefix, probe_targets).float()
    torch.testing.assert_close(active, bare, atol=1e-5, rtol=1e-5)
    write(args.output / "zero-base-gate.json", {"all_B_zero": True, "only_finite_A_B": True,
          "configuration_checked": True, "actual_loaded_state_verified": True,
          "active_vs_disabled_max_abs_delta": float((active - bare).abs().max()),
          "scope": "pinned original backbone through a structurally zero-effect adapter, not checkpoint64 behavior"})
    zero_path = args.output / "zero-effect-base"
    model.save_pretrained(zero_path, selected_adapters=["default"], safe_serialization=True)
    args.context_limit, args.reply_limit, args.feedback_mode = 16384, 2048, "minimal"
    campaign = ReservedEvaluation(args)
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
        for step, item in enumerate(items):
            if time.monotonic() - started > 6800:
                raise TimeoutError("bounded total experiment time")
            path = zero_path if step == 0 else Path(item["path"])
            if step:
                apply(model, load_file(str(path / "adapter_model.safetensors")))
                await rollout.refresh()
            exported = Path(rollout.refreshes[-1]["path"])
            provenance.append({"name": item["name"], "kind": item["kind"], "checkpoint": str(path),
                               **matching_export(path, exported)})
            write(args.output / "candidate-provenance.json", provenance)
            await rollout.score_probe(model, probe_prefix, probe_targets)
            await campaign.collect(step, names)
            await rollout.suspend_for_update()
            print(json.dumps({"stage": "candidate_complete", "candidate": item["name"]}), flush=True)
        write(args.output / "receipt.json", {"completed": True, "candidates": len(items),
              "tasks_per_candidate": 24, "episodes_per_candidate": 48, "optimizer_updates": 0,
              "external_calls": 0, "elapsed_s": time.monotonic() - started,
              "primary": selection["primary"], "primary_qualified_before_reserve": selection["primary_qualified"],
              "scope": "Reserved training-pool transfer; primary comparison to fresh base predeclared. "
                       "Other arms descriptive; never select best by reserve. Not heldout62."})
    finally:
        await runner.cleanup()
        await rollout.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["selection", "reserve", "protocol", "pool", "heldout", "output"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-name", required=True)
    asyncio.run(main(parser.parse_args()))
