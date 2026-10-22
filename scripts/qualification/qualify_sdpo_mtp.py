"""Zero-update MTP/LoRA/token-score/native-tool gate on the pinned workstation."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


async def run(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from sdpo_agent_ablation import Agent, SamplingRollout
    from sdpo_native_tool_gate import qualify
    from sdpo_schema_parser import parser_class
    from sdpo_speculation import speculative_config
    from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed

    args.output.mkdir(parents=True, exist_ok=False)
    context = 65536 if args.stochastic_cohorts or args.native_prefixes or args.native_mixed_fixture else 32768
    reply = 8192 if args.native_mixed_fixture else 4096
    set_seed(18)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer._sdpo_model_family = "qwen3.5"
    base = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="sdpa", local_files_only=True)
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.,
        target_modules="all-linear", exclude_modules=r".*visual.*", task_type="CAUSAL_LM"))
    if args.capacity_reference:
        initial = args.output / "initial-student"
        model.save_pretrained(initial, selected_adapters=["default"])
        model.load_adapter(initial, adapter_name="sdpo_teacher", is_trainable=False)
        model.set_adapter("default")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    config = speculative_config(args.method, args.mtp_tokens)
    (args.output / "requested.json").write_text(json.dumps({
        "method": args.method, "speculative_config": config, "concurrency": 40,
        "context": context, "reply": reply,
        "prefix_caching": not args.disable_prefix_caching,
        "chunked_prefill": not args.disable_chunked_prefill,
        "enforce_eager": args.enforce_eager,
        "stochastic_native": bool(args.stochastic_cohorts), "optimizer_updates": 0}, indent=2))
    rollout = await SamplingRollout.create(args.model, model, tokenizer, args.output,
        context, 40, 16,
        speculative_config=config, prefix_caching=not args.disable_prefix_caching,
        chunked_prefill=not args.disable_chunked_prefill, enforce_eager=args.enforce_eager)
    rollout.reply_limit = reply
    runner = None
    try:
        prefix = tokenizer.encode("Local readiness", add_special_tokens=False)
        target = tokenizer.encode(" verified.", add_special_tokens=False)
        parity = await rollout.score_probe(model, prefix, target)
        batch = await rollout.batch_probe(prefix, maximum=40)
        mixed = None
        if args.native_mixed_fixture:
            from sdpo_native_mixed_gate import MixedPrefixMismatch
            from sdpo_native_mixed_gate import qualify as qualify_mixed

            path = args.output / "native-mixed-qualification.json"
            try:
                mixed = await qualify_mixed(rollout, json.loads(args.native_mixed_fixture.read_text()))
            except MixedPrefixMismatch as error:
                path.write_text(json.dumps(error.receipt, indent=2))
                raise
            path.write_text(json.dumps(mixed, indent=2))
        native_prefix_gate = None
        if args.native_prefixes:
            from sdpo_native_prefix_gate import NativePrefixMismatch
            from sdpo_native_prefix_gate import qualify as qualify_native_prefixes

            path = args.output / "native-prefix-qualification.json"
            try:
                native_prefix_gate = await qualify_native_prefixes(
                    rollout, json.loads(args.native_prefixes.read_text()))
            except NativePrefixMismatch as error:
                path.write_text(json.dumps(error.receipt, indent=2))
                raise
            path.write_text(json.dumps(native_prefix_gate, indent=2))
        agent = Agent(SimpleNamespace(output=args.output, reply_limit=4096, context_limit=32768,
            thinking=True, feedback_mode="minimal", model_family="qwen3.5", tool_parser_mode="schema-bound-v1"))
        agent.rollout, agent.tokenizer, agent.model = rollout, tokenizer, model
        agent.schema_parser, _ = parser_class()
        native = await qualify(agent)
        stochastic = None
        if args.stochastic_cohorts:
            from aiohttp import web
            from verifiers_preinstalled import configure_preinstalled_runtime

            os.environ.update(POSTTRAIN_VERIFIERS_PREINSTALLED="1", SELF_LOCAL_KEY="EMPTY")
            configure_preinstalled_runtime(interpreter=sys.executable)
            agent.args.run_profile = "guidance-five"
            agent.args.concurrency = 40
            agent.args.run_name = args.run_name
            agent.args.teacher_feedback = "native-feedback"
            agent.args.online_rollouts = 2
            names = json.loads(args.stochastic_cohorts.read_text())["train"]
            if len(set(names)) != 20:
                raise ValueError("stochastic qualification requires the frozen20tasks")
            app = web.Application(client_max_size=16 * 1024 * 1024)
            app.router.add_post("/v1/chat/completions", agent.completions)
            runner = web.AppRunner(app)
            await runner.setup()
            await web.TCPSite(runner, "127.0.0.1", 18774).start()
            greedy = None
            if args.native_greedy:
                from sdpo_greedy_repeatability import audit

                _, greedy_calls = await agent.collect(0, "train", names, temperature=0.)
                greedy = audit(greedy_calls)
                (args.output / "native-greedy-repeatability.json").write_text(json.dumps(greedy, indent=2))
                if not greedy["qualified"]:
                    raise ValueError("actual native greedy repeatability failed; preserve zero-update evidence")
            episodes, calls = await agent.collect(0, "online_train", names, temperature=1.)
            stochastic = {"qualified": True, "episodes": len(episodes), "calls": len(calls),
                "temperature": 1., "tasks": names, "optimizer_updates": 0,
                "native_greedy_repeatability": greedy}
        actor = None
        if args.capacity_reference:
            from sdpo_agent_ablation import loss_for, score_sampled, teacher_logits
            from sdpo_capacity import qualify_actor, qualify_rollout

            reference = json.loads(args.capacity_reference.read_text())
            if (reference["context_limit"] != 32768 or reference["reply_limit"] != 4096
                    or not reference["batch_invariance"]["exact_batch_invariance"]):
                raise ValueError("capacity reference differs from the selected profile")
            seed = tokenizer.encode("A local capacity probe uses neutral repeated text. ", add_special_tokens=False)
            prefix = (seed * (28672 // len(seed) + 1))[:28672]
            target = (seed * (4096 // len(seed) + 1))[:4096]
            agent.args.alpha, agent.args.topk = 1., 20
            if args.disable_prefix_caching:
                agent.args.context_limit, agent.args.reply_limit = 32768, 4096
                rollout.temperature = 0.
                prefix, target, reference = await qualify_rollout(agent.args, model, tokenizer, rollout)
            await rollout.suspend_for_update()
            actor = qualify_actor(agent.args, model, tokenizer, (prefix, target, reference),
                                  teacher_logits, loss_for, score_sampled)
        receipt = {"qualified": True, "optimizer_updates": 0, "method": args.method,
            "parity": parity, "batch_invariance": batch, "native": native,
            "stochastic_native": stochastic, "actor_capacity": actor,
            "native_prefixes": native_prefix_gate, "prefix_caching": not args.disable_prefix_caching,
            "native_mixed": mixed, "chunked_prefill": not args.disable_chunked_prefill,
            "enforce_eager": args.enforce_eager,
            "scope": "Initial LoRA serving plus explicit stochastic/native and actor-capacity fields when requested. "
                     "Actual changed-adapter qualification, learning and comparative throughput remain separate."}
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    finally:
        from sdpo_runtime_diagnostics import engine_status

        (args.output / "engine-process-status.json").write_text(json.dumps(engine_status(rollout), indent=2))
        if runner is not None:
            await runner.cleanup()
        await rollout.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("none", "mtp"), required=True)
    parser.add_argument("--mtp-tokens", type=int, choices=(1, 2), default=2)
    parser.add_argument("--stochastic-cohorts", type=Path)
    parser.add_argument("--capacity-reference", type=Path)
    parser.add_argument("--run-name", default="mtp-gate")
    parser.add_argument("--native-prefixes", type=Path)
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--native-greedy", action="store_true")
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--native-mixed-fixture", type=Path)
    parser.add_argument("--enforce-eager", action="store_true")
    asyncio.run(run(parser.parse_args()))
