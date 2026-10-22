"""Zero-update native diagnostic: matched base versus compact-guided teacher.

Not an SDPO learning result or a replacement for optimizer/EMA qualification.
"""

import argparse
import asyncio
import copy
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from sdpo_compact_guidance import compile_guidance, digest, instructions


def guided_messages(messages, rows, compiled):
    matches = [r for r in rows if any(m.get("role") == "user" and m.get("content") == r["material"]["user_request"]
                                     for m in messages)]
    if len(matches) != 1:
        raise ValueError("teacher diagnostic requires exact unique public-task match")
    task = matches[0]["task"]
    text = ("Teacher-only compact task constraints. Source roles and withheld values are not observations. "
            "Retrieve current records and applicable policies; resolve references from their results. "
            "Use native tools to complete the original public task; do not print the sheet or its symbols.\n"
            + instructions(compiled[task])[0])
    return [*messages, {"role": "user", "content": text}], task


async def run(args):
    import torch
    from aiohttp import web
    from peft import LoraConfig, get_peft_model
    from sdpo_agent_ablation import Agent, SamplingRollout
    from sdpo_schema_parser import parser_class
    from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed
    from verifiers_preinstalled import configure_preinstalled_runtime

    rows = json.loads(args.materials.read_text())["rows"]
    rows = [r for r in rows if r["task"] in args.tasks]
    if len(rows) != 4 or len(set(args.tasks)) != 4 or set(args.tasks) != {r["task"] for r in rows}:
        raise ValueError("exact four prespecified training tasks required")
    compiled = {r["task"]: compile_guidance(r["material"]) for r in rows}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "compiled.json").write_text(json.dumps(compiled, indent=2) + "\n")
    set_seed(18)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer._sdpo_model_family = "qwen3.5"
    model = get_peft_model(AutoModelForImageTextToText.from_pretrained(args.model,
        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa", local_files_only=True),
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0., target_modules="all-linear",
                   exclude_modules=r".*visual.*", task_type="CAUSAL_LM"))
    rollout = await SamplingRollout.create(args.model, model, tokenizer, args.output,
        65536, 40, 16, speculative_config=None, prefix_caching=False, chunked_prefill=False)
    if rollout.refreshes[-1]["weights_sha256"] != "e71dae81f1b4723c83aacd130bfbf60b072b3580212133a4fc45dcb9f403421c":
        raise ValueError("teacher qualification initial adapter differs from shared base")
    os.environ.update(POSTTRAIN_VERIFIERS_PREINSTALLED="1", SELF_LOCAL_KEY="EMPTY")
    configure_preinstalled_runtime(interpreter=sys.executable)

    class DiagnosticAgent(Agent):
        guided = False

        async def completions(self, request):
            body = await request.json()
            if self.guided:
                original = body["messages"]
                body["messages"], name = guided_messages(original, rows, compiled)
                with (args.output / "teacher-contexts.jsonl").open("a") as stream:
                    stream.write(json.dumps({"task": name, "original_messages": original,
                        "teacher_messages": body["messages"], "sheet_sha256": digest(compiled[name]["sheet"]),
                        "policy_version": rollout.version, "context_scope": "guided teacher diagnostic only"}) + "\n")

            class Request:
                async def json(self):
                    return body

            return await super().completions(Request())

    agent = DiagnosticAgent(SimpleNamespace(output=args.output, context_limit=32768, reply_limit=4096,
        thinking=True, feedback_mode="minimal", model_family="qwen3.5", tool_parser_mode="schema-bound-v1",
        run_profile="guidance-five", concurrency=40, run_name=args.output.parent.name + "-compact-four",
        teacher_feedback="native-feedback", online_rollouts=2, native_repeatability=True))
    agent.rollout, agent.tokenizer, agent.model = rollout, tokenizer, model
    agent.schema_parser, _ = parser_class()
    app = web.Application(client_max_size=16 * 1024 * 1024)
    app.router.add_post("/v1/chat/completions", agent.completions)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", 18774).start()
        results = {}
        population_ids = []
        for mode in ("base", "compact_teacher"):
            agent.guided = mode == "compact_teacher"
            (args.output / "progress.json").write_text(json.dumps({"stage": mode, "optimizer_updates": 0}) + "\n")
            await agent.collect(0, mode, args.tasks, temperature=0.)
            from automationbench_v1.world_codec import load_world

            retained = [json.loads(line) for line in (agent.folder / "native/traces.jsonl").read_text().splitlines()]
            if Counter(e["task"]["data"]["name"] for e in retained) != Counter({name: 2 for name in args.tasks}):
                raise ValueError("native task/replica population mismatch")
            for row in rows:
                initial = copy.deepcopy(row["material"]["native_data"]["initial_state"])
                load_world(initial)
                for episode in retained:
                    if episode["task"]["data"]["name"] == row["task"] and episode["task"]["data"]["initial_state"] != initial:
                        raise ValueError("native initial state differs from compiled source")
            population_ids.extend(e["id"] for e in retained)
            results[mode] = json.loads((agent.folder / "outcomes.json").read_text())
        if len(population_ids) != len(set(population_ids)) or len(population_ids) != 16:
            raise ValueError("native episode identities must be16unique")
        (args.output / "receipt.json").write_text(json.dumps({
            "runtime_qualified": True, "semantic_qualified": None, "optimizer_updates": 0,
            "native_episodes": 16, "tasks": args.tasks, "results": results,
            "episode_ids": population_ids,
            "model_revision": Path(args.model).name, "initial_adapter": rollout.refreshes[-1],
            "compiler_sha256": hashlib.sha256(Path(__file__).with_name("sdpo_compact_guidance.py").read_bytes()).hexdigest(),
            "scope": "8fresh base and8compact-guided teacher native episodes; no learning/gradient or heldout claim. "
                     "Independent semantic/source/action audit still required before training."}, indent=2) + "\n")
    finally:
        await runner.cleanup()
        await rollout.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--materials", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
