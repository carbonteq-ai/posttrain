"""Collect fresh native AutomationBench transitions with a local Transformers policy."""

from __future__ import annotations

import argparse
import asyncio
import copy
import faulthandler
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from aiohttp import web
from datasets import Dataset
from live import environment_config
from peft import LoraConfig, get_peft_model
from self_feedback_probe import UpdateAudit
from self_pilot_config import admit_sequences, resolve_pilot_tasks, sample_transitions
from transformers import AutoModelForImageTextToText, AutoTokenizer
from trl import SFTConfig
from trl.experimental.self_feedback import SELFTrainer
from verifiers_preinstalled import configure_preinstalled_runtime
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser


def prompt_ids(tokenizer, messages, tools):
    # OpenAI transports arguments as JSON strings; the native Qwen XML
    # tokenizer template requires mappings. Preserve the actual values.
    messages = copy.deepcopy(messages)
    for message in messages:
        for call in message.get("tool_calls", []):
            function = call["function"]
            if isinstance(function["arguments"], str):
                function["arguments"] = json.loads(function["arguments"])
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )
    return encoded["input_ids"][0].tolist()


def collect_transitions(calls, episodes, tokenizer, revision, excluded=None):
    tools_by_id, episode_by_tool, terminal_unexecuted = {}, {}, set()
    for episode in episodes:
        value = episode.model_dump(mode="json")
        for trace in value["traces"]:
            if trace.get("errors"):
                raise ValueError("native rollout has errors; inspect retained traces before training")
            for node in trace["nodes"]:
                msg = node.get("message") or {}
                if msg.get("role") == "tool":
                    tools_by_id[msg["tool_call_id"]] = msg
                    episode_by_tool[msg["tool_call_id"]] = value["id"]
            if trace.get("stop_condition") == "max_turns":
                last = trace["nodes"][-1]["message"]
                if last.get("role") == "assistant":
                    terminal_unexecuted.update(call["id"] for call in last.get("tool_calls", []))
    rows = []
    for call in calls:
        if not call["complete"] and excluded is not None:
            excluded.append({"reason": call.get("limit_reason", "incomplete_action"),
                             "generation_index": len(rows), "action_tokens": len(call["action_ids"]),
                             "prompt_tokens": len(call["prefix_ids"])})
            continue
        expected = [tool["id"] for tool in call["message"].get("tool_calls", [])]
        if not expected:
            continue
        if not call["complete"]:
            raise ValueError("truncated or unpaired tool action retained; refuse to train incomplete collection")
        if all(key in terminal_unexecuted and key not in tools_by_id for key in expected):
            continue  # Native turn limit prevented execution; there is no observed transition.
        if any(key not in tools_by_id for key in expected):
            raise ValueError("executed tool action lacks its observed response")
        # Keep transport UUIDs in native evidence, not the prediction target:
        # they are random identifiers, not consequences the policy can infer.
        feedback = json.dumps(
            [{"tool": tools_by_id[key].get("name"), "response": tools_by_id[key]["content"]} for key in expected],
            sort_keys=True,
        )
        from self_feedback_signals import teacher_message

        teacher_messages = call["messages"] + [
            {
                "role": "user",
                "content": teacher_message(call, feedback, call.get("teacher_goals")),
            }
        ]
        feedback_messages = call["messages"] + [
            call["message"],
            {"role": "user", "content": "Predict the next tool response as JSON."},
        ]
        rows.append(
            {
                "trajectory_id": episode_by_tool[expected[0]],
                "policy_revision": revision,
                "student_prefix_ids": call["prefix_ids"],
                "action_ids": call["action_ids"],
                "teacher_prefix_ids": prompt_ids(tokenizer, teacher_messages, call["tools"]),
                "feedback_prefix_ids": prompt_ids(tokenizer, feedback_messages, call["tools"]),
                "feedback_ids": tokenizer.encode(feedback, add_special_tokens=False),
            }
        )
    return rows, sorted(terminal_unexecuted - tools_by_id.keys())


async def main_async(args):
    from verifiers.v1.cli.eval.runner import run_eval
    from verifiers.v1.configs.cli.eval import EvalConfig

    args.output.mkdir(parents=True, exist_ok=False)
    heldout = set(json.loads(args.heldout.read_text())["task_names"])
    if set(args.task) & heldout:
        raise ValueError("heldout62 tasks cannot be used by the training qualifier")
    (args.output / "run-inputs.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2)
    )
    os.environ["POSTTRAIN_VERIFIERS_PREINSTALLED"] = "1"
    configure_preinstalled_runtime(interpreter=sys.executable)
    torch.manual_seed(17)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa", local_files_only=True
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules="all-linear",
            exclude_modules=r".*visual.*",
            task_type="CAUSAL_LM",
        ),
    )
    rollout = None
    if args.rollout_backend == "vllm":
        from self_feedback_vllm import LocalVLLMRollout

        if os.environ.get("VLLM_BATCH_INVARIANT") != "1":
            raise ValueError("optimized qualification requires VLLM_BATCH_INVARIANT=1 before imports")
        rollout = await LocalVLLMRollout.create(
            args.model, model, tokenizer, args.output, args.context_limit, args.server_sequences, args.kv_cache_gib
        )
    if args.qualify_context_only:
        from self_context_qualification import qualify_context

        if rollout is None:
            raise ValueError("context qualification requires the actual vLLM rollout path")
        await qualify_context(args, model, tokenizer, rollout)
        return
    calls = []
    round_output = args.output

    async def completions(request):
        body = await request.json()
        print(json.dumps({"stage": "request_received", "backend": args.rollout_backend}), flush=True)
        messages, tools = body["messages"], body.get("tools", [])
        prefix = prompt_ids(tokenizer, messages, tools)
        print(json.dumps({"stage": "prompt_encoded", "tokens": len(prefix)}), flush=True)
        with (round_output / "request-starts.jsonl").open("a") as stream:
            stream.write(json.dumps({"messages": messages, "tools": tools, "prefix_ids": prefix}) + "\n")
        engine_evidence = {}
        if len(prefix) + 512 > args.context_limit:
            # Preserve the full request. End this episode with a typed length
            # completion instead of retryable HTTP500; execute no partial action.
            action_ids, complete = [], False
            engine_evidence = {"limit_reason": "context_limit", "policy_version": rollout.version if rollout else 0}
        elif rollout is not None:
            action_ids, complete, engine_evidence = await rollout.generate(prefix)
            print(json.dumps({"stage": "generation_finished", "tokens": len(action_ids)}), flush=True)
        else:
            model.eval()
            with torch.no_grad():
                ids = torch.tensor([prefix], device="cuda")
                generated = model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=512,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            action_ids = generated[0, len(prefix) :].tolist()
            complete = action_ids[-1] == tokenizer.eos_token_id
        raw = tokenizer.decode(action_ids, skip_special_tokens=True)
        native_request = ChatCompletionRequest(model="local-self", messages=messages, tools=tools)
        parser = Qwen3EngineToolParser(tokenizer, tools=native_request.tools)
        parsed = parser.extract_tool_calls(raw, native_request) if complete else None
        tool_calls = [call.model_dump() for call in parsed.tool_calls] if parsed else []
        message = {"role": "assistant", "content": parsed.content if parsed else raw}
        if tool_calls:
            message["tool_calls"] = tool_calls
        record = {
            "messages": messages,
            "tools": tools,
            "prefix_ids": prefix,
            "action_ids": action_ids,
            "raw": raw,
            "message": message,
            "complete": complete,
            **engine_evidence,
        }
        calls.append(record)
        with (round_output / "generation-calls.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        return web.json_response(
            {
                "id": f"self-{len(calls)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "local-self",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "length" if not complete else "tool_calls" if tool_calls else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prefix),
                    "completion_tokens": len(action_ids),
                    "total_tokens": len(prefix) + len(action_ids),
                },
            }
        )

    app = web.Application(client_max_size=16 * 1024 * 1024)
    app.router.add_post("/v1/chat/completions", completions)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 18774).start()
    os.environ["SELF_LOCAL_KEY"] = "EMPTY"
    env_config = environment_config(8, "limited_zapier")
    env_config["agent"]["max_output_tokens"] = args.episode_output_tokens
    env_config["taskset"]["task"].update(capture_actions=True, deterministic_world=True)
    config = EvalConfig.model_validate(
        {
            "env": env_config,
            "model": "local-self",
            "client": {"type": "eval", "base_url": "http://127.0.0.1:18774/v1", "api_key_var": "SELF_LOCAL_KEY"},
            "sampling": {"temperature": 0.0, "max_tokens": 512},
            "num_tasks": len(args.task),
            "task_keys": args.task,
            "num_rollouts": args.rollouts,
            "max_concurrent": args.episode_concurrency,
            "shuffle": False,
            "output_dir": args.output / "native",
            "run": {"name": args.run_name, "dir": "."},
            "push": False,
            "rich": None,
            "serve": None,
        }
    )
    try:
        episodes = await run_eval(config)
    except BaseException:
        await runner.cleanup()
        raise
    excluded = []
    rows, terminal_unexecuted_ids = collect_transitions(calls, episodes, tokenizer, "round0", excluded)
    (args.output / "all-complete-transitions.json").write_text(json.dumps(rows))
    rows, overlong = admit_sequences(rows, args.context_limit)
    (args.output / "training-admission.json").write_text(json.dumps({"incomplete_generations": excluded,
        "overlong_transitions": overlong, "admitted": len(rows), "truncated_training_sequences": 0}, indent=2))
    if not rows:
        raise ValueError("native collection produced no executed transitions")
    probe_prefix, probe_targets = rows[0]["student_prefix_ids"], rows[0]["action_ids"][:16]
    before = None
    if rollout is not None:
        before = await rollout.score_probe(model, probe_prefix, probe_targets)
        await rollout.batch_probe(probe_prefix)
        await rollout.suspend_for_update()
    groups = defaultdict(list)
    for row in rows:
        groups[row["trajectory_id"]].append(row)
    sampled = sample_transitions(rows, args.steps * 2, seed=17)
    (args.output / "transitions.json").write_text(json.dumps(rows))
    trainer = SELFTrainer(
        sequence_limit=args.context_limit,
        profile_scoring=True,
        model=model,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(sampled),
        data_collator=lambda x: {"transitions": x},
        args=SFTConfig(
            output_dir=str(args.output / "checkpoint"),
            max_steps=args.steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=args.learning_rate,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=None,
            report_to="none",
            logging_steps=1,
            save_strategy="no",
        ),
    )
    audit = UpdateAudit(trainer)
    trainer.add_callback(audit)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = trainer.train()
    trainer.save_model(str(args.output / "student-and-teacher"))
    training_wall = time.perf_counter() - started
    refreshed = None
    if rollout is not None:
        assert before is not None
        await rollout.refresh()
        after = await rollout.score_probe(model, probe_prefix, probe_targets)
        movement = max(abs(a - b) for a, b in zip(before["rollout"], after["rollout"], strict=True))
        if movement < 0.01:
            raise ValueError("updated rollout log probabilities moved less than 0.01 nats")
        await rollout.batch_probe(probe_prefix)
        round_output = args.output / "round1"
        round_output.mkdir()
        calls.clear()
        next_config = config.model_copy(update={"output_dir": round_output / "native"})
        next_episodes = await run_eval(next_config)
        next_excluded = []
        next_rows, next_unexecuted = collect_transitions(calls, next_episodes, tokenizer, "round1", next_excluded)
        (round_output / "incomplete-generations.json").write_text(json.dumps(next_excluded, indent=2))
        if any(call["policy_version"] != 1 for call in calls):
            raise ValueError("second collection used a stale policy")
        (round_output / "transitions.json").write_text(json.dumps(next_rows))
        refreshed = {
            "second_collection_terminal_unexecuted_tool_ids": next_unexecuted,
            "second_collection_episodes": len(next_episodes),
            "second_collection_transitions": len(next_rows),
            "selected_token_logprob_movement": movement,
            "refreshes": rollout.refreshes,
            "before_max_actor_rollout_delta": before["max_abs_delta"],
            "after_max_actor_rollout_delta": after["max_abs_delta"],
        }
    await runner.cleanup()
    if rollout is not None:
        await rollout.aclose()
    receipt = {
        "purpose": "bounded local SELF pilot; fresh collection, updates and post-update training-task diagnostics",
        "task_count": len(args.task),
        "rollouts_per_task": args.rollouts,
        "context_limit": args.context_limit,
        "episode_concurrency": args.episode_concurrency,
        "server_sequences": args.server_sequences,
        "kv_cache_gib": args.kv_cache_gib,
        "episode_output_tokens": args.episode_output_tokens,
        "training_overlong_exclusions": len(overlong),
        "incomplete_generation_exclusions": len(excluded),
        "truncated_training_sequences": 0,
        "learning_rate": args.learning_rate,
        "sampled_transition_exposures": len(sampled),
        "sampled_trajectory_count": len({row["trajectory_id"] for row in sampled}),
        "episodes": len(episodes),
        "transitions": len(rows),
        "heldout62_overlap": [],
        "applied_steps": trainer.state.global_step,
        "last_losses": trainer.last_self_metrics,
        "train_metrics": result.metrics,
        "optimizer_updates": audit.updates,
        "ema_verified": True,
        "training_wall_s": training_wall,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "terminal_unexecuted_tool_ids": terminal_unexecuted_ids,
        "external_model_calls": 0,
        "rollout_backend": args.rollout_backend,
        "refreshed_policy_validation": refreshed,
        "environment_source": "native evaluation image e976e7277861; original a0c7378 environment",
    }
    (args.output / "native-receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


def main():
    faulthandler.dump_traceback_later(90, repeat=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", action="append")
    parser.add_argument("--context-limit", type=int, choices=(8192, 16384), default=8192)
    parser.add_argument("--episode-concurrency", type=int, default=1)
    parser.add_argument("--server-sequences", type=int, default=4)
    parser.add_argument("--kv-cache-gib", type=int, default=2)
    parser.add_argument("--episode-output-tokens", type=int, default=12288)
    parser.add_argument("--qualify-context-only", action="store_true")
    parser.add_argument("--pilot-protocol", type=Path)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--run-name", default="self-native-qualification")
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollout-backend", choices=("transformers", "vllm"), default="transformers")
    args = parser.parse_args()
    if not 1 <= args.episode_concurrency <= args.server_sequences <= 32 or not 1 <= args.kv_cache_gib <= 32:
        parser.error("episode concurrency must fit server sequences (1..32), with KV cache in 1..32 GiB")
    if args.episode_output_tokens < 1 or args.rollouts < 1 or args.steps < 1 or not 0 < args.learning_rate <= 3e-4:
        parser.error("positive rollout/update counts and learning rate <=3e-4 are required")
    if args.pilot_protocol:
        if args.task:
            parser.error("choose the sealed pilot protocol or explicit tasks, not both")
        args.task = resolve_pilot_tasks(args.pilot_protocol, args.heldout)
    if not args.task or len(args.task) != len(set(args.task)):
        parser.error("unique task identities are required")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
