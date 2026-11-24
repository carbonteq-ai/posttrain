"""Replay recorded assistant turns through the frozen environment on CPU (plan D9, D20).

The mock model answers each turn with the assistant message recorded for that
task, so the real harness, tool server and manifest assessments execute the
recorded calls. This yields native traces with manifest findings for episodes
that were recorded before manifest assessment existed (builder validation only).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from sdpo_frozen_environment import LIMITS, taskset_config


def environment(task_names, max_turns=LIMITS["max_turns"], max_output_tokens=LIMITS["max_context_tokens"],
                rollout_timeout=3600):
    return {
        "taskset": taskset_config(task_names),
        "retries": {"max_retries": 0},
        "agent": {
            "harness": {"id": "null"},
            "runtime": {"type": "subprocess"},
            "timeout": {"setup": 120, "rollout": rollout_timeout, "finalize": 120, "scoring": 900},
            "max_turns": max_turns,
            "max_output_tokens": max_output_tokens,
        },
    }


def wire_message(recorded):
    message = {"role": "assistant", "content": recorded.get("content") or ""}
    if recorded.get("tool_calls"):
        message["tool_calls"] = [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": c["arguments"]
                          if isinstance(c["arguments"], str) else json.dumps(c["arguments"])}}
            for c in recorded["tool_calls"]]
    return message


async def serve(port, turns_by_prompt):
    from aiohttp import web

    async def completions(request):
        body = await request.json()
        messages = body.get("messages", [])
        user = next(m for m in messages if m.get("role") == "user")
        recorded = turns_by_prompt.get(json.dumps(user["content"], sort_keys=True), [])
        turn = sum(m.get("role") == "assistant" for m in messages)
        if turn < len(recorded):
            message = wire_message(recorded[turn])
        else:
            message = {"role": "assistant", "content": "Done."}
        finish = "tool_calls" if message.get("tool_calls") else "stop"
        return web.json_response({"id": f"replay-{time.time_ns()}", "object": "chat.completion",
            "created": int(time.time()), "model": "replay",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_post("/v1/chat/completions", completions)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner


async def main(args):
    from verifiers.v1.cli.eval.runner import run_eval
    from verifiers.v1.configs.cli.eval import EvalConfig

    if os.environ.get("POSTTRAIN_VERIFIERS_PREINSTALLED") == "1":
        import sys

        from verifiers_preinstalled import configure_preinstalled_runtime

        configure_preinstalled_runtime(interpreter=sys.executable)
    cases = json.loads(args.cases.read_text())["cases"]
    turns, names = {}, []
    for case in cases:
        trace = case["episode"]["traces"][0]
        name = case["episode"]["task"]["data"]["name"]
        if args.tasks and name not in args.tasks:
            continue
        messages = [n["message"] for n in trace["nodes"] if n.get("message")]
        user = next(m for m in messages if m["role"] == "user")
        turns[json.dumps(user["content"], sort_keys=True)] = [m for m in messages if m["role"] == "assistant"]
        names.append(name)
    os.environ.setdefault("REPLAY_KEY", "EMPTY")
    runner = await serve(args.port, turns)
    try:
        config = EvalConfig.model_validate({
            "env": environment(names), "model": "replay",
            "client": {"type": "eval", "base_url": f"http://127.0.0.1:{args.port}/v1", "api_key_var": "REPLAY_KEY"},
            "sampling": {"temperature": 0.0, "max_tokens": 4096}, "num_tasks": len(names),
            "task_keys": names, "num_rollouts": 1, "max_concurrent": 4, "shuffle": False,
            "output_dir": args.output, "run": {"name": "replay", "dir": "."},
            "push": False, "rich": None, "serve": None})
        episodes = await run_eval(config)
    finally:
        await runner.cleanup()
    print(json.dumps({"episodes": len(episodes), "tasks": names}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18791)
    asyncio.run(main(parser.parse_args()))
