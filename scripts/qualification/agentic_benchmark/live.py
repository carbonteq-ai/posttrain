#!/usr/bin/env python3
"""Run real AutomationBench episodes through Verifiers and record timing.

Builds the same environment configuration the LFM2.5 AutomationBench training
binding activates (``limited_zapier`` toolset, ``null`` harness, subprocess
runtime, 12 turns) and runs it with Verifiers' ``run_eval``. Episodes execute
the real tool server and harness processes, so host-side costs (process
startup, tool calls, prompt rendering) are measured, not assumed.

``--mock`` serves an OpenAI-compatible model that answers every request with a
short final message, so each episode is setup + one turn + scoring. That
isolates harness startup cost from model time. ``--replay-model TRACES`` makes
the mock answer each turn with the assistant message recorded for that task and
turn (tool calls included), so the real harness and tool server execute the
recorded tool calls: host-side cost per turn without a GPU. Without either the
runner talks to a real endpoint given by ``--base-url``.

Production job images run harness scripts with the packed job interpreter
instead of ``uv`` (``POSTTRAIN_VERIFIERS_PREINSTALLED``); ``--preinstalled``
applies the same policy with this interpreter so startup matches production.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import statistics
import sys
import time
from pathlib import Path

# Modules the AutomationBench tool server and the harness chat program import.
FORK_SERVER_PRELOAD = [
    "automationbench_v1.tools",
    "anyio",
    "httpx",
    "httpx2",
    "openai",
    "openai.resources",
    "tenacity",
    "mcp",
    "mcp.client.streamable_http",
]


def environment_config(max_turns: int, toolset: str) -> dict:
    # Mirrors apps/lab/.posttrain/catalog/lfm26-automationbench-comparison.yaml
    # (automationbench-lfm26-train-mix-episode-judged-v3) without its judge,
    # which has zero reward weight and needs a separate judge endpoint.
    return {
        "taskset": {
            "id": "automationbench-v1",
            "domains": ["simple", "sales", "marketing", "operations", "support", "finance", "hr"],
            "task": {"toolset": toolset, "search_top_k": 20},
        },
        "retries": {"max_retries": 0},
        "agent": {
            "harness": {"id": "null"},
            "runtime": {"type": "subprocess"},
            "timeout": {"setup": 120, "rollout": 1800, "finalize": 60, "scoring": 900},
            "max_turns": max_turns,
            "max_output_tokens": 12288,
        },
    }


def install_preinstalled_runtime(python: str) -> None:
    """Run harness scripts with ``python`` directly, as Posttrain job images do."""
    import hashlib
    import uuid

    from verifiers.v1.runtimes import base as runtime_base

    async def prepare(runtime, script, env=None, *, activate=True):
        del activate
        data = script.encode() if isinstance(script, str) else script
        digest = hashlib.sha256(data).hexdigest()
        path = f"/tmp/vf-scripts/{digest}.py"
        if digest not in runtime._uv_interpreters:
            async with runtime._uv_script_locks.setdefault(digest, asyncio.Lock()):
                if digest not in runtime._uv_interpreters:
                    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
                    await runtime.write(tmp, data)
                    result = await runtime.run(
                        ["sh", "-c", f"mkdir -p /tmp/vf-scripts && mv -f {shlex.quote(tmp)} {shlex.quote(path)}"],
                        env or {},
                    )
                    if result.exit_code != 0:
                        raise RuntimeError(result.stderr[-2000:])
                    runtime._uv_interpreters[digest] = python
        return [runtime._uv_interpreters[digest], path]

    runtime_base.Runtime.prepare_uv_script = prepare  # type: ignore[method-assign]


def recorded_turns(traces: Path) -> dict[str, list[dict]]:
    """First user message -> the first recorded sample's assistant messages."""
    turns: dict[str, list[dict]] = {}
    for line in traces.read_text().splitlines():
        record = json.loads(line)
        messages = [n["message"] for n in record["nodes"] if n.get("message")]
        user = next(m for m in messages if m["role"] == "user")
        key = json.dumps(user["content"], sort_keys=True)
        turns.setdefault(key, [m for m in messages if m["role"] == "assistant"])
    return turns


async def serve_mock(port: int, replay: dict[str, list[dict]] | None = None) -> object:
    from aiohttp import web

    async def completions(request):
        body = await request.json()
        messages = body.get("messages", [])
        user = next((m for m in messages if m.get("role") == "user"), None)
        recorded = replay.get(json.dumps(user["content"], sort_keys=True), []) if replay and user else []
        turn = sum(m.get("role") == "assistant" for m in messages)
        if turn < len(recorded):
            message = {k: v for k, v in recorded[turn].items() if k in ("role", "content")}
            if recorded[turn].get("tool_calls"):
                # Traces store Verifiers' flat tool calls; the wire format nests them.
                message["tool_calls"] = [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in recorded[turn]["tool_calls"]
                ]
            finish = "tool_calls" if message.get("tool_calls") else "stop"
        else:
            message, finish = {"role": "assistant", "content": "Done."}, "stop"
        return web.json_response(
            {
                "id": f"mock-{time.time_ns()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "mock"),
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    async def models(_request):
        return web.json_response({"object": "list", "data": [{"id": "mock", "object": "model"}]})

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_post("/v1/chat/completions", completions)
    app.router.add_get("/v1/models", models)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner


def timing_summary(traces_path: Path) -> dict:
    phases: dict[str, list[float]] = {}
    for line in traces_path.read_text().splitlines():
        record = json.loads(line)
        traces = record.get("traces") or [record]
        for trace in traces:
            timing = trace.get("timing") or {}
            for name in ("boot", "setup", "agent", "finalize", "scoring"):
                span = timing.get(name) or {}
                if span.get("start") is not None and span.get("end") is not None:
                    phases.setdefault(name, []).append(span["end"] - span["start"])
            agent = timing.get("agent") or {}
            for part in ("model", "harness"):
                value = (agent.get(part) or {}).get("duration")
                if value is not None:
                    phases.setdefault(f"agent.{part}", []).append(value)
            turns = len(trace.get("calls") or [])
            phases.setdefault("turns", []).append(turns)
            harness = (agent.get("harness") or {}).get("duration")
            if harness is not None and turns:
                phases.setdefault("harness_per_turn", []).append(harness / turns)
    return {
        name: {
            "n": len(values),
            "median_s": statistics.median(values),
            "mean_s": statistics.fmean(values),
            "max_s": max(values),
        }
        for name, values in phases.items()
    }


async def main_async(args) -> None:
    if args.fork_server:
        # Process-wide, as a job would set it: covers the agent runtime and the
        # tool servers' own runtimes.
        os.environ["VF_FORK_SERVER"] = "1"
        os.environ["VF_FORK_SERVER_PRELOAD"] = ",".join(FORK_SERVER_PRELOAD)
    if args.preinstalled:
        install_preinstalled_runtime(sys.executable)
    from verifiers.v1.cli.eval.runner import run_eval

    try:
        from verifiers.v1.configs.eval import EvalConfig
    except ImportError:
        from verifiers.v1.configs.cli.eval import EvalConfig

    replay = recorded_turns(args.replay_model) if args.replay_model else None
    use_mock = args.mock or replay is not None
    mock = await serve_mock(args.mock_port, replay) if use_mock else None
    base_url = f"http://127.0.0.1:{args.mock_port}/v1" if use_mock else args.base_url
    os.environ.setdefault("LIVE_BENCH_API_KEY", "EMPTY")
    task_keys = [line.strip() for line in args.tasks.read_text().splitlines() if line.strip()] if args.tasks else None
    output_dir = args.output.parent / f"{args.output.stem}-run"
    config = EvalConfig.model_validate(
        {
            "env": environment_config(args.max_turns, args.toolset),
            "model": args.model,
            "client": {"type": "eval", "base_url": base_url, "api_key_var": "LIVE_BENCH_API_KEY"},
            "sampling": {"temperature": args.temperature, "max_tokens": args.max_tokens, "top_p": args.top_p},
            "num_tasks": len(task_keys) if task_keys else args.num_tasks,
            "num_rollouts": args.rollouts,
            "max_concurrent": args.concurrency,
            "shuffle": False,
            "output_dir": output_dir,
            "run": {"name": args.label, "dir": "."},
            "push": False,
            "rich": None,
            "serve": None,
            **({"task_keys": task_keys} if task_keys else {}),
        }
    )
    started = time.perf_counter()
    try:
        episodes = await run_eval(config)
    finally:
        if mock is not None:
            await mock.cleanup()
    wall = time.perf_counter() - started
    traces_file = next(output_dir.rglob("traces.jsonl"))
    summary = {
        "label": args.label,
        "episodes": len(episodes),
        "ok": sum(bool(getattr(e, "ok", False)) for e in episodes),
        "concurrency": args.concurrency,
        "wall_s": wall,
        "phases": timing_summary(traces_file),
        "traces": str(traces_file),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=1))
    phases = summary["phases"]
    parts = ", ".join(f"{name} {v['median_s']:.2f}s (max {v['max_s']:.2f})" for name, v in phases.items() if v["n"])
    print(f"[{args.label}] {summary['ok']}/{summary['episodes']} ok, wall {wall:.1f}s; medians: {parts}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--replay-model", type=Path, default=None, help="recorded traces whose assistant turns the mock replays"
    )
    parser.add_argument("--mock-port", type=int, default=18765)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="mock")
    parser.add_argument("--tasks", type=Path, default=None, help="file of task keys, one per line")
    parser.add_argument("--num-tasks", type=int, default=16)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--toolset", default="limited_zapier")
    parser.add_argument("--preinstalled", action="store_true")
    parser.add_argument(
        "--fork-server", action="store_true", help="start Python programs from a warm, preloaded fork server"
    )
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
