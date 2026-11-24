"""Execute scripted tool calls in the frozen environment on CPU (plan D24, R4 and S2).

A correction or a cleaned solution is shown to the teacher only after it has run in
the same environment the attempts ran in. This module runs scripted assistant turns
through the real harness, tool server, native assertions and manifest assessments,
exactly as `sdpo_ab_replay.py` does for recorded episodes: a local mock model
answers each model request with the next scripted assistant message.

Why the real harness and not an in-process call of the tool function: the limited
toolset validates arguments at the MCP boundary (pydantic type checks, rejection of
undeclared keys, decoding of JSON-looking strings), runs each call inside the
deterministic-world seed, and the manifest assessments read the tool-server
execution receipts. Re-implementing any of that would be a second, unverified copy.

Several scripts may target the same task. Each script's tool-call ids carry its
number, so later model requests are routed back to the right script by the id of
the first tool call in the conversation; the first request of a rollout takes the
next unassigned script of that task.

Entry point: `run_scripts(scripts)`; each script is
    {"key": any hashable, "task": task name,
     "turns": [[{"name": tool, "arguments": dict}, ...], ...]}
and the result maps key -> native episode record (the same JSON form as the
`episodes.jsonl.gz` files), or an {"error": ...} dict when the rollout failed.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import tempfile
import time
from collections import defaultdict
from pathlib import Path

# World clock pin (opt-in, for replays only). The environment builds a task's initial world at UTC midnight of the
# REAL current date when the task's initial_state declares no meta.current_time, so schema-default timestamps (and
# the initial-world digest) change from one UTC day to the next. Setting SDPO_AB_WORLD_DATE=YYYY-MM-DD (or this
# constant) makes replays and in-process worlds use that day instead. None = environment default (not pinned).
WORLD_DATE = None


def world_date():
    return os.environ.get("SDPO_AB_WORLD_DATE") or WORLD_DATE


def pin_world_clock(day=None):
    """Patch the environment's world_clock_base (where taskset.setup and simulation look it up) so a task without
    meta.current_time uses UTC midnight of `day`; tasks that declare a current_time are unchanged. Returns the day
    pinned, or None when nothing is pinned. Idempotent."""
    from datetime import UTC, datetime

    day = day or world_date()
    if not day:
        return None
    import automationbench_v1.simulation as simulation
    import automationbench_v1.taskset as taskset

    original = getattr(simulation, "_sdpo_original_world_clock_base", simulation.world_clock_base)
    fixed = datetime.fromisoformat(str(day)).replace(tzinfo=UTC)

    def pinned(initial_state, *, today=None):
        return original(initial_state, today=today or fixed)

    simulation._sdpo_original_world_clock_base = original
    simulation.world_clock_base = pinned
    taskset.world_clock_base = pinned
    # naive date strings in task data are read in the process's local zone: the recordings ran in UTC containers,
    # so a pinned replay also runs in UTC (a UTC+5 laptop otherwise shifts every message date by 5 hours)
    if os.environ.get("TZ") != "UTC":
        os.environ["TZ"] = "UTC"
        time.tzset()
    return day

DONE_TEXT = "Done."


def _wait_for_headroom():
    """Block until the machine has headroom for another batch of rollouts, when a gate script is configured
    (environment variable SDPO_CAN_LAUNCH: a command that exits 0 when it is safe to start more workers)."""
    import subprocess

    gate = os.environ.get("SDPO_CAN_LAUNCH")
    while gate and subprocess.run(gate, shell=True, capture_output=True).returncode != 0:
        time.sleep(20)


def _run_sync(make):
    """Run a coroutine to completion from sync code, also when the caller is inside a running event loop
    (the training runner is async): then it runs on its own loop in a worker thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make())
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(make())).result()


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _call_id(script, turn, position):
    return f"call_s{script}_t{turn}_p{position}"


def _script_of(call_id):
    if not isinstance(call_id, str) or not call_id.startswith("call_s"):
        return None
    try:
        return int(call_id[len("call_s"):].split("_", 1)[0])
    except ValueError:
        return None


def _message(script_index, turn_index, calls):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": _call_id(script_index, turn_index, p), "type": "function",
                            "function": {"name": c["name"],
                                         "arguments": c["arguments"] if isinstance(c["arguments"], str)
                                         else json.dumps(c["arguments"], ensure_ascii=False)}}
                           for p, c in enumerate(calls)]}


async def _serve(port, scripts, prompts):
    """Mock chat-completions server answering from the scripts."""
    from aiohttp import web

    pending = defaultdict(list)  # user prompt -> unassigned script indices (FIFO)
    for index, script in enumerate(scripts):
        pending[prompts[script["task"]]].append(index)
    fillers = {"count": 0}

    async def completions(request):
        body = await request.json()
        messages = body.get("messages", [])
        user = next(m for m in messages if m.get("role") == "user")
        prompt = json.dumps(user["content"], sort_keys=True)
        assistants = [m for m in messages if m.get("role") == "assistant"]
        index = None
        for m in assistants:
            for call in m.get("tool_calls") or []:
                index = _script_of(call.get("id"))
                if index is not None:
                    break
            if index is not None:
                break
        if not assistants:
            queue = pending.get(prompt)
            index = queue.pop(0) if queue else None
            if index is None:
                fillers["count"] += 1
        turn = len(assistants)
        if index is not None and turn < len(scripts[index]["turns"]):
            message = _message(index, turn, scripts[index]["turns"][turn])
        else:
            message = {"role": "assistant", "content": DONE_TEXT}
        finish = "tool_calls" if message.get("tool_calls") else "stop"
        return web.json_response({"id": f"script-{time.time_ns()}", "object": "chat.completion",
                                  "created": int(time.time()), "model": "script",
                                  "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                                  "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_post("/v1/chat/completions", completions)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner, fillers


def task_prompts(task_names):
    """The public user prompt of each task, as the harness sends it."""
    from automationbench_v1.taskset import AutomationBenchConfig, AutomationBenchTaskset
    from sdpo_frozen_environment import taskset_config

    taskset = AutomationBenchTaskset(AutomationBenchConfig.model_validate(taskset_config(sorted(task_names))))
    out = {}
    for task in taskset.load():
        prompt = task.data.prompt
        user = next(m for m in prompt if (m.get("role") if isinstance(m, dict) else m.role) == "user")
        content = user.get("content") if isinstance(user, dict) else user.content
        out[task.data.task_name] = json.dumps(content, sort_keys=True)
    return out


def _script_index_of_record(record):
    from sdpo_ab_metrics import trace_of

    for node in trace_of(record)["nodes"]:
        message = node.get("message") or {}
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                index = _script_of(call.get("id"))
                if index is not None:
                    return index
    return None


async def _run(scripts, concurrency, prompts):
    pin_world_clock()
    from sdpo_ab_replay import environment
    from verifiers.v1.cli.eval.runner import run_eval
    from verifiers.v1.configs.cli.eval import EvalConfig

    names = sorted({s["task"] for s in scripts})
    rollouts = max(sum(s["task"] == n for s in scripts) for n in names)
    port = _free_port()
    os.environ.setdefault("SDPO_SCRIPT_KEY", "EMPTY")
    runner, fillers = await _serve(port, scripts, prompts)
    try:
        with tempfile.TemporaryDirectory(prefix="sdpo-exec-") as folder:
            config = EvalConfig.model_validate({
                "env": environment(names), "model": "script",
                "client": {"type": "eval", "base_url": f"http://127.0.0.1:{port}/v1",
                           "api_key_var": "SDPO_SCRIPT_KEY"},
                "sampling": {"temperature": 0.0, "max_tokens": 4096}, "num_tasks": len(names),
                "task_keys": names, "num_rollouts": rollouts, "max_concurrent": concurrency,
                "shuffle": False, "output_dir": Path(folder) / "native", "run": {"name": "script", "dir": "."},
                "push": False, "rich": None, "serve": None})
            episodes = await run_eval(config)
            records = [e.model_dump(mode="json") for e in episodes]
    finally:
        await runner.cleanup()
    out = {}
    for record in records:
        index = _script_index_of_record(record)
        if index is not None:
            out[index] = record
    return out


def initial_digest(record):
    from sdpo_ab_metrics import trace_of

    return ((trace_of(record).get("info") or {}).get("automationbench_capture") or {}).get("initial_digest")


def _execute(scripts, concurrency, prompts, chunk):
    """One pass: index in `scripts` -> native record (missing when the rollout failed).

    A run gives every task the same number of rollouts, so scripts are packed per task: each run takes up to
    `chunk // 8` scripts of every task that still has some (unused rollouts of a task answer "Done." at once)."""
    per_task = max(1, chunk // 8)
    queues = defaultdict(list)
    for index, script in enumerate(scripts):
        queues[script["task"]].append(index)
    out = {}
    while any(queues.values()):
        picked = []
        for task in list(queues):
            picked += queues[task][:per_task]
            queues[task] = queues[task][per_task:]
        part = [scripts[i] for i in picked]
        _wait_for_headroom()
        records = _run_sync(lambda: _run(part, concurrency, prompts))
        for position, record in records.items():
            out[picked[position]] = record
    return out


def run_scripts(scripts, concurrency=8, chunk=64, rounds=3):
    """Run every script in the frozen environment; key -> native record or {"error": ...}.

    The environment builds one of two byte-different initial worlds for a task: the first rollout of a task
    in a run gets one encoding, every later rollout the other (they differ only in internal row ids; recorded
    batches hold both, 1 of 8 and 7 of 8). Minted message and record ids depend on the world, so a script that
    names `initial_digest` must start from exactly that world. Scheduling:
      1. all scripts in one concurrent pass;
      2. scripts that got the wrong world run alone for their task (a task's only rollout is its first);
      3. scripts still wrong run as two copies side by side (only one copy can be its task's first rollout);
    a script whose world never matched returns {"error": "initial world mismatch"}."""
    scripts = list(scripts)
    for s in scripts:
        if not s["turns"] or not all(s["turns"]):
            raise ValueError(f"script {s['key']!r} needs at least one call in every turn")
    prompts = task_prompts({s["task"] for s in scripts})
    results = {}

    def settle(batch, records, copies=1):
        wrong = []
        for position, script in enumerate(batch):
            got = [records.get(position * copies + c) for c in range(copies)]
            got = [r for r in got if r is not None]
            want = script.get("initial_digest")
            match = next((r for r in got if not want or initial_digest(r) == want), None)
            if match is not None:
                results[script["key"]] = match
            elif got:
                results[script["key"]] = {"error": "initial world mismatch"}
                wrong.append(script)
            else:
                results[script["key"]] = {"error": "rollout missing"}
        return wrong

    wrong = settle(scripts, _execute(scripts, concurrency, prompts, chunk))
    for _ in range(rounds):
        if not wrong:
            break
        solo, rest = [], []
        seen = set()
        for script in wrong:
            (rest if script["task"] in seen else solo).append(script)
            seen.add(script["task"])
        wrong = settle(solo, _execute(solo, concurrency, prompts, chunk)) + rest
        if not wrong:
            break
        doubled = [dict(script, key=(script["key"], c)) for script in wrong for c in range(2)]
        wrong = settle(wrong, _execute(doubled, concurrency, prompts, chunk), copies=2)
    return results


def call_results(record):
    """Tool results of a scripted record in call order: (turn, position, name, arguments, result, error)."""
    from sdpo_ab_context import episode_calls

    calls, _, _ = episode_calls(record)
    return calls


def save_cache(path, cache):
    Path(path).write_text(json.dumps(cache, indent=1, sort_keys=True))


def load_cache(path):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------- one call on a recorded world, in process

class InProcessTools:
    """The task's own limited toolset and MCP tool manager, run in this process on a given world.

    Used to verify a corrected call (plan D24, R4) at the exact point of a recorded attempt: the world before
    the call is the attempt's verified snapshot (`WorldTimeline`), the call goes through the same MCP argument
    validation, strict-argument rejection, JSON-string decoding and deterministic-world seeding as in a
    rollout. Not exercised: the harness transport and the manifest assessments (guards), which need a rollout
    (`run_scripts`)."""

    _cache = {}

    def __new__(cls, task_name):
        if task_name not in cls._cache:
            obj = super().__new__(cls)
            obj._setup(task_name)
            cls._cache[task_name] = obj
        return cls._cache[task_name]

    def _setup(self, task_name):
        pin_world_clock()
        import inspect

        from automationbench_v1.limited_tools import _ZAPIER_TOOLS
        from automationbench_v1.taskset import AutomationBenchConfig, AutomationBenchTaskset
        from automationbench_v1.tools import AutomationBenchState
        from mcp.server.mcpserver import MCPServer
        from sdpo_frozen_environment import taskset_config

        task = AutomationBenchTaskset(AutomationBenchConfig.model_validate(taskset_config([task_name]))).load()[0]
        self.task = task
        self.toolset = type(task).toolsets(task.config)[0]
        self.state_cls = AutomationBenchState
        self._holder = {}

        async def pull():
            return self._holder["state"]

        async def push(before):
            return None

        self.toolset._pull_state, self.toolset._push_state = pull, push
        self.toolset._state_cls = AutomationBenchState
        self.mcp = MCPServer("sdpo-in-process")
        self.toolset.register(self.mcp)
        self.allowed = {name: frozenset(p for p in inspect.signature(_ZAPIER_TOOLS[name]).parameters if p != "world")
                        for name in task.config.allowed_tools}

    def clock(self, fallback_day=None):
        from datetime import UTC, datetime

        from automationbench_v1.simulation import world_clock_base

        today = datetime.fromisoformat(fallback_day).astimezone(UTC) if fallback_day else None
        return world_clock_base(self.task.data.initial_state, today=today).isoformat()

    def call(self, world, clock, revision, name, arguments):
        """(error text or None, result text, world after) of one call on `world` (a JSON world dict)."""
        state = self.state_cls(world=world, capture_actions=False, deterministic_world=True, world_clock=clock,
                               world_revision=revision, search_top_k=self.task.config.search_top_k)
        self._holder["state"] = state
        if name not in self.allowed:
            return f"Unknown tool: {name}", "", world
        if not isinstance(arguments, dict):
            return "arguments are not an object", "", world
        extra = set(arguments) - self.allowed[name]
        if extra and self.task.config.strict_tool_arguments:
            message = (f"Unsupported arguments for {name}: {', '.join(sorted(extra))}. "
                       "Use the declared tool parameter names. No action was executed.")
            return message, message, world
        try:
            result = _run_sync(lambda: self.mcp.call_tool(name, dict(arguments)))
        except Exception as error:  # noqa: BLE001 - a tool error is a result here
            text = str(error) if str(error).startswith("Error executing tool") else f"Error executing tool {name}: {error}"
            return text, text, world
        text = "".join(getattr(part, "text", "") for part in (result.content or []))
        from sdpo_ab_metrics import tool_error_text

        error = text if getattr(result, "is_error", False) or getattr(result, "isError", False) else tool_error_text(text)
        return error, text, self._holder["state"].world
