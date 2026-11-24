"""One recorded AutomationBench attempt's world, rebuilt with the environment's own code (precheck, check 2).

The functions down to `score` are copied unchanged in behaviour from the replica tool
`docs/research/evidence/sdpo-ab-20261104-abtool/abtool.py` (D60; its exactness test replayed 147 recorded episodes of
two control runs to their recorded partial credit, success and per-assertion results, 147/147): the task loaded
through AutomationBenchTaskset with the study's frozen options and tool additions, the world built by the task's own
`setup()` under the study's clock pin, the environment's own `AutomationBenchLimitedToolset` on a real MCP server
called in process through the MCP client with verifiers' middleware, verifiers' null-harness argument handling and
tool-output truncation, and `automationbench_v1.scoring.score_world`. The only changes: imports resolve from this
directory (the GPU job ships it), the world clock is pinned by the caller, and no session files are written.

Added for the precheck: the recorded calls and outcome of an episode record (as abtool's exactness test reads them),
the initial-world variant chosen from the recorded initial digest (first or later rollout of a task in a process;
README "Two initial worlds"), `rebuild` (replay + verification) and `execute`/`grade` on a rebuilt world.

Every function here is synchronous and runs its own event loop (`asyncio.run`), so the GPU job calls them from a
worker thread (never from its running loop).
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as _dt
import json
import os
import time as _time
import uuid
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("AUTOMATIONBENCH_STRICT_ASSERTIONS", "1")
# D31: study runs execute in UTC (naive dates in task data are read in the local zone).
os.environ["TZ"] = "UTC"
_time.tzset()

from sdpo_frozen_environment import DOMAINS, LIMITS, TASK_OPTIONS, TOOL_ADDITIONS  # noqa: E402

MAX_TURNS = int(LIMITS["max_turns"])
SERVER_NAME = "AutomationBenchLimitedToolset"  # what the real server announces (TOOL_PREFIX None)


def check_imports():
    """Import everything the replica uses, so a missing module stops the job before the model is loaded."""
    import automationbench_v1.limited_tools  # noqa: F401
    import automationbench_v1.scoring  # noqa: F401
    import automationbench_v1.taskset  # noqa: F401
    import automationbench_v1.tools  # noqa: F401
    from mcp import Client  # noqa: F401
    from mcp.server.mcpserver import MCPServer  # noqa: F401
    from verifiers.v1.harnesses.utils.compaction import bound_tool_message  # noqa: F401
    from verifiers.v1.harnesses.utils.mcp import mcp_content_to_chat_content  # noqa: F401
    from verifiers.v1.mcp.server import _call_state, _execution_metadata_middleware  # noqa: F401


# ----------------------------------------------------------------------------- task loading (abtool)
def load_task(name: str):
    """The task exactly as the study's taskset config loads it (same options, same tool additions)."""
    from automationbench_v1.taskset import AutomationBenchConfig, AutomationBenchTaskset

    config: dict[str, Any] = {"domains": list(DOMAINS), "task_names": [name], "task": dict(TASK_OPTIONS)}
    if name in TOOL_ADDITIONS:
        config["additional_tools_by_task"] = {name: tuple(TOOL_ADDITIONS[name])}
    (task,) = AutomationBenchTaskset(AutomationBenchConfig.model_validate(config)).load()
    return task


def initial_state(task, today: _dt.date, fresh: bool = False):
    """Run the task's own setup() under the study's world-clock pin (abtool.initial_state; see its docstring).
    fresh=False: the later-rollout world (the task's own validate() first); fresh=True: the first-rollout world."""
    from automationbench_v1.tools import AutomationBenchState
    from sdpo_ab_execution import pin_world_clock

    pin_world_clock(today.isoformat())
    trace = SimpleNamespace(state=AutomationBenchState(), info={})

    async def run():
        if not fresh:
            await task.validate(None)
        await task.setup(trace, None)

    asyncio.run(run())
    return trace.state


def prepare(name: str, today: _dt.date, fresh: bool = False):
    """Load the task and run its setup, as the environment server does for every rollout (abtool.prepare)."""
    task = load_task(name)
    return task, initial_state(task, today, fresh)


def _state_adapter():
    from automationbench_v1.tools import AutomationBenchState
    from pydantic import TypeAdapter

    return TypeAdapter(AutomationBenchState)


# ----------------------------------------------------------------------------- tool server (abtool)
class ToolHost:
    """One rollout's MCP tool server, in process, over the environment's own toolset class."""

    def __init__(self, task, state):
        from automationbench_v1.limited_tools import AutomationBenchLimitedToolset
        from automationbench_v1.taskset import AutomationBenchTask, AutomationBenchTaskConfig

        adapter = _state_adapter()
        host = self

        class Toolset(AutomationBenchLimitedToolset):
            # State channel: pull/push the rollout state as JSON bytes, exactly like the HTTP channel.
            async def _pull_state(self):
                return adapter.validate_json(host.state_bytes)

            async def _push_state(self, before: bytes) -> None:
                from verifiers.v1.mcp.server import _call_state

                current = _call_state.get()
                after = adapter.dump_json(current)
                if after != before:
                    host.state_bytes = after

            async def _emit_execution_receipt(self, receipt) -> None:
                host.receipts.append(receipt.model_dump(mode="json"))

        # Same construction as AutomationBenchTask.tool_servers() in the environment.
        config = AutomationBenchTaskConfig.model_validate(
            {**task.config.model_dump(), "allowed_tools": task.data.zapier_tools}
        )
        (real,) = AutomationBenchTask.toolsets(config)
        if type(real) is not AutomationBenchLimitedToolset:
            raise RuntimeError(f"unexpected toolset {type(real).__name__}")
        self.toolset = Toolset(real.config)
        self.state_bytes = adapter.dump_json(state)
        self.receipts: list[dict] = []
        self.adapter = adapter

    def state(self):
        return self.adapter.validate_json(self.state_bytes)

    @contextlib.asynccontextmanager
    async def client(self):
        from mcp import Client
        from mcp.server.mcpserver import MCPServer
        from verifiers import __version__
        from verifiers.v1.mcp.server import _execution_metadata_middleware

        mcp = MCPServer(SERVER_NAME, version=__version__, middleware=[_execution_metadata_middleware])
        self.toolset.register(mcp)
        async with Client(mcp) as client:
            yield client


async def tool_definitions(host: ToolHost) -> list[dict]:
    """What the null harness gives the model: MCP list_tools, bare names (TOOL_PREFIX None)."""
    async with host.client() as client:
        result = await client.list_tools()
    return [
        {"name": tool.name, "description": tool.description or "", "parameters": tool.input_schema}
        for tool in result.tools
    ]


async def harness_call(client, dispatch: set[str], name: str, raw_arguments: str | None):
    """One tool call as verifiers' null harness (harnesses/utils/core.py run_chat_loop) makes it.

    Returns (content, status) where status is "ok", "rejected" (harness-side) or "raised"."""
    from verifiers.v1.harnesses.utils.compaction import bound_tool_message
    from verifiers.v1.harnesses.utils.mcp import mcp_content_to_chat_content

    try:
        tool_args = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as e:
        content, status = f"error: invalid JSON in tool arguments ({e}); resend the call with valid JSON", "rejected"
    else:
        if not isinstance(tool_args, dict):
            content = (f"error: tool arguments must be a JSON object, got {type(tool_args).__name__}; "
                       "resend as an object")
            status = "rejected"
        elif name not in dispatch:
            content, status = f"error: unknown tool {name!r}", "rejected"
        else:
            meta = {"verifiers.execution": {"dispatch_ticket": uuid.uuid4().hex,
                                            "parent_execution_id": uuid.uuid4().hex,
                                            "transport_attempt_index": 0}}
            result = await client.call_tool(name, tool_args, meta=meta)
            content = mcp_content_to_chat_content(result.content)
            status = "error_result" if result.is_error else "ok"
    message = bound_tool_message({"role": "tool", "tool_call_id": "x", "content": content, "name": name})
    return message["content"], status


async def run_calls(host: ToolHost, calls: list[tuple[str, str | None]]) -> list[tuple[Any, str]]:
    async with host.client() as client:
        dispatch = {tool.name for tool in (await client.list_tools()).tools}
        out = []
        for name, raw in calls:
            out.append(await harness_call(client, dispatch, name, raw))
        return out


# ----------------------------------------------------------------------------- scoring (abtool)
def score(task, state) -> dict:
    from automationbench_v1.scoring import score_world

    snap = score_world(world=state.world, initial_state=task.data.initial_state, assertions=task.data.assertions)
    return {
        "task": task.data.task_name,
        "partial_credit": snap.partial_credit,
        "task_completed_correctly": snap.task_completed_correctly,
        "assertions_passed": snap.assertions_passed,
        "assertions_scored": snap.assertions_scored,
        "assertions_excluded": snap.assertions_excluded,
        "assertion_results": list(snap.assertion_results),
    }


def _arguments_text(value: Any) -> str | None:
    """The raw function.arguments string; objects are encoded as an OpenAI client would."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value)


# ----------------------------------------------------------------------------- recorded episodes (abtool test)
def path_nodes(nodes: list[dict]) -> list[dict]:
    """Nodes on the path from the root to the deepest leaf (the conversation the harness ran)."""
    depth = {}
    for i, n in enumerate(nodes):
        depth[i] = 0 if n["parent"] is None else depth[n["parent"]] + 1
    leaf = max(depth, key=lambda i: (depth[i], i))
    out = []
    while leaf is not None:
        out.append(nodes[leaf])
        leaf = nodes[leaf]["parent"]
    return out[::-1]


def recorded_calls(record: dict) -> tuple[list[dict], int]:
    """The episode's answered tool calls in order ({"tool", "arguments", "id", "recorded_result"}) and the number of
    calls without a recorded result (abtool's exactness test `extract`)."""
    (trace,) = record["traces"]
    path = [n["message"] for n in path_nodes(trace["nodes"])]
    results = {m["tool_call_id"]: m["content"] for m in path if m.get("role") == "tool"}
    calls, unanswered = [], 0
    for m in path:
        if m.get("role") != "assistant":
            continue
        for c in m.get("tool_calls") or []:
            if c["id"] not in results:
                unanswered += 1
                continue
            calls.append({"tool": c["name"], "arguments": c["arguments"], "id": c["id"],
                          "recorded_result": results[c["id"]]})
    return calls, unanswered


def recorded_score(record: dict) -> dict:
    (trace,) = record["traces"]
    return {"partial_credit": trace["rewards"]["partial_credit"]["score"],
            "task_completed_correctly": trace["metrics"]["task_completed_correctly"],
            "assertion_results": trace["info"]["automationbench"]["assertions"]}


def task_name(record: dict) -> str:
    return record["traces"][0]["task"]["data"]["task_name"]


class World:
    """A rebuilt world: the post-setup task (the grader's initial state and assertions) and its tool host."""

    def __init__(self, task, host, variant):
        self.task, self.host, self.variant = task, host, variant


def rebuild(record: dict, today: _dt.date) -> tuple[World, dict]:
    """The attempt's world after its recorded calls, and the verification report.

    The initial-world variant is the one whose digest equals the recorded `initial_digest` (abtool's exactness test;
    "unmatched" builds the later-rollout world, as that test does). `verified` is the registered condition: the
    replayed world scores exactly the recorded partial credit and success. The report also says whether every
    per-assertion result and every replayed tool-result text equals the recording."""
    name = task_name(record)
    digest = record["traces"][0]["info"]["automationbench_capture"]["initial_digest"]
    later = prepare(name, today)
    first = prepare(name, today, fresh=True)
    if later[1].action_initial_digest == digest:
        (task, state), variant = later, "later_rollout"
    elif first[1].action_initial_digest == digest:
        (task, state), variant = first, "first_rollout"
    else:
        (task, state), variant = later, "unmatched"
    host = ToolHost(task, state)
    calls, unanswered = recorded_calls(record)
    pairs = [(c["tool"], _arguments_text(c.get("arguments"))) for c in calls]
    results = asyncio.run(run_calls(host, pairs))
    got = score(task, host.state())
    expected = recorded_score(record)
    diffs = [i for i, ((content, _status), c) in enumerate(zip(results, calls)) if content != c["recorded_result"]]
    report = {
        "variant": variant, "calls_replayed": len(calls), "unanswered_calls": unanswered,
        "replayed": {k: got[k] for k in ("partial_credit", "task_completed_correctly", "assertions_passed",
                                         "assertions_scored", "assertions_excluded")},
        "recorded": {k: expected[k] for k in ("partial_credit", "task_completed_correctly")},
        "verified": (got["partial_credit"] == expected["partial_credit"]
                     and got["task_completed_correctly"] == expected["task_completed_correctly"]),
        "assertions_equal": got["assertion_results"] == expected["assertion_results"],
        "result_text_diffs": diffs,
        "assertion_results": got["assertion_results"],
    }
    return World(task, host, variant), report


def execute(world: World, pairs: list[tuple[str, str | None]]) -> list[tuple[Any, str]]:
    """Run tool calls (name, raw arguments text) in order on the world, as the null harness does."""
    return asyncio.run(run_calls(world.host, pairs))


def grade(world: World) -> dict:
    return score(world.task, world.host.state())


def tools_of(world: World) -> list[dict]:
    return asyncio.run(tool_definitions(world.host))
