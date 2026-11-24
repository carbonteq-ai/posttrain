"""abtool2: solve one AutomationBench task through its own tools, one assistant TURN per command (SFT Route B).

Derived from abtool.py (docs/research/evidence/sdpo-ab-20261104-abtool/abtool.py, D60). The task loading, world
build, clock pin, tool server, null-harness call path, 20,000-byte result truncation, scoring and the `start` rendering
are copied unchanged (test_abtool2.py asserts the copied functions are source-identical). What is new:

  * `turn`  executes a JSON list of calls IN ORDER as ONE assistant turn (the runner executes a turn's tool_calls in
            order: verifiers harnesses/utils/core.py `for call in message.tool_calls`) and records the turn's visible
            note, every call (arguments object with its key order) and every exact result text.
  * `reply` records the final reply (the assistant message without tool calls) and closes the session.
  * the session holds everything the training row needs (prompt.json, turns.jsonl, reply.json), so no transcript
    parsing is required; transcript text is only a cross-check.
  * an exclusive, non-blocking lock (fcntl.flock on DIR/.lock): a second command on the same session while one runs
    is REJECTED (exit 4) without touching the session. Writes are atomic (temp file + os.replace) and every load
    checks that meta, state and turns agree.
  * limits: the runner's 50 model turns, counting the final reply (so at most 49 calling turns), and at most 45 calls
    in total (the admission cap). A turn that would exceed either is rejected whole, before any call runs.
  * the environment pin is enforced: the automationbench_v1 package must come from a git checkout at the expected
    commit (default bd655be7ae63, override ABTOOL2_EXPECT_ENV); the commit is recorded in meta.json.

Run with the environment interpreter that loads env-fix (bd655be):
  PY=python3  # with PYTHONPATH loading environments/automationbench_v1/src of the pinned checkout
  $PY abtool2.py start  --task NAME --session DIR [--today YYYY-MM-DD] [--fresh-setup] [--json]
  $PY abtool2.py turn   --session DIR --note "TEXT" --calls '[{"tool": "NAME", "args": {...}}, ...]'
  $PY abtool2.py reply  --session DIR --text "TEXT"
  $PY abtool2.py finish --session DIR                     # coordinator only: score the final world
  $PY abtool2.py replay --task NAME --calls FILE [--today D] [--fresh-setup] [--json] [--no-cap]

Exit codes: 0 ok; 2 bad input (nothing executed, no turn used); 3 refused (limit reached or session closed);
4 session busy (another command holds the lock); 5 session inconsistent or environment pin mismatch.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as _dt
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("AUTOMATIONBENCH_STRICT_ASSERTIONS", "1")
# D31: study runs execute in UTC (naive dates in task data are read in the local zone).
os.environ["TZ"] = "UTC"
import time as _time  # noqa: E402

_time.tzset()

HERE = Path(__file__).resolve().parent
QUALIFICATION = HERE.parent  # wt-sft scripts/qualification (sdpo_frozen_environment, sdpo_ab_execution)
sys.path.insert(0, str(QUALIFICATION))

from sdpo_frozen_environment import DOMAINS, LIMITS, TASK_OPTIONS, TOOL_ADDITIONS  # noqa: E402

STUDY_WORLD_DATE = _dt.date(2026, 10, 28)  # D31 (SDPO_AB_WORLD_DATE of every study run)
MAX_TURNS = int(LIMITS["max_turns"])  # 50 model turns per episode, as in the study's evaluation config
MAX_CALLING_TURNS = MAX_TURNS - 1  # the final reply is a model turn too
MAX_CALLS = 45  # admission cap on calls per trajectory
EXPECTED_ENV = os.environ.get("ABTOOL2_EXPECT_ENV", "86994309c1b90efb443fbd36c70b5cc03796a963")  # task fixes v1 (plan v4)
STATE_FILE = "state.json"
META_FILE = "meta.json"
PROMPT_FILE = "prompt.json"
TURNS_FILE = "turns.jsonl"
REPLY_FILE = "reply.json"
SCORE_FILE = "score.json"
LOCK_FILE = ".lock"
SERVER_NAME = "AutomationBenchLimitedToolset"  # what the real server announces (TOOL_PREFIX None)
FORBIDDEN_MARKUP = ("<tool_call>", "</tool_call>", "<function=", "<parameter=", "<tool_response>", "<|", "<think>",
                    "</think>")

EXIT_OK, EXIT_INPUT, EXIT_REFUSED, EXIT_BUSY, EXIT_BROKEN = 0, 2, 3, 4, 5


class Refusal(Exception):
    """A command that is not executed; carries the exit code and the message shown to the solver."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


# ----------------------------------------------------------------------------- task loading (copied from abtool.py)
def load_task(name: str):
    """The task exactly as the study's taskset config loads it (same options, same tool additions)."""
    from automationbench_v1.taskset import AutomationBenchConfig, AutomationBenchTaskset

    config: dict[str, Any] = {"domains": list(DOMAINS), "task_names": [name], "task": dict(TASK_OPTIONS)}
    if name in TOOL_ADDITIONS:
        config["additional_tools_by_task"] = {name: tuple(TOOL_ADDITIONS[name])}
    (task,) = AutomationBenchTaskset(AutomationBenchConfig.model_validate(config)).load()
    return task


def initial_state(task, today: _dt.date | None = None, fresh: bool = False):
    """Run the task's own setup() under the study's world-clock pin.

    ``today`` is the UTC day used as the world clock when the task declares no
    ``meta.current_time``; default STUDY_WORLD_DATE (D31: every study run pins 2026-10-28 and
    TZ=UTC through sdpo_ab_execution.pin_world_clock, which is called here).
    setup()'s seed is a digest of task.data.initial_state, and WorldState validation flattens
    nested Google Sheets data *in place* in that dict. In an evaluation process the first rollout
    of a task is set up on the pristine data and every later rollout of that task on the data the
    earlier setup flattened, so their schema-default ids (and every id minted later) differ.
    Default (fresh=False) is the later-rollout world, the case of all but one attempt per task:
    the task's own validate() is run first to flatten the data. fresh=True gives the
    first-rollout world."""
    from sdpo_ab_execution import pin_world_clock
    from automationbench_v1.tools import AutomationBenchState

    pin_world_clock((today or STUDY_WORLD_DATE).isoformat())
    trace = SimpleNamespace(state=AutomationBenchState(), info={})

    async def run():
        if not fresh:
            await task.validate(None)
        await task.setup(trace, None)

    asyncio.run(run())
    return trace.state


def prepare(name: str, today: _dt.date | None = None, fresh: bool = False):
    """Load the task and run its setup, as the environment server does for every rollout.

    setup() matters beyond the returned state: WorldState validation flattens nested Google
    Sheets (and similar) structures *in place* inside ``task.data.initial_state``, and the
    environment scores against that mutated initial state (and records it in the trace). So
    every command that scores or executes rebuilds the task through setup() first."""
    task = load_task(name)
    return task, initial_state(task, today, fresh)


def _state_adapter():
    from pydantic import TypeAdapter
    from automationbench_v1.tools import AutomationBenchState

    return TypeAdapter(AutomationBenchState)


def save_state(session: Path, state) -> None:
    """Persist the rollout state as the tool server's state channel would, minus the grader inputs.

    ``initial_state`` and ``assertions`` are rebuilt from the dataset (deterministic) so the
    session never stores the assertions."""
    hidden = state.model_copy(update={"initial_state": {}, "assertions": ()})
    (session / STATE_FILE).write_bytes(_state_adapter().dump_json(hidden))


def load_state(session: Path, task):
    state = _state_adapter().validate_json((session / STATE_FILE).read_bytes())
    state.initial_state = task.data.initial_state
    state.assertions = task.data.assertions
    return state


# ----------------------------------------------------------------------------- tool server (copied from abtool.py)
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


# ----------------------------------------------------------------------------- scoring (copied from abtool.py)
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


# ----------------------------------------------------------------------------- rendering (copied from abtool.py)
def render_start(prompt: list[dict], tools: list[dict]) -> str:
    parts = []
    for message in prompt:
        role = message.get("role")
        title = {"system": "SYSTEM PROMPT", "user": "USER MESSAGE"}.get(role, f"{str(role).upper()} MESSAGE")
        parts.append(f"===== {title} =====\n{message.get('content')}")
    parts.append(f"===== TOOLS ({len(tools)}) =====\n" + json.dumps(tools, indent=2, ensure_ascii=False))
    return "\n\n".join(parts) + "\n"


def _arguments_text(value: Any) -> str | None:
    """The raw function.arguments string; objects are encoded as an OpenAI client would."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value)


def _parse_day(text: str | None) -> _dt.date | None:
    return _dt.date.fromisoformat(text) if text else None


# ----------------------------------------------------------------------------- new: pin, lock, atomic session files
def environment_pin() -> dict:
    """Where automationbench_v1 is imported from and that checkout's git commit (must be the expected one)."""
    import automationbench_v1

    src = Path(automationbench_v1.__file__).resolve().parent
    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(src), "status", "--porcelain", "--", "."], capture_output=True,
                           text=True).stdout.strip()
    return {"env_commit": head, "env_src": str(src), "env_src_dirty": bool(dirty)}


def check_pin(pin: dict) -> None:
    if not pin["env_commit"].startswith(EXPECTED_ENV) and not EXPECTED_ENV.startswith(pin["env_commit"] or "x"):
        raise Refusal(EXIT_BROKEN, f"abtool2: environment at {pin['env_commit'] or '(no git)'} ({pin['env_src']}), "
                                   f"expected {EXPECTED_ENV}")
    if pin["env_src_dirty"]:
        raise Refusal(EXIT_BROKEN, f"abtool2: environment source {pin['env_src']} has uncommitted changes")


def tool_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with open(tmp, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


@contextlib.contextmanager
def session_lock(session: Path):
    """Exclusive non-blocking lock: a concurrent command on the same session is rejected, never interleaved."""
    session.mkdir(parents=True, exist_ok=True)
    fd = os.open(session / LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Refusal(EXIT_BUSY, "abtool2: another command is running on this session; wait for it to finish, "
                                     "then run this command again (nothing was executed)") from None
        yield
    finally:
        os.close(fd)  # closing the descriptor releases the lock


def read_meta(session: Path) -> dict:
    try:
        return json.loads((session / META_FILE).read_text())
    except FileNotFoundError:
        raise Refusal(EXIT_INPUT, f"abtool2: {session} is not a session (no {META_FILE}); run start first") from None


def read_turns(session: Path) -> list[dict]:
    path = session / TURNS_FILE
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def consistent(session: Path, meta: dict) -> list[dict]:
    """Refuse a session whose files disagree (an interrupted write); returns the recorded turns."""
    turns = read_turns(session)
    problems = []
    if len(turns) != meta["turns"]:
        problems.append(f"{len(turns)} recorded turns, meta says {meta['turns']}")
    if sum(len(t["calls"]) for t in turns) != meta["calls"]:
        problems.append("recorded calls disagree with meta")
    if _sha256((session / STATE_FILE).read_bytes()) != meta["state_sha256"]:
        problems.append("state.json digest differs from meta")
    if meta["replied"] != (session / REPLY_FILE).exists():
        problems.append("reply flag disagrees with reply.json")
    if problems:
        raise Refusal(EXIT_BROKEN, "abtool2: session files are inconsistent (" + "; ".join(problems) +
                      "); discard this session and start a new one")
    return turns


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


def clean_text(text: str, what: str) -> str:
    """The visible text as it will be trained: surrounding whitespace removed (the chat template trims it too)."""
    stripped = (text or "").strip()
    if not stripped:
        raise Refusal(EXIT_INPUT, f"abtool2: the {what} is empty; write it and run the command again "
                                  "(nothing was executed)")
    bad = [m for m in FORBIDDEN_MARKUP if m in stripped]
    if bad:
        raise Refusal(EXIT_INPUT, f"abtool2: the {what} contains reserved markup {bad}; remove it "
                                  "(nothing was executed)")
    return stripped


def parse_calls(text: str) -> list[dict]:
    """The --calls JSON: a non-empty list of {"tool": NAME, "args": OBJECT}. Key order of each args object is kept."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        raise Refusal(EXIT_INPUT, f"abtool2: --calls is not valid JSON ({e}); nothing was executed") from None
    if not isinstance(value, list) or not value:
        raise Refusal(EXIT_INPUT, "abtool2: --calls must be a non-empty JSON list of {\"tool\": NAME, \"args\": "
                                  "{...}}; nothing was executed")
    calls = []
    for i, item in enumerate(value, 1):
        if not isinstance(item, dict) or not isinstance(item.get("tool"), str) or not item["tool"].strip():
            raise Refusal(EXIT_INPUT, f"abtool2: call {i} needs a \"tool\" name; nothing was executed")
        extra = set(item) - {"tool", "args"}
        if extra:
            raise Refusal(EXIT_INPUT, f"abtool2: call {i} has unknown keys {sorted(extra)} (use \"tool\" and "
                                      "\"args\"); nothing was executed")
        args = item.get("args", {})
        if not isinstance(args, dict):
            raise Refusal(EXIT_INPUT, f"abtool2: call {i} \"args\" must be a JSON object; nothing was executed")
        calls.append({"tool": item["tool"], "args": args})
    return calls


def render_results(turn: int, records: list[dict]) -> str:
    parts = []
    for r in records:
        content = r["result"] if isinstance(r["result"], str) else json.dumps(r["result"])
        parts.append(f"===== RESULT {r['index']} of {len(records)}: {r['tool']} =====\n{content}")
    return "\n\n".join(parts) + "\n"


# ----------------------------------------------------------------------------- commands
def cmd_start(args) -> int:
    session = Path(args.session)
    pin = environment_pin()
    check_pin(pin)
    with session_lock(session):
        if any(p.name != LOCK_FILE for p in session.iterdir()):
            raise Refusal(EXIT_INPUT, f"abtool2: session directory {session} is not empty; use a new one per attempt")
        today = _parse_day(args.today)
        task, state = prepare(args.task, today, args.fresh_setup)
        host = ToolHost(task, state)
        tools = asyncio.run(tool_definitions(host))
        prompt = [dict(m) for m in task.data.model_dump(mode="json")["prompt"]]
        save_state(session, state)
        _atomic_write(session / PROMPT_FILE, json.dumps({"prompt": prompt, "tools": tools},
                                                         ensure_ascii=False).encode())
        (session / TURNS_FILE).touch()
        meta = {"task": args.task, "tool": "abtool2", "tool_sha256": tool_sha256(), "turns": 0, "calls": 0,
                "max_turns": MAX_TURNS, "max_calling_turns": MAX_CALLING_TURNS, "max_calls": MAX_CALLS,
                "replied": False, "finished": False, "world_date": (today or STUDY_WORLD_DATE).isoformat(),
                "fresh_setup": args.fresh_setup, "world_clock": state.world_clock, "created": _now(),
                "python": sys.executable, **pin,
                "state_sha256": _sha256((session / STATE_FILE).read_bytes())}
        _atomic_write(session / META_FILE, json.dumps(meta, indent=1).encode())
    if args.json:
        print(json.dumps({"prompt": prompt, "tools": tools}, ensure_ascii=False))
    else:
        sys.stdout.write(render_start(prompt, tools))
    return EXIT_OK


def cmd_turn(args) -> int:
    session = Path(args.session)
    with session_lock(session):
        meta = read_meta(session)
        consistent(session, meta)
        if meta["replied"] or meta["finished"]:
            raise Refusal(EXIT_REFUSED, "abtool2: the final reply was already given; no further tool calls are "
                                        "accepted")
        note = clean_text(args.note, "note")
        calls = parse_calls(args.calls)
        if meta["turns"] >= MAX_CALLING_TURNS:
            raise Refusal(EXIT_REFUSED, f"abtool2: turn budget exhausted ({MAX_TURNS} turns including the final "
                                        "reply); only the final reply is accepted now")
        if meta["calls"] + len(calls) > MAX_CALLS:
            raise Refusal(EXIT_REFUSED, f"abtool2: this turn has {len(calls)} calls but only "
                                        f"{MAX_CALLS - meta['calls']} of {MAX_CALLS} remain; nothing was executed")
        pin = environment_pin()
        check_pin(pin)
        if pin["env_commit"] != meta["env_commit"]:
            raise Refusal(EXIT_BROKEN, "abtool2: environment commit changed since start; discard this session")
        started = _now()
        task, _ = prepare(meta["task"], _parse_day(meta["world_date"]), meta["fresh_setup"])
        host = ToolHost(task, load_state(session, task))
        pairs = [(c["tool"], _arguments_text(c["args"])) for c in calls]
        results = asyncio.run(run_calls(host, pairs))
        number = meta["turns"] + 1
        records = [{"index": j, "tool": c["tool"], "arguments": c["args"], "arguments_text": raw, "status": status,
                    "result": content}
                   for j, (c, (_, raw), (content, status)) in enumerate(zip(calls, pairs, results), 1)]
        record = {"turn": number, "note": note, "calls": records, "started": started, "ended": _now()}
        if note != args.note:
            record["note_raw"] = args.note
        state_bytes = _state_adapter().dump_json(host.state().model_copy(update={"initial_state": {},
                                                                                  "assertions": ()}))
        old = (session / TURNS_FILE).read_bytes()
        meta.update(turns=number, calls=meta["calls"] + len(calls), state_sha256=_sha256(state_bytes))
        _atomic_write(session / STATE_FILE, state_bytes)
        _atomic_write(session / TURNS_FILE, old + (json.dumps(record, ensure_ascii=False) + "\n").encode())
        _atomic_write(session / META_FILE, json.dumps(meta, indent=1).encode())
    sys.stdout.write(render_results(number, records))
    print(f"[turn {number} of at most {MAX_CALLING_TURNS} calling turns; {meta['calls']} of {MAX_CALLS} calls used]",
          file=sys.stderr)
    return EXIT_OK


def cmd_reply(args) -> int:
    session = Path(args.session)
    with session_lock(session):
        meta = read_meta(session)
        consistent(session, meta)
        if meta["replied"] or meta["finished"]:
            raise Refusal(EXIT_REFUSED, "abtool2: the final reply was already given; the session is closed")
        text = clean_text(args.text, "reply")
        record = {"text": text, "turn": meta["turns"] + 1, "created": _now()}
        if text != args.text:
            record["text_raw"] = args.text
        _atomic_write(session / REPLY_FILE, json.dumps(record, ensure_ascii=False, indent=1).encode())
        meta["replied"] = True
        _atomic_write(session / META_FILE, json.dumps(meta, indent=1).encode())
    print("Reply recorded. The task is closed.")
    return EXIT_OK


def cmd_finish(args) -> int:
    session = Path(args.session)
    with session_lock(session):
        meta = read_meta(session)
        consistent(session, meta)
        task, _ = prepare(meta["task"], _parse_day(meta["world_date"]), meta["fresh_setup"])
        result = score(task, load_state(session, task))
        result.update(turns=meta["turns"], calls=meta["calls"])
        _atomic_write(session / SCORE_FILE, json.dumps(result, indent=1).encode())
        meta["finished"] = True
        _atomic_write(session / META_FILE, json.dumps(meta, indent=1).encode())
    print(json.dumps({k: v for k, v in result.items() if k != "assertion_results"}, indent=1))
    for i, item in enumerate(result["assertion_results"]):
        flag = "EXCLUDED" if item.get("excluded") else ("PASS" if item.get("passed") else "FAIL")
        print(f"  [{i}] {flag:8} {item.get('type')}")
    return EXIT_OK


def replay(task_name: str, calls: list[dict], today: _dt.date | None = None, session: Path | None = None,
           cap: bool = True, fresh: bool = False) -> dict:
    """start + every call + finish in one process (same code path as the separate commands; from abtool.py)."""
    task, state = prepare(task_name, today, fresh)
    host = ToolHost(task, state)
    pairs = [(c["tool"], _arguments_text(c.get("arguments"))) for c in calls]
    if cap and len(pairs) > MAX_CALLS:
        raise ValueError(f"{len(pairs)} calls exceed the {MAX_CALLS}-call cap")
    results = asyncio.run(run_calls(host, pairs))
    final = host.state()
    out = score(task, final)
    out["calls"] = [{"tool": n, "arguments": a, "status": s, "result": c} for (n, a), (c, s) in zip(pairs, results)]
    if session is not None:
        session.mkdir(parents=True, exist_ok=True)
        save_state(session, final)
        (session / SCORE_FILE).write_text(json.dumps(out, indent=1))
    return out


def cmd_replay(args) -> int:
    calls = json.loads(Path(args.calls).read_text())
    out = replay(args.task, calls, _parse_day(args.today), Path(args.session) if args.session else None,
                 cap=not args.no_cap, fresh=args.fresh_setup)
    if args.json:
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(json.dumps({k: v for k, v in out.items() if k not in ("assertion_results", "calls")}, indent=1))
    return EXIT_OK


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="abtool2", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("start", help="build the task's world and print what the model receives")
    p.add_argument("--task", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--today", help="UTC day for tasks without a declared clock (default 2026-10-28, the study's D31 pin)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--fresh-setup", action="store_true",
                   help="world of a task's first rollout in a process (default: a later rollout's world)")
    p = sub.add_parser("turn", help="one assistant turn: a visible note plus one or more calls, executed in order")
    p.add_argument("--session", required=True)
    p.add_argument("--note", required=True, help="the visible text written before the calls")
    p.add_argument("--calls", required=True, help='JSON list of {"tool": NAME, "args": {...}}')
    p = sub.add_parser("reply", help="record the final reply and close the session")
    p.add_argument("--session", required=True)
    p.add_argument("--text", required=True)
    p = sub.add_parser("finish", help="score the final world (coordinator only, never the solver)")
    p.add_argument("--session", required=True)
    p = sub.add_parser("replay", help="start, apply a recorded list of calls, finish")
    p.add_argument("--task", required=True)
    p.add_argument("--calls", required=True, help='JSON list of {"tool": NAME, "arguments": JSON text or object}')
    p.add_argument("--today")
    p.add_argument("--session")
    p.add_argument("--json", action="store_true")
    p.add_argument("--fresh-setup", action="store_true")
    p.add_argument("--no-cap", action="store_true", help=f"allow more than {MAX_CALLS} calls")
    args = parser.parse_args(argv)
    handler = {"start": cmd_start, "turn": cmd_turn, "reply": cmd_reply, "finish": cmd_finish,
               "replay": cmd_replay}[args.command]
    try:
        return handler(args)
    except Refusal as refusal:
        print(str(refusal))
        return refusal.code


if __name__ == "__main__":
    sys.exit(main())
