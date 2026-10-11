# ruff: noqa: E701, E702, E741, B905
"""Phase 1: re-execute recorded tool calls with one environment revision's tools.

    python replay_worker.py --src <dir containing automationbench/ and automationbench_v1/>
        < episodes.jsonl > replays.jsonl

Version agnostic: the simulator (``automationbench``) and the limited toolset's
own ``invoke`` come from ``--src``, so an episode recorded with an older
environment is replayed with the tools it actually ran against. Capture is done
here (full before/after world snapshots per call) in the receipt format that
the current manifest readers consume, so this file imports nothing from the
current environment.

Input episode: {"id", "data": task data (initial_state, zapier_tools, ...),
"nodes": trace nodes, "end_state": recorded final world or null}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Any

RAW_ACTION_KIND = "automationbench_raw_action"
# REPLAY_ERA_CAPTURE=1 reproduces the capture argument form of the episode's own
# environment revision (to explain recorded values); default is the current form.
ERA_CAPTURE = __import__("os").environ.get("REPLAY_ERA_CAPTURE") == "1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ alignment
_HEX = set("0123456789abcdefABCDEF-_")


def random_id(a: str, b: str) -> bool:
    """Same-shape minted identifiers: equal length, shared prefix, random tail."""
    if len(a) == len(b) and 4 <= len(a) <= 8 and a.isdigit() and b.isdigit():
        return True  # short numeric minted ids
    if len(a) != len(b) or not 6 <= len(a) <= 64:
        return False
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    ta, tb = a[i:], b[i:]
    if len(ta) < 2:
        return False
    if set(ta) <= _HEX and set(tb) <= _HEX:
        return True
    if ta.isalnum() and tb.isalnum():
        return True
    return ta.replace(".", "").isdigit() and tb.replace(".", "").isdigit()  # epoch "ts" strings


_VOLATILE_TOKEN = re.compile(
    r"[0-9a-fA-F-]{12,}|\d{4}-\d\d-\d\d(?:[T ][\d:.]+(?:Z|[+-]\d\d:?\d\d)?)?|\d{10,}(?:\.\d+)?"
)


def same_modulo_tokens(a: str, b: str) -> bool:
    """Equal once embedded minted ids, dates and epoch values are masked."""
    return _VOLATILE_TOKEN.sub("#", a) == _VOLATILE_TOKEN.sub("#", b)


def looks_time(s: str) -> bool:
    return len(s) >= 10 and s[:4].isdigit() and s[4] == "-"


def volatile(a: Any, b: Any) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return random_id(a, b) or looks_time(a) and looks_time(b) or same_modulo_tokens(a, b)
    if isinstance(a, int) and isinstance(b, int) and not isinstance(a, bool):
        return a >= 10**8 and b >= 10**8  # minted numeric ids, epoch second/millisecond clock reads
    return False


def align(recorded: Any, replayed: Any, mapping: dict[str, str]) -> int:
    """Learn recorded->replayed substitutions; return non-volatile mismatches."""
    if isinstance(recorded, dict) and isinstance(replayed, dict):
        bad = 0 if set(recorded) == set(replayed) else 1
        return bad + sum(align(recorded[k], replayed[k], mapping) for k in set(recorded) & set(replayed))
    if isinstance(recorded, list) and isinstance(replayed, list):
        bad = 0 if len(recorded) == len(replayed) else 1
        return bad + sum(align(a, b, mapping) for a, b in zip(recorded, replayed))
    if recorded == replayed:
        return 0
    if isinstance(recorded, str) and isinstance(replayed, str):
        mapping.setdefault(recorded, replayed)
    elif (
        isinstance(recorded, int)
        and isinstance(replayed, int)
        and not isinstance(recorded, bool)
        and volatile(recorded, replayed)
    ):
        mapping.setdefault(str(recorded), str(replayed))  # numeric ids: as numbers and inside text
    return 0 if volatile(recorded, replayed) else 1


def substitute(value: Any, mapping: dict[str, str]) -> Any:
    if not mapping:
        return value
    if isinstance(value, dict):
        return {k: substitute(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute(v, mapping) for v in value]
    if isinstance(value, int) and not isinstance(value, bool) and str(value) in mapping:
        replaced = mapping[str(value)]
        return int(replaced) if replaced.isdigit() else replaced
    if isinstance(value, str):
        if value in mapping:
            return mapping[value]
        for old, new in mapping.items():
            if len(old) >= 8 and old in value:
                value = value.replace(old, new)
    return value


def world_diff(recorded: Any, replayed: Any) -> int:
    if isinstance(recorded, dict) and isinstance(replayed, dict):
        return sum(
            world_diff(recorded.get(k), replayed.get(k))
            for k in set(recorded) | set(replayed)
            if k not in ("_automationbench_v1_private", "meta")
        )
    if isinstance(recorded, list) and isinstance(replayed, list):
        return sum(world_diff(a, b) for a, b in zip(recorded, replayed)) + abs(len(recorded) - len(replayed))
    return 0 if recorded == replayed or volatile(recorded, replayed) else 1


def parse(content: Any) -> Any:
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    if isinstance(content, str):
        try:
            return json.loads(content)
        except ValueError:
            return content
    return content


def recorded_calls(nodes: list[dict]) -> list[tuple[str, Any, Any]]:
    """(tool, raw arguments, recorded content) in the order results came back."""
    pending: dict[str, dict] = {}
    out = []

    def entry(call, content):
        fn = call.get("function") or {}
        return (
            call.get("name") or fn.get("name"),
            call["arguments"] if "arguments" in call else fn.get("arguments"),
            content,
        )

    def flush():
        # Calls the server executed but whose results never reached the transcript
        # (the episode ended on that turn): replayed without a recorded result.
        out.extend(entry(call, UNRECORDED) for call in pending.values())
        pending.clear()

    for node in nodes:
        message = node["message"]
        if message.get("role") == "assistant":
            flush()
            pending.update({c["id"]: c for c in message.get("tool_calls") or []})
        elif message.get("role") == "tool":
            call = pending.pop(message.get("tool_call_id"), None)
            if call is not None:
                out.append(entry(call, message.get("content")))
    flush()
    return out


UNRECORDED = object()


# ------------------------------------------------------------------ environment
class Env:
    def __init__(self, src: str):
        sys.path.insert(0, src)
        import automationbench_v1.limited_tools as lt
        from automationbench.schema.world import WorldState
        from mcp.server.mcpserver.utilities.func_metadata import func_metadata
        from verifiers.v1.mcp.execution import ToolServerReceipt
        from verifiers.v1.trace import StateWriteReceipt, ToolServerExecutionEvent

        self.WorldState, self.lt, self.func_metadata = WorldState, lt, func_metadata
        self.Receipt, self.Write, self.Event = ToolServerReceipt, StateWriteReceipt, ToolServerExecutionEvent
        try:
            from automationbench_v1 import simulation
        except ImportError:
            simulation = None
        self.simulation = simulation
        try:
            from automationbench_v1.taskset import _allowed_services
        except ImportError:
            _allowed_services = None
        self.allowed_services = _allowed_services
        self.meta: dict[str, Any] = {}

    def metadata(self, name: str):
        if name not in self.meta:
            stub = SimpleNamespace(invoke=lambda *a, **k: None)
            wrapper = self.lt.AutomationBenchLimitedToolset._tool_wrapper(stub, name, self.lt._ZAPIER_TOOLS[name])
            self.meta[name] = self.func_metadata(wrapper)
        return self.meta[name]

    def initial_world(self, data: dict, recorded_now: str | None, deterministic: bool):
        initial = data["initial_state"]
        clock = None
        if deterministic:
            today = datetime.fromisoformat(recorded_now.replace("Z", "+00:00")) if recorded_now else None
            clock = self.simulation.world_clock_base(initial, today=today)
            with self.simulation.simulated_setup(initial, clock):
                world = self.WorldState.model_validate(initial)
            world.meta.current_time = clock
        else:
            world = self.WorldState.model_validate(initial)
            if not (initial.get("meta") or {}).get("current_time") and recorded_now:
                world.meta.current_time = datetime.fromisoformat(recorded_now.replace("Z", "+00:00"))
        if self.allowed_services is not None:
            world.meta.allowed_services = self.allowed_services(
                initial, data.get("assertions") or (), data["zapier_tools"]
            )
        return world.model_dump(mode="json"), clock

    def replay(self, episode: dict, deterministic: bool) -> dict:
        data = episode["data"]
        end = episode.get("end_state") or {}
        world, clock = self.initial_world(data, (end.get("meta") or {}).get("current_time"), deterministic)
        state = SimpleNamespace(
            world=world,
            capture_actions=False,
            deterministic_world=deterministic,
            world_clock=clock.isoformat() if clock else None,
            world_revision=0,
            action_count=0,
            action_published=(),
            action_initial_digest=None,
        )
        allowed = tuple(data["zapier_tools"])
        toolset = SimpleNamespace(config=SimpleNamespace(allowed_tools=allowed), state=state)
        invoke = self.lt.AutomationBenchLimitedToolset.invoke
        mapping: dict[str, str] = {}
        events, writes = [], []
        stats = dict(calls=0, invoked=0, raised=0, skipped=0, mismatch=0, status_mismatch=0)
        revision = 0
        for index, (name, raw, content) in enumerate(recorded_calls(episode["nodes"])):
            stats["calls"] += 1
            unrecorded = content is UNRECORDED
            stats["unrecorded"] = stats.get("unrecorded", 0) + unrecorded
            recorded = None if unrecorded else parse(content)
            recorded_error = isinstance(recorded, str) and recorded.lstrip().lower().startswith("error")
            try:
                args = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw or {})
            except ValueError:
                args = None
            if name not in allowed or name not in self.lt._ZAPIER_TOOLS or not isinstance(args, dict):
                stats["skipped"] += 1  # unknown tool or unparsable call: never dispatched
                stats["status_mismatch"] += int(not recorded_error and not unrecorded)
                continue
            args = substitute(args, mapping)
            meta = self.metadata(name)
            try:
                validated = dict(meta.validate_arguments(meta.pre_parse_json(args)))
            except Exception:  # noqa: BLE001 - MCP argument validation precedes dispatch
                stats["skipped"] += 1
                stats["status_mismatch"] += int(not recorded_error and not unrecorded)
                continue
            if ERA_CAPTURE and not hasattr(self.lt, "_call_arguments"):
                called = {"args": [], "kwargs": validated}  # pre-d766982 capture form (diagnostic only)
            else:
                called = {k: v for k, v in validated.items() if v is not None}
            before_text = canonical_json(state.world)
            error = result = None
            try:
                result = invoke(toolset, name, **validated)
            except Exception as exc:  # noqa: BLE001
                error = exc
            after_text = canonical_json(state.world)
            stats["invoked"] += 1
            before, after = sha(before_text), sha(after_text)
            action = {
                "occurrence_index": index,
                "tool_name": name,
                "arguments_json": canonical_json(called),
                "before_digest": before,
                "after_digest": after,
                "status": "returned" if error is None else "raised",
                "result_json": canonical_json(result) if error is None else None,
                "error_json": None
                if error is None
                else canonical_json({"type": type(error).__name__, "message": str(error)}),
                "ordering_scope": "local_state_history",
                "native_join_status": "unqualified",
            }
            envelope = canonical_json(
                {"kind": RAW_ACTION_KIND, "action": action, "snapshots": {before: before_text, after: after_text}}
            )
            invocation = f"replay-invocation-{index}"
            identity = {
                "invocation_id": invocation,
                "tool_name": name,
                "arguments_json": canonical_json({"args": [], "kwargs": args}),
                "state_read_revision": revision,
            }
            dispatch = self.Receipt(**identity, event_index=0, phase="dispatch")
            if error is None:
                terminal = self.Receipt(
                    **identity,
                    event_index=1,
                    phase="returned",
                    evidence_json=(envelope,),
                    result_json=canonical_json(result),
                    state_write_revision=revision + 1,
                    state_conflict=False,
                    state_persistence="applied",
                )
            else:
                stats["raised"] += 1
                terminal = self.Receipt(
                    **identity,
                    event_index=1,
                    phase="raised",
                    evidence_json=(envelope,),
                    error_json=action["error_json"],
                )
            for receipt in (dispatch, terminal):
                events.append(
                    self.Event(
                        invocation_id=invocation,
                        event_index=receipt.event_index,
                        phase=receipt.phase,
                        receipt_seq=len(events),
                        state_revision=revision + (receipt.phase == "returned"),
                        receipt_json=canonical_json(receipt.model_dump(mode="json")),
                    ).model_dump(mode="json")
                )
            if error is None:
                writes.append(
                    self.Write(
                        write_id=invocation,
                        body_digest=after,
                        expected_revision=revision,
                        applied_revision=revision + 1,
                        conflict=False,
                    ).model_dump(mode="json")
                )
                revision += 1
                if not unrecorded and (recorded_error or align(recorded, parse(result), mapping)):
                    stats["mismatch"] += 1
            else:
                stats["status_mismatch"] += int(not recorded_error and not unrecorded)
        diff = None
        if end:
            diff = world_diff(end, substitute(state.world, {v: k for k, v in mapping.items()}))
        return {
            "events": events,
            "writes": writes,
            "final": state.world,
            "stats": stats,
            "world_diff": diff,
            "deterministic": deterministic,
            "fidelity": [stats["mismatch"] + stats["status_mismatch"], diff or 0],
        }

    def best_replay(self, episode: dict) -> dict:
        best = None
        for deterministic in (False, True) if self.simulation is not None else (False,):
            result = self.replay(episode, deterministic)
            if best is None or result["fidelity"] < best["fidelity"]:
                best = result
            if result["fidelity"] == [0, 0]:
                break
        return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    args = parser.parse_args()
    env = Env(args.src)
    for line in sys.stdin:
        if not line.strip():
            continue
        episode = json.loads(line)
        started = time.perf_counter()
        try:
            out = env.best_replay(episode)
        except Exception as error:  # noqa: BLE001
            out = {"error": f"replay {type(error).__name__}: {error}"[:400]}
        out["id"] = episode["id"]
        out["replay_seconds"] = round(time.perf_counter() - started, 4)
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
