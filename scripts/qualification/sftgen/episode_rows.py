"""Teacher episodes -> SupervisedDataset rows (plan v4 P4).

Input: a teacher_runner.py output folder (episodes.jsonl.gz, summary.json) and the teacher_proxy.py folder of the
same run (requests.jsonl.gz, ledger.jsonl). For every episode whose run ended with the agent's own final reply
(stop_condition agent_completed) the row is:
  messages: the trace messages in order: system and user as recorded; each assistant turn with its visible text
            (`content`, "" if none), its reasoning (`reasoning_content`, exactly as recorded) and its tool calls
            (`tool_calls`, arguments parsed from the recorded JSON string with key order kept); each tool message
            with its call id and the exact result text; the final assistant reply;
  tools:    the trace's tool definitions in the chat-completions shape;
  trainable_message_indices: every assistant message.
Cross-checks recorded in metadata (a failed one marks the row `exact: false`, which validate_row does not admit):
  - the proxy saw a request whose messages begin with exactly this system and user message and whose tools equal the
    row's tools (byte-equal JSON) -- the row is what the teacher saw;
  - every tool message answers the preceding calls in order.
The episode's own benchmark score, stop condition and the run's environment/eval-code commits are recorded;
admission is decided by validate_row.py (replay, checks), not here.

usage: python episode_rows.py RUN_DIR PROXY_DIR OUT.jsonl --model qwen3.5-397b-a17b [--prefix sft-phase2a-v4]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path


def text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def chat_tool(tool: dict) -> dict:
    if tool.get("type") == "function" and "function" in tool:
        return tool
    out = {"name": tool["name"], "description": tool.get("description", ""), "parameters": tool.get("parameters", {})}
    if tool.get("strict") is not None:
        out["strict"] = tool["strict"]
    return {"type": "function", "function": out}


class MalformedArguments(ValueError):
    """A tool call whose arguments are not valid JSON: the call cannot be written into a row faithfully."""


def parse_arguments(raw):
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except ValueError as error:
        raise MalformedArguments(f"call arguments are not valid JSON: {raw[:120]!r}") from error


CANONICAL_ID = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")  # posttrain.data adapters' _CANONICAL_ID


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def episode_row(record: dict, *, model: str, prefix: str, run_summary: dict) -> dict:
    trace = record["traces"][0]
    raw_task = record.get("task")
    task = raw_task if isinstance(raw_task, str) else (raw_task or {}).get("key") or ((raw_task or {}).get("data") or {}).get("task_name")
    if not isinstance(task, str) or not task:
        raise ValueError(f"episode {record.get('id')} has no task name")
    messages, problems, pending = [], [], []
    for node in trace["nodes"]:
        message = node.get("message")
        if not message:
            continue
        role = message.get("role")
        if role in ("system", "user"):
            messages.append({"role": role, "content": text_of(message.get("content"))})
        elif role == "assistant":
            out = {"role": "assistant", "content": text_of(message.get("content"))}
            reasoning = message.get("reasoning_content")
            if reasoning is not None:
                out["reasoning_content"] = reasoning
            calls = []
            for call in message.get("tool_calls") or []:
                calls.append({"id": call["id"], "type": "function",
                              "function": {"name": call["name"], "arguments": parse_arguments(call["arguments"])}})
            if calls:
                out["tool_calls"] = calls
            if pending:
                problems.append(f"assistant turn before all calls were answered: {pending}")
            pending = [c["id"] for c in calls]
            messages.append(out)
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not pending or pending[0] != call_id:
                problems.append(f"tool message {call_id} out of order (expected {pending[:1]})")
            else:
                pending.pop(0)
            tool_message = {"role": "tool", "tool_call_id": call_id, "content": text_of(message.get("content"))}
            if message.get("name"):
                tool_message["name"] = message["name"]
            messages.append(tool_message)
    tools = [chat_tool(t) for t in trace.get("tools") or []]
    metrics = trace.get("metrics") or {}
    first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
    row_id = f"{prefix}/{model}/{task}/{record['id']}"
    if not CANONICAL_ID.fullmatch(row_id):  # the data adapter replaces any other id with rows/<index>
        raise ValueError(f"row id {row_id!r} is not canonical (lowercase letters, digits, . _ / -)")
    return {
        "id": row_id,
        "messages": messages,
        "tools": tools,
        "trainable_message_indices": [i for i, m in enumerate(messages) if m["role"] == "assistant"],
        "metadata": {
            "task": task, "episode_id": record["id"], "source": "teacher-in-runner", "model": model,
            "stop_condition": trace.get("stop_condition"), "metrics": metrics,
            "task_completed_correctly": metrics.get("task_completed_correctly"),
            "environment": (run_summary.get("environment") or {}).get("commit"),
            "eval_code": (run_summary.get("eval_code") or {}).get("head"),
            "limits": run_summary.get("limits"), "first_user_sha256": sha(first_user),
            "structure_problems": problems, "final_reply": "teacher (recorded)",
        },
    }


def proxy_requests(proxy_dir: Path) -> dict:
    """Compact index of the proxy's request log, streamed (P7 logs thousands of long requests): for each distinct
    first two messages (role, text), the sha256 of the tools JSON of the first request that began with them."""
    path = proxy_dir / "requests.jsonl.gz"
    index: dict = {}
    if not path.exists():
        return index
    with gzip.open(path, "rt") as stream:
        for line in stream:
            if not line.strip():
                continue
            body = json.loads(line)["body"]
            sent = body.get("messages") or []
            if len(sent) < 2:
                continue
            head = json.dumps([(m.get("role"), text_of(m.get("content"))) for m in sent[:2]], ensure_ascii=False)
            index.setdefault(head, sha(json.dumps(body.get("tools"), ensure_ascii=False)))
    return index


def cross_check(row: dict, requests: dict) -> dict:
    """The row's system and user messages begin a request the proxy saw, and that request's tools are byte-equal to
    the row's tools (compared through sha256 of the same JSON text)."""
    head = json.dumps([(m["role"], m["content"]) for m in row["messages"][:2]], ensure_ascii=False)
    if head not in requests:
        return {"request_found": False, "tools_equal_request": False}
    return {"request_found": True,
            "tools_equal_request": requests[head] == sha(json.dumps(row["tools"], ensure_ascii=False))}


from settings import FROZEN_SETS as FROZEN  # noqa: E402


def evaluation_tasks(frozen: Path = FROZEN) -> set[str]:
    sets = json.loads(frozen.read_text())["sets"]
    return set(sets["test120"]["per_task"]) | set(sets["guard10"]["per_task"])


def build(run_dir: Path, proxy_dir: Path, *, model: str, prefix: str) -> tuple[list[dict], list[dict]]:
    """Quarantine: a run containing any test120 or guard10 task is refused outright (evaluation trajectories, e.g.
    the teacher's P6 baseline, must never become training data)."""
    summary = json.loads((run_dir / "summary.json").read_text())
    leaked = sorted(set(summary.get("tasks") or []) & evaluation_tasks())
    if leaked:
        raise ValueError(f"refusing to build rows from a run with evaluation tasks: {leaked[:5]} ({len(leaked)})")
    requests = proxy_requests(proxy_dir)
    rows, skipped = [], []
    with gzip.open(run_dir / "episodes.jsonl.gz", "rt") as stream:
        for line in stream:
            record = json.loads(line)
            try:
                row = episode_row(record, model=model, prefix=prefix, run_summary=summary)
            except MalformedArguments as error:  # P7 r1 c4: the teacher wrote {"id": zoom_noise_011}
                task = ((record.get("task") or {}).get("data") or {}).get("name")
                if task in evaluation_tasks():
                    raise ValueError(f"refusing evaluation task {task} in training rows")
                skipped.append({"episode_id": record["id"], "task": task, "reason": f"malformed call arguments: {error}"})
                continue
            meta = row["metadata"]
            if meta["task"] in evaluation_tasks():
                raise ValueError(f"refusing evaluation task {meta['task']} in training rows")
            if meta["stop_condition"] != "agent_completed":
                skipped.append({"episode_id": record["id"], "task": meta["task"], "reason": meta["stop_condition"]})
                continue
            meta["cross_check"] = cross_check(row, requests)
            meta["exact"] = (not meta["structure_problems"] and meta["cross_check"]["request_found"]
                             and meta["cross_check"]["tools_equal_request"])
            rows.append(row)
    return rows, skipped


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("proxy_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefix", default="sft-phase2a-v4")
    args = parser.parse_args(argv)
    rows, skipped = build(args.run_dir, args.proxy_dir, model=args.model, prefix=args.prefix)
    with args.out.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "exact": sum(r["metadata"]["exact"] for r in rows), "skipped": skipped}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
