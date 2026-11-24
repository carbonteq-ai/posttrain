"""The SFT model's recorded attempts -> whole-attempt rows.

One row per attempt of the W6 run (P9 on the 204 training tasks, 8 attempts each), in the SFT row shape so that the
SFT gates (sftgen/validate_row.py) run on it unchanged:
  messages   system and user as recorded; each assistant turn with its visible text (`content`), its reasoning
             (`reasoning_content`, the text before </think> of P9's own sampled reply in the token chain) and its tool
             calls (arguments parsed, key order kept); one tool message per call with the exact result; the final reply
  tools      the chain's tool list (what P9 was given)
  trainable_message_indices  every assistant message
  metadata   task, episode_id, source, the attempt's recorded outcome and failed checks, chain cross-checks
Nothing is edited. Cross-checks (each recorded; a failed one sets metadata.exact false):
  - the trace's conversation path has as many assistant messages as the chain has sampled segments;
  - each assistant message's visible text equals the chain's visible text (after </think>, tool-call markup removed);
  - no sampled reply ends inside an unclosed tool call (a render would add tokens P9 never sampled);
  - each assistant message's call ids equal the chain segment's call ids;
  - every tool message answers the preceding calls in order.
Helpers path_nodes, split_reasoning and the tool-call markup rule are copied from wt-weekend-v4 sdpo_ab_world.py and
sdpo_a3.py (969d6472), which built the SDPO A3 rows from the same records.

usage: python3 -I attempt_rows.py RUN_DIR OUT.jsonl [--tasks T1,T2,...] [--episodes FILE]
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

THINK_CLOSE = "</think>"
SOURCE = "p9-w6-attempt"
_TOOL_CALL = re.compile(r"<tool_call>.*?</tool_call>", re.S)


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


def split_reasoning(raw: str) -> tuple[str, str]:
    """(reasoning, visible text) of one generated reply (the generation prompt opened `<think>`)."""
    if THINK_CLOSE not in raw:
        return "", raw.strip()
    reasoning, visible = raw.split(THINK_CLOSE, 1)
    return reasoning.strip(), visible.strip()


def strip_calls(visible: str) -> str:
    return _TOOL_CALL.sub("", visible).strip()


def parse_arguments(raw):
    return json.loads(raw) if isinstance(raw, str) else raw


def attempt_row(record: dict, chain: dict, outcome: dict) -> dict:
    (trace,) = record["traces"]
    path = [n["message"] for n in path_nodes(trace["nodes"])]
    head = [m for m in path if m.get("role") in ("system", "user")][:2]
    actions = [s for s in chain["segments"] if s["kind"] == "action"]
    assistants = [m for m in path if m.get("role") == "assistant"]
    problems = []
    if len(assistants) != len(actions):
        problems.append(f"assistant messages {len(assistants)} != chain actions {len(actions)}")
    results = {m["tool_call_id"]: m for m in path if m.get("role") == "tool"}
    messages = [{"role": m["role"], "content": m["content"]} for m in head]
    trainable = []
    for k, m in enumerate(assistants):
        raw = actions[k].get("raw") or "" if k < len(actions) else ""
        reasoning, visible = split_reasoning(raw)
        content = m.get("content") or ""
        if visible.count("<tool_call>") > visible.count("</tool_call>"):
            # the sampled reply ends inside a tool call (no closing tags); the parser accepted the call, but a render
            # of this turn would add closing tokens P9 never sampled, so the attempt fails the token-parity rule
            problems.append(f"turn {k + 1}: unclosed tool call in the sampled reply")
            visible = visible[: visible.rfind("<tool_call>")]
        if strip_calls(visible) != content.strip():
            problems.append(f"turn {k + 1}: visible text differs from the chain")
        calls = [{"id": c["id"], "type": "function",
                  "function": {"name": c["name"], "arguments": parse_arguments(c["arguments"])}}
                 for c in m.get("tool_calls") or []]
        if k < len(actions) and [c["id"] for c in calls] != list(actions[k].get("tool_call_ids") or []):
            problems.append(f"turn {k + 1}: call ids differ from the chain")
        trainable.append(len(messages))
        messages.append({"role": "assistant", "content": content, "reasoning_content": reasoning,
                         **({"tool_calls": calls} if calls else {})})
        for c in m.get("tool_calls") or []:
            r = results.get(c["id"])
            if r is None:
                problems.append(f"turn {k + 1}: call {c['id']} has no result")
                continue
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": r["content"], "name": c["name"]})
    if not assistants or assistants[-1].get("tool_calls"):
        problems.append("no final reply")
    assertions = trace["info"]["automationbench"]["assertions"]
    failed = [{"type": a["type"], "params": a.get("params"), "excluded": bool(a.get("excluded"))}
              for a in assertions if not a.get("passed")]
    task = trace["task"]["data"]["task_name"]
    return {"id": f"dpo-w6/{task}/{record['id']}", "messages": messages, "tools": chain["tools"],
            "trainable_message_indices": trainable,
            "metadata": {"task": task, "episode_id": record["id"], "source": SOURCE,
                         "outcome": {k: outcome.get(k) for k in ("success", "partial_credit", "harm", "manifest_harms",
                                                                  "tool_errors", "stop_condition", "runaway",
                                                                  "reply_limit_cut", "tokens", "infrastructure_failure",
                                                                  "chain_joined", "preservation_break")},
                         "recorded_failed_checks": failed, "exact": not problems, "chain_problems": problems}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir")
    parser.add_argument("out")
    parser.add_argument("--tasks", help="comma-separated task names (default: all)")
    parser.add_argument("--episodes", help="file with one episode id per line (default: all)")
    args = parser.parse_args(argv)
    run = Path(args.run_dir)
    tasks = set(args.tasks.split(",")) if args.tasks else None
    episodes = set(Path(args.episodes).read_text().split()) if args.episodes else None
    outcomes = {o["episode_id"]: o for o in json.load(open(run / "outcomes.json"))}

    def wanted(eid):
        o = outcomes.get(eid)
        return o is not None and (tasks is None or o["task"] in tasks) and (episodes is None or eid in episodes)

    chains = {}
    with gzip.open(run / "chains.jsonl.gz", "rt") as s:
        for line in s:
            c = json.loads(line)
            if wanted(c["episode_id"]):
                chains[c["episode_id"]] = c
    written = inexact = 0
    with gzip.open(run / "episodes.jsonl.gz", "rt") as s, open(args.out, "w") as w:
        for line in s:
            r = json.loads(line)
            if not wanted(r["id"]):
                continue
            row = attempt_row(r, chains[r["id"]], outcomes[r["id"]])
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            inexact += not row["metadata"]["exact"]
    print(json.dumps({"rows": written, "inexact": inexact}))
    return 0 if inexact == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
