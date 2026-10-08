#!/usr/bin/env python3
"""Turn harvested agent trajectories into SFT and DPO JSONL datasets.

Input: ``trajectories.jsonl.gz`` written by ``harvest_agent_trajectories.py`` (one
compact episode per line with ``side`` best/worst, task name, tools, nodes, reward and
verified harm count).

- ``sft.jsonl``: one supervised example per best episode without harms. ``messages`` is
  the trained trajectory (system, user, assistant turns with ``reasoning_content`` and
  ``tool_calls``, tool results); every sampled assistant message is trainable.
- ``dpo.jsonl``: per task, best episodes (no harms) paired with worst episodes of the same
  task. ``prompt`` is the shared system and user messages; ``chosen`` and ``rejected`` are
  the rest of each trajectory. Each best and each worst episode is used at most twice.

usage: trajectories_to_training_data.py IN.jsonl.gz --out DIR [--sft] [--dpo]
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path


def trajectory(nodes: list[dict]) -> list[dict]:
    """Messages of the trained trajectory: the deepest path, with sampled replies in place.

    A reply whose tool call did not parse is kept as a sampled leaf, and the conversation
    continues from an unsampled canonical copy; training uses the sampled reply."""

    if not nodes:
        return []
    sampled_by_parent = {
        node.get("parent"): node
        for node in nodes
        if node.get("sampled") and (node.get("message") or {}).get("role") == "assistant"
    }
    depth: dict[int, int] = {}
    for index, node in enumerate(nodes):
        parent = node.get("parent")
        depth[index] = 0 if parent is None else depth.get(parent, 0) + 1
    leaf = max(range(len(nodes)), key=lambda index: (depth[index], index))
    path = []
    index: int | None = leaf
    while index is not None:
        path.append(nodes[index])
        index = nodes[index].get("parent")
    messages = []
    for node in reversed(path):
        message = node.get("message") or {}
        if message.get("role") == "assistant" and not node.get("sampled"):
            message = (sampled_by_parent.get(node.get("parent")) or node).get("message") or message
        messages.append(clean(message))
    return messages


def clean(message: dict) -> dict:
    keep = {
        key: message[key]
        for key in ("role", "content", "reasoning_content", "tool_calls", "tool_call_id", "name")
        if message.get(key) not in (None, "", [])
    }
    if message.get("role") == "assistant":
        keep.setdefault("content", "")
    return keep


def tools(record: dict) -> list[dict]:
    return [
        tool if "function" in tool else {"type": "function", "function": tool} for tool in record.get("tools") or []
    ]


def example_id(record: dict, prefix: str) -> str:
    return re.sub(r"[^a-z0-9._/-]", "-", f"{prefix}/{record['task_name']}/{record['id']}".lower())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sft", action="store_true")
    parser.add_argument("--dpo", action="store_true")
    parser.add_argument("--pairs-per-episode", type=int, default=2)
    args = parser.parse_args()
    records = [json.loads(line) for line in gzip.open(args.source, "rt", encoding="utf-8")]
    best = [record for record in records if record["side"] == "best" and record.get("harms", 0) == 0]
    worst = [record for record in records if record["side"] == "worst"]
    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "best_without_harms": len(best),
        "best_with_harms_dropped": sum(
            1 for record in records if record["side"] == "best" and record.get("harms", 0) > 0
        ),
        "worst": len(worst),
    }

    if args.sft:
        with (args.out / "sft.jsonl").open("w", encoding="utf-8") as stream:
            for record in best:
                messages = trajectory(record["nodes"])
                stream.write(
                    json.dumps(
                        {
                            "id": example_id(record, "sft"),
                            "messages": messages,
                            "trainable_message_indices": [
                                index for index, message in enumerate(messages) if message["role"] == "assistant"
                            ],
                            "tools": tools(record),
                            "metadata": {
                                "task_name": record["task_name"],
                                "run": record["run"],
                                "step": record["step"],
                                "reward": record["reward"],
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        summary["sft_examples"] = len(best)
        summary["sft_tasks"] = len({record["task_name"] for record in best})

    if args.dpo:
        by_task: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"best": [], "worst": []})
        for record in best:
            by_task[record["task_name"]]["best"].append(record)
        for record in worst:
            by_task[record["task_name"]]["worst"].append(record)
        pairs = 0
        with (args.out / "dpo.jsonl").open("w", encoding="utf-8") as stream:
            for task, sides in sorted(by_task.items()):
                if not sides["best"] or not sides["worst"]:
                    continue
                # Harmful rejected episodes first: they carry the clearest "do not" signal.
                rejected_pool = sorted(sides["worst"], key=lambda record: (-record.get("harms", 0), record["reward"]))
                uses: dict[str, int] = defaultdict(int)
                for chosen, rejected in zip(  # noqa: B905 - pairs stop at the shorter side by design
                    itertools.chain.from_iterable(itertools.repeat(sides["best"], args.pairs_per_episode)),
                    itertools.chain.from_iterable(itertools.repeat(rejected_pool, args.pairs_per_episode)),
                ):
                    if uses[chosen["id"]] >= args.pairs_per_episode or uses[rejected["id"]] >= args.pairs_per_episode:
                        continue
                    chosen_messages, rejected_messages = trajectory(chosen["nodes"]), trajectory(rejected["nodes"])
                    prompt = [message for message in chosen_messages if message["role"] in {"system", "user"}][:2]
                    if [m for m in rejected_messages[: len(prompt)]] != prompt:
                        continue  # the two attempts must share the task prompt exactly
                    uses[chosen["id"]] += 1
                    uses[rejected["id"]] += 1
                    stream.write(
                        json.dumps(
                            {
                                "id": example_id(chosen, "dpo") + "--" + rejected["id"].lower()[:24],
                                "prompt": prompt,
                                "chosen": chosen_messages[len(prompt) :],
                                "rejected": rejected_messages[len(prompt) :],
                                "tools": tools(chosen),
                                "metadata": {
                                    "task_name": task,
                                    "chosen_reward": chosen["reward"],
                                    "rejected_reward": rejected["reward"],
                                    "rejected_harms": rejected.get("harms", 0),
                                },
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    pairs += 1
        summary["dpo_pairs"] = pairs
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
