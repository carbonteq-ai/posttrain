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
  Records from a ``--selection`` harvest carry a failure ``category``: then each task's
  rejected episodes are taken per category (up to ``--per-category``, each used once) and
  each is paired with the least-used chosen episode of the same task (from any run)
  that shares its prompt and tool list, each chosen used at most ``--max-chosen-reuse`` times, so every failure kind is contrasted with a perfect run.

usage: trajectories_to_training_data.py IN.jsonl.gz --out DIR [--sft] [--dpo]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
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


def preference(chosen: dict, rejected: dict, prompt: list[dict], chosen_messages, rejected_messages) -> dict:
    return {
        "id": example_id(chosen, "dpo") + "--" + hashlib.sha256(rejected["id"].encode()).hexdigest()[:16],
        "prompt": prompt,
        "chosen": chosen_messages[len(prompt) :],
        "rejected": rejected_messages[len(prompt) :],
        "tools": tools(chosen),
        "metadata": {
            "task_name": chosen["task_name"],
            "category": rejected.get("category"),
            "chosen_reward": chosen["reward"],
            "rejected_reward": rejected["reward"],
            "rejected_harms": rejected.get("harms", 0),
        },
    }


def contrast_pairs(best: list[dict], worst: list[dict], path: Path, per_category: int, max_reuse: int) -> dict:
    """One pair per rejected episode: up to ``per_category`` per task and failure category."""

    chosen_by_task: dict[str, list[dict]] = defaultdict(list)
    for record in best:
        chosen_by_task[record["task_name"]].append(record)
    rejected_by_slot: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in worst:
        rejected_by_slot[(record["task_name"], record["category"])].append(record)
    messages = {record["id"]: trajectory(record["nodes"]) for record in [*best, *worst]}
    uses: dict[str, int] = defaultdict(int)
    by_category: dict[str, int] = defaultdict(int)
    unpaired = 0
    with path.open("w", encoding="utf-8") as stream:
        for (task, category), pool in sorted(rejected_by_slot.items()):
            # Near misses: the closest first; harmful: the most harms first; else the lowest credit.
            if category == "near-miss":
                pool.sort(key=lambda record: -record["reward"])
            elif category == "harmful":
                pool.sort(key=lambda record: (-record.get("harms", 0), record["reward"]))
            else:
                pool.sort(key=lambda record: record["reward"])
            made = 0
            for rejected in pool:
                if made >= per_category:
                    break
                rejected_messages = messages[rejected["id"]]
                for chosen in sorted(
                    chosen_by_task.get(task, []), key=lambda record: (uses[record["id"]], record["id"])
                ):
                    if uses[chosen["id"]] >= max_reuse:
                        continue
                    chosen_messages = messages[chosen["id"]]
                    prompt = [message for message in chosen_messages if message["role"] in {"system", "user"}][:2]
                    if rejected_messages[: len(prompt)] == prompt and tools(rejected) == tools(chosen):
                        break
                else:
                    unpaired += 1
                    continue
                uses[chosen["id"]] += 1
                made += 1
                by_category[category] += 1
                stream.write(
                    json.dumps(
                        preference(chosen, rejected, prompt, chosen_messages, rejected_messages), ensure_ascii=False
                    )
                    + "\n"
                )
    return {
        "dpo_pairs": sum(by_category.values()),
        "dpo_pairs_by_category": dict(sorted(by_category.items())),
        "dpo_tasks": len({task for task, _ in rejected_by_slot if task in chosen_by_task}),
        "max_chosen_reuse": max(uses.values(), default=0),
        "rejected_unpaired": unpaired,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sft", action="store_true")
    parser.add_argument("--dpo", action="store_true")
    parser.add_argument("--pairs-per-episode", type=int, default=2)
    parser.add_argument("--per-category", type=int, default=4, help="rejected episodes per task and category")
    parser.add_argument(
        "--max-chosen-reuse", type=int, default=4, help="pairs per chosen episode, so one success is not memorized"
    )
    parser.add_argument("--max-weight", type=float, default=4.0, help="cap on an SFT example's task weight")
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
        # Equal total weight per task: a task with many examples weighs each one less. The
        # weights average 1, so the overall loss scale is unchanged.
        per_task = defaultdict(int)
        for record in best:
            per_task[record["task_name"]] += 1
        mean_per_task = len(best) / max(len(per_task), 1)
        # Cap each example's weight so a task with one or two examples is not drawn so often
        # that the model memorizes it; renormalize so the weights still average 1.
        raw = {task: min(mean_per_task / count, args.max_weight) for task, count in per_task.items()}
        scale = len(best) / sum(raw[record["task_name"]] for record in best)
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
                                "sample_weight": round(raw[record["task_name"]] * scale, 6),
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        summary["sft_examples"] = len(best)
        summary["sft_tasks"] = len({record["task_name"] for record in best})

    if args.dpo and any("category" in record for record in records):
        summary.update(contrast_pairs(best, worst, args.out / "dpo.jsonl", args.per_category, args.max_chosen_reuse))
    elif args.dpo:
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
                # Distinct (chosen, rejected) combinations, diagonal first so every episode is
                # used before any is reused, each at most --pairs-per-episode times.
                best_side = sides["best"]
                combinations = sorted(
                    itertools.product(range(len(best_side)), range(len(rejected_pool))),
                    key=lambda pair: ((pair[1] - pair[0]) % len(rejected_pool), pair[0]),
                )
                for chosen_index, rejected_index in combinations:
                    chosen, rejected = best_side[chosen_index], rejected_pool[rejected_index]
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
                                "id": example_id(chosen, "dpo")
                                + "--"
                                + hashlib.sha256(rejected["id"].encode()).hexdigest()[:16],
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
