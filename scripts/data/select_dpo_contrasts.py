#!/usr/bin/env python3
"""Choose DPO episodes: perfect runs against runs that failed in a named way, per task.

Input is the TSV from ``episode_signals.py``. Only episodes whose harm status is known
are used, on both sides:

- ``--harm-results``: harm counts re-scored offline with the current manifests (JSON
  lines with ``id``, ``task``, ``harms`` and ``guarded``). Recorded harm flags are not
  trusted (some environment revisions under-counted), so in a guarded task only
  re-scored episodes are used;
- ``--unguarded-tasks``: tasks whose reviewed manifest defines no guard, so no action in
  them can be harmful.

Episodes of any other task (no manifest) are never used. A perfect episode passed the task with full partial credit, made no tool
mistakes, recorded no harm, and was not truncated or errored.

Each failed episode gets one category, the most specific that applies, in this order:

- A harmful: a manifest guard recorded a harm;
- F out-of-room: truncated, or ended at a turn, token or context limit;
- D tool-error: an unknown id, unknown tool or invalid arguments, and the task failed;
- E empty-results: acted on an empty lookup result, and the task failed;
- C near-miss: partial credit at least 0.8 without passing;
- B wrong-outcome: a clean run (no tool mistakes) with partial credit below 0.8.

A pair must share one setup (the system prompt and tool list fingerprint from
``episode_signals.py``), but may come from any training run or rollout group. So per task,
rejected episodes are drawn only from setups that have a chosen-eligible perfect episode,
and each setup gets enough perfect episodes for about ``--pairs-per-chosen`` pairs each.
Held-out tasks are never used. Picks rotate across training runs so one run does not dominate, and
are oversampled (``--oversample``) so later deduplication can still fill each slot.

Output TSV columns: trace_id, task, side (best = chosen, worst = rejected), category.

usage: select_dpo_contrasts.py SIGNALS.tsv --out SELECTION.tsv --exclude-tasks HELDOUT.txt
       [--categories ABCDEF] [--per-category 4] [--pairs-per-chosen 2]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

LIMIT_ENDINGS = {"turn_limit", "reply_token_limit", "token_budget", "context_limit_reply_cut", "context_rejected"}
CATEGORY_NAMES = {
    "A": "harmful",
    "F": "out-of-room",
    "D": "tool-error",
    "E": "empty-results",
    "C": "near-miss",
    "B": "wrong-outcome",
}


def number(row: dict, key: str) -> float:
    return float(row[key] or 0)


def flag(row: dict, key: str) -> bool:
    return row[key] in ("1", "True", "true")


def perfect(row: dict) -> bool:
    return (
        number(row, "pass") >= 1
        and number(row, "reward") >= 0.999
        and number(row, "mistakes") == 0
        and not flag(row, "harmful")
        and not flag(row, "truncated")
        and row["ending"] not in LIMIT_ENDINGS
    )


def category(row: dict) -> str | None:
    failed = number(row, "pass") < 1
    if flag(row, "harmful"):
        return "A"
    if flag(row, "truncated") or row["ending"] in LIMIT_ENDINGS:
        return "F"
    if failed and number(row, "unknown_id") + number(row, "unknown_tool") + number(row, "invalid_args") > 0:
        return "D"
    if failed and number(row, "empty_results") > 0:
        return "E"
    if failed and number(row, "reward") >= 0.8:
        return "C"
    if failed and number(row, "mistakes") == 0:
        return "B"
    return None


def across_runs(rows: list[dict], count: int, rng: random.Random, key=None) -> list[dict]:
    """Up to ``count`` rows, taking one per run in turn (each run's rows in ``key`` order)."""

    by_run: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_run[row["run"]].append(row)
    queues = []
    for run_rows in by_run.values():
        rng.shuffle(run_rows)
        if key:
            run_rows.sort(key=key)
        queues.append(run_rows)
    rng.shuffle(queues)
    picked: list[dict] = []
    while len(picked) < count and any(queues):
        for queue in queues:
            if queue and len(picked) < count:
                picked.append(queue.pop(0))
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("signals", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude-tasks", type=Path, required=True)
    parser.add_argument("--harm-results", type=Path, action="append", default=[])
    parser.add_argument("--unguarded-tasks", type=Path, help="tasks whose manifest defines no guard, one per line")
    parser.add_argument("--categories", default="ABCDEF")
    parser.add_argument("--per-category", type=int, default=4, help="rejected episodes per task and category")
    parser.add_argument("--pairs-per-chosen", type=int, default=2)
    parser.add_argument("--oversample", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    excluded = {line.strip() for line in args.exclude_tasks.read_text().splitlines() if line.strip()}
    rescored: dict[str, int] = {}
    for path in args.harm_results:
        for line in path.read_text().splitlines():
            result = json.loads(line)
            # Only exact replays (no tool-result or final-world mismatch) whose every guard ran.
            exact = result.get("fidelity") == [0, 0] and not result.get("abstained")
            if result.get("guarded") and result.get("harms") is not None and exact:
                if not result.get("unavailable_effects"):
                    rescored[result["id"]] = int(result["harms"])
    unguarded = set()
    if args.unguarded_tasks:
        unguarded = {line.strip() for line in args.unguarded_tasks.read_text().splitlines() if line.strip()}
    by_task: dict[str, list[dict]] = defaultdict(list)
    with args.signals.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if not row["task"] or row["task"] in excluded or flag(row, "has_error"):
                continue
            if row["trace_id"] in rescored:
                row["harmful"] = "1" if rescored[row["trace_id"]] > 0 else "0"
            elif row["task"] in unguarded:
                row["harmful"] = "0"
            else:
                continue  # harm status unknown
            by_task[row["task"]].append(row)

    selection: list[tuple[str, str, str, str]] = []
    counts: Counter[str] = Counter()
    tasks_per_category: dict[str, set[str]] = defaultdict(set)
    for task, rows in sorted(by_task.items()):
        chosen_pool = [row for row in rows if perfect(row)]
        if not chosen_pool:
            continue
        setups = {row["setup"] for row in chosen_pool}
        rejected: list[tuple[dict, str]] = []
        for letter in args.categories:
            pool = [row for row in rows if not perfect(row) and category(row) == letter and row["setup"] in setups]
            # Near misses: the closest misses first; elsewhere the lowest credit first.
            order = (lambda row: -number(row, "reward")) if letter == "C" else (lambda row: number(row, "reward"))
            picks = across_runs(pool, round(args.per_category * args.oversample), rng, key=order)
            rejected += [(row, letter) for row in picks]
            if picks:
                tasks_per_category[letter].add(task)
        if not rejected:
            continue
        per_setup = Counter(row["setup"] for row, _ in rejected)
        for setup, count in per_setup.items():
            pool = [row for row in chosen_pool if row["setup"] == setup]
            for row in across_runs(pool, math.ceil(count / args.pairs_per_chosen), rng):
                selection.append((row["trace_id"], task, "best", "chosen"))
                counts["chosen"] += 1
        for row, letter in rejected:
            selection.append((row["trace_id"], task, "worst", CATEGORY_NAMES[letter]))
            counts[CATEGORY_NAMES[letter]] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        stream.write("trace_id\ttask\tside\tcategory\n")
        for item in selection:
            stream.write("\t".join(item) + "\n")
    print(
        json.dumps(
            {
                "tasks": len({task for _, task, _, _ in selection}),
                "episodes": dict(counts),
                "tasks_per_category": {CATEGORY_NAMES[k]: len(v) for k, v in sorted(tasks_per_category.items())},
            }
        )
    )


if __name__ == "__main__":
    main()
