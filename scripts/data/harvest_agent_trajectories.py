#!/usr/bin/env python3
"""Select the best and worst AutomationBench training episodes per task from Trackio's Doris store.

Runs read-only SQL against the Trackio Doris database (connection from the
TRACKIO_DORIS_* environment variables, as the Trackio server is configured). For each
task it ranks the training episodes of the selected runs and keeps:

- best: highest partial credit (``fact_task_reward``) at or above ``--best-min``,
  fewest output tokens first, not truncated or errored; for SFT and as DPO "chosen";
- worst: lowest partial credit at or below ``--worst-max``, not truncated or errored;
  for DPO "rejected".

Held-out evaluation tasks (``--exclude-tasks``, one name per line) are never selected.
Only the selected payloads are read. Each output line is a compact Verifiers-style
episode: task name, tools, nodes (messages with reasoning and tool calls), metrics, and
the number of verified manifest harms (guards whose outcome value is 1), plus its source
run, step and reward.

Diversity: picks within a task rotate across runs (the best of each run first), and a
trajectory with the same sequence of tool calls as one already kept is dropped.

With ``--selection`` (the TSV from ``select_dpo_contrasts.py``) it skips ranking and
fetches exactly the listed episodes, keeping each one's side and failure category; only
rejected episodes repeating another's tool calls are dropped.

usage: harvest_agent_trajectories.py --out DIR [--runs-like PATTERN ...] [--per-task 8]
       harvest_agent_trajectories.py --out DIR --selection SELECTION.tsv
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

import pymysql


def connect() -> pymysql.connections.Connection:
    return pymysql.connect(
        host=os.environ["TRACKIO_DORIS_HOST"],
        port=int(os.environ.get("TRACKIO_DORIS_PORT", "9030")),
        user=os.environ["TRACKIO_DORIS_USER"],
        password=os.environ["TRACKIO_DORIS_PASSWORD"],
        database=os.environ.get("TRACKIO_DORIS_DATABASE", "trackio"),
        read_timeout=900,
        connect_timeout=10,
    )


SELECT_SQL = """
SELECT trace_id, run_name, fact_task_id, fact_task_reward, fact_rollout_step, fact_model_output_tokens, side FROM (
  SELECT *, ROW_NUMBER() OVER (
           PARTITION BY fact_task_id, side
           ORDER BY rank_in_run, CASE WHEN side = 'best' THEN -fact_task_reward ELSE fact_task_reward END,
                    fact_model_output_tokens) AS rank_in_task
  FROM (
    SELECT trace_id, run_name, fact_task_id, fact_task_reward, fact_rollout_step, fact_model_output_tokens, side,
           ROW_NUMBER() OVER (
             PARTITION BY fact_task_id, side, run_name
             ORDER BY CASE WHEN side = 'best' THEN -fact_task_reward ELSE fact_task_reward END,
                      fact_model_output_tokens) AS rank_in_run
    FROM (
      SELECT trace_id, run_name, fact_task_id, fact_task_reward, fact_rollout_step, fact_model_output_tokens,
             CASE WHEN fact_task_reward >= %(best_min)s THEN 'best' ELSE 'worst' END AS side
      FROM traces
      WHERE project_id = %(project)s AND trace_type = 'verifiers' AND run_name LIKE 'train.%%'
        AND fact_model LIKE %(model)s AND ({runs})
        AND NOT COALESCE(fact_is_truncated, FALSE) AND NOT COALESCE(fact_has_error, FALSE)
        AND (fact_task_reward >= %(best_min)s OR fact_task_reward <= %(worst_max)s)
    ) candidates
  ) by_run
) ranked
WHERE rank_in_task <= %(oversample)s
"""


def harms(payload: dict) -> int:
    """Verified harmful actions: complete guard assessments with outcome value 1, counted once."""

    seen: set[str] = set()
    count = 0
    # Training traces keep results only: manifest runs record harms per turn in their turn evidence.
    evidence = ((payload.get("info") or {}).get("posttrain_turn_rewards") or {}).get("assessments") or []
    recorded = [
        component.get("value") or 0
        for turn in evidence
        for component in turn.get("components") or []
        if component.get("name") == "manifest_harms"
    ]
    if recorded:
        return int(sum(recorded))
    for trace in payload.get("traces") or [payload]:
        for batch in trace.get("assessment_batches") or []:
            assessments = batch.get("assessments")
            if not isinstance(assessments, list):
                continue
            for assessment in assessments:
                if assessment.get("assessment_id") in seen:
                    continue
                seen.add(assessment.get("assessment_id"))
                signal = assessment.get("signal") or {}
                if (
                    assessment.get("status") == "valid"
                    and signal.get("semantics") == "outcome"
                    and signal.get("direction") == "lower"
                    and float(assessment.get("value") or 0) >= 1
                ):
                    count += 1
    return count


def tool_signature(record: dict) -> str:
    """The trajectory's sequence of (tool, arguments) calls, for dropping duplicates."""

    calls = []
    for node in record["nodes"]:
        message = node.get("message") or {}
        for call in message.get("tool_calls") or []:
            function = call.get("function") or call
            calls.append([function.get("name"), function.get("arguments")])
    return json.dumps(calls, sort_keys=True)


def compact(payload: dict, row: tuple) -> dict:
    trace_id, run_name, task, reward, step, output_tokens, side = row
    return {
        "id": trace_id,
        "side": side,
        "task_name": task,
        "run": run_name.removeprefix("train."),
        "step": step,
        "reward": reward,
        "output_tokens": output_tokens,
        "harms": harms(payload),
        "metrics": payload.get("metrics") or {},
        "tools": payload.get("tools") or [],
        "nodes": [
            {key: node.get(key) for key in ("message", "sampled", "parent") if key in node}
            for node in payload.get("nodes") or []
        ],
        "is_truncated": payload.get("is_truncated", False),
        "ok": payload.get("ok", True),
    }


def harvest_selection(connection: pymysql.connections.Connection, args: argparse.Namespace) -> None:
    lines = args.selection.read_text().splitlines()[1:]
    wanted = {trace_id: (side, category) for trace_id, _task, side, category in (line.split("\t") for line in lines)}
    ids = list(wanted)
    args.out.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    signatures: set[tuple[str, str, str]] = set()
    duplicates = 0
    with gzip.open(args.out / "trajectories.jsonl.gz", "wt", encoding="utf-8") as stream, connection.cursor() as cursor:
        for start in range(0, len(ids), 50):
            cursor.execute(
                "SELECT trace_id, run_name, fact_task_id, fact_task_reward, fact_rollout_step,"
                " fact_model_output_tokens, payload FROM traces WHERE project_id = %s AND trace_id IN %s",
                (args.project, tuple(ids[start : start + 50])),
            )
            for *row, raw in cursor.fetchall():
                side, category = wanted[row[0]]
                payload = json.loads(raw)
                record = compact(payload, (*row, side))
                record["task_name"] = ((payload.get("task") or {}).get("data") or {}).get("task_name") or row[2]
                record["category"] = category
                signature = (record["task_name"], category, tool_signature(record))
                # Perfect runs with the same tool calls still differ in their reasoning: keep them.
                if side == "worst" and signature in signatures:
                    duplicates += 1
                    continue
                signatures.add(signature)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                written[category] = written.get(category, 0) + 1
    summary = {"selected": len(ids), "written": written, "duplicates_dropped": duplicates}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--project", default="posttrain-lab")
    parser.add_argument("--model", default="%LFM2.5%")
    parser.add_argument("--runs-like", action="append", default=[], help="SQL LIKE pattern on run_name")
    parser.add_argument("--exclude-tasks", type=Path)
    parser.add_argument("--per-task", type=int, default=8)
    parser.add_argument(
        "--domain-cap",
        action="append",
        default=[],
        metavar="DOMAIN=N",
        help="lower per-task limit for one task domain, e.g. simple=3, so easy tasks do not dominate",
    )
    parser.add_argument("--selection", type=Path, help="fetch exactly these episodes (select_dpo_contrasts.py)")
    parser.add_argument("--best-min", type=float, default=0.9)
    parser.add_argument("--worst-max", type=float, default=0.3)
    args = parser.parse_args()

    domain_caps = {item.split("=", 1)[0]: int(item.split("=", 1)[1]) for item in args.domain_cap}
    excluded = set()
    if args.exclude_tasks:
        excluded = {line.strip() for line in args.exclude_tasks.read_text().splitlines() if line.strip()}
    patterns = args.runs_like or ["train.%"]
    runs = " OR ".join("run_name LIKE %({})s".format(f"run{index}") for index in range(len(patterns)))
    params = {
        "project": args.project,
        "model": args.model,
        "best_min": args.best_min,
        "worst_max": args.worst_max,
        # Twice the target, so duplicate trajectories can be dropped and still fill each task.
        "oversample": args.per_task * 2,
        **{f"run{index}": pattern for index, pattern in enumerate(patterns)},
    }
    connection = connect()
    if args.selection:
        harvest_selection(connection, args)
        return
    with connection.cursor() as cursor:
        cursor.execute(SELECT_SQL.format(runs=runs), params)
        rows = [row for row in cursor.fetchall() if row[2] not in excluded]
    args.out.mkdir(parents=True, exist_ok=True)
    written = {"best": 0, "worst": 0}
    kept: dict[tuple[str, str], int] = {}
    signatures: set[tuple[str, str, str]] = set()
    duplicates = 0
    with gzip.open(args.out / "trajectories.jsonl.gz", "wt", encoding="utf-8") as stream, connection.cursor() as cursor:
        for start in range(0, len(rows), 50):
            chunk = rows[start : start + 50]
            cursor.execute(
                "SELECT trace_id, payload FROM traces WHERE project_id = %s AND trace_id IN %s",
                (args.project, tuple(row[0] for row in chunk)),
            )
            payloads = dict(cursor.fetchall())
            for row in chunk:
                raw = payloads.get(row[0])
                task, side = row[2], row[6]
                if not raw or kept.get((task, side), 0) >= args.per_task:
                    continue
                payload = json.loads(raw)
                # Older runs record a hashed task id; the payload always names the task.
                task = ((payload.get("task") or {}).get("data") or {}).get("task_name") or task
                limit = domain_caps.get(task.split(".", 1)[0], args.per_task)
                if task in excluded or kept.get((task, side), 0) >= limit:
                    continue
                record = compact(payload, row)
                record["task_name"] = task
                signature = (task, side, tool_signature(record))
                if signature in signatures:
                    duplicates += 1
                    continue
                signatures.add(signature)
                kept[(task, side)] = kept.get((task, side), 0) + 1
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                written[side] += 1
    best_tasks = {task for (task, side) in kept if side == "best"}
    worst_tasks = {task for (task, side) in kept if side == "worst"}
    summary = {
        "candidates": len(rows),
        "written": written,
        "duplicates_dropped": duplicates,
        "tasks_with_best": len(best_tasks),
        "tasks_with_worst": len(worst_tasks),
        "tasks_with_a_pair": len(best_tasks & worst_tasks),
        "tasks_with_only_worst": sorted(worst_tasks - best_tasks),
        "excluded_tasks": len(excluded),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
