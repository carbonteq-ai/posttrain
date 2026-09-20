"""Sample complete training traces from one Trackio run without leaking judge oracles.

The sampler first reads bounded trace metadata, selects task types round-robin
across domains, then fetches complete native payloads only for the selected
trace IDs.  Each selected task contributes a fixed number of trajectories,
preferring the earliest and latest optimizer steps available for that task.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from posttrain.tracking import TraceQuery, TraceRecord
from posttrain_tracking_trackio import TrackioDataSource


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--project", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--provider-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-types", type=int, default=25)
    parser.add_argument("--per-task", type=int, default=2)
    parser.add_argument("--fetch-concurrency", type=int, default=12)
    args = parser.parse_args()
    if args.task_types < 1 or args.per_task < 1 or args.fetch_concurrency < 1:
        parser.error("task-types, per-task, and fetch-concurrency must be positive")
    return args


def _stable_rank(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


async def _metadata(source: TrackioDataSource, provider_run_id: str) -> list[TraceRecord]:
    records: list[TraceRecord] = []
    cursor: str | None = None
    while True:
        page = await source.traces_by_provider_run_id(
            provider_run_id,
            TraceQuery(cursor=cursor, limit=1000, include_payload=False),
        )
        records.extend(page.items)
        if page.next_cursor is None:
            return records
        cursor = page.next_cursor


def _select(records: list[TraceRecord], *, task_types: int, per_task: int) -> list[TraceRecord]:
    eligible = [
        record
        for record in records
        if record.trace_type == "verifiers"
        and record.attributes.get("is_truncated") is False
        and record.attributes.get("has_error") is False
        and isinstance(record.attributes.get("task_index"), int)
        and isinstance(record.attributes.get("domain"), str)
    ]
    by_task: dict[tuple[str, int], list[TraceRecord]] = defaultdict(list)
    for record in eligible:
        key = (str(record.attributes["domain"]), int(record.attributes["task_index"]))
        by_task[key].append(record)
    candidates: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key, rows in by_task.items():
        if len(rows) >= per_task:
            candidates[key[0]].append(key)
    for domain, keys in candidates.items():
        keys.sort(key=lambda key: (_stable_rank(f"{domain}:{key[1]}"), key[1]))

    chosen_tasks: list[tuple[str, int]] = []
    domains = sorted(candidates)
    while len(chosen_tasks) < task_types and domains:
        remaining: list[str] = []
        for domain in domains:
            if candidates[domain] and len(chosen_tasks) < task_types:
                chosen_tasks.append(candidates[domain].pop(0))
            if candidates[domain]:
                remaining.append(domain)
        domains = remaining
    if len(chosen_tasks) != task_types:
        raise ValueError(f"only {len(chosen_tasks)} task types have {per_task} eligible traces")

    selected: list[TraceRecord] = []
    for key in chosen_tasks:
        rows = sorted(
            by_task[key],
            key=lambda row: (
                int(row.attributes.get("optimizer_step", -1)),
                int(row.attributes.get("rollout_ordinal", -1)),
                row.external_id,
            ),
        )
        if per_task == 1:
            picked = [rows[-1]]
        else:
            picked = [rows[0], rows[-1]]
            remaining = [row for row in rows if row.external_id not in {item.external_id for item in picked}]
            while len(picked) < per_task:
                target = round((len(rows) - 1) * len(picked) / (per_task - 1))
                picked.append(min(remaining, key=lambda row: abs(rows.index(row) - target)))
                remaining.remove(picked[-1])
        selected.extend(picked)
    return selected


async def _main(args: argparse.Namespace) -> None:
    source = TrackioDataSource(args.project, server_url=args.server_url)
    metadata = await _metadata(source, args.provider_run_id)
    selected = _select(metadata, task_types=args.task_types, per_task=args.per_task)
    semaphore = asyncio.Semaphore(args.fetch_concurrency)

    async def fetch(record: TraceRecord) -> tuple[TraceRecord, TraceRecord]:
        async with semaphore:
            complete = await source.get_trace(args.run_id, record.external_id)
        if complete is None:
            raise ValueError(f"selected trace disappeared: {record.external_id}")
        return record, complete

    complete = await asyncio.gather(*(fetch(record) for record in selected))
    args.output.mkdir(parents=True, exist_ok=False)
    traces_path = args.output / "native-traces.jsonl"
    traces_path.write_text("".join(json.dumps(item.payload, ensure_ascii=False) + "\n" for _, item in complete))
    rows: list[dict[str, Any]] = []
    for metadata_record, full_record in complete:
        payload = full_record.payload
        rows.append(
            {
                "trace_id": full_record.external_id,
                "domain": metadata_record.attributes["domain"],
                "task_index": metadata_record.attributes["task_index"],
                "example_id": metadata_record.attributes.get("example_id"),
                "optimizer_step": metadata_record.attributes.get("optimizer_step"),
                "rollout_ordinal": metadata_record.attributes.get("rollout_ordinal"),
                "is_truncated": metadata_record.attributes.get("is_truncated"),
                "has_error": metadata_record.attributes.get("has_error"),
                "task_name": payload.get("task", {}).get("data", {}).get("task_name"),
                "native_rewards": payload.get("rewards", {}),
                "native_metrics": payload.get("metrics", {}),
                "stop_condition": payload.get("stop_condition"),
                "node_count": len(payload.get("nodes", ())),
                "tool_call_count": len(payload.get("calls", ())),
            }
        )
    (args.output / "sample-index.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    manifest = {
        "project": args.project,
        "server_url": args.server_url,
        "provider_run_id": args.provider_run_id,
        "run_id": args.run_id,
        "population_trace_count": len(metadata),
        "eligibility": {"trace_type": "verifiers", "is_truncated": False, "has_error": False},
        "sampling": {
            "task_types": args.task_types,
            "per_task": args.per_task,
            "domain_balancing": "round_robin",
            "within_task": "earliest_and_latest_optimizer_step_then_stable_ordinal",
        },
        "selected_trace_count": len(rows),
        "selected_trace_ids": [row["trace_id"] for row in rows],
    }
    (args.output / "sample-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    asyncio.run(_main(_arguments()))
