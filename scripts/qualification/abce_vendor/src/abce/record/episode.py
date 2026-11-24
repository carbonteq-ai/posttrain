"""Recorded episodes as the environment stores them (one JSON line per episode).

`iter_episodes` never yields an excluded task. `Episode` is a thin read-only view with the
accessors the record builder needs.
"""

from __future__ import annotations

import glob
import gzip
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import paths
from ..tasks import is_sealed


@dataclass(frozen=True)
class Episode:
    record: dict
    path: str | None = None

    @property
    def trace(self) -> dict:
        return self.record["traces"][0]

    @property
    def id(self) -> str:
        return str(self.record.get("id") or self.trace.get("id"))

    @property
    def task(self) -> str:
        return self.trace["task"]["data"]["name"]

    @property
    def task_data(self) -> dict:
        return self.trace["task"]["data"]

    @property
    def messages(self) -> list[dict]:
        return [n["message"] for n in self.trace["nodes"] if n.get("message")]

    @property
    def nodes(self) -> list[dict]:
        return self.trace["nodes"]

    @property
    def events(self) -> list[dict]:
        return self.trace.get("tool_execution_events") or []

    @property
    def native_results(self) -> list[dict]:
        return ((self.trace.get("info") or {}).get("automationbench") or {}).get("assertions") or []

    @property
    def metrics(self) -> dict:
        return self.trace.get("metrics") or {}

    @property
    def success(self) -> bool:
        return self.metrics.get("task_completed_correctly") == 1.0

    @property
    def assessment_batches(self) -> list[dict]:
        return self.trace.get("assessment_batches") or self.record.get("assessment_batches") or []

    @property
    def stop_condition(self) -> str | None:
        return self.trace.get("stop_condition")

    @property
    def turn_progress(self) -> list[dict]:
        return (self.trace.get("info") or {}).get("automationbench_turn_progress") or []


def episode_files(pattern: str | None = None) -> list[str]:
    pattern = pattern or str(paths.EVIDENCE / "*/run/**/episodes.jsonl.gz")
    return sorted(glob.glob(pattern, recursive=True))


def iter_episodes(files: list[str] | None = None, *, tasks: set[str] | None = None,
                  limit: int | None = None) -> Iterator[Episode]:
    """Episodes of non-excluded tasks, optionally restricted to `tasks`."""
    count = 0
    for path in files if files is not None else episode_files():
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                traces = record.get("traces") or []
                if not traces or not traces[0].get("task"):
                    continue
                name = traces[0]["task"]["data"].get("name")
                if name is None or is_sealed(name):
                    continue
                if tasks is not None and name not in tasks:
                    continue
                yield Episode(record, str(path))
                count += 1
                if limit is not None and count >= limit:
                    return


def episode_census(files: list[str] | None = None) -> dict[str, Any]:
    """Counts over the evidence files (docs/progress.md M1 numbers)."""
    total = 0
    with_snapshots = 0
    with_batches = 0
    failed = 0
    tasks: dict[str, int] = {}
    for ep in iter_episodes(files):
        total += 1
        tasks[ep.task] = tasks.get(ep.task, 0) + 1
        if any(e.get("source") == "tool_server" and e.get("phase") in ("returned", "raised") for e in ep.events):
            with_snapshots += 1
        if ep.assessment_batches:
            with_batches += 1
        if not ep.success:
            failed += 1
    return {"files": len(files if files is not None else episode_files()), "episodes": total,
            "with_snapshots": with_snapshots, "with_batches": with_batches, "failed": failed,
            "tasks": len(tasks), "episodes_per_task": tasks}
