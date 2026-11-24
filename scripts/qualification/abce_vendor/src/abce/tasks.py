"""Task definitions of the design target: the fork's public tasks minus the excluded names.

The fork's datasets are loaded through its own loaders and cached as JSON (loading all seven
domains takes a few seconds). The excluded set (heldout62) is applied at load time and in every
consumer, so no excluded task is ever read or counted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from . import paths


@dataclass(frozen=True)
class Task:
    name: str
    domain: str
    prompt: list[dict]
    tools: tuple[str, ...]
    initial_state: dict
    assertions: tuple[dict, ...]
    manifest_path: str | None = None  # relative to contracts/

    @property
    def request(self) -> str:
        return next((m["content"] for m in reversed(self.prompt) if m.get("role") == "user"), "")

    @property
    def apps(self) -> tuple[str, ...]:
        return tuple(sorted(self.initial_state))

    def contract(self) -> dict | None:
        if not self.manifest_path:
            return None
        return json.loads((paths.CONTRACTS / self.manifest_path).read_text())


@lru_cache(maxsize=1)
def excluded_names() -> frozenset[str]:
    return frozenset(json.load(open(paths.EXCLUDED_NAMES))["task_names"])


@lru_cache(maxsize=1)
def pool_names() -> frozenset[str]:
    return frozenset(json.load(open(paths.POOL_NAMES))["task_names"])


def is_excluded(name: str) -> bool:
    return name in excluded_names()


@lru_cache(maxsize=1)
def sealed_names() -> frozenset[str]:
    """Tasks whose recorded episodes are never read: heldout62 and the new test (test60b, D66). The new test's tasks
    stay in the design target; only their episodes are sealed."""
    return excluded_names() | frozenset(json.load(open(paths.EXCLUDED_TEST))["task_names"])


def is_sealed(name: str) -> bool:
    return name in sealed_names()


def _load_from_fork() -> list[dict]:
    from automationbench.domains import DOMAINS  # the fork's loaders (read-only import)

    catalog = json.load(open(paths.CATALOG))["tasks"]
    rows = []
    for domain, loader in DOMAINS.items():
        for row in loader():
            name = row["task"]
            if is_excluded(name):
                continue
            info = json.loads(row["info"]) if isinstance(row["info"], str) else row["info"]
            rows.append(
                {
                    "name": name,
                    "domain": domain,
                    "prompt": row["prompt"],
                    "tools": list(info.get("zapier_tools", [])),
                    "initial_state": info.get("initial_state", {}),
                    "assertions": info.get("assertions", []),
                    "manifest_path": catalog.get(name),
                }
            )
    return rows


def _cache_file() -> Path:
    return paths.CACHE / "tasks.json"


@lru_cache(maxsize=1)
def load_tasks(refresh: bool = False) -> dict[str, Task]:
    """All design-target tasks by name (excluded names never appear)."""
    cache = _cache_file()
    if cache.exists() and not refresh:
        rows = json.load(open(cache))
    else:
        rows = _load_from_fork()
        cache.parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(cache, "w"))
    tasks = {}
    for row in rows:
        if is_excluded(row["name"]):
            continue
        tasks[row["name"]] = Task(
            name=row["name"],
            domain=row["domain"],
            prompt=row["prompt"],
            tools=tuple(row["tools"]),
            initial_state=row["initial_state"],
            assertions=tuple(row["assertions"]),
            manifest_path=row.get("manifest_path"),
        )
    return tasks


def get_task(name: str) -> Task:
    if is_excluded(name):
        raise KeyError(f"{name} is in the excluded set")
    return load_tasks()[name]


def assertion_params(assertion: dict) -> dict:
    return {k: v for k, v in assertion.items() if k != "type"}


def assertion_shape(assertion: dict) -> tuple[str, tuple[str, ...]]:
    return assertion["type"], tuple(sorted(assertion_params(assertion)))
