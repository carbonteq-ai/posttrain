"""Census of the design target: the numbers in docs/survey.md, computed, not typed.

`census()` returns a dict of counts; `coverage_report()` adds, per native shape and per manifest
source adapter, whether the engine has what it needs (filled in by later milestones through
`registries`). The report is the acceptance instrument for every milestone.
"""

from __future__ import annotations

import collections
import json
from typing import Any

from .tasks import Task, assertion_shape, load_tasks, pool_names


def _walk_ops(node: Any, ops: collections.Counter) -> None:
    if isinstance(node, dict):
        op = node.get("op")
        if isinstance(op, str):
            ops[op] += 1
        for value in node.values():
            _walk_ops(value, ops)
    elif isinstance(node, list):
        for value in node:
            _walk_ops(value, ops)


def census(tasks: dict[str, Task] | None = None) -> dict:
    tasks = tasks or load_tasks()
    pool = pool_names()
    out: dict[str, Any] = {"tasks": len(tasks), "pool_tasks": sum(n in pool for n in tasks)}
    out["by_domain"] = dict(collections.Counter(t.domain for t in tasks.values()))
    out["without_manifest"] = sorted(n for n, t in tasks.items() if not t.manifest_path)
    out["with_manifest"] = len(tasks) - len(out["without_manifest"])
    out["pool_with_manifest"] = sum(bool(t.manifest_path) for n, t in tasks.items() if n in pool)

    types = collections.Counter()
    type_tasks = collections.Counter()
    shapes = set()
    pool_types = collections.Counter()
    pool_shapes = set()
    per_task = []
    for name, task in tasks.items():
        seen = set()
        per_task.append(len(task.assertions))
        for a in task.assertions:
            types[a["type"]] += 1
            seen.add(a["type"])
            shapes.add(assertion_shape(a))
            if name in pool:
                pool_types[a["type"]] += 1
                pool_shapes.add(assertion_shape(a))
        for t in seen:
            type_tasks[t] += 1
    out["assertions"] = sum(types.values())
    out["assertion_types"] = len(types)
    out["assertion_shapes"] = len(shapes)
    out["pool_assertions"] = sum(pool_types.values())
    out["pool_assertion_types"] = len(pool_types)
    out["pool_assertion_shapes"] = len(pool_shapes)
    out["types_absent_from_pool"] = len(set(types) - set(pool_types))
    out["singleton_task_types"] = sum(1 for t, c in type_tasks.items() if c == 1)
    out["assertions_per_task"] = {"min": min(per_task), "mean": sum(per_task) / len(per_task), "max": max(per_task)}

    tools = collections.Counter(t for task in tasks.values() for t in task.tools)
    out["tools"] = len(tools)
    out["pool_tools"] = len({t for n, task in tasks.items() if n in pool for t in task.tools})

    ops = collections.Counter()
    op_tasks = collections.Counter()
    roles = collections.Counter()
    adapters = collections.Counter()
    adapter_tasks = collections.Counter()
    exprs = collections.Counter()
    nchecks = 0
    for task in tasks.values():
        contract = task.contract()
        if not contract:
            continue
        seen_ops = set()
        for check in contract["checks"]:
            nchecks += 1
            ops[check["operator"]] += 1
            seen_ops.add(check["operator"])
            roles[check.get("role")] += 1
            _walk_ops(check, exprs)
        for op in seen_ops:
            op_tasks[op] += 1
        seen_ad = set()
        for source in contract.get("sources", {}).values():
            key = f"{source.get('adapter')}/{source.get('kind') or '-'}"
            adapters[key] += 1
            seen_ad.add(source.get("adapter"))
        for ad in seen_ad:
            adapter_tasks[ad] += 1
    out["manifest_checks"] = nchecks
    out["manifest_operators"] = dict(ops)
    out["manifest_operator_tasks"] = dict(op_tasks)
    out["manifest_roles"] = dict(roles)
    out["manifest_adapters"] = dict(adapters)
    out["manifest_adapter_tasks"] = dict(adapter_tasks)
    out["manifest_expression_ops"] = dict(exprs)
    return out


def shape_table(tasks: dict[str, Task] | None = None) -> list[dict]:
    """Every native (type, key-set) shape with its instance and task counts."""
    tasks = tasks or load_tasks()
    pool = pool_names()
    rows: dict[tuple, dict] = {}
    for name, task in tasks.items():
        for a in task.assertions:
            shape = assertion_shape(a)
            row = rows.setdefault(shape, {"type": shape[0], "keys": list(shape[1]), "instances": 0,
                                          "tasks": set(), "pool_tasks": set()})
            row["instances"] += 1
            row["tasks"].add(name)
            if name in pool:
                row["pool_tasks"].add(name)
    out = []
    for row in rows.values():
        out.append({**row, "tasks": len(row["tasks"]), "pool_tasks": len(row["pool_tasks"])})
    out.sort(key=lambda r: (-r["instances"], r["type"], r["keys"]))
    return out


def adapter_table(tasks: dict[str, Task] | None = None) -> list[dict]:
    tasks = tasks or load_tasks()
    rows = collections.Counter()
    for task in tasks.values():
        contract = task.contract()
        if not contract:
            continue
        for source in contract.get("sources", {}).values():
            rows[(source.get("adapter"), source.get("kind"))] += 1
    return [{"adapter": a, "kind": k, "sources": c} for (a, k), c in rows.most_common()]


def format_census(c: dict) -> str:
    lines = [
        f"tasks {c['tasks']} (pool {c['pool_tasks']}); with manifest {c['with_manifest']}; without: {', '.join(c['without_manifest'])}",
        f"assertions {c['assertions']} / types {c['assertion_types']} / shapes {c['assertion_shapes']}; "
        f"pool {c['pool_assertions']} / {c['pool_assertion_types']} / {c['pool_assertion_shapes']}; "
        f"types absent from pool {c['types_absent_from_pool']}; singleton-task types {c['singleton_task_types']}",
        f"tools {c['tools']} (pool {c['pool_tools']})",
        f"manifest checks {c['manifest_checks']}; operators {json.dumps(c['manifest_operators'], sort_keys=True)}",
        f"roles {json.dumps(c['manifest_roles'], sort_keys=True)}",
    ]
    return "\n".join(lines)
