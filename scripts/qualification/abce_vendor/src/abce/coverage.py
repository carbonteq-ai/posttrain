"""Core-layer coverage over every design-target task (738), with or without recorded episodes.

For each task the static probe builds the native Items from the task's own assertions (the same
code path as for a recorded result) and exercises the grader on the task's initial world:

- `handler`: the handler runs on the initial world without raising (so flips can be probed);
- `initial`: the verdict on the untouched world (free assertions; negative checks that are not
  pure prohibitions show up as False here);
- `clause_probe`: the address-only variant of the assertion runs (so clause ablation is possible);
- `classified`: every parameter key is classified (no `unclassified_keys`);
- `polarity`: the rubric has a negative/positive marker for the type.

The per-shape table lists which of these hold, so a gap is a named row, never a silent one.
"""

from __future__ import annotations

import collections
from typing import Any

from .record.model import Item
from .record.native import ADDRESS_KEYS, CONTROL_KEYS, native_item
from .record.timeline import check
from .tasks import Task, assertion_shape, load_tasks, pool_names


def _initial_world(task: Task):
    from automationbench.schema.world import WorldState

    try:
        return WorldState(**task.initial_state)
    except Exception:  # noqa: BLE001
        return None


def static_probe_task(task: Task) -> list[dict[str, Any]]:
    world = _initial_world(task)
    rows = []
    for index, assertion in enumerate(task.assertions):
        result = {"type": assertion["type"], "passed": False, "excluded": False,
                  "params": {k: v for k, v in assertion.items() if k != "type"}}
        item: Item = native_item(index, result, None)
        initial = check(world, assertion) if world is not None else None
        address = {k: v for k, v in assertion.items() if k in ADDRESS_KEYS}
        params = {k: v for k, v in assertion.items() if k != "type" and k not in CONTROL_KEYS}
        clause_ok = None
        if address and world is not None:
            clause_ok = check(world, {"type": assertion["type"], **address}) is not None
            if not clause_ok:  # drop-one fallback (see probe.probe_clauses)
                clause_ok = any(check(world, {"type": assertion["type"], **{k: v for k, v in params.items() if k != key}})
                                is not None for key in params)
        rows.append({
            "task": task.name, "index": index, "shape": assertion_shape(assertion),
            "handler": initial is not None, "initial": initial, "clause_probe": clause_ok,
            "classified": not item.unclassified_keys, "unclassified": list(item.unclassified_keys),
            "polarity": item.polarity, "polarity_source": item.polarity_source,
            "effect": (item.effect.app, item.effect.object, item.effect.kind),
            "marker_name_disagreement": "marker_name_disagreement" in item.semantics,
            "world_loaded": world is not None,
        })
    return rows


def manifest_pairing(task: Task) -> dict[str, Any] | None:
    """How many of a task's native checks have a same-effect manifest check (static, by values)."""
    from .record.manifest import compile_contract, manifest_items
    from .record.merge import same_effect

    contract = task.contract()
    if not contract:
        return None
    from .record.manifest import _spec_values
    from .record.model import EffectRef, Item, Location

    specs = compile_contract(contract, task.initial_state, task.prompt)
    m_items = []
    for spec in specs:
        labels = list(spec["members"])[:50] or [None]
        polarity = {"harm": "forbid"}.get(spec["role"], "require")
        for label in labels:
            address, content, _meta = _spec_values(spec, None, label)
            m_items.append(Item(id=f"m:{spec['check_id']}:{label}", origin="manifest", type=spec["operator"],
                                outcome="abstained", polarity=polarity, polarity_source="manifest_role",
                                effect=spec["effect"], address=address, content=content, location=Location(kind="unknown")))
    n_items = []
    for index, assertion in enumerate(task.assertions):
        result = {"type": assertion["type"], "passed": False, "excluded": False,
                  "params": {k: v for k, v in assertion.items() if k != "type"}}
        n_items.append(native_item(index, result, None))
    paired = sum(1 for n in n_items if any(same_effect(n, m) and n.polarity == m.polarity for m in m_items))
    paired_checks = {m.id.split(":")[1] for m in m_items if any(same_effect(n, m) and n.polarity == m.polarity for n in n_items)}
    return {"native": len(n_items), "paired": paired, "manifest_checks": len(specs),
            "manifest_with_native": len(paired_checks)}


def _pairing_summary(tasks: dict[str, Task]) -> dict[str, int]:
    totals = collections.Counter()
    for task in tasks.values():
        p = manifest_pairing(task)
        if p is None:
            totals["tasks_without_manifest"] += 1
            continue
        totals["tasks"] += 1
        totals["native"] += p["native"]
        totals["native_paired"] += p["paired"]
        totals["manifest_checks"] += p["manifest_checks"]
        totals["manifest_paired"] += p["manifest_with_native"]
    return dict(totals)


def static_coverage(tasks: dict[str, Task] | None = None) -> dict[str, Any]:
    tasks = tasks or load_tasks()
    pool = pool_names()
    rows = [r for task in tasks.values() for r in static_probe_task(task)]
    by_shape: dict[tuple, dict] = {}
    for r in rows:
        s = by_shape.setdefault(r["shape"], {"type": r["shape"][0], "keys": list(r["shape"][1]), "instances": 0,
                                             "tasks": set(), "pool_tasks": set(), "handler_ok": 0, "initial_true": 0,
                                             "clause_probe_ok": 0, "clause_probe_na": 0, "classified": 0,
                                             "polarity": collections.Counter(), "disagree": 0})
        s["instances"] += 1
        s["tasks"].add(r["task"])
        if r["task"] in pool:
            s["pool_tasks"].add(r["task"])
        s["handler_ok"] += bool(r["handler"])
        s["initial_true"] += r["initial"] is True
        if r["clause_probe"] is None:
            s["clause_probe_na"] += 1
        else:
            s["clause_probe_ok"] += bool(r["clause_probe"])
        s["classified"] += bool(r["classified"])
        s["polarity"][r["polarity"]] += 1
        s["disagree"] += bool(r["marker_name_disagreement"])
    table = []
    for s in by_shape.values():
        table.append({**s, "tasks": len(s["tasks"]), "pool_tasks": len(s["pool_tasks"]), "polarity": dict(s["polarity"])})
    table.sort(key=lambda s: (-s["instances"], s["type"]))
    summary = {
        "tasks": len(tasks), "assertions": len(rows), "shapes": len(table),
        "worlds_loaded": sum(1 for t in tasks.values() if _initial_world(t) is not None),
        "handler_ok": sum(r["handler"] for r in rows),
        "handler_failed_shapes": sorted({r["shape"][0] for r in rows if not r["handler"]}),
        "initial_true": sum(r["initial"] is True for r in rows),
        "negative_marked_initial_false": sum(1 for r in rows if r["polarity"] == "forbid" and r["initial"] is False),
        "negative_initial_false_types": sorted({r["shape"][0] for r in rows if r["polarity"] == "forbid" and r["initial"] is False}),
        "clause_probe_ok": sum(1 for r in rows if r["clause_probe"]),
        "clause_probe_na": sum(1 for r in rows if r["clause_probe"] is None),
        "clause_probe_failed_shapes": sorted({r["shape"][0] for r in rows if r["clause_probe"] is False}),
        "unclassified_keys": dict(collections.Counter(k for r in rows for k in r["unclassified"])),
        "polarity_sources": dict(collections.Counter(r["polarity_source"] for r in rows)),
        "marker_name_disagreement_types": sorted({r["shape"][0] for r in rows if r["marker_name_disagreement"]}),
        "manifest_pairing": _pairing_summary(tasks),
        "pool_assertions": sum(1 for r in rows if r["task"] in pool),
        "pool_handler_ok": sum(1 for r in rows if r["task"] in pool and r["handler"]),
    }
    return {"summary": summary, "shapes": table}
