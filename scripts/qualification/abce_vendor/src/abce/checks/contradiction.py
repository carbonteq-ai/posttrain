"""Contradiction test at the Record level (docs/design.md §5.4, before any rendering exists).

A forbid item's offending object says "the attempt should not have produced this". If a
*successful* attempt at the same task produced an object with the same identity (same collection,
same address-like fields), then a statement built from that item would forbid something a correct
solution does — a contradiction. Objects are compared on identity fields only (recipient, channel,
subject, record id), not on free text, so paraphrased bodies do not hide a contradiction.
"""

from __future__ import annotations

import collections
import json
from typing import Any

from ..record import Episode, build_record, iter_episodes
from ..record.observed import normalize
from ..record.probe import _objects
from ..record.timeline import Timeline

IDENTITY_FIELDS = ("to", "cc", "recipients", "recipient", "channel_id", "channel", "user_id", "subject", "id",
                   "record_id", "row_id", "ticket_id", "conversation_id", "contact_id", "email", "spreadsheet_id",
                   "worksheet_id", "name", "title", "summary", "topic")


def _identity(collection: str, fields: dict[str, Any]) -> tuple:
    ident = tuple((k, json.dumps(fields[k], sort_keys=True)) for k in IDENTITY_FIELDS if k in fields and fields[k] not in ("", None, []))
    return (collection, ident)


def added_objects(episode: Episode) -> list[tuple[tuple, str]]:
    """(identity, normalised text) of every object a successful attempt added or changed."""
    tl = Timeline(episode)
    if not tl.available:
        return []
    from automationbench_v1.capture import world_patch

    initial, final = tl.json(tl.initial_digest), tl.json(tl.final_digest)
    if initial is None or final is None:
        return []
    patch = world_patch(initial, final)
    objects: list = []
    _objects(patch or {}, (), initial, objects)
    out = []
    for path, kind, key in objects:
        node: Any = final
        try:
            for step in path + (key,):
                node = node[step]
        except (KeyError, IndexError, TypeError):
            continue
        if isinstance(node, dict):
            out.append((_identity("/".join(str(p) for p in path), node), normalize(json.dumps(node, sort_keys=True))))
    return out


def contradiction_report(files: list[str] | None = None, *, limit: int | None = None) -> dict:
    """Per evidence file (siblings are the same file's episodes): two streaming passes, the first
    collecting successful attempts' object identities, the second checking failed attempts."""
    from ..record import episode_files

    checked = contradictions = 0
    rows = []
    tasks_with_successes: set[str] = set()
    for path in (files if files is not None else episode_files()):
        success_objects: dict[str, dict[tuple, list[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))
        for ep in iter_episodes([path], limit=limit):
            if ep.success:
                tasks_with_successes.add(ep.task)
                for ident, text in added_objects(ep):
                    if ident[1]:
                        success_objects[ep.task][ident].append(text)
        for ep in iter_episodes([path], limit=limit):
            if ep.success or ep.task not in success_objects:
                continue
            task = ep.task
            record = build_record(ep, probe=True)
            for item in record.items:
                obj = item.probe.offending_object
                if not obj or item.polarity not in ("forbid", "keep"):
                    continue
                checked += 1
                ident = _identity(obj["collection"], obj["fields"])
                # Same identity in a successful attempt, and (when the item names content) that
                # attempt's object carries the same content: the prohibition would forbid a correct act.
                forms = [f for v in item.content.values() for f in v.forms if f]
                matches = [t for t in success_objects[task].get(ident, []) if all(f in t for f in forms)]
                if ident[1] and matches:
                    contradictions += 1
                    rows.append({"task": task, "episode": ep.id, "item": item.id, "type": item.type,
                                 "object": obj["collection"], "identity": ident[1]})
    return {"tasks_with_successes": len(tasks_with_successes),
            "forbid_items_checked": checked, "contradictions": contradictions, "rows": rows}
