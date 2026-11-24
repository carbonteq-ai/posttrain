"""Minimal diagnostic rendering of a Record: one line per item, for inspecting the core layer.

This is not a teacher context (raw and plain renderers come later, after the recipe comparison);
it exists so humans and tests can read what the Record says.
"""

from __future__ import annotations

from ..record.model import Item, Record, Value


def _v(value: Value) -> str:
    raw = value.raw
    text = raw if isinstance(raw, str) else repr(raw)
    if len(text) > 60:
        text = text[:57] + "..."
    return text + ("" if value.observed else "†")  # † = never observed by the attempt


def item_line(item: Item) -> str:
    loc = item.location
    where = {"at_call": f"turn {loc.turn}", "after_turn": f"after turn {loc.turn}", "end_of_attempt": "end",
             "unknown": "?"}[loc.kind]
    parts = [f"[{item.polarity}/{item.outcome}]", item.type, f"@{where}"]
    if item.address:
        parts.append("to " + ", ".join(f"{k}={_v(v)}" for k, v in item.address.items()))
    if item.content:
        parts.append("with " + ", ".join(f"{k}={_v(v)}" for k, v in item.content.items()))
    p = item.probe
    if p.status != "pending":
        bits = [p.status]
        if p.failing_clauses:
            bits.append("failing=" + ",".join(p.failing_clauses))
        if p.required_keys:
            bits.append("missing=" + ",".join(p.required_keys))
        if p.offending_object:
            bits.append("offender=" + p.offending_object["collection"])
        if p.found_count is not None:
            bits.append(f"found={p.found_count}")
        parts.append("probe:" + " ".join(bits))
    if item.candidate:
        parts.append(f"for {item.candidate['label']}")
    if item.merged_from:
        parts.append("+" + ",".join(item.merged_from))
    if item.semantics.get("effect_required"):
        parts.append("(effect required)")
    if item.type == "tool_error":
        parts.append(item.semantics.get("error", "")[:80])
    return " ".join(parts)


def record_text(record: Record, *, failed_only: bool = True) -> str:
    lines = [f"# {record.task} {record.episode_id} success={record.success} turns={record.turns}"]
    for item in record.items:
        if failed_only and item.outcome not in ("failed",):
            continue
        lines.append(item_line(item))
    for note in record.notes:
        lines.append("note: " + note)
    return "\n".join(lines)
