"""Implication test: do manifest findings and native checks agree where they describe the same effect?

For every merged-or-disagreeing (native, manifest) pair across recorded episodes of a task, count
agreements and disagreements per manifest check id. A check whose disagreement rate exceeds the
threshold is reported; the engine renders such pairs unmerged (both statements, each attributed)
and never patches either side (docs/design.md §5.2, open-questions Q5).
"""

from __future__ import annotations

import collections
from typing import Any

from ..record import build_record, iter_episodes
from ..record.merge import _address_forms


def pair_outcomes(record) -> list[dict[str, Any]]:
    """(native item, manifest item, agree?) for every same-effect pair in one record."""
    pairs = []
    natives = [i for i in record.items if i.origin == "native" and i.type != "tool_error"]
    manifests = {i.id: i for i in record.items if i.origin == "manifest"}
    for n in natives:
        for mid in n.merged_from:
            pairs.append({"task": record.task, "check_id": mid.split(":")[1], "native_type": n.type, "agree": True})
    for note in record.notes:
        if note.startswith("disagreement"):
            parts = note.split()  # "disagreement native:<i> <outcome> vs manifest:<check>:<inst> <outcome>"
            pairs.append({"task": record.task, "check_id": parts[4].split(":")[1], "native_type": None, "agree": False})
    return pairs


def implication_report(files: list[str] | None = None, *, threshold: float = 0.2, limit: int | None = None) -> dict:
    by_check: dict[tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)
    episodes = 0
    for ep in iter_episodes(files, limit=limit):
        episodes += 1
        record = build_record(ep, probe=True, manifest=True)
        for pair in pair_outcomes(record):
            by_check[(pair["task"], pair["check_id"])]["agree" if pair["agree"] else "disagree"] += 1
    rows = []
    for (task, check_id), counts in sorted(by_check.items()):
        total = counts["agree"] + counts["disagree"]
        rate = counts["disagree"] / total if total else 0.0
        rows.append({"task": task, "check_id": check_id, "agree": counts["agree"], "disagree": counts["disagree"],
                     "disagreement_rate": rate, "flagged": rate > threshold and counts["disagree"] >= 2})
    return {"episodes": episodes, "pairs": sum(r["agree"] + r["disagree"] for r in rows),
            "checks": len(rows), "flagged": [r for r in rows if r["flagged"]], "rows": rows}


__all__ = ["implication_report", "pair_outcomes", "_address_forms"]
