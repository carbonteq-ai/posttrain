"""Merge native and manifest Items that describe the same effect (docs/design.md §2).

Two items merge when they have the same polarity, the same effect app, both are decided
(passed/failed) with the same outcome, and share at least one address value (normalised forms
intersect). The native item is kept as the primary (its probe carries the location); the
manifest item's evidence and candidate are attached. Disagreeing pairs (same effect+address,
different outcome) are not merged; they are recorded for the implication test, and the manifest item
is marked `semantics["native_disagrees"]` (the native check's id): the grader decides success, so
where its own check of the same effect disagrees, the manifest's reading is unsure and is not rendered.
A manifest item that asks for more on one address field than the native check names (one message to
the employee *and* the manager, where the check names the employee) is a different requirement: it is
neither merged nor contradicted (`asks_more_on_a_shared_field`).
"""

from __future__ import annotations

from .model import Item


def _forms(values) -> set[str]:
    forms = set()
    for value in values:
        for f in value.forms:
            if f and len(f) >= 3:
                forms.add(f)
    return forms


def _address_forms(item: Item) -> set[str]:
    return _forms(item.address.values())


def _content_forms(item: Item) -> set[str]:
    return _forms(item.content.values())


def same_effect(n: Item, m: Item) -> bool:
    """Same app, a shared address value, and — when both name content — a shared content value.
    A recipient alone is not enough: many checks of one task address the same person."""
    if m.effect.app != n.effect.app or not (_address_forms(n) & _address_forms(m)):
        return False
    if n.content and m.content:
        return bool(_content_forms(n) & _content_forms(m))
    return not n.content and not m.content


def asks_more_on_a_shared_field(m: Item, n: Item) -> bool:
    """The manifest item names several values on one address field (`recipients`, `recipients#1`:
    one message to the employee and the manager) and the native check names one of them: the
    native check deciding that the one message was sent does not decide the goal. Such a pair is
    neither a merge nor a disagreement."""
    native = _address_forms(n)
    for key, value in m.address.items():
        if "#" not in key:
            continue
        first = m.address.get(key.split("#")[0])
        if first is not None and (_forms([first]) & native) and not (_forms([value]) <= native):
            return True
    return False


def merge_items(items: list[Item]) -> tuple[list[Item], list[dict]]:
    native = [i for i in items if i.origin == "native" and i.type != "tool_error"]
    manifest = [i for i in items if i.origin == "manifest"]
    others = [i for i in items if i.type == "tool_error"]
    used: set[str] = set()
    merged: list[Item] = []
    disagreements: list[dict] = []
    disagreeing: dict[str, str] = {}
    for n in native:
        partner = None
        for m in manifest:
            if m.id in used or m.polarity != n.polarity or not same_effect(n, m):
                continue
            if asks_more_on_a_shared_field(m, n):
                continue  # a different requirement: neither merged into the native item nor contradicted by it
            if m.outcome in ("passed", "failed") and n.outcome in ("passed", "failed") and m.outcome != n.outcome:
                disagreements.append({"native": n.id, "manifest": m.id, "native_outcome": n.outcome,
                                      "manifest_outcome": m.outcome})
                disagreeing.setdefault(m.id, n.id)
                continue
            if m.outcome == n.outcome:
                partner = m
                break
        if partner is None:
            merged.append(n)
            continue
        used.add(partner.id)
        update = {"evidence": tuple(n.evidence) + tuple(partner.evidence), "merged_from": (partner.id,),
                  "candidate": n.candidate or partner.candidate,
                  "semantics": {**n.semantics, "manifest": partner.semantics}}
        if n.location.kind == "unknown" and partner.location.kind != "unknown":
            update["location"] = partner.location
        merged.append(n.model_copy(update=update))
    # the loop above stops at a native item's first partner; a later manifest item can still disagree with it
    decided = ("passed", "failed")
    for m in manifest:
        if m.id in used or m.id in disagreeing or m.outcome not in decided:
            continue
        for n in native:
            if (n.polarity == m.polarity and n.outcome in decided and n.outcome != m.outcome and same_effect(n, m)
                    and not asks_more_on_a_shared_field(m, n)):
                disagreements.append({"native": n.id, "manifest": m.id, "native_outcome": n.outcome,
                                      "manifest_outcome": m.outcome})
                disagreeing[m.id] = n.id
                break
    merged.extend(m.model_copy(update={"semantics": {**m.semantics, "native_disagrees": disagreeing[m.id]}})
                  if m.id in disagreeing else m for m in manifest if m.id not in used)
    merged.extend(others)
    return merged, disagreements
