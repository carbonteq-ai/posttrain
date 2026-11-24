"""Build a Record from an Episode: native items (M2), probing and location (M3)."""

from __future__ import annotations

import json
from typing import Any

from .calls import episode_calls
from .episode import Episode
from .model import Call, CallRef, EffectRef, Item, Location, ProbeResult, Record, Value
from .native import native_item
from .observed import Corpus

TURN_LIMIT_MARKERS = ("max_turns", "turn_limit", "max_steps", "budget")


def initial_pass_flags(episode: Episode) -> list[bool | None]:
    """Whether each native assertion already passed on the initial world (the free-assertion rule)."""
    try:
        import automationbench.rubric  # noqa: F401
        from automationbench.rubric.registry import AssertionRegistry
        from automationbench.schema.world import WorldState
    except ImportError:
        return [None] * len(episode.native_results)
    initial = episode.task_data.get("initial_state")
    assertions = episode.task_data.get("assertions") or []
    if not initial or len(assertions) != len(episode.native_results):
        return [None] * len(episode.native_results)
    try:
        world = WorldState(**initial)
    except Exception:  # noqa: BLE001
        return [None] * len(episode.native_results)
    flags: list[bool | None] = []
    for assertion in assertions:
        try:
            flags.append(bool(AssertionRegistry.check(world, assertion)))
        except Exception:  # noqa: BLE001
            flags.append(None)
    return flags


def tool_error_items(calls: list[Call]) -> list[Item]:
    items = []
    for call in calls:
        if call.error is None:
            continue
        items.append(Item(
            id=f"tool_error:{call.turn}:{call.index}", origin="native", type="tool_error", outcome="failed",
            polarity="require", polarity_source="call", effect=EffectRef(app=call.app, object="call", kind="tool_error"),
            location=Location(kind="at_call", turn=call.turn, call=CallRef(turn=call.turn, index=call.index), source="cause"),
            semantics={"error": call.error, "tool": call.tool},
            evidence=(f"call:{call.turn}:{call.index}",),
        ))
    return items


def locate(item: Item, probe: ProbeResult, calls_by_invocation: dict[str, Call]) -> Location:
    """Location from probe results (docs/design.md §2)."""
    def at(invocation: str, kind: str, source: str) -> Location | None:
        call = calls_by_invocation.get(invocation)
        if call is None:
            return None
        return Location(kind=kind, turn=call.turn, call=CallRef(turn=call.turn, index=call.index), source=source)

    if item.outcome == "failed" and item.polarity in ("forbid", "keep"):
        if probe.reason and "negative_check_requires_effect" in probe.reason:
            # the effect exists and a located call gave it the forbidden content (located only when
            # ablation isolated the object that call produced)
            if probe.offending_invocation and probe.offending_object is not None:
                loc = at(probe.offending_invocation, "at_call", "flip")
                if loc is not None:
                    return loc
            return Location(kind="end_of_attempt", source="unmet")
        for inv in probe.flips:  # the first pass->fail flip is the offending call
            loc = at(inv, "at_call", "flip")
            if loc is not None and probe.flip_direction == "pass_to_fail":
                return loc
        return Location(kind="unknown", source="no_flip")
    if item.outcome == "failed":
        return Location(kind="end_of_attempt", source="unmet")
    if item.outcome == "passed" and item.polarity == "require" and probe.flips:
        loc = at(probe.flips[0], "after_turn", "witness")
        if loc is not None:
            return loc
    return Location(kind="unknown")


def mark_present_effects(items: list[Item], timeline) -> list[Item]:
    """A failed manifest goal whose effect, as a line would state it (address and content), is already
    carried by an object the attempt added (or changed, for a goal about an existing record) is
    marked `semantics["effect_present"]`. Such a goal failed for a reason the engine cannot state (a
    condition it could not express, a stricter comparison than the line's words), so no "Not done"
    line may claim the effect is missing. Read goals leave no object and are not judged here."""
    from .probe import added_objects, effect_present, item_requirements

    if not timeline.available:
        return items
    initial_json, final_json = timeline.json(timeline.initial_digest), timeline.json(timeline.final_digest)
    if initial_json is None or final_json is None:
        return items
    texts = None
    out = []
    for item in items:
        if not (item.origin == "manifest" and item.outcome == "failed" and item.polarity == "require"
                and item.location.kind != "at_call" and item.effect.kind != "read"):
            out.append(item)
            continue
        if texts is None:
            texts = added_objects(initial_json, final_json)
        address, required, negated = item_requirements(item)
        # a sent message or a created record must be new; a goal about an existing record (update,
        # field, exist) reads the records the attempt changed as well
        new_only = item.effect.kind in ("send", "create", "post")
        present = effect_present(texts, address, required, negated, app=item.effect.app,
                                 collection=item.semantics.get("collection") or item.effect.object,
                                 world=final_json, new_only=new_only)
        # the identity alone (an object the attempt produced with every address value): "Not met" rather than
        # "Not done" when the line could not state the whole condition
        # the identity must be the candidate's own: its label (the contact, the vendor) is on the object too,
        # else another candidate's object would pass as this one's
        label = (item.candidate or {}).get("label")
        with_candidate = address + ([label] if isinstance(label, (str, int)) and str(label).strip() else [])
        address_present = bool(address) and effect_present(texts, with_candidate, [], [], app=item.effect.app,
                                                             collection=item.semantics.get("collection") or item.effect.object,
                                                             world=final_json, new_only=new_only)
        named = None
        if address_present or present:
            # the object itself (fix round, not_met): the first added/changed object of the effect's collection that
            # carries the identity (and the candidate's label); its own id names it on the line
            for view in texts:
                if effect_present([view], with_candidate, [], [], app=item.effect.app,
                                  collection=item.semantics.get("collection") or item.effect.object,
                                  world=final_json, new_only=new_only):
                    fields = view.get("fields") or {}
                    oid = next((fields[k] for k in ("id", "row_id", "message_id", "ts") if fields.get(k) not in (None, "", "null")), None)
                    if oid is not None:
                        named = {"id": str(oid), "path": view.get("path"), "kind": view.get("kind")}
                    break
        out.append(item.model_copy(update={"semantics": {**item.semantics, "effect_present": present,
                                                         "address_present": address_present, "present_object": named}}))
    return out


def _sheet_row(initial_state: dict | None, address: dict) -> dict | None:
    """The one row of the task's initial Sheets state that the address names (by row id and the
    spreadsheet/worksheet id or title), or None."""
    sheets = (initial_state or {}).get("google_sheets") or {}
    rows = sheets.get("rows") or []
    if not isinstance(rows, list) or "row_id" not in address:
        return None
    row_id = str(address["row_id"].raw)
    names = {}
    for key in ("spreadsheet_id", "spreadsheet", "worksheet_id", "worksheet", "worksheet_name"):
        if key in address and not isinstance(address[key].raw, (dict, list)):
            names[key.split("_")[0]] = str(address[key].raw)
    titles = {}
    for kind in ("spreadsheets", "worksheets"):
        for obj in sheets.get(kind) or []:
            if isinstance(obj, dict) and obj.get("id") is not None:
                titles[(kind[:-1], str(obj.get("title")))] = str(obj["id"])
    wanted = {k: titles.get((k, v), v) for k, v in names.items()}
    found = [r for r in rows if isinstance(r, dict) and str(r.get("row_id")) == row_id
             and all(str(r.get(f"{k}_id")) == v for k, v in wanted.items())]
    return found[0] if len(found) == 1 else None


def label_rows(items: list[Item], initial_state: dict | None, corpus: Corpus) -> list[Item]:
    """A native check that names a sheet row by number gets the row's first cell value as a label
    (`semantics["address_labels"]["row_id"]`), so a line can say `row 5 ("Sterling & Associates")`
    instead of a bare row number; the label is a value like any other (shown only when the attempt
    saw or wrote it)."""
    out = []
    for item in items:
        row = _sheet_row(initial_state, item.address) if item.origin == "native" else None
        cells = row.get("cells") if isinstance(row, dict) else None
        label = next((str(v) for v in (cells or {}).values() if isinstance(v, (str, int, float)) and str(v).strip()), None)
        if label is None:
            out.append(item)
            continue
        labels = {**(item.semantics.get("address_labels") or {}),
                  "row_id": {"text": label, "observed": corpus.is_observed(label), "written": corpus.is_written(label)}}
        out.append(item.model_copy(update={"semantics": {**item.semantics, "address_labels": labels}}))
    return out


def _dict_objects(node: Any, depth: int = 0, out: list | None = None) -> list:
    """Every dict inside a list of a world, any depth up to 4."""
    out = [] if out is None else out
    if depth > 4:
        return out
    if isinstance(node, dict):
        for v in node.values():
            _dict_objects(v, depth + 1, out)
    elif isinstance(node, list):
        for v in node:
            if isinstance(v, dict):
                out.append(v)
            _dict_objects(v, depth + 1, out)
    return out


def narrow_list_addresses(items: list[Item], timeline, calls: list[Call]) -> list[Item]:
    """A located manifest harm whose address is a list of ids (`id in [...]`, the rule's candidate domain)
    keeps only the ids the located call named or changed; with none left the item is marked
    `address_unverified` (withheld). The line then states what the call did, not the rule's domain."""
    from .observed import normalize, value_forms
    from .probe import _all_forms_in, added_objects

    def as_list(raw: Any) -> list | None:
        if isinstance(raw, list):
            return raw
        if isinstance(raw, str) and raw.startswith("[") and raw.endswith("]"):
            try:  # the rule's domain stored as JSON text
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, list) and parsed and all(not isinstance(p, (dict, list)) for p in parsed) else None
            except ValueError:
                return None
        return None

    by_site = {(c.turn, c.index): c for c in calls}
    out = []
    for item in items:
        lists = {k: v for k, v in item.address.items() if as_list(v.raw) is not None and len(as_list(v.raw)) > 1}
        if not (lists and item.origin == "manifest" and item.polarity in ("forbid", "keep") and item.location.kind == "at_call"
                and item.location.call is not None):
            out.append(item)
            continue
        call = by_site.get((item.location.call.turn, item.location.call.index))
        texts = [normalize(json.dumps(call.args, sort_keys=True, default=str))] if call else []
        before = timeline.json(call.before_digest) if call and timeline.available and call.before_digest else None
        after = timeline.json(call.after_digest) if call and timeline.available and call.after_digest else None
        if before is not None and after is not None:
            texts += [v["full"] for v in added_objects(before, after)]
            # a container whose sub-records grew (a ticket that got a message) carries the id the call acted on
            seen = {normalize(json.dumps(o, sort_keys=True, default=str)) for o in _dict_objects(before)}
            texts += [t for t in (normalize(json.dumps(o, sort_keys=True, default=str)) for o in _dict_objects(after)) if t not in seen]
        address = dict(item.address)
        unverified = False
        for key, value in lists.items():
            elements = as_list(value.raw)
            kept = [i for i, e in enumerate(elements) if any(_all_forms_in([e], t) for t in texts)]
            if not kept:
                unverified = True
                break
            hidden = set(value.hidden_elements)
            address[key] = Value(raw=[elements[i] for i in kept], observed=value.observed, written=value.written,
                                 forms=value_forms([elements[i] for i in kept]),
                                 hidden_elements=tuple(str(n) for n, i in enumerate(kept) if str(i) in hidden))
        if unverified:
            out.append(item.model_copy(update={"semantics": {**item.semantics, "address_unverified": True}}))
        else:
            out.append(item.model_copy(update={"address": address}))
    return out


def mark_matched_branches(items: list[Item], timeline, calls: list[Call]) -> list[Item]:
    """A located manifest harm whose condition has alternatives (`any`) is marked with the branches that hold
    on an object the located call added or changed in the effect's collection (`semantics["matched_branches"]`),
    so its line names the content the call produced rather than every branch of the rule."""
    from .probe import _same_app, _same_collection, added_objects, matched_branches

    if not timeline.available:
        return items
    by_site = {(c.turn, c.index): c for c in calls}
    out = []
    for item in items:
        groups = item.semantics.get("alternatives") or {}
        if not (item.origin == "manifest" and item.polarity in ("forbid", "keep") and item.location.kind == "at_call"
                and item.location.call is not None and groups):
            out.append(item)
            continue
        call = by_site.get((item.location.call.turn, item.location.call.index))
        before = timeline.json(call.before_digest) if call and call.before_digest else None
        after = timeline.json(call.after_digest) if call and call.after_digest else None
        if before is None or after is None:
            out.append(item)
            continue
        views = added_objects(before, after)
        app_keys = [k for k in after if _same_app(str(k), item.effect.app)]
        collection = item.semantics.get("collection") or item.effect.object
        collection_keys = [c for k in app_keys if isinstance(after[k], dict) for c in after[k] if _same_collection(str(c), collection or "")]
        views = [v for v in views if (not app_keys or _same_app(v["app"], item.effect.app))
                 and (not collection_keys or any(_same_collection(p, collection or "") for p in v.get("path", [])[1:]))]
        matched = matched_branches(item, views) if views else {}
        out.append(item.model_copy(update={"semantics": {**item.semantics, "matched_branches": matched}}) if matched else item)
    return out


def task_contract(episode: Episode) -> dict | None:
    """The task's manifest from the catalog (None for the 6 tasks without one)."""
    try:
        from ..tasks import get_task

        return get_task(episode.task).contract()
    except (KeyError, FileNotFoundError):
        return None


def _changed_texts(before: dict | None, after: dict | None) -> list[str]:
    """Normalised JSON of every dict inside a list (depth <= 5) of `after` that is not in `before` byte for byte."""
    from .observed import normalize

    def walk(node, depth, out):
        if depth > 5:
            return out
        if isinstance(node, dict):
            for v in node.values():
                walk(v, depth + 1, out)
        elif isinstance(node, list):
            for v in node:
                if isinstance(v, dict):
                    out.append(v)
                walk(v, depth + 1, out)
        return out

    if before is None or after is None:
        return []
    seen = {json.dumps(o, sort_keys=True, default=str) for o in walk(before, 0, [])}
    return [normalize(t) for t in (json.dumps(o, sort_keys=True, default=str) for o in walk(after, 0, [])) if t not in seen]


def _list_dicts(node: Any, depth: int = 0, out: list | None = None) -> list:
    out = [] if out is None else out
    if depth > 4:  # the same depth the truth checker walks
        return out
    if isinstance(node, dict):
        for v in node.values():
            _list_dicts(v, depth + 1, out)
    elif isinstance(node, list):
        for v in node:
            if isinstance(v, dict):
                out.append(v)
            _list_dicts(v, depth + 1, out)
    return out


TARGET_ALIASES = {"recipients": "to", "recipient": "to", "to_email": "to", "email_to": "to", "to_number": "to"}
TARGET_SUFFIXES = ("_ids", "_id", "_name", "_title")
ID_LIKE = ("id",)
NAME_LIKE = ("name", "title", "label", "display_name", "full_name", "username", "email", "handle")


def target_family(key: str) -> str:
    """The argument family an address key or a call's argument key belongs to: `channel_name`, `channel_id` and
    `channel` are one family ("channel"), `spreadsheet_id` and `spreadsheet` another, `recipients` is "to"."""
    import re as _re

    k = str(key).split("#")[0].split(".")[-1]
    k = _re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", k).lower()
    k = TARGET_ALIASES.get(k, k)
    for suffix in TARGET_SUFFIXES:
        if k.endswith(suffix) and len(k) > len(suffix):
            return k[: -len(suffix)]
    return k


def _arg_scalars(node: Any, key: str | None = None, depth: int = 0):
    """(argument key, scalar) for every scalar argument of a call, list elements under their list's key."""
    if isinstance(node, dict) and depth < 5:
        for k, v in node.items():
            yield from _arg_scalars(v, str(k), depth + 1)
    elif isinstance(node, list) and depth < 5:
        for v in node:
            yield from _arg_scalars(v, key, depth + 1)
    elif key is not None and node is not None and not isinstance(node, bool):
        yield key, node


def _same_text(a: Any, b: Any) -> bool:
    return str(a).strip().lower() == str(b).strip().lower()


def _token_in(value: Any, text: Any) -> bool:
    """`value` is the whole written value, or a whole token inside it ("a@x.com" in "a@x.com, b@x.com")."""
    import re as _re

    v, t = str(value).strip(), str(text)
    if not v:
        return False
    return bool(_re.search(r"(?<![\w@.\-])" + _re.escape(v) + r"(?![\w@\-]|\.\w)", t, _re.IGNORECASE))


def world_maps(world: dict | None, app: str, written: Any, named: Any) -> dict | None:
    """Proof, from a recorded world, that a value the call wrote and the value a check names are the same object:
    one object of the app's collections carries one of them under an id field and the other under a name-like
    field ({"collection", "index", "id_field", "name_field"}); None without such an object or with several."""
    found = []
    a = app.replace("_", "").lower()
    for app_key, data in (world or {}).items():
        k = str(app_key).replace("_", "").lower()
        if not isinstance(data, dict) or not (k == a or k.startswith(a) or a.startswith(k)):
            continue
        for coll, records in data.items():
            if not isinstance(records, list):
                continue
            for i, rec in enumerate(records):
                if not isinstance(rec, dict):
                    continue
                for id_value, name_value in ((written, named), (named, written)):
                    id_fields = [f for f in rec if (f in ID_LIKE or f.endswith("_id")) and not isinstance(rec[f], (dict, list))
                                 and rec[f] is not None and _same_text(rec[f], id_value)]
                    name_fields = [f for f in rec if f in NAME_LIKE and isinstance(rec[f], str) and _same_text(rec[f], name_value)]
                    if id_fields and name_fields:
                        found.append({"collection": [str(app_key), str(coll)], "index": i, "id_field": id_fields[0],
                                      "name_field": name_fields[0]})
    unique = {json.dumps(f, sort_keys=True) for f in found}
    return found[0] if len(unique) == 1 else None


def target_proof(item: Item, call: Call, world: dict | None) -> dict:
    """For every scalar address value of a located prohibition: what the located call wrote under the same argument
    family (`written`), the value quoted as written (`as_written`: the check's value, spelled as the call wrote it,
    when the call wrote it), else a written value the recorded world proves to be the same object (`mapped`, with
    the proof). A key the call wrote nothing for has no entry (the check's value stays)."""
    leaves = list(_arg_scalars(call.args))
    out: dict[str, dict] = {}
    for key, value in item.address.items():
        raw = value.raw
        if isinstance(raw, (dict, list)) or raw is None or isinstance(raw, bool):
            continue
        family = target_family(key)
        written = list(dict.fromkeys(v for k, v in leaves if target_family(k) == family))
        if not written:
            continue
        entry: dict[str, Any] = {"written": written[:10], "as_written": None, "mapped": None}
        exact = [w for w in written if _same_text(w, raw)]
        if exact:
            entry["as_written"] = exact[0]
        elif any(isinstance(w, str) and _token_in(raw, w) for w in written):
            entry["as_written"] = raw  # the call wrote it inside a longer value (one of several recipients)
        else:
            for w in written:
                proof = world_maps(world, item.effect.app, w, raw)
                if proof is not None:
                    entry["mapped"] = {"written": w, "named": raw, **proof}
                    break
        out[key] = entry
    return out


def mark_located_proof(items: list[Item], timeline, calls: list[Call], initial_state: dict | None) -> list[Item]:
    """For every located prohibition or broken record: what the snapshots prove about the located call
    (`semantics.located_proof`): the call's app is the effect's (`app_match`); the effect's own collection
    (`collection`, when it resolves in the initial world) changed across the call (`collection_changed`, None
    without snapshots); for a broken record, a changed object carries the address (`changed_carries_address`).
    The plain renderer (version c) stays silent where the proof is missing (engine-final specs R2, R3)."""
    from .. import fix
    from .observed import Corpus, value_forms

    by_ref = {(c.turn, c.index): c for c in calls}
    out = []
    for item in items:
        if item.polarity not in ("forbid", "keep") or item.location.kind != "at_call" or item.location.call is None:
            out.append(item)
            continue
        call = by_ref.get((item.location.call.turn, item.location.call.index))
        if call is None:
            out.append(item)
            continue
        a, b = call.app.replace("_", "").lower(), item.effect.app.replace("_", "").lower()
        proof: dict[str, Any] = {"app_match": bool(a and b) and (a == b or a.startswith(b) or b.startswith(a))}
        target = fix.effect_collection(item, initial_state)
        proof["collection"] = list(target) if target else None
        before = timeline.json(call.before_digest) if getattr(timeline, "available", False) and call.before_digest else None
        after = timeline.json(call.after_digest) if getattr(timeline, "available", False) and call.after_digest else None
        if target and before is not None and after is not None:
            proof["collection_changed"] = (before.get(target[0]) or {}).get(target[1]) != (after.get(target[0]) or {}).get(target[1])
        else:
            proof["collection_changed"] = None
        obj = item.probe.offending_object if item.origin == "native" else None
        if obj is not None and before is not None and after is not None and isinstance(obj.get("collection"), str):
            # the offending object exists after the call (its identity fields as the probe recorded them) and the
            # call produced or changed it: its node differs before and after
            def node(world):
                cur: Any = world
                for p in [int(x) if x.isdigit() else x for x in obj["collection"].split("/") if x != ""] + [obj.get("key")]:
                    try:
                        cur = cur[p]
                    except (KeyError, IndexError, TypeError):
                        return None
                return cur
            na, nb = node(after), node(before)
            fields = obj.get("fields") or {}
            ids = {k: v for k, v in fields.items() if k in ("id", "row_id", "rowId", "ticket_id", "message_id", "email", "name", "subject", "ts")}
            container = None
            if "value" in fields and not isinstance(fields["value"], (dict, list)):
                cur_a: Any = after
                cur_b: Any = before
                for p in [int(x) if x.isdigit() else x for x in obj["collection"].split("/") if x != ""]:
                    try:
                        cur_a = cur_a[p]
                    except (KeyError, IndexError, TypeError):
                        cur_a = None
                    try:
                        cur_b = cur_b[p]
                    except (KeyError, IndexError, TypeError):
                        cur_b = None
                container = (cur_a, cur_b) if isinstance(cur_a, list) else None
            if container is not None:
                # correction r0: a scalar element of a list (a label on a message) is produced by the call only when it
                # is in the list after the call and was in no position of it before: a removal shifts a pre-existing
                # element into the probe's index, which proves nothing about this call
                in_after = any(str(v) == str(fields["value"]) for v in container[0] if not isinstance(v, (dict, list)))
                in_before = isinstance(container[1], list) and any(
                    str(v) == str(fields["value"]) for v in container[1] if not isinstance(v, (dict, list)))
                proof["object_produced"] = bool(in_after and not in_before)
            elif isinstance(na, dict):
                ok = all(str(na.get(k)) == str(v) for k, v in ids.items())
            elif na is not None and "value" in fields:
                ok = str(na) == str(fields["value"])
            else:
                ok = na is not None
            if container is None:
                proof["object_produced"] = bool(ok and na != nb) if not ("value" in fields and isinstance(na, list)) else None
        # what the call wrote into the world: the scalar fields of the objects it added or changed (and cells),
        # so a line can quote the value as written (removal round: never a matcher token)
        written: list[list] = []
        if before is not None and after is not None:
            seen = {json.dumps(o, sort_keys=True, default=str) for o in _list_dicts(before)}
            for o in _list_dicts(after):
                if json.dumps(o, sort_keys=True, default=str) in seen:
                    continue
                for k, v in o.items():
                    if isinstance(v, dict):
                        written.extend([str(k2), v2] for k2, v2 in v.items() if not isinstance(v2, (dict, list)))
                    elif not isinstance(v, (dict, list)):
                        written.append([str(k), v])
        proof["written"] = written[:400]
        # Fix 1: the target as the call wrote it (the world before the call, else the task's initial world, proves a
        # mapping between the call's id or name and the check's)
        proof["targets"] = target_proof(item, call, before if before is not None else initial_state)
        if item.polarity == "keep":
            if item.origin == "native" and item.probe.offending_object is not None:
                proof["changed_carries_address"] = True
            else:
                values = [v.raw for v in item.address.values() if not (isinstance(v.raw, dict) and "unresolved" in v.raw)]
                texts = _changed_texts(before, after)
                proof["changed_carries_address"] = bool(values) and any(
                    all(any(Corpus._form_in(f, t) for f in value_forms(v) if f) for v in values) for t in texts)
        out.append(item.model_copy(update={"semantics": {**item.semantics, "located_proof": proof}}))
    return out


def _world_object_texts(world: Any) -> list[str]:
    """Normalised JSON of every object (a dict inside a list, depth <= 4) of a world."""
    from .observed import normalize

    out: list[str] = []

    def walk(node: Any, depth: int) -> None:
        if depth > 4:
            return
        if isinstance(node, dict):
            for v in node.values():
                walk(v, depth + 1)
        elif isinstance(node, list):
            for v in node:
                if isinstance(v, dict):
                    out.append(normalize(json.dumps(v, sort_keys=True, default=str)))
                walk(v, depth + 1)

    walk(world, 0)
    return out


def mark_identity_evidence(items: list[Item], timeline, calls: list[Call]) -> list[Item]:
    """Correction r0: for a native requirement the grader's address-only verdict says exists
    (`probe.identity_present`), whether the address values are literally there: in an object of the final world or in
    the tool and arguments of a successful call of the effect's app (`semantics.identity_literal`). The grader may
    normalise a value ("update_record" for an action recorded as "updateRecord"); a line saying "a X <address> exists"
    then quotes something the attempt's world does not hold, and version (c) says only what is required."""
    from .observed import Corpus, normalize, value_forms

    if not getattr(timeline, "available", False):
        return items
    texts: list[str] | None = None
    out = []
    for item in items:
        if item.origin != "native" or item.type == "tool_error" or item.probe.identity_present is not True:
            out.append(item)
            continue
        values = [v.raw for v in item.address.values() if not (isinstance(v.raw, dict) and "unresolved" in v.raw)]

        def forms(value: Any) -> list[str]:
            if isinstance(value, dict):
                return [f for v in value.values() for f in forms(v)]
            return [f for f in value_forms(value) if f]

        def all_in(text: str) -> bool:
            return all(not forms(v) or any(Corpus._form_in(f, text) for f in forms(v)) for v in values)

        if texts is None:
            final = timeline.json(timeline.final_digest) if timeline.final_digest else None
            texts = _world_object_texts(final)
        a = item.effect.app.replace("_", "")
        in_calls = any(c.error is None and (c.app.replace("_", "") == a or c.app.replace("_", "").startswith(a)
                                            or a.startswith(c.app.replace("_", "")))
                       and all_in(normalize(c.tool + " " + json.dumps(c.args, sort_keys=True, default=str))) for c in calls)
        literal = bool(values) and (in_calls or any(all_in(t) for t in texts))
        out.append(item.model_copy(update={"semantics": {**item.semantics, "identity_literal": literal}}))
    return out


def build_record(episode: Episode, siblings: list[Episode] | None = None, *, probe: bool = False,
                 manifest: bool = False) -> Record:
    calls, node_turns, turns, complete = episode_calls(episode)
    corpus = Corpus(episode.messages, tuple(episode.task_data.get("zapier_tools") or ()))
    flags = initial_pass_flags(episode)
    items: list[Item] = []
    assertions = episode.task_data.get("assertions") or []
    for index, result in enumerate(episode.native_results):
        items.append(native_item(index, result, corpus, flags[index] if index < len(flags) else None))
    items = label_rows(items, episode.task_data.get("initial_state"), corpus)
    notes = []
    timeline = None
    if probe:
        from .probe import probe_item
        from .timeline import Timeline

        timeline = Timeline(episode)
        if not timeline.available:
            notes.append("no snapshots: probing skipped")
        by_invocation = {c.invocation_id: c for c in calls if c.invocation_id}
        probed = []
        for item in items:
            index = int(item.id.split(":")[1])
            assertion = assertions[index] if index < len(assertions) else {"type": item.type, **{
                k: v.raw for k, v in {**item.address, **item.content}.items()}, **(item.cardinality or {})}
            result = probe_item(timeline, assertion, item.polarity, item.outcome, calls=tuple(calls),
                                app=item.effect.app) if timeline.available else ProbeResult(
                status="none", reason="no_snapshots")
            update: dict[str, Any] = {"probe": result, "location": locate(item, result, by_invocation)}
            if result.reason and "negative_check_requires_effect" in result.reason:
                # The probe showed this negative-marked check needs the effect to exist.
                update["polarity"] = "require"
                update["polarity_source"] = "probe"
                update["semantics"] = {**item.semantics, "effect_required": True,
                                       "forbidden_content_keys": list(result.failing_clauses)}
            probed.append(item.model_copy(update=update))
        items = probed
    disagreements: list[dict] = []
    if manifest:
        from .manifest import compile_contract, manifest_items
        from .merge import merge_items

        contract = task_contract(episode)
        if contract is None:
            notes.append("no manifest for task")
        else:
            specs = compile_contract(contract, episode.task_data.get("initial_state"), episode.task_data.get("prompt"))
            by_invocation = {c.invocation_id: c for c in calls if c.invocation_id}
            items.extend(manifest_items(specs, episode.assessment_batches, by_invocation, corpus))
            items, disagreements = merge_items(items)
            if timeline is None:
                from .timeline import Timeline

                timeline = Timeline(episode)
            items = mark_present_effects(items, timeline)
            items = narrow_list_addresses(items, timeline, calls)
            items = mark_matched_branches(items, timeline, calls)
    if timeline is not None:
        items = mark_located_proof(items, timeline, calls, episode.task_data.get("initial_state"))
        items = mark_identity_evidence(items, timeline, calls)
    items.extend(tool_error_items(calls))
    metrics = episode.metrics
    score = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    stop = episode.stop_condition or ""
    budget: dict[str, Any] = {"turns_used": turns, "stop_condition": stop,
                              "ended_at_limit": any(m in stop for m in TURN_LIMIT_MARKERS)}
    if not complete:
        notes.append("invocation mapping incomplete")
    for d in disagreements:
        notes.append(f"disagreement {d['native']} {d['native_outcome']} vs {d['manifest']} {d['manifest_outcome']}")
    return Record(
        task=episode.task, episode_id=episode.id, success=episode.success, score=score, turns=turns,
        calls=tuple(calls), items=tuple(items), observed_digest=corpus.digest, budget=budget,
        siblings=tuple({"episode_id": s.id, "success": s.success} for s in (siblings or [])),
        notes=tuple(notes),
    )
