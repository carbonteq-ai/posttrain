"""Behavioural probing: derive meaning and location of a native check by running the grader.

Three probes, all generic over handlers (docs/design.md §2, alternatives.md option C):

- flips: evaluate the handler on the world before and after every mutating invocation;
- object ablation: `world_patch(initial, final)` enumerates every object the attempt added or
  changed; remove/revert each one in a copy of the final world and re-check; the object whose
  removal changes the verdict is the offending (negative check) or witness (positive check);
- clause ablation: for a failed positive check, test the address-only assertion, then add
  content keys one at a time; the keys that turn it false are the failing clauses.

Count checks are probed by substituting the count parameter.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from .model import ProbeResult
from .native import ADDRESS_KEYS, CARDINALITY_KEYS, CONTROL_KEYS
from .timeline import Timeline, check, load_world_safe

PRIVATE_KEY = "_automationbench_v1_private"
MAX_OBJECTS = 400
MAX_COUNT = 200


# ---------------------------------------------------------------------------------------------
# Object enumeration over a structural patch
# ---------------------------------------------------------------------------------------------
def _objects(patch: Any, path: tuple, before: Any, out: list) -> None:
    """Objects added or changed between two worlds, as (path, kind, index/key)."""
    if not isinstance(patch, dict):
        return
    if "+" in patch:
        base = len(before) if isinstance(before, list) else 0
        for offset, _ in enumerate(patch["+"]):
            out.append((path, "added", base + offset))
        return
    if "[]" in patch:
        for index, child in patch["[]"].items():
            index = int(index)
            child_before = before[index] if isinstance(before, list) and index < len(before) else None
            # A changed list element is one object (revert it whole) unless it is itself a container
            # of lists (then recurse so sub-objects are found).
            if isinstance(child, dict) and "{}" in child and isinstance(child_before, dict) and any(
                isinstance(v, list) for v in child_before.values()
            ):
                _objects(child, path + (index,), child_before, out)
            else:
                out.append((path, "changed", index))
        return
    if "{}" in patch:
        for key, child in patch["{}"].items():
            child_before = before.get(key) if isinstance(before, dict) else None
            if isinstance(child, dict) and "=" in child and child_before is None:
                out.append((path, "added_key", key))
            elif isinstance(child, dict) and "=" in child:
                out.append((path, "changed", key))
            else:
                _objects(child, path + (key,), child_before, out)
        for key in patch.get("-", ()):
            out.append((path, "removed", key))


def _get(data: Any, path: tuple) -> Any:
    for step in path:
        data = data[step]
    return data


def _ablate(final: dict, initial: dict, path: tuple, kind: str, key: Any) -> dict | None:
    """A copy of `final` with one object removed (added) or reverted (changed)."""
    world = copy.deepcopy(final)
    try:
        container = _get(world, path)
        if kind == "added":
            if not isinstance(container, list) or key >= len(container):
                return None
            removed = container.pop(key)
            _forget_private(world, removed)
        elif kind == "added_key":
            removed = container.pop(key)
            _forget_private(world, removed)
        elif kind == "changed":
            source = _get(initial, path)
            if isinstance(container, list):
                container[key] = copy.deepcopy(source[key])
            else:
                container[key] = copy.deepcopy(source[key])
            _forget_private(world, container[key])
        elif kind == "removed":
            source = _get(initial, path)
            container[key] = copy.deepcopy(source[key])
        else:
            return None
    except (KeyError, IndexError, TypeError):
        return None
    return world


def _forget_private(world: dict, obj: Any) -> None:
    """Drop private Sheets update keys that name the ablated row (environment private state)."""
    private = world.get(PRIVATE_KEY)
    if not isinstance(private, dict) or not isinstance(obj, dict):
        return
    ids = {str(v) for k, v in obj.items() if isinstance(v, (str, int)) and k in ("id", "row_id", "rowId")}
    for name, value in list(private.items()):
        if isinstance(value, list):
            private[name] = [v for v in value if not (set(re.split(r"[^A-Za-z0-9_]+", str(v))) & ids)]


def _describe(final: dict, path: tuple, kind: str, key: Any) -> dict[str, Any]:
    try:
        obj = _get(final, path + (key,))
    except (KeyError, IndexError, TypeError):
        obj = None
    fields = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (str, int, float, bool)) or (isinstance(v, list) and all(isinstance(x, str) for x in v)):
                text = v if not isinstance(v, str) else (v if len(v) <= 200 else v[:200] + "…")
                fields[k] = text
    else:
        fields = {"value": obj if not isinstance(obj, str) or len(obj) <= 200 else obj[:200] + "…"}
    collection = "/".join(str(p) for p in path)
    return {"collection": collection, "kind": kind, "key": key, "fields": fields}


# ---------------------------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------------------------
def probe_flips(timeline: Timeline, assertion: dict) -> tuple[list[str], str | None, bool | None, bool | None, list[str]]:
    """(flip invocations, first direction, initial verdict, final verdict, tools at flips)."""
    initial = check(timeline.world(timeline.initial_digest), assertion) if timeline.available else None
    previous = initial
    flips, tools = [], []
    direction = None
    for inv, before, after, tool in timeline.mutating_steps():
        verdict = check(timeline.world(after), assertion)
        if verdict is None:
            continue
        if previous is not None and verdict != previous:
            flips.append(inv)
            tools.append(tool or "")
            if direction is None:
                direction = "pass_to_fail" if previous else "fail_to_pass"
        previous = verdict
    final = check(timeline.world(timeline.final_digest), assertion) if timeline.available else None
    return flips, direction, initial, final, tools


def probe_objects(initial_json: dict, final_json: dict, assertion: dict, want: bool) -> list[dict]:
    """Objects whose removal/revert makes the handler return `want` on the final world."""
    from automationbench_v1.capture import world_patch

    patch = world_patch(initial_json, final_json)
    if patch is None:
        return []
    objects: list = []
    _objects(patch, (), initial_json, objects)
    # The environment's private Sheets record is not an object of the world; it is adjusted when
    # the row it names is ablated (see _forget_private).
    objects = [o for o in objects if not ((o[0] and o[0][0] == PRIVATE_KEY) or (not o[0] and o[2] == PRIVATE_KEY))]
    hits = []
    for path, kind, key in objects[:MAX_OBJECTS]:
        ablated = _ablate(final_json, initial_json, path, kind, key)
        if ablated is None:
            continue
        world = load_world_safe(ablated)
        if world is None:
            continue
        if check(world, assertion) is want:
            hits.append(_describe(final_json, path, kind, key))
    return hits


NONCE = "zq9x-nonce-never-present-7k3v"


def _nonce_like(value: Any) -> Any | None:
    """A value of the same shape as `value` that no world text can contain (None when there is none)."""
    if isinstance(value, str):
        return NONCE
    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
        return [NONCE]
    if isinstance(value, dict) and value and all(isinstance(v, str) for v in value.values()):
        return {k: NONCE for k in value}
    return None


def _split_params(assertion: dict) -> tuple[dict, dict, dict]:
    params = {k: v for k, v in assertion.items() if k != "type" and k not in CONTROL_KEYS}
    address = {k: v for k, v in params.items() if k in ADDRESS_KEYS}
    content = {k: v for k, v in params.items() if k not in ADDRESS_KEYS and k not in CARDINALITY_KEYS}
    return params, address, content


def _all_forms_in(values: list[Any], text: str) -> bool:
    from .observed import Corpus, value_forms

    for value in values:
        forms = [f for f in value_forms(value) if f]
        if forms and not any(Corpus._form_in(f, text) for f in forms):
            return False
    return True


def world_objects(world: Any, depth: int = 0, out: list | None = None) -> list[Any]:
    """Every object of a world: the dicts inside lists, at any depth up to 4 (apps hold collections
    of records; records may hold lists of sub-records)."""
    out = [] if out is None else out
    if depth > 4:
        return out
    if isinstance(world, dict):
        for v in world.values():
            world_objects(v, depth + 1, out)
    elif isinstance(world, list):
        for v in world:
            if isinstance(v, dict):
                out.append(v)
            world_objects(v, depth + 1, out)
    return out


def id_record(final_json: Any, app: str, ids: dict) -> dict | None:
    """judge 37/69/70/121/136: the one object of the app's part of the final world that carries every id value of the
    check under its own key (`ticket_id`) or as its `id`; None when there is no such object or several."""
    item_app = app.replace("_", "")
    hits = []
    for key, part in (final_json or {}).items():
        k_app = str(key).replace("_", "")
        if item_app and not (k_app == item_app or k_app.startswith(item_app) or item_app.startswith(k_app)):
            continue
        for node in world_objects(part):
            if all(str(node.get(k)) == str(v) or str(node.get("id")) == str(v) for k, v in ids.items()):
                hits.append(node)
    return hits[0] if len(hits) == 1 else None


def identity_evidence(initial_json: dict, final_json: dict, address: dict, calls: tuple = (), app: str = "") -> bool:
    """Whether anything in the final world or in the attempt's calls carries the identity values: an
    object of the final world (added or pre-existing) whose text holds every non-schema address
    value, or a successful mutating call of the same app whose arguments do. Used only to decide
    whether an *absence* may be claimed (False here = nothing matches = "no such X" is safe); an
    address made only of schema names (field, column, …) identifies nothing, so it counts as
    evidence (no absence claim)."""
    from .native import SCHEMA_KEYS
    from .observed import normalize

    values = [v for k, v in address.items() if k not in SCHEMA_KEYS]
    if not values:
        return True
    for node in world_objects(final_json):
        if _all_forms_in(values, normalize(json.dumps(node, sort_keys=True, default=str))):
            return True
    item_app = app.replace("_", "")
    for call in calls:
        if call.error is not None or call.mutated is not True:
            continue
        call_app = call.app.replace("_", "")
        if item_app and not (call_app == item_app or call_app.startswith(item_app) or item_app.startswith(call_app)):
            continue
        if _all_forms_in(values, normalize(json.dumps(call.args, sort_keys=True, default=str))):
            return True
    return False


def added_identity_evidence(initial_json: dict, final_json: dict, address: dict, calls: tuple = (), app: str = "") -> bool | None:
    """Whether the attempt itself produced an object carrying the (non-schema) address values: True when
    an object it added does, or a successful mutating call of the same app names them; False when
    neither does and no object of the final world that carries the values (a channel the address
    names) has an id some added object of the app refers to (the attempt may have posted there by id);
    None otherwise (no claim). Used to say "the attempt sent no X <address>" instead of "X exists" when
    the only matching objects were there before the attempt."""
    from .native import SCHEMA_KEYS
    from .observed import normalize

    values = [v for k, v in address.items() if k not in SCHEMA_KEYS]
    if not values or initial_json is None or final_json is None:
        return None
    views = added_objects(initial_json, final_json)
    added = [v for v in views if v["kind"] == "added" and (not app or _same_app(v["app"], app))]
    if any(_all_forms_in(values, v["full"]) for v in added):
        return True
    item_app = app.replace("_", "")
    for call in calls:
        if call.error is not None or call.mutated is not True:
            continue
        call_app = call.app.replace("_", "")
        if item_app and not (call_app == item_app or call_app.startswith(item_app) or item_app.startswith(call_app)):
            continue
        if _all_forms_in(values, normalize(json.dumps(call.args, sort_keys=True, default=str))):
            return True
    # an alias: a container object that carries the values and whose id an added object refers to
    for node in world_objects(final_json):
        if not isinstance(node, dict) or not _all_forms_in(values, normalize(json.dumps(node, sort_keys=True, default=str))):
            continue
        ids = [str(v) for k, v in node.items() if (k == "id" or str(k).endswith("_id")) and isinstance(v, (str, int)) and str(v)]
        if any(_all_forms_in([i], v["full"]) for i in ids for v in added):
            return None
    return False


TEXT_FIELDS = frozenset({"body", "body_plain", "body_html", "body_text", "text", "description", "notes", "note", "content",
                         "snippet", "message", "comment", "html", "plain_text"})
TEXT_STEMS = ("body", "text", "description", "notes", "note", "content", "message", "comment", "summary", "snippet")


def _short(value: Any) -> bool:
    if isinstance(value, (dict, list)):
        return all(not isinstance(v, (dict, list)) and _short(v) for v in (value.values() if isinstance(value, dict) else value))
    return len(str(value)) <= 160


def object_view(node: Any) -> dict[str, Any]:
    """{"identity", "full", "fields"}: the full text is the object's normalised JSON; the identity text is
    the JSON of its short, non-prose fields (ids, recipients, channels, statuses), where an address value
    must be found (a recipient's address quoted inside another message's body is not a message to them);
    `fields` holds each field's raw text (nested one level as "k.k2"), newlines intact, so a mention or an
    "on one line" set is judged inside the field the task names."""
    from .observed import normalize

    full = normalize(json.dumps(node, sort_keys=True, default=str))
    if not isinstance(node, dict):
        return {"identity": full, "full": full, "fields": {}}
    identity = {k: v for k, v in node.items() if str(k).lower() not in TEXT_FIELDS and _short(v)}
    fields: dict[str, str] = {}
    for k, v in node.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if not isinstance(v2, (dict, list)):
                    fields[f"{k}.{k2}".lower()] = str(v2)
            continue
        fields[str(k).lower()] = v if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str)
    return {"identity": normalize(json.dumps(identity, sort_keys=True, default=str)), "full": full, "fields": fields}


def added_objects(initial_json: dict, final_json: dict) -> list[dict[str, Any]]:
    """One view per object the attempt added or changed between the two worlds, with `app` (its top-level
    collection key), `path` (the collection keys above it) and `kind` ("added" | "changed"); a whole
    collection added or replaced under a key is read element by element. The environment's private
    Sheets record is not an object of the world."""
    from automationbench_v1.capture import world_patch

    patch = world_patch(initial_json, final_json)
    if patch is None:
        return []
    objects: list = []
    _objects(patch, (), initial_json, objects)
    out = []
    for path, kind, key in objects:
        if kind == "removed" or (path and path[0] == PRIVATE_KEY) or (not path and key == PRIVATE_KEY):
            continue
        try:
            node = _get(final_json, path + (key,))
        except (KeyError, IndexError, TypeError):
            continue
        full_path = [str(p) for p in path + (key,) if not isinstance(p, int)]
        app = full_path[0] if full_path else ""
        kind = "changed" if kind == "changed" else "added"
        for element in (node if isinstance(node, list) else [node]):
            out.append({"app": app, "path": full_path, "kind": kind, **object_view(element)})
    return out


def _same_app(a: str, b: str) -> bool:
    a, b = a.replace("_", ""), b.replace("_", "")
    return bool(a and b) and (a == b or a.startswith(b) or b.startswith(a))


def _same_collection(key: str, want: str) -> bool:
    """A world collection key names the effect's collection: equal, or its plural ("messages" for
    "message", "sms_messages" for "sms_message", "customer_lists" for "customer_list")."""
    a, b = key.lower().replace("_", ""), want.lower().replace("_", "")
    return bool(a and b) and (a == b or a == b + "s" or a == b + "es" or (b.endswith("y") and a == b[:-1] + "ies"))


def candidate_identity_value(item) -> Any | None:
    """The candidate's native id when the candidates are records of the effect's own world collection
    (a goal about contact 207 is not met by contact 201 turning high): the object must carry it."""
    candidate = item.candidate or {}
    population = item.semantics.get("population_collection")
    collection = item.semantics.get("collection") or item.effect.object
    if not candidate or not population or not _same_collection(str(population), str(collection)):
        return None
    return candidate.get("key")


def group_parents(group_branches: dict[Any, list[str]]) -> dict[Any, tuple[Any, str] | None]:
    """For every `any` group, the (group, branch) it is nested in, from the branch ids the compiler gave its
    conjuncts ("0.1" is branch 1 of an `any` inside branch 0 of an enclosing `any`): the parent is the closest
    preceding group whose branch ids are one segment shorter (groups are numbered in compile order, an
    enclosing `any` before the ones inside it). None for a top-level group or when no parent is found."""
    def depth(ids: list[str]) -> int:
        return max((str(i).count(".") + 1 for i in ids), default=1)

    def sortable(g: Any) -> tuple:
        return (0, g) if isinstance(g, int) else (1, str(g))

    depths = {g: depth(ids) for g, ids in group_branches.items()}
    parents: dict[Any, tuple[Any, str] | None] = {}
    for g, ids in group_branches.items():
        d = depths[g]
        prefixes = {str(i).rsplit(".", 1)[0] for i in ids if "." in str(i)}
        if d == 1 or len(prefixes) != 1:
            parents[g] = None
            continue
        prefix = next(iter(prefixes))
        candidates = [h for h in group_branches if depths[h] == d - 1 and sortable(h) < sortable(g)]
        parents[g] = (max(candidates, key=sortable), prefix) if candidates else None
    return parents


def content_structure(item) -> tuple[list[list[dict]], list[tuple[str, str, Any]]]:
    """(content requirements, negated terms) of an item, read as a line would state them. A term is
    (field, mode, value). Each requirement is a list of alternatives (the branches of one `any`, or the
    one way of a plain conjunct); an alternative is {"terms", "negated", "sets"}: the terms that must all
    hold, the terms that must not, and the "on one line" sets, each {"scope", "terms", "excluded"}: its
    terms within one unit of that scope that carries none of its excluded terms. Negated terms outside
    any `any` are returned apart (they must hold nowhere in the object). Unresolved values are not
    requirements (the line never names them)."""
    def unresolved(value) -> bool:
        return isinstance(value.raw, dict) and "unresolved" in value.raw

    negated_keys = set(item.semantics.get("negated_keys") or ())
    excluded_keys = set(item.semantics.get("excluded_keys") or ())
    groups = item.semantics.get("alternatives") or {}
    branches = item.semantics.get("branches") or {}
    together = item.semantics.get("together") or {}
    scopes = item.semantics.get("together_scope") or {}
    modes = item.semantics.get("modes") or {}
    negated: list = []
    required: dict[Any, dict[Any, dict]] = {}

    def alternative(requirement: Any, branch: Any) -> dict:
        return required.setdefault(requirement, {}).setdefault(
            branch, {"terms": [], "negated": [], "sets": {}, "nested": [],
                     "group": requirement[1] if requirement[0] == "group" else None, "branch": str(branch)})

    def put(alt: dict, key: str, term: tuple) -> None:
        t = together.get(key)
        if t is not None:
            s = alt["sets"].setdefault(t, {"scope": scopes.get(str(t), "line"), "terms": [], "excluded": []})
            (s["excluded"] if key in excluded_keys else s["terms"]).append(term)
        elif key in negated_keys:
            alt["negated"].append(term)
        else:
            alt["terms"].append(term)

    for key, value in item.content.items():
        if unresolved(value):
            continue
        term = (key.split("#")[0], modes.get(key) or "mentions", value.raw)
        g, t = groups.get(key), together.get(key)
        if g is not None:
            branch = branches.get(key, ("set", t) if t is not None else key)  # no branch numbers: a set is one branch
            put(alternative(("group", g), branch), key, term)
        elif t is not None:
            put(alternative(("together", t), 0), key, term)
        elif key in negated_keys:
            negated.append(term)
        else:
            put(alternative(("single", key), 0), key, term)
    # an `any` inside a branch of another `any` is a requirement of that branch, not of the whole item
    group_ids = {req[1]: [str(b) for b in alts] for req, alts in required.items() if req[0] == "group"}
    parents = group_parents(group_ids)
    finished: dict[Any, list[dict]] = {}

    def finish(requirement: Any) -> list[dict]:
        alts = required[requirement]
        out_alts = []
        for branch, a in alts.items():
            nested = [finish(("group", g)) for g, parent in parents.items()
                      if parent is not None and parent[0] == requirement[1] and parent[1] == str(branch)]
            out_alts.append({**a, "sets": list(a["sets"].values()), "nested": nested})
        return out_alts

    out = []
    for requirement in required:
        if requirement[0] == "group" and parents.get(requirement[1]) is not None:
            continue  # rendered inside its parent branch
        out.append(finish(requirement))
    return out, negated


def matched_branches(item, views: list[dict[str, Any]]) -> dict[str, list[str]]:
    """For every top-level `any` of a located manifest harm, the branches that hold on some object the
    located call added or changed (group id -> branch ids, as strings); a group none of whose branches
    holds is left out (the line then keeps every branch)."""
    required, _negated = content_structure(item)
    out: dict[str, list[str]] = {}
    for alts in required:
        group = alts[0].get("group") if alts else None
        if group is None:
            continue
        held = [a["branch"] for a in alts if any(alternative_holds(v, a) for v in views)]
        if held:
            out[str(group)] = held
    return out


def item_requirements(item) -> tuple[list[Any], list[list[dict]], list[tuple[str, str, Any]]]:
    """(address values, content requirements, negated terms) of an item (see `content_structure`); the
    candidate's own record id is part of the address when the candidates are records of the effect's
    collection."""
    def unresolved(value) -> bool:
        return isinstance(value.raw, dict) and "unresolved" in value.raw

    address = [v.raw for v in item.address.values() if not unresolved(v)]
    own_record = candidate_identity_value(item)
    if own_record is not None:
        address.append(own_record)
    required, negated = content_structure(item)
    return address, required, negated


def field_value_in(text: str, field: str, value: Any) -> bool | None:
    """Whether the object's own field `field` holds `value` (one of its forms, as a scalar or a list
    element); None when the object has no such field."""
    m = re.search(r'"' + re.escape(field.lower()) + r'": ', text)
    if m is None:
        return None
    rest = text[m.end():]
    if rest.startswith("["):
        segment = rest[: rest.find("]") + 1]
    else:
        end = re.search(r', "|[}\]]', rest)
        segment = rest[: end.start()] if end else rest
    if isinstance(value, str) and value == "":
        return segment.strip() in ('""', "null", "")  # an empty value: the field is empty, not merely present
    return _all_forms_in([value], segment)


def field_text(view: dict[str, Any], field: str) -> str | None:
    """The raw text of the object's field the term names (`body_text` reads the body fields, `text` the
    text; a nested `record.body` its last segment), or None when the object has no such field."""
    fields = view.get("fields") or {}
    name = field.split(".")[-1].lower()
    if name in fields:
        return fields[name]
    stem = name.split("_")[0]
    if stem in TEXT_STEMS:
        found = [v for k, v in fields.items() if k.split(".")[-1].split("_")[0] == stem]
        if found:
            return "\n".join(found)
    return None


def _units(text: str, scope: str, *, escaped: bool = False) -> list[str]:
    """The units of a text for a scope: the whole text, its lines or its blocks; `escaped` when the text
    is JSON (newlines written as backslash-n)."""
    from .observed import normalize

    if scope == "text":
        return [text if escaped else normalize(text)]
    nl = "\\n" if escaped else "\n"
    sep = nl + nl if scope == "block" else nl
    return [u if escaped else normalize(u) for u in text.split(sep) if u.strip()]


def term_holds(view: dict[str, Any], term: tuple[str, str, Any], unit: str | None = None) -> bool:
    """An equality on a named field holds in that field, and in no other (an absent field is not
    present; an identity field such as `recipients` may be spelt otherwise, so it is read in the
    identity fields); a mention holds in the text field the term names (the given unit of it), and
    anywhere in the object only when the object has no such field."""
    from .manifest import EFFECT_IDENTITY_FIELDS
    from .observed import normalize

    field, mode, value = term
    name = field.split(".")[-1]
    if mode in ("eq", "ne", "in"):
        found = field_value_in(view["full"], name, value)
        if found is None:
            return name in EFFECT_IDENTITY_FIELDS and _all_forms_in([value], view["identity"])
        return found
    if unit is not None:
        return _all_forms_in([value], unit)
    text = field_text(view, field)
    return _all_forms_in([value], normalize(text) if text is not None else view["full"])


def set_holds(view: dict[str, Any], together: dict) -> bool:
    """An "on one line" set holds when one unit of the named field's text (a line, a block, or the whole
    text) carries every term and none of the set's excluded terms."""
    terms, excluded, scope = together["terms"], together.get("excluded") or [], together.get("scope", "line")
    if not terms:
        return True
    field = terms[0][0]
    if scope == "text" and not excluded:
        return all(term_holds(view, t) for t in terms)
    text = field_text(view, field)
    units = _units(text, scope) if text is not None else _units(view["full"], scope, escaped=True)
    return any(all(term_holds(view, t, unit) for t in terms) and not any(term_holds(view, e, unit) for e in excluded)
               for unit in units)


def alternative_holds(view: dict[str, Any], alternative: dict) -> bool:
    """One alternative of a requirement holds: every plain term, no negated term, every set (see
    `content_structure`). The earlier shape {"scope", "terms"} is read as one set."""
    if "sets" not in alternative:
        return set_holds(view, {"scope": alternative.get("scope", "line"), "terms": alternative["terms"], "excluded": []})
    return (all(term_holds(view, t) for t in alternative["terms"])
            and not any(term_holds(view, t) for t in alternative.get("negated") or [])
            and all(set_holds(view, s) for s in alternative.get("sets") or [])
            and all(any(alternative_holds(view, a) for a in nested) for nested in alternative.get("nested") or []))


def effect_present(views: list[dict[str, Any]], address: list, required: list[list[dict]], negated: list,
                   *, app: str = "", collection: str | None = None, world: dict | None = None, new_only: bool = True) -> bool:
    """Whether an object the attempt added (or, unless `new_only`, changed) carries every address value
    in its identity fields, one alternative of every content requirement and none of the negated
    terms: everything a "Not done" line would name as missing. A goal for a new effect (a message
    sent, a record created) is not met by a record the attempt only updated, so such goals read added
    objects alone. Objects are read from the effect's own collection when the final world keeps one
    under that name (a hubspot ticket never stands for a contact), else from the app's collections,
    else from all. Used for a failed manifest goal: such an object means the goal failed for a
    reason the engine cannot state, and the line is withheld (docs: silent when unsure)."""
    if not address and not required:
        return False
    world = world or {}
    app_keys = [k for k in world if _same_app(str(k), app)]
    own_app = bool(app_keys)
    collection_keys = [c for k in app_keys if isinstance(world[k], dict) for c in world[k] if _same_collection(str(c), collection or "")]
    for view in views:
        if (own_app and not _same_app(view["app"], app)) or (new_only and view["kind"] != "added"):
            continue
        if collection_keys and not any(_same_collection(p, collection or "") for p in view.get("path", [])[1:]):
            continue
        if not _all_forms_in(address, view["identity"]):
            continue
        if not all(any(alternative_holds(view, alt) for alt in alts) for alts in required):
            continue
        if any(term_holds(view, t) for t in negated):
            continue
        return True
    return False


def probe_clauses(final_world, assertion: dict, *, initial_json: dict | None = None, final_json: dict | None = None,
                  calls: tuple = (), app: str = "") -> tuple[list[str], list[str], dict[str, bool | None], bool | None]:
    """(failing clauses, required identity keys, step results, identity present) for a failed positive check.

    First the address-only variant (identity present at all?), then one content key at a time on
    top of it. Handlers that raise without a content key (they index it directly) are probed by
    dropping one key at a time from the full assertion instead. Identity presence is True only
    when the grader found the address on its own; it is False only when, in addition, nothing the
    attempt added or called carries the address values (so "no such X" cannot be wrong); otherwise
    None, and the renderer makes no existence claim."""
    params, address, content = _split_params(assertion)
    results: dict[str, bool | None] = {}
    base = {"type": assertion["type"], **address}
    base_verdict = check(final_world, base) if address else None
    results["address_only"] = base_verdict
    present: bool | None = True if base_verdict is True else None
    if address and base_verdict is False and not content and final_json is not None:
        # judge 37/69/70/121/136: the record the check names by its id exists, only somewhere else (a ticket in another
        # group): the requirement is the state that record must reach, never a record to create
        from .native import OBJECT_ID_KEYS

        ids = {k: v for k, v in address.items() if k in OBJECT_ID_KEYS and isinstance(v, (str, int)) and not isinstance(v, bool)}
        if ids and len(ids) < len(address):
            results["id_only"] = check(final_world, {"type": assertion["type"], **ids})
            node = id_record(final_json, app, ids) if results["id_only"] is True else None
            if node is not None:
                results["id_record"] = {k: node.get(k) for k in address if k not in ids and k in node}
                return [], [], results, None
    if address and base_verdict is False:
        evidence = (identity_evidence(initial_json, final_json, address, calls, app)
                    if initial_json is not None and final_json is not None else True)
        if not evidence:
            return [], sorted(address), results, False  # no object with this identity at all
        present = None  # the grader needs a content key it did not get; say nothing about existence
    failing: list[str] = []
    if base_verdict is not None or not address:
        for key, value in content.items():
            verdict = check(final_world, {**base, key: value})
            results[key] = verdict
            if verdict is False:
                failing.append(key)
    elif base_verdict is None:
        # drop-one: the handler needs its content key to run at all
        for key in params:
            verdict = check(final_world, {"type": assertion["type"], **{k: v for k, v in params.items() if k != key}})
            results["drop:" + key] = verdict
            if verdict is True:
                failing.append(key)
        if not failing:
            # Dropping any key alone does not rescue the check; the content keys are reported as
            # failing, which is all the handler lets us know (no identity claim either way).
            return sorted(content), [], results, None
    if not failing and content:
        full = check(final_world, {"type": assertion["type"], **params})
        if full is False:
            failing = sorted(content)  # clauses fail only in combination
    if present is True:
        # list-element focus: which elements of a failing list value are missing on their own
        for key in failing:
            value = content.get(key)
            if isinstance(value, list) and len(value) > 1 and all(isinstance(v, str) for v in value):
                verdicts = [check(final_world, {**base, key: [v]}) for v in value]
                missing = tuple(i for i, v in enumerate(verdicts) if v is False)
                if missing and len(missing) < len(value) and all(v is not None for v in verdicts):
                    results["elements:" + key] = missing
    return failing, [], results, present


def probe_effect_required(timeline: Timeline, assertion: dict, initial_json: dict, final_json: dict, final_world,
                          calls: tuple = (), app: str = "") -> dict[str, Any]:
    """A negative-marked check that is False on the untouched world needs the effect to exist and its
    content to avoid something. Substituting each content value by a nonce no text contains tells
    which key carries the forbidden content: the variant passes exactly when the identity exists
    and that content is the only failing clause. The offending call is the first mutating step after
    which the variant passes while the real check fails; the offending object is the added/changed
    object of that step whose removal makes the variant fail again."""
    params, address, content = _split_params(assertion)
    out: dict[str, Any] = {"failing_clauses": (), "required_keys": (), "identity_present": None}
    forbidden_key, variant = None, None
    for key, value in content.items():
        nonce = _nonce_like(value)
        if nonce is None:
            continue
        candidate = {**assertion, key: nonce}
        if check(final_world, candidate) is True:
            forbidden_key, variant = key, candidate
            break
    if forbidden_key is None:
        evidence = identity_evidence(initial_json, final_json, address, calls, app) if address else True
        if address and not evidence:
            out["required_keys"] = tuple(sorted(address))
            out["identity_present"] = False
        return out
    out["failing_clauses"] = (forbidden_key,)
    out["identity_present"] = True
    for inv, before, after, _tool in timeline.mutating_steps():
        world_after = timeline.world(after)
        if check(world_after, variant) is True and check(world_after, assertion) is False:
            out["offending_invocation"] = inv
            before_json, after_json = timeline.json(before), timeline.json(after)
            if before_json is not None and after_json is not None:
                from automationbench_v1.capture import world_patch

                objects: list = []
                _objects(world_patch(before_json, after_json) or {}, (), before_json, objects)
                for path, kind, key in objects[:MAX_OBJECTS]:
                    if kind == "removed":
                        continue
                    ablated = _ablate(after_json, before_json, path, kind, key)
                    world = load_world_safe(ablated) if ablated is not None else None
                    if world is None:
                        continue
                    # the object this step produced is the offender when, without it, the check
                    # passes again (another, clean effect remains) or the identity disappears
                    if check(world, assertion) is True or check(world, variant) is False:
                        out["offending_object"] = _describe(after_json, path, kind, key)
                        break
            break
    return out


def probe_count(final_world, assertion: dict) -> int | None:
    """The attempt's own number of matching objects: for an exact `count`, the first n the grader
    accepts; for `min_count`, the largest n it accepts (every smaller n is accepted too)."""
    key = next((k for k in CARDINALITY_KEYS if k in assertion), None)
    if key is None:
        return None
    best: int | None = None
    for n in range(0, MAX_COUNT + 1):
        verdict = check(final_world, {**assertion, key: n})
        if verdict is None:
            return None
        if verdict is True:
            if key != "min_count":
                return n
            best = n
        elif best is not None:
            break
    return best


def probe_item(timeline: Timeline, assertion: dict, polarity: str, outcome: str, *, calls: tuple = (),
               app: str = "") -> ProbeResult:
    """Full probe of one native assertion against the recorded attempt."""
    if not timeline.available:
        return ProbeResult(status="none", reason="no_snapshots")
    initial_json = timeline.json(timeline.initial_digest)
    final_json = timeline.json(timeline.final_digest)
    if initial_json is None or final_json is None:
        return ProbeResult(status="none", reason="snapshot_unavailable")
    final_world = timeline.world(timeline.final_digest)
    flips, direction, initial, final, _ = probe_flips(timeline, assertion)
    kwargs: dict[str, Any] = {"flips": tuple(flips), "flip_direction": direction, "initial_result": initial,
                              "final_result": final}
    reasons = []
    if final is None:
        return ProbeResult(status="none", reason="handler_raised", **kwargs)
    negative = polarity == "forbid"
    if outcome == "failed":
        if negative and initial is False:
            # A "negative" check that already failed on the untouched world is not a pure
            # prohibition: it needs the effect to exist (e.g. *_body_not_contains). Probe it as a
            # requirement first (even when an object's removal would rescue it: that object is the
            # one with the forbidden content, not a forbidden effect as such): the nonce variant says
            # whether the effect is missing or its content is forbidden, and which call produced it.
            kwargs.update(probe_effect_required(timeline, assertion, initial_json, final_json, final_world,
                                                calls=calls, app=app))
            reasons.append("negative_check_requires_effect")
        elif negative or polarity == "keep":
            hits = probe_objects(initial_json, final_json, assertion, want=True)
            if hits:
                kwargs["offending_object"] = hits[0]
                if len(hits) > 1:
                    reasons.append(f"{len(hits)}_objects_each_sufficient")
            else:
                reasons.append("no_single_object_explains_failure")
        else:
            failing, required, results, present = probe_clauses(final_world, assertion, initial_json=initial_json,
                                                                final_json=final_json, calls=calls, app=app)
            kwargs["failing_clauses"] = tuple(failing)
            kwargs["required_keys"] = tuple(required)
            kwargs["identity_present"] = present
            if "id_record" in results:
                kwargs["id_record_fields"] = results["id_record"]
            if present is True and initial_json is not None and final_json is not None:
                _p, address, _c = _split_params(assertion)
                kwargs["identity_added"] = added_identity_evidence(initial_json, final_json, address, calls, app)
            kwargs["missing_elements"] = {k[len("elements:"):]: v for k, v in results.items() if k.startswith("elements:")}
            if results.get("address_only") is False and present is None:
                # the grader fails on the address alone, but the world or the calls carry the identity
                # values: explained, without an absence claim
                reasons.append("address_only_false_identity_unverified")
            elif not failing and not required:
                reasons.append("no_clause_isolated")
    elif outcome == "passed" and not negative:
        hits = probe_objects(initial_json, final_json, assertion, want=False)
        if hits:
            kwargs["witness_object"] = hits[0]
    if any(k in assertion for k in CARDINALITY_KEYS):
        kwargs["found_count"] = probe_count(final_world, assertion)
        if kwargs["found_count"] is not None and "no_clause_isolated" in reasons:
            reasons.remove("no_clause_isolated")
        # engine-fix 9 (pattern 2): the same count on the initial world, so a line can say how many of the objects the
        # grader counted were there before the attempt
        initial_world = timeline.world(timeline.initial_digest)
        kwargs["initial_count"] = probe_count(initial_world, assertion) if initial_world is not None else None
    # "partial" means unexplained: a failure with neither an offending object, a failing clause,
    # a missing identity nor a count. Notes about how it was explained stay in `reason`.
    explained = (kwargs.get("offending_object") is not None or kwargs.get("failing_clauses")
                 or kwargs.get("required_keys") or kwargs.get("found_count") is not None
                 or "address_only_false_identity_unverified" in reasons or kwargs.get("id_record_fields") is not None)
    unexplained = outcome == "failed" and not explained
    return ProbeResult(status="partial" if unexplained else "full", reason=";".join(reasons) or None, **kwargs)
