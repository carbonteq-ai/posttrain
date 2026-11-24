"""Truth test: every rendered plain line and every per-turn note is checked against the recorded
worlds and calls, independently of the probe that produced it (docs/design.md §5, "never wrong").

For each line the checker finds the Record item it was rendered from (by re-rendering the item's
statement) and verifies the claim the line makes:

- a located line ("At turn T, <tool> …") names a call that exists at (T, index), ran that tool and
  changed the world; a native offending object is present after that call and not before it;
- "a X <address> exists, but none with …": the grader's address-only verdict on the final world
  is True and some object of the final world, or a successful call of the same app, carries the
  address values;
- "no X <address> exists" / "no such X": no object the attempt added or changed carries the
  address values and no successful mutating call of the same app does;
- "N required; M found": the grader agrees with M (exactly / at least);
- a manifest "Not done: a X <address> <content>" line: no object the attempt added (or changed, for a
  goal about an existing record) in the effect's own collection carries every address value in its
  identity fields, one alternative of every content requirement (a mention in the named text field,
  an equality in the named field, "on one line" terms within one unit of that field) and none of the
  negated terms: the line says that effect is missing, and where one exists the line is false
  whatever the grader's reason;
- a forbid line never targets a call whose content the task requires (a require item of the same
  app whose address and content the call carries, or a passed goal witnessed at that call);
- no line carries unresolved reference text; notes sit at valid turns and hold only lines located
  at that turn;
- no manifest line is shown where a native check of the same effect (same app, polarity and address,
  `abce.record.merge.same_effect`) was decided with the other outcome, and no manifest goal is shown as
  missing on an attempt the grader passed: the grader decides success;
- a shown solution comes from a sibling that passed every native check with no manifest harm and
  none of its calls carries what a forbid line of the same context forbids;
- (correction r0) with `not_done_source` "benchmark", no line, note or hint comes from a rule-file-only requirement
  (`rule_file_only_not_done_shown`, `rule_file_only_hint_shown`); a final note ("Still not done at this point: …")
  sits at the turn of the final reply (the last assistant message, without a tool call, after the last call), never
  on a successful attempt, and lists every failed benchmark requirement whose "Not done" line the context renders,
  each once and checked as that line, numbered ("(1) A; (2) B." or one numbered item per line when an item holds
  "; "), and nothing else: no item stating a count, no "must not" item (a prohibition, a record to keep, a check
  the benchmark's registry marks negative), no tool error, no passed or grader-excluded check, no rule-file-only item
  (`final_note_*`).

Violations are returned as rows {episode, task, item, kind, line}; the sampled pass streams one
evidence file at a time (memory discipline).
"""

from __future__ import annotations

import collections
import gc
import json
import random
import re
from pathlib import Path
from typing import Any

from ..record import Episode, Record, build_record, iter_episodes
from ..record.calls import error_message
from ..record.model import Item
from ..record.native import ADDRESS_KEYS, CARDINALITY_KEYS, CONTROL_KEYS
from ..record.observed import Corpus, normalize, value_forms
from ..record.probe import _get, _objects
from ..record.timeline import Timeline, check
from ..render import common, plain

UNRESOLVED = re.compile(r"the candidate's|unresolved|\{'derived'|\bcandidate\.|\bjoined\.|\bjoin\.|\blookup\.|\bselected\.|"
                        r"\bselection\.|\bmember\.|\baggregate\.|\bpolicy\.")
COUNT_LINE = re.compile(r"(?:at least|exactly) (\d+) .*required; (\d+) found")
TURN_PREFIX = re.compile(r"^At turn (\d+), ")
TURNS_PREFIX = re.compile(r"^At turns ((?:\d+, )*\d+), ")
NONCE = "zq9x-nonce-never-present-7k3v"
# "<object> has <thing>" check types (kept here so the checker does not depend on the renderer's table)
THINGS = ("message", "note", "comment", "reply", "tag", "label", "signer", "member", "event", "property")


# ---------------------------------------------------------------------------------------------
# Evidence helpers (worlds and calls)
# ---------------------------------------------------------------------------------------------
def _forms(value: Any) -> list[str]:
    forms = [f for f in value_forms(value) if f]
    if isinstance(value, dict):
        return [f for v in value.values() for f in _forms(v)]
    return forms


def _all_forms_in(values: list[Any], text: str) -> bool:
    """Every value (one of its forms) occurs in `text`."""
    for value in values:
        forms = _forms(value)
        if forms and not any(Corpus._form_in(f, text) for f in forms):
            return False
    return True


def _address_values(item: Item) -> list[Any]:
    return [v.raw for v in item.address.values() if not (isinstance(v.raw, dict) and "unresolved" in v.raw)]


def _content_values(item: Item) -> list[Any]:
    return [v.raw for v in item.content.values() if not (isinstance(v.raw, dict) and "unresolved" in v.raw)]


def _object_texts(world: dict | None, *, only_added_from: dict | None = None) -> list[str]:
    """Normalised JSON of every object of a world (dicts inside lists, any depth ≤ 4); with
    `only_added_from`, only objects added or changed since that world."""
    if world is None:
        return []
    out: list[str] = []
    if only_added_from is not None:
        from automationbench_v1.capture import world_patch

        patch = world_patch(only_added_from, world)
        objects: list = []
        _objects(patch or {}, (), only_added_from, objects)
        for path, kind, key in objects:
            if kind == "removed":
                continue
            try:
                node = _get(world, path + (key,))
            except (KeyError, IndexError, TypeError):
                continue
            out.append(normalize(json.dumps(node, sort_keys=True, default=str)))
        return out

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


def _requirements(item: Item) -> tuple[list[list[dict]], list[tuple[str, str, Any]]]:
    """(content requirements, negated terms) as a line states them: a term is (field, mode, value); each
    requirement a list of alternatives (the branches of one `any`), each alternative {"terms", "negated",
    "sets"}: the terms that must all hold, those that must not, and the "on one line" sets {"scope",
    "terms", "excluded"} whose terms share one unit of the named field that carries none of the
    excluded terms. Negated terms outside any `any` come apart (they must hold nowhere). Written here
    again so the checker reads the item on its own."""
    negated_keys = set(item.semantics.get("negated_keys") or ())
    excluded_keys = set(item.semantics.get("excluded_keys") or ())
    groups = item.semantics.get("alternatives") or {}
    branches = item.semantics.get("branches") or {}
    together = item.semantics.get("together") or {}
    scopes = item.semantics.get("together_scope") or {}
    modes = item.semantics.get("modes") or {}
    required: dict[Any, dict[Any, dict]] = {}
    negated = []
    for key, value in item.content.items():
        if isinstance(value.raw, dict) and "unresolved" in value.raw:
            continue
        term = (key.split("#")[0], modes.get(key) or "mentions", value.raw)
        g, t = groups.get(key), together.get(key)
        if g is None and t is None and key in negated_keys:
            negated.append(term)
            continue
        requirement = ("group", g) if g is not None else (("together", t) if t is not None else ("single", key))
        branch = branches.get(key, ("set", t) if t is not None else key) if g is not None else 0
        alt = required.setdefault(requirement, {}).setdefault(branch, {"terms": [], "negated": [], "sets": {}, "nested": []})
        if t is not None:
            s = alt["sets"].setdefault(t, {"scope": scopes.get(str(t), "line"), "terms": [], "excluded": []})
            (s["excluded"] if key in excluded_keys else s["terms"]).append(term)
        elif key in negated_keys:
            alt["negated"].append(term)
        else:
            alt["terms"].append(term)
    # an `any` nested in a branch of another `any` (branch ids "0.1") is a requirement of that branch alone
    from ..record.probe import group_parents

    parents = group_parents({req[1]: [str(b) for b in alts] for req, alts in required.items() if req[0] == "group"})

    def finish(requirement: Any) -> list[dict]:
        return [{**a, "sets": list(a["sets"].values()),
                 "nested": [finish(("group", g)) for g, parent in parents.items()
                            if parent is not None and parent[0] == requirement[1] and parent[1] == str(branch)]}
                for branch, a in required[requirement].items()]

    return [finish(req) for req in required if not (req[0] == "group" and parents.get(req[1]) is not None)], negated


def _addressed_object_texts(after: dict | None, args: dict) -> list[str]:
    """Normalised JSON of the objects of `after` whose own id (`id`, `row_id`, `*_id`) the call's arguments
    name: a row rewritten with its own values is not a changed object, but the call addressed it, so its
    cells are what the call acted on."""
    if after is None or not isinstance(args, dict):
        return []
    ids = {str(v) for k, v in args.items() if (k == "id" or k.endswith("_id")) and isinstance(v, (str, int)) and not isinstance(v, bool)}
    if not ids:
        return []
    out = []
    for node in _walk_objects(after):
        if isinstance(node, dict) and any(str(node.get(k)) in ids for k in ("id", "row_id", "rowId") if k in node):
            out.append(normalize(json.dumps(node, sort_keys=True, default=str)))
    return out


def _changed_object_texts(before: dict | None, after: dict | None) -> list[str]:
    """Normalised JSON of every object (a dict inside a list, depth ≤ 4) of `after` that is not in
    `before` byte for byte: added objects and the containers whose sub-records changed (a
    conversation whose messages grew), so a field of the container counts as touched too."""
    if before is None or after is None:
        return []
    seen = {normalize(json.dumps(o, sort_keys=True, default=str)) for o in _walk_objects(before)}
    return [t for t in (normalize(json.dumps(o, sort_keys=True, default=str)) for o in _walk_objects(after)) if t not in seen]


def _walk_objects(node: Any, depth: int = 0, out: list | None = None) -> list:
    out = [] if out is None else out
    if depth > 4:
        return out
    if isinstance(node, dict):
        for v in node.values():
            _walk_objects(v, depth + 1, out)
    elif isinstance(node, list):
        for v in node:
            if isinstance(v, dict):
                out.append(v)
            _walk_objects(v, depth + 1, out)
    return out


def _app_matches(call_app: str, item_app: str) -> bool:
    a, b = call_app.replace("_", ""), item_app.replace("_", "")
    return a == b or a.startswith(b) or b.startswith(a)


def _successful_calls_with(record: Record, item: Item, values: list[Any], *, mutating: bool) -> list:
    found = []
    for call in record.calls:
        if call.error is not None or not _app_matches(call.app, item.effect.app):
            continue
        if mutating and call.mutated is not True:
            continue
        text = normalize(call.tool + " " + json.dumps(call.args, sort_keys=True, default=str))
        if values and _all_forms_in(values, text):
            found.append(call)
    return found


def _call_at(record: Record, item: Item):
    loc = item.location
    if loc.kind != "at_call" or loc.call is None:
        return None
    return next((c for c in record.calls if c.turn == loc.call.turn and c.index == loc.call.index), None)


def _assertion(ep: Episode, item: Item) -> dict | None:
    if item.origin != "native" or item.type == "tool_error":
        return None
    index = int(item.id.split(":")[1])
    assertions = ep.task_data.get("assertions") or []
    return assertions[index] if index < len(assertions) else None


def _object_node(world: dict | None, obj: dict) -> Any:
    if world is None:
        return None
    path = tuple(int(p) if p.isdigit() else p for p in obj["collection"].split("/") if p != "")
    try:
        return _get(world, path + (obj["key"],))
    except (KeyError, IndexError, TypeError):
        return None


def _object_present(world: dict | None, obj: dict) -> bool:
    """The probe's offending object (collection path + key + fields) is in `world` with those fields.
    A scalar element (an email in a member list) is looked up anywhere in its list: such lists may be
    kept sorted, so its index moves as later elements arrive."""
    fields = obj.get("fields") or {}
    if "value" in fields and not isinstance(fields["value"], (dict, list)):
        path = tuple(int(p) if p.isdigit() else p for p in obj["collection"].split("/") if p != "")
        try:
            container = _get(world, path) if world is not None else None
        except (KeyError, IndexError, TypeError):
            container = None
        if isinstance(container, list):
            return any(str(v) == str(fields["value"]) for v in container if not isinstance(v, (dict, list)))
        if isinstance(container, dict):  # a changed scalar field of a record (a ticket's status)
            return obj["key"] in container and str(container[obj["key"]]) == str(fields["value"])
        return False
    node = _object_node(world, obj)
    if node is None:
        return False
    if isinstance(node, dict):
        if obj.get("kind") == "changed":
            # a changed record is described from the final world; at the located call only its
            # identity must match (later calls may have changed its other fields again)
            ids = {k: v for k, v in fields.items() if k in ("id", "row_id", "rowId", "ticket_id", "message_id", "email",
                                                            "name", "subject", "ts", "spreadsheet_id", "worksheet_id")}
            fields = ids or fields
        return all(k in node and (str(node[k]) == str(v) or (isinstance(v, str) and v.endswith("…") and str(node[k]).startswith(v[:-1])))
                   for k, v in fields.items())
    if isinstance(node, list):
        return True  # a whole collection added under a key: its presence is what the ablation removed
    return "value" in fields and str(node) == str(fields["value"])


def _object_produced_by(before: dict | None, after: dict | None, obj: dict, args_text: str = "") -> bool:
    """The object is in the after-world as described and the call produced it: absent before (an
    added object) or different before (a changed one); a record rewritten with identical values is
    accepted when the call's arguments name its id (the grader's private update record changed)."""
    if not _object_present(after, obj):
        return False
    fields = obj.get("fields") or {}
    if "value" in fields and not isinstance(fields["value"], (dict, list)):
        return not _object_present(before, obj)  # a scalar element: absent from its list before the call
    node_before, node_after = _object_node(before, obj), _object_node(after, obj)
    if node_before != node_after:
        return True
    if node_before is None:
        return True
    if isinstance(node_after, dict):
        # the object itself is unchanged by this call, but the call named it (a label applied to a
        # message, a row rewritten with its own values): the grader's flip happened here
        ids = [v for k, v in node_after.items() if k in ("id", "row_id", "rowId") and isinstance(v, (str, int))]
        return any(_all_forms_in([v], args_text) for v in ids)
    return False


# ---------------------------------------------------------------------------------------------
# Line attribution
# ---------------------------------------------------------------------------------------------
def _body_lines(first_message: str) -> list[str]:
    if common.FAILED_HEADING not in first_message and common.SUCCESS_HEADING not in first_message:
        return []  # no feedback block (every line withheld): the request text is not a feedback line
    body = first_message
    for heading in (common.FAILED_HEADING, common.SUCCESS_HEADING):
        if heading in body:
            body = body.split(heading, 1)[1]
    if "Correct solution:" in body and body.find("Correct solution:") < body.find(common.CLOSING):
        # the solution block sits before the closing; statement lines (if any) follow a heading
        parts = body.split(common.CLOSING)[0]
        body = parts.split(common.FAILED_HEADING)[-1] if common.FAILED_HEADING in parts else ""
    else:
        body = body.split(common.CLOSING)[0]
    return [line.strip() for line in body.split("\n") if line.strip() and not line.startswith("Correct solution:")
            and not line.startswith("Not listed:")]


def _statements(record: Record, items: list[Item], expected_values: bool) -> dict[str, list[Item]]:
    """statement text (without any turn prefix) -> items that render to it."""
    out: dict[str, list[Item]] = collections.defaultdict(list)
    for item in items:
        try:
            text = plain.statement(item, record, expected_values, with_turn=False)
        except Exception:  # noqa: BLE001 - a renderer crash is itself a finding
            continue
        if text:
            out[text].append(item)
    return out


WHY_MARKERS = re.compile(r" \(also required: | \(the task also requires: |; this task forbids it when | \(the task's rule: ")
BOOKKEEPING = re.compile(r"(_at|_time|timestamp|modified|updated|version|etag|_date|_count|count|total|_by)$", re.IGNORECASE)
REMOVED_KINDS = {"cause_diff", "hint_rule_population", "solution_note", "harm_as_made", "not_met"}  # fix round
SOLUTION_HEADINGS = ("Correct solution:", "Solution that passed the task's checks:")


def _whole(name: str, text: str) -> bool:
    return bool(re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", text, re.IGNORECASE))


def _solution_heading(first: str) -> str | None:
    found = [(first.find(h), h) for h in SOLUTION_HEADINGS if h in first]
    return min(found)[1] if found else None


def _solution_block(first: str) -> str | None:
    heading = _solution_heading(first)
    if heading is None:
        return None
    text = first.split(heading, 1)[1].split(common.CLOSING)[0]
    return text.split(common.FAILED_HEADING)[0].split(common.SUCCESS_HEADING)[0]


def fix_cause_match(line: str):
    from .. import fix

    return fix.CAUSE_LINE.match(line)


def _flat_all(node: Any) -> list:
    if isinstance(node, dict) and node.get("op") == "all":
        out = []
        for a in node.get("args") or []:
            out.extend(_flat_all(a))
        return out
    return [node] if isinstance(node, dict) and node.get("op") else []


def _node_paths(node: Any) -> list[tuple]:
    out = []
    if isinstance(node, dict):
        if node.get("kind") in ("field", "input") and isinstance(node.get("path"), list):
            out.append(tuple(node["path"]))
        for v in node.values():
            out.extend(_node_paths(v))
    elif isinstance(node, list):
        for v in node:
            out.extend(_node_paths(v))
    return out


def _reads(node: Any, root: str) -> bool:
    return any(p[:1] == (root,) for p in _node_paths(node))


def _request_only(node: Any) -> bool:
    paths = _node_paths(node)
    return bool(paths) and all(p[:1] == ("request",) for p in paths)


def _is_selector(node: Any, keys: list[str]) -> bool:
    if not isinstance(node, dict) or node.get("op") not in ("eq", "in"):
        return False
    left = node.get("left") or {}
    path = left.get("path") if isinstance(left, dict) and left.get("kind") == "field" else None
    return bool(path and len(path) == 2 and path[0] == "request" and path[1] in keys)


def _expressible(node: Any) -> bool:
    """The checker's own classification of what the compiler states as address or content (independent of
    abce.why): eq/in/ne with an effect field operand, mentions on an effect text, and junctions of those."""
    if not isinstance(node, dict):
        return False
    op = node.get("op")
    if op in ("all", "any", "not"):
        args = node.get("args") or ([node.get("arg")] if op == "not" else [])
        return bool(args) and all(_expressible(a) for a in args)
    if op == "proven":
        return _expressible(node.get("arg"))
    if op in ("eq", "in", "ne"):
        return any(isinstance(x, dict) and x.get("kind") == "field" and (x.get("path") or [None])[0] == "effect"
                   for x in (node.get("left"), node.get("right")))
    if op in ("mentions", "mentions_together"):
        t = node.get("text")
        return isinstance(t, dict) and t.get("kind") == "field" and (t.get("path") or [None])[0] == "effect"
    return False


def _population_member(initial: dict, population: dict, candidate: dict | None) -> dict | None:
    """The candidate's population member as the environment sees it: a sheet row's cells, or the raw initial
    record whose identity path value is the candidate's key."""
    if not candidate or not population:
        return None
    key = str((candidate.get("identity") or [candidate.get("key")])[-1])
    adapter = population.get("adapter")
    if adapter == "google_sheets.rows@1":
        for row in ((initial.get("google_sheets") or {}).get("rows")) or []:
            if (isinstance(row, dict) and str(row.get("row_id")) == key and row.get("spreadsheet_id") == population.get("spreadsheet_id")
                    and row.get("worksheet_id") == population.get("worksheet_id")):
                return dict(row.get("cells") or {})
        return None
    if adapter == "initial.records@1":
        node: Any = initial
        for step in (population.get("path") or [])[2:]:
            node = node.get(step) if isinstance(node, dict) else None
        for rec in node if isinstance(node, list) else []:
            ident: Any = rec
            for step in population.get("identity_path") or ["id"]:
                ident = ident.get(step) if isinstance(ident, dict) else None
            if ident is not None and str(ident) == key:
                return rec
    return None


# ---------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------
def _rule_file_only(item: Item) -> bool:
    """Correction r0, the checker's own reading: a failed requirement (not a prohibition, a record to keep or a tool
    error) built from the rule file and not merged into a benchmark assertion's item."""
    return item.type != "tool_error" and item.polarity == "require" and item.origin == "manifest"


def _benchmark_required_failed(item: Item) -> bool:
    """A required benchmark check (a native assertion result, polarity require, not a check the benchmark marks
    negative) that failed in the final graded state."""
    return (item.origin == "native" and item.type != "tool_error" and item.polarity == "require" and item.outcome == "failed"
            and not _negative_check(item))


def _states_count(item: Item, text: str | None) -> bool:
    """The checker's reading of a count item: a cardinality check, or a text saying "N required; M found"."""
    return bool(item.cardinality) or item.probe.found_count is not None or bool(
        text and re.search(r"(?:at least|exactly) \d+ .*required; \d+ found", text))


def _negative_check(item: Item) -> bool:
    """The checker's own reading of a "must not" check: the benchmark's registry marks the type negative (a
    `*_not_contains` check the probe re-read as needing the effect is one), or the Record says so."""
    if item.polarity in ("forbid", "keep") or item.semantics.get("effect_required") or item.polarity_source == "probe":
        return True
    if item.origin == "native" and item.type != "tool_error":
        from ..record.native import negative_marker

        return negative_marker(item.type) is True
    return False


class Checker:
    def __init__(self, ep: Episode, record: Record, siblings: list[Episode] | None = None):
        self.ep, self.record, self.siblings = ep, record, siblings or []
        self.tl = Timeline(ep)
        self.rows: list[dict] = []
        self.checked = 0
        self._final_json = self.tl.json(self.tl.final_digest) if self.tl.available else None
        self._initial_json = self.tl.json(self.tl.initial_digest) if self.tl.available else None
        self._final_world = self.tl.world(self.tl.final_digest) if self.tl.available else None
        self._all_texts: list[str] | None = None
        self._added_texts: list[str] | None = None
        self._added_objects: list[dict[str, Any]] | None = None

    # evidence caches
    def all_texts(self) -> list[str]:
        if self._all_texts is None:
            self._all_texts = _object_texts(self._final_json)
        return self._all_texts

    def added_texts(self) -> list[str]:
        if self._added_texts is None:
            self._added_texts = _object_texts(self._final_json, only_added_from=self._initial_json) if self._initial_json else []
        return self._added_texts

    def added_objects(self, app: str, collection: str | None, *, new_only: bool) -> list[dict[str, Any]]:
        """Views (identity text, full text, fields) of the objects added (or, unless `new_only`, changed)
        in the effect's own collection when the final world keeps one under that name, else in the
        app's collections, else in every collection; from the recorded initial and final worlds
        (abce.record.probe.added_objects)."""
        from ..record.probe import _same_collection, added_objects

        if self._added_objects is None:
            self._added_objects = (added_objects(self._initial_json, self._final_json)
                                   if self._initial_json is not None and self._final_json is not None else [])
        world = self._final_json or {}
        app_keys = [k for k in world if _app_matches(str(k), app)]
        collections = [c for k in app_keys if isinstance(world[k], dict) for c in world[k] if _same_collection(str(c), collection or "")]
        return [v for v in self._added_objects
                if (not app_keys or _app_matches(v["app"], app)) and (v["kind"] == "added" or not new_only)
                and (not collections or any(_same_collection(p, collection or "") for p in v.get("path", [])[1:]))]

    def fail(self, item: Item | None, kind: str, line: str, **extra: Any) -> None:
        self.rows.append({"episode": self.record.episode_id, "task": self.record.task, "item": item.id if item else None,
                          "type": item.type if item else None, "kind": kind, "line": line[:300], **extra})

    # -- one line ---------------------------------------------------------------------------
    def check_line(self, item: Item, line: str, *, turn: int | None, in_note: bool) -> None:
        self.checked += 1
        if UNRESOLVED.search(line):
            self.fail(item, "unresolved_reference", line)
        if item.origin == "manifest":
            self._check_native_agrees(item, line)
            if plain.contradictory_fields(item):
                self.fail(item, "condition_contradicts_itself", line, fields=plain.contradictory_fields(item))
        if item.candidate and f" for {json.dumps(str(item.candidate.get('label')))}" in line and not plain.candidate_tied(item):
            self.fail(item, "candidate_label_not_tied_to_line", line, candidate=item.candidate.get("label"))
        if WHY_MARKERS.search(line):
            if not getattr(self, "version_c", False):
                self.fail(item, "why_clause_outside_version_c", line)
            elif not getattr(self, "clauses", True):
                self.fail(item, "why_clause_not_selected", line)
            self._check_why(item, line)
        kind = common.item_kind(item)
        if getattr(self, "version_c", False) and kind == "prohibited":
            self._check_written_values(item, line)
            self._check_harm_target(item, line)
        if getattr(self, "version_c", False) and line.startswith("Not met: "):
            self._check_not_met(item, line)
        if kind in ("prohibited", "changed", "tool_error"):
            self._check_located(item, line, turn, in_note)
            if kind != "tool_error":
                self._check_forbid_not_required(item, line)
                if "forbids it as made" in line:
                    self._check_as_made(item, line)
            return
        if re.search(r"(?:^|: )the attempt never read ", line):
            self._check_never_read(item, line)
        # missing (require failed)
        if turn is not None and item.location.kind == "at_call":
            self._check_located(item, line, turn, in_note)
        if " exists, but " in line:
            self._check_exists(item, line)
        elif re.search(r"(?:^|: )the attempt (?:sent|created) no ", line):
            self._check_produced_nothing(item, line)
        elif re.search(r"(?:^|: )no (?:such )?", line) and " exists" in line:
            self._check_absent(item, line)
        elif item.origin == "manifest" and line.startswith("Not done: "):
            self._check_not_done(item, line)
        m = COUNT_LINE.search(line)
        if m:
            self._check_count(item, line, int(m.group(2)))

    def _check_native_agrees(self, item: Item, line: str) -> None:
        from ..record.merge import asks_more_on_a_shared_field, same_effect

        if item.polarity == "require" and self.record.success:
            self.fail(item, "goal_missing_on_successful_attempt", line)
            return

        for other in self.record.items:
            # a native check that names one recipient of a goal for a message to two decides less than the
            # goal asks: its pass does not contradict "Not done: a message to both"
            if (other.origin == "native" and other.type != "tool_error" and other.polarity == item.polarity
                    and other.outcome in ("passed", "failed") and other.outcome != item.outcome
                    and same_effect(other, item) and not asks_more_on_a_shared_field(item, other)):
                self.fail(item, "contradicts_native_check", line, native=other.id)
                return

    def _check_located(self, item: Item, line: str, turn: int | None, in_note: bool) -> None:
        call = _call_at(self.record, item)
        if call is None:
            self.fail(item, "located_without_call", line)
            return
        if turn is not None and turn != call.turn:
            self.fail(item, "turn_mismatch", line, said=turn, actual=call.turn)
        if call.tool not in line:
            self.fail(item, "tool_not_named", line, tool=call.tool)
        if item.type == "tool_error":
            if call.error is None:
                self.fail(item, "tool_error_without_error", line)
            elif normalize(error_message(call.error)[:100]) not in normalize(line):
                # the line must carry the tool's own message (its suggestion list may be left out)
                self.fail(item, "error_text_mismatch", line)
            return
        if call.mutated is not True:
            self.fail(item, "located_call_did_not_change_world", line)
            return
        obj = item.probe.offending_object
        before, after = self.tl.json(call.before_digest), self.tl.json(call.after_digest)
        if getattr(self, "version_c", False) and item.polarity in ("forbid", "keep"):
            from .. import fix

            # G1: the noun of the line is the call's: same app and, where the snapshots tell, its collection changed
            if not _app_matches(call.app, item.effect.app):
                self.fail(item, "harm_call_app_mismatch", line, app=call.app)
            target = fix.effect_collection(item, self.ep.task_data.get("initial_state"))
            if target and before is not None and after is not None and \
                    (before.get(target[0]) or {}).get(target[1]) == (after.get(target[0]) or {}).get(target[1]):
                self.fail(item, "harm_collection_unchanged", line, collection=list(target))
            if item.polarity == "keep" and obj is None:
                # R3: "changed" is a diff: the address must be in an object the call changed
                values = _address_values(item)
                if not values or not any(_all_forms_in(values, t) for t in _changed_object_texts(before, after)):
                    self.fail(item, "changed_claim_without_diff", line)
        if obj is not None:
            if not _object_present(after, obj):
                self.fail(item, "offending_object_absent_after_call", line)
            elif not _object_produced_by(before, after, obj, normalize(json.dumps(call.args, sort_keys=True, default=str))):
                self.fail(item, "offending_object_already_present_before_call", line)
        else:
            # no object isolated: the call's arguments or the objects it changed must carry the
            # address values (or, when the address is an alias the attempt never wrote, the content)
            args = normalize(json.dumps(call.args, sort_keys=True, default=str))
            texts = [args, *_changed_object_texts(before, after), *_addressed_object_texts(after, call.args)]
            negated = set(item.semantics.get("negated_keys") or ())
            groups = item.semantics.get("alternatives") or {}
            branches = item.semantics.get("branches") or {}
            # positive content as requirements: every plain conjunct; of an `any`, every term of one branch
            requirements: dict[Any, dict[Any, list]] = {}
            for k, v in item.content.items():
                if k in negated or (isinstance(v.raw, dict) and "unresolved" in v.raw):
                    continue
                g = groups.get(k)
                requirements.setdefault(("single", k) if g is None else ("group", g), {}).setdefault(
                    branches.get(k, k) if g is not None else 0, []).append(v.raw)  # a set without branch numbers: each term alone (weaker)

            def content_in(text: str) -> bool:
                return all(any(_all_forms_in(values, text) for values in alternatives.values())
                           for alternatives in requirements.values())

            address = _address_values(item)
            if address and any(_all_forms_in(address, t) for t in texts):
                pass
            elif requirements and any(content_in(t) for t in texts):
                pass
            elif address or requirements:
                self.fail(item, "address_not_in_located_call", line)
            # a line restricted to the branches the object matched: those branches must be in the call's objects
            for g, shown in (item.semantics.get("matched_branches") or {}).items():
                alternatives = requirements.get(("group", int(g) if str(g).isdigit() else g), {})
                values = [v for b, v in alternatives.items() if str(b) in shown]
                if values and not any(_all_forms_in(v, t) for v in values for t in texts):
                    self.fail(item, "matched_branch_not_in_located_call", line, group=g)

    def _check_forbid_not_required(self, item: Item, line: str) -> None:
        """R2: a forbid line must not cover an action the task requires. It does when everything the
        line names (address and content) is among what some require item of the same app names and
        that require item is met by, or asks for, this very call (its address is in the call's
        arguments, or it was witnessed here). A line that names something more (another address
        part, extra content) is narrower than the requirement and is allowed."""
        call = _call_at(self.record, item)
        if call is None:
            return
        args = normalize(json.dumps(call.args, sort_keys=True, default=str))
        negated_keys = set(item.semantics.get("negated_keys") or ())
        negated = {f for k, v in item.content.items() if k in negated_keys for f in _forms(v.raw)}
        forbids = {f for k, v in {**item.address, **item.content}.items() if k not in negated_keys
                   and not (isinstance(v.raw, dict) and "unresolved" in v.raw) for f in _forms(v.raw)}
        if not forbids:
            return
        for other in self.record.items:
            if other.polarity != "require" or other.type == "tool_error" or other.id == item.id:
                continue
            if not _app_matches(call.app, other.effect.app) and other.effect.app != item.effect.app:
                continue
            if other.effect.object != item.effect.object and other.effect.object not in ("", other.effect.app) \
                    and item.effect.object not in ("", item.effect.app):
                continue  # a draft is not a message: a different effect of the same app
            needs_address = _address_values(other)
            # what the requirement must NOT contain is not something it needs; a value that is one
            # of several alternatives is not needed either (the goal is met without it)
            avoided = set(other.semantics.get("forbidden_content_keys") or ()) | set(other.semantics.get("negated_keys") or ())
            if other.semantics.get("effect_required"):
                avoided |= set(other.probe.failing_clauses)
            groups = other.semantics.get("alternatives") or {}
            branches = other.semantics.get("branches") or {}
            group_branches: dict[Any, set] = {}
            for k, g in groups.items():
                group_branches.setdefault(g, set()).add(branches.get(k, k))
            needs_content = [v.raw for k, v in other.content.items() if k not in avoided and k.split("#")[0] not in avoided
                             and not (k in groups and len(group_branches[groups[k]]) > 1)
                             and not (isinstance(v.raw, dict) and "unresolved" in v.raw)]
            witnessed_here = (other.location.kind == "after_turn" and other.location.call is not None
                              and (other.location.call.turn, other.location.call.index) == (call.turn, call.index))
            address_in_call = bool(needs_address) and _all_forms_in(needs_address, args)
            if not (witnessed_here or address_in_call):
                continue
            needed = {f for v in needs_address + needs_content for f in _forms(v)}
            if negated & needed:
                continue  # "not X" forbidden where X is required: consistent, the line is narrower
            if forbids <= needed:
                kind = "forbids_required_action" if not _content_values(item) else "forbids_required_content"
                self.fail(item, kind, line, requires=other.id)
                return

    def _check_as_made(self, item: Item, line: str) -> None:
        """A harm stated with only part of its condition ("forbids it as made") may not share its whole
        address with any requirement of the same app: the teacher could read it as forbidding the required
        action (R2)."""
        def address_forms(it: Item) -> set[str]:
            return {f for v in _address_values(it) for f in _forms(v) if len(f) >= 3}

        scalars = {f for v in _address_values(item) if not isinstance(v, list) for f in _forms(v) if len(f) >= 3}
        lists = [{f for e in v for f in _forms(e) if len(f) >= 3} for v in _address_values(item) if isinstance(v, list)]
        if not scalars and not any(lists):
            self.fail(item, "as_made_harm_without_address", line)
            return
        for other in self.record.items:
            if other.id == item.id or other.polarity != "require" or other.type == "tool_error" or other.effect.app != item.effect.app:
                continue
            theirs = address_forms(other)
            if scalars <= theirs and all(lst & theirs for lst in lists if lst):
                self.fail(item, "as_made_harm_at_a_required_address", line, requires=other.id)
                return

    def _check_never_read(self, item: Item, line: str) -> None:
        """"The attempt never read X": no successful call of the app returned or named X's address values."""
        values = _address_values(item)
        if not values:
            self.fail(item, "never_read_claim_without_identity", line)
            return
        for call in self.record.calls:
            if call.error is not None or not _app_matches(call.app, item.effect.app):
                continue
            text = normalize(json.dumps([call.args, call.result], sort_keys=True, default=str))
            if _all_forms_in(values, text):
                self.fail(item, "never_read_claim_object_returned", line, call=f"{call.turn}:{call.index}")
                return

    def _check_exists(self, item: Item, line: str) -> None:
        assertion = _assertion(self.ep, item)
        values = _address_values(item)
        if assertion is not None and self._final_world is not None and values:
            address = {k: v for k, v in assertion.items() if k in ADDRESS_KEYS}
            verdict = check(self._final_world, {"type": assertion["type"], **address})
            if verdict is not True:
                self.fail(item, "exists_claim_grader_disagrees", line, verdict=verdict)
        if not values:
            self.fail(item, "exists_claim_without_address", line)
            return
        # "<object> has <thing>": the address identifies the object; a line claiming that the thing
        # (a tag, a note, a member) exists is not what the address-only verdict established
        thing = next((t for t in THINGS if f"has_{t}" in item.type or f"missing_{t}" in item.type), None)
        if thing and re.search(rf"\b{thing} .*exists, but none", line):
            self.fail(item, "exists_claim_about_thing_not_object", line)
        in_world = any(_all_forms_in(values, t) for t in self.all_texts())
        in_calls = bool(_successful_calls_with(self.record, item, values, mutating=False))
        if not (in_world or in_calls):
            self.fail(item, "exists_claim_unsupported", line)

    def _check_absent(self, item: Item, line: str) -> None:
        from ..record.native import SCHEMA_KEYS

        values = [v.raw for k, v in item.address.items() if k not in SCHEMA_KEYS
                  and not (isinstance(v.raw, dict) and "unresolved" in v.raw)]
        if not values:
            self.fail(item, "absence_claim_without_identity", line)
            return
        if any(_all_forms_in(values, t) for t in self.all_texts()):
            self.fail(item, "absence_claim_object_exists", line)
            return
        if _successful_calls_with(self.record, item, values, mutating=True):
            self.fail(item, "absence_claim_call_made", line)

    def _check_produced_nothing(self, item: Item, line: str) -> None:
        """"The attempt sent/created no X <address>": no object the attempt added carries the address
        values, no successful mutating call of the app names them, and no object of the final world that
        carries them (a channel named by the address) has an id an added object of the app refers to."""
        from ..record.native import OBJECT_ID_KEYS, SCHEMA_KEYS

        values = [v.raw for k, v in item.address.items() if k not in SCHEMA_KEYS
                  and not (isinstance(v.raw, dict) and "unresolved" in v.raw)]
        if not values:
            self.fail(item, "produced_nothing_claim_without_identity", line)
            return
        if any(k.split("#")[0] in OBJECT_ID_KEYS for k in item.address):
            self.fail(item, "produced_nothing_claim_about_existing_record", line)
        # objects the attempt added in the item's own app (a word like "facilities" in another app's record is not a message there)
        added = [v["full"] for v in self.added_objects(item.effect.app, None, new_only=True)]
        if any(_all_forms_in(values, t) for t in added):
            self.fail(item, "produced_nothing_claim_object_added", line)
            return
        if _successful_calls_with(self.record, item, values, mutating=True):
            self.fail(item, "produced_nothing_claim_call_made", line)
            return
        for node in _walk_objects(self._final_json):
            if not isinstance(node, dict) or not _all_forms_in(values, normalize(json.dumps(node, sort_keys=True, default=str))):
                continue
            ids = [str(v) for k, v in node.items() if (k == "id" or str(k).endswith("_id")) and isinstance(v, (str, int)) and str(v)]
            if any(_all_forms_in([i], t) for i in ids for t in added):
                self.fail(item, "produced_nothing_claim_alias_referred", line)
                return

    def _check_not_done(self, item: Item, line: str) -> None:
        """A manifest "Not done" line says the effect it names is missing. It is false when an object the
        attempt added (or changed, for a goal about an existing record) in the app's collections carries
        every address value, one alternative of every content requirement and none of the negated
        values, whatever the grader's reason for failing the check (a condition the line does not
        state, a stricter comparison). A goal for a new effect is not met by a record the attempt only
        updated. Read goals leave no object and are not judged."""
        if item.effect.kind == "read":
            return
        from ..record.probe import alternative_holds, candidate_identity_value, term_holds

        address = _address_values(item)
        own_record = candidate_identity_value(item)  # a goal about record 207 is not met by record 201
        if own_record is not None:
            address.append(own_record)
        required, negated = _requirements(item)
        if not address and not required:
            return

        new_only = item.effect.kind in ("send", "create", "post")
        for view in self.added_objects(item.effect.app, item.semantics.get("collection") or item.effect.object, new_only=new_only):
            # the address in the object's identity fields (not quoted in some body); a mention in the text
            # field the term names; an equality in the named field; "on one line" terms within one unit
            if (_all_forms_in(address, view["identity"]) and all(any(alternative_holds(view, alt) for alt in alts) for alts in required)
                    and not any(term_holds(view, t) for t in negated)):
                self.fail(item, "not_done_effect_exists", line)
                return

    def _check_count(self, item: Item, line: str, found: int) -> None:
        assertion = _assertion(self.ep, item)
        if assertion is None or self._final_world is None:
            return
        key = next((k for k in CARDINALITY_KEYS if k in assertion), None)
        if key is None:
            return
        if check(self._final_world, {**assertion, key: found}) is not True:
            self.fail(item, "count_grader_disagrees", line, found=found)
        elif key == "min_count" and found < 200 and check(self._final_world, {**assertion, key: found + 1}) is True:
            self.fail(item, "count_understated", line, found=found)

    # -- the whole context ------------------------------------------------------------------
    def check_context(self, ctx: dict, *, expected_values: bool = True) -> None:
        from .. import why

        # N1 clauses and the version-(c) strengthenings apply exactly where the renderer used them
        self.version_c = ctx.get("fix_hints") is not None and expected_values
        # the fix-hint kinds the context was built with (`fix_hints.kinds` names the selected kinds when not all
        # were; "why_clause" is selected exactly when a version-(c) context carries an N1 clause renderer-side)
        from ..fix import ALL_KINDS

        report = ctx.get("fix_hints") or {}
        self.hint_kinds = set(ALL_KINDS) if "kinds" not in report else set(report["kinds"])
        self.clauses = "why_clause" in self.hint_kinds
        # correction r0: with "benchmark", no line, note or hint may come from a rule-file-only requirement
        self.benchmark_only = ctx.get("not_done_source") == "benchmark"
        self.ctx = ctx
        self.attributed: list[dict] = []  # {where, text, kind, flags}: what the sweep reports per line
        with why.enabled_as(self.version_c, clauses=self.clauses):
            self._check_context(ctx, expected_values=expected_values)
        if self.version_c:
            self._check_selected_kinds(ctx)

    def _check_selected_kinds(self, ctx: dict) -> None:
        """Only the selected fix-hint kinds: no hint of another kind in the report, no line of another kind anywhere."""
        from .kinds import SELECTABLE

        for meta in (ctx.get("fix_hints") or {}).get("lines", []):
            if meta.get("kind") not in self.hint_kinds:
                self.fail(None, "hint_kind_not_selected", meta.get("line", ""), hint_kind=meta.get("kind"))
        for entry in self.attributed:
            for name in [entry["kind"], *entry["flags"]]:
                selectable = SELECTABLE.get(name)
                if selectable is not None and selectable not in self.hint_kinds:
                    self.fail(None, "line_kind_not_selected", entry["text"], line_kind=name, hint_kind=selectable)

    def _attribute(self, where: str, text: str, attributed_as: str) -> None:
        from .kinds import classify, compatible

        kind, flags = classify(text)
        if self.version_c and kind in REMOVED_KINDS:
            self.fail(None, "removed_kind_present", text, line_kind=kind)
        if self.version_c and not compatible(kind, attributed_as):
            self.fail(None, "kind_attribution_mismatch", text, line_kind=kind, attributed=attributed_as)
        self.attributed.append({"where": where, "text": text, "kind": kind, "flags": flags, "attributed": attributed_as})

    def _check_context(self, ctx: dict, *, expected_values: bool = True) -> None:
        items = common.facts(self.record)
        by_text = _statements(self.record, items, expected_values)
        # a line that names several turns and the objects of several same-shape harms: every member is
        # checked on its own line at its own turn, and the merged text must be what the members merge to
        collapsed_harms = {line: members for line, members in plain.collapse_harms(
            plain.harm_items_shown(self.record, items), self.record, expected_values) if len(members) > 1}
        located_turns = {}
        for item in items:
            t = common.located_turn(item)
            if t is not None:
                located_turns.setdefault(t, []).append(item)
        hint_lines = {m["line"]: m for m in (ctx.get("fix_hints") or {}).get("lines", [])}
        first = ctx.get("first_message") or ""
        if self.version_c:
            # G5: the failed heading claims the attempt failed
            if common.FAILED_HEADING in first and self.ep_success():
                self.fail(None, "failed_heading_on_success", common.FAILED_HEADING)
            # G4: "Not listed: N more." counts the planned lines the budget dropped
            m_nl = re.search(r"^Not listed: (\d+) more\.$", first, re.M)
            dropped = list((ctx.get("coverage") or {}).get("dropped") or [])
            if m_nl:
                self._attribute("body", m_nl.group(0), "not_listed")
                ids = {i.id for i in self.record.items if i.outcome == "failed"} | {"tool_error", "budget"}
                if int(m_nl.group(1)) != len(dropped) or any(not set(d.split(",")) <= ids for d in dropped):
                    self.fail(None, "not_listed_count_mismatch", m_nl.group(0), dropped=len(dropped))
        body = _body_lines(first)
        for position, line in enumerate(body):
            if line == "The attempt ended at its turn limit.":
                self._attribute("body", line, "turn_limit")
                if not self.record.budget.get("ended_at_limit"):
                    self.fail(None, "turn_limit_claim_false", line)
                continue
            if line in collapsed_harms:
                members = collapsed_harms[line]
                self._attribute("body", line, "collapsed")
                m_turns, m_turn = TURNS_PREFIX.match(line), TURN_PREFIX.match(line)
                said = ([int(t) for t in m_turns.group(1).split(", ")] if m_turns
                        else [int(m_turn.group(1))] if m_turn else [])
                if sorted(said) != sorted({i.location.turn for i in members}):
                    self.fail(members[0], "collapsed_turns_mismatch", line, said=said)
                if self.version_c:
                    # G2: the count of objects is the count of distinct located calls
                    m_n = re.match(r"^At turns? [\d, ]+, \S+ (?:produced|updated|added|changed) (\d+) ", line)
                    calls = {(i.location.turn, i.location.call.index if i.location.call else 0) for i in members}
                    if m_n and int(m_n.group(1)) != len(calls):
                        self.fail(members[0], "collapsed_count_mismatch", line, said=int(m_n.group(1)), calls=len(calls))
                    if WHY_MARKERS.search(line) and len({i.check_id for i in members}) > 1:
                        self.fail(members[0], "why_collapsed_members_differ", line)
                for member in members:
                    own = plain.statement(member, self.record, expected_values, with_turn=False)
                    self.check_line(member, own or line, turn=member.location.turn, in_note=False)
                continue
            turn = None
            text = line
            m = TURN_PREFIX.match(line)
            turns: list[int] = []
            if m:
                turn = int(m.group(1))
                turns = [turn]
                text = line[m.end():]
            else:
                m2 = TURNS_PREFIX.match(line)
                if m2:
                    turns = [int(t) for t in m2.group(1).split(", ")]
                    text = line[m2.end():]
            collapsed = re.search(r" \(and \d+ other (\S+) errors?(?: at turns? ([\d, ]+))?\)$", text)  # several errors of one tool
            if collapsed:
                text = text[:collapsed.start()]
                if collapsed.group(2):
                    turns = turns + [int(t) for t in collapsed.group(2).split(", ")]
            if line in hint_lines:
                meta = hint_lines[line]
                self._attribute("body", line, "hint:" + str(meta.get("kind")))
                self._check_hint_source(meta, line)
                if meta.get("kind") == "rule_source" and self.version_c:
                    self._check_rule_placement(meta, line, body[position - 1] if position else None, by_text)
                if meta.get("kind") == "cause_diff":
                    self._check_cause_placement(meta, line, body[:position], hint_lines)
                if meta.get("kind") == "call_fix":
                    back = position - 1  # several fixes of one failure line follow it one after another
                    while back >= 0 and (hint_lines.get(body[back]) or {}).get("kind") == "call_fix":
                        back -= 1
                    previous = body[back] if back >= 0 else ""
                    if not (re.search(r"(?:^|, )" + re.escape(str(meta.get("tool"))) + r" failed", previous)
                            and meta.get("turn") in plain.failure_turns(previous)):
                        self.fail(None, "call_fix_not_after_its_failure", line)
                self._check_fix_hint(meta, line)
                continue
            joined = None
            for hint, meta in hint_lines.items():
                if meta.get("kind") == "missed_write" and line.endswith(" " + hint):
                    self._check_fix_hint(meta, hint)
                    self._check_hint_source(meta, hint)
                    joined = hint
                    line = text = line[:-len(hint) - 1]
                    break
            rep = plain.REPEATED_FAILURES.match(line)
            if rep:
                self._attribute("body", line, "repeated")
                self._check_repeated_failures(rep, line)
                continue
            candidates = by_text.get(text) or []
            if not candidates:
                self.fail(None, "unattributed_line", line)
                continue
            if getattr(self, "benchmark_only", False) and all(_rule_file_only(i) for i in candidates):
                self.fail(candidates[0], "rule_file_only_not_done_shown", line)
            self._attribute("body", line + (" " + joined if joined else ""), "statement:" + common.item_kind(candidates[0]))
            if turns:
                for t in turns:
                    matching = [i for i in candidates if i.location.turn == t]
                    if not matching and collapsed:
                        tool = collapsed.group(1)
                        matching = [i for i in items if i.type == "tool_error" and i.location.turn == t
                                    and i.semantics.get("tool") == tool]
                        if matching:
                            self.check_line(matching[0], f"{tool} failed: {error_message(matching[0].semantics.get('error', ''))}",
                                            turn=t, in_note=False)
                            continue
                    if not matching:
                        self.fail(candidates[0], "turn_without_item", line, said=t)
                        continue
                    self.check_line(matching[0], line, turn=t, in_note=False)
            else:
                self.check_line(candidates[0], line, turn=None, in_note=False)
        final_notes: list[int] = []
        for t, note in (ctx.get("notes") or {}).items():
            t = int(t)
            if not 1 <= t <= max(self.record.turns, 1):
                self.fail(None, "note_turn_out_of_range", note, turn=t)
            rest = note
            if rest.startswith(common.NOTE_PREFIX):
                rest = rest[len(common.NOTE_PREFIX):]
            if plain.FINAL_NOTE_HEAD in rest:  # correction r0: always the last text of its note
                position = rest.index(plain.FINAL_NOTE_HEAD)
                final_notes.append(t)
                self._check_final_note(t, rest[position:], expected_values=expected_values)
                rest = rest[:position].rstrip()
                if not rest:
                    continue
            # decision-point / solution blocks are not per-turn notes
            if common.CLOSING in rest or "Correct solution:" in rest:
                continue
            original = rest
            for hint, meta in hint_lines.items():
                if hint in rest:
                    if meta.get("kind") == "rule_source" and self.version_c:
                        pos = original.find(hint)
                        self._check_rule_placement(meta, hint, original[:pos].strip() or None, by_text, in_note=True)
                    rest = rest.replace(hint, " ", 1)
                    self._attribute(f"note:{t}", hint, "hint:" + str(meta.get("kind")))
                    self._check_hint_source(meta, hint)
                    owner = next((i for i in items if i.id == meta["item"]), None)
                    kind = meta.get("kind")
                    if kind == "cause_diff":
                        at_call = t == min(c.turn for c in self.record.calls if c.tool == meta.get("tool") and c.error is not None) \
                            if any(c.tool == meta.get("tool") and c.error is not None for c in self.record.calls) else False
                    elif kind == "unrequested_change":
                        at_call = meta.get("turn") == t
                    elif kind == "rule_source":
                        at_call = owner is not None and common.located_turn(owner) == t
                    else:
                        at_call = meta.get("at_turn") == t and any(
                            c.turn == t and c.error and (c.tool == meta.get("at_tool") or meta.get("kind") == "wrong_value" or c.error is None)
                            for c in self.record.calls) or (meta.get("kind") == "wrong_value" and meta.get("turn") == t)
                        at_call = at_call or (owner is not None and common.located_turn(owner) == t)
                    if not at_call:
                        self.fail(owner, "hint_note_not_at_item_turn", hint, turn=t)
                    else:
                        self._check_fix_hint(meta, hint)
            for text in sorted(by_text, key=len, reverse=True):  # longest first: a statement may be a prefix of another
                its = by_text[text]
                while text in rest:
                    rest = rest.replace(text, " ", 1)
                    here = [i for i in its if i.location.turn == t and common.located_turn(i) == t]
                    if not here:
                        self.fail(its[0], "note_line_not_located_at_turn", text, turn=t)
                        break
                    if getattr(self, "benchmark_only", False) and all(_rule_file_only(i) for i in here):
                        self.fail(here[0], "rule_file_only_not_done_shown", text, turn=t)
                    self._attribute(f"note:{t}", text, "statement:" + common.item_kind(here[0]))
                    self.check_line(here[0], text, turn=t, in_note=True)
            if rest.strip():
                self.fail(None, "unattributed_note_text", rest.strip(), turn=t)
        self._check_final_note_placement(ctx, final_notes)
        if _solution_heading(first) is not None:
            self._check_solution(ctx)

    # -- correction r0: rule-file-only requirements and the final note --------------------------------------------
    def _check_hint_source(self, meta: dict, line: str) -> None:
        """With not_done_source "benchmark", a hint derived from a rule-file-only requirement is not shown."""
        if not getattr(self, "benchmark_only", False):
            return
        by_id = {i.id: i for i in self.record.items}
        owners = [by_id.get(i) for i in (meta.get("items") or [meta.get("item")])]
        if any(o is not None and _rule_file_only(o) for o in owners):
            self.fail(next(o for o in owners if o is not None and _rule_file_only(o)), "rule_file_only_hint_shown", line)

    def _final_reply_turn(self) -> int | None:
        """The turn of the final reply, re-derived from the episode: the last assistant message, without a tool call,
        with an earlier assistant message that made one."""
        nodes = getattr(self.ep, "nodes", None)
        messages = [n.get("message") or {} for n in nodes] if nodes is not None else list(getattr(self.ep, "messages", None) or [])
        assistants = [m for m in messages if m.get("role") == "assistant"]
        if not assistants or assistants[-1].get("tool_calls") or not any(m.get("tool_calls") for m in assistants[:-1]):
            return None
        return len(assistants)

    def _check_final_note(self, turn: int, text: str, *, expected_values: bool = True) -> None:
        """The note before the final reply: it sits at the final reply's turn (the last assistant message, no tool call,
        after the last call), never on a successful attempt, and lists only failed benchmark requirements (each item
        text re-derived from its item and checked as that "Not done" line), every one of them once."""
        self.checked += 1
        self._attribute(f"note:{turn}", text, plain.FINAL_NOTE_KIND)
        if self.ep_success() or self.record.success:
            self.fail(None, "final_note_on_successful_attempt", text, turn=turn)
        final = self._final_reply_turn()
        if final is None or final != turn or turn != self.record.turns:
            self.fail(None, "final_note_not_before_final_reply", text, turn=turn, final_reply=final)
        if any(c.turn == turn for c in self.record.calls):
            self.fail(None, "final_note_reply_has_tool_call", text, turn=turn)
        rem = text[len(plain.FINAL_NOTE_HEAD):] if text.startswith(plain.FINAL_NOTE_HEAD) else None
        one_per_line = rem is not None and rem.startswith("\n")
        if rem is None or not (one_per_line or (rem.startswith(" (1) ") and rem.endswith("."))):
            self.fail(None, "final_note_malformed", text, turn=turn)
            return
        # every statement any item of the record renders to, as an item text ("Not done: " and the final "." off) and
        # whole (a planted harm or tool error is then named for what it is, not only as unattributed text)
        texts: dict[str, list[Item]] = collections.defaultdict(list)
        for item in self.record.items:
            try:
                statement = plain.statement(item, self.record, expected_values, with_turn=False)
            except Exception:  # noqa: BLE001 - a renderer crash leaves the text unattributed
                continue
            if not statement:
                continue
            short = statement[len("Not done: "):] if statement.startswith("Not done: ") else statement
            texts[short[:-1] if short.endswith(".") else short].append(item)
        listed: list[str] = []
        ordered = sorted(texts, key=len, reverse=True)  # an item text may be a prefix of another
        if one_per_line:  # "\n1. A\n2. B": every line numbered in order, each an item text
            for k, row in enumerate(rem[1:].split("\n"), 1):
                m = re.match(r"^(\d+)\. (.*)$", row)
                if m is None or int(m.group(1)) != k:
                    self.fail(None, "final_note_malformed", row, turn=turn)
                    continue
                if m.group(2) not in texts:
                    self.fail(None, "final_note_unattributed_text", m.group(2), turn=turn)
                    continue
                listed.append(m.group(2))
        else:  # " (1) A; (2) B."
            rest, k = rem[1:-1], 1
            while rest:
                if not rest.startswith(f"({k}) "):
                    self.fail(None, "final_note_malformed", rest, turn=turn)
                    break
                rest = rest[len(f"({k}) "):]
                match = next((b for b in ordered if rest == b or re.match(re.escape(b) + r"; \(\d+\) ", rest)), None)
                if match is None:
                    self.fail(None, "final_note_unattributed_text", rest, turn=turn)
                    break
                listed.append(match)
                rest, k = rest[len(match):].removeprefix("; "), k + 1
        # one per line exactly when an item holds "; " itself (then "; " would not separate items unambiguously)
        if listed and one_per_line != any("; " in b for b in listed):
            self.fail(None, "final_note_format_mismatch", text, turn=turn)
        if len(listed) != len(set(listed)):
            self.fail(None, "final_note_item_repeated", text, turn=turn)
        native = getattr(self.ep, "native_results", None)
        for short in listed:
            its = texts[short]
            good = [i for i in its if _benchmark_required_failed(i)]
            if not good:
                worst = its[0]
                if any(common.item_kind(i) in ("prohibited", "changed") or _negative_check(i) for i in its):
                    kind = "final_note_lists_must_not"
                elif any(i.type == "tool_error" for i in its):
                    kind = "final_note_lists_tool_error"
                elif any(i.outcome != "failed" for i in its):
                    kind = "final_note_item_not_failed"
                else:
                    kind = "final_note_item_rule_file_only"
                self.fail(worst, kind, short, turn=turn)
                continue
            item = good[0]
            if _states_count(item, short):
                self.fail(item, "final_note_lists_count", short, turn=turn)
                continue
            if isinstance(native, list) and item.type != "tool_error":
                index = int(item.id.split(":")[1])
                recorded = native[index] if index < len(native) else None
                if recorded is None or recorded.get("passed") or recorded.get("excluded"):
                    self.fail(item, "final_note_item_not_failed_by_grader", short, turn=turn)
            self.check_line(item, "Not done: " + short + ".", turn=None, in_note=True)
        # the list is the whole set: every failed benchmark requirement shown as a "Not done" line in this context
        # (one merged into a fuller line, `near_duplicate_requirement`, is said by that line)
        merged = {s.get("id") for s in (self.ctx.get("coverage") or {}).get("suppressed") or []
                  if s.get("reason") == "near_duplicate_requirement"}
        for item in self.record.items:
            if not _benchmark_required_failed(item) or item.id in merged:
                continue
            statement = plain.statement(item, self.record, expected_values, with_turn=False)
            if _states_count(item, statement):
                continue  # count items are left out of the final note
            if statement and statement.startswith("Not done: ") and statement[len("Not done: "):].removesuffix(".") not in listed:
                self.fail(item, "final_note_missing_item", statement, turn=turn)

    def _check_final_note_placement(self, ctx: dict, found: list[int]) -> None:
        """The described final note (`ctx["final_note"]`) is the one in the notes, at its turn."""
        meta = ctx.get("final_note")
        said = meta.get("turn") if isinstance(meta, dict) and meta.get("line") else None
        if len(found) > 1:
            self.fail(None, "final_note_repeated", str(found))
        if said is not None:
            note = (ctx.get("notes") or {}).get(said) or (ctx.get("notes") or {}).get(str(said)) or ""
            if not note.endswith(meta["line"]) or found != [int(said)]:
                self.fail(None, "final_note_not_in_notes", meta["line"], turn=said)
        elif found and isinstance(meta, dict):
            self.fail(None, "final_note_not_described", str(found))

    def ep_success(self) -> bool:
        return bool(getattr(self.ep, "success", self.record.success))

    # -- N1 "why" clauses: parsed back and compared with the contract; the condition evaluated by the
    # environment's own evaluator on the candidate's member rebuilt from the initial world ----------------
    def _contract_check(self, item: Item) -> tuple[dict | None, dict | None]:
        from ..record.build import task_contract

        if not hasattr(self, "_contract"):
            self._contract = task_contract(self.ep) if hasattr(self.ep, "task") else None
        contract = self._contract or {}
        check = next((c for c in contract.get("checks") or [] if c.get("check_id") == item.check_id), None)
        return contract, check

    def _check_why(self, item: Item, line: str) -> None:
        from . import why_parse as W

        self.checked += 1
        contract, check = self._contract_check(item)
        if item.origin != "manifest" or check is None:
            self.fail(item, "why_clause_not_in_contract", line, reason="no contract check")
            return
        sources = contract.get("sources") or {}
        variant = check
        alias = item.semantics.get("alias")
        for alt in check.get("alternatives") or []:  # a guard located at an alternative source's effect
            if alias is not None and isinstance(alt, dict) and alt.get("alias") == alias:
                variant = {**check, "source": alt.get("source"), "effect_match": alt.get("effect_match") or {}}
        effect_top = _flat_all(variant.get("effect_match") or check.get("expected"))
        role = "required_when" if check.get("required_when") else ("prohibited_when" if check.get("prohibited_when") else None)
        condition = check.get(role) if role else None
        cond_top = _flat_all(condition)
        population = sources.get(check.get("population") or "", {}) if check.get("population") else {}
        keys = list(population.get("key_fields") or [])

        def canon_set(nodes):
            out = {}
            for n in nodes:
                try:
                    out[W.canon(n)] = n
                except W.NotCanonical:
                    out[("unrenderable", json.dumps(n, sort_keys=True, default=str))] = n
            return out

        effect_resid = [c for c in effect_top if not _expressible(c) and not _request_only(c)]
        effect_request = [c for c in effect_top if not _expressible(c) and _request_only(c)]
        cond_effect = [c for c in cond_top if _reads(c, "effect") and not _expressible(c)]
        cond_request = [c for c in cond_top if _request_only(c) and not _is_selector(c, keys)]
        n1a_allowed = canon_set(effect_resid + (cond_effect if role == "prohibited_when" and item.polarity == "forbid" else []))
        applicability_only = canon_set(cond_effect if role == "required_when" else [])
        n1b_allowed = canon_set(cond_request + effect_request)
        segments = []
        for opener, kind in ((" (also required: ", "a"), (" (the task also requires: ", "a"), (" (the task's rule: ", "b")):
            seg = W.balanced_segment(line, opener)
            if seg is not None:
                segments.append((kind, seg[2]))
        harm_when = None
        if "; this task forbids it when " in line:
            rest = line.split("; this task forbids it when ", 1)[1].rstrip(".")
            seg = W.balanced_segment(rest, " (the task's rule: ")
            if seg is not None:
                rest = rest[:seg[0]]
            cand = plain._candidate(item, True)
            if cand and rest.endswith(cand):
                rest = rest[: -len(cand)]
            harm_when = rest
            segments.append(("a", rest))
        for kind, text in segments:
            marker_kept = False
            if kind == "b":
                verb = "forbidden" if item.polarity == "forbid" else "required"
                if not text.startswith(f"{verb} when "):
                    self.fail(item, "why_polarity_mismatch", line)
                    continue
                text = text[len(f"{verb} when "):]
            elif text.endswith("; plus a further condition the task states"):
                marker_kept = True
                text = text[: -len("; plus a further condition the task states")]
            if kind == "a" and harm_when is None and item.polarity == "forbid" and "also require" in line:
                self.fail(item, "why_polarity_mismatch", line)
            if kind == "a" and harm_when is not None and text == harm_when and item.polarity != "forbid":
                self.fail(item, "why_polarity_mismatch", line)
            try:
                parsed = W.parse_clauses(text)
            except W.NotCanonical as exc:
                self.fail(item, "why_clause_not_in_contract", line, reason=str(exc))
                continue
            allowed = n1a_allowed if kind == "a" else n1b_allowed
            for node in parsed:
                if node not in allowed:
                    reason = "why_clause_from_applicability" if node in applicability_only else "why_clause_not_in_contract"
                    self.fail(item, reason, line, clause=str(node)[:200])
            if kind == "a" and not marker_kept:
                # completeness: no unstated effect condition is left once the marker is gone
                rendered = set(parsed)
                if (set(n1a_allowed) - rendered or (item.semantics.get("unresolved") or {}).get("address")
                        or (item.semantics.get("unresolved") or {}).get("content") or applicability_only
                        or (effect_request and " (the task's rule: " not in line)):
                    self.fail(item, "why_marker_dropped_with_condition_left", line)
            if kind == "b":
                if set(parsed) != set(n1b_allowed):
                    self.fail(item, "why_condition_incomplete", line)
                self._check_why_applies(item, line, condition, effect_request, population)
                if item.polarity == "forbid":
                    self._check_why_rule2(item, line, condition, effect_request, population)

    def _check_why_applies(self, item: Item, line: str, condition: dict | None, extra: list, population: dict) -> None:
        """The whole condition (selectors included) holds for this candidate, by the environment's evaluator on
        the member rebuilt from the initial world."""
        try:
            from automationbench_v1.contracts.predicates import evaluate_predicate, parse_predicate
        except Exception:  # noqa: BLE001 - without the environment's evaluator nothing is established
            self.fail(item, "why_condition_not_established", line, reason="no evaluator")
            return
        member = _population_member(self.ep.task_data.get("initial_state") or {}, population, item.candidate)
        if member is None:
            self.fail(item, "why_condition_not_established", line, reason="member not found")
            return
        tree = {"op": "all", "args": ([condition] if condition else []) + list(extra)}
        try:
            result = evaluate_predicate(parse_predicate(tree), {"request": member})
        except Exception as exc:  # noqa: BLE001
            self.fail(item, "why_condition_not_established", line, reason=str(exc)[:120])
            return
        if result.value is not True:
            self.fail(item, "why_condition_not_established", line, value=result.value)

    def _check_why_rule2(self, item: Item, line: str, condition: dict | None, extra: list, population: dict) -> None:
        """Rule 2 (fix round): no "forbidden when" clause whose condition holds (or cannot be decided) for a row or
        record that a requirement of the task needs written. Members rebuilt from the initial world; the condition
        evaluated by the environment's evaluator."""
        try:
            from automationbench_v1.contracts.predicates import evaluate_predicate, parse_predicate
        except Exception:  # noqa: BLE001
            self.fail(item, "why_forbidden_clause_covers_required_record", line, reason="no evaluator")
            return
        initial = self.ep.task_data.get("initial_state") or {}
        members: list[tuple[str, dict, str]] = []
        if population.get("adapter") == "google_sheets.rows@1":
            keys = population.get("key_fields") or []
            for row in ((initial.get("google_sheets") or {}).get("rows")) or []:
                if isinstance(row, dict) and row.get("spreadsheet_id") == population.get("spreadsheet_id") \
                        and row.get("worksheet_id") == population.get("worksheet_id"):
                    cells = dict(row.get("cells") or {})
                    members.append((str(row.get("row_id")), cells, str(cells.get(keys[0], "")) if keys else ""))
        elif population.get("adapter") == "initial.records@1":
            node: Any = initial
            for step in (population.get("path") or [])[2:]:
                node = node.get(step) if isinstance(node, dict) else None
            for rec in node if isinstance(node, list) else []:
                ident: Any = rec
                for step in population.get("identity_path") or ["id"]:
                    ident = ident.get(step) if isinstance(ident, dict) else None
                if ident is not None:
                    members.append((str(ident), rec, ""))
        needed_keys, needed_values = set(), set()
        for other in self.record.items:
            if other.polarity != "require" or other.type == "tool_error" or other.effect.kind == "read":
                continue
            if other.candidate and other.candidate.get("key") is not None:
                needed_keys.add(str(other.candidate["key"]).strip().lower())
            for v in other.address.values():
                for x in (v.raw if isinstance(v.raw, list) else [v.raw]):
                    if not isinstance(x, (dict, list)) and x is not None:
                        needed_values.add(str(x).strip().lower())
        tree = parse_predicate({"op": "all", "args": ([condition] if condition else []) + list(extra)})
        for key, member, label in members:
            k = key.strip().lower()
            if not (k in needed_keys or k in needed_values or (label and label.strip().lower() in needed_values)):
                continue
            try:
                value = evaluate_predicate(tree, {"request": member}).value
            except Exception:  # noqa: BLE001
                value = None
            if value is not False:
                self.fail(item, "why_forbidden_clause_covers_required_record", line, member=key)
                return

    # -- H4 placement (R6.2) ---------------------------------------------------------------------------
    def _check_rule_placement(self, meta: dict, line: str, previous: str | None, by_text: dict, in_note: bool = False) -> None:
        owner = next((i for i in self.record.items if i.id == meta.get("item")), None)
        population = owner.semantics.get("population") if owner is not None else None
        ok = False
        if previous and population:
            for text, its in by_text.items():
                if previous.endswith(text) and any(i.origin == "manifest" and i.semantics.get("population") == population for i in its):
                    ok = True
                    break
        if not ok:
            self.fail(owner, "hint_rule_source_misplaced", line, in_note=in_note)

    # -- N2 placement: after the failure line of that tool (hint lines in between allowed) -------------
    def _check_cause_placement(self, meta: dict, line: str, before: list[str], hint_lines: dict) -> None:
        m = fix_cause_match(line)
        tool = m.group("tool") if m else meta.get("tool")
        for prev in reversed(before):
            if prev in hint_lines:
                continue
            if re.search(r"(?:^|, )" + re.escape(str(tool)) + r" failed", prev):
                return
            break
        self.fail(None, "cause_line_not_after_failure", line)

    # -- N2: argument differences, re-derived from the episodes ----------------------------------------
    def _check_cause_diff(self, meta: dict, line: str) -> None:
        from .. import fix
        from ..record.calls import episode_calls, parse_arguments
        from ..solution import qualify

        m = fix.CAUSE_LINE.match(line)
        if not m:
            self.fail(None, "hint_line_unparsed", line)
            return
        tool = m.group("tool")
        clauses = []
        for c in fix.CAUSE_CLAUSE.finditer(m.group("clauses")):
            kind = "with_value" if c.group("value") is not None else c.group("kind")
            clauses.append((kind, c.group("key"), json.loads(c.group("value")) if c.group("value") is not None else None))
        rebuilt = ", ".join((f'without "{k}"' if kind == "without" else f'with "{k}"' if kind == "with" else f'with "{k}": {json.dumps(v)}')
                            for kind, k, v in clauses)
        if not clauses or rebuilt != m.group("clauses"):
            self.fail(None, "hint_line_unparsed", line)
            return
        sibling = next((s for s in self.siblings if s.id == meta.get("sibling")), None)
        if sibling is None:
            self.fail(None, "cause_sibling_unknown", line)
            return
        ok, _why = qualify(sibling)
        if not ok:
            self.fail(None, "cause_sibling_not_qualified", line)
        if m.group("subject") == "The correct solution above":
            if self.ctx.get("solution_episode") != sibling.id:
                self.fail(None, "cause_solution_mismatch", line)
            block = _solution_block(self.ctx.get("first_message") or "") or ""
            success = []
            for l in block.split("\n"):
                name, _, rest = l.strip().partition("(")
                if name == tool and rest.endswith(")"):
                    success.append(parse_arguments(rest[:-1]))
        else:
            success = [c.args for c in episode_calls(sibling)[0] if c.tool == tool and c.error is None]
        if not success:
            self.fail(None, "cause_sibling_never_called_tool", line)
            return
        failed = [c for c in self.record.calls if c.tool == tool and c.error is not None]
        if not failed:
            self.fail(None, "cause_no_failed_call", line)
            return
        for kind, k, v in clauses:
            if kind == "without":
                holds, opposite = all(k not in a for a in success), all(k in c.args for c in failed)
            elif kind == "with":
                holds, opposite = all(k in a for a in success), all(k not in c.args for c in failed)
            else:
                holds = all(k in a and json.dumps(a[k], sort_keys=True) == json.dumps(v, sort_keys=True) for a in success)
                opposite = all(k not in c.args or json.dumps(c.args[k], sort_keys=True) != json.dumps(v, sort_keys=True) for c in failed)
            if not holds:
                self.fail(None, "cause_clause_false_for_sibling", line, key=k)
            if not opposite:
                self.fail(None, "cause_clause_not_a_difference", line, key=k)
            if self._cause_contradicts_requirement(kind, k, v, failed):
                self.fail(None, "cause_clause_contradicts_requirement", line, key=k)

    def _cause_contradicts_requirement(self, kind: str, key: str, value: Any, failed: list) -> bool:
        """R2 re-derived: the requirements of the tool's app (record items, any outcome; the contract's goal
        literals; the native assertions' parameters) and the request's own words."""
        from .. import fix
        from ..record.build import task_contract

        app = failed[0].app
        knorm = re.sub(r"[^a-z0-9]", "", key.lower())
        named: dict[str, set] = {}
        expected: set = set()

        def add(name: str, raw: Any) -> None:
            n = re.sub(r"[^a-z0-9]", "", name.lower())
            named.setdefault(n, set())
            for x in (raw if isinstance(raw, list) else [raw]):
                if not isinstance(x, (dict, list)) and fix.norm(x) is not None:
                    named[n].add(fix.norm(x))
                    expected.add(fix.norm(x))
        for i in self.record.items:
            if i.polarity != "require" or i.type == "tool_error" or not fix.app_matches(app, i.effect.app):
                continue
            for k, v in [*i.address.items(), *i.content.items()]:
                add(k.split("#")[0].split(".")[-1].removesuffix("_contains"), v.raw)
        for a in self.ep.task_data.get("assertions") or []:
            if isinstance(a, dict) and str(a.get("type", "")).replace("_", "").startswith(app.replace("_", "")[:6]):
                for k, v in a.items():
                    if k != "type":
                        add(str(k).removesuffix("_contains"), v)
        contract = task_contract(self.ep) if hasattr(self.ep, "task") else None
        for name, values in fix.contract_requirements(contract, app).items():
            named.setdefault(name, set()).update(values)
            expected |= values
        failed_values = {fix.norm(c.args[key]) for c in failed if key in c.args and not isinstance(c.args[key], (dict, list))}
        if failed_values & expected:
            return True
        if kind == "without":
            request = common.request_text(self.record, getattr(self.ep, "messages", None))
            return knorm in named or fix.key_words_in_request(key, request)
        if kind == "with_value" and named.get(knorm) and fix.norm(value) not in named[knorm]:
            return True
        return False

    # -- N3: a change to a record no requirement names, re-derived from the snapshots and the raw task data -
    def _check_unrequested(self, meta: dict, line: str) -> None:
        from .. import fix
        from ..record.build import task_contract

        m = fix.UNREQUESTED_LINE.match(line)
        if not m:
            self.fail(None, "hint_line_unparsed", line)
            return
        turn, tool = int(m.group("turn")), m.group("tool")
        call = next((c for c in self.record.calls if c.turn == turn and c.tool == tool and c.error is None and c.mutated), None)
        if call is None or not self.tl.available:
            self.fail(None, "unrequested_call_missing", line)
            return
        before, after = self.tl.json(call.before_digest), self.tl.json(call.after_digest)
        if before is None or after is None:
            self.fail(None, "unrequested_call_missing", line)
            return
        fields_text = m.group("fields")
        said = re.findall(r'"([^"]+)"', fields_text)
        count = int(fields_text.split()[0]) if fields_text.endswith("fields") else None
        target = None
        for app_key, data in after.items():
            if not isinstance(data, dict) or app_key.replace("_", " ") != m.group("app"):
                continue
            for coll, recs in data.items():
                if not isinstance(recs, list) or fix.singular(coll) != m.group("coll"):
                    continue
                old = {json.dumps(fix.record_key(r)): r for r in ((before.get(app_key) or {}).get(coll) or [])
                       if isinstance(r, dict) and fix.record_key(r) is not None}
                for r in recs:
                    if not isinstance(r, dict) or fix.record_key(r) is None:
                        continue
                    b = old.get(json.dumps(fix.record_key(r)))
                    if b is None or b == r:
                        continue
                    own = [fix.norm(b.get(k)) for k in fix.own_label_fields(coll) + ("email", "id")]
                    own.append(fix.norm(f"{b.get('first_name', '')} {b.get('last_name', '')}".strip()))
                    if coll.lower().endswith("rows") and isinstance(b.get("cells"), dict):
                        own = [fix.norm(v) for v in b["cells"].values()][:1]
                    if fix.norm(m.group("label")) in [o for o in own if o is not None]:
                        target = (app_key, coll, b, r)
                        break
        if target is None:
            self.fail(None, "unrequested_field_not_changed", line, reason="record not found")
            return
        app_key, coll, b, r = target
        changed = [k for k in dict.fromkeys([*b, *r]) if b.get(k) != r.get(k) and not BOOKKEEPING.search(str(k))]
        if count is not None:
            if count != len([f for f in changed if not fix.field_named(f, self.ep.task_data.get("assertions") or [], task_contract(self.ep))]):
                self.fail(None, "unrequested_field_not_changed", line, count=count, changed=len(changed))
        elif not said or any(f not in changed for f in said):
            self.fail(None, "unrequested_field_not_changed", line)
        # NAMES from both snapshots, and the scans of the raw task data
        names = {fix.norm(fix.record_key(b)), fix.norm(m.group("label"))}
        for rec in (b, r):
            for k, v in rec.items():
                if BOOKKEEPING.search(str(k)):
                    continue
                stack = [v]
                while stack:
                    x = stack.pop()
                    if isinstance(x, dict):
                        stack.extend(x.values())
                    elif isinstance(x, list):
                        stack.extend(x)
                    elif fix.norm(x) is not None:
                        names.add(fix.norm(x))
        names.discard(None)
        args = json.dumps(call.args, default=str)
        if not any(_whole(n, args) for n in names):
            self.fail(None, "unrequested_call_not_targeting_record", line)
        assertions = self.ep.task_data.get("assertions") or []
        raw_texts = [json.dumps(a, default=str) for a in assertions]
        contract = task_contract(self.ep) if hasattr(self.ep, "task") else None
        raw_texts.append(json.dumps(contract or {}, default=str))
        sing = fix.singular(coll).replace(" ", "_").lower()
        named = any(_whole(n, t) for n in names for t in raw_texts)
        named = named or any(re.search(r"(?:^|_)" + re.escape(sing) + r"(?:_|$)", str(a.get("type", "")).lower()) for a in assertions if isinstance(a, dict))
        named = named or fix.collection_named(app_key, coll, contract, assertions)
        named = named or any(fix.norm(x) in names for i in self.record.items for v in [*i.address.values(), *i.content.values()]
                             for x in (v.raw if isinstance(v.raw, list) else [v.raw]) if not isinstance(x, (dict, list)))
        if named:
            self.fail(None, "unrequested_record_named_by_requirement", line)
        for f in said:
            if fix.field_named(f, assertions, contract):
                self.fail(None, "unrequested_record_named_by_requirement", line, field=f)
        request = common.request_text(self.record, getattr(self.ep, "messages", None))
        if any(_whole(n, request) for n in names):
            self.fail(None, "unrequested_record_in_request", line)

    def _check_written_values(self, item: Item, line: str) -> None:
        """Removal round: a located prohibition quotes only what the call wrote. For every content value, the line
        quotes a value the call wrote exactly (its arguments, or a field of an object it added or changed in the
        recorded snapshots), or the expected value as a literal part of a written value after "containing"."""
        call = _call_at(self.record, item)
        if call is None:
            return
        norm_key = lambda k: re.sub(r"[^a-z0-9]", "", str(k).lower())  # noqa: E731
        leaves = []
        stack = [((), call.args)]
        while stack:
            path, node = stack.pop()
            if isinstance(node, dict):
                stack.extend(((*path, str(k)), v) for k, v in node.items())
            elif isinstance(node, list):
                stack.extend((path, v) for v in node)
            elif path and node is not None:
                leaves.append((norm_key(path[-1]), node))
        before, after = self.tl.json(call.before_digest), self.tl.json(call.after_digest)
        if before is not None and after is not None:
            old = {json.dumps(o, sort_keys=True, default=str) for o in _walk_objects(before)}
            for o in _walk_objects(after):
                if not isinstance(o, dict) or json.dumps(o, sort_keys=True, default=str) in old:
                    continue
                for k, v in o.items():
                    if isinstance(v, dict):
                        leaves += [(norm_key(k2), v2) for k2, v2 in v.items() if not isinstance(v2, (dict, list)) and v2 is not None]
                    elif not isinstance(v, (dict, list)) and v is not None:
                        leaves.append((norm_key(k), v))
        quoted = [json.loads(q) for q in re.findall(r'"(?:[^"\\]|\\.)*"', line)]
        negated = set(item.semantics.get("negated_keys") or ())
        for key, value in item.content.items():
            if key in negated or isinstance(value.raw, (dict, list)) or value.raw is None:
                continue
            base = key.split("#")[0]
            last = base.split(".")[-1]
            if base.startswith("after_cells."):
                field = base[len("after_cells."):]
            elif last in ("value", "expected_value") and any(k in item.address for k in ("field", "property", "column")):
                field = str(next(item.address[k].raw for k in ("field", "property", "column") if k in item.address))
            else:
                field = re.sub(r"(_not)?_contains$", "", last)
            written = [str(w).strip() for k, w in leaves if k == norm_key(field)]
            if not written:
                self.fail(item, "harm_value_not_written", line, field=field)
                continue
            if not isinstance(value.raw, str):  # a number or a boolean is shown bare, exactly as written
                shown = json.dumps(value.raw) if isinstance(value.raw, bool) else str(value.raw)
                if str(value.raw) not in written or not re.search(r"(?<![\w.])" + re.escape(shown) + r"(?![\w.])", line):
                    self.fail(item, "harm_value_not_written", line, field=field)
                continue
            exact = any(isinstance(q, str) and q.strip() in written for q in quoted)
            part = any(isinstance(q, str) and q and any(q in w for w in written) and re.search(
                r'containing (?:"(?:[^"\\]|\\.)*"(?:, | or | and ))*' + re.escape(json.dumps(q)), line) for q in quoted)
            if not (exact or part):
                self.fail(item, "harm_value_not_written", line, field=field)

    def _check_call_fix(self, meta: dict, line: str, initial: dict) -> None:
        """Fix 2: a failed call's fix sentence, re-derived from the recorded call, its error and the initial world.
        World form: the failed call at that turn errored and gave the value as an argument, its error message names
        it; the value is at the stated record and field, at one place of the whole initial world (or the record's own
        id with every other place a reference to it); the record's id, label and parent are its own; "not a X" is
        the kind the error says it could not find, and the record is not of that kind; "TOOL accepted "I" as "A"" is
        a recorded call without error (the attempt at that turn, or a sibling) of that tool with that argument.
        List form: the error the attempt received lists exactly one entry containing the value, the quoted one."""
        from .. import fix
        from ..record.calls import episode_calls, error_message

        turn = int(meta.get("turn") or 0)
        call = next((c for c in self.record.calls if c.turn == turn and c.index == meta.get("call_index")), None)
        m_world, m_list = fix.CALL_FIX_WORLD.match(line), fix.CALL_FIX_LIST.match(line)
        if call is None or not call.error or call.tool != meta.get("tool"):
            self.fail(None, "call_fix_without_failed_call", line)
            return
        m = m_world or m_list
        if m is None:
            self.fail(None, "hint_line_unparsed", line)
            return
        value = m.group("v")
        if int(m.group("turn")) != call.turn:
            self.fail(None, "call_fix_turn_mismatch", line)
        args = [str(v).strip() for p, v in fix._flat_args(call.args) if p and not isinstance(v, (dict, list, bool))]
        message = error_message(call.error)
        if value not in args or not fix.not_found_names(message, value):
            self.fail(None, "call_fix_value_not_in_failed_call", line)
        if m_list is not None:
            entries = fix.error_suggestions(call.error) or []
            pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(value) + r"(?![A-Za-z0-9])", re.IGNORECASE)
            hits = [e for e in entries if pattern.search(e)]
            if hits != [m.group("entry")]:
                self.fail(None, "call_fix_list_entry_not_unique", line, hits=hits[:4])
            return
        try:
            rec = initial[meta["app"]][meta["collection"]][meta["index"]]
        except (KeyError, IndexError, TypeError):
            self.fail(None, "call_fix_record_missing", line)
            return
        node: Any = rec
        for k in meta["field"]:
            node = node.get(k) if isinstance(node, dict) else None
        if fix.norm(node) != fix.norm(value):
            self.fail(None, "call_fix_value_not_at_place", line)
        if f'{m.group("app")} {m.group("coll")}' != f'{meta["app"].replace("_", " ")} {fix.singular(meta["collection"])}' \
                or m.group("field") != fix.field_words(tuple(meta["field"])):
            self.fail(None, "call_fix_place_mismatch", line)
        places = []
        for app_key, data in initial.items():
            for coll_key, records in (data.items() if isinstance(data, dict) else []):
                for i, r in enumerate(records if isinstance(records, list) else []):
                    if isinstance(r, dict):
                        places += [(app_key, coll_key, i, p) for p, v in fix.scalar_fields(r) if fix.norm(v) == fix.norm(value)]
        stated = (meta["app"], meta["collection"], meta["index"], tuple(meta["field"]))
        others = [p for p in places if p != stated]
        if stated not in places or (len(places) != 1 and not (stated[3] == ("id",) and all(
                p[3][-1].lower().endswith("_id") or p[3][-1] in fix.ID_KEYS for p in others))):
            self.fail(None, "call_fix_value_not_unique", line, occurrences=len(places))
        if m.group("id") is not None and fix.norm(rec.get("id")) != fix.norm(m.group("id")):
            self.fail(None, "call_fix_id_mismatch", line)
        if m.group("label") is not None:
            own = [fix.norm(rec.get(k)) for k in fix.own_label_fields(meta["collection"]) + ("email", "id")]
            own.append(fix.norm(f"{rec.get('first_name', '')} {rec.get('last_name', '')}".strip()))
            cells = rec.get("cells") if isinstance(rec.get("cells"), dict) else {}
            own += [fix.norm(v) for v in cells.values()][:1]
            if fix.norm(m.group("label")) not in [o for o in own if o is not None]:
                self.fail(None, "call_fix_label_not_records_own_name", line)
        if m.group("pid") is not None:
            parent = meta.get("parent") or {}
            try:
                prec = initial[meta["app"]][parent["collection"]][parent["index"]]
            except (KeyError, IndexError, TypeError):
                prec = None
            ref = fix.singular(parent.get("collection", "")).replace(" ", "_") + "_id"
            if prec is None or fix.norm(prec.get("id")) != fix.norm(m.group("pid")) or fix.norm(rec.get(ref)) != fix.norm(m.group("pid")):
                self.fail(None, "call_fix_parent_mismatch", line)
            elif fix.norm(m.group("plabel")) not in {fix.norm(prec.get(k)) for k in fix.own_label_fields(parent["collection"]) + ("email", "id")}:
                self.fail(None, "call_fix_parent_label_mismatch", line)
        if m.group("noun") is not None:
            said = fix.error_noun(message, value)
            if said != m.group("noun") or fix.singular(meta["collection"]).split()[-1] == said:
                self.fail(None, "call_fix_kind_claim_wrong", line)
        if m.group("tool") is not None:
            target, arg, tool = m.group("target"), m.group("arg"), m.group("tool")
            if fix.norm(target) != fix.norm(rec.get("id")):
                self.fail(None, "call_fix_accepted_not_the_record", line)
            pools = ([c for c in self.record.calls if m.group("at") is not None and c.turn == int(m.group("at"))]
                     if m.group("at") is not None else
                     [c for s in self.siblings if s.id != self.record.episode_id for c in episode_calls(s)[0]])
            ok = any(c.tool == tool and not c.error and any(p and p[-1] == arg and fix.norm(v) == fix.norm(target)
                                                             for p, v in fix._flat_args(c.args)) for c in pools)
            if not ok:
                self.fail(None, "call_fix_accepted_call_missing", line)

    def _check_harm_target(self, item: Item, line: str) -> None:
        """Fix 1 (version c): a located prohibition names its target as the call wrote it, scopes a per-record
        prohibition to that record, and appends a rule's candidate only when it is the record the address names.
        - For every scalar address value whose argument family (channel / channel_id / channel_name, …) the call
          wrote: the line quotes a written value equal to it (or holding it as a whole token), or quotes a written
          value followed by the check's value in brackets, and then one object of the world before the call (else
          the task's initial world) carries the two under an id field and a name-like field.
        - A line that says "for that record" names a record by its own id; a per-record prohibition says it.
        - " for "LABEL"" after the tail only for a candidate whose key or label is one of the line's values."""
        from ..record.build import target_family

        call = _call_at(self.record, item)
        if call is None:
            return
        leaves: list[tuple[str, Any]] = []
        stack: list[tuple[str | None, Any]] = [(None, call.args)]
        while stack:
            key, node = stack.pop()
            if isinstance(node, dict):
                stack.extend((str(k), v) for k, v in node.items())
            elif isinstance(node, list):
                stack.extend((key, v) for v in node)
            elif key is not None and node is not None and not isinstance(node, bool):
                leaves.append((key, node))
        before = self.tl.json(call.before_digest) if call.before_digest else None
        world = before if before is not None else (self.ep.task_data.get("initial_state") or {})

        def shown(v: Any) -> str:
            return json.dumps(v) if isinstance(v, str) else str(v)

        def in_line(v: Any) -> bool:
            text = shown(v)
            return bool(re.search(r"(?<![\w.])" + re.escape(text) + r"(?![\w])", line)) if not isinstance(v, str) else text in line

        def mapped_by_world(a: Any, b: Any) -> bool:
            for data in (world or {}).values():
                if not isinstance(data, dict):
                    continue
                for records in data.values():
                    for rec in records if isinstance(records, list) else []:
                        if not isinstance(rec, dict):
                            continue
                        vals = {f: str(v).strip().lower() for f, v in rec.items() if not isinstance(v, (dict, list)) and v is not None}
                        for x, y in ((a, b), (b, a)):
                            if any((f == "id" or f.endswith("_id")) and v == str(x).strip().lower() for f, v in vals.items()) and \
                                    any(f in ("name", "title", "label", "display_name", "full_name", "username", "email", "handle")
                                        and v == str(y).strip().lower() for f, v in vals.items()):
                                return True
            return False

        for key, value in item.address.items():
            raw = value.raw
            if isinstance(raw, (dict, list)) or raw is None or isinstance(raw, bool):
                continue
            written = [w for k, w in leaves if target_family(k) == target_family(key)]
            if not written:
                continue
            same = [w for w in written if str(w).strip().lower() == str(raw).strip().lower()]
            inside = [w for w in written if isinstance(w, str) and re.search(
                r"(?<![\w@.\-])" + re.escape(str(raw).strip()) + r"(?![\w@\-]|\.\w)", w, re.IGNORECASE)]
            if any(in_line(w) for w in same) or (inside and in_line(raw)):
                continue
            bracketed = [w for w in written if f"{shown(w)} ({json.dumps(str(raw))})" in line]
            if bracketed and any(mapped_by_world(w, raw) for w in bracketed):
                continue
            self.fail(item, "harm_target_not_as_written" if not bracketed else "harm_target_mapping_unproven", line,
                      key=key, written=[str(w)[:60] for w in written[:3]])
        scoped = plain.SCOPED_TAIL in line or plain.SCOPED_TAIL_MANY in line
        ids = [(k, v.raw) for k, v in item.address.items() if k.split("#")[0].split(".")[-1] in plain.RECORD_KEYS
               and v.raw is not None and not isinstance(v.raw, (dict, list, bool))]
        lists = [(k, v.raw) for k, v in item.address.items() if k.split("#")[0].split(".")[-1] in plain.RECORD_KEYS
                 and isinstance(v.raw, list) and v.raw]
        if scoped and not ids and not lists:
            self.fail(item, "record_scope_without_record", line)
        if (ids or lists) and not scoped and re.search(r'; this task forbids (?:it|them)(?:\.$| for ")', line):
            self.fail(item, "record_prohibition_unscoped", line)
        if scoped and ids and not any(in_line(v) or in_line(str(v)) for _k, v in ids):
            self.fail(item, "record_scope_record_not_named", line)
        if item.candidate:
            label = json.dumps(str(item.candidate.get("label")))
            if re.search(r"; this task forbids (?:it|them)(?: for that record| for those records)? for " + re.escape(label), line) \
                    and not plain.candidate_named(item):
                self.fail(item, "harm_candidate_not_named_by_line", line, candidate=item.candidate.get("label"))

    def _check_not_met(self, item: Item, line: str) -> None:
        """Fix round: "Not met: the attempt made A NOUN ADDR (id "I") …; the grader marks this requirement as not met."
        The object with that id is one the attempt added (or changed) and carries the address; the requirement is a
        failed manifest goal."""
        m = re.search(r' \(id ("(?:[^"\\]|\\.)*")\)', line)
        if not m or not line.endswith("; the grader marks this requirement as not met.") or item.origin != "manifest" \
                or item.outcome != "failed" or item.polarity != "require":
            self.fail(item, "not_met_unsupported", line)
            return
        oid = json.loads(m.group(1))
        values = _address_values(item)
        new_only = item.effect.kind in ("send", "create", "post")
        for view in self.added_objects(item.effect.app, item.semantics.get("collection") or item.effect.object, new_only=new_only):
            fields = view.get("fields") or {}
            if any(str(fields.get(k)) == oid for k in ("id", "row_id", "message_id", "ts")) and _all_forms_in(values, view["identity"]):
                return
        self.fail(item, "not_met_object_not_found", line, id=oid)

    def _check_solution_notes(self, notes: list[str], solution_text: str) -> None:
        """`Note: this solution wrote "X" in FIELD of TOOL; the task needs "Y" or "Z".`: a shown call of TOOL
        carries X under an argument naming FIELD; the Ys are exactly the expected values of one failed
        requirement's group; X is none of them and none of them is in the solution."""
        from .. import fix
        from ..record.calls import parse_arguments

        calls = []
        for line in solution_text.strip().split("\n"):
            name, _, rest = line.strip().partition("(")
            if rest.endswith(")"):
                calls.append((name, parse_arguments(rest[:-1])))
        low = solution_text.lower()
        for note in notes:
            self.checked += 1
            m = fix.SOLUTION_NOTE.match(note)
            absent = fix.SOLUTION_ABSENCE.match(note)
            if not m and not absent:
                self.fail(None, "solution_note_unparsed", note)
                continue
            needs = re.findall(r'"([^"]*)"', (m or absent).group("needs"))
            written = m.group("written") if m else None
            if m:
                wrote = any(name == m.group("tool") and any(
                    path and fix._key_norm(path[-1]) == fix._key_norm(m.group("field")) and written in str(v)
                    for path, v in fix._flat_args(args)) for name, args in calls)
                if not wrote:
                    self.fail(None, "solution_note_value_not_written", note)
            groups = []
            for item in self.record.items:
                if not fix.binding_item(item):
                    continue
                alt = item.semantics.get("alternatives") or {}
                by_group: dict = {}
                for key, value in item.content.items():
                    raws = value.raw if isinstance(value.raw, list) else [value.raw]
                    for raw in raws:
                        if not isinstance(raw, (dict, list)):
                            by_group.setdefault(alt.get(key, ("own", key)), []).append(str(raw).strip())
                groups += [sorted(set(v)) for v in by_group.values()]
            if sorted(set(needs)) not in groups:
                self.fail(None, "solution_note_values_not_expected", note)
            # the solution itself must not satisfy the requirement: a check the grader passed on the solution's
            # episode, or a sibling check of the same requirement the solution carries, allows no note
            from ..fix import _same_requirement, _satisfied, binding_item

            def item_groups(i):
                alt = i.semantics.get("alternatives") or {}
                by: dict = {}
                for key, value in i.content.items():
                    for raw in (value.raw if isinstance(value.raw, list) else [value.raw]):
                        if not isinstance(raw, (dict, list)):
                            by.setdefault(alt.get(key, ("own", key)), []).append(str(raw).strip())
                return [sorted(set(v)) for v in by.values()]

            owners = [i for i in self.record.items if binding_item(i) and sorted(set(needs)) in item_groups(i)]
            outcomes = getattr(self, "_solution_outcomes", {}) or {}
            for owner in owners:
                if owner.origin == "native" and outcomes.get(int(owner.id.split(":")[1])) not in (None, "failed"):
                    self.fail(owner, "solution_note_check_passed_for_solution", note)
                    break
                key = (owner.check_id, str(owner.candidate.get("key")) if owner.candidate else None)
                if owner.origin == "manifest" and (getattr(self, "_solution_manifest", {}) or {}).get(key) == 1:
                    self.fail(owner, "solution_note_check_passed_for_solution", note)
                    break
                if any(o is not owner and _same_requirement(owner, o) and _satisfied(o, low) for o in self.record.items if binding_item(o)):
                    self.fail(owner, "solution_note_requirement_satisfied_by_alternative", note)
                    break
            if notes.count(note) > 1:
                self.fail(None, "solution_note_duplicated", note)
            if any((written is not None and fix.norm(written) == fix.norm(n)) or fix.norm(n) in low for n in needs):
                self.fail(None, "solution_note_no_disagreement", note)

    def _check_repeated_failures(self, m, line: str) -> None:
        """`TOOL failed N times (turns a-b), first: MESSAGE`: N tool errors of that tool, spanning exactly
        those turns, the first with that message."""
        self.checked += 1
        tool = m.group("tool")
        errors = sorted((i for i in self.record.items if i.type == "tool_error" and i.semantics.get("tool") == tool),
                        key=lambda i: (i.location.turn or 0))
        turns = sorted({i.location.turn for i in errors})
        if len(errors) != int(m.group("count")) or not turns or turns[0] != int(m.group("first")) or turns[-1] != int(m.group("last")):
            self.fail(None, "repeated_failures_count_mismatch", line, found=len(errors), turns=(turns[:1] + turns[-1:]))
            return
        if plain.shorten(error_message(errors[0].semantics.get("error", ""))) != m.group("message"):
            self.fail(errors[0], "repeated_failures_message_mismatch", line)
        if getattr(self, "version_c", False):
            # G3: recounted from the recorded calls, not from the builder's items
            calls = [c for c in self.record.calls if c.tool == tool and c.error is not None]
            if len(calls) != int(m.group("count")) or not calls or calls[0].turn != int(m.group("first")) \
                    or max(c.turn for c in calls) != int(m.group("last")):
                self.fail(None, "repeated_failures_calls_mismatch", line, found=len(calls))

    # -- fix hints (abce.fix): the value is at the stated place and nowhere else; the tool returned that
    # collection in a recorded result; no step changed the collection; the wrong value was written ----
    def _check_fix_hint(self, meta: dict, line: str) -> None:
        from .. import fix

        self.checked += 1
        initial = self.ep.task_data.get("initial_state") or {}
        if meta.get("kind") == "cause_diff":
            self._check_cause_diff(meta, line)
            return
        if meta.get("kind") == "unrequested_change":
            self._check_unrequested(meta, line)
            return
        if meta.get("kind") == "call_fix":
            self._check_call_fix(meta, line, initial)
            return
        if meta.get("kind") == "rule_source":
            m = fix.RULE_LINE.match(line)
            item = next((i for i in self.record.items if i.id == meta["item"]), None)
            from ..record.build import task_contract

            src = fix.sheet_source(task_contract(self.ep), item, initial) if item is not None and item.outcome == "failed" else None
            if not m or src is None:
                self.fail(item, "hint_rule_source_unknown", line)
                return
            if item.semantics.get("population") is None or (m.group("ws"), m.group("ss")) != (src["worksheet"], src["spreadsheet"]):
                self.fail(item, "hint_rule_source_mismatch", line)
            if bool(m.group("unread")) != (not fix.worksheet_read(list(self.record.calls), src["worksheet_id"], src["worksheet"], initial)):
                self.fail(item, "hint_rule_source_read_claim_wrong", line)
            return
        if meta.get("kind") == "wrong_value":
            m = fix.WROTE_LINE.match(line)
            if not m:
                self.fail(None, "hint_line_unparsed", line)
                return
            item = next((i for i in self.record.items if i.id == meta["item"]), None)
            calls = [c for c in self.record.calls if c.turn == int(m.group("turn")) and c.tool == m.group("tool") and c.error is None]
            if not calls or item is None or item.outcome != "failed":
                self.fail(item, "hint_wrote_call_missing", line)
                return
            # several calls of that tool at one turn: the line is true when one of them carries every claim
            rows_before = len(self.rows)
            best = None
            for call in calls:
                del self.rows[rows_before:]
                self._check_wrote_call(item, meta, m, line, call)
                if len(self.rows) == rows_before:
                    best = None
                    break
                best = list(self.rows[rows_before:])
            if best is not None:
                del self.rows[rows_before:]
                self.rows.extend(best)
            return
        if meta.get("kind") == "value_location":
            return self._check_value_location(meta, line, initial)
        self._check_fix_hint_rest(meta, line, initial)

    def _check_wrote_call(self, item: Item, meta: dict, m, line: str, call) -> None:
        from .. import fix

        if True:
            args_text = json.dumps(call.args, sort_keys=True).lower()
            if not all(fix.norm(v) in args_text for v in meta["address"].values()):
                self.fail(item, "hint_wrote_address_not_in_call", line)
            written = [v for path, v in fix._flat_args(call.args) if path and fix._key_norm(path[-1]) == fix._key_norm(m.group("field"))]
            if not any(fix.norm(v) == fix.norm(m.group("written")) for v in written):
                self.fail(item, "hint_wrote_value_not_in_call", line)
            # the call changed that field of that record itself (snapshots), never through a reference argument
            on_record = fix.changed_record_field(self.tl, call, item.effect.app, str(item.semantics.get("collection") or item.effect.object),
                                                 next(iter(meta["address"].values()), None), m.group("field"))
            if on_record is None or fix.norm(on_record) != fix.norm(m.group("written")):
                self.fail(item, "hint_wrote_not_on_that_record", line)
            needs = re.findall(r'"([^"]*)"', m.group("needs"))
            negated = set(item.semantics.get("negated_keys") or ())
            groups = item.semantics.get("alternatives") or {}
            keys = [k for k, v in item.content.items() if fix.expected_field(item, k) == meta["field"]
                    and not isinstance(v.raw, (dict, list)) and fix.norm(v.raw) == fix.norm(needs[0] if needs else None)]
            if not keys:
                self.fail(item, "hint_wrote_expected_not_required", line)
            for k in keys[:1]:
                if k in negated:
                    self.fail(item, "hint_wrote_expected_negated", line)
                members = ([str(v.raw).strip() for kk, v in item.content.items() if groups.get(kk) == groups[k] and kk not in negated]
                           if k in groups else [str(item.content[k].raw).strip()])
                if sorted(set(fix.norm(x) for x in members)) != sorted(set(fix.norm(x) for x in needs)):
                    self.fail(item, "hint_wrote_expected_not_required", line, members=members)
                if any(fix.norm(x) == fix.norm(m.group("written")) or fix.norm(x) in fix.norm(m.group("written")) for x in members):
                    self.fail(item, "hint_wrote_alternative_satisfied", line)
            if any(fix.norm(m.group("written")) == fix.norm(n) or fix.norm(n) in fix.norm(m.group("written")) for n in needs):
                self.fail(item, "hint_wrote_value_is_the_expected", line)
            # fix round: the wording follows the check: a containment check says the written value lacks it
            k0 = keys[0] if keys else None
            base = (k0 or "").split("#")[0].split(".")[-1]
            mode = (item.semantics.get("modes") or {}).get(k0)
            contains = (mode not in (None, "eq", "in")) if item.origin == "manifest" else (base.endswith("_contains") or "contains" in item.type)
            if k0 is not None and contains != bool(m.group("tail")):
                self.fail(item, "hint_wrote_wording_mismatch", line)
            return

    def _check_value_location(self, meta: dict, line: str, initial: dict) -> None:
        from .. import fix

        if True:
            m = fix.VALUE_LINE.match(line)
            if not m:
                self.fail(None, "hint_line_unparsed", line)
                return
            named = m.group("v2") is not None
            value = m.group("v2") if named else m.group("v1")
            app, coll = (m.group("app2"), m.group("coll2")) if named else (m.group("app"), m.group("coll"))
            tools = (m.group("tools") or "").split(" or ") if m.group("tools") else []
            if (value != meta.get("value") or tools != list(meta.get("tools") or [])
                    or f"{app} {coll}" != f"{meta['app'].replace('_', ' ')} {fix.singular(meta['collection'])}"):
                self.fail(None, "hint_line_differs_from_evidence", line)
                return
            try:
                rec = initial[meta["app"]][meta["collection"]][meta["index"]]
            except (KeyError, IndexError, TypeError):
                self.fail(None, "hint_record_missing", line)
                return
            node = rec
            for key in meta["field"]:
                node = node.get(key) if isinstance(node, dict) else None
            at_place = fix.norm(node) == fix.norm(value) if not isinstance(node, list) else any(
                fix.norm(v) == fix.norm(value) for v in node)
            if not at_place:
                self.fail(None, "hint_value_not_at_place", line)
            # uniqueness, by a scan of every record of the initial world: one place, or one record whose own
            # `id` it is while every other place is a reference to it (`*_id`)
            places = []
            for app_key, data in initial.items():
                if not isinstance(data, dict):
                    continue
                for coll_key, records in data.items():
                    if not isinstance(records, list):
                        continue
                    for i, r in enumerate(records):
                        if isinstance(r, dict):
                            places += [(app_key, coll_key, i, p) for p, v in fix.scalar_fields(r) if fix.norm(v) == fix.norm(value)]
            stated = (meta["app"], meta["collection"], meta["index"], tuple(meta["field"]))
            others = [p for p in places if p != stated]
            if len(places) != 1 and not (stated in places and stated[3] == ("id",)
                                         and all(p[3][-1].lower().endswith("_id") or p[3][-1] in fix.ID_KEYS for p in others)):
                self.fail(None, "hint_value_not_unique", line, occurrences=len(places))
            if named:
                if m.group("id2") is not None and fix.norm(rec.get("id")) != fix.norm(m.group("id2")):
                    self.fail(None, "hint_id_mismatch", line)
            else:
                label = m.group("label")
                coll = meta["collection"]
                if coll.lower().endswith("rows"):
                    cells = rec.get("cells") if isinstance(rec.get("cells"), dict) else {}
                    own = [fix.norm(v) for v in cells.values()][:1]  # a row is named by its first cell
                else:
                    own = [fix.norm(rec.get(k)) for k in fix.own_label_fields(coll) + ("email", "id")]
                    own.append(fix.norm(f"{rec.get('first_name', '')} {rec.get('last_name', '')}".strip()))
                if fix.norm(label) not in [o for o in own if o is not None]:
                    self.fail(None, "hint_label_not_records_own_name", line)
                if m.group("ws") is not None and fix.worksheet_title(initial, rec) != m.group("ws"):
                    self.fail(None, "hint_worksheet_mismatch", line)
            for tool in tools:
                if not self._tool_returned(tool, meta["app"], meta["collection"], rec):
                    # correction r0: a learned map is no evidence that the tool returns this record (no excuse)
                    self.fail(None, "hint_tool_never_returned_collection", line, tool=tool)
            return

    def _check_fix_hint_rest(self, meta: dict, line: str, initial: dict) -> None:
        from .. import fix

        if meta.get("kind") == "missed_write":
            m = fix.WRITE_LINE.match(line)
            if not m:
                self.fail(None, "hint_line_unparsed", line)
                return
            from ..record.probe import _same_collection

            for item_id in meta.get("items", []):
                item = next((i for i in self.record.items if i.id == item_id), None)
                if (item is None or item.outcome != "failed" or item.polarity != "require"
                        or not _app_matches(meta["app"], item.effect.app)
                        or not _same_collection(meta["collection"], str(item.semantics.get("collection") or item.effect.object))
                        or plain.noun(item) != m.group("noun")):
                    self.fail(item, "hint_write_item_mismatch", line)
            if not meta.get("items"):
                self.fail(None, "hint_write_item_mismatch", line)
            if not self.tl.available:
                self.fail(None, "hint_write_without_snapshots", line)
                return
            invocations = {inv for _occ, inv, _b, _a, _t in self.tl.steps}
            if any(c.invocation_id not in invocations for c in self.record.calls if c.error is None and c.mutated is not False):
                self.fail(None, "write_steps_incomplete", line)
            for _occ, _inv, before, after, _tool in self.tl.steps:
                if before == after:
                    continue
                b, a = self.tl.json(before), self.tl.json(after)
                if b is None or a is None:
                    self.fail(None, "hint_write_snapshot_missing", line)
                    return
                if (b.get(meta["app"]) or {}).get(meta["collection"]) != (a.get(meta["app"]) or {}).get(meta["collection"]):
                    self.fail(None, "hint_write_collection_was_written", line, invocation=_inv)
                    return
            return
        self.fail(None, "hint_kind_unknown", line)

    def _tool_returned(self, tool: str, app: str, coll: str, rec: dict) -> bool:
        """A recorded, error-free call of `tool` (the attempt or a sibling) whose result holds a dict carrying
        the record's id under an id-like key, or two of the record's field values."""
        from .. import fix
        from ..record.calls import episode_calls

        rec_values = {fix.norm(v) for _p, v in fix.scalar_fields(rec)} - {None}
        rec_id = fix.norm(rec.get("id"))
        pools = [list(self.record.calls)] + [episode_calls(s)[0] for s in self.siblings if s.id != self.record.episode_id]
        for calls in pools:
            for call in calls:
                if call.tool != tool or call.error or call.result is None or not fix.app_matches(call.app, app):
                    continue
                for node in fix.result_dicts(call.result):
                    scalars = {k: fix.norm(v) for k, v in node.items() if not isinstance(v, (dict, list))}
                    if rec_id is not None and any(v == rec_id and (k in fix.ID_KEYS or k.lower().endswith("id"))
                                                  for k, v in scalars.items()):
                        return True
                    if len({v for v in scalars.values() if v in rec_values}) >= 2:
                        return True
        return False

    def _check_solution(self, ctx: dict) -> None:
        from ..fix import solution_agrees

        sid = ctx.get("solution_episode")
        sibling = next((s for s in self.siblings if s.id == sid), None)
        text = _solution_block(ctx.get("first_message") or "") or ""  # the feedback lines follow the block
        if getattr(self, "version_c", False):
            # R7: "Correct solution:" only when the solution's own obligation findings show no unmet goal
            from ..fix import sibling_manifest_outcomes

            if _solution_heading(ctx.get("first_message") or "") == "Correct solution:" and \
                    any(v == 0 for v in sibling_manifest_outcomes(sibling).values()):
                self.fail(None, "solution_heading_overclaims", "Correct solution:", solution=sid)
        notes = [l.strip() for l in text.split("\n") if l.strip().startswith("Note: this solution ")]
        text = "\n".join(l for l in text.split("\n") if not l.strip().startswith("Note: this solution "))
        if ctx.get("expected_values", True) and ctx.get("fix_hints") is not None:
            from ..fix import sibling_outcomes

            from ..fix import sibling_manifest_outcomes

            self._solution_outcomes = sibling_outcomes(sibling)
            self._solution_manifest = sibling_manifest_outcomes(sibling)
            missing = solution_agrees(text, self.record, self._solution_outcomes, self._solution_manifest)
            noted = {tuple(n.get("needs") or ()) for n in ctx.get("solution_notes") or []}
            for miss in missing:
                if tuple(miss["values"]) not in noted:
                    self.fail(None, "solution_contradicts_expected_values", text[:200], solution=sid, values=[miss])
            self._check_solution_notes(notes, text)
        if sibling is None:
            self.fail(None, "solution_sibling_unknown", text[:200], solution=sid)
            return
        if not all(r.get("passed") or r.get("excluded") for r in sibling.native_results):
            self.fail(None, "solution_failed_native_check", text[:200], solution=sid)
        try:
            from ..record.manifest import GUARD_OUTPUT, findings

            harms = [p for p in findings(sibling.assessment_batches)[GUARD_OUTPUT]
                     if p.get("kind") == "finding" and p.get("status") == "valid" and p.get("value") == 1]
        except Exception:  # noqa: BLE001
            harms = []
        if harms:
            self.fail(None, "solution_triggered_manifest_harm", text[:200], solution=sid, harms=[h.get("check_id") for h in harms][:3])
        # S2: the shown call list, replayed from the task's initial world, must pass every scored native check
        from ..record.calls import READ_VERBS, app_of, parse_arguments
        from ..record.model import Call
        from ..solution import replay_native

        from ..record.calls import episode_calls

        recorded = {}
        for c in episode_calls(sibling)[0]:
            recorded.setdefault((c.tool, json.dumps(c.args, sort_keys=True, default=str)), c.invocation_id)
        shown = []
        for n, line in enumerate(l.strip() for l in text.strip().split("\n") if l.strip()):
            name, _, rest = line.partition("(")
            if not rest.endswith(")") or "[" in name:
                continue  # a masked line (values hidden) cannot be replayed as written
            args = parse_arguments(rest[:-1])
            shown.append(Call(turn=n + 1, index=0, id=None, tool=name, app=app_of(name), args=args,
                              invocation_id=recorded.get((name, json.dumps(args, sort_keys=True, default=str)))))
        if shown and common.UNSEEN not in text:
            excluded = {i for i, r in enumerate(sibling.native_results) if r.get("excluded")}
            replay = replay_native(sibling.task_data, shown, excluded, sibling=sibling)
            if replay.get("ok") is False:
                self.fail(None, "solution_fails_on_replay", text[:200], solution=sid, reason=replay.get("reason"))

        writes = [line.strip() for line in text.strip().split("\n")
                  if line.strip() and not any(v in line.split("(")[0] for v in READ_VERBS)]
        for item in common.facts(self.record):
            if common.item_kind(item) != "prohibited" or not _content_values(item):
                continue
            values = _address_values(item) + _content_values(item)
            # a write of the same app carries the forbidden content at the forbidden address
            for line in writes:
                if _app_matches(app_of(line.split("(")[0]), item.effect.app) and _all_forms_in(values, normalize(line)):
                    self.fail(item, "solution_contradicts_forbid_line", line)
                    break


def check_episode(ep: Episode, record: Record, ctx: dict, siblings: list[Episode] | None = None,
                  *, expected_values: bool = True) -> tuple[list[dict], int]:
    checker = Checker(ep, record, siblings)
    checker.check_context(ctx, expected_values=expected_values)
    return checker.rows, checker.checked


# ---------------------------------------------------------------------------------------------
# Sampled pass over evidence files (streamed)
# ---------------------------------------------------------------------------------------------
def default_sample_files() -> list[tuple[str, Path]]:
    """Six non-heldout runs across all six domains plus the simple tasks (the audit's sample)."""
    from .. import paths

    e = paths.EVIDENCE
    return [
        ("base", e / "sdpo-ab-20261027-base-r1/run/base/episodes.jsonl.gz"),
        ("screen96", e / "sdpo-ab-20261030-screen96-r1/run/screen96/episodes.jsonl.gz"),
        ("m62ctrl", e / "sdpo-ab-20261030-m62-control-ours-r1/run/control-m62-ours/episodes.jsonl.gz"),
        ("pool124u05", e / "sdpo-ab-20261030-handover-pool124-r1/run/handover/update-05/episodes.jsonl.gz"),
        ("t16ctrl", e / "sdpo-ab-20261030-t16-control-r1/run/control-transfer-t16/episodes.jsonl.gz"),
        ("simple30", e / "sdpo-ab-20261031-easy-final-r1/run/final-simple30/episodes.jsonl.gz"),
    ]


def select_sample(path: str, *, per_task: int, seed: int, label: str) -> tuple[dict[int, str], dict[str, int]]:
    """(line index -> task) of ≤ `per_task` failed episodes per task (seeded, as the audit sampled) and
    (task -> line index) of one successful reference per task."""
    rng = random.Random(f"{seed}:{label}")
    failed_by_task: dict[str, list[int]] = collections.defaultdict(list)
    refs: dict[str, int] = {}
    for i, ep in enumerate(iter_episodes([path])):
        if ep.success:
            refs.setdefault(ep.task, i)
        else:
            failed_by_task[ep.task].append(i)
    chosen: dict[int, str] = {}
    for task, idxs in failed_by_task.items():
        for i in rng.sample(idxs, min(per_task, len(idxs))):
            chosen[i] = task
    return chosen, refs


def load_sample(path: str, *, per_task: int, seed: int, label: str) -> tuple[list[Episode], dict[str, Episode]]:
    chosen, refs = select_sample(path, per_task=per_task, seed=seed, label=label)
    wanted = set(refs.values())
    selected, references = [], {}
    for i, ep in enumerate(iter_episodes([path])):
        if i in wanted:
            references[ep.task] = ep
        if i in chosen:
            selected.append(ep)
    return selected, references


def truth_pass(files: list[tuple[str, Path]] | None = None, *, per_task: int = 2, seed: int = 7,
               solution: str = "none", expected_values: bool = True, out_path: str | Path | None = None) -> dict[str, Any]:
    """Stream the sample; render plain/both contexts under the given value policy; return counts and
    violation rows. With `expected_values=False` the leak detector runs too: a value the attempt never
    observed must not appear anywhere in the text (lines, notes or the solution block)."""
    from ..context import build_context
    from .leak import leaks

    files = files or default_sample_files()
    rows: list[dict] = []
    counts: collections.Counter = collections.Counter()
    by_kind: collections.Counter = collections.Counter()
    suppressed: collections.Counter = collections.Counter()
    for label, path in files:
        if not Path(path).exists():
            continue
        selected, references = load_sample(str(path), per_task=per_task, seed=seed, label=label)
        for ep in selected:
            sib = [references[ep.task]] if ep.task in references else []
            record = build_record(ep, sib, probe=True, manifest=True)
            ctx = build_context(record, sib, style="plain", placement="both", expected_values=expected_values,
                                episode=ep, solution=solution)
            found, checked = check_episode(ep, record, ctx, sib, expected_values=expected_values)
            if not expected_values:
                for leak in leaks(record, ctx, episode_messages=ep.messages, task_assertions=ep.task_data.get("assertions"),
                                  tool_names=tuple(ep.task_data.get("zapier_tools") or ())):
                    found.append({"episode": ep.id, "task": ep.task, "item": leak.get("item"), "type": leak.get("key"),
                                  "kind": "leak_hidden_value", "line": str(leak.get("form"))})
                    counts["leaks"] += 1
            counts["episodes"] += 1
            counts["lines_checked"] += checked
            counts["lines_first_message"] += len(_body_lines(ctx.get("first_message") or ""))
            counts["notes"] += len(ctx.get("notes") or {})
            counts["episodes_with_violation"] += bool(found)
            counts["violations"] += len(found)
            counts["with_solution"] += ctx.get("solution_episode") is not None
            counts["skipped"] += ctx.get("skip_reason") is not None
            for s in (ctx.get("coverage") or {}).get("suppressed") or []:
                suppressed[s.get("reason")] += 1
            for s in (ctx.get("coverage") or {}).get("dropped") or []:
                counts["dropped_lines"] += 1
            report = (ctx.get("coverage") or {}).get("solution") or {}
            for candidate in report.get("candidates") or []:
                verdict = "qualified" if candidate.get("qualified") else str(candidate.get("reason")).split(":")[0]
                suppressed["solution:" + verdict] += 1
            if report.get("withheld"):
                suppressed["solution:withheld_" + str(report["withheld"].get("reason"))] += 1
            for r in found:
                by_kind[r["kind"]] += 1
                r["label"] = label
            rows.extend(found)
            del record, ctx
        del selected, references
        gc.collect()
    result = {"counts": dict(counts), "by_kind": dict(by_kind), "suppressed": dict(suppressed), "rows": rows,
              "sample": {"files": [l for l, _ in files], "per_task": per_task, "seed": seed, "solution": solution,
                         "expected_values": expected_values}}
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as handle:
            json.dump(result, handle, indent=1)
    return result
