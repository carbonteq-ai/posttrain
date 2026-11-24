"""Fix hints (version c, `build_context(..., expected_values=True, fix_hints=...)`): line forms added to the plain
context for failed requirements and failed calls, each verified by `abce.checks.truth` against the recorded initial
world and the recorded tool results. `fix_hints` selects them by kind (FIX_HINT_KINDS; True = all of them):

- "value_location": where a needed value lives. An expected value of a failed requirement (an address or content
  value) is searched in the task's recorded initial world; found in exactly one place (collection, record, field), the
  line names the place with the record's own label:
     `In the initial world, "C_ENG" is the id of slack channel "engineering" (slack_list_channels returns it).`
     `In the initial world, there is a salesforce account named "MiniTech" (id "001xx000007ACC5") (…).`
  A tool is named only when one of the attempt's or its siblings' recorded calls returned that very record
  (`returned_record`: its id under an id-like key, or two of its field values); never from a hand table and never from
  other runs. With no such call, the location alone is stated.
- "wrong_value": a call wrote another value where the requirement expects one.
- "missed_write": a failed requirement whose effect is a write to a collection no call of the attempt changed:
     `No call wrote a salesforce task.` (joined to the requirement's own "Not done" line).
- "unrequested_change": a change to an existing record that no requirement of the task names.
- "call_fix" (Fix 2 of correction a7): a failed call's line followed by one sentence naming the fix, only when the
  recorded initial world or the error's own list of known values proves it (`call_fix`).
- "why_clause": the N1 clauses of version (c) (abce.why: a manifest rule's own condition in plain words).
- "solution_note": the disagreement of a sibling solution with the expected values (since the fix round such a
  solution is withheld, so this kind adds no line; it decides only the reason recorded in coverage).

Every value that could not be hinted is counted with its reason (`fix_hints.withheld`): found in several
places, not in the initial world (a computed or derived value), unsearchable (a boolean, a short token, a
compound), or the requirement's own statement was withheld.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any, Iterator

from .record.calls import Call
from .record.model import Item, Record

HINT_LINE_KINDS = ("value_location", "wrong_value", "missed_write", "unrequested_change", "call_fix")
FIX_HINT_KINDS = HINT_LINE_KINDS + ("why_clause", "solution_note")
ALL_KINDS = frozenset(FIX_HINT_KINDS)
# the only kind the owner approved for the study (a7 Fix 2: "proven fix lines for failed calls")
APPROVED_KINDS = frozenset({"call_fix"})


def selected_kinds(fix_hints: Any) -> frozenset[str] | None:
    """The fix-hint kinds a `build_context(fix_hints=...)` value selects: None (nothing) for False, None or an empty
    collection; every kind for True; else the given kind names (an iterable of FIX_HINT_KINDS names)."""
    if fix_hints is None or fix_hints is False:
        return None
    if fix_hints is True:
        return ALL_KINDS
    if isinstance(fix_hints, (str, bytes, dict)) or not hasattr(fix_hints, "__iter__"):
        raise ValueError(f"fix_hints must be True, False, None or a collection of kind names {FIX_HINT_KINDS}, not {fix_hints!r}")
    kinds = frozenset(fix_hints)
    unknown = sorted(str(k) for k in kinds - ALL_KINDS)
    if unknown:
        raise ValueError(f"unknown fix_hints kinds {unknown}; known: {FIX_HINT_KINDS}")
    return kinds or None


OWN_NAME_FIELDS = ("name", "full_name", "display_name", "username")
OWN_TITLE_FIELDS = ("title", "subject")
LABEL_FIELDS = OWN_NAME_FIELDS + ("email", "label", "summary", "file_name", "filename")


def own_label_fields(coll: str) -> tuple[str, ...]:
    """The fields that name a record of `coll` itself: `name`, `title`, ..., and `<singular>_name`
    (`account_name` on accounts); never a related object's name (`account_name` on a contact)."""
    return OWN_NAME_FIELDS + (singular(coll).replace(" ", "_") + "_name",) + OWN_TITLE_FIELDS
SCHEMA_KEYS = {"object_type", "collection", "field", "column", "property", "worksheet", "worksheet_name", "table",
               "action_key", "lookup_column", "target_column", "params", "fields", "cells", "criteria", "after_cells",
               "changed_fields"}
ID_KEYS = ("id", "Id", "ID", "row_id", "message_id", "ts")
MAX_VALUE_LINES = 6
MAX_WRITE_LINES = 4
VALUE_LINE = re.compile(r'^In the initial world, (?:"(?P<v1>.*?)" is the (?P<field>.+?) of (?P<app>[a-z0-9 ]+?) '
                        r'(?P<coll>[a-z0-9 ]+?) "(?P<label>.*?)"(?: in worksheet "(?P<ws>.*?)")?|there is an? (?P<app2>[a-z0-9 ]+?) (?P<coll2>[a-z0-9 ]+?) '
                        r'named "(?P<v2>.*?)"(?: \(id "(?P<id2>.*?)"\))?)(?: \((?P<tools>[A-Za-z0-9_]+(?: or [A-Za-z0-9_]+)*) returns it\))?\.$')
WRITE_LINE = re.compile(r"^No call wrote an? (?P<noun>.+?)\.$")
WROTE_LINE = re.compile(r'^At turn (?P<turn>\d+), (?P<tool>\S+) wrote "(?P<written>.*?)" as (?P<field>.+?) (?P<addr>(?:to|for|in|on|of|at|with|named|from|row|worksheet|message|envelope|meeting|label|field|column|property|assigned|related|due|cc|message|thread).*?); the task needed (?P<needs>"(?:[^"]*)"(?: or "(?:[^"]*)")*) in that field(?P<tail>, and the written value (?:does not contain it|contains none of them))?\.$')


# ---------------------------------------------------------------------------------------------
# The initial world
# ---------------------------------------------------------------------------------------------
def norm(value: Any) -> str | None:
    """The comparable text of a scalar: strings of 3+ characters and numbers of 2+ digits; never booleans,
    never compounds."""
    if isinstance(value, bool) or value is None or isinstance(value, (dict, list, tuple)):
        return None
    text = str(value).strip()
    if isinstance(value, (int, float)):
        text = text[:-2] if text.endswith(".0") else text
        return text if len(text) >= 2 else None
    return text.lower() if len(text) >= 3 else None


def collections(initial_state: dict | None) -> Iterator[tuple[str, str, list]]:
    for app, data in (initial_state or {}).items():
        if not isinstance(data, dict):
            continue
        for coll, records in data.items():
            if isinstance(records, list) and records and all(isinstance(r, dict) for r in records):
                yield str(app), str(coll), records


def scalar_fields(record: dict, depth: int = 2) -> Iterator[tuple[tuple[str, ...], Any]]:
    """(field path, scalar) for every scalar of a record, two levels deep (a row's cells), list elements
    under their list's key."""
    def walk(node: Any, path: tuple[str, ...], d: int) -> Iterator[tuple[tuple[str, ...], Any]]:
        if isinstance(node, dict):
            if d >= depth:
                return
            for k, v in node.items():
                yield from walk(v, path + (str(k),), d + 1)
        elif isinstance(node, list):
            for v in node:
                if not isinstance(v, (dict, list)):
                    yield path, v
        else:
            yield path, node
    yield from walk(record, (), 0)


class WorldIndex:
    """value text -> places (app, collection, record index, field path) in a recorded initial world."""

    def __init__(self, initial_state: dict | None):
        self.state = initial_state or {}
        self.places: dict[str, list[tuple[str, str, int, tuple[str, ...]]]] = defaultdict(list)
        self.ids: dict[str, set[tuple[str, str, int]]] = defaultdict(set)
        for app, coll, records in collections(self.state):
            for i, rec in enumerate(records):
                for path, v in scalar_fields(rec):
                    t = norm(v)
                    if t is None:
                        continue
                    self.places[t].append((app, coll, i, path))
                    if len(path) == 1 and path[0] in ID_KEYS:
                        self.ids[t].add((app, coll, i))

    def find(self, value: Any) -> list[tuple[str, str, int, tuple[str, ...]]]:
        t = norm(value)
        return list(self.places.get(t, ())) if t is not None else []

    def substring_hit(self, value: Any) -> bool:
        t = norm(value)
        return t is not None and any(t in k for k in self.places)

    def record(self, app: str, coll: str, index: int) -> dict:
        return self.state[app][coll][index]


def record_label(record: dict, coll: str) -> tuple[str, tuple[str, ...]] | None:
    """(label text, the field path it came from): a sheet row by its first cell, else the first label-like
    field, else first and last name, else the id."""
    cells = record.get("cells")
    if coll.lower().endswith("rows") and isinstance(cells, dict):
        for k, v in cells.items():
            if isinstance(v, (str, int, float)) and not isinstance(v, bool) and str(v).strip():
                return str(v).strip(), ("cells", str(k))
        return None
    names = [k for k in own_label_fields(coll) if k not in OWN_TITLE_FIELDS]
    for key in names:
        v = record.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip(), (key,)
    first, last = record.get("first_name"), record.get("last_name")
    if isinstance(first, str) and isinstance(last, str) and (first + last).strip():
        return f"{first} {last}".strip(), ("first_name", "last_name")  # a person's name before their title
    for key in OWN_TITLE_FIELDS:
        v = record.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip(), (key,)
    v = record.get("email")
    if isinstance(v, str) and v.strip():
        return v.strip(), ("email",)
    v = record.get("id")
    if isinstance(v, (str, int)) and not isinstance(v, bool) and str(v).strip():
        return str(v), ("id",)
    return None


def singular(coll: str) -> str:
    words = coll.replace("_", " ")
    if words.endswith("ies"):
        return words[:-3] + "y"
    if words.endswith("ses") or words.endswith("xes"):
        return words[:-2]
    if words.endswith("s"):
        return words[:-1]
    return words


def field_words(path: tuple[str, ...]) -> str:
    if len(path) == 2 and path[0] == "cells":
        return f'"{path[1]}" cell'
    return " ".join(p.replace("_", " ").strip() for p in path).strip() or "value"


def worksheet_title(initial_state: dict | None, row: dict) -> str | None:
    """The title of the worksheet a sheet row belongs to, from the initial state's worksheets."""
    sheets = (initial_state or {}).get("google_sheets") or {}
    for ws in sheets.get("worksheets") or []:
        if isinstance(ws, dict) and str(ws.get("id")) == str(row.get("worksheet_id")) and isinstance(ws.get("title"), str):
            return ws["title"]
    return None


IDENTIFYING_HEADER_WORDS = ("name", "email", "id", "url", "link", "employee", "person", "contact", "owner", "customer",
                            "vendor", "company", "account", "candidate", "role", "title", "phone", "address", "channel",
                            "subject", "handle", "profile", "signer", "recipient", "manager", "assignee", "user")


def identifying_cell(header: str, value: Any) -> bool:
    """A sheet cell worth pointing at: its header names a person, place or thing (Name, Email, Vendor, Role,
    ...), or the value is an email or a url; a number or a status word in a data cell is not."""
    text = str(value)
    if "@" in text or "://" in text:
        return True
    words = re.findall(r"[a-z]+", header.lower())
    return any(w in IDENTIFYING_HEADER_WORDS for w in words)


def rank_places(places: list[tuple[str, str, int, tuple[str, ...]]]) -> tuple[str, str, int, tuple[str, ...]] | None:
    """Among several places, the one record whose own `id` the value is, when every other place is a
    reference to it (a `*_id` field or an id-like key elsewhere); None when the places are not of that shape."""
    own = [p for p in places if p[3] == ("id",)]
    refs = [p for p in places if p[3] != ("id",) and (p[3][-1].lower().endswith("_id") or p[3][-1] in ID_KEYS)]
    if len(own) == 1 and len(refs) == len(places) - 1:
        return own[0]
    return None


# ---------------------------------------------------------------------------------------------
# Which tools return which collections (learned from recorded results)
# ---------------------------------------------------------------------------------------------
def result_dicts(result: Any, depth: int = 6) -> Iterator[dict]:
    """Every dict inside a tool result (a JSON string or already parsed)."""
    if isinstance(result, str):
        text = result.strip()
        if not text.startswith(("{", "[")):
            return
        try:
            result = json.loads(text)
        except ValueError:
            return
    stack = [(result, 0)]
    while stack:
        node, d = stack.pop()
        if d > depth:
            continue
        if isinstance(node, dict):
            yield node
            stack.extend((v, d + 1) for v in node.values() if isinstance(v, (dict, list)))
        elif isinstance(node, list):
            stack.extend((v, d + 1) for v in node if isinstance(v, (dict, list)))


def dict_returns(node: dict, index: WorldIndex) -> set[tuple[str, str]]:
    """The initial-world collections of which `node` (one dict of a tool result) is a record: its own id
    field equals a record's id, or two of its scalar fields equal two fields of one record."""
    matches: dict[tuple[str, str, int], set[tuple[str, ...]]] = defaultdict(set)
    strong: set[tuple[str, str]] = set()
    for k, v in node.items():
        if isinstance(v, (dict, list)):
            continue
        t = norm(v)
        if t is None:
            continue
        if k in ID_KEYS or k.lower().endswith("_id") or k.lower().endswith("id"):
            for app, coll, i in index.ids.get(t, ()):
                strong.add((app, coll))
        for app, coll, i, path in index.places.get(t, ()):
            matches[(app, coll, i)].add(path)
    out = set(strong)
    for (app, coll, _i), paths_ in matches.items():
        if len(paths_) >= 2:
            out.add((app, coll))
    return out


def app_matches(call_app: str, world_app: str) -> bool:
    a, b = call_app.lower().replace("_", ""), world_app.lower().replace("_", "")
    return bool(a and b) and (a == b or a.startswith(b) or b.startswith(a))


def call_returns(call: Call, index: WorldIndex) -> set[tuple[str, str]]:
    """The collections a recorded call returned records of; a call that changed the world teaches nothing
    (a create's result echoes its own arguments, it does not look anything up)."""
    if call.error or call.result is None or call.mutated:
        return set()
    found: set[tuple[str, str]] = set()
    for node in result_dicts(call.result):
        found |= {(app, coll) for app, coll in dict_returns(node, index) if app_matches(call.app, app)}
    return found


def returned_record(call: Call, tool: str, app: str, rec: dict) -> bool:
    """A recorded, error-free call of `tool` whose result holds a dict carrying the record's id under an id-like key, or
    two of the record's field values: the evidence that this tool returns that very record (the truth checker's test)."""
    if call.tool != tool or call.error or call.result is None or not app_matches(call.app, app) or not isinstance(rec, dict):
        return False
    rec_values = {norm(v) for _p, v in scalar_fields(rec)} - {None}
    rec_id = norm(rec.get("id"))
    for node in result_dicts(call.result):
        scalars = {k: norm(v) for k, v in node.items() if not isinstance(v, (dict, list))}
        if rec_id is not None and any(v == rec_id and (k in ID_KEYS or k.lower().endswith("id")) for k, v in scalars.items()):
            return True
        if len({v for v in scalars.values() if v in rec_values}) >= 2:
            return True
    return False


class ToolMap:
    """tool -> {"app/collection": {"calls": n, "example": [episode id, turn]}}, built per record from the calls of the
    attempt and its siblings only (nothing is learned from other runs, read or written to disk)."""

    def __init__(self, data: dict | None = None):
        self.data: dict[str, dict[str, dict]] = {k: dict(v) for k, v in (data or {}).items()}

    def add(self, tool: str, app: str, coll: str, episode_id: str, turn: int) -> None:
        entry = self.data.setdefault(tool, {}).setdefault(f"{app}/{coll}", {"calls": 0, "example": [episode_id, turn]})
        entry["calls"] += 1

    def learn_calls(self, calls: list[Call], index: WorldIndex, episode_id: str) -> None:
        for call in calls:
            for app, coll in call_returns(call, index):
                self.add(call.tool, app, coll, episode_id, call.turn)

    def tools_for(self, app: str, coll: str) -> list[str]:
        key = f"{app}/{coll}"
        found = [(v[key]["calls"], tool) for tool, v in self.data.items() if key in v]
        return [tool for _, tool in sorted(found, key=lambda p: (-p[0], p[1]))]


# ---------------------------------------------------------------------------------------------
# The hints for one record
# ---------------------------------------------------------------------------------------------
def _candidate_values(item: Item) -> list[tuple[str, Any, bool, bool]]:
    """(key, scalar, is_address, written) worth looking up: address and content values that are scalars
    (list elements one by one), not schema names, not unresolved; `written` when the attempt itself wrote
    the value in an argument (a hint would restate what it already used)."""
    out = []
    for is_address, mapping in ((True, item.address), (False, item.content)):
        for key, value in mapping.items():
            base = key.split(".")[-1]
            if base in SCHEMA_KEYS:
                continue
            raw = value.raw
            if isinstance(raw, dict):
                continue
            for element in (raw if isinstance(raw, list) else [raw]):
                if isinstance(element, (dict, list)):
                    continue
                out.append((key, element, is_address, bool(value.written)))
    return out


def identifying_field(path: tuple[str, ...], value: Any = None) -> bool:
    """An id-like or label-like field: where a content value (a name, an email, an id in a body) is worth
    pointing at; a sheet cell counts when the value is text (a name, an email, a role), not a number."""
    if len(path) == 2 and path[0] == "cells":
        return identifying_cell(path[1], value)
    if len(path) != 1:
        return False
    key = path[0].lower()
    return key in ID_KEYS or key.endswith("_id") or key.endswith("id") or key in LABEL_FIELDS or key in ("first_name", "last_name")


def value_line(index: WorldIndex, place: tuple[str, str, int, tuple[str, ...]], value: Any, tools: list[str]) -> dict | None:
    app, coll, i, path = place
    rec = index.record(app, coll, i)
    label = record_label(rec, coll)
    if label is None:
        return None
    app_words, coll_word = app.replace("_", " "), singular(coll)
    tool_text = f" ({' or '.join(tools)} returns it)" if tools else ""
    shown = str(value).strip() if not isinstance(value, float) or not value.is_integer() else str(int(value))
    if path == label[1] and not coll.lower().endswith("rows"):
        rid = rec.get("id")
        id_text = f' (id "{rid}")' if isinstance(rid, (str, int)) and not isinstance(rid, bool) and norm(rid) != norm(value) else ""
        line = f'In the initial world, there is {_article(app_words)} {app_words} {coll_word} named "{shown}"{id_text}{tool_text}.'
    else:
        ws = worksheet_title(index.state, rec) if coll.lower().endswith("rows") else None
        ws_text = f' in worksheet "{ws}"' if ws else ""
        line = f'In the initial world, "{shown}" is the {field_words(path)} of {app_words} {coll_word} "{label[0]}"{ws_text}{tool_text}.'
    return {"kind": "value_location", "line": line, "value": shown, "app": app, "collection": coll, "index": i,
            "field": list(path), "label": label[0], "tools": tools}


def _article(word: str) -> str:
    return "an" if word[:1] in "aeiou" else "a"


def effect_collection(item: Item, initial_state: dict | None) -> tuple[str, str] | None:
    """The initial-world collection the item's effect writes to (app key, collection key), or None."""
    from .record.probe import _same_collection

    want = str(item.semantics.get("collection") or item.effect.object)
    for app, coll, _records in collections(initial_state):
        if app_matches(item.effect.app, app) and _same_collection(coll, want):
            return app, coll
    for app, data in (initial_state or {}).items():  # an empty collection is a list too
        if isinstance(data, dict) and app_matches(item.effect.app, str(app)):
            for coll, records in data.items():
                if isinstance(records, list) and _same_collection(str(coll), want):
                    return str(app), str(coll)
    return None


def steps_complete(calls: list[Call], timeline) -> bool:
    """Every call that may have changed the world (no error, not known unchanged) has its invocation among the
    timeline's steps: only then does "no step changed X" mean "no call changed X" (R4)."""
    if timeline is None or not getattr(timeline, "available", False):
        return False
    invocations = {inv for _occ, inv, _b, _a, _t in timeline.steps}
    return all(c.invocation_id in invocations for c in calls if c.error is None and c.mutated is not False)


def collection_written(timeline, app: str, coll: str) -> bool | None:
    """Whether any recorded step changed `initial[app][coll]`; None without snapshots."""
    if timeline is None or not timeline.available:
        return None
    for _occ, _inv, before, after, _tool in timeline.steps:
        if before == after:
            continue
        b, a = timeline.json(before), timeline.json(after)
        if b is None or a is None:
            return None
        if (b.get(app) or {}).get(coll) != (a.get(app) or {}).get(coll):
            return True
    return False


RULE_LINE = re.compile(r'^This applies to the rows of worksheet "(?P<ws>.*?)" in spreadsheet "(?P<ss>.*?)"(?P<unread>, which the attempt never read)?\.$')


def sheet_source(contract: dict | None, item: Item, initial_state: dict | None) -> dict | None:
    """The sheet rows a manifest rule applies to (its `population` in the contract's `sources`, adapter
    google_sheets.rows@1; a `source` is not that): {worksheet_id, spreadsheet_id, worksheet, spreadsheet} or None."""
    if not contract or item.origin != "manifest":
        return None
    sources = contract.get("sources") or {}
    for name in (item.semantics.get("population"),):
        src = sources.get(name) if isinstance(name, str) else None
        if not src or src.get("adapter") != "google_sheets.rows@1":
            continue
        sheets = (initial_state or {}).get("google_sheets") or {}
        ws = next((w for w in sheets.get("worksheets") or [] if isinstance(w, dict) and str(w.get("id")) == str(src.get("worksheet_id"))), None)
        ss = next((w for w in sheets.get("spreadsheets") or [] if isinstance(w, dict) and str(w.get("id")) == str(src.get("spreadsheet_id"))), None)
        if ws and ss and isinstance(ws.get("title"), str) and isinstance(ss.get("title"), str):
            return {"worksheet_id": str(src["worksheet_id"]), "spreadsheet_id": str(src["spreadsheet_id"]),
                    "worksheet": ws["title"], "spreadsheet": ss["title"]}
    return None


def worksheet_read(calls: list[Call], worksheet_id: str, worksheet: str, initial_state: dict | None = None) -> bool:
    """A successful google sheets call named or returned the worksheet (its id or title), or returned one of its
    rows: a dict whose `row_id` is a row of that worksheet in the initial world and which carries two or more of
    that row's cell values (a lookup result that names neither the title nor the id)."""
    wanted = {worksheet_id.lower(), worksheet.lower()}
    rows = [r for r in ((initial_state or {}).get("google_sheets") or {}).get("rows") or []
            if isinstance(r, dict) and str(r.get("worksheet_id")) == str(worksheet_id)]
    by_id = {str(r.get("row_id")): {norm(v) for v in (r.get("cells") or {}).values()} - {None} for r in rows}
    for call in calls:
        if call.error or not app_matches(call.app, "google_sheets"):
            continue
        text = (json.dumps(call.args) + " " + str(call.result or "")).lower()
        if any(w in text for w in wanted):
            return True
        for node in result_dicts(call.result):
            cells = by_id.get(str(node.get("row_id"))) if "row_id" in node else None
            if not cells:
                continue
            values = set()
            for v in node.values():
                if isinstance(v, dict):
                    values |= {norm(x) for x in v.values() if not isinstance(x, (dict, list))}
                elif not isinstance(v, list):
                    values.add(norm(v))
            if len(values & cells) >= 2:
                return True
    return False


def rule_source_line(record: Record, item: Item, contract: dict | None, initial_state: dict | None) -> dict | None:
    src = sheet_source(contract, item, initial_state)
    if src is None:
        return None
    unread = not worksheet_read(list(record.calls), src["worksheet_id"], src["worksheet"], initial_state)
    line = (f'This applies to the rows of worksheet "{src["worksheet"]}" in spreadsheet "{src["spreadsheet"]}"'
            + (", which the attempt never read" if unread else "") + ".")
    return {"kind": "rule_source", "item": item.id, "line": line, **src, "unread": unread,
            "population": item.semantics.get("population")}


# ---------------------------------------------------------------------------------------------
# Fix 2: a failed call's fix, only when the recorded world or the error the attempt received proves it
# ---------------------------------------------------------------------------------------------
MAX_CALL_FIX_LINES = 4
CALL_FIX_WORLD = re.compile(
    r'^The value "(?P<v>.+?)" given at turn (?P<turn>\d+) is the (?P<field>.+?) of (?P<app>[a-z0-9 ]+?) (?P<coll>[a-z0-9 ]+?)'
    r'(?: "(?P<label>.*?)")?(?: \(id "(?P<id>.*?)"\))?(?: of (?P<papp>[a-z0-9 ]+?) (?P<pcoll>[a-z0-9 ]+?) "(?P<plabel>.*?)"'
    r' \(id "(?P<pid>.*?)"\))? in the initial world(?:, not an? (?P<noun>[a-z][a-z ]*?))?'
    r'(?:; (?P<tool>[A-Za-z0-9_]+) accepted "(?P<target>.*?)" as "(?P<arg>.*?)" (?:at turn (?P<at>\d+)|in another recorded attempt at this task))?\.$')
CALL_FIX_LIST = re.compile(r'^The error at turn (?P<turn>\d+) lists "(?P<entry>.*?)", the only listed entry containing "(?P<v>.*?)"\.$')


def error_suggestions(error: str) -> list[str] | None:
    """The entries of the suggestion list a tool appended to its error ("Known spreadsheets: ['A (ss_a)', 'B']"),
    as the attempt received them; None without such a list."""
    from .record.calls import _SUGGESTION

    text = re.sub(r"\s+", " ", error or "").strip()
    m = _SUGGESTION.search(text)
    if not m:
        return None
    clause = m.group(0)
    body = clause.split(":", 1)[1] if ":" in clause else clause
    quoted = re.findall(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"", body)
    entries = [a or b for a, b in quoted]
    if not entries:
        entries = [e.strip(" []().") for e in body.split(",")]
    return [e for e in entries if e.strip()]


def not_found_names(message: str, value: Any) -> bool:
    """The error says that this very value was not found ("User 'X' not found", "Template not found: X",
    "Ticket with ID X not found"), not merely that it mentions it ("… not found in spreadsheet 'X'")."""
    v = re.escape(str(value).strip())
    return bool(re.search(r"['\"]?(?<![A-Za-z0-9_@.\-])" + v + r"['\"]?\s+(?:was\s+)?not found\b", message)
                or re.search(r"\bnot found:\s*['\"]?" + v + r"(?![A-Za-z0-9_@\-]|\.[A-Za-z0-9])", message))


def failing_value(call: Call) -> tuple[str, Any] | None:
    """(argument key, value): the scalar argument of the failed call (3+ characters) that the error message says
    was not found."""
    from .record.calls import error_message

    message = error_message(call.error or "")
    for path, v in _flat_args(call.args):
        if not path or not isinstance(v, (str, int)) or isinstance(v, bool) or norm(v) is None:
            continue
        if len(str(v).strip()) >= 3 and not_found_names(message, v):
            return path[-1], v
    return None


def error_noun(message: str, value: Any) -> str | None:
    """The kind of object the error says it could not find ("User 'X' not found" -> "user")."""
    v = re.escape(str(value).strip())
    m = re.search(r"\b([A-Za-z]+)(?: with (?:id|ID|name))?\s+['\"]?" + v + r"['\"]?\s+(?:was\s+)?not found", message)
    if not m:
        m = re.search(r"\b([A-Za-z]+) not found:\s*['\"]?" + v, message)
    if not m:
        return None
    word = m.group(1).lower()
    return None if word in ("error", "the", "a", "an", "with", "id", "value", "matching") else word


def _parent(index: WorldIndex, app: str, rec: dict) -> tuple[str, int, dict] | None:
    """The record of the same app another record belongs to: a `<singular>_id` field whose value is the id of one
    record of the collection with that singular (a worksheet's spreadsheet)."""
    for key, value in rec.items():
        if key == "id" or not key.endswith("_id") or isinstance(value, (dict, list)) or norm(value) is None:
            continue
        word = key[: -len("_id")]
        for coll_app, coll, records in collections(index.state):
            if coll_app != app or singular(coll).replace(" ", "_") != word:
                continue
            hits = [(i, r) for i, r in enumerate(records) if norm(r.get("id")) == norm(value)]
            if len(hits) == 1:
                return coll, hits[0][0], hits[0][1]
    return None


def accepted_by(record: Record, sibling_calls: list[list[Call]], app: str, target: Any, failed: Call) -> dict | None:
    """A recorded call without error (the attempt at another turn, else a sibling) of the app that took `target` as
    one of its arguments: {"tool", "arg", "turn" (None for a sibling)}; the failed call's own tool first, then a tool
    of the same kind (a read for a read, a write for a write), then the attempt's own calls."""
    from .record.calls import READ_VERBS

    def reads(tool: str) -> bool:
        return any(w in READ_VERBS for w in tool.split("_"))
    want = norm(target)
    if want is None:
        return None
    found = []
    for pool, own in ((list(record.calls), True), *[(calls, False) for calls in sibling_calls or []]):
        for call in pool:
            if call.error or not app_matches(call.app, app) or (own and (call.turn, call.index) == (failed.turn, failed.index)):
                continue
            for path, v in _flat_args(call.args):
                if path and not isinstance(v, (dict, list, bool)) and norm(v) == want:
                    found.append((call.tool != failed.tool, reads(call.tool) != reads(failed.tool), not own,
                                  {"tool": call.tool, "arg": path[-1], "turn": call.turn if own else None}))
                    break
    if not found:
        return None
    found.sort(key=lambda f: f[:3])
    return found[0][3]


def call_fix(record: Record, item: Item, index: WorldIndex, sibling_calls: list[list[Call]] | None) -> dict | None:
    """Fix 2: one sentence naming the fix of a failed call, only when it is proved:
    - by the recorded initial world: the value the error names is, in exactly one place of the app's world, the id or
      an own name of a record (with the record's id, and the record it belongs to); when the error names another kind
      ("User 'C_RECRUIT' not found" for a channel id) the sentence says "not a user"; a recorded call without error
      that took the record's id (or the value) names the call and argument;
    - or by the error itself: its list of known values holds exactly one entry containing the value.
    None otherwise (the bare error stays)."""
    from .record.calls import error_message

    call = next((c for c in record.calls if item.location.call is not None and (c.turn, c.index) ==
                 (item.location.call.turn, item.location.call.index)), None)
    if call is None or not call.error:
        return None
    found = failing_value(call)
    if found is None:
        return None
    key, value = found
    message = error_message(call.error)
    noun = error_noun(message, value)
    base = {"kind": "call_fix", "item": item.id, "turn": call.turn, "call_index": call.index, "tool": call.tool,
            "at_turn": call.turn, "at_tool": call.tool, "value": str(value).strip(), "arg": key}
    places = [p for p in index.find(value) if app_matches(call.app, p[0])]
    if len(places) > 1 and rank_places(places) is not None:
        places = [rank_places(places)]
    all_places = index.find(value)
    if len(places) == 1 and (len(all_places) == 1 or rank_places(all_places) == places[0]):
        app, coll, i, path = places[0]
        rec = index.record(app, coll, i)
        is_id = path == ("id",)
        own_name = len(path) == 1 and path[0] in own_label_fields(coll) + ("email", "label")
        word = singular(coll).split()[-1]
        rid = rec.get("id")
        if (is_id or own_name) and not (is_id and noun == word) and isinstance(rid, (str, int)) and not isinstance(rid, bool):
            label = record_label(rec, coll)
            app_words = app.replace("_", " ")
            label_text = f' "{label[0]}"' if label and norm(label[0]) != norm(value) else ""
            id_text = f' (id "{rid}")' if not is_id else ""
            parent = _parent(index, app, rec)
            parent_text = ""
            meta = {**base, "route": "world", "app": app, "collection": coll, "index": i, "field": list(path),
                    "label": label[0] if label_text else None, "id": None if is_id else str(rid), "noun": None, "accepted": None}
            if parent is not None:
                plabel = record_label(parent[2], parent[0])
                if plabel is not None:
                    parent_text = f' of {app_words} {singular(parent[0])} "{plabel[0]}" (id "{parent[2]["id"]}")'
                    meta["parent"] = {"collection": parent[0], "index": parent[1], "label": plabel[0], "id": str(parent[2]["id"])}
            not_text = ""
            if noun and noun != word:
                not_text = f", not {'an' if noun[:1] in 'aeio' else 'a'} {noun}"
                meta["noun"] = noun
            accepted = accepted_by(record, sibling_calls or [], app, rid, call)
            acc_text = ""
            if accepted is not None:
                where = f"at turn {accepted['turn']}" if accepted["turn"] is not None else "in another recorded attempt at this task"
                acc_text = f'; {accepted["tool"]} accepted "{rid}" as "{accepted["arg"]}" {where}'
                meta["accepted"] = {**accepted, "value": str(rid)}
            if not (acc_text or not_text or id_text or parent_text):
                return None  # nothing the bare error does not already say
            meta["line"] = (f'The value "{meta["value"]}" given at turn {call.turn} is the {field_words(path)} of {app_words} '
                            f'{singular(coll)}{label_text}{id_text}{parent_text} in the initial world{not_text}{acc_text}.')
            return meta
    entries = error_suggestions(call.error)
    if entries:
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(str(value).strip()) + r"(?![A-Za-z0-9])", re.IGNORECASE)
        hits = [e for e in entries if pattern.search(e)]
        if len(hits) == 1 and norm(hits[0]) != norm(value):
            return {**base, "route": "error_list", "entry": hits[0],
                    "line": f'The error at turn {call.turn} lists "{hits[0]}", the only listed entry containing "{str(value).strip()}".'}
    return None


def call_fixes(record: Record, texts: dict[str, str | None], index: WorldIndex,
               sibling_calls: list[list[Call]] | None) -> list[dict]:
    """Fix 2 for every rendered failed-call statement (one per distinct sentence, at most MAX_CALL_FIX_LINES)."""
    out: list[dict] = []
    seen: set[str] = set()
    for item in record.items:
        if item.type != "tool_error" or item.outcome != "failed" or texts.get(item.id) is None:
            continue
        meta = call_fix(record, item, index, sibling_calls)
        if meta is None or meta["line"] in seen:
            continue
        seen.add(meta["line"])
        out.append(meta)
        if len(out) >= MAX_CALL_FIX_LINES:
            break
    return out


def hints(record: Record, texts: dict[str, str | None], *, initial_state: dict | None, timeline,
          sibling_calls: list[list[Call]] | None = None,
          contract: dict | None = None, siblings: list | None = None, solution: dict | None = None,
          assertions: list[dict] | None = None, request_text: str = "",
          kinds: frozenset[str] | None = None) -> dict[str, Any]:
    """The hint lines for a record's failed requirements and the count of every silence.

    `kinds`: the selected fix-hint kinds (FIX_HINT_KINDS names; None = all); lines of other kinds are not returned,
    and the report then names the selected kinds (`kinds`). Returns {"lines": [{item, kind, line, ...}], "withheld": [{item, key, value,
    reason}], "by_item": {item id: status}}."""
    from .render import plain

    index = WorldIndex(initial_state)
    live = ToolMap()
    live.learn_calls(list(record.calls), index, record.episode_id)
    for calls in sibling_calls or []:
        live.learn_calls(calls, index, "sibling")
    # correction r0: no map learned from other runs; only what this attempt's and its siblings' calls returned, and a
    # tool is named only when one of those calls returned the very record (`returned_record`)
    merged = live
    pools = [list(record.calls)] + [list(c) for c in (sibling_calls or [])]
    lines: list[dict] = []
    withheld: list[dict] = []
    by_item: dict[str, str] = {}
    seen_lines: set[str] = set()
    wrote: dict[tuple[str, str], bool | None] = {}
    for item in record.items:
        if item.outcome != "failed" or item.type == "tool_error":
            continue
        if texts.get(item.id) is not None:
            rule = None  # fix round: "This applies to the rows of worksheet …" removed (judged not useful)
            if rule is not None and rule["line"] not in seen_lines and sum(1 for m in lines if m["kind"] == "rule_source") < MAX_WRITE_LINES:
                seen_lines.add(rule["line"])
                lines.append(rule)
        if item.polarity != "require":
            continue
        if texts.get(item.id) is None:
            by_item[item.id] = "statement_withheld"
            withheld.append({"item": item.id, "key": None, "value": None, "reason": "statement_withheld"})
            continue
        status = "other"
        reasons: list[str] = []
        for key, value, is_address, written in _candidate_values(item):
            if norm(value) is None:
                withheld.append({"item": item.id, "key": key, "value": value, "reason": "value_unsearchable"})
                reasons.append("value_unsearchable")
                continue
            if written and value_used(list(record.calls), value):
                withheld.append({"item": item.id, "key": key, "value": value, "reason": "value_already_used"})
                reasons.append("value_already_used")
                continue
            places = index.find(value)
            if len(places) > 1 and rank_places(places) is not None:
                places = [rank_places(places)]
            if not places:
                reason = "not_in_initial_world_substring" if index.substring_hit(value) else "not_in_initial_world"
                withheld.append({"item": item.id, "key": key, "value": value, "reason": reason})
                reasons.append(reason)
                continue
            if len(places) > 1:
                withheld.append({"item": item.id, "key": key, "value": value, "reason": "several_places",
                                 "places": [f"{a}/{c}[{i}].{'.'.join(p)}" for a, c, i, p in places[:6]]})
                reasons.append("several_places")
                continue
            place = places[0]
            if not is_address and not identifying_field(place[3], value):
                withheld.append({"item": item.id, "key": key, "value": value, "reason": "content_value_in_plain_field",
                                 "places": [f"{place[0]}/{place[1]}[{place[2]}].{'.'.join(place[3])}"]})
                reasons.append("content_value_in_plain_field")
                continue
            rec = index.record(place[0], place[1], place[2])
            tools = [t for t in merged.tools_for(place[0], place[1])
                     if any(returned_record(c, t, place[0], rec) for calls in pools for c in calls)][:2]
            meta = value_line(index, place, value, tools)
            if meta is None:
                withheld.append({"item": item.id, "key": key, "value": value, "reason": "record_unlabelled"})
                reasons.append("record_unlabelled")
                continue
            meta.update({"item": item.id, "key": key})
            at = failed_call_for(record, meta["app"], meta["collection"], key)
            if at is not None:
                meta["at_turn"], meta["at_tool"] = at.turn, at.tool
            if meta["line"] not in seen_lines and sum(1 for m in lines if m["kind"] == "value_location") < MAX_VALUE_LINES:
                seen_lines.add(meta["line"])
                lines.append(meta)
            status = "hint" if tools else ("hint_location_only" if status != "hint" else status)
        if status == "other" and reasons:
            for r in ("several_places", "not_in_initial_world_substring", "not_in_initial_world",
                      "content_value_in_plain_field", "record_unlabelled", "value_already_used", "value_unsearchable"):
                if r in reasons:
                    status = r
                    break
        if status == "other" and not reasons:
            status = "no_value_to_look_up"
            withheld.append({"item": item.id, "key": None, "value": None, "reason": "no_value_to_look_up"})
        by_item[item.id] = status
        # the wrong value: a call that wrote another value where the requirement expects one
        for meta in wrong_values(record, item, timeline):
            if meta["line"] not in seen_lines:
                seen_lines.add(meta["line"])
                lines.append(meta)
        # the missed call
        if item.effect.kind in ("read", "tool_error", "other", "action", "flag", "count"):
            continue
        target = effect_collection(item, initial_state)
        if target is None:
            withheld.append({"item": item.id, "key": None, "value": None, "reason": "write_collection_unknown"})
            continue
        if target not in wrote:
            wrote[target] = collection_written(timeline, *target)
        if wrote[target] is None:
            withheld.append({"item": item.id, "key": None, "value": None, "reason": "write_no_snapshots"})
            continue
        if not steps_complete(list(record.calls), timeline):
            withheld.append({"item": item.id, "key": None, "value": None, "reason": "write_steps_incomplete"})
            continue
        if wrote[target]:
            continue
        noun = plain.noun(item)
        existing = next((m for m in lines if m["kind"] == "missed_write" and (m["app"], m["collection"]) == target
                         and m["noun"] == noun), None)
        if existing is not None:
            existing["more"] += 1
            existing["items"].append(item.id)
            existing["line"] = _write_line(existing)
            continue
        if sum(1 for m in lines if m["kind"] == "missed_write") >= MAX_WRITE_LINES:
            continue
        meta = {"kind": "missed_write", "item": item.id, "items": [item.id], "app": target[0], "collection": target[1],
                "noun": noun, "more": 0}
        meta["line"] = _write_line(meta)
        lines.append(meta)
    # N2: what a successful sibling's calls of a failed tool did differently (facts about the two calls only)
    # N2 (cause_diff) removed in the fix round (judged misleading; one false claim about an attempt not shown)
    # N3: a change to an existing record no requirement names (never at a call a located line already covers)
    located = {(i.location.call.turn, i.location.call.index) for i in record.items
               if i.outcome == "failed" and i.polarity in ("forbid", "keep") and i.location.kind == "at_call"
               and i.location.call is not None and texts.get(i.id) is not None}
    lines.extend(unrequested_changes(record, timeline, contract, assertions, request_text, located))
    # Fix 2: the fix of a failed call, when the recorded world or the error itself proves it
    lines.extend(m for m in call_fixes(record, texts, index, sibling_calls) if m["line"] not in seen_lines)
    report = {"lines": lines, "withheld": withheld, "by_item": by_item,
              "summary": dict(Counter(by_item.values())), "withheld_reasons": dict(Counter(w["reason"] for w in withheld))}
    if kinds is not None and set(kinds) != ALL_KINDS:
        # only the selected line kinds (the others were computed only for the counts above); the report names the
        # selected kinds, which the renderer and the truth checker read
        report["lines"] = [m for m in lines if m["kind"] in kinds]
        report["kinds"] = sorted(kinds)
    return report


def _write_line(meta: dict) -> str:
    """Joined to the requirement's own "Not done" line (which already says what was needed)."""
    return f"No call wrote {_article(meta['noun'])} {meta['noun']}."


def _flat_args(node: Any, prefix: tuple[str, ...] = (), depth: int = 0) -> Iterator[tuple[tuple[str, ...], Any]]:
    if isinstance(node, dict) and depth < 4:
        for k, v in node.items():
            yield from _flat_args(v, prefix + (str(k),), depth + 1)
    elif isinstance(node, list) and depth < 4:
        for v in node:
            yield from _flat_args(v, prefix, depth + 1)
    else:
        yield prefix, node


def value_used(calls: list[Call], value: Any) -> bool:
    """The attempt wrote the value itself: as a whole argument value, or as a whole token inside one
    (word characters and "_" bound it: "tmpl_offer_letter_001" does not use "tmpl_offer_letter")."""
    want = norm(value)
    if want is None:
        return False
    pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(want) + r"(?![A-Za-z0-9_])", re.IGNORECASE)
    for call in calls:
        for _path, v in _flat_args(call.args):
            if norm(v) == want or (isinstance(v, str) and pattern.search(v)):
                return True
    return False


def _key_norm(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def failed_call_for(record: Record, app: str, coll: str, key: str) -> Call | None:
    """The first failed call of the collection's app whose arguments name the collection (an argument key
    containing its singular, "template_id" for templates) or the requirement's own key: where the hint acts."""
    wanted = {_key_norm(singular(coll)), _key_norm(key.split(".")[-1].split("#")[0])}
    for call in record.calls:
        if not call.error or not app_matches(call.app, app):
            continue
        keys = {_key_norm(p[-1]) for p, _v in _flat_args(call.args) if p}
        if any(w and any(w in k for k in keys) for w in wanted):
            return call
    return None


def expected_field(item: Item, key: str) -> str | None:
    """The field a content value is expected in: the address's `field`/`property`/`column` for a bare
    `value`, else the content key without its `_contains` suffix."""
    base = key.split(".")[-1].split("#")[0]
    if base in ("value", "expected_value", "cell_contains", "cells"):
        for k in ("field", "property", "column"):
            if k in item.address and not isinstance(item.address[k].raw, (dict, list)):
                return str(item.address[k].raw)
        return None
    if base in SCHEMA_KEYS or base.startswith("body") or base.startswith("text"):
        return None
    return base[:-9] if base.endswith("_contains") else base


def changed_record_field(timeline, call: Call, app: str, collection_object: str, record_id: Any, field: str) -> Any:
    """The value of `field` on the object of the effect's collection carrying `record_id` after the call,
    when that object differs before and after it (the call changed that record itself, not a record that
    references it); None otherwise."""
    from .record.probe import _same_collection

    if timeline is None or not getattr(timeline, "available", False) or not call.before_digest or not call.after_digest:
        return None
    before, after = timeline.json(call.before_digest), timeline.json(call.after_digest)
    if before is None or after is None:
        return None
    want = norm(record_id)
    for app_key, data in after.items():
        if not isinstance(data, dict) or not app_matches(app, str(app_key)):
            continue
        for coll, records in data.items():
            if not isinstance(records, list) or not _same_collection(str(coll), collection_object):
                continue
            old = {norm(r.get("id")): r for r in ((before.get(app_key) or {}).get(coll) or []) if isinstance(r, dict)}
            for rec in records:
                if not isinstance(rec, dict) or norm(rec.get("id")) != want or old.get(want) == rec:
                    continue
                for k, v in rec.items():
                    if _key_norm(k) == _key_norm(field) and norm(v) is not None:
                        return v
    return None


def contains_mode(item: Item, key: str) -> bool:
    """Whether the check accepts the value inside a longer one (a containment or mention), not only equal."""
    base = key.split("#")[0].split(".")[-1]
    mode = (item.semantics.get("modes") or {}).get(key)
    if item.origin == "manifest":
        return mode not in (None, "eq", "in")
    return base.endswith("_contains") or "contains" in item.type


def wrong_values(record: Record, item: Item, timeline=None) -> list[dict]:
    """"At turn t, TOOL wrote "X" as FIELD ADDRESS; the task needed "Y"": a successful call of the effect's
    app whose arguments carry every address value and, under a key naming the expected field, a scalar
    other than the expected value."""
    from .render import plain

    address = [(k, v.raw) for k, v in item.address.items() if k.split(".")[-1] not in SCHEMA_KEYS
               and not isinstance(v.raw, (dict, list)) and norm(v.raw) is not None]
    if not address:
        return []
    addr_keys = {k for k, _ in address}
    negated = set(item.semantics.get("negated_keys") or ())
    groups = item.semantics.get("alternatives") or {}
    out = []
    for key, value in item.content.items():
        field = expected_field(item, key)
        expected = value.raw
        if field is None or isinstance(expected, (dict, list)) or norm(expected) is None:
            continue
        if key in negated:
            continue  # the content must NOT be this value: "the task needed" it would be false
        if key in groups:  # one alternative of an `any`: every member is named, and any of them satisfies it
            members = [v.raw for k, v in item.content.items() if groups.get(k) == groups[key] and k not in negated]
            if any(isinstance(m, (dict, list)) or norm(m) is None for m in members):
                continue
            needed = [str(m).strip() for m in dict.fromkeys(members)]
        else:
            needed = [str(expected).strip()]
        for call in record.calls:
            if call.error or not app_matches(call.app, item.effect.app) or call.mutated is False:
                continue
            args_text = json.dumps(call.args, sort_keys=True).lower()
            if not all(norm(v) in args_text for _k, v in address):
                continue
            record_id = next((v for k, v in address if k.split(".")[-1] in ("record_id", "native_record_id", "id", "contact_id",
                                                                           "ticket_id", "row_id", "message_id", "envelope_id")
                              or k.split(".")[-1].endswith("_id")), None)
            for path, written in _flat_args(call.args):
                if not path or _key_norm(path[-1]) != _key_norm(field) or norm(written) is None or len(str(written)) > 80:
                    continue
                if any(norm(written) == norm(n) or norm(n) in norm(written) for n in needed):
                    continue
                # the call must have changed that field of that record itself (per the snapshots), never a
                # record that merely references it through an argument
                on_record = changed_record_field(timeline, call, item.effect.app, str(item.semantics.get("collection") or item.effect.object),
                                                record_id, field) if record_id is not None else None
                if on_record is None or norm(on_record) != norm(written):
                    continue
                shown = str(written).strip()
                needs = " or ".join(f'"{n}"' for n in needed)
                # fix round: "in that field", never "instead of"; for a containment check, what is missing is said
                tail = ""
                if contains_mode(item, key):
                    tail = ", and the written value does not contain it" if len(needed) == 1 else ", and the written value contains none of them"
                line = (f'At turn {call.turn}, {call.tool} wrote "{shown}" as {field.replace("_", " ")} '
                        f'{plain.address_phrase(item, True, keys=addr_keys)}; the task needed {needs} in that field{tail}.')
                out.append({"kind": "wrong_value", "item": item.id, "line": line, "turn": call.turn, "index": call.index,
                            "tool": call.tool, "field": field, "written": shown, "expected": needed[0], "needs": needed,
                            "key": key, "address": {k: v for k, v in address}, "at_turn": call.turn})
                break
            else:
                continue
            break
    return out


def binding_value(raw: Any) -> bool:
    """A value a correct solution must write verbatim (a number, an amount, a date, an email, a phrase);
    a single state word ("sent", "High") describes an effect the environment sets and is not compared."""
    text = str(raw).strip()
    return bool(re.search(r"\d|@| ", text))


SOLUTION_ABSENCE = re.compile(r'^Note: this solution does not write (?P<needs>"(?:[^"]*)"(?: or "(?:[^"]*)")*); the task needs it\.$')
SOLUTION_NOTE = re.compile(r'^Note: this solution wrote "(?P<written>.*?)" in (?P<field>.+?) of (?P<tool>\S+); the task needs (?P<needs>"(?:[^"]*)"(?: or "(?:[^"]*)")*)\.$')


def _shape(token: str) -> str:
    return re.sub(r"[A-Za-z]", "a", re.sub(r"\d", "0", token))


def written_value_for(calls: list[Call], field: str, expected: Any) -> tuple[str, str, str] | None:
    """(tool, argument key, token) the solution wrote where `expected` was needed: the argument whose key
    names the field, and in it the token of the same shape as the expected value (digits for digits) that
    shares the longest prefix with it. None when nothing of that shape was written: the disagreement cannot
    be stated precisely."""
    want = str(expected).strip()
    if not re.search(r"\d", want):
        return None
    best = None
    for call in calls:
        for path, value in _flat_args(call.args):
            if not path or _key_norm(field) not in _key_norm(path[-1]) or not isinstance(value, (str, int, float)):
                continue
            for token in re.findall(r"[\w.,%$:/-]+", str(value)):
                token = token.strip(".,:")
                if _shape(token) != _shape(want) or norm(token) == norm(want):
                    continue
                prefix = 0
                for a, b in zip(token, want):
                    if a != b:
                        break
                    prefix += 1
                if best is None or prefix > best[0]:
                    best = (prefix, call.tool, path[-1], token)
    return (best[1], best[2], best[3]) if best else None


def disagreement_notes(calls: list[Call], record: Record, outcomes: dict[int, str] | None = None,
                       manifest_outcomes: dict | None = None) -> tuple[list[dict], list[dict]]:
    """One note per expected-value group the solution does not satisfy (de-duplicated); the groups no note
    can state."""
    text = "\n".join(f"{c.tool}({json.dumps(c.args)})" for c in calls)
    notes, unstated = [], []
    seen: set[str] = set()
    for miss in solution_agrees(text, record, outcomes, manifest_outcomes):
        if miss.get("unstatable"):
            unstated.append(miss)
            continue
        item = next(i for i in record.items if i.id == miss["item"])
        key = next((k for k, v in item.content.items() if str(v.raw).strip() == miss["values"][0]
                    or (isinstance(v.raw, list) and miss["values"][0] in [str(x).strip() for x in v.raw])), None)
        field = (expected_field(item, key) if key else None) or (key or "").split("#")[0].split("_contains")[0].split("_text")[0]
        found = written_value_for(calls, field, miss["values"][0]) if field else None
        if found is None:
            # no written token of the expected value's shape: the absence is what can be said precisely
            needs = " or ".join(f'"{v}"' for v in miss["values"])
            line = f"Note: this solution does not write {needs}; the task needs it."
            if line not in seen:
                seen.add(line)
                notes.append({"kind": "solution_absence", "item": item.id, "line": line, "needs": miss["values"],
                              **({"flags": ["unverified_by_solution_findings"]} if miss.get("unverified") else {})})
            continue
        tool, arg_key, token = found
        needs = " or ".join(f'"{v}"' for v in miss["values"])
        line = f'Note: this solution wrote "{token}" in {arg_key.replace("_", " ")} of {tool}; the task needs {needs}.'
        if line in seen:
            continue
        seen.add(line)
        notes.append({"kind": "solution_disagreement", "item": item.id, "line": line, "tool": tool, "field": arg_key,
                      "written": token, "needs": miss["values"],
                      **({"flags": ["unverified_by_solution_findings"]} if miss.get("unverified") else {})})
    return notes, unstated


def binding_item(item: Item) -> bool:
    """A requirement whose expected values a solution must write: a scored require check (failed or passed)
    about something written, never a read requirement, a tool error or an excluded check."""
    return (item.polarity == "require" and item.type != "tool_error" and item.outcome in ("failed", "passed")
            and item.effect.kind != "read" and not item.control.get("excluded"))


def sibling_outcomes(sibling) -> dict[int, str]:
    """index -> outcome of every native check recorded for a sibling (passed / failed / excluded ...)."""
    from .record.native import native_item

    if sibling is None:
        return {}
    out = {}
    for index, result in enumerate(sibling.native_results):
        try:
            out[index] = native_item(index, result, None).outcome
        except Exception:  # noqa: BLE001 - an unreadable result is "unknown": no note is derived from it
            out[index] = "unknown"
    return out


def sibling_manifest_outcomes(sibling) -> dict[tuple, int]:
    """(check_id, candidate key or None) -> obligation value (1 met, 0 unmet) of a sibling's own required
    obligation findings (its assessment batches): what the solution itself did for each manifest goal."""
    from .record.manifest import OBLIGATION_OUTPUT, findings

    if sibling is None:
        return {}
    out: dict[tuple, int] = {}
    try:
        payloads = findings(sibling.assessment_batches)[OBLIGATION_OUTPUT]
    except Exception:  # noqa: BLE001 - unreadable batches: nothing is known
        return {}
    for p in payloads:
        if p.get("kind") != "finding" or p.get("status") != "valid" or p.get("required") is False:
            continue
        cand = p.get("candidate_identity")
        key = (p.get("check_id"), str(cand[-1]) if cand else None)
        value = 1 if p.get("value") == 1 else 0
        out[key] = min(out.get(key, 1), value) if key in out else value
    return out


def manifest_outcome(item: Item, manifest_outcomes: dict | None) -> int | None:
    if not manifest_outcomes or item.origin != "manifest":
        return None
    key = (item.check_id, str(item.candidate.get("key")) if item.candidate else None)
    return manifest_outcomes.get(key)


def _satisfied(item: Item, low: str) -> bool:
    """Every content group of the item has one alternative in the solution text."""
    groups = item.semantics.get("alternatives") or {}
    negated = set(item.semantics.get("negated_keys") or ())
    by_group: dict[Any, list[Any]] = {}
    for key, value in item.content.items():
        if key in negated or key.split(".")[-1] in SCHEMA_KEYS:
            continue
        raws = value.raw if isinstance(value.raw, list) else [value.raw]
        for raw in raws:
            if isinstance(raw, (dict, list)) or norm(raw) is None or not binding_value(raw):
                continue
            by_group.setdefault(groups.get(key, ("own", key)), []).append(raw)
    return all(any(norm(raw) in low for raw in members) for members in by_group.values())


def _same_requirement(a: Item, b: Item) -> bool:
    """Two checks of one requirement: same effect and the same address values (sibling checks that each
    accept one alternative of a value)."""
    return (a.effect.app, a.effect.object) == (b.effect.app, b.effect.object) and a.polarity == b.polarity and \
        {k: str(v.raw) for k, v in a.address.items()} == {k: str(v.raw) for k, v in b.address.items()}


def solution_agrees(text: str, record: Record, outcomes: dict[int, str] | None = None,
                    manifest_outcomes: dict | None = None) -> list[dict]:
    """The expected values of the binding requirements that the solution does not satisfy (one alternative
    of an `any` group suffices; a sibling check of the same requirement that the solution satisfies counts;
    a native check recorded as passed for the solution's own episode, `outcomes`, never yields one)."""
    missing = []
    low = text.lower()
    binding = [i for i in record.items if binding_item(i)]
    for item in binding:
        if item.origin == "native" and outcomes is not None:
            index = int(item.id.split(":")[1])
            if outcomes.get(index) != "failed":
                continue  # the grader passed (or did not score) this check on the solution itself
        if manifest_outcome(item, manifest_outcomes) == 1:
            continue  # the solution's own obligation finding met this goal
        if any(other is not item and _same_requirement(item, other) and _satisfied(other, low) for other in binding):
            continue
        groups = item.semantics.get("alternatives") or {}
        negated = set(item.semantics.get("negated_keys") or ())
        by_group: dict[Any, list[tuple[str, Any]]] = {}
        for key, value in item.content.items():
            if key in negated or key.split(".")[-1] in SCHEMA_KEYS:
                continue
            raws = value.raw if isinstance(value.raw, list) else [value.raw]
            for raw in raws:
                if isinstance(raw, (dict, list)) or norm(raw) is None or not binding_value(raw):
                    continue
                by_group.setdefault(groups.get(key, ("own", key)), []).append((key, raw))
        every: dict[Any, list] = {}
        for key, value in item.content.items():
            if key in negated or key.split(".")[-1] in SCHEMA_KEYS:
                continue
            for raw in (value.raw if isinstance(value.raw, list) else [value.raw]):
                if not isinstance(raw, (dict, list)):
                    every.setdefault(groups.get(key, ("own", key)), []).append(str(raw).strip())
        for group, members in by_group.items():
            if any(norm(raw) in low for _k, raw in members):
                continue
            if sorted(set(every.get(group, []))) != sorted(set(str(raw).strip() for _k, raw in members)):
                # scale fix (shape removed): an alternative the note would leave out; the note cannot be exact
                missing.append({"item": item.id, "values": [str(raw) for _k, raw in members], "unstatable": True})
                continue
            missing.append({"item": item.id, "values": [str(raw) for _k, raw in members],  # every accepted alternative
                            **({"unverified": True} if item.origin == "manifest" and manifest_outcome(item, manifest_outcomes) is None else {})})
    return missing


# ---------------------------------------------------------------------------------------------
# N2: argument differences between this attempt's failed calls of a tool and a successful sibling's calls
# ---------------------------------------------------------------------------------------------
CAUSE_LINE = re.compile(r'^(?P<subject>A successful attempt at this task|The correct solution above) called (?P<tool>\S+) '
                        r'(?P<clauses>.+)\.$')
CAUSE_CLAUSE = re.compile(r'(?:^|, )(?P<kind>without|with) "(?P<key>[^"]+)"(?:: (?P<value>"(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?|true|false|null))?')
MAX_CAUSE_LINES = 3
MAX_CAUSE_CLAUSES = 3


def short_scalar(v: Any) -> bool:
    """A value worth naming in a difference: a number, a boolean, or a string without spaces of at most 40
    characters (an id, an enum, a method, a date); free text is never compared."""
    if isinstance(v, bool) or isinstance(v, (int, float)):
        return True
    return isinstance(v, str) and 0 < len(v) <= 40 and not re.search(r"\s", v)


def _req_items(record: Record, app: str) -> list[Item]:
    return [i for i in record.items if i.polarity == "require" and i.type != "tool_error" and app_matches(app, i.effect.app)]


def _expected_values(items: list[Item]) -> set[str]:
    out = set()
    for i in items:
        for v in [*i.address.values(), *i.content.values()]:
            for x in (v.raw if isinstance(v.raw, list) else [v.raw]):
                if not isinstance(x, (dict, list)) and norm(x) is not None:
                    out.add(norm(x))
    return out


def _named_keys(items: list[Item], assertions: list[dict] | None, app: str) -> dict[str, set[str]]:
    """normalised key name -> the expected values named for it (by a requirement's address/content key or an
    assertion parameter of the app)."""
    out: dict[str, set[str]] = defaultdict(set)
    for i in items:
        for k, v in [*i.address.items(), *i.content.items()]:
            name = _key_norm(k.split("#")[0].split(".")[-1].removesuffix("_contains"))
            for x in (v.raw if isinstance(v.raw, list) else [v.raw]):
                if not isinstance(x, (dict, list)) and norm(x) is not None:
                    out[name].add(norm(x))
            out.setdefault(name, set())
    for a in assertions or []:
        if not isinstance(a, dict) or not str(a.get("type", "")).replace("_", "").startswith(app.replace("_", "")[:6]):
            continue
        for k, v in a.items():
            if k == "type":
                continue
            name = _key_norm(str(k).removesuffix("_contains"))
            out.setdefault(name, set())
            if not isinstance(v, (dict, list)) and norm(v) is not None:
                out[name].add(norm(v))
    return out


def contract_requirements(contract: dict | None, app: str) -> dict[str, set[str]]:
    """normalised effect field name -> literal values the contract's goal checks of the app compare it with (a
    manifest goal merged into a native item keeps no content of its own in the Record)."""
    from .record.manifest import source_effect

    out: dict[str, set[str]] = defaultdict(set)
    sources = (contract or {}).get("sources") or {}
    for check in (contract or {}).get("checks") or []:
        if check.get("role") != "goal":
            continue
        src = sources.get(check.get("source") or "") or {}
        try:
            if not app_matches(source_effect(src).app, app):
                continue
        except Exception:  # noqa: BLE001
            continue
        for node in _walk_nodes(check.get("effect_match") or {}):
            if not isinstance(node, dict) or node.get("op") not in ("eq", "in", "mentions"):
                continue
            sides = [node.get("left"), node.get("right"), node.get("text"), node.get("value")]
            fields = [x["path"] for x in sides if isinstance(x, dict) and x.get("kind") == "field"
                      and isinstance(x.get("path"), list) and x["path"][:1] == ["effect"]]
            lits = [x.get("value") for x in sides if isinstance(x, dict) and x.get("kind") == "literal"]
            for path in fields:
                name = _key_norm(str([p for p in path if not isinstance(p, int)][-1]))
                out.setdefault(name, set())
                for lit in lits:
                    for v in (lit if isinstance(lit, list) else [lit]):
                        if norm(v) is not None:
                            out[name].add(norm(v))
    return out


def key_words_in_request(key: str, request_text: str) -> bool:
    """A key whose words the request uses ("scheduled_at" in "Schedule it for tomorrow"): a five-letter stem of
    any part of four or more letters starts a word of the request."""
    parts = [p for p in re.split(r"[_\W]+|(?<=[a-z])(?=[A-Z])", key) if len(p) >= 4]
    words = re.findall(r"[a-z]+", request_text.lower())
    return any(w.startswith(p.lower()[:5]) for p in parts for w in words)


def cause_clauses(failed: list[Call], success: list[Call], record: Record, assertions: list[dict] | None,
                  contract: dict | None = None, request_text: str = "") -> list[tuple]:
    """[(kind, key, value)]: kind "without" (every success call lacks the key, every failed call has it), "with"
    (every success call has it, every failed call lacks it) or "with_value" (every success call has the same short
    value, no failed call has it); the R2 drops applied (engine-final specs N2, R10)."""
    if not failed or not success:
        return []
    app = failed[0].app
    req = _req_items(record, app)
    expected = _expected_values(req)
    named = _named_keys(req, assertions, app)
    for values in named.values():  # scale fix: an assertion parameter's value is an expected value too
        expected |= values
    for name, values in contract_requirements(contract, app).items():
        named.setdefault(name, set()).update(values)
        expected |= values
    keys = list(dict.fromkeys([k for c in failed for k in c.args] + [k for c in success for k in c.args]))
    out: list[tuple] = []
    for k in keys:
        kn = _key_norm(str(k))
        in_success = [k in c.args for c in success]
        in_failed = [k in c.args for c in failed]
        failed_values = {norm(c.args[k]) for c in failed if k in c.args and not isinstance(c.args[k], (dict, list))}
        if not any(in_success) and all(in_failed):
            if kn in named or (failed_values & expected) or key_words_in_request(str(k), request_text):
                continue  # R2: a requirement (or the request's own words) names this key, or the attempt's value
            out.append(("without", k, None))
            continue
        if not all(in_success):
            continue
        values = [c.args[k] for c in success]
        same = all(json.dumps(v, sort_keys=True) == json.dumps(values[0], sort_keys=True) for v in values)
        if failed_values & expected:
            continue  # R2: the attempt's value is one a requirement expects
        if same and short_scalar(values[0]) and not any(
                k in c.args and json.dumps(c.args[k], sort_keys=True) == json.dumps(values[0], sort_keys=True) for c in failed):
            if kn in named and named[kn] and norm(values[0]) not in named[kn]:
                continue  # R2: a requirement expects another value for this key
            out.append(("with_value", k, values[0]))
            continue
        if not any(in_failed):
            out.append(("with", k, None))
    order = {"without": 0, "with": 1, "with_value": 2}
    return sorted(out, key=lambda c: order[c[0]])[:MAX_CAUSE_CLAUSES]


def cause_line(subject: str, tool: str, clauses: list[tuple]) -> str:
    parts = []
    for kind, key, value in clauses:
        if kind == "without":
            parts.append(f'without "{key}"')
        elif kind == "with":
            parts.append(f'with "{key}"')
        else:
            parts.append(f'with "{key}": {json.dumps(value)}')
    return f"{subject} called {tool} {', '.join(parts)}."


def cause_diffs(record: Record, siblings: list | None, solution: dict | None, assertions: list[dict] | None,
                contract: dict | None = None, request_text: str = "") -> list[dict]:
    """N2 lines, one per tool that failed (first failure order), at most three."""
    from .record.calls import episode_calls
    from .solution import qualify

    failed_by_tool: dict[str, list[Call]] = {}
    for c in record.calls:
        if c.error is not None:
            failed_by_tool.setdefault(c.tool, []).append(c)
    qualified = []
    for sib in siblings or []:
        if sib.id == record.episode_id:
            continue
        try:
            ok, _why = qualify(sib)
        except Exception:  # noqa: BLE001
            ok = False
        if ok:
            qualified.append(sib)
    out = []
    for tool, failed in failed_by_tool.items():
        if len(out) >= MAX_CAUSE_LINES:
            break
        shown = [c for c in (solution or {}).get("calls") or [] if c.tool == tool] if solution else []
        if shown:
            subject, sibling_id, success = "The correct solution above", solution["episode_id"], shown
        else:
            found = None
            for sib in qualified:
                calls = [c for c in episode_calls(sib)[0] if c.tool == tool and c.error is None]
                if calls:
                    found = (sib.id, calls)
                    break
            if found is None:
                continue
            subject, (sibling_id, success) = "A successful attempt at this task", found
        clauses = cause_clauses(failed, success, record, assertions, contract, request_text)
        if not clauses:
            continue
        out.append({"kind": "cause_diff", "item": None, "line": cause_line(subject, tool, clauses), "tool": tool,
                    "sibling": sibling_id, "subject": subject, "clauses": [list(c) for c in clauses],
                    "failed_turns": sorted({c.turn for c in failed}), "at_turn": failed[0].turn, "at_tool": tool})
    return out


# ---------------------------------------------------------------------------------------------
# N3: a change to an existing record that no requirement names
# ---------------------------------------------------------------------------------------------
BOOKKEEPING = re.compile(r"(_at|_time|timestamp|modified|updated|version|etag|_date|_count|count|total|_by)$", re.IGNORECASE)
UNREQUESTED_LINE = re.compile(r'^At turn (?P<turn>\d+), (?P<tool>\S+) changed (?P<fields>(?:"[^"]+"(?:, "[^"]+")*(?: and "[^"]+")?)|\d+ fields) '
                              r'of (?P<app>[a-z0-9 ]+?) (?P<coll>[a-z0-9 ]+?) "(?P<label>.*)"; no requirement of this task names that record\.$')
MAX_UNREQUESTED = 2


def record_key(rec: dict) -> Any:
    for k in ("id", "row_id"):
        if k in rec and isinstance(rec[k], (str, int)) and not isinstance(rec[k], bool):
            return rec[k]
    return None


def _scalars(node: Any, depth: int = 0) -> Iterator[Any]:
    if depth > 6:
        return
    if isinstance(node, dict):
        for v in node.values():
            yield from _scalars(v, depth + 1)
    elif isinstance(node, list):
        for v in node:
            yield from _scalars(v, depth + 1)
    else:
        yield node


def _token_in(name: str, text: str) -> bool:
    return bool(re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", text, re.IGNORECASE))


def record_names(rec_before: dict, rec_after: dict, label: str) -> set[str]:
    names = {norm(record_key(rec_before)), norm(label)}
    for rec in (rec_before, rec_after):
        for k, v in rec.items():
            if BOOKKEEPING.search(str(k)):
                continue
            for x in _scalars(v):
                names.add(norm(x))
    return {n for n in names if n is not None}


def collection_named(app: str, coll: str, contract: dict | None, assertions: list[dict] | None) -> bool:
    """A contract source watches the collection, or an assertion names it (rules 1 and 3 of N3)."""
    from .record.manifest import source_effect
    from .record.probe import _same_collection

    wanted = {norm(coll), norm(singular(coll)), norm(f"{app}_{coll}")} - {None}
    sing = singular(coll).replace(" ", "_").lower()
    for a in assertions or []:
        if not isinstance(a, dict):
            continue
        t = str(a.get("type", "")).lower()
        if re.search(r"(?:^|_)" + re.escape(sing) + r"(?:_|$)", t):
            return True
        for v in _scalars({k: v for k, v in a.items() if k != "type"}):
            if norm(v) in wanted:
                return True
    for src in ((contract or {}).get("sources") or {}).values():
        if not isinstance(src, dict):
            continue
        path = src.get("path") or []
        if src.get("adapter") == "initial.records@1" and len(path) >= 4 and app_matches(str(path[2]), app) and _same_collection(str(path[3]), coll):
            return True
        collection = src.get("collection")
        collection = collection[-1] if isinstance(collection, list) and collection else collection
        if isinstance(collection, str) and _same_collection(collection, coll) and app_matches(str(src.get("service") or app), app):
            return True
        if src.get("adapter") == "google_sheets.rows@1" and app_matches(app, "google_sheets") and coll == "rows":
            return True
        try:
            eff = source_effect(src)
        except Exception:  # noqa: BLE001
            continue
        if app_matches(eff.app, app) and _same_collection(coll, eff.object):
            return True
    return False


def names_in_requirements(names: set[str], assertions: list[dict] | None, contract: dict | None, record: Record) -> bool:
    texts = []
    for a in assertions or []:
        texts.extend(str(x) for x in _scalars({k: v for k, v in (a or {}).items() if k != "type"}) if x is not None)
    texts.extend(str(x) for x in _scalars(contract or {}) if isinstance(x, str))
    for n in names:
        for t in texts:
            if norm(t) == n or _token_in(n, t):
                return True
    for i in record.items:
        for v in [*i.address.values(), *i.content.values()]:
            for x in (v.raw if isinstance(v.raw, list) else [v.raw]):
                if not isinstance(x, (dict, list)) and norm(x) in names:
                    return True
    return False


def field_named(field: str, assertions: list[dict] | None, contract: dict | None) -> bool:
    f = _key_norm(field)
    for a in assertions or []:
        for k, v in (a or {}).items():
            if _key_norm(str(k).removesuffix("_contains")) == f or (isinstance(v, str) and _key_norm(v) == f):
                return True
    for node in _walk_nodes(contract or {}):
        if isinstance(node, dict) and isinstance(node.get("path"), list) and any(_key_norm(str(p)) == f for p in node["path"]):
            return True
        if isinstance(node, dict):
            for k, v in node.items():
                if _key_norm(str(k)) == f:
                    return True
                if k in ("field", "fields", "required_fields", "key_fields", "identity_path") and (
                        (isinstance(v, str) and _key_norm(v) == f) or (isinstance(v, list) and any(_key_norm(str(x)) == f for x in v))):
                    return True
    return False


def _walk_nodes(node: Any, depth: int = 0) -> Iterator[Any]:
    if depth > 30:
        return
    yield node
    if isinstance(node, dict):
        for v in node.values():
            yield from _walk_nodes(v, depth + 1)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_nodes(v, depth + 1)


def changed_records(before: dict, after: dict) -> Iterator[tuple[str, str, dict, dict, list[str]]]:
    """(app, collection, record before, record after, changed non-bookkeeping fields) for every record present
    before and after with the same id whose fields differ."""
    for app, data in (after or {}).items():
        if not isinstance(data, dict):
            continue
        for coll, recs in data.items():
            if not isinstance(recs, list):
                continue
            old = {}
            for r in ((before or {}).get(app) or {}).get(coll) or []:
                if isinstance(r, dict) and record_key(r) is not None:
                    old[json.dumps(record_key(r))] = r
            for r in recs:
                if not isinstance(r, dict) or record_key(r) is None:
                    continue
                b = old.get(json.dumps(record_key(r)))
                if b is None or b == r:
                    continue
                fields = [k for k in dict.fromkeys([*b, *r]) if b.get(k) != r.get(k) and not BOOKKEEPING.search(str(k))]
                if fields:
                    yield str(app), str(coll), b, r, fields


def unrequested_line(turn: int, tool: str, fields: list[str], app: str, coll: str, label: str) -> str:
    if len(fields) > 3:
        what = f"{len(fields)} fields"
    elif len(fields) == 1:
        what = f'"{fields[0]}"'
    else:
        what = ", ".join(f'"{f}"' for f in fields[:-1]) + f' and "{fields[-1]}"'
    return (f'At turn {turn}, {tool} changed {what} of {app.replace("_", " ")} {singular(coll)} "{label}"; '
            f"no requirement of this task names that record.")


def unrequested_changes(record: Record, timeline, contract: dict | None, assertions: list[dict] | None,
                        request_text: str, located: set[tuple]) -> list[dict]:
    out = []
    if timeline is None or not getattr(timeline, "available", False):
        return out
    for call in record.calls:
        if len(out) >= MAX_UNREQUESTED:
            break
        if call.error is not None or call.mutated is not True or not call.before_digest or not call.after_digest:
            continue
        if (call.turn, call.index) in located:
            continue
        before, after = timeline.json(call.before_digest), timeline.json(call.after_digest)
        if before is None or after is None:
            continue
        args_scalars = [x for x in _scalars(call.args) if norm(x) is not None]
        for app, coll, b, r, fields in changed_records(before, after):
            label = record_label(b, coll)
            if label is None:
                continue
            names = record_names(b, r, label[0])
            rid = norm(record_key(b))
            # the call targeted this record: its id or one of its values is in the call's own arguments
            if not any(norm(x) == rid or norm(x) in names for x in args_scalars):
                continue
            if any(_token_in(n, request_text) for n in names):
                continue
            if collection_named(app, coll, contract, assertions) or names_in_requirements(names, assertions, contract, record):
                continue
            kept = [f for f in fields if not field_named(f, assertions, contract)]
            if not kept:
                continue
            out.append({"kind": "unrequested_change", "item": None, "line": unrequested_line(call.turn, call.tool, kept, app, coll, label[0]),
                        "turn": call.turn, "index": call.index, "tool": call.tool, "app": app, "collection": coll,
                        "record_id": record_key(b), "label": label[0], "fields": kept, "at_turn": call.turn})
            break
    return out
