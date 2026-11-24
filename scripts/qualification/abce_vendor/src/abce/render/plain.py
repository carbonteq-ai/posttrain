"""Plain style: short grammatical lines from the generic grammar (docs/design.md §3, no polish table).

Nouns are derived from the item's effect (app + object); parameter keys are labelled by a small
key-label table; values pass through the visibility rule. One statement per failed item, ordered
harms first, within a line budget. A statement is withheld (None, listed in the context's coverage
report) whenever the Record cannot establish what it would claim: an existence claim without a
verdict, a prohibition whose condition the compiler could not resolve, a value the engine does not
know. "Silent when unsure" is the rule; the raw style is untouched by any of this.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..record.calls import error_message
from ..record.model import Item, Record, Value
from . import common

LINE_BUDGET = 8
CHAR_BUDGET = 2400  # of the statement lines of the first message (headings and closing excluded)
# Fix 3: the ordering policies of the first message's statement lines. "worst_first" (the default, today's order):
# located harms, unrecovered tool errors, missing work (native before manifest), recovered tool errors, the
# turn-limit line first; "turn": the same selection under the budget, shown by the first turn each line names
# (lines naming no turn after them, in worst-first order). A budget of None means no limit.
ORDERS = ("worst_first", "turn")
DEFAULT_ORDER = "worst_first"
_LINE_TURN = re.compile(r"^At turns? (\d+)|\(turns (\d+)-\d+\)")
# Correction r0: which "Not done" lines (the lines of failed requirements, item kind "missing") are shown. "all" (the
# default, the released behaviour); "benchmark": only those backed by a benchmark assertion (`benchmark_backed`), a
# rule-file-only requirement (and every note or hint derived from it) is withheld. Harms and tool errors are unchanged.
NOT_DONE_SOURCES = ("all", "benchmark")
DEFAULT_NOT_DONE_SOURCE = "all"
# Correction r0: the note placed before the attempt's final reply (`build_context(final_note=True)`)
FINAL_NOTE_KIND = "final_not_done"
FINAL_NOTE_HEAD = "Still not done at this point:"
NOT_DONE_PREFIX = "Not done: "


def benchmark_backed(item: Item) -> bool:
    """A failed requirement backed by a benchmark assertion: the item was built from one of the task's own recorded
    assertion results (`origin == "native"`, abce.record.native; a scored assertion, since an assertion the benchmark
    excludes from scoring has the outcome "excluded"), a rule-file item merged into it included (abce.record.merge keeps
    the native item and records the manifest one in `merged_from`). A manifest item left on its own (`origin ==
    "manifest"`) has only the rule file behind it, whatever native check names a related effect. Tool errors are not
    requirements."""
    return item.origin == "native" and item.type != "tool_error"


def must_not_item(item: Item) -> bool:
    """A "must not" item: a prohibition, a record that had to stay as it was, or a check the benchmark marks negative
    (`gmail_message_sent_to_with_body_not_contains` …) that the probe re-read as needing the effect
    (`semantics.effect_required`, polarity "require" from the probe): its unmet part is content to leave out."""
    if item.polarity in ("forbid", "keep") or item.semantics.get("effect_required") or item.polarity_source == "probe":
        return True
    if item.origin == "native" and item.type != "tool_error":
        from ..record.native import negative_marker

        return negative_marker(item.type) is True
    return False


def rule_file_only_requirement(item: Item) -> bool:
    """A "Not done" item (a failed requirement, item kind "missing") whose only source is the task's rule file."""
    return common.item_kind(item) == "missing" and not benchmark_backed(item)


def line_turn(line: str) -> int | None:
    """The first turn a statement line names ("At turn 4, …", "At turns 3, 5, …", "… (turns 2-9), first: …")."""
    m = _LINE_TURN.search(line)
    return int(m.group(1) or m.group(2)) if m else None

ADDRESS_LABELS = {
    "to": "to", "recipients": "to", "recipient": "to", "cc": "cc", "to_number": "to", "to_email": "to", "email": "for",
    "channel": "in channel", "channel_id": "in channel", "channel_name": "in channel", "user_id": "to user",
    "recipient_id": "to user", "recipient_user_id": "to user", "spreadsheet_id": "in spreadsheet",
    "spreadsheet": "in spreadsheet", "worksheet_id": "worksheet", "worksheet": "worksheet", "worksheet_name": "worksheet",
    "row_id": "row", "ticket_id": "on ticket", "conversation_id": "on conversation", "contact_id": "for contact",
    "record_id": "on record", "native_record_id": "on record", "parent_id": "on record", "message_id": "message",
    "thread_id": "in thread", "campaign_id": "in campaign", "envelope_id": "envelope", "meeting_id": "meeting",
    "profile_id": "to profile", "profile_url": "to profile", "signer_email": "for signer", "recipient_email": "for",
    "organization_id": "for organization", "account_id": "for account", "project": "in project", "collection": "in",
    "object_type": "of type", "field": "field", "column": "column", "property": "property", "label_id": "label",
    "label": "label", "action_key": "", "customer_name": "for customer", "contact_name": "for contact",
    "display_name": "named", "template_id": "from template", "template_name": "from template", "id": "with id",
    "list_id": "in list", "mailbox_id": "in mailbox", "group_id": "in group", "page_id": "on page",
    "event_type_name": "of event type", "assignee_email": "assigned to", "to_contains": "to", "calendar_id": "in calendar",
}
CONTENT_LABELS = {
    "body_contains": "with body containing", "body_not_contains": "with body not containing",
    "text_contains": "with text containing", "text_not_contains": "with text not containing",
    "subject": "with subject", "subject_contains": "with subject containing", "cells": "with cells",
    "cell_contains": "with cells containing", "contains": "containing", "value": "with value",
    "value_contains": "with value containing", "expected_value": "with value", "content_contains": "containing",
    "summary": "with summary", "summary_contains": "with summary containing", "topic_contains": "with topic containing",
    "title_contains": "with title containing", "title": "with title", "caption_contains": "with caption containing",
    "description_contains": "with description containing", "message_contains": "with message containing",
    "note_contains": "with note containing", "tag": "with tag", "status": "with status", "priority": "with priority",
    "stage": "at stage", "text": "with text", "params": "with", "fields": "with fields", "criteria": "matching",
    "name": "named", "name_contains": "with name containing", "first_name": "first name", "last_name": "last name",
    "company": "company", "domain": "domain", "phone": "phone", "body_text": "with body containing",
    "values_text": "with values containing", "changed_fields": "changing fields", "after_cells": "with cell",
    "template_id": "from template", "email": "for email", "assigned_to_id": "assigned to", "related_to_id": "related to",
    "due_date": "due", "group_id": "in group",
    "body_regex": "with body matching the pattern", "total_amt": "with total",  # task fixes v1.1 (14 Nov)
}
DEFAULT_LABEL = lambda key: key.replace("_", " ")  # noqa: E731
THINGS = ("message", "note", "comment", "reply", "tag", "label", "signer", "member", "event", "property", "email")


def article(noun: str) -> str:
    return "an" if noun[:1].lower() in "aeio" else "a"


def noun_parts(item: Item) -> tuple[str, str | None]:
    """(base noun, the thing a "has_<thing>" check is about or None)."""
    app = item.effect.app.replace("_", " ")
    obj = item.effect.object.replace("_", " ")
    if obj in ("", app) and item.origin == "native":
        # a type without an object word ("salesforce_field_contains"): the predicate's first word names it
        rest = item.type[len(item.effect.app) + 1:] if item.type.startswith(item.effect.app + "_") else ""
        first = rest.split("_")[0] if rest else ""
        obj = first if first in ("field", "record", "action", "task") else obj
    base = app if obj in ("", app, "population", "call") else f"{app} {obj}"
    if obj == "collection":
        # engine-fix (D79 count lines): a "<app>_collection_count_*" check counts the records of a collection ("cases"),
        # never collections: "exactly 6 salesforce records in "cases" required"
        base = f"{app} record"
    for thing in THINGS:
        if f"has_{thing}" in item.type or f"missing_{thing}" in item.type:
            return base, thing
    return base, None


# engine-fix (D79 required_effect lines): a native check named for the state its object must reach
# ("docusign_envelope_voided") says that state ("is required to be voided"), never only the object
STATE_PREDICATES = frozenset({"read", "voided", "archived", "cancelled", "canceled", "paused", "enabled", "deleted", "closed",
                              "resolved", "completed", "approved", "declined", "rejected", "merged", "pinned", "starred",
                              "unsubscribed", "suspended", "deactivated", "activated"})


STATE_PREDICATES = STATE_PREDICATES | {"liked", "followed", "no_show"}
STATE_PHRASES = {"read": "marked as read", "no_show": "marked as a no-show"}
# engine-fix 7 (judge-final3 91): the content keys that name the object itself (its name, subject), said before "is
# required to be STATE"; the other content (a reason) after the state
IDENTITY_CONTENT_KEYS = {"name", "name_contains", "subject", "subject_contains", "title", "title_contains", "summary",
                         "summary_contains", "topic", "topic_contains", "email", "dealname"}


def state_predicate(item: Item) -> str | None:
    """The state predicate of a native check's type name ("is_read" -> "read", "voided"); None for other checks."""
    if item.origin != "native" or item.type == "tool_error":
        return None
    from ..record.native import split_type

    predicate = split_type(item.type)[2]
    if predicate.startswith("is_"):  # engine-fix 7 (judge-final3 91): "gmail_message_is_read" names the state "read"
        predicate = predicate[len("is_"):]
    return predicate if predicate in STATE_PREDICATES else None


def state_word(item: Item) -> str | None:
    """The state a native check requires of its object ("voided"), from its own type name; None for other checks."""
    predicate = state_predicate(item)
    # engine-fix 2 (judge v1 9/14/33): "read" is a flag the message carries (fetching it does not set it)
    return STATE_PHRASES.get(predicate, predicate) if predicate is not None else None


def split_identity_content(item: Item, expected_values: bool) -> tuple[str, str]:
    """(identity content, other content) of a state check: "named "X"" goes before "is required to be paused", "with
    reason containing "Y"" after it."""
    bases = {k.split("#")[0] for k in item.content}
    identity = {k for k in bases if k in IDENTITY_CONTENT_KEYS}
    if not identity:
        return "", content_phrase(item, expected_values)
    return content_phrase(item, expected_values, keys=identity), content_phrase(item, expected_values, keys=bases - identity)


# engine-fix 7 (judge-final3 2 / 40): a "changed" line names what the keep check protects (a field and its value, a
# cell, a status, a mailbox, a label), never "it had to stay as it was" for a check that protects one attribute
KEEP_TAIL_WHOLE = "; it had to stay as it was"
KEEP_TAIL_WHOLE_MANY = "; they had to stay as they were"
# an unquoted value never ends with the sentence's full stop ("its priority had to stay 2." -> "2"; "1.08" keeps its dot)
_KEEP_VALUE = r'(?:"(?:[^"\\]|\\.)*"|[^\s";]*[^\s";.])'
KEEP_TAIL_RE = re.compile(r'; (?:its|their) (?P<clauses>.+? had to stay(?: as it was| ' + _KEEP_VALUE + r')?'
                          r'(?: and (?:its|their) .+? had to stay(?: as it was| ' + _KEEP_VALUE + r')?)*)(?= for "|\.$|$)')
KEEP_CLAUSE_RE = re.compile(r'^(?P<label>.+?) had to stay(?: (?P<value>as it was|' + _KEEP_VALUE + r'))?$')
KEEP_THING_RE = re.compile(r'^(?P<thing>[a-z]+) (?P<value>' + _KEEP_VALUE + r') had to stay$')
_ADDRESS_NOUNS = {"mailbox_id": "mailbox", "group_id": "group", "list_id": "list", "channel": "channel", "channel_id": "channel",
                  "channel_name": "channel", "project": "project", "collection": "collection", "campaign_id": "campaign",
                  "spreadsheet_id": "spreadsheet", "worksheet_id": "worksheet", "worksheet": "worksheet", "calendar_id": "calendar",
                  "assignee_email": "assignee", "sender_type": "sender type", "author_type": "author type"}


def protected_fields(item: Item, expected_values: bool) -> list[dict]:
    """What a native keep check protects, each as {label, value (shown text, None when hidden), keys (the address keys the
    tail stands for), thing (True for a "has_<thing>" check: the thing itself had to stay)}: the cells of a
    `cell_contains`, a `value` with its field / column / property, a status or phone, the thing of a has_<thing> check,
    or (without content) the non-identity address (a mailbox, a group). Empty when the check protects the whole record."""
    if item.origin != "native" or item.polarity != "keep":
        return []
    from ..record.native import OBJECT_ID_KEYS, SCHEMA_KEYS

    out: list[dict] = []
    negated = set(item.semantics.get("negated_keys") or ())
    base, thing = noun_parts(item)
    if thing:
        thing_keys = [k for k in item.address if k.split("#")[0] in (thing, f"{thing}_name", f"{thing}_id", f"{thing}s")]
        if len(thing_keys) == 1 and not item.content:
            shown = show(item.address[thing_keys[0]], expected_values)
            if shown is not None and shown != common.UNSEEN:
                return [{"label": thing, "value": shown, "keys": [thing_keys[0]], "thing": True}]
            return []
    qualifier = next((k for k in item.address if k.split("#")[0] in QUALIFIER_ADDRESS_KEYS), None)
    for key, value in item.content.items():
        if key in negated or (isinstance(value.raw, dict) and "unresolved" in value.raw):
            continue
        field = key.split("#")[0].split(".")[-1]
        if data_field(key) and isinstance(value.raw, dict):
            hidden = set(value.hidden_elements) if not common.visible(value, expected_values) else set()
            for cell, raw in value.raw.items():
                shown = None if str(cell) in hidden or (not common.visible(value, expected_values) and hidden == set()) \
                    else _show_value(raw, False)
                out.append({"label": str(cell), "value": shown, "keys": [], "thing": False})
            continue
        shown = show(value, expected_values, data_field(key))
        if shown is None:
            continue
        if shown == common.UNSEEN:
            shown = None
        if field in ("value", "expected_value") and qualifier is not None:
            name = show(item.address[qualifier], expected_values)
            if name is None:
                continue
            label = f"{ADDRESS_LABELS.get(qualifier.split('#')[0], DEFAULT_LABEL(qualifier.split('#')[0]))} {name}"
            out.append({"label": label, "value": shown, "keys": [qualifier], "thing": False})
        else:
            out.append({"label": DEFAULT_LABEL(field), "value": shown, "keys": [], "thing": False})
    if out or item.content:
        return out
    for key, value in item.address.items():
        field = key.split("#")[0]
        if field in OBJECT_ID_KEYS or field in SCHEMA_KEYS:
            continue
        shown = show(value, expected_values)
        if shown is None:
            continue
        label = _ADDRESS_NOUNS.get(field, DEFAULT_LABEL(field[:-3] if field.endswith("_id") else field))
        out.append({"label": label, "value": None if shown == common.UNSEEN else shown, "keys": [key], "thing": False})
    return out


def statement_noun(item: Item, expected_values: bool) -> str:
    """The noun a statement uses for the item: the base noun for a keep check whose tail names the thing it protects
    ("changed a gmail message "M"; its label "UNREAD" had to stay"), else `noun`."""
    if common.item_kind(item) == "changed" and any(f["thing"] for f in protected_fields(item, expected_values)):
        return noun_parts(item)[0]
    return noun(item)


def keep_tail(fields: list[dict], *, many: bool = False) -> str:
    """The tail of a "changed" line from `protected_fields`: "; its FX Rate had to stay "1.08"", "; its field "stage" had to
    stay "Prospecting" and its field "amount" had to stay 5", "; its label "UNREAD" had to stay", "; its mailbox had to
    stay "mb_general""; a hidden value: "; its FX Rate had to stay as it was". Without fields, the whole-record tail."""
    if not fields:
        return KEEP_TAIL_WHOLE_MANY if many else KEEP_TAIL_WHOLE
    own = "their" if many else "its"
    clauses = []
    for f in fields:
        if f["thing"]:
            clauses.append(f"{own} {f['label']} {f['value']} had to stay")
        elif f["value"] is None:
            clauses.append(f"{own} {f['label']} had to stay as it was")
        else:
            clauses.append(f"{own} {f['label']} had to stay {f['value']}")
    return "; " + " and ".join(clauses)


def keep_tail_many(text: str) -> str:
    """A collapsed "changed" line's tail in the plural ("; their stage had to stay …")."""
    m = KEEP_TAIL_RE.search(text)
    if m is None:
        return text.replace(KEEP_TAIL_WHOLE, KEEP_TAIL_WHOLE_MANY)
    tail = m.group(0).replace("; its ", "; their ", 1).replace(" and its ", " and their ")
    return text[:m.start()] + tail + text[m.end():]


def keep_tail_clauses(line: str) -> list[tuple[str, str | None, bool]] | None:
    """(label, value or None for "as it was", thing) per clause of a line's keep tail; None when the line has none."""
    m = KEEP_TAIL_RE.search(line)
    if m is None:
        return None
    out = []
    for clause in re.split(r" and (?:its|their) ", m.group("clauses")):
        t = KEEP_THING_RE.match(clause)
        if t:
            out.append((t.group("thing"), t.group("value"), True))
            continue
        c = KEEP_CLAUSE_RE.match(clause)
        if c is None:
            return None
        value = c.group("value")
        out.append((c.group("label"), None if value in (None, "as it was") else value, False))
    return out


# engine-fix 7 (judge-final3 71): a negative attribute check the object fails from the start (a tag the subscriber has, a
# status the conversation is in) asks for a removal, never "is required with tag not "warm-lead""
REMOVAL_MARKERS = (" had to be removed; ", " must not be there; ", " must not have ")


def negative_attribute_fields(item: Item) -> list[str]:
    """The content keys a negative attribute check forbids on an existing object (a has_<thing> check, a status of a
    record), when the probe read the check as requiring an effect; empty for a send or a new object."""
    if item.origin != "native" or not item.semantics.get("effect_required") or "_not_" not in item.type:
        return []
    if item.effect.kind not in ("has", "exist", "field", "update", "other") or item.probe.identity_present is False:
        return []
    keys = [k for k in (item.semantics.get("forbidden_content_keys") or item.probe.failing_clauses or ()) if k in item.content]
    return keys


def negative_attribute_line(item: Item, expected_values: bool) -> str | None:
    """"Not done: tag "warm-lead" on mailchimp subscriber for "x" in list "aud_main" had to be removed; it was there at the
    start and still is." / "Not done: a reamaze conversation on conversation "rm_906" must not have status "archived"; it
    had that status at the start and still has it."; with the attribute absent at the start: "the attempt added it and it
    is still there" / "the attempt gave it that status and it still has it"; unknown start: "it is there now"."""
    keys = negative_attribute_fields(item)
    if not keys:
        return None
    base, thing = noun_parts(item)
    initial = item.probe.initial_result
    candidate = _candidate(item, expected_values)
    if thing and len(keys) == 1 and keys[0].split("#")[0] == thing:
        shown = show(item.content[keys[0]], expected_values)
        if shown is None or shown == common.UNSEEN:
            return None
        addr = address_phrase(item, expected_values)
        where = f" {addr}" if addr else ""
        if initial is False:
            return f"Not done: {thing} {shown} on {base}{where}{candidate} had to be removed; it was there at the start and still is."
        if initial is True:
            return f"Not done: {thing} {shown} on {base}{where}{candidate} must not be there; the attempt added it and it is still there."
        return f"Not done: {thing} {shown} on {base}{where}{candidate} must not be there; it is there now."
    attr = content_phrase(item, expected_values, keys=set(keys), affirm=set(keys))
    if not attr or common.UNSEEN in attr:
        return None
    attr = attr[len("with "):] if attr.startswith("with ") else attr
    word = attr.split(" ")[0]
    n, addr = noun(item), address_phrase(item, expected_values)
    where = f" {addr}" if addr else ""
    head = f"Not done: {article(n)} {n}{where}{candidate} must not have {attr}"
    if initial is False:
        return f"{head}; it had that {word} at the start and still has it."
    if initial is True:
        return f"{head}; the attempt gave it that {word} and it still has it."
    return f"{head}; it has that {word} now."


def thing_required_line(item: Item, expected_values: bool) -> str | None:
    """engine-fix (D79 absent lines): a native "<object> has <thing>" check whose thing is named by an address value
    (gmail_message_has_label: message_id, label) and has no content: "Not done: label "L" on gmail message "M" is
    required." (never "no gmail message label message "M" label "L" exists", which reads as the label not existing)."""
    from ..record.native import OBJECT_ID_KEYS

    base, thing = noun_parts(item)
    if item.origin != "native" or thing is None or f"has_{thing}" not in item.type or item.content or item.cardinality \
            or item.semantics.get("effect_required"):
        return None  # a "missing_<thing>" check asks for the thing's absence: never said as required
    thing_keys = [k for k in item.address if k.split("#")[0] in (thing, f"{thing}_name", f"{thing}_id", f"{thing}s")]
    ids = [k for k in item.address if k.split("#")[0] in OBJECT_ID_KEYS and k not in thing_keys]
    if len(thing_keys) != 1 or not ids or set(item.address) != set(thing_keys) | set(ids):
        return None
    shown_thing = show(item.address[thing_keys[0]], expected_values)
    shown_ids = [show(item.address[k], expected_values) for k in ids]
    if shown_thing is None or any(x is None for x in shown_ids) or common.UNSEEN in [shown_thing, *shown_ids]:
        return None
    return f"Not done: {thing} {shown_thing} on {base} {' and '.join(shown_ids)} is required{_candidate(item, expected_values)}."


def noun(item: Item) -> str:
    base, thing = noun_parts(item)
    # "<object> has <thing>" checks are about the thing (a message on a ticket, a tag on a contact).
    return f"{base} {thing}" if thing else base


def plural(n: str) -> str:
    """The plural of an object noun. engine-fix 9 (pattern 3): "opportunity" -> "opportunities", "reply" -> "replies",
    "process" -> "processes", "match" -> "matches", "box" -> "boxes"; a noun ending in a plain "s" ("records", "sms") is
    left as it is; "day" -> "days"."""
    if n.endswith(("ss", "x", "z", "ch", "sh")):
        return n + "es"
    if n.endswith("s"):
        return n
    if n.endswith("y") and len(n) > 1 and n[-2] not in "aeiou":
        return n[:-1] + "ies"
    return n + "s"


def _param_label(key: str) -> str | None:
    """A plain label for a grader parameter key inside a compound value ("summary_contains" -> "summary
    containing", "space_id" -> "space id", "applicationId" -> "application id", "commentText_contains" ->
    "comment text containing"); None for a key that is task data (a column name such as "Employee ID")."""
    label = CONTENT_LABELS.get(key)
    if label:
        return label.removeprefix("with ")
    label = ADDRESS_LABELS.get(key)
    if label:  # "in project" would read "with in project" inside a compound value: the plain key instead
        return key.replace("_", " ") if label.split()[0] in ("in", "to", "on", "for", "from", "of", "at", "with") else label
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", key).replace("_", " ").lower()  # camelCase and snake_case alike
    if key.endswith("_contains") or key.endswith("Contains"):
        return words[: -len(" contains")] + " containing"
    if ("_" in key or re.search(r"[a-z][A-Z]", key)) and " " not in key:
        return words
    return None


def _show_value(v: Any, hidden: bool) -> str:
    if hidden:
        return common.UNSEEN
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, dict):  # a nested mapping (an action's fields): its own keys, never a Python repr
        return "(" + _show_dict(v, set()) + ")"
    if isinstance(v, list):
        return ", ".join(_show_value(x, False) for x in v)
    return str(v)


# engine-fix 3 (judge-final 57): the keys of a row's cells are the sheet's own headers ("Serial_Number"), task data
# quoted as the world has them, never relabelled as a grader parameter ("serial number")
DATA_FIELDS = {"cells", "cell_contains", "after_cells", "before_cells"}


def data_field(key: str) -> bool:
    return key.split("#")[0].split(".")[-1] in DATA_FIELDS


def _show_dict(raw: dict, hidden: set[str], data: bool = False) -> str:
    out = []
    for k, v in raw.items():
        shown = _show_value(v, k in hidden)
        label = None if data else _param_label(str(k))
        out.append(f"{label} {shown}" if label else f"{k} = {shown}")
    return "; ".join(out)


def show(value: Value, expected_values: bool, data: bool = False) -> str | None:
    """The value as text, or None when the engine does not know it (an unresolved reference). A
    mapping shows each key through the parameter-key labels (never a raw `name_contains = …`)."""
    raw = value.raw
    if isinstance(raw, dict) and "unresolved" in raw:
        return None
    if not common.visible(value, expected_values):
        # a compound value is shown element by element: only the elements the attempt neither
        # observed nor wrote are hidden (a list mixing seen and written elements has none)
        hidden = set(value.hidden_elements)
        if isinstance(raw, list):
            # engine-fix-v11 (14 Nov, judged misleading): shown elements first, the hidden ones after "and", so a
            # placeholder never reads as an apposition of a shown value ("a value this attempt never observed, "job"")
            shown_el = [json.dumps(x) if isinstance(x, str) else str(x) for i, x in enumerate(raw) if str(i) not in hidden]
            n_hidden = sum(1 for i in range(len(raw)) if str(i) in hidden)
            if not shown_el:
                return ", ".join([common.UNSEEN] * n_hidden)
            return ", ".join(shown_el) + (" and " + ", ".join([common.UNSEEN] * n_hidden) if n_hidden else "")
        if isinstance(raw, dict):
            return _show_dict(raw, hidden, data)
        return common.UNSEEN
    if isinstance(raw, str):
        if raw.startswith("[") and raw.endswith("]"):
            try:  # a compound identity stored as JSON text (a channel and a timestamp): its parts, not the JSON
                parts = json.loads(raw)
                if isinstance(parts, list) and parts and all(not isinstance(p, (dict, list)) for p in parts):
                    return ", ".join(json.dumps(p) if isinstance(p, str) else str(p) for p in parts)
            except ValueError:
                pass
        return json.dumps(raw)
    if isinstance(raw, bool) or raw is None:
        return json.dumps(raw)
    if isinstance(raw, (int, float)):
        return str(raw)
    if isinstance(raw, list):
        return ", ".join(json.dumps(x) if isinstance(x, str) else str(x) for x in raw)
    if isinstance(raw, dict):
        return _show_dict(raw, set(), data)
    return str(raw)




def address_phrase(item: Item, expected_values: bool, keys: set[str] | None = None, labels: bool = True) -> str:
    """Address parts as "to …", "in channel …"; several conjuncts on one field (`recipients`,
    `recipients#1`: one message to both the employee and the manager) read as one part, "to A and B".
    With `keys`, only those address keys; `labels` adds a row's name after its number."""
    parts: dict[str, tuple[str, list[str]]] = {}  # field -> (label, shown values)
    last = noun(item).split()[-1]
    for key, value in item.address.items():
        if keys is not None and key not in keys:
            continue
        shown = show(value, expected_values, data_field(key))
        if shown is None:
            continue
        field = key.split("#")[0]
        base = field.split(".")[-1]
        label = ADDRESS_LABELS.get(base, DEFAULT_LABEL(base))
        words = label.split()
        if not parts and words and words[0] == last:  # "meeting meeting_id" -> drop the adjacent repeated word
            label = " ".join(words[1:])
        row_label = (item.semantics.get("address_labels") or {}).get(key) if labels else None
        if row_label and (expected_values or row_label.get("observed") or row_label.get("written")):
            shown = f"{shown} ({json.dumps(str(row_label['text']))})"  # row 5 ("Sterling & Associates")
        elif row_label:
            shown = f"{shown} (named by {common.UNSEEN})"  # the row's name is a value the attempt never saw
        values = parts.setdefault(field, (label, []))[1]
        if shown not in values:
            values.append(shown)
    return " ".join(f"{label} {' and '.join(values)}".strip() for label, values in parts.values())


def produced_nothing(item: Item) -> bool:
    """Whether a "no X exists, but none with" case is better said as "the attempt sent/created no X
    <address>": the only objects with that identity were there before the attempt, and the address
    names where a new object goes (a channel, a recipient, a project), not an existing record."""
    from ..record.native import OBJECT_ID_KEYS

    return (item.origin == "native" and item.probe.identity_added is False and bool(item.address)
            and not any(k.split("#")[0] in OBJECT_ID_KEYS for k in item.address)
            and item.effect.kind in ("send", "exist", "create", "post"))


def _negate_label(label: str) -> str:
    if "not containing" in label:
        return label
    if "containing" in label:
        return label.replace("containing", "not containing")
    if label.startswith("with ") or label.startswith("at "):
        return label + " not"
    return "not " + label


def _affirm_label(label: str) -> str:
    return label.replace("not containing", "containing").removeprefix("not ").removesuffix(" not")


# engine-fix 9 (judge-final5 29, 127): the content keys that name no field of the object ("text", "value"): the field a native
# check of the shape `<app>_<object>_<field>_contains` / `_equals` reads comes from the type name instead
GENERIC_CONTENT_KEYS = frozenset({"text", "text_contains", "text_not_contains", "value", "value_contains", "contains",
                                  "expected_value"})
# predicate words that are a parameter naming the field (salesforce_field_contains: `field`), or the text itself
FIELD_PARAM_WORDS = frozenset({"field", "property", "cell", "row_cell", "column", "text", "count"})
_FIELD_PREDICATE = re.compile(r"^(?P<field>[a-z]+(?:_[a-z]+)*?)_(?P<mode>not_contains|contains|not_equals|equals)$")


def checked_field(type_name: str) -> tuple[str, str] | None:
    """engine-fix 9 (pattern 1): the object field a native `<app>_<object>_<field>_(not_)contains / _(not_)equals` check
    reads, with its mode: ("description", "contains") for docusign_workspace_description_contains, ("details", "contains")
    for zendesk_organization_details_contains, ("body", "not_contains") for gmail_draft_body_not_contains; None when the
    predicate has no field word, or the word is a parameter that names the field (salesforce_field_contains)."""
    from ..record.native import split_type

    predicate = split_type(type_name)[2]
    if "_with_" in predicate:  # gmail_message_sent_to_with_body_contains: the field follows "with"
        predicate = predicate.rsplit("_with_", 1)[1]
    m = _FIELD_PREDICATE.match(predicate)
    if m is None or m.group("field") in FIELD_PARAM_WORDS or m.group("field") == "not":
        return None  # "not" is no field: google_sheets_cell_not_equals (object "cell", predicate "not_equals")
    return m.group("field").replace("_", " "), m.group("mode")


def _field_label(item: Item, base: str) -> str | None:
    """engine-fix 9 (pattern 1): "with description containing" for a generic key ("text") of a native check whose type
    names the field it reads; None for every other key (a slack message's `text` is the message text itself)."""
    if item.origin != "native" or item.type == "tool_error" or base.split(".")[-1] not in GENERIC_CONTENT_KEYS:
        return None
    found = checked_field(item.type)
    if found is None:
        return None
    field, mode = found
    if mode == "contains":
        return f"with {field} containing"
    if mode == "not_contains":
        return f"with {field} not containing"
    return f"with {field}" + (" not" if mode == "not_equals" else "")


def _label(item: Item, key: str, want: bool | None) -> str:
    """The label of a content key; `want` True forces the negative form, False the positive form
    (a `*_not_contains` key read as what the offending object did contain), None keeps the key's own."""
    base = key.split("#")[0]
    field = base.split(".")[-1]
    label = _field_label(item, base) or CONTENT_LABELS.get(base, CONTENT_LABELS.get(field))
    if label is None:
        if field.endswith("_contains"):
            label = "with " + DEFAULT_LABEL(field[: -len("_contains")]) + " containing"  # "with reason containing"
        else:
            label = "with " + DEFAULT_LABEL(field)  # an unlisted field name: "with owner id"
    if base.startswith("after_cells."):
        label = f"with cell {json.dumps(base[len('after_cells.'):])} ="
    if key in (item.semantics.get("containing_keys") or ()) and "containing" not in label:
        label = label[:-2] + " containing" if label.endswith(" =") else label + " containing"
    if want is True:
        label = _negate_label(label)
    elif want is False:
        label = _affirm_label(label)
    words = label.split()
    if words and words[0] == noun(item).split()[-1]:  # "label label_name" -> drop the repeated noun word
        label = " ".join(words[1:])
    return label


def _hidden_negated(label: str) -> str:
    """A negated term whose value is hidden: no placeholder in a negated position ("= not a value this
    attempt never observed"); the line says what the object lacks instead."""
    if label.endswith(" not containing"):
        # engine-fix-v11 (14 Nov, judged false twice): "with body not containing" + "lacking" read "not lacking", the
        # opposite of the rule; a label that is already negative keeps its sense
        return label[: -len(" not containing")] + " without a phrase the task states"
    if label.endswith(" containing"):
        return label[: -len(" containing")] + " lacking a phrase the task states"
    if label.endswith(" ="):
        return label[:-2] + " lacking the value the task states"
    return label + " not the value the task states"


def _set_phrase(label: str, positives: list[str], excluded: list[str], hidden_excluded: bool, scope: str) -> str:
    """Terms that must appear together within one unit of a text (a line, a paragraph, the whole text),
    with the terms that must be absent from that same unit."""
    unit = {"line": ("on one line", "on that line"), "block": ("in one paragraph", "in that paragraph")}.get(scope, ("", ""))
    text = f"{label} {' and '.join(positives)}".strip()
    if len(positives) > 1 and unit[0]:
        text += " " + unit[0]
    if excluded or hidden_excluded:
        names = ", ".join(excluded)
        if hidden_excluded:
            names = (names + " or " if names else "") + "a phrase the task states"
        text += f", without {names}" + (f" {unit[1]}" if unit[1] else "")
    return text


def _branch(branches: dict, together: dict, key: str) -> Any:
    """The branch of an `any` a conjunct belongs to; an item compiled without branch numbers reads each
    "on one line" set as one branch and every other conjunct as its own."""
    if key in branches:
        return branches[key]
    return ("set", together[key]) if together.get(key) is not None else key


def content_phrase(item: Item, expected_values: bool, keys: set[str] | None = None, negate: set[str] = frozenset(),
                   affirm: set[str] = frozenset()) -> str:
    """Content conjuncts as "with …, with …". The branches of one `any` are alternatives, each rendered
    whole ("(with text "A" and "B" or with text "C")": a branch that needs A and B is never "A or B");
    terms that must appear together on one line say so, with the terms that must be absent from that
    line; negated conjuncts (manifest `not`/`ne`, or `negate`) say "not"; a negated term whose value is
    hidden says what the object lacks (no placeholder in a negated position); `affirm` forces the
    positive reading of a key (what an offending object did contain)."""
    negated_keys = set(item.semantics.get("negated_keys") or ())
    excluded_keys = set(item.semantics.get("excluded_keys") or ())
    groups: dict[Any, Any] = item.semantics.get("alternatives") or {}
    branches: dict[Any, Any] = item.semantics.get("branches") or {}
    together: dict[Any, Any] = item.semantics.get("together") or {}
    scopes: dict[Any, Any] = item.semantics.get("together_scope") or {}
    parts: list[dict] = []
    for key, value in item.content.items():
        base = key.split("#")[0]
        if keys is not None and base not in keys and key not in keys:
            continue
        shown = show(value, expected_values, data_field(key))
        if shown is None:
            continue
        negated = (base in negate or key in negate or key in negated_keys) and not (base in affirm or key in affirm)
        if negated and value.raw == "" and key in negated_keys:
            continue  # engine-fix 4 (judge 1d4e581 97): "not empty" ("spreadsheet id not \"\"") says nothing a reader can use
        parts.append({"group": groups.get(key), "branch": _branch(branches, together, key), "tog": together.get(key), "key": key,
                      "label": _label(item, key, False if (base in affirm or key in affirm) else None),
                      "neglabel": _label(item, key, True), "shown": shown, "negated": negated,
                      "excluded": key in excluded_keys and key in negated_keys, "hidden": shown == common.UNSEEN})

    def set_text(members: list[dict]) -> str:
        scope = scopes.get(str(members[0]["tog"]), "line")
        positives = list(dict.fromkeys(m["shown"] for m in members if not m["excluded"] and not m["negated"]))
        negatives = list(dict.fromkeys(m["shown"] for m in members if not m["excluded"] and m["negated"]))
        excluded = list(dict.fromkeys(m["shown"] for m in members if m["excluded"] and not m["hidden"]))
        hidden_excluded = any(m["excluded"] and m["hidden"] for m in members)
        label = members[0]["label"]
        if negatives and not positives:  # a negated set: not (A and B on one line)
            return _set_phrase(members[0]["neglabel"], negatives, [], False, scope)
        text = _set_phrase(label, positives, excluded, hidden_excluded, scope)
        for value in negatives:
            text += f", {members[0]['neglabel']} {value}"
        return text

    def conjunction(members: list[dict], joiner: str) -> str:
        """Plain terms of one scope (top level or one branch): same-label positives merge into one phrase."""
        pieces: list[str] = []
        by_label: dict[str, list[str]] = {}
        done_sets: set[Any] = set()
        for m in members:
            if m["tog"] is not None:
                if m["tog"] in done_sets:
                    continue
                done_sets.add(m["tog"])
                pieces.append(set_text([x for x in members if x["tog"] == m["tog"]]))
                continue
            if m["negated"] and m["hidden"]:
                text = _hidden_negated(m["label"])
            elif m["negated"]:
                text = f"{m['neglabel']} {m['shown']}".strip()
            else:
                label = m["label"]
                if label in by_label:
                    if m["shown"] not in by_label[label]:
                        by_label[label].append(m["shown"])
                    continue
                by_label[label] = [m["shown"]]
                pieces.append(("__label__", label))  # placeholder, filled below
                continue
            if text not in pieces:
                pieces.append(text)
        out = []
        for piece in pieces:
            if isinstance(piece, tuple):
                label = piece[1]
                out.append(f"{label} {', '.join(by_label[label])}".strip())
            else:
                out.append(piece)
        return joiner.join(out)

    from ..record.probe import group_parents

    top = [m for m in parts if m["group"] is None]
    group_members: dict[Any, list[dict]] = {}
    for m in parts:
        if m["group"] is not None:
            group_members.setdefault(m["group"], []).append(m)
    parents = group_parents({g: [str(x["branch"]) for x in ms] for g, ms in group_members.items()})

    matched: dict[str, list[str]] = item.semantics.get("matched_branches") or {}

    def render_group(g: Any) -> str:
        """The alternatives of one `any`, each branch a conjunction of its own terms and of the `any`s nested
        in it (an `any` inside a branch is a requirement of that branch alone, never of the whole line). A
        located harm names only the branches the offending object matched, when the Record found them."""
        members = group_members[g]
        children = {h: parent[1] for h, parent in parents.items() if parent is not None and parent[0] == g}
        order = list(dict.fromkeys([str(x["branch"]) for x in members] + list(children.values())))
        if str(g) in matched:
            kept = [b for b in order if b in matched[str(g)]]
            order = kept or order
        texts = []
        for b in order:
            own = conjunction([x for x in members if str(x["branch"]) == b], " and ")
            inner = [render_group(h) for h, prefix in children.items() if prefix == b]
            texts.append(" and ".join(p for p in [own, *inner] if p))
        texts = list(dict.fromkeys(t for t in texts if t))
        labels = {x["label"] for x in members}
        # "with text A or B or C" only when every branch is one plain term of one label and nothing is nested
        simple = (all(x["tog"] is None and not x["negated"] for x in members) and len(labels) == 1
                  and len(order) == len(members) and not children)
        if len(texts) == 1:
            return texts[0]
        if simple:
            label = members[0]["label"]
            return f"{label} {' or '.join(t.removeprefix(label).strip() for t in texts)}".strip()
        return "(" + " or ".join(texts) + ")"

    rendered = [render_group(g) for g in group_members if parents.get(g) is None]
    plain_text = conjunction(top, ", ")
    # keep the item's own order: plain conjuncts first when they come first
    first_group = next((i for i, m in enumerate(parts) if m["group"] is not None), len(parts))
    first_plain = next((i for i, m in enumerate(parts) if m["group"] is None), len(parts))
    pieces = [plain_text, *rendered] if first_plain <= first_group else [*rendered, plain_text]
    return ", ".join(p for p in pieces if p)


def candidate_tied(item: Item) -> bool:
    """The requirement's own address or content identifies the candidate: its key, label, an identity element
    or one of its member fields is one of the line's values (a whole value, or an element of a list)."""
    if "candidate_tied" in item.semantics:  # a written view keeps the tie of the requirement's own values
        return bool(item.semantics["candidate_tied"])
    candidate = item.candidate or {}
    names = {str(candidate.get("key", "")), str(candidate.get("label", ""))}
    names |= {str(x) for x in candidate.get("identity") or [] if not isinstance(x, (dict, list))}
    names |= {str(x) for x in candidate.get("fields") or []}
    names = {n.strip().lower() for n in names if n.strip()}  # whole-value equality: length adds no safety
    values = []
    for value in [*item.address.values(), *item.content.values()]:
        raw = value.raw
        for element in (raw if isinstance(raw, list) else [raw]):
            if not isinstance(element, (dict, list)):
                values.append(str(element).strip().lower())
    return any(v in names for v in values)


def internal_candidate(item: Item) -> bool:
    """engine-fix (D79): a candidate of the task-request population ("public.request@1"): its label is the request's own
    key ("support08-public-case"), an identifier no task text or tool result shows, so it is never appended."""
    identity = (item.candidate or {}).get("identity") or []
    return bool(identity) and isinstance(identity[0], str) and identity[0].split("@")[0].endswith("request")


def _candidate(item: Item, expected_values: bool = True) -> str:
    if not item.candidate:
        return ""
    if internal_candidate(item):
        return ""
    if not (expected_values or item.candidate.get("observed", True)):
        return ""
    if not candidate_tied(item):
        return ""  # a label the line's own values do not identify would read as a condition of the line
    if item.polarity == "require" and "candidate_tied" not in item.semantics and not candidate_named(item):
        # engine-fix 3 (judge-final 103): a requirement's candidate tied only through another of its fields (the
        # expense row whose Category the body names, labelled by its Date "2026-01-25") would append a value none of
        # the line's own values supports; the tail is said only when the line's values name the candidate itself
        return ""
    label = str(item.candidate["label"])
    if label.startswith("[") and label.endswith("]"):
        try:
            parts = json.loads(label)
            if isinstance(parts, list) and parts and all(not isinstance(p, (dict, list)) for p in parts):
                return " for " + ", ".join(json.dumps(p) if isinstance(p, str) else str(p) for p in parts)
        except ValueError:
            pass
    return f" for {json.dumps(label)}"


def _residual(item: Item) -> str:
    from .. import why

    has = _has_residual(item)
    text = " (plus a further condition the task states)" if has else ""
    replaced = why.goal_suffix(item, has)  # N1a: the stated conditions replace (or extend) the marker
    if replaced is not None:
        text = replaced
    text += why.rule_suffix(item)  # N1b: when the rule applies to this candidate
    if item.semantics.get("alternatives_allowed") and item.polarity == "require":
        text += " (or another effect the task allows)"  # the check accepts an effect from another source too
    return text


def _effect_residual(semantics: dict) -> list:
    return list(semantics.get("residual") or []) + list(semantics.get("condition_effect_residual") or [])


def _has_residual(item: Item) -> bool:
    semantics = item.semantics
    # engine-fix 6 (judge-final2 33): the grader's check is complete as it states it (cells Order Number = "4511";
    # Action = "Draft"); the condition its manifest partner could not express is no condition of that check, so no
    # "(plus a further condition the task states)" on a native line
    manifest = (semantics.get("manifest") or {}) if item.origin != "native" else {}
    unresolved = semantics.get("unresolved") or manifest.get("unresolved") or {}
    return bool(_effect_residual(semantics) or _effect_residual(manifest) or unresolved.get("address")
                or unresolved.get("content"))


def contradictory_fields(item: Item) -> list[str]:
    """Fields whose content both asks for a value and negates it: platform = not "LinkedIn", "Twitter", …,
    platform = "Twitter" or "X", platform = "LinkedIn". The compiled predicate was a disjunction of cases (one
    platform per kind of post, plus "none of these"), and a plain line joins its clauses with "and": read that
    way it describes no possible object. It holds for any comparison (equality, membership, mentions). Terms that
    must (not) appear together on one line are left out: "not A and B on one line" does not negate A."""
    negated = set(item.semantics.get("negated_keys") or ())
    together = item.semantics.get("together") or {}
    asked: dict[str, set[str]] = {}
    denied: dict[str, set[str]] = {}
    for key, value in item.content.items():
        if key in together:
            continue
        side = denied if key in negated else asked
        side.setdefault(key.split("#")[0], set()).add(json.dumps(value.raw, sort_keys=True, default=str))
    return sorted(f for f in asked if asked[f] & denied.get(f, set()))


def _tool_at(item: Item, record: Record) -> str:
    call = common.located_call(record, item)
    return call.tool if call else ""


def _resolved_address_values(item: Item) -> list[Any]:
    return [v.raw for v in item.address.values() if not (isinstance(v.raw, dict) and "unresolved" in v.raw)]


def _raw_forms(values) -> set[str]:
    """Normalised forms (3+ characters) of the raw values; unresolved references contribute none."""
    from ..record.observed import value_forms

    out = set()
    for value in values:
        raw = value.raw
        if isinstance(raw, dict) and "unresolved" in raw:
            continue
        out.update(f for f in value_forms(raw) if f and len(f) >= 3)
    return out


def address_required_elsewhere(item: Item, record: Record) -> bool:
    """Whether some requirement of the same app names every address value of this item (then a line that
    forbids an effect at this address without stating its full condition could read as forbidding the
    required action: R2, say nothing). A list-valued address (`id in [...]`) is covered when the requirement
    names any of its elements."""
    scalars, lists = set(), []
    for value in item.address.values():
        raw = value.raw
        if isinstance(raw, dict) and "unresolved" in raw:
            continue
        if isinstance(raw, list):
            lists.append(_raw_forms([Value(raw=e, observed=True) for e in raw]))
        else:
            scalars |= _raw_forms([value])
    if not scalars and not any(lists):
        return True  # nothing to name
    for o in record.items:
        if o.id == item.id or o.polarity != "require" or o.type == "tool_error" or o.effect.app != item.effect.app:
            continue
        theirs = _raw_forms(o.address.values())
        if scalars <= theirs and all(lst & theirs for lst in lists if lst):
            return True
    return False


def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _arg_leaves(node: Any, path: tuple = (), depth: int = 0):
    if isinstance(node, str) and depth < 5 and node[:1] in ("{", "[") and path:
        # engine-fix 8 (judge-final4 94): a mapping argument given as JSON text (`cells: "{\"Status\": \"Done\"}"`) holds
        # the fields the call wrote
        try:
            parsed = json.loads(node)
        except ValueError:
            parsed = None
        if isinstance(parsed, (dict, list)):
            yield from _arg_leaves(parsed, path, depth + 1)
            return
    if isinstance(node, dict) and depth < 5:
        for k, v in node.items():
            yield from _arg_leaves(v, path + (str(k),), depth + 1)
    elif isinstance(node, list) and depth < 5:
        for v in node:
            yield from _arg_leaves(v, path, depth + 1)
    elif path:
        yield path[-1], node


def content_field(key: str) -> str:
    """The field a content key is about: `after_cells.Status` -> "Status", `record.status` -> "status",
    `text_contains` -> "text"."""
    base = key.split("#")[0]
    if base.startswith("after_cells."):
        return base[len("after_cells."):]
    field = base.split(".")[-1]
    for suffix in ("_not_contains", "_contains"):
        if field.endswith(suffix):
            return field[: -len(suffix)]
    return field


def written_field(item: Item, key: str) -> str:
    """The field a content value sits in: a bare `value` names the address's field/property/column."""
    base = key.split("#")[0].split(".")[-1]
    if base in ("value", "expected_value"):
        for k in ("field", "property", "column"):
            if k in item.address and not isinstance(item.address[k].raw, (dict, list)):
                return str(item.address[k].raw)
    return content_field(key)


def written_values(item: Item, record: Record, key: str) -> list:
    """What the located call wrote under the content key's field: its arguments' leaves of that name and the fields
    of the objects it added or changed in the recorded world (`semantics.located_proof.written`)."""
    call = common.located_call(record, item)
    field = _norm_key(written_field(item, key))
    out = []
    if call is not None:
        out += [v for k, v in _arg_leaves(call.args) if _norm_key(k) == field and v is not None and not isinstance(v, (dict, list))]
    proof = item.semantics.get("located_proof") or {}
    out += [v for k, v in proof.get("written") or [] if _norm_key(k) == field and v is not None]
    return out


def written_view(item: Item, record: Record) -> Item | None:
    """Removal round (harm, harm_collapsed): a located prohibition quotes only what the call wrote. For every
    content value: when the call wrote exactly that value, it stays; when the call wrote a value of at most 60
    characters that holds it (any case), that written value is quoted exactly instead; when it is a literal part of
    a longer written value, it is said as "containing" it; otherwise (nothing written under that field, or no
    literal match) the line is withheld."""
    negated = set(item.semantics.get("negated_keys") or ())
    content = dict(item.content)
    containing: set[str] = set()
    members: dict[str, list] = {}
    for key, value in item.content.items():
        raw = value.raw
        if key not in negated and isinstance(raw, list) and raw and all(isinstance(x, str) for x in raw):
            # engine-fix (D79 "ten Recommendation values"): a value that is one of a list (membership) is quoted as the
            # one member the call wrote, exactly; the list's other members are said apart (`also_scope`)
            written = {str(w).strip() for w in written_values(item, record, key)}
            hits = list(dict.fromkeys(x for x in raw if x.strip() in written))
            if len(hits) == 1:
                content[key] = value.model_copy(update={"raw": hits[0], "forms": (hits[0].lower(),), "hidden_elements": ()})
                members[key] = list(raw)
            continue
        if key in negated or isinstance(raw, (dict, list)) or raw is None:
            continue
        written = [str(w).strip() for w in written_values(item, record, key)]
        text = str(raw).strip()
        if not written or not text:
            return None
        if text in written:
            continue
        if not isinstance(raw, str):
            return None  # a number or boolean is quoted only when written exactly so
        short = [w for w in written if text.lower() in w.lower() and len(w) <= 60]
        if short:
            content[key] = value.model_copy(update={"raw": short[0], "forms": (short[0].lower(),)})
            continue
        if any(text in w for w in written):
            if "containing" not in _label(item, key, None):
                containing.add(key)
            continue
        return None
    address, labels = written_targets(item)
    if address is None:
        return None  # the call named the target by another value the record cannot map (harm_target_not_written)
    return item.model_copy(update={"content": content, "address": address,
                                   "semantics": {**item.semantics, "containing_keys": sorted(containing),
                                                 "candidate_tied": candidate_tied(item), "address_labels": labels,
                                                 "list_members": members}})


def written_targets(item: Item) -> tuple[dict | None, dict]:
    """Fix 1: a located prohibition quotes its target (channel, spreadsheet, recipient, …) exactly as the call wrote
    it. Per address key (`semantics.located_proof.targets`, abce.record.build.target_proof): the check's value
    spelled as written when the call wrote it; else the value the call wrote, followed by the check's (the world's)
    name in brackets, when the recorded world proves both name one object; else (the call wrote another value the
    record cannot map) None. A key the call wrote nothing for keeps the check's value."""
    targets = (item.semantics.get("located_proof") or {}).get("targets") or {}
    address = dict(item.address)
    labels = dict(item.semantics.get("address_labels") or {})
    for key, target in targets.items():
        value = item.address.get(key)
        if value is None:
            continue
        if target.get("as_written") is not None:
            shown = target["as_written"]
            if str(shown) != str(value.raw) and isinstance(value.raw, str):
                address[key] = value.model_copy(update={"raw": str(shown), "forms": (str(shown).lower(),)})
            continue
        mapped = target.get("mapped")
        if not mapped:
            return None, labels
        address[key] = value.model_copy(update={"raw": mapped["written"], "observed": True, "written": True,
                                                "forms": (str(mapped["written"]).lower(),)})
        labels[key] = {"text": str(mapped["named"]), "observed": True, "written": False, "target": True}
    return address, labels


# engine-fix (D79, the 9.3 scope issue): a located prohibition whose check forbids a combination (a case with this
# subject AND this priority; a calendar row for this link WITHOUT the batch reference; an event row with one of several
# recommendations) names what is forbidden about the object instead of forbidding the whole action:
# - value scope: "; this task forbids it with priority "Low"" — the values of the line that no requirement of the
#   same app and object needs together with the line's other (required) values; said only when some requirement names
#   at least one content value of the line and not all of them;
# - unless scope: "; this task forbids it unless its "batch_ref" cell contains "SCH-W05-2026"" — a rule-file prohibition
#   whose whole condition is the negation of one term on the written object; the body quotes what the call wrote there;
# - also scope: ", and also with cell "Recommendation" = "sponsor" or "speak"" — the other members (one spelling each)
#   of the list the quoted value belongs to, when the rule's condition does not read the written object.
# A scoped line is never merged with other harms (each keeps its own tail).
SCOPE_UNLESS = "; this task forbids it unless its "
SCOPE_ALSO = "; it also forbids a "
SCOPE_VALUE = "; what this task forbids is "


def _reads_effect(node: Any) -> bool:
    """Whether a compiled condition reads a field of the written object (`path` ["effect", …])."""
    if isinstance(node, dict):
        if node.get("kind") == "field" and list(node.get("path") or [])[:1] == ["effect"]:
            return True
        return any(_reads_effect(v) for v in node.values())
    if isinstance(node, list):
        return any(_reads_effect(v) for v in node)
    return False


def _unless_term(item: Item) -> tuple[str, str, Any] | None:
    """(content key, "contains" | "is", needed value) when the prohibition's whole condition is `not` of one term on the
    written object (a mention or an equality) and nothing else of it is left unstated; else None."""
    sem = item.semantics
    negated = list(sem.get("negated_keys") or ())
    cond = sem.get("condition")
    unresolved = sem.get("unresolved") or {}
    if (item.origin != "manifest" or len(negated) != 1 or not isinstance(cond, dict) or cond.get("op") != "not"
            or sem.get("condition_residual") or _effect_residual(sem) or unresolved.get("content") or unresolved.get("address")
            or sem.get("condition_role") != "prohibited_when"):
        return None
    arg = cond.get("arg") or {}
    op = arg.get("op")
    field = arg.get("text") if op == "mentions" else arg.get("left") if op == "eq" else None
    literal = arg.get("value") if op == "mentions" else arg.get("right") if op == "eq" else None
    key = negated[0]
    if not isinstance(field, dict) or not isinstance(literal, dict) or literal.get("kind") != "literal":
        return None
    path = list(field.get("path") or [])
    value = item.content.get(key)
    if path[:1] != ["effect"] or ".".join(path[1:]) != key.split("#")[0] or value is None or value.raw != literal.get("value"):
        return None
    if not isinstance(value.raw, str) or not value.raw.strip():
        return None
    return key, ("contains" if op == "mentions" else "is"), value.raw


def unless_scope(item: Item, view: Item, record: Record) -> tuple[Item, str] | None:
    """The unless scope: (the view with the negated field showing what the call wrote, the tail)."""
    term = _unless_term(item)
    if term is None:
        return None
    key, verb, needed = term
    written = list(dict.fromkeys(str(w).strip() for w in written_values(item, record, key) if isinstance(w, str) and str(w).strip()))
    if len(written) != 1:
        return None
    from ..record.observed import normalize

    if normalize(needed) in normalize(written[0]) or needed.lower() in written[0].lower():
        return None  # the written value holds the needed text by some reading: the line could not say it lacks it
    content = dict(view.content)
    content[key] = view.content[key].model_copy(update={"raw": written[0], "forms": (written[0].lower(),), "observed": True,
                                                        "written": True, "hidden_elements": ()})
    sem = {**view.semantics, "negated_keys": [], "excluded_keys": []}
    groups = dict(sem.get("alternatives") or {})
    # an `any` of identity terms whose every term the call wrote exactly: the object carries them all
    for g in set(groups.values()):
        keys = [k for k, x in groups.items() if x == g]
        if all(str(view.content[k].raw).strip() in [str(w).strip() for w in written_values(item, record, k)]
               for k in keys if k in view.content) and not any(k in (sem.get("containing_keys") or ()) for k in keys):
            for k in keys:
                groups.pop(k, None)
    sem["alternatives"] = groups
    sem["branches"] = {k: v for k, v in (sem.get("branches") or {}).items() if k in groups}
    field = (f'{json.dumps(key.split("#")[0][len("after_cells."):])} cell' if key.startswith("after_cells.")
             else DEFAULT_LABEL(content_field(key)))
    return view.model_copy(update={"content": content, "semantics": sem}), f"{SCOPE_UNLESS}{field} {verb} {json.dumps(needed)}"


def _needed_forms(other: Item) -> set[str]:
    """Forms of what a requirement needs: its address and its content, without what it must not contain and without the
    terms of an `any` with several branches (the goal is met without them) — the R2 reading of `forbids_required`."""
    from ..record.observed import value_forms

    def forms(raw):
        if isinstance(raw, dict):
            return {f for v in raw.values() for f in forms(v)}
        if isinstance(raw, list):
            return {f for v in raw for f in forms(v)}
        return {f for f in value_forms(raw) if f}

    avoided = set(other.semantics.get("forbidden_content_keys") or ()) | set(other.semantics.get("negated_keys") or ())
    if other.semantics.get("effect_required"):
        avoided |= set(other.probe.failing_clauses)
    groups = other.semantics.get("alternatives") or {}
    branches = other.semantics.get("branches") or {}
    group_branches: dict = {}
    for k, g in groups.items():
        group_branches.setdefault(g, set()).add(branches.get(k, k))
    out: set[str] = set()
    for v in other.address.values():
        if not (isinstance(v.raw, dict) and "unresolved" in v.raw):
            out |= forms(v.raw)
    for k, v in other.content.items():
        if k in avoided or k.split("#")[0] in avoided or (k in groups and len(group_branches[groups[k]]) > 1):
            continue
        if not (isinstance(v.raw, dict) and "unresolved" in v.raw):
            out |= forms(v.raw)
    return out


def _scalar_forms(raw: Any) -> set[str]:
    from ..record.observed import value_forms

    return {f for f in value_forms(raw) if f}


def value_scope(item: Item, view: Item, record: Record, expected_values: bool) -> str | None:
    """The value scope's tail, or None. The line's values (address and content of the check, scalar, shown) split into
    those some requirement of the same app and object needs and the rest; the requirement needing the most of them
    (at least one content value, not all) decides; the rest is what the tail names, as the body shows it."""
    negated = set(item.semantics.get("negated_keys") or ())
    if negated or item.semantics.get("alternatives"):
        return None
    keys = [("address", k) for k, v in item.address.items() if not isinstance(v.raw, (dict, list)) and v.raw is not None]
    keys += [("content", k) for k, v in item.content.items() if not isinstance(v.raw, (dict, list)) and v.raw is not None]
    if len(keys) < 2 or not any(side == "content" for side, _k in keys):
        return None
    from ..record.observed import Corpus, normalize

    call = common.located_call(record, item)
    args = normalize(json.dumps(call.args, sort_keys=True, default=str)) if call is not None else ""
    ranked: list[tuple[tuple, frozenset]] = []
    # how many requirements of the same app and object name each key: a key every requirement names (a case's
    # subject) identifies the object; a key only some name (its priority) is an attribute of it
    peers = [o for o in record.items if o.polarity == "require" and o.type != "tool_error" and _same_app(o.effect.app, item.effect.app)
             and (o.effect.object == item.effect.object or o.effect.object in ("", o.effect.app) or item.effect.object in ("", item.effect.app))]
    frequency = {k: sum(1 for o in peers if k in (o.address if side == "address" else o.content)) for side, k in keys}
    for other in record.items:
        if other.polarity != "require" or other.type == "tool_error" or other.id == item.id:
            continue
        if not _same_app(other.effect.app, item.effect.app):
            continue
        if other.effect.object != item.effect.object and other.effect.object not in ("", other.effect.app) \
                and item.effect.object not in ("", item.effect.app):
            continue
        needed = _needed_forms(other)
        covered = [(side, k) for side, k in keys
                   if _scalar_forms((item.address if side == "address" else item.content)[k].raw) <= needed]
        # engine-fix 2: a message the task requires to that recipient (a send) whose content is what is forbidden
        send_cover = (item.effect.kind == "send" and bool(covered)
                      and all(side == "content" for side, k in keys if (side, k) not in covered))
        if (not any(side == "content" for side, _k in covered) and not send_cover) or len(covered) == len(keys):
            continue
        # the requirement this very call serves ranks first: its address is in the call's arguments, or it was
        # witnessed at this call; then the one needing the most of the line's values
        address = [v.raw for v in other.address.values() if not (isinstance(v.raw, dict) and "unresolved" in v.raw)]
        in_call = bool(address) and all(any(Corpus._form_in(f, args) for f in _scalar_forms(v)) for v in address
                                         if not isinstance(v, (dict, list)))
        witnessed = (call is not None and other.location.kind == "after_turn" and other.location.call is not None
                     and (other.location.call.turn, other.location.call.index) == (call.turn, call.index))
        # a failed requirement whose probe found its object and failed exactly on the line's other values (priority)
        # is about this very object: those values are what is wrong
        rest = {k for side, k in keys if side == "content" and (side, k) not in covered}
        about = bool(rest) and other.outcome == "failed" and other.probe.identity_present is True and \
            rest <= {c.split("#")[0] for c in other.probe.failing_clauses}
        ranked.append(((in_call or witnessed, about, sum(frequency[k] for _side, k in covered), len(covered)), frozenset(covered)))
    if not ranked:
        return None
    top = max(score for score, _c in ranked)
    choices = {c for score, c in ranked if score == top}
    if len(choices) != 1:
        return None  # two requirements of equal standing need different values of the line: which is wrong is unknown
    best = next(iter(choices))
    parts = []
    affirmed = affirmed_keys(item)
    for side, k in keys:
        if (side, k) in best:
            continue
        value = (view.address if side == "address" else view.content).get(k)
        own = item.content.get(k) if side == "content" else None
        # D89 (harm_scope_object_not_required): the body quotes what the call wrote (a written value holding the
        # check's own value); what the task forbids is the check's own value, which that written value contains
        stand_in = own is not None and value is not None and value.raw != own.raw
        shown = show(own if stand_in else value, expected_values) if value is not None else None
        if shown is None or common.UNSEEN in shown:
            return None
        if side == "content":
            label = _label(view, k, False if k in affirmed else None).removeprefix("with ")
            if stand_in and "containing" not in label:
                label = label[:-2] + " containing" if label.endswith(" =") else label + " containing"
        else:
            base = k.split("#")[0].split(".")[-1]
            label = ADDRESS_LABELS.get(base, DEFAULT_LABEL(base))
            label = ("sending it " + label) if item.effect.kind == "send" else label
        parts.append(f"{label} {shown}".strip())
    # engine-fix 2 (judge v1 3): the forbidden part named as such, and the object itself said to be not what is forbidden
    # (a requirement of the same app and object needs the line's other values)
    if not parts:
        return None
    if item.effect.kind == "send" and any(side == "address" for side, k in keys if (side, k) not in best):
        # engine-fix 5 (judge 2a79e7e 136): a message sent to the forbidden recipient is the forbidden thing itself;
        # "not the gmail message itself" would vouch for its content, which no check proves
        return f"{SCOPE_VALUE}{' and '.join(parts)}"
    # engine-fix 8 (judge-final4 59): "; what this task forbids is P, not the NOUN itself" vouched for the object (the
    # announcement in the wrong channel is itself the problem); the attribute tail (engine-fix 6) names the forbidden
    # attribute and the object tail stands when the check forbids the object in that place outright
    return None


def also_scope(item: Item, view: Item, expected_values: bool, candidate: str = "") -> str | None:
    """The also scope: the other members of a membership list whose one written member the line quotes, one spelling
    each (letter case aside), when the rule's condition does not read the written object."""
    members = view.semantics.get("list_members") or {}
    if not members or _reads_effect(item.semantics.get("condition")) or not candidate.startswith(" for "):
        return None  # engine-fix 2 (judge v1 18/28): said only as the rule of that very record (its candidate)
    parts = []
    for key, listed in members.items():
        shown_value = view.content.get(key)
        if shown_value is None or not isinstance(shown_value.raw, str):
            return None
        seen = {shown_value.raw.strip().lower()}
        others = []
        for x in listed:
            if x.strip().lower() not in seen:
                seen.add(x.strip().lower())
                others.append(json.dumps(x))
        if others:
            base = key.split("#")[0]
            field = (f'{json.dumps(base[len("after_cells."):])} cell' if base.startswith("after_cells.")
                     else DEFAULT_LABEL(content_field(key)))
            parts.append(f"{field} is {' or '.join(others)}")
    if not parts:
        return None
    word = noun(view).split()[-1]
    return f"{SCOPE_ALSO.removesuffix('a ')}{article(word)} {word}{candidate} whose {' and whose '.join(parts)}"


def affirmed_keys(item: Item) -> set[str]:
    """D89: the `*_not_contains` content keys of a pure prohibition (a check that passes on the untouched world and
    forbids a message whose body holds the value): the offending object did contain the value, so its line says
    "containing", never "not containing" (that is what the body of the line describes: the object the call made)."""
    if item.polarity != "forbid" or item.semantics.get("effect_required"):
        return set()
    negated = set(item.semantics.get("negated_keys") or ())
    return {k for k in item.content if k not in negated and k.split("#")[0].split(".")[-1].endswith("_not_contains")}


def scope_forbids_required(item: Item, record: Record) -> bool:
    """D89 (harm_scope_forbids_required), the truth checker's own reading: every value the prohibition's line can name
    (its address and its non-negated scalar content) is among what one requirement of the same app needs. Such a line
    cannot tell the forbidden object from the required one, whatever its tail says: nothing is said."""
    negated = set(item.semantics.get("negated_keys") or ())
    forbids: set[str] = set()
    for k, v in {**item.address, **item.content}.items():
        if k in negated or isinstance(v.raw, (dict, list)) or v.raw is None:
            continue
        forbids |= _scalar_forms(v.raw)
    if not forbids:
        return False
    from ..checks.truth import _app_matches

    return any(forbids <= _needed_forms(o) for o in record.items
               if o.polarity == "require" and o.type != "tool_error" and _app_matches(o.effect.app, item.effect.app))


def _value_scoped(item: Item, record: Record) -> bool:
    """Whether the located prohibition's line would carry the value scope ("; what this task forbids is …")."""
    if item.location.kind != "at_call" or record_scoped(item):
        return False
    view = written_view(item, record)
    return view is not None and unless_scope(item, view, record) is None and value_scope(item, view, record, True) is not None


def requirements_needing_all(item: Item, record: Record) -> list[Item]:
    """The requirements of the item's app that need every value the prohibition's line can name (`_needed_forms`)."""
    negated = set(item.semantics.get("negated_keys") or ())
    forbids: set[str] = set()

    def flat(raw: Any) -> set[str]:  # a check's mapping parameters ({"name_contains": "TXN-5005"}) name values too
        if isinstance(raw, dict):
            return {f for x in raw.values() for f in flat(x)}
        if isinstance(raw, list):
            return {f for x in raw for f in flat(x)}
        return _scalar_forms(raw) if raw is not None else set()

    for k, v in {**item.address, **item.content}.items():
        if k in negated or (isinstance(v.raw, dict) and "unresolved" in v.raw):
            continue
        forbids |= flat(v.raw)
    if not forbids:
        return []
    from ..checks.truth import _app_matches

    return [o for o in record.items if o.polarity == "require" and o.type != "tool_error" and o.id != item.id
            and _app_matches(o.effect.app, item.effect.app) and forbids <= _needed_forms(o)]


def forbids_required_elsewhere(item: Item, record: Record) -> bool:
    """D89-2 (hard rule: never forbid what the task requires): a located prohibition whose line names no value that
    tells it from a requirement — no negated value, no record of its own, no "unless" condition — while one requirement
    of the same app (at any call) needs every value it names. "This task forbids it" would forbid that requirement."""
    if item.semantics.get("negated_keys") or record_scoped(item):
        return False
    view = written_view(item, record)
    if view is not None and unless_scope(item, view, record) is not None:
        return False
    return bool(requirements_needing_all(item, record))


STILL_REQUIRED = " still required"


def still_required_tail(item: Item, record: Record, expected_values: bool = True) -> str | None:
    """D89-2: for a prohibition whose every named value a requirement elsewhere needs, the clause that tells the two apart,
    or None (then nothing is said). Said only when every such requirement is about another kind of object (a draft, not
    a sent message): "; a gmail draft to "X" is still required", each requirement in its own words (address, content)."""
    needing = requirements_needing_all(item, record)
    if not needing or any(noun(o) == noun(item) or o.effect.app.replace("_", "") != item.effect.app.replace("_", "")
                          for o in needing):
        return None
    phrases = []
    for o in needing:
        addr, content = address_phrase(o, expected_values), content_phrase(o, expected_values)
        text = f"{article(noun(o))} {noun(o)}{(' ' + addr) if addr else ''}{(' ' + content) if content else ''}"
        if common.UNSEEN in text:
            return None
        if text not in phrases:
            phrases.append(text)
    phrases = [p for p in phrases if not any(q != p and q.startswith(p + " ") for q in phrases)]  # the fuller one says it
    return f"; {' and '.join(phrases)} {'is' if len(phrases) == 1 else 'are'}{STILL_REQUIRED}"


# engine-fix 3 (judge-final 116): a prohibition naming only recipients (a message to "d.optout@…") located at a send that
# also went to other recipients forbids the message to those recipients, never the whole message (the recording email to
# the others): the tail names the recipient as what is forbidden
RECIPIENT_KEYS = {"to", "cc", "bcc", "recipient", "recipients"}
RECIPIENT_ARG_KEYS = RECIPIENT_KEYS | {"to_email", "to_emails", "cc_emails", "bcc_emails", "to_addresses", "cc_addresses"}
RECIPIENT_SCOPE = "; this task forbids sending to th"
_EMAIL = re.compile(r"[A-Za-z0-9._%+'-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def recipient_tail(n: str, objects: int = 1, recipients: int = 1) -> str:
    # engine-fix 8 (judge-final4 59): never ", not the NOUN itself" (it vouched for the message's content)
    who = "that recipient" if recipients == 1 else "those recipients"
    return f"; this task forbids sending to {who}"


def call_recipients(call) -> list[str]:
    """Every address the send call wrote to a recipient argument (to, cc, bcc), lower-cased, in order."""
    out: list[str] = []
    for k, v in (call.args or {}).items() if isinstance(call.args, dict) else ():
        if str(k).lower() in RECIPIENT_ARG_KEYS:
            for e in _EMAIL.findall(json.dumps(v, default=str)):
                if e.lower() not in out:
                    out.append(e.lower())
    return out


def recipient_scope(item: Item, view: Item | None, record: Record) -> list[str] | None:
    """The forbidden recipients when the located prohibition of a send names nothing but recipients (no content, no
    negated value, no unresolved part) and its call also wrote to someone else; else None."""
    if item.effect.kind != "send" or item.location.kind != "at_call" or item.content or (view is not None and view.content):
        return None
    if item.semantics.get("negated_keys") or not item.address:
        return None
    forbidden = []
    for k, v in item.address.items():
        if k.split("#")[0] not in RECIPIENT_KEYS or not isinstance(v.raw, str) or not _EMAIL.fullmatch(v.raw.strip()):
            return None
        forbidden.append(v.raw.strip().lower())
    call = common.located_call(record, item)
    if call is None:
        return None
    written = call_recipients(call)
    if not set(forbidden) <= set(written) or not [e for e in written if e not in forbidden]:
        return None
    return forbidden


def written_row_cell(item: Item, record: Record) -> tuple[str, Any] | None:
    """engine-fix 4: (header, value) of the first non-empty cell the located add-row call wrote, or None."""
    call = common.located_call(record, item)
    cells = (call.args or {}).get("cells") if call is not None and isinstance(call.args, dict) else None
    if isinstance(cells, str):
        try:
            cells = json.loads(cells)
        except ValueError:
            return None
    if not isinstance(cells, dict):
        return None
    for k, v in cells.items():
        if isinstance(v, (str, int, float)) and not isinstance(v, bool) and str(v).strip():
            return str(k), v
    return None


def names_written_row(item: Item) -> bool:
    """Whether the prohibition's line names its row by the written cell: its only content term is "not empty"."""
    negated = set(item.semantics.get("negated_keys") or ())
    return any(v.raw == "" for k, v in item.content.items() if k in negated) and \
        not content_phrase(item, True, affirm=affirmed_keys(item))


def written_row_name(item: Item, record: Record) -> str | None:
    """engine-fix 4 (judge 1d4e581 97): the row a located add-row call wrote, named by its first non-empty cell as the call
    wrote it (`with cell "Case" = "DSP-004"`), for a prohibition whose only term was "not empty"."""
    cell = written_row_cell(item, record)
    if cell is None:
        return None
    k, v = cell
    return f"with cell {json.dumps(k)} = {json.dumps(v) if isinstance(v, str) else v}"


def field_check_phrase(item: Item, expected_values: bool, keys: set[str] | None = None) -> str | None:
    """engine-fix 4 (judge 1d4e581 61): a native "exists with field" check (address {"field": F}, content value V and
    additional checks) names the record that has the value: `with email "V" and phone containing "7777"`, never
    `field "email" with value "V", with additional checks ...` (read as a field to write). None for any other check."""
    if item.origin != "native" or set(item.address) != {"field"} or "value" not in item.content:
        return None
    field = item.address["field"].raw
    if not isinstance(field, str) or set(item.content) - {"value", "additional_checks"}:
        return None
    parts = []
    if keys is None or "value" in keys:
        shown = show(item.content["value"], expected_values)
        if shown is None:
            return None
        label = field[: -len("_contains")].replace("_", " ") + " containing" if field.endswith("_contains") else field.replace("_", " ")
        parts.append(f"{label} {shown}")  # the record's own field name ("email"), never a parameter label
    extra = item.content.get("additional_checks")
    if extra is not None and (keys is None or "additional_checks" in keys):
        if not isinstance(extra.raw, dict):
            return None
        shown = show(extra, expected_values)
        if shown is None:
            return None
        parts.extend(p.strip() for p in shown.split("; "))
    return ("with " + " and ".join(parts)) if parts else None


FIELD_VALUE_TAIL = "; this task forbids that "


def field_check_forbidden(item: Item, record: Record) -> str | None:
    """engine-fix 5 (judge 2a79e7e 144): for a native "not exists with field" prohibition with one additional check, the
    words of that check's field ("stage name") when a requirement of the same app and object needs the record's field
    value (the opportunity's name) and not the additional value: what is forbidden is that value on that record."""
    if item.origin != "native" or item.polarity != "forbid" or set(item.address) != {"field"} or "value" not in item.content:
        return None
    extra = item.content.get("additional_checks")
    if extra is None or not isinstance(extra.raw, dict) or len(extra.raw) != 1 or set(item.content) - {"value", "additional_checks"}:
        return None
    (key, raw), = extra.raw.items()
    if isinstance(raw, (dict, list)) or raw is None or not isinstance(key, str):
        return None
    identity, forbidden = _scalar_forms(item.content["value"].raw), _scalar_forms(raw)
    if not identity or not forbidden:
        return None
    for o in record.items:
        if o.polarity != "require" or o.type == "tool_error" or not _same_app(o.effect.app, item.effect.app):
            continue
        if o.effect.object != item.effect.object and o.effect.object not in ("", o.effect.app):
            continue
        needed = _needed_forms(o)
        if identity <= needed and not forbidden <= needed:
            return key[: -len("_contains")].replace("_", " ") if key.endswith("_contains") else key.replace("_", " ")
    return None


def harm_scoped(item: Item, record: Record, expected_values: bool) -> bool:
    """Whether the located prohibition's line carries a scope (never merged with other harms)."""
    from .. import why

    if not (why.ENABLED.get() and common.item_kind(item) == "prohibited" and item.location.kind == "at_call") or record_scoped(item):
        return False
    if forbids_required_elsewhere(item, record):
        return True  # D89-2: its own "still required" clause
    view = written_view(item, record)
    if view is None:
        return False
    candidate = _candidate(item, expected_values) if candidate_named(item) else ""
    return bool(unless_scope(item, view, record) or value_scope(item, view, record, expected_values)
                or also_scope(item, view, expected_values, candidate))


RECORD_KEYS = {"row_id", "record_id", "native_record_id", "ticket_id", "contact_id", "lead_id", "deal_id", "opportunity_id",
               "case_id", "task_id", "card_id", "issue_key", "envelope_id", "invoice_number", "employee_id", "candidate_id",
               "event_id", "meeting_id", "file_id", "id", "customer_id", "conversation_id"}
SCOPED_TAIL = "; this task forbids it for that record"
SCOPED_TAIL_MANY = "; this task forbids them for those records"


def record_scoped(item: Item) -> bool:
    """Fix 1: the prohibition is about specific records (a per-record check: its address names the record by its own
    id; a rule-file population member): the line names the record and says the task forbids it for that record,
    never the whole action."""
    return any(k.split("#")[0].split(".")[-1] in RECORD_KEYS and v.raw is not None and not isinstance(v.raw, (dict, bool))
               and not (isinstance(v.raw, list) and not v.raw) for k, v in item.address.items())


# engine-fix 6 (judge-final2 46/55/67/68/161): a prohibition whose check forbids an attribute of an object (text or body
# content, a cc, a field value, a stage name) and not the object's existence never ends "; this task forbids it/them" (that
# forbids the object, which the task may require: the rejection emails whose cc is forbidden). The tail names exactly the
# forbidden attribute(s), the check's own values, and asserts nothing else: "; this task forbids text containing "Bob Chen"
# in that slack message", "; this task forbids cc "ceo@…" on these gmail messages", "; this task forbids the text shown in
# each of these slack dms" (members with different values, or a phrase of branches). "This task forbids it" stays only when the check names no attribute
# (a message to a recipient, a row named by its written cell, a record's update). Never "not the message itself" (136).
ATTRIBUTE_ADDRESS_KEYS = {"cc", "bcc", "cc_emails", "bcc_emails", "cc_addresses", "bcc_addresses"}
QUALIFIER_ADDRESS_KEYS = {"column", "field", "property"}
_OBJECT_TAIL = re.compile(r"; this task forbids (it|them)( for th(?:at|ose) records?)?(?=\.$| for \"|; | \()")
ATTRIBUTE_TAIL_RE = re.compile(r'; this task forbids (?!it\b|them\b|sending to )(?P<attrs>.+?) (?P<prep>in|on) '
                               r'(?P<ref>that|these|those|each of these) (?P<noun>[A-Za-z0-9][A-Za-z0-9 _-]*?)(?=\.$| for "|; | \()')


IDENTITY_FIELDS = {"subject", "title", "name", "summary", "topic", "email", "id", "key", "url", "link", "sku", "number",
                   "order_number", "invoice_number", "case", "ticket", "cells"}
TEXT_MODES = {"mentions", "words", "verbatim", "contains", "icontains"}


def identity_key(item: Item, key: str, record: Record | None = None) -> bool:
    """Whether a content term names the object rather than an attribute of it: a term that is the rule's candidate (the
    event row's name), an exact value in a column that keys a row the call made (`column "Ticket ID" with value "tkt_q07"`),
    exact cells of a row the call made, or an exact (never "containing") subject, title, name, email or id. A mention
    ("text containing"), a status, a priority, a stage, a recommendation cell, a value written by an update are attributes."""
    base = key.split("#")[0]
    field = base.split(".")[-1]
    value = item.content.get(key)
    if value is None or isinstance(value.raw, list):
        return False
    if item.origin == "native" and item.effect.kind not in UPDATE_KINDS and item.polarity == "forbid" \
            and not field.endswith("_not_contains") and content_field(key) in IDENTITY_FIELDS and field not in ("cells",):
        # engine-fix 8 (judge-final4 152): a title / name / subject term of a not-exists check, exact or "containing",
        # identifies which object must not exist (the skip@skipcorp.com tickets): the check forbids the object
        return True
    if field.endswith("_contains") or field in ("contains", "content_contains", "text", "body", "body_text", "values_text"):
        return False
    if item.origin == "manifest" and (item.semantics.get("modes") or {}).get(key) in TEXT_MODES:
        return False
    label = (item.candidate or {}).get("label")
    if label is not None and not isinstance(value.raw, (dict, list)) and candidate_named(item) \
            and str(value.raw).strip().lower() == str(label).strip().lower():
        return True
    made = item.effect.kind not in ("update", "field")
    if not made and record is not None:
        # a cell or field of an updated record that the call did not write (the "Item" of the row whose Status it
        # changed) names the record; what the call wrote is the attribute
        call = common.located_call(record, item)
        if call is not None:
            want = _norm_key(written_field(item, key))
            if not any(_norm_key(k) == want for k, _v in _arg_leaves(call.args)):
                return True
    if made and field in ("value", "expected_value") and any(k.split("#")[0].split(".")[-1] in QUALIFIER_ADDRESS_KEYS for k in item.address):
        return True
    if made and (field == "cells" or base.startswith("after_cells.")) and item.origin == "native":
        return True
    return field in IDENTITY_FIELDS or field.endswith("_id")


def attribute_keys(item: Item, record: Record | None = None) -> tuple[list[str], list[str]]:
    """(content keys, address keys) of what the prohibition forbids about the object: its non-negated, stated content
    that is not the object's identity (`identity_key`; a "not empty" term and a field check's identity value aside) and
    its cc/bcc recipients. Both empty: the check forbids the object itself."""
    negated = set(item.semantics.get("negated_keys") or ())
    content = []
    for k, v in item.content.items():
        if k in negated or v.raw is None or (isinstance(v.raw, dict) and "unresolved" in v.raw):
            continue
        if isinstance(v.raw, str) and not v.raw.strip():
            continue
        if field_check_phrase(item, True) is not None and k == "value":
            continue  # a native "not exists with field" check: the value names the record (engine-fix 5, 144)
        if identity_key(item, k, record):
            continue
        content.append(k)
    address = [k for k, v in item.address.items() if k.split("#")[0].split(".")[-1] in ATTRIBUTE_ADDRESS_KEYS
               and v.raw is not None and not isinstance(v.raw, dict)]
    return content, address


def attribute_labels(item: Item, record: Record | None = None) -> list[str]:
    """The field words of the forbidden attributes ("text", "cells", "cc"), one each, in the item's order."""
    content, address = attribute_keys(item, record)
    out: list[str] = []
    for k in content:
        base = k.split("#")[0].split(".")[-1]
        raw = item.content[k].raw
        if k.split("#")[0].startswith("after_cells."):
            words = [f"cell {json.dumps(content_field(k))}"]
        elif base in ("cells", "cell_contains"):
            words = ["cells"]
        elif isinstance(raw, dict):  # the parameters of an action: each member's field ("topic")
            words = [content_field(str(k2)) .replace("_", " ") for k2 in raw]
        else:
            words = [DEFAULT_LABEL(content_field(k))]
        for word in words:
            if word not in out:
                out.append(word)
    for k in address:
        word = ADDRESS_LABELS.get(k.split("#")[0].split(".")[-1], DEFAULT_LABEL(k.split("#")[0].split(".")[-1]))
        if word not in out:
            out.append(word)
    return out


def attribute_phrase(item: Item, record: Record | None, expected_values: bool) -> str | None:
    """The forbidden attributes as the tail names them, the check's own values: `text containing "Bob Chen"`,
    `cc "ceo@company.example.com"`, `value "https://…" in column "URL"`; None when a value cannot be shown."""
    content, address = attribute_keys(item, record)
    parts: list[str] = []
    if content:
        if field_check_phrase(item, True) is not None:
            phrase = field_check_phrase(item, expected_values, keys={"additional_checks"})
        else:
            # the labels of the written view (a mention is "containing"), the check's own values: the body quotes what
            # the call wrote, which holds the check's value (D89, harm_scope_object_not_required); a membership value
            # keeps the one written member (engine-fix D79)
            src = item
            view = written_view(item, record) if record is not None and item.location.kind == "at_call" else None
            if view is not None:
                containing = set(view.semantics.get("containing_keys") or ())
                values = dict(view.content)
                for k in content:
                    own = item.content[k]
                    if k in values and values[k].raw != own.raw and isinstance(own.raw, str):
                        values[k] = own
                        containing.add(k)
                src = view.model_copy(update={"content": values, "semantics": {**view.semantics, "containing_keys": sorted(containing)}})
            phrase = content_phrase(src, expected_values, keys=set(content) | {k.split("#")[0] for k in content},
                                    affirm=affirmed_keys(item))
        if not phrase or common.UNSEEN in phrase:
            return None
        if phrase.startswith("(") or "(with " in phrase or " on one line" in phrase or ", without " in phrase:
            return None  # branches of an `any`, terms bound to one line: the field is named, the body holds the values
        phrase = phrase.removeprefix("with ").replace(", with ", " and ")
        phrase = re.sub(r"(^|\band )named(?= containing | \")", r"\1name", phrase)  # the field, not the body's "named X"
        qualifier = [k for k in item.address if k.split("#")[0].split(".")[-1] in QUALIFIER_ADDRESS_KEYS
                     and not isinstance(item.address[k].raw, (dict, list)) and item.address[k].raw is not None]
        if qualifier and any(k.split("#")[0].split(".")[-1] in ("value", "expected_value", "value_contains") for k in content):
            k = qualifier[0]
            shown = show(item.address[k], expected_values)
            if shown is None or common.UNSEEN in shown:
                return None
            phrase += f" in {DEFAULT_LABEL(k.split('#')[0].split('.')[-1])} {shown}"
        parts.append(phrase)
    for k in address:
        shown = show(item.address[k], expected_values)
        if shown is None or common.UNSEEN in shown:
            return None
        base = k.split("#")[0].split(".")[-1]
        parts.append(f"{ADDRESS_LABELS.get(base, DEFAULT_LABEL(base))} {shown}")
    return " and ".join(parts) if parts else None


def attribute_tail(members: list[Item], record: Record | None, expected_values: bool, *, plural_objects: bool,
                   scoped: bool) -> str | None:
    """The attribute tail for the line of these members (one check shape), or None when the check forbids the object
    itself. Members with one phrase name it; members with different values name the field ("that text")."""
    item = members[0]
    content, address = attribute_keys(item, record)
    if not content and not address:
        return None
    phrases = list(dict.fromkeys(attribute_phrase(m, record, expected_values) for m in members))
    labels = attribute_labels(item, record)
    if len(phrases) == 1 and phrases[0] is not None:
        attrs = phrases[0]
        each = ""
    else:  # the body holds the values: "the text shown", "the summary and description shown"
        attrs = "the " + " and ".join(labels) + " shown"
        each = "each of " if plural_objects else ""
    prep = "in" if content else "on"
    if scoped:
        ref = "those records" if plural_objects else "that record"
        return f"; this task forbids {attrs} on {ref}" if not each else f"; this task forbids {attrs} on each of those records"
    n = noun(item)
    ref = f"{each}these {plural(n)}" if plural_objects else f"that {n}"
    return f"; this task forbids {attrs} {prep} {ref}"


def attribute_rewrite(line: str | None, members: list[Item], record: Record | None, expected_values: bool) -> str | None:
    """The line with its object tail ("; this task forbids it/them[ for that record]") replaced by the attribute tail
    when the check forbids an attribute; unchanged otherwise (other tails already name what is forbidden)."""
    if not line:
        return line
    m = _OBJECT_TAIL.search(line)
    if m is None:
        return line
    tail = attribute_tail(members, record, expected_values, plural_objects=m.group(1) == "them", scoped=m.group(2) is not None)
    if tail is None:
        return line
    return line[:m.start()] + tail + line[m.end():]


def attribute_tail_match(line: str) -> dict | None:
    """The parts of an attribute tail in the line ({"attrs", "prep", "ref", "noun"}), or None."""
    m = ATTRIBUTE_TAIL_RE.search(line)
    return m.groupdict() if m else None


def _written_alternatives(item: Item, group: Any) -> set[str]:
    """judge 145: the distinct values of one `any` whose branches the Record found the located object matching
    (`matched_branches`); empty when the Record did not find them."""
    branches = item.semantics.get("branches") or {}
    matched = set((item.semantics.get("matched_branches") or {}).get(str(group)) or ())
    groups = item.semantics.get("alternatives") or {}
    return {str(v.raw).strip().lower() for k, v in [*item.address.items(), *item.content.items()]
            if groups.get(k) == group and str(branches.get(k, k)) in matched and not isinstance(v.raw, (dict, list))}


def candidate_named(item: Item) -> bool:
    """Fix 1: a rule's candidate is named on a prohibition only when the line's own values name it: its key, label or
    record key is an address or content value (the row "Brand Awareness" a message mentions; row 3 the update
    addresses). A candidate tied only through one of its other fields (contact "c14", whose email the message
    contains) is a qualifier no check of the line supports."""
    candidate = item.candidate or {}
    if internal_candidate(item):
        return False
    identity = [x for x in candidate.get("identity") or [] if not isinstance(x, (dict, list))]
    names = {str(candidate.get("key", "")), str(candidate.get("label", ""))} | ({str(identity[-1])} if identity else set())
    names = {n.strip().lower() for n in names if n.strip()}
    groups = item.semantics.get("alternatives") or {}
    matched = item.semantics.get("matched_branches") or {}
    for key, value in [*item.address.items(), *item.content.items()]:
        raw = value.raw
        if key in groups and len(_written_alternatives(item, groups[key])) > 1:
            continue  # judge 145: one of several values the call wrote that each break the rule; naming it alone hides the others
        for element in (raw if isinstance(raw, list) else [raw]):
            if not isinstance(element, (dict, list)) and str(element).strip().lower() in names:
                return True
    return False


def forbids_required(item: Item, record: Record, call=None) -> bool:
    """R2, the renderer's mirror of the checker's `forbids_required_action` / `forbids_required_content`: the
    located call carries a requirement of the same app and effect (its address in the call's arguments, or the
    requirement was witnessed at this call), and every value the forbid line names (address and non-negated
    content) is among what that requirement needs. A line naming something more is narrower and is shown."""
    from ..record.observed import Corpus, normalize, value_forms

    call = call if call is not None else common.located_call(record, item)
    if call is None:
        return False
    args = normalize(json.dumps(call.args, sort_keys=True, default=str))

    def forms(raw):
        if isinstance(raw, dict):
            return {f for v in raw.values() for f in forms(v)}
        return {f for f in value_forms(raw) if f}

    negated_keys = set(item.semantics.get("negated_keys") or ())
    negated = {f for k, v in item.content.items() if k in negated_keys for f in forms(v.raw)}
    forbids = {f for k, v in {**item.address, **item.content}.items() if k not in negated_keys
               and not (isinstance(v.raw, dict) and "unresolved" in v.raw) for f in forms(v.raw)}
    if not forbids:
        return False
    for other in record.items:
        if other.polarity != "require" or other.type == "tool_error" or other.id == item.id:
            continue
        if not _same_app(call.app, other.effect.app) and other.effect.app != item.effect.app:
            continue
        if other.effect.object != item.effect.object and other.effect.object not in ("", other.effect.app) \
                and item.effect.object not in ("", item.effect.app):
            continue
        needs_address = [v.raw for v in other.address.values() if not (isinstance(v.raw, dict) and "unresolved" in v.raw)]
        avoided = set(other.semantics.get("forbidden_content_keys") or ()) | set(other.semantics.get("negated_keys") or ())
        if other.semantics.get("effect_required"):
            avoided |= set(other.probe.failing_clauses)
        groups = other.semantics.get("alternatives") or {}
        branches = other.semantics.get("branches") or {}
        group_branches: dict = {}
        for k, g in groups.items():
            group_branches.setdefault(g, set()).add(branches.get(k, k))
        needs_content = [v.raw for k, v in other.content.items() if k not in avoided and k.split("#")[0] not in avoided
                         and not (k in groups and len(group_branches[groups[k]]) > 1)
                         and not (isinstance(v.raw, dict) and "unresolved" in v.raw)]
        witnessed = (other.location.kind == "after_turn" and other.location.call is not None
                     and (other.location.call.turn, other.location.call.index) == (call.turn, call.index))
        in_call = bool(needs_address) and all(not forms(v) or any(Corpus._form_in(f, args) for f in forms(v)) for v in needs_address)
        if not (witnessed or in_call):
            continue
        needed = {f for v in needs_address + needs_content for f in forms(v)}
        if negated & needed:
            continue
        if forbids <= needed:
            return True
    return False


def read_evidence(item: Item, record: Record) -> bool:
    """Whether a successful call of the item's app returned or named the object a read goal is about (then
    "the attempt never read it" cannot be said)."""
    from ..record.observed import Corpus, normalize, value_forms

    values = _resolved_address_values(item)
    if not values:
        return True
    for call in record.calls:
        if call.error is not None or not _same_app(call.app, item.effect.app):
            continue
        text = normalize(json.dumps([call.args, call.result], sort_keys=True, default=str))
        if all(any(Corpus._form_in(f, text) for f in value_forms(v) if f) for v in values):
            return True
    return False


def _same_app(a: str, b: str) -> bool:
    a, b = a.replace("_", ""), b.replace("_", "")
    return bool(a and b) and (a == b or a.startswith(b) or b.startswith(a))


def suppression_reason(item: Item, record: Record) -> str | None:
    """Why no statement may be made for a failed item (docs: never wrong, silent when unsure)."""
    kind = common.item_kind(item)
    if kind == "tool_error":
        return None
    if kind in ("prohibited", "changed") and item.location.kind != "at_call":
        # a prohibition that cannot name its call cannot name the exact mistake: say nothing
        return "harm_not_located"
    if kind == "prohibited" and item.effect.kind in UPDATE_KINDS and not update_call_counts(item, record):
        # engine-fix 8 (judge-final4 94): an update call counts toward a prohibition on a field value only when its
        # arguments wrote that field; a call that wrote only the description of a record already in stage "Proposal"
        # is no update to the stage, although the record returned shows it
        return "harm_update_value_not_in_args"
    from .. import why

    proof = item.semantics.get("located_proof")
    if why.ENABLED.get() and kind in ("prohibited", "changed") and isinstance(proof, dict):
        # version (c): the located call must be of the effect's app and, where the snapshots tell, must have
        # changed the effect's own collection (the noun of the line); a broken record must be a changed object
        if not proof.get("app_match") or proof.get("collection_changed") is False:
            return "harm_noun_unverified"
        if proof.get("object_produced") is False or (
                "object_produced" in proof and proof.get("object_produced") is None):
            return "harm_object_unverified"  # scale fix: the probe's object is not shown produced by this call
        if item.origin == "native" and kind == "prohibited" and noun_parts(item)[1] and item.probe.offending_object is not None:
            return "harm_thing_object_removed"  # shape removed after its fix still violated: "added a <thing>" natives
        if kind == "prohibited" and item.location.kind == "at_call" and written_view(item, record) is None:
            return "harm_target_not_written" if written_targets(item)[0] is None else "harm_value_not_written"
        if kind == "prohibited" and forbids_required(item, record):
            return "harm_forbids_required"  # R2, exactly the checker's case: everything the line names is required here
        if kind == "prohibited" and _value_scoped(item, record) and scope_forbids_required(item, record):
            return "harm_forbids_required"  # D89: one requirement of the app needs every value the scoped line names
        if kind == "prohibited" and item.location.kind == "at_call" and forbids_required_elsewhere(item, record) \
                and still_required_tail(item, record) is None:
            return "harm_forbids_required"  # D89-2: no wording tells the forbidden object from the required one
        if item.origin == "manifest" and kind == "prohibited" and (
                (item.semantics.get("unresolved") or {}).get("content") or (item.semantics.get("unresolved") or {}).get("address")
                or _effect_residual(item.semantics)) and why.harm_when(item) is None:
            return "harm_as_made_removed"  # fix round: "forbids it as made" lines are removed
        if kind == "changed" and proof.get("changed_carries_address") is False:
            return "changed_claim_without_diff"
    if kind == "prohibited" and item.location.kind == "at_call" and forbids_required(item, record):
        # engine-fix-v11 (14 Nov): R2 holds for every located prohibition, not only those with a located proof; a manifest
        # prohibition without one ("no escalation update" on zv_08) still forbade the required "not a vendor" comment
        return "harm_forbids_required"
    if item.origin == "manifest":
        if item.semantics.get("source_ambiguous"):
            return "harm_source_ambiguous"  # which of the check's alternative effects the call produced is not known
        if item.semantics.get("address_unverified"):
            return "harm_address_unverified"  # none of the ids the rule lists is in the located call's objects
        if contradictory_fields(item):
            # a disjunction of cases rendered as one conjunction would describe no possible object: say nothing
            return "condition_contradicts_itself"
        unresolved = item.semantics.get("unresolved") or {}
        residual = _effect_residual(item.semantics)
        if kind in ("prohibited", "changed"):
            # R2: a prohibition names the offending content exactly, or says nothing. A harm whose
            # condition the compiler could not express is stated only as "this task forbids it as made (a
            # further condition the task states)", and only where no requirement of the same app names
            # the same address (else the line could read as forbidding the required action).
            if unresolved.get("content") or unresolved.get("address") or residual:
                if item.location.kind != "at_call" or address_required_elsewhere(item, record):
                    return "harm_condition_unresolved"
                return None
            if item.location.kind != "at_call":
                return "harm_not_located"
            return None
        if item.effect.kind == "read":
            if not _resolved_address_values(item):
                return "goal_unresolved"
            if read_evidence(item, record):
                return "read_goal_object_was_returned"  # a call returned it: "never read" cannot be said
            return None
        if not item.content and (unresolved.get("address") or unresolved.get("content") or residual or not item.address):
            return "goal_unresolved"  # nothing precise can be asked for
        if item.semantics.get("effect_present") and not _resolved_address_values(item) and not item.candidate:
            # an object carrying everything the line would name exists (abce.record.build): the goal
            # failed for a reason the engine cannot state, "Not done" would be false, and with neither an
            # address nor a candidate there is nothing to point at
            return "goal_effect_exists"
        return None
    probe = item.probe
    if kind == "missing" and item.semantics.get("effect_required"):
        if probe.identity_present is None and not probe.failing_clauses:
            # neither "no such X" nor "X exists with forbidden content" could be established; the
            # requirement itself is still stated (it is true) unless even that names nothing
            return None
    return None


def must_not_rule(item: Item, expected_values: bool = True) -> str | None:
    """engine-fix (silent must-not checks): the rule a benchmark must-not check states, as "no <thing> …" in the
    engine's own words, every value of the check shown; None when a value cannot be shown or the check's terms
    contradict each other. "no message on gorgias ticket "g_211" with sender type "agent", with body containing "g_202"",
    "no tag "cat-account" on helpscout conversation "hs_cat11"", "no google sheets row in spreadsheet "ss_recs" with
    cells containing "Local Networking"", "no update of google sheets row 5 in spreadsheet "ss_breaks""."""
    from ..record.native import OBJECT_ID_KEYS

    if item.origin != "native" or contradictory_fields(item):
        return None
    for value in [*item.address.values(), *item.content.values()]:
        shown = show(value, expected_values)
        if shown is None or common.UNSEEN in shown:
            return None
    base, thing = noun_parts(item)

    def term(key: str, value: Value, side: str) -> list[str]:
        field = key.split("#")[0].split(".")[-1]
        raw = value.raw
        if isinstance(raw, dict):  # parameters of an action, cells of a row: one term per member
            out = []
            for k2, v2 in raw.items():
                v_text = json.dumps(v2) if isinstance(v2, str) else _show_value(v2, False)
                if field == "cell_contains":
                    out.append(f"with cell {json.dumps(str(k2))} containing {v_text}")
                else:
                    label = str(k2) if field in DATA_FIELDS else (_param_label(str(k2)) or str(k2).replace("_", " "))
                    out.append(f"with {label} {v_text}")
            return out
        shown = show(value, expected_values, data_field(key))
        if side == "content":
            return [f"{_label(item, key, None)} {shown}".strip()]
        label = ADDRESS_LABELS.get(field, DEFAULT_LABEL(field))
        if label.split()[:1] and label.split()[0] not in ("in", "to", "on", "for", "from", "of", "at", "with", "named", "assigned"):
            label = "with " + label if label else ""
        return [f"{label} {shown}".strip()]

    if thing:
        thing_key = next((k for k in [*item.address, *item.content] if k.split("#")[0] in (
            thing, f"{thing}_email", f"{thing}_name", f"{thing}_id", f"{thing}s")), None)
        anchors = [k for k in item.address if k.split("#")[0] in OBJECT_ID_KEYS and k != thing_key]
        named = [k for k in item.content if k.split("#")[0] == "name" and k != thing_key]
        own = ""
        if thing_key is not None:
            own = " " + show((item.address.get(thing_key) or item.content.get(thing_key)), expected_values)
        rest: list[str] = []
        for k, v in item.address.items():
            if k not in anchors and k != thing_key:
                rest += term(k, v, "address")
        for k, v in item.content.items():
            if k not in named and k != thing_key:
                rest += term(k, v, "content")
        if anchors or named:
            where = " and ".join(show(item.address[k], expected_values) for k in anchors)
            where += "".join(f" named {show(item.content[k], expected_values)}" for k in named)
            head = f"{thing}{own} on {base} {where.strip()}"
        else:
            head = f"{base} {thing}{own}"
        return " ".join([head, ", ".join(rest)]).strip() if rest else head
    ordered = {k: v for k, v in item.address.items() if k.split("#")[0] in OBJECT_ID_KEYS}
    ordered.update({k: v for k, v in item.address.items() if k not in ordered})
    view = item.model_copy(update={"address": ordered})
    where = address_phrase(view, expected_values, labels=False)  # a row's own name is no value of the check
    terms = [t for k, v in item.content.items() for t in term(k, v, "content")]
    head = f"update of {base}" if item.effect.kind == "update" else base
    if item.effect.kind == "send" and terms:
        # engine-fix 2 (judge v1 31): the message is what is forbidden: its content first, then where it went
        return " ".join(p for p in [head, ", ".join(terms), where] if p)
    return " ".join(p for p in [head, where, ", ".join(terms)] if p)


def _found_count_line(item: Item, record: Record, expected_values: bool, where: str) -> str:
    """A count check counts what its predicate names: `<object>_member_count` counts members of the object
    ("2 members of a google ads customer list named X"), `<object>_count` counts the objects themselves."""
    from ..record.native import split_type

    n = noun(item)
    probe = item.probe
    want = ", ".join(f"{'at least ' if k == 'min_count' else 'exactly '}{v}" for k, v in item.cardinality.items())
    count = max((int(v) for v in item.cardinality.values() if isinstance(v, int)), default=2)
    content = content_phrase(item, expected_values)
    from ..record.native import OBJECTS

    predicate = split_type(item.type)[2] if item.origin == "native" else ""
    word = predicate[: -len("_count")] if predicate.endswith("_count") and predicate != "count" else ""
    # "member_count" counts members of the object; "sent_count" (a state, not a thing) counts the objects
    counted = word.replace("_", " ") if word and (word in THINGS or word in OBJECTS) else ""
    if counted:
        noun_text = f"{plural(counted) if count != 1 else counted} of {article(n)} {n}{where}{(' ' + content) if content else ''}"
        return f"Not done: {want} {noun_text} required; {found_text(probe)}{_candidate(item, expected_values)}."
    noun_text = plural(n) if count != 1 else n
    return f"Not done: {want} {noun_text}{where}{(' ' + content) if content else ''} required; {found_text(probe)}{_candidate(item, expected_values)}."


def found_text(probe) -> str:
    """engine-fix 9 (pattern 2): "16 found, 1 of them there at the start" when the grader counted objects of the initial
    world too (gmail_email_sent_count counts every SENT message; the task's world started with one), so the reader knows
    how many the attempt made; "3 found; 5 were there at the start" when fewer are left than there were; "16 found" when
    the initial world had none (or its count is unknown)."""
    text = f"{probe.found_count} found"
    initial = probe.initial_count
    if initial is None or initial <= 0 or probe.found_count is None:
        return text
    if initial <= probe.found_count:
        return f"{text}, {initial} of them there at the start"
    return f"{text}; {initial} were there at the start"


def statement(item: Item, record: Record, expected_values: bool, *, with_turn: bool = True, final: bool = True) -> str | None:
    """One grammatical line for a failed item, or None when nothing true and precise can be said. `final` False keeps the
    object tail ("; this task forbids it") the collapse of same-shape harms works on; the shown line (`final`) carries the
    attribute tail where the check forbids an attribute (engine-fix 6, `attribute_rewrite`)."""
    from .. import why as _w

    token = _w.RECORD.set(record)
    try:
        text = _statement(item, record, expected_values, with_turn=with_turn)
    finally:
        _w.RECORD.reset(token)
    if final and text is not None and common.item_kind(item) == "prohibited":
        text = attribute_rewrite(text, [item], record, expected_values)
    return text


def _statement(item: Item, record: Record, expected_values: bool, *, with_turn: bool = True) -> str | None:
    if suppression_reason(item, record) is not None:
        return None
    kind = common.item_kind(item)
    turn = item.location.turn
    prefix = f"At turn {turn}, " if with_turn and turn is not None and item.location.kind == "at_call" else ""
    tool = _tool_at(item, record)
    n, addr = noun(item), address_phrase(item, expected_values)
    a = article(n)
    where = f" {addr}" if addr else ""
    if kind == "tool_error":
        # the tool's own message, without the list of known objects it may append (abce.record.calls)
        error = shorten(error_message(item.semantics.get("error") or ""))
        return f"{prefix}{tool or item.semantics.get('tool', 'a call')} failed: {error}"
    if kind == "prohibited":
        from .. import why as _why

        original_item = item
        tail, candidate = "; this task forbids it", _candidate(item, expected_values)
        also = ""
        if _why.ENABLED.get() and item.location.kind == "at_call":
            if not candidate_named(item):
                candidate = ""  # Fix 1: a qualifier no check of this line supports is never appended
            scoped = record_scoped(item)
            if scoped:
                tail = SCOPED_TAIL  # Fix 1: the prohibition is about that record, not the whole action
            original = item
            item = written_view(item, record)
            if item is None:
                return None  # a quoted value the call did not write (harm_value_not_written / harm_target_not_written)
            if not scoped:  # engine-fix: what about the object is forbidden, never the whole action
                unless = unless_scope(original, item, record)
                if unless is not None:
                    item, tail = unless
                    candidate = ""  # the unless clause names the condition; a label after it would read as part of it
                else:
                    scope = value_scope(original, item, record, expected_values)
                    if scope is not None:
                        tail, candidate = scope, ""  # the scope names the object; a label after it would read as part of it
                also = also_scope(original, item, expected_values, _candidate(original, expected_values)
                                  if candidate_named(original) else "") or ""
                if tail == "; this task forbids it" and not candidate and not also and original.origin == "native" \
                        and recipient_scope(original, item, record) is not None \
                        and not forbids_required_elsewhere(original, record):
                    tail = recipient_tail(noun(item))  # engine-fix 3 (judge-final 116)
            n, addr = noun(item), address_phrase(item, expected_values)
            where = f" {addr}" if addr else ""
        content = content_phrase(item, expected_values, affirm=affirmed_keys(item))
        if not content and any(v.raw == "" for k, v in item.content.items() if k in set(item.semantics.get("negated_keys") or ())):
            content = written_row_name(item, record) or ""  # engine-fix 4: name the forbidden record the call wrote
        verb = "updated" if item.effect.kind in ("update", "field") else "produced"
        if noun_parts(item)[1]:  # "<object> has <thing>": the call added the thing (an email to a list, a tag)
            verb = "added"
        field_view = field_check_phrase(original_item, expected_values) if original_item.polarity == "forbid" else None
        if field_view is not None:
            # engine-fix 5 (judge 2a79e7e 144): a native "not exists with field" check names the record by its values
            # (never 'field "name" with value …, with additional checks …'), said as made by the call; when a
            # requirement needs the record's field value, only the forbidden value is named
            where, content = "", field_view
            if (original_item.semantics.get("located_proof") or {}).get("object_produced"):
                verb = "produced"
            split = field_check_forbidden(original_item, record) if original_item.location.kind == "at_call" else None
            if split is not None:
                only = field_check_phrase(original_item, expected_values, keys={"additional_checks"})
                if only is not None:
                    content, candidate, also = only, "", ""
                    tail = f"{FIELD_VALUE_TAIL}{split} for that {n}"
        produced = f"{tool} {verb} " if tool else f"The attempt {verb} "
        several = [v.raw for v in item.address.values() if isinstance(v.raw, list) and len(v.raw) > 1]
        if several and item.origin == "manifest":  # one call that changed several records: say so
            a, n = str(len(several[0])), plural(n)
        unresolved = item.semantics.get("unresolved") or {}
        from .. import why

        rule = why.rule_suffix(item)
        if _why.ENABLED.get() and item.location.kind == "at_call" and forbids_required_elsewhere(original_item, record):
            rule += still_required_tail(original_item, record, expected_values) or ""  # D89-2
        if item.origin == "manifest" and (unresolved.get("content") or unresolved.get("address") or _effect_residual(item.semantics)):
            when = why.harm_when(item)  # N1a: every unstated condition rendered, or the line stays "as made"
            if when is not None:
                return (f"{prefix}{produced}{a} {n}{where}{(' ' + content) if content else ''}; this task forbids it{when}"
                        f"{candidate}{rule}.")
            # the condition is only partly stated: the effect as made is forbidden, not the effect as named
            return (f"{prefix}{produced}{a} {n}{where}{(' ' + content) if content else ''}; this task forbids it as made "
                    f"(a further condition the task states){candidate}{rule}.")
        return f"{prefix}{produced}{a} {n}{where}{(' ' + content) if content else ''}{tail}{candidate}{also}{rule}."
    if kind == "changed":
        fields = protected_fields(item, expected_values)
        if fields:  # engine-fix 7 (judge-final3 2 / 40): the protected field and its value, the object named by its ids
            drop = {k for f in fields for k in f["keys"]}
            if any(f["thing"] for f in fields):
                n = noun_parts(item)[0]
                a = article(n)
            addr = address_phrase(item, expected_values, keys=set(item.address) - drop)
            if addr.startswith(n.split()[-1] + " "):  # "gmail message message "M"" -> "gmail message "M""
                addr = addr[len(n.split()[-1]) + 1:]
            where = f" {addr}" if addr else ""
        return f"{prefix}{tool + ' ' if tool else 'The attempt '}changed {a} {n}{where}{keep_tail(fields)}{_candidate(item, expected_values)}."
    # missing (require failed)
    probe = item.probe
    if item.origin == "manifest" and item.effect.kind == "read":
        # a read obligation: what was returned ("returned fields") is an internal, never rendered; the unread
        # object is named by its visible parts, a hidden part by what it is ("a worksheet the task names")
        resolved = {k: v for k, v in item.address.items() if not (isinstance(v.raw, dict) and "unresolved" in v.raw)}
        visible = {k for k, v in resolved.items() if common.visible(v, expected_values)}
        hidden = [k for k in resolved if k not in visible]
        if not visible:
            return None  # nothing the teacher could recognise: withheld (read_goal_address_hidden)
        shown = address_phrase(item, expected_values, keys=visible)
        from .. import why

        if not hidden:
            return f"Not done: the attempt never read {a} {n} {shown}{_candidate(item, expected_values)}{why.rule_suffix(item)}."
        kinds = {"worksheet": "a worksheet", "spreadsheet": "a spreadsheet", "channel": "a channel"}
        what = next((kinds[k] for key in hidden for k in kinds if key.split("#")[0].split(".")[-1].startswith(k)), f"the {n}")
        return f"Not done: the attempt never read {what} the task names {shown}{_candidate(item, expected_values)}{why.rule_suffix(item)}."
    present = item.semantics.get("effect_present") or (
        item.semantics.get("address_present") and (_has_residual(item) or any(
            not common.visible(v, expected_values) for v in item.content.values()
            if not (isinstance(v.raw, dict) and "unresolved" in v.raw))))
    if item.origin == "manifest" and present:
        # an object carrying everything the line would name exists (abce.record.build), or one with the named
        # identity exists and the line could not state the whole condition (a residual, a hidden value): "Not
        # done: a X" would read as the effect missing; what is true is that the task's condition on X is not met
        from .. import why

        if why.ENABLED.get():
            # fix round: name the object that exists (its own id, from the recorded world) and say only that the
            # grader marks the requirement unmet; without such an object and an address, nothing precise: silent
            return None  # removal round: "Not met" lines are removed (judged not useful)
            obj = item.semantics.get("present_object")
            if not obj or not where:
                return None
            return (f"Not met: the attempt made {a} {n}{where} (id {json.dumps(str(obj['id']))}){_candidate(item, expected_values)}"
                    f"; the grader marks this requirement as not met.")
        return (f"Not met: a further condition the task states on {a} {n}{where}{_candidate(item, expected_values)}"
                f"{why.not_met_suffix(item)}{why.rule_suffix(item)}.")
    thing_line = thing_required_line(item, expected_values)
    if thing_line is not None:
        return thing_line
    state = state_word(item)
    content = content_phrase(item, expected_values)
    if item.semantics.get("effect_required"):
        forbidden = set(item.semantics.get("forbidden_content_keys") or probe.failing_clauses or [])
        required = content_phrase(item, expected_values, negate=forbidden) if content else "as the task states"
        if probe.offending_object is not None and tool and item.location.kind == "at_call":
            offending = content_phrase(item, expected_values, keys=set(probe.failing_clauses), affirm=set(probe.failing_clauses))
            return (f"{prefix}{tool} produced {a} {n}{where} {offending}; the task requires one {required}"
                    f"{_candidate(item, expected_values)}.")
        removal = negative_attribute_line(item, expected_values)  # engine-fix 7 (judge-final3 71)
        if removal is not None:
            return removal
        if probe.identity_present is False:
            return f"Not done: no {n}{where} exists; one is required {required}{_candidate(item, expected_values)}."
        if state:
            identity, rest = split_identity_content(item, expected_values)
            rest = content_phrase(item, expected_values, keys={k.split("#")[0] for k in item.content} - IDENTITY_CONTENT_KEYS,
                                  negate=forbidden) if rest else ""
            return (f"Not done: {a} {n}{where}{(' ' + identity) if identity else ''} is required to be {state}"
                    f"{(' ' + rest) if rest else ''}{_candidate(item, expected_values)}.")
        return f"Not done: {a} {n}{where} is required {required}{_candidate(item, expected_values)}."
    if probe.found_count is not None and item.cardinality:
        return _found_count_line(item, record, expected_values, where)
    if probe.id_record_fields is not None and not content:
        # judge 37/69/70/121/136: the record exists, elsewhere: the state it must reach, and where it is, never
        # "no X exists; one is required" (which reads as a record to create)
        from ..record.native import OBJECT_ID_KEYS

        ids = {k for k in item.address if k.split("#")[0] in OBJECT_ID_KEYS}
        others = set(item.address) - ids
        which = address_phrase(item, expected_values, keys=ids)
        rest = address_phrase(item, expected_values, keys=others)
        if not which or not rest:
            return None
        now = {k: v for k, v in probe.id_record_fields.items()
               if k in others and isinstance(v, (str, int)) and not isinstance(v, bool) and str(v).strip()}
        current = ""
        if set(now) == others:
            view = item.model_copy(update={"address": {k: item.address[k].model_copy(update={
                "raw": now[k], "forms": (str(now[k]).lower(),), "observed": True, "written": False}) for k in others}})
            current = address_phrase(view, expected_values, keys=others)
        verb = "to be " if rest.split()[0] in ("in", "on", "at", "assigned") else ""
        return (f"Not done: the {n} {which} is required {verb}{rest}{_candidate(item, expected_values)}"
                f"{('; it is ' + current) if current else ''}.")
    field_view = field_check_phrase(item, expected_values)
    if field_view is not None:  # engine-fix 4 (judge 1d4e581 61): the record that has the value, not a field to write
        where, content = "", field_view
    if probe.identity_present is False and probe.required_keys:
        from ..record.native import OBJECT_ID_KEYS

        ids = {k for k in item.address if k.split("#")[0] in OBJECT_ID_KEYS}
        if ids and len(item.address) > len(ids) and item.effect.kind in ("field", "has", "update", "flag", "content", "other"):
            # the check is about an existing record (a ticket) and where it must be (a group): not an existence claim
            which = address_phrase(item, expected_values, keys=ids)
            rest = address_phrase(item, expected_values, keys=set(item.address) - ids)
            if rest and rest.split()[0] not in ("in", "to", "on", "for", "from", "of", "at", "with", "assigned"):
                rest = "with " + rest  # "is required with sender type "agent""
            if state:
                rest = f"to be {state}" + (f" {rest}" if rest else "")
                identity, content = split_identity_content(item, expected_values)  # engine-fix 7
                which = f"{which} {identity}" if identity else which
            return f"Not done: {a} {n} {which} is required {rest}{(' ' + content) if content else ''}{_candidate(item, expected_values)}."
        return f"Not done: no {n}{where} exists; one is required{(' ' + content) if content else ''}{_candidate(item, expected_values)}."
    from .. import why as _why_c

    literal = not (_why_c.ENABLED.get() and item.semantics.get("identity_literal") is False)  # correction r0, version c
    if probe.failing_clauses and probe.identity_present is True and literal:
        focused = item
        if probe.missing_elements:
            # list-element focus: name only the elements that are missing
            content_values = dict(item.content)
            for key, indexes in probe.missing_elements.items():
                value = content_values.get(key)
                if value is not None and isinstance(value.raw, list):
                    keep = [i for i in indexes if i < len(value.raw)]
                    hidden = tuple(str(n) for n, i in enumerate(keep) if str(i) in set(value.hidden_elements))
                    content_values[key] = value.model_copy(update={"raw": [value.raw[i] for i in keep], "hidden_elements": hidden})
            focused = item.model_copy(update={"content": content_values})
        failing = content_phrase(focused, expected_values, keys=set(probe.failing_clauses))
        if field_view is not None:
            failing = field_check_phrase(focused, expected_values, keys=set(probe.failing_clauses)) or failing
        base, thing = noun_parts(item)
        if thing:
            return f"Not done: {article(base)} {base}{where} exists, but has no {thing} {failing.split(' ', 2)[-1] if failing.startswith('with ' + thing) else failing}{_candidate(item, expected_values)}."
        if produced_nothing(item):
            # the matching objects were there before the attempt (a channel's earlier messages): what
            # the attempt did is that it produced none
            verb = "sent" if item.effect.kind == "send" else "created"
            return f"Not done: the attempt {verb} no {n}{where}; one is required {failing or 'as the task states'}{_candidate(item, expected_values)}."
        return f"Not done: {a} {n}{where} exists, but none {failing or 'as required'}{_candidate(item, expected_values)}."
    if item.origin == "manifest" and not content and item.effect.kind in ("update", "field") and addr:
        return f"Not done: {a} {n}{where} was to be updated{_candidate(item, expected_values)}{_residual(item)}."
    if state:
        identity, rest = split_identity_content(item, expected_values)  # engine-fix 7: "named "X" is required to be paused"
        return (f"Not done: {a} {n}{where}{(' ' + identity) if identity else ''} is required to be {state}"
                f"{(' ' + rest) if rest else ''}{_candidate(item, expected_values)}{_residual(item)}.")
    return f"Not done: {a} {n}{where}{(' ' + content) if content else ''}{_candidate(item, expected_values)}{_residual(item)}."


def _collapse_tool_errors(items: list[Item], record: Record, expected_values: bool) -> list[tuple[str, str, list[int]]]:
    """Identical failed calls repeated across turns become one line with the turns listed; several
    different errors of one tool become one line with the first error and the count of the rest.
    Returns (line, tool, turns)."""
    by_text: dict[str, list[int]] = {}
    by_tool: dict[str, list[str]] = {}
    for item in items:
        key = statement(item, record, expected_values, with_turn=False)
        if key is None:
            continue
        by_text.setdefault(key, []).append(item.location.turn or 0)
        by_tool.setdefault(key.split(" failed: ")[0], []).append(key)
    out = []
    done: set[str] = set()
    for key, turns in by_text.items():
        tool = key.split(" failed: ")[0]
        texts = list(dict.fromkeys(by_tool[tool]))
        if len(texts) > 1:
            if tool in done:
                continue
            done.add(tool)
            first_turns = sorted(set(by_text[texts[0]]))
            others = [t for text in texts[1:] for t in by_text[text]]  # one per error, not per distinct message
            other_turns = sorted(set(others))
            all_turns = sorted(set(first_turns) | set(other_turns))
            if len(all_turns) > 3:
                # engine-fix-v11 (14 Nov): one per error always (the line says "failed N times"; counting the first
                # message once per turn made it false whenever a turn repeated it: 24 said, 42 errors, refused twice)
                first_count = len(by_text[texts[0]])
                out.append((repeated_failures_line(tool, first_count + len(others), all_turns, texts[0].split(" failed: ", 1)[1]),
                            tool, all_turns))
                continue
            where = f"At turns {', '.join(map(str, first_turns))}, " if len(first_turns) > 1 else f"At turn {first_turns[0]}, "
            at = f" at turn{'s' if len(other_turns) > 1 else ''} {', '.join(map(str, other_turns))}" if other_turns else ""
            out.append((f"{where}{texts[0]} (and {len(others)} other {tool} error{'s' if len(others) > 1 else ''}{at})",
                        tool, sorted(set(first_turns) | set(other_turns))))
            continue
        turns = sorted(set(turns))
        if len(turns) > 3:
            out.append((repeated_failures_line(tool, len(by_text[key]), turns, key.split(" failed: ", 1)[1]), tool, turns))
            continue
        where = f"At turn {turns[0]}, " if len(turns) == 1 else f"At turns {', '.join(map(str, turns))}, "
        out.append((where + key, tool, turns))
    return out


def shorten(text: str, limit: int = 200) -> str:
    """A long error message cut at a word boundary, never inside a value, with the cut marked."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    cut = max(head.rfind(" "), head.rfind(","), head.rfind(";"))
    return (head[:cut] if cut > limit // 2 else head).rstrip(" ,;") + " (...)"


def repeated_failures_line(tool: str, count: int, turns: list[int], first_message: str) -> str:
    """One line for a tool that failed again and again: the count, the turn span and the first message."""
    return f"{tool} failed {count} times (turns {turns[0]}-{turns[-1]}), first: {first_message}"


def failure_turns(line: str) -> list[int]:
    """The turns a failed-call line names: its "At turn(s) …" prefix, its "(and k other … at turns …)" tail, or the
    span of a repeated-failures line."""
    m = REPEATED_FAILURES.match(line)
    if m:
        return list(range(int(m.group("first")), int(m.group("last")) + 1))
    turns = []
    m = re.match(r"^At turns? ([\d, ]+), ", line)
    if m:
        turns += [int(t) for t in m.group(1).split(", ") if t.strip()]
    m = re.search(r" at turns? ([\d, ]+)\)$", line)
    if m:
        turns += [int(t) for t in m.group(1).split(", ") if t.strip()]
    return turns


REPEATED_FAILURES = re.compile(r"^(?P<tool>\S+) failed (?P<count>\d+) times \(turns (?P<first>\d+)-(?P<last>\d+)\), first: (?P<message>.*)$")


def recovered(record: Record, tool: str, turns: list[int]) -> bool:
    """A tool error is recovered when the same tool later ran without an error (the attempt got past it;
    such a line explains less than a missing requirement and ranks below it in the budget)."""
    last = max(turns) if turns else 0
    return any(c.tool == tool and c.error is None and c.turn > last for c in record.calls)


_QUOTED = re.compile(r'"(?:[^"\\]|\\.)*"')


def _shape(text: str) -> str:
    return _QUOTED.sub('"…"', text)


def _merge_values(texts: list[str], *, one_object: bool = False) -> str | None:
    """One text from several of the same shape. Across objects (`one_object` False): exactly one quoted slot
    may differ, and it lists the values in order ("A", "B" and "C" (one each)); None when several slots
    differ (the pairing of values would be lost) or none does. For one object (several checks on the same
    written object): every differing slot lists its values ("A", "B" and "C"), no marker."""
    slots = [_QUOTED.findall(t) for t in texts]
    segments = _QUOTED.split(texts[0])
    if len({len(s) for s in slots}) != 1:  # the texts differ in shape (one lists three values, another two)
        return texts[0] if one_object else None
    varying = [i for i in range(len(slots[0])) if len({s[i] for s in slots}) > 1]
    if not one_object and len(varying) != 1:
        return None
    if one_object and not varying:
        return texts[0]
    out = []
    for i, segment in enumerate(segments):
        out.append(segment)
        if i < len(slots[0]):
            values = list(dict.fromkeys(s[i] for s in slots))
            if len(values) == 1:
                out.append(values[0])
            else:
                out.append(", ".join(values[:-1]) + " and " + values[-1] + ("" if one_object else " (one each)"))
    return "".join(out)


def paired_terms(item: Item) -> bool:
    """The item's content holds several terms of one key family (`text`, `text#1`): values that belong to
    one object together and must never be listed slot-wise across checks."""
    bases = [k.split("#")[0] for k in item.content]
    return any(bases.count(b) >= 2 for b in set(bases))


def _split_statement(text: str, head: str) -> tuple[str, str, str]:
    """(body, tail, candidate) of a harm statement: what follows the head ("tool produced a noun"), the
    fixed tail ("; this task forbids it" / "; it had to stay as it was") and the candidate suffix."""
    rest = text[len(head):].strip()
    recipient = re.search(r"; this task forbids sending to th(?:at|ose) recipients?", rest)
    if recipient:
        body, after = rest[:recipient.start()], rest[recipient.end():]
        return body.strip(), recipient.group(0), after.rstrip(".").strip()
    for tail in ("; this task forbids it as made (a further condition the task states)", SCOPED_TAIL, "; this task forbids it",
                 KEEP_TAIL_WHOLE):
        if tail in rest:
            body, _, after = rest.partition(tail)
            return body.strip(), tail, after.rstrip(".").strip()
    keep = KEEP_TAIL_RE.search(rest)  # engine-fix 7: "; its FX Rate had to stay "1.08""
    if keep:
        return rest[:keep.start()].strip(), keep.group(0), rest[keep.end():].rstrip(".").strip()
    return rest.rstrip("."), "", ""


def collapse_harms(items: list[Item], record: Record, expected_values: bool) -> list[tuple[str, list[Item]]]:
    """Located prohibitions (and broken records) of one tool whose lines differ only in their values become
    one line naming the turns and the objects: "At turns 20, 21, 22, google_sheets_add_row produced 3 google
    sheets rows … with value "A", "B" and "C"; this task forbids them." Returns (line, member items) for
    every group, singletons included (their line is the item's own)."""
    from .. import why

    groups: dict[tuple, list[Item]] = {}
    for item in items:
        if common.item_kind(item) not in ("prohibited", "changed") or item.location.kind != "at_call":
            continue
        with why.enabled_as(False):  # the shape without the rule's own words: those differ only between checks
            text = statement(item, record, expected_values, with_turn=False, final=False)
        own = statement(item, record, expected_values, with_turn=False, final=False)
        if text is None or own is None:
            continue
        shape = _shape(text) if not harm_scoped(item, record, expected_values) else "scoped:" + item.id
        if RECIPIENT_SCOPE in own:  # engine-fix 3: recipient-scoped lines merge only with each other
            shape = "recipient:" + shape
        groups.setdefault((_tool_at(item, record), common.item_kind(item), shape), []).append(item)
    out = []
    for key, members in groups.items():
        # N1 clauses only on a line whose members are all one check (one rule): R9.1
        # D89 (unattributed_line): only the clauses depend on that; the version-(c) line itself (what the call wrote,
        # the scopes) stays as each member's own statement says it, or a member's line would read "with content "X""
        # for a page that only contains X
        with why.enabled_as(why.ENABLED.get(), clauses=why.CLAUSES.get() and len({i.check_id for i in members}) == 1):
            out.extend(_collapse_group(key, members, record, expected_values))
    return out


def writes_named_values(item: Item, record: Record, call) -> bool:
    """Whether the call's arguments carry every value the line names; a recipient (to/cc/bcc) only in a recipient argument
    (an address the body merely mentions is not a message to it)."""
    from ..record.observed import Corpus, normalize

    recipients = {str(v.raw).strip().lower() for k, v in item.address.items()
                  if k.split("#")[0] in RECIPIENT_KEYS and isinstance(v.raw, str)}
    if recipients and not recipients <= set(call_recipients(call)):
        return False
    args = normalize(json.dumps(call.args, sort_keys=True, default=str))
    return all(any(Corpus._form_in(f, args) for f in _scalar_forms(v)) for v in named_values(item, record)
               if str(v).strip().lower() not in recipients)


# engine-fix 8 (judge-final4 94): an update call counts toward a prohibition on a field value only when its arguments wrote
# that field to that value; the record's fields after the call (`located_proof.written`, which `written_values` also reads)
# show what the record holds, not what the call wrote. A collapsed update line counts distinct records (what each call named
# its record by) and says "(N times)" when one record was updated N times.
UPDATE_KINDS = ("update", "field")


def _scalar_content_terms(item: Item) -> list[tuple[str, str, Any]]:
    from .. import fix

    return fix.scalar_content_terms(item)


def args_wrote(call, field: str, value: Any) -> bool:
    """Whether the call's arguments wrote `value` under `field`: a leaf of that name holding a form of the value (as the
    check reads it), or a leaf equal to the value under any name; a state the call's own verb is (`calendly_cancel_event`
    for status "canceled"). The record's fields after the call are not read."""
    from ..record.observed import Corpus, normalize

    forms = _scalar_forms(value)
    want = _norm_key(field)
    if isinstance(value, str):
        stem = re.sub(r"(l?ed|ing|d)$", "", value.strip().lower())
        if len(stem) >= 4 and stem in (call.tool or "").lower():
            return True
    for name, leaf in _arg_leaves(call.args if isinstance(call.args, dict) else {}):
        if leaf is None or isinstance(leaf, (dict, list)):
            continue
        text = normalize(str(leaf))
        if _norm_key(name) == want and any(Corpus._form_in(f, text) for f in forms):
            return True
        if text in forms:
            return True
    return False


def update_written_terms(item: Item, record: Record, call=None) -> list[str] | None:
    """The content keys of a located update prohibition whose value the located call's arguments wrote; None when the
    item states no scalar content (the update of the record itself is what is forbidden)."""
    terms = _scalar_content_terms(item)
    if not terms:
        return None
    call = call or common.located_call(record, item)
    if call is None:
        return []
    return [k for k, field, value in terms if args_wrote(call, field, value)]


def update_call_counts(item: Item, record: Record) -> bool:
    """Whether the located call counts toward an update prohibition: it wrote one of the content values (the forbidden
    value; an unwritten term then names the record, engine-fix 6), or every content term is a selector of the record
    (`located_proof.selector_keys`: the value picks out one record of the collection and the address names none; the
    rule's candidate label; an identity field). A call that wrote none of the values of a record already holding them
    does not count."""
    terms = _scalar_content_terms(item)
    if not terms:
        return True
    written = update_written_terms(item, record)
    if written:
        return True
    selectors = set((item.semantics.get("located_proof") or {}).get("selector_keys") or ())
    label = (item.candidate or {}).get("label")
    for k, field, value in terms:
        if k in selectors or (label is not None and str(value).strip().lower() == str(label).strip().lower()):
            continue
        if field in IDENTITY_FIELDS or field.endswith("_id"):
            continue
        return False
    return True


def update_record_identity(item: Item, record: Record) -> tuple:
    """What a located update call named its record by: the item's scalar address values (their forms) found in the
    call's arguments; without one, the call itself (every call its own record)."""
    from ..record.observed import Corpus, normalize

    call = common.located_call(record, item)
    key = located_key(item)
    if call is None:
        return ("call",) + key
    args = normalize(json.dumps(call.args, sort_keys=True, default=str))
    named = frozenset(f for k, v in item.address.items() if not isinstance(v.raw, (dict, list)) and v.raw is not None
                      for f in _scalar_forms(v.raw) if Corpus._form_in(f, args))
    return ("record", named) if named else ("call",) + key


def with_times(text: str, head: str, n: str, times: int) -> str | None:
    """The single-object statement with "(N times)" after its noun: "TOOL updated a NOUN (2 times) with id …"; None when
    the statement does not start with the head, verb and noun."""
    for verb in ("updated", "changed", "produced", "added"):
        start = f"{head}{verb} {article(n)} {n}"
        if text.startswith(start):
            return f"{start} ({times} times){text[len(start):]}"
    return None


def named_values(item: Item, record: Record) -> list[Any]:
    """The scalar values a located prohibition's line names: the check's own (negated aside), those of its written view,
    and (engine-fix 5) the written cell that names the row when the check's only term is "not empty"."""
    negated = set(item.semantics.get("negated_keys") or ())
    view = written_view(item, record) if item.location.kind == "at_call" else None
    out = []

    def leaves(raw: Any):  # engine-fix 5: a mapping parameter's values ({"Order Number": "4503"}) are named values too
        if isinstance(raw, dict):
            for x in raw.values():
                yield from leaves(x)
        elif isinstance(raw, list):
            for x in raw:
                yield from leaves(x)
        elif isinstance(raw, (str, int, float)) and not isinstance(raw, bool) and str(raw).strip():
            yield raw

    for it in [item] + ([view] if view is not None else []):
        for k, v in {**it.address, **it.content}.items():
            if k in negated or (isinstance(v.raw, dict) and "unresolved" in v.raw):
                continue
            for x in ([v.raw] if not isinstance(v.raw, (dict, list)) else list(leaves(v.raw)) if isinstance(v.raw, dict) else []):
                if isinstance(x, (str, int, float)) and not isinstance(x, bool) and str(x).strip() and x not in out:
                    out.append(x)
    if item.location.kind == "at_call" and names_written_row(item):
        cell = written_row_cell(item, record)
        if cell is not None and cell[1] not in out:
            out.append(cell[1])
    return out


def located_key(item: Item) -> tuple:
    return (item.location.turn, item.location.call.index if item.location.call else 0)


def same_object_calls(item: Item, record: Record, members: list[Item] | None = None) -> set[tuple[int, int]]:
    """engine-fix 4 (judge 1d4e581 127): the calls that produced the forbidden object of a located prohibition: its own
    call and every other successful, world-changing call of the same tool whose arguments carry every value the line names
    (address and content, negated values aside) — the same row written twice at t7.4 and t7.5 is two rows.
    engine-fix 5 (judge 2a79e7e 107/132-134): such another call counts only when it is itself located for one of the
    line's items (`members`), or when the item is a benchmark check without condition that the call breaks by itself
    (every value it names written): a rule-file prohibition holds only at the calls its condition located (the DSP-001
    row written to the same sheet is not one of them)."""
    call = common.located_call(record, item)
    if call is None:
        return set()
    own = {(call.turn, call.index)}
    values = named_values(item, record)
    if not values:
        return own
    located = {located_key(m) for m in (members or [item]) if m.location.kind == "at_call"}
    proves = item.origin == "native" and not item.semantics.get("condition")
    for c in record.calls:
        if c.tool == call.tool and c.error is None and c.mutated is not False and writes_named_values(item, record, c) \
                and (proves or (c.turn, c.index) in located):
            own.add((c.turn, c.index))
    return own


def _collapse_group(key: tuple, members: list[Item], record: Record, expected_values: bool) -> list[tuple[str, list[Item]]]:
    tool, kind, _ = key
    out: list[tuple[str, list[Item]]] = []
    if True:
        members = sorted(members, key=lambda i: (i.location.turn or 0, i.location.call.index if i.location.call else 0))
        turns = sorted({i.location.turn for i in members if i.location.turn is not None})
        # one located call writes one object: the checks located at the same call describe that one object, so
        # their values merge into one description (a message mentioning A, B and C), never into several objects
        by_call: dict[tuple, list[str]] = {}
        for i in members:
            key = (i.location.turn, i.location.call.index if i.location.call else 0)
            text = statement(i, record, expected_values, with_turn=False, final=False)
            if text not in by_call.setdefault(key, []):
                by_call[key].append(text)
        # a check whose content is a set of terms under one key ("text", "text#1": a first and a last name) keeps
        # its values together: at a call with several such checks each is rendered on its own, as it appears
        # engine-fix 2 (judge v1 3): several checks at one call are several forbidden things, never one merged object
        for key in [k for k, ts in by_call.items() if len(ts) > 1]:
            for text in by_call.pop(key):
                own = [i for i in members if (i.location.turn, i.location.call.index if i.location.call else 0) == key
                       and statement(i, record, expected_values, with_turn=False, final=False) == text]
                out.append((f"At turn {key[0]}, {attribute_rewrite(text, own, record, expected_values)}", own))
                members = [i for i in members if i not in own]
        if not by_call:
            return out
        turns = sorted({i.location.turn for i in members if i.location.turn is not None})
        texts = list(dict.fromkeys(_merge_values(ts, one_object=True) if len(ts) > 1 else ts[0] for ts in by_call.values()))
        objects = len(by_call)
        # engine-fix 4 (judge 1d4e581 127): a produced object written again by another call is counted and named
        produced = kind == "prohibited" and all(
            statement(i, record, expected_values, with_turn=False, final=False).startswith(f"{tool} produced ") for i in members) if tool else False
        per_text: dict[str, set] = {}
        if produced:
            for k, ts in by_call.items():
                text = _merge_values(ts, one_object=True) if len(ts) > 1 else ts[0]
                own_members = [i for i in members if (i.location.turn, i.location.call.index if i.location.call else 0) == k]
                per_text.setdefault(text, set()).add(k)
                for i in own_members:
                    per_text[text] |= same_object_calls(i, record, members)
        every = set().union(*per_text.values()) if per_text else set()
        duplicated = produced and (len(every) > objects or any(len(v) > 1 for v in per_text.values()))
        if duplicated:
            objects = len(every)
            turns = sorted(set(turns) | {t for t, _i in every})
        if objects == 1:
            out.append((f"At turn {turns[0]}, {attribute_rewrite(texts[0], members, record, expected_values)}", members))
            return out
        n = statement_noun(members[0], expected_values)
        a = article(n)
        head = f"{tool} " if tool else "The attempt "
        single = next((f"{head}{verb} {a} {n}" for verb in ("produced", "updated", "added", "changed")
                       if texts[0].startswith(f"{head}{verb} {a} {n}")), None)
        # engine-fix 8 (judge-final4 94): an update line counts distinct records, never calls: two updates of one
        # opportunity are "updated a salesforce opportunity (2 times) …", several records some updated more than once
        # are listed one clause per record with "(k times)"
        repeats: dict[tuple, int] = {}
        if kind == "changed" or all(i.effect.kind in UPDATE_KINDS for i in members):
            by_record: dict[tuple, list[tuple]] = {}
            for k_call in by_call:
                i = next(i for i in members if (i.location.turn, i.location.call.index if i.location.call else 0) == k_call)
                by_record.setdefault(update_record_identity(i, record), []).append(k_call)
            if len(by_record) < objects:
                where = f"At turns {', '.join(map(str, turns))}, " if len(turns) > 1 else f"At turn {turns[0]}, "
                if len(by_record) == 1:
                    all_texts = list(dict.fromkeys(t for ts in by_call.values() for t in ts))
                    text = _merge_values(all_texts, one_object=True) if len(all_texts) > 1 else all_texts[0]
                    timed = with_times(text, head, n, objects) if text is not None else None
                    if timed is not None:
                        out.append((f"{where}{attribute_rewrite(timed, members, record, expected_values)}", members))
                        return out
                else:
                    merged_calls: dict[tuple, list[str]] = {}
                    for identity, k_calls in by_record.items():
                        ts = list(dict.fromkeys(t for k_call in k_calls for t in by_call[k_call]))
                        merged_calls[identity] = ts
                        repeats[identity] = len(k_calls)
                    if single is not None and all(len(ts) == 1 or _merge_values(ts, one_object=True) is not None
                                                  for ts in merged_calls.values()):
                        by_call = {identity: ([_merge_values(ts, one_object=True)] if len(ts) > 1 else ts)
                                   for identity, ts in merged_calls.items()}
                        objects = len(by_call)
                        texts = list(dict.fromkeys(ts[0] for ts in by_call.values()))
                    else:
                        repeats = {}
                if len(by_record) == 1 or not repeats:
                    for key, ts in by_call.items():  # one line per call: the count could not be said
                        own = [i for i in members if (i.location.turn, i.location.call.index if i.location.call else 0) == key]
                        text = _merge_values(ts, one_object=True) if len(ts) > 1 else ts[0]
                        out.append((f"At turn {key[0]}, {attribute_rewrite(text, own, record, expected_values)}", own))
                    return out
        merged = _merge_values(texts) if len(texts) > 1 else texts[0]
        if repeats and any(k > 1 for k in repeats.values()):
            merged = None  # engine-fix 8: a repeated record is listed with its "(k times)"
        if merged is None and single is not None:
            # several slots differ: list each object with its own values, so nothing is paired wrongly; past
            # three objects the rest are counted (the notes at their turns carry every one)
            parts = []
            for k_call, ts in by_call.items():  # one clause per object (per located call), never de-duplicated
                text = _merge_values(ts, one_object=True) if len(ts) > 1 else ts[0]
                body, _, candidate = _split_statement(text, single)
                # engine-fix 4: this object's own repeats (calls no other clause of the line stands for)
                times = len({c for i in members if (i.location.turn, i.location.call.index if i.location.call else 0) == k_call
                             for c in same_object_calls(i, record, members)} - (set(by_call) - {k_call})) if duplicated else 1
                parts.append(body + (f" for {candidate[len('for '):]}" if candidate.startswith("for ") else "")
                             + (f" ({times} {plural(n.split()[-1])})" if times > 1 else "")
                             + (f" ({repeats[k_call]} times)" if repeats.get(k_call, 1) > 1 else ""))
            tail = _split_statement(texts[0], single)[1]
            listed = parts[:3] + ([f"and {len(parts) - 3} more like them"] if len(parts) > 3 else [])
            merged = f"{single} ({'; '.join(listed)}){tail}."
        if merged is None:
            for key, ts in by_call.items():  # could not be merged: one line per object (its own call)
                own = [i for i in members if (i.location.turn, i.location.call.index if i.location.call else 0) == key]
                text = _merge_values(ts, one_object=True) if len(ts) > 1 else ts[0]
                out.append((f"At turn {key[0]}, {attribute_rewrite(text, own, record, expected_values)}", own))
            return out
        if duplicated and " (one each)" in merged and len(texts) > 1:
            slots = [_QUOTED.findall(t) for t in texts]
            vary = [i for i in range(len(slots[0])) if len({s_[i] for s_ in slots}) > 1]
            if len(vary) == 1:
                values = list(dict.fromkeys(s_[vary[0]] for s_ in slots))
                counts = {v: sum(len(per_text.get(t, ())) for t, s_ in zip(texts, slots) if s_[vary[0]] == v) for v in values}
                word = n.split()[-1]
                listed = ", ".join(values[:-1]) + " and " + values[-1] + " (one each)"
                pieces = [f"{v} ({counts[v]} {word if counts[v] == 1 else plural(word)})" for v in values]
                counted = ", ".join(pieces[:-1]) + " and " + pieces[-1]
                merged = merged.replace(listed, counted, 1)
        if single is not None and merged.startswith(single):
            merged = f"{head}{single[len(head):].split(' ')[0]} {objects} {plural(n)}" + merged[len(single):]
        if RECIPIENT_SCOPE in merged:  # engine-fix 3: the recipients of the objects, each forbidden
            forbidden = {e for i in members for e in (recipient_scope(i, None, record) or [])}
            merged = merged.replace(recipient_tail(n), recipient_tail(n, objects, len(forbidden)))
        merged = keep_tail_many(merged.replace(SCOPED_TAIL, SCOPED_TAIL_MANY).replace("; this task forbids it", "; this task forbids them"))
        where = f"At turns {', '.join(map(str, turns))}, " if len(turns) > 1 else f"At turn {turns[0]}, "
        out.append((f"{where}{attribute_rewrite(merged, members, record, expected_values)}", members))
    return out


def _twins(items: list[Item]) -> set[str]:
    """Manifest harms located at a call where a native harm of the same app and kind is located and whose
    every value the native line (or its offending object) already carries: the native line says it; a
    manifest harm naming something else at the same call (another record, other content) is no twin."""
    from ..record.observed import Corpus, normalize

    native_at: dict[tuple, list[Item]] = {}
    for i in items:
        if i.origin == "native" and i.location.kind == "at_call" and i.location.call is not None:
            native_at.setdefault((i.location.call.turn, i.location.call.index, i.effect.app, common.item_kind(i)), []).append(i)
    twins = set()
    for i in items:
        if not (i.origin == "manifest" and i.location.kind == "at_call" and i.location.call is not None):
            continue
        natives = native_at.get((i.location.call.turn, i.location.call.index, i.effect.app, common.item_kind(i))) or []
        negated = set(i.semantics.get("negated_keys") or ())
        own = _raw_forms(v for k, v in {**i.address, **i.content}.items() if k not in negated)
        for n in natives:
            covered = _raw_forms(list(n.address.values()) + list(n.content.values()))
            obj = n.probe.offending_object or {}
            text = normalize(json.dumps(obj.get("fields") or {}, sort_keys=True, default=str))
            if all(f in covered or Corpus._form_in(f, text) for f in own):
                twins.add(i.id)
                break
    return twins


def harm_items_shown(record: Record, items: list[Item]) -> list[Item]:
    """The located harms and broken records the plan collapses: not twins of a native harm at the same call,
    not manifest items the grader's own check contradicts (the same selection as `plan`, so the truth
    checker re-derives the same collapsed lines)."""
    twins = _twins(items)
    return [i for i in items if common.item_kind(i) in ("prohibited", "changed") and i.location.kind == "at_call"
            and i.id not in twins and not (i.origin == "manifest" and i.semantics.get("native_disagrees"))]


def plan(record: Record, items: list[Item], expected_values: bool, *, budget: int | None = LINE_BUDGET,
         chars: int | None = CHAR_BUDGET, order: str = DEFAULT_ORDER,
         not_done_source: str = DEFAULT_NOT_DONE_SOURCE) -> dict[str, Any]:
    order_policy = order
    """Statements in order (prohibited, changed, missing, tool errors collapsed, turn limit), the
    per-item texts, and the coverage report (suppressed items with reasons, lines dropped by budget).
    `budget` (lines) and `chars` (characters of the statement lines) bound the first message; None is no limit.
    `order` is the display policy (ORDERS); the budget always keeps the worst lines first.
    `not_done_source` "benchmark" withholds the "Not done" line of every rule-file-only requirement (its text is None,
    so no note or hint is derived from it), before the near-duplicate pass, so a benchmark line never yields to it."""
    if order_policy not in ORDERS:
        raise ValueError(f"unknown order {order_policy!r}; one of {ORDERS}")
    if not_done_source not in NOT_DONE_SOURCES:
        raise ValueError(f"unknown not_done_source {not_done_source!r}; one of {NOT_DONE_SOURCES}")
    texts: dict[str, str | None] = {}
    suppressed: list[dict] = []
    twins = _twins(items)
    lines: list[tuple[str, str, tuple]] = []  # (item id, line, budget priority)
    seen = set()
    errors = [i for i in items if common.item_kind(i) == "tool_error"]
    shown_harms: list[Item] = []
    for item in items:
        if common.item_kind(item) == "tool_error":
            texts[item.id] = statement(item, record, expected_values, with_turn=False)
            continue
        if item.id in twins:
            texts[item.id] = None
            suppressed.append({"id": item.id, "reason": "native_twin_at_same_call"})
            continue
        if item.origin == "manifest" and item.semantics.get("native_disagrees"):
            # the grader's own check of this effect has the other outcome (abce.record.merge)
            texts[item.id] = None
            suppressed.append({"id": item.id, "reason": "native_check_disagrees"})
            continue
        if item.origin == "manifest" and item.polarity == "require" and record.success:
            # the grader passed the attempt: a manifest goal it reads as missing is unsure (located harms stay)
            texts[item.id] = None
            suppressed.append({"id": item.id, "reason": "goal_on_successful_attempt"})
            continue
        text = statement(item, record, expected_values)
        texts[item.id] = text
        if text is None:
            reason = suppression_reason(item, record) or (
                "read_goal_address_hidden" if item.origin == "manifest" and item.effect.kind == "read" else "nothing_to_say")
            suppressed.append({"id": item.id, "reason": reason})
            continue
        if not_done_source == "benchmark" and rule_file_only_requirement(item):
            texts[item.id] = None  # no line, and no note or hint derived from it (they are built from the texts)
            suppressed.append({"id": item.id, "reason": "not_done_rule_file_only"})
            continue
        if common.item_kind(item) in ("prohibited", "changed") and item.location.kind == "at_call":
            shown_harms.append(item)  # collapsed below: same-shape lines of one tool become one line
            continue
        if text in seen:
            continue
        seen.add(text)
        # budget priority (worst first): located harms, unrecovered tool errors, missing work (native
        # before manifest; a line whose every value is a placeholder after the lines that name something),
        # recovered tool errors; the turn-limit line is never dropped
        origin = 0 if item.origin == "native" else 1
        all_placeholder = common.UNSEEN in text and '"' not in text
        lines.append((item.id, text, (3 + origin, 1 if all_placeholder else 0, len(lines))))
    # a "Not done" line whose quoted values are a subset of another line's for the same noun states the
    # same requirement twice (a native check and its manifest twin): the fuller line stands
    quoted = {p: (set(_QUOTED.findall(l)), re.sub(r"^Not done: (?:a |an |the attempt \S+ no |no )?([a-z_ ]+?)\b", r"\1", l).split(" ")[0:2])
              for p, (_i, l, _pr) in enumerate(lines) if l.startswith("Not done: ")}
    dropped_dupes = set()
    for p, (vals, noun) in quoted.items():
        for q, (vals2, noun2) in quoted.items():
            if p != q and noun == noun2 and vals and vals <= vals2 and (vals < vals2 or (q < p)):
                dropped_dupes.add(p)
                break
    for p in sorted(dropped_dupes, reverse=True):
        suppressed.append({"id": lines[p][0], "reason": "near_duplicate_requirement"})
        texts[lines[p][0]] = None
        del lines[p]
    for line, members in collapse_harms(shown_harms, record, expected_values):
        origin = 0 if any(i.origin == "native" for i in members) else 1
        lines.append((",".join(i.id for i in members), line, (origin, 0, len(lines))))
    for text, tool, turns in _collapse_tool_errors(errors, record, expected_values):
        lines.append(("tool_error", text, (5 if recovered(record, tool, turns) else 2, 0, len(lines))))
        if len(turns) > 3:  # one note, at the first failure; the repeats are counted on the line
            for item in errors:
                if item.semantics.get("tool") == tool and item.location.turn in turns[1:]:
                    texts[item.id] = None
                    suppressed.append({"id": item.id, "reason": "repeated_tool_error"})
    if record.budget.get("ended_at_limit"):
        lines.append(("budget", "The attempt ended at its turn limit.", (-1, 0, len(lines))))
    # selection: one line of every kind first (a harm, a missing requirement, a tool error), so eight
    # same-shape harms never hide the one requirement the attempt missed; then worst first
    order = sorted(range(len(lines)), key=lambda p: lines[p][2])
    classes = {0: "harm", 1: "harm", 2: "error", 3: "missing", 4: "missing", 5: "error", -1: "budget"}
    first_of: dict[str, int] = {}
    for p in order:
        first_of.setdefault(classes[lines[p][2][0]], p)
    kept_ids: set[int] = set()
    used = 0
    dropped: list[str] = []
    for position in dict.fromkeys([*first_of.values(), *order]):
        item_id, line, _ = lines[position]
        if (budget is None or len(kept_ids) < budget) and (chars is None or used + len(line) <= chars):
            kept_ids.add(position)
            used += len(line) + 1
        else:
            dropped.append(item_id)
    if order_policy == "turn":
        far = 10 ** 9
        order = sorted(order, key=lambda p: (0 if lines[p][0] != "budget" else 1, line_turn(lines[p][1]) or far))
    out_lines = [lines[p][1] for p in order if p in kept_ids]
    if dropped:
        out_lines.append(f"Not listed: {len(dropped)} more.")
    return {"lines": out_lines, "texts": texts,
            "coverage": {"rendered": len(kept_ids), "suppressed": suppressed, "dropped": dropped}}


def lines(record: Record, items: list[Item], expected_values: bool, *, budget: int | None = LINE_BUDGET,
          chars: int | None = CHAR_BUDGET, order: str = DEFAULT_ORDER) -> list[str]:
    return plan(record, items, expected_values, budget=budget, chars=chars, order=order)["lines"]


def not_done_body(text: str | None) -> str | None:
    """The item text of a "Not done" line ("Not done: a X …." -> "a X …"), or None for any other line."""
    if not text or not text.startswith(NOT_DONE_PREFIX):
        return None
    body = text[len(NOT_DONE_PREFIX):]
    return body[:-1] if body.endswith(".") else body


def final_not_done(items: list[Item], texts: dict[str, str | None]) -> list[tuple[str, list[str]]]:
    """Correction r0: the work still not done when the attempt gave its final reply (after its last call the world is the
    final graded world): exactly the benchmark-backed "Not done" items, i.e. failed benchmark requirements whose line is
    a "Not done" line, with the item text of that line (`texts`, so an item the plan withheld or merged into a fuller
    line is not listed again). Never a "must not" item (`must_not_item`: a prohibition, a record to keep, a check the
    benchmark marks negative), an item stating a count (`count_item`), a passed check, a tool error or a rule-file-only item. Returns (item text, item ids) in
    the order of the facts, one entry per distinct text."""
    out: dict[str, list[str]] = {}
    for item in items:
        if (item.outcome != "failed" or common.item_kind(item) != "missing" or not benchmark_backed(item)
                or must_not_item(item) or count_item(item, texts.get(item.id))):
            continue
        body = not_done_body(texts.get(item.id))
        if body:
            out.setdefault(body, []).append(item.id)
    return list(out.items())


def count_item(item: Item, text: str | None = None) -> bool:
    """An item that states a count ("exactly N … required; M found", a cardinality check): left out of the final note,
    where "still not done" could push the attempt to do more of what it may already have overdone."""
    return bool(item.cardinality) or item.probe.found_count is not None or bool(text and COUNT_TEXT.search(text))


COUNT_TEXT = re.compile(r"(?:at least|exactly) \d+ .*required; \d+ found")


def final_note_line(bodies: list[str]) -> str:
    """The numbered list: "(1) A; (2) B." on one line, or one item per line ("\n1. A\n2. B") when an item contains "; "
    itself (a "; " separator would then be ambiguous)."""
    if any("; " in b for b in bodies):
        return FINAL_NOTE_HEAD + "".join(f"\n{k}. {b}" for k, b in enumerate(bodies, 1))
    return FINAL_NOTE_HEAD + " " + "; ".join(f"({k}) {b}" for k, b in enumerate(bodies, 1)) + "."


def feedback_for_action(record: Record, turn: int, *, expected_values: bool, solution: dict | None,
                        budget: int | None = LINE_BUDGET, chars: int | None = CHAR_BUDGET,
                        order: str = DEFAULT_ORDER) -> str | None:
    """The plain-style text for one action: the same facts as the raw text, as grammar lines. None
    when there is nothing to say (no failed item, no turn limit, no solution)."""
    items = common.facts(record)
    heading = common.FAILED_HEADING if not record.success else common.SUCCESS_HEADING
    body_lines = lines(record, items, expected_values, budget=budget, chars=chars, order=order)
    parts = []
    if solution is not None:
        parts.append("A previous attempt on this same task satisfied all native checks. Use it as a reference, "
                     "not as a new observation in your history.\n\nCorrect solution:\n" + solution["text"])
    elif body_lines:
        parts.append(heading + "\n\n" + "\n".join(body_lines))
    else:
        return None
    parts.append("Reconsider your next action at this point and " + common.CLOSING[0].lower() + common.CLOSING[1:])
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------------------------
# plain_matched: the raw text's information, presented as plain lines (one-factor style test)
# ---------------------------------------------------------------------------------------------
MATCHED_HEADING = ("The following is feedback from this earlier attempt. These are hindsight observations, "
                   "not actions already executed at the original decision point.")
MATCHED_CLOSING = ("Correctly continue the original task from the original decision point. "
                   "Do not repeat a search unchanged when its response supplies no matching evidence. "
                   "Do not invent identifiers or claim completion without evidence.")
MATCHED_SOLUTION_HEADING = ("A previous attempt on this same task satisfied all native checks. "
                            "Use it as a reference, not as a new observation in the student's history.")
MATCHED_SOLUTION_CLOSING = "Reconsider the next action at the original decision point."


def matched_statement(item: Item, record: Record) -> str:
    """One line per failed native check that names every parameter value (address and content), then
    the probe's explanation. Nothing the raw `params` carry is dropped; no existence claim is made
    without a verdict."""
    expected_values = True
    n, addr = noun(item), address_phrase(item, expected_values, labels=False)  # the raw text's information only
    a = article(n)
    where = f" {addr}" if addr else ""
    content = content_phrase(item, expected_values)
    what = f"{n}{where}{(' ' + content) if content else ''}"
    extra = []
    if item.cardinality:
        extra.append(", ".join(f"{'at least ' if k == 'min_count' else 'exactly '}{v}" for k, v in item.cardinality.items()) + " required")
    tool = _tool_at(item, record)
    turn = item.location.turn if item.location.kind == "at_call" else None
    kind = common.item_kind(item)
    probe = item.probe
    if kind == "prohibited":
        where_call = f"At turn {turn}, {tool} produced" if turn and tool else "The attempt produced"
        line = attribute_rewrite(f"{where_call} {a} {what}; this task forbids it.", [item], record, expected_values)
    elif kind == "changed":
        where_call = f"At turn {turn}, {tool} changed" if turn and tool else "The attempt changed"
        line = f"{where_call} {a} {what}; it had to stay as it was."
    else:
        if item.semantics.get("effect_required"):
            forbidden = set(item.semantics.get("forbidden_content_keys") or probe.failing_clauses or [])
            what = f"{n}{where}{(' ' + content_phrase(item, expected_values, negate=forbidden)) if content else ''}"
            if probe.offending_object is not None and tool and turn:
                offending = content_phrase(item, expected_values, keys=set(probe.failing_clauses), affirm=set(probe.failing_clauses))
                line = f"At turn {turn}, {tool} produced {a} {n}{where} {offending}; a {what} is required."
            else:
                why = "; no such " + n + " exists" if probe.identity_present is False else ""
                line = f"Not done: {a} {what} is required{why}."
        elif probe.found_count is not None and item.cardinality:
            line = f"Not done: {extra[0]} {plural(n)}{where}{(' ' + content) if content else ''}; {found_text(probe)}."
            extra = []
        elif probe.identity_present is False and probe.required_keys:
            line = f"Not done: {a} {what}; no such {n} exists."
        elif probe.failing_clauses and probe.identity_present is True:
            failing = content_phrase(item, expected_values, keys=set(probe.failing_clauses))
            line = f"Not done: {a} {what}; {a} {n}{where} exists, but none {failing or 'as required'}."
        else:
            line = f"Not done: {a} {what}."
    if extra:
        line = line[:-1] + f" ({'; '.join(extra)})."
    return line


def _solution_lines(solution_messages: list[dict]) -> list[str]:
    """A successful rollout as readable lines: each assistant tool call, then its tool result verbatim."""
    results = {m.get("tool_call_id"): m for m in solution_messages if m.get("role") == "tool"}
    out = []
    for message in solution_messages:
        if message.get("role") != "assistant":
            continue
        if message.get("content"):
            out.append(f"assistant: {message['content']}")
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = call.get("name") or function.get("name")
            args = call.get("arguments", function.get("arguments"))
            out.append(f"call: {name}({args if isinstance(args, str) else json.dumps(args)})")
            result = results.get(call.get("id"))
            if result is not None:
                body = result.get("content")
                out.append(f"result: {body if isinstance(body, str) else json.dumps(body)}")
    return out


def feedback_for_action_matched(record: Record, turn: int, *, solution_messages: list[dict] | None,
                                observations: list[dict]) -> str:
    if solution_messages is not None:
        return "\n".join([MATCHED_SOLUTION_HEADING, "Correct solution:", *_solution_lines(solution_messages),
                          MATCHED_SOLUTION_CLOSING])
    items = [i for i in record.items if i.origin == "native" and i.type != "tool_error" and i.outcome == "failed"]
    lines_out: list[str] = []
    for item in items:
        line = matched_statement(item, record)
        if line not in lines_out:
            lines_out.append(line)
    if not lines_out:
        lines_out.append("Every native check of this attempt passed.")
    obs_lines = [f"{o.get('tool')}: {o.get('response') if isinstance(o.get('response'), str) else json.dumps(o.get('response'))}"
                 for o in observations] or ["(none)"]
    return "\n".join([MATCHED_HEADING, "Native checks that failed:", *lines_out, "Observed tool responses to this action:",
                      *obs_lines, MATCHED_CLOSING])


def render(record: Record, *, placement: str, expected_values: bool, episode_messages: list[dict] | None = None,
           solution: dict | None = None, budget: int | None = LINE_BUDGET, fix_hints: dict | None = None,
           chars: int | None = CHAR_BUDGET, order: str = DEFAULT_ORDER, not_done_source: str = DEFAULT_NOT_DONE_SOURCE,
           final_note: bool = False) -> dict:
    from .. import why

    # N1 "why" clauses belong to version (c) only: the a/b contexts stay byte-identical; within version (c) they are the
    # fix-hint kind "why_clause" (`fix_hints["kinds"]`; absent = every kind)
    kinds = (fix_hints or {}).get("kinds")
    with why.enabled_as(fix_hints is not None and expected_values, clauses=kinds is None or "why_clause" in kinds):
        return _render(record, placement=placement, expected_values=expected_values, episode_messages=episode_messages,
                       solution=solution, budget=budget, fix_hints=fix_hints, chars=chars, order=order,
                       not_done_source=not_done_source, final_note=final_note)


def _render(record: Record, *, placement: str, expected_values: bool, episode_messages: list[dict] | None = None,
            solution: dict | None = None, budget: int | None = LINE_BUDGET, fix_hints: dict | None = None,
            chars: int | None = CHAR_BUDGET, order: str = DEFAULT_ORDER, not_done_source: str = DEFAULT_NOT_DONE_SOURCE,
            final_note: bool = False) -> dict:
    if final_note and placement not in ("notes", "both"):
        raise ValueError(f"final_note needs per-turn notes (placement 'notes' or 'both'), not {placement!r}")
    items = common.facts(record)
    request = common.request_text(record, episode_messages)
    heading = common.FAILED_HEADING if not record.success else common.SUCCESS_HEADING
    planned = plan(record, items, expected_values, budget=budget, chars=chars, order=order, not_done_source=not_done_source)
    body_lines, texts, coverage = planned["lines"], planned["texts"], planned["coverage"]
    hint_report = None
    hint_notes: dict[int, list[str]] = {}
    fix_notes: dict[int, list[str]] = {}  # Fix 2: placed after the statements of the note
    if fix_hints is not None:
        # version (c): where a needed value lives, which tool returns it, which collection no call wrote;
        # each hint line follows the statement of its requirement (or the body when that line was dropped)
        hint_report = fix_hints["build"](record, texts, **{k: v for k, v in fix_hints.items() if k != "build"})
        by_item = {i.id: i for i in items}
        trailing: list[str] = []
        joined: dict[str, str] = {}  # a "No call wrote" hint is joined to its requirement's own line
        rule_notes: list[dict] = []  # H4: placed right after its owner's statement in the note, never first
        dropped_hints: set[str] = set()
        for meta in hint_report["lines"]:
            owner = by_item.get(meta["item"])
            text = texts.get(meta["item"])
            after = next((b for b in body_lines if text and text in b), None)
            if meta["kind"] == "rule_source":
                # R6.2: only directly after its owner's statement; with that line not shown, not in the body
                if after is not None:
                    meta["after"] = after
                else:
                    dropped_hints.add(meta["line"])
                if owner is not None and common.located_turn(owner) is not None and text is not None:
                    rule_notes.append({"turn": common.located_turn(owner), "after": text, "line": meta["line"]})
                continue
            if after is None:
                trailing.append(meta["line"])
            elif meta["kind"] == "missed_write":
                joined[after] = meta["line"]
                meta["after"] = after
            else:
                meta["after"] = after
            # the note where the hint acts: the failing call it fixes, else the item's own turn
            t = meta.get("at_turn") or (common.located_turn(owner) if owner is not None else None)
            if meta["kind"] == "call_fix":  # after the failure it fixes, in the note at that call
                if t is not None and meta["line"] not in fix_notes.get(t, []):
                    fix_notes.setdefault(t, []).append(meta["line"])
                continue
            if t is not None and meta["line"] not in hint_notes.get(t, []):
                hint_notes.setdefault(t, []).append(meta["line"])
        for meta in hint_report["lines"]:
            # a hint that unblocks a failed call (an id or value the attempt needed and never had) follows the
            # line of that call's failure, or opens the list when no such line exists
            if meta.get("at_tool") and meta["kind"] == "call_fix":
                # Fix 2: right after the line of that very failure (the tool's line naming its turn); with that line
                # not shown, only in the note at the failed call
                fail_line = next((b for b in body_lines if re.search(r"(?:^|, )" + re.escape(meta["at_tool"]) + r" failed", b)
                                  and meta["at_turn"] in failure_turns(b)), None)
                meta["after"], meta["first"] = fail_line, False
                if meta["line"] in trailing:
                    trailing.remove(meta["line"])
                if fail_line is None:
                    meta["body"] = False
                continue
            if meta.get("at_tool"):
                fail_line = next((b for b in body_lines if meta["at_tool"] in b and " failed" in b), None)
                if meta["kind"] == "cause_diff":  # exactly that tool's failure line (B8/B9/B10)
                    fail_line = next((b for b in body_lines if re.search(r"(?:^|, )" + re.escape(meta["at_tool"]) + r" failed", b)), None)
                meta["after"] = fail_line
                meta["first"] = fail_line is None
                if meta["line"] in trailing:
                    trailing.remove(meta["line"])
                if meta["kind"] == "cause_diff" and fail_line is None:
                    # N2 follows the failure line it explains; with that line not shown it is not said at all
                    meta["first"] = False
                    dropped_hints.add(meta["line"])
                    for t_ in list(hint_notes):
                        hint_notes[t_] = [h for h in hint_notes[t_] if h != meta["line"]]
        hint_report["lines"] = [m for m in hint_report["lines"] if m["line"] not in dropped_hints or m["kind"] == "rule_source"]
        for m in hint_report["lines"]:
            if m["line"] in dropped_hints:
                m["body"] = False  # H4 kept for its note only
        new_body: list[str] = [m["line"] for m in hint_report["lines"] if m.get("first")]
        for b in body_lines:
            new_body.append(b + " " + joined[b] if b in joined else b)
            # D89 (call_fix_not_after_its_failure): the fixes of a failure come first, right after its line
            followers = [m for m in hint_report["lines"] if m.get("after") == b and m["kind"] != "missed_write"]
            followers.sort(key=lambda m: m["kind"] != "call_fix")
            for m in followers:
                if m["line"] not in new_body:
                    new_body.append(m["line"])
        body_lines = new_body + [t for t in trailing if t not in new_body]
    reason = common.skip_reason(record, items, solution)
    if reason is None and not body_lines and solution is None:
        reason = "nothing_true_to_say"  # every failed item was withheld
    block_parts = []
    if solution is not None:
        block = solution.get("heading", "Correct solution:") + "\n\n" + solution["text"]
        if solution.get("notes"):  # one true line per value the solution wrote differently from what the task needs
            block += "\n\n" + "\n".join(n["line"] for n in solution["notes"])
        block_parts.append(block)
    if body_lines:
        block_parts.append(heading + "\n\n" + "\n".join(body_lines))
    block_parts.append(common.CLOSING)
    block = "\n\n".join(block_parts)
    notes: dict[int, str] = {}
    body_first = None
    turn = common.decision_turn(record, items)
    if placement in ("first", "both"):
        body_first = block
    if placement == "decision_point":
        notes[turn] = block
    if placement in ("notes", "both"):
        per_turn: dict[int, list[str]] = {}
        for item in items:
            t = common.located_turn(item)
            if t is None or texts.get(item.id) is None:
                continue
            text = statement(item, record, expected_values, with_turn=False)
            if text is None:
                continue
            per_turn.setdefault(t, [])
            if text not in per_turn[t]:
                per_turn[t].append(text)
        for t in list(per_turn):
            per_turn[t] = per_turn[t][:budget] if budget is not None else per_turn[t]
        if fix_hints is not None:
            for rn in rule_notes:  # H4 right after its owner's statement at the owner's turn, or not in the note
                own = per_turn.get(rn["turn"]) or []
                if rn["after"] in own and rn["line"] not in own:
                    own.insert(own.index(rn["after"]) + 1, rn["line"])
        for t in sorted(set(per_turn) | set(hint_notes) | set(fix_notes)):
            notes[t] = common.NOTE_PREFIX + " ".join(hint_notes.get(t, []) + per_turn.get(t, []) + fix_notes.get(t, []))
        if placement == "notes" and not notes:
            notes[turn] = block
    final = None
    if final_note:
        # Correction r0: before the final reply (the last assistant message, no tool call, after the last call), the
        # benchmark-backed work still not done; not bound by the line budget; none on a successful attempt
        final = {"kind": FINAL_NOTE_KIND, "turn": None, "line": None, "items": [], "texts": [], "reason": None}
        listed = final_not_done(items, texts)
        final_turn = common.final_reply_turn(episode_messages, record)
        if record.success:
            final["reason"] = "attempt_succeeded"
        elif not listed:
            final["reason"] = "nothing_required_missing"
        elif final_turn is None:
            final["reason"] = "no_final_reply"
        else:
            line = final_note_line([body for body, _ids in listed])
            final.update({"turn": final_turn, "line": line, "items": [i for _body, ids in listed for i in ids],
                          "texts": [body for body, _ids in listed]})
            # appended last: a note already at that turn keeps its text, the final note follows it
            notes[final_turn] = (notes[final_turn] + " " + line) if final_turn in notes else common.NOTE_PREFIX + line
    context = common.assemble(record, request, body_first, notes, style="plain", placement=placement,
                              expected_values=expected_values, items=items,
                              solution_episode=solution["episode_id"] if solution else None, reason=reason,
                              texts=texts, coverage=coverage)
    if final is not None:
        context["final_note"] = final
    if not_done_source != DEFAULT_NOT_DONE_SOURCE:
        context["not_done_source"] = not_done_source
    if solution is not None and solution.get("notes"):
        context["solution_notes"] = solution["notes"]
    if hint_report is not None:
        context["fix_hints"] = hint_report
        context["coverage"]["fix_hints"] = {"lines": len(hint_report["lines"]), "by_item": hint_report["summary"],
                                            "withheld": hint_report["withheld_reasons"]}
    return context
