"""Teacher-context builder for the SDPO study (plan D5, D16, D20).

Pure functions: (native episode, its siblings, environment findings, tool list)
-> the teacher's first user message plus per-turn notes. Facts come from the
environment's own manifest assessments and native assertion results; turn tags
come from the harness execution link of each tool-server invocation. A literal
is printed only if the attempt observed it (public request or a tool result it
received); otherwise it is described by role.

Excluded by construction: the student's own text, grader values the attempt
never observed, whole manifests, per-task rules, generic instruction
paragraphs, external-model text and re-quoted policy passages.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict, defaultdict

from sdpo_ab_metrics import manifest_findings, messages_of, negative_family, tool_error_text, trace_of

MAX_LINES = 8
RESULT_CHARS = 300
FAILED_HEADING = "The following is feedback from your unsuccessful earlier attempt:"
SUCCESS_HEADING = "The following is feedback on your earlier attempt:"
SOLUTION_HEADING = "Correct solution:"
CLOSING = "Correctly solve the original question."
NOTE_PREFIX = "Note on your next action: "
UNSEEN = "a value this attempt never observed"
KIND_ORDER = ("prohibited", "changed", "tool_error", "missing", "unread", "budget", "correct")
KIND_LABEL = {"prohibited": "prohibited effects", "changed": "changed records", "tool_error": "tool errors",
              "missing": "missing effects", "unread": "unread sources", "budget": "turn budget",
              "correct": "correct effects"}
READ_MARKERS = ("find", "search", "get", "list", "read", "lookup", "retrieve", "fetch", "query", "describe")
UNCHANGED_MARKERS = ("not_updated", "not_changed", "not_voided")
SOURCE_WORDS = ("policy", "policies", "procedure", "guideline", "guidelines", "playbook", "handbook",
                "rules", "matrix", "sop", "criteria", "schedule", "rate card")
SALIENT_KEYS = ("to", "cc", "recipient", "recipients", "email", "channel", "channel_name", "channel_id",
                "user_id", "recipient_id", "spreadsheet", "spreadsheet_id", "worksheet", "worksheet_id",
                "row_id", "ticket_id", "record_id", "id", "conversation_id", "organization_id", "contact_id",
                "project", "issue_key", "profile_id", "to_number", "envelope_id", "name", "title", "subject")


# ---------------------------------------------------------------- values

def normalize(text):
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def _own_free_text(message):
    """Free-text values (with whitespace) the attempt wrote into a call's arguments, by tool call id."""
    out = {}
    for call in message.get("tool_calls") or []:
        try:
            arguments = call.get("arguments") or (call.get("function") or {}).get("arguments") or "{}"
            arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        except ValueError:
            continue
        out[call.get("id")] = [a for a in atoms(arguments) if re.search(r"\s", a.strip()) and len(a.strip()) >= 4]
    return out


def observed_corpus(messages):
    """Everything the attempt observed: the public request and every tool result it received (J). Text the
    attempt wrote itself is not observed when a later tool result merely echoes it back (a created record, a sent
    email): free text from its own arguments is removed from every later tool result, unless a tool result before
    that call already showed it."""
    parts, own = [], []
    for m in messages:
        if m["role"] == "assistant":
            seen = normalize("\n".join(parts))
            for texts in _own_free_text(m).values():
                own += [t for t in texts if normalize(t) not in seen]
            continue
        if m["role"] not in ("user", "tool"):
            continue
        content = m.get("content") or ""
        content = content if isinstance(content, str) else json.dumps(content)
        if m["role"] == "tool":
            for text in own:
                for form in (text, json.dumps(text)[1:-1]):
                    content = content.replace(form, " ")
        parts.append(content)
    return normalize("\n".join(parts))


def value_forms(value):
    if isinstance(value, bool):
        return [str(value).lower()]
    if isinstance(value, (int, float)):
        forms = [str(value)]
        if float(value).is_integer():
            forms += [str(int(value)), f"{int(value):,}"]
        else:
            forms += [f"{value:,.2f}", f"{value:.2f}"]
        return forms
    return [str(value)]


_NUMBER = re.compile(r"^\$?-?\d[\d,]*(\.\d+)?%?$")
_SUPERSEDED = re.compile(r"supersed|outdated|obsolete|deprecated|no longer (valid|current|applies|in effect)|"
                         r"replaced by|old version|previous version|out of date", re.IGNORECASE)


def _is_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)) or bool(
        isinstance(value, str) and _NUMBER.match(value.strip()))


def standalone_number(value, text):
    """A number located in a text only as a whole token: never inside a longer number, a range, a date or an id."""
    for form in value_forms(value) if not isinstance(value, str) else [value.strip()]:
        form = normalize(form)
        if form and re.search(r"(?<![\w.,/-])" + re.escape(form) + r"(?![\w]|[.,/:]\d|-\d)", normalize(text)):
            return True
    return False


def observable(value, corpus):
    """Whole tokens or phrases only, case-insensitive (J): a stem such as "negotiat" is not observed through
    "Negotiation"."""
    if value is None or value == "":
        return True
    for form in value_forms(value):
        form = normalize(form)
        if form and form in corpus and re.search(r"(?<!\w)" + re.escape(form) + r"(?!\w)", corpus):
            return True
    return False


def show(value, corpus):
    """The literal if the attempt observed it, else its role."""
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(show(v, corpus) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {show(v, corpus)}" for k, v in value.items()) + "}"
    if observable(value, corpus):
        return json.dumps(value) if isinstance(value, str) else str(value)
    return f"({UNSEEN})"


def literals(value):
    if isinstance(value, dict):
        return [x for v in value.values() for x in literals(v)]
    if isinstance(value, (list, tuple)):
        return [x for v in value for x in literals(v)]
    if value is None or isinstance(value, bool):
        return []
    text = normalize(value)
    return [text] if len(text) >= 3 else []


# ---------------------------------------------------------------- episode structure

def parse_arguments(raw):
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except ValueError:
        return {"_raw": raw}
    return value if isinstance(value, dict) else {"_value": value}


def episode_calls(record):
    """Ordered tool calls: turn (1-based assistant ordinal), call index, name, arguments, result, error."""
    trace = trace_of(record)
    results = {}
    for node in trace["nodes"]:
        message = node.get("message") or {}
        if message.get("role") == "tool":
            results[message.get("tool_call_id")] = message.get("content")
    calls, turn, node_turns = [], 0, {}
    for index, node in enumerate(trace["nodes"]):
        message = node.get("message") or {}
        if message.get("role") != "assistant":
            continue
        turn += 1
        node_turns[index] = turn
        for position, call in enumerate(message.get("tool_calls") or []):
            name = call.get("name") or (call.get("function") or {}).get("name")
            raw = call.get("arguments", (call.get("function") or {}).get("arguments"))
            result = results.get(call.get("id"))
            error = tool_error_text(result) if result is not None else None
            if error:
                error = re.sub(r"\s*For further information visit \S+", "", error)
                error = re.sub(r"\s+", " ", error).strip()
            calls.append({"turn": turn, "call_index": position, "id": call.get("id"), "name": name,
                          "arguments": parse_arguments(raw), "result": result, "error": error})
    return calls, node_turns, turn


def invocation_map(record, node_turns):
    """Tool-server invocation id -> (turn, call index), through the issuing harness execution.

    Returns (mapping, mapped_all). Uses the environment's `invocation_turns` for the turn when
    importable and cross-checks it."""
    trace = trace_of(record)
    events = trace.get("tool_execution_events") or []
    executions, dispatches = {}, []
    for event in events:
        if event.get("source") in ("harness", "interceptor") and isinstance(event.get("node_index"), int):
            executions.setdefault(str(event.get("execution_id")),
                                  (event["node_index"], event.get("emitted_call_index", 0)))
        elif event.get("source") == "tool_server" and event.get("phase") == "dispatch":
            receipt = json.loads(event["receipt_json"])
            dispatches.append((event["invocation_id"], receipt.get("parent_execution_id")))
    mapping, complete = {}, True
    for invocation, parent in dispatches:
        link = executions.get(str(parent))
        if link is None or link[0] not in node_turns:
            complete = False
            continue
        mapping[invocation] = (node_turns[link[0]], link[1] or 0)
    try:
        from automationbench_v1.manifest_step_credit import invocation_turns

        turns = [n.get("message") for n in trace["nodes"] if (n.get("message") or {}).get("role") == "assistant"]
        native = invocation_turns(turns, events, node_turns)
        if native is None:
            complete = False
        elif any(native.get(k) != v[0] for k, v in mapping.items()):
            raise ValueError("invocation turn mapping disagrees with the environment")
    except ImportError:
        pass
    return mapping, complete


def is_read(name):
    lowered = (name or "").lower()
    return any(marker in lowered for marker in READ_MARKERS)


def destination(call, corpus):
    args = call["arguments"]
    keys = [k for k in SALIENT_KEYS if k in args][:3]
    if not keys:
        keys = list(args)[:2]
    return ", ".join(f"{k}={show(args[k], corpus)}" for k in keys)


def humanize(identifier):
    return identifier.replace(".", " ").replace("_", " ").replace("-", " ").strip()


def assertion_text(assertion, corpus):
    params = {k: v for k, v in (assertion.get("params") or {}).items() if k not in ("excluded", "scored")}
    shown = "; ".join(f"{k} {show(v, corpus)}" for k, v in params.items())
    return f"{humanize(assertion['type'])} ({shown})" if shown else humanize(assertion["type"])


def atoms(value):
    """Every scalar inside a value, as strings."""
    if isinstance(value, dict):
        return [x for v in value.values() for x in atoms(v)]
    if isinstance(value, (list, tuple)):
        return [x for v in value for x in atoms(v)]
    return [] if value is None else [str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)]


def record_label(identity, corpus):
    """A manifest candidate identity such as [sheet, worksheet, "int", 3] as a short record label."""
    parts = [show(x, corpus) if not isinstance(x, int) else f"row {x}" for x in identity]
    return " ".join(parts)


IDENTITY_KEYS = ("to", "email", "channel_name", "channel", "recipient_id", "to_number", "profile_id", "profile_url",
                 "signer_email", "customer_name", "name", "title", "project")


def identity_values(params):
    """The identifying values of a native check: ids and addressees, not contents."""
    out = []
    for key, value in (params or {}).items():
        if key.endswith("id") or key.endswith("_ids") or key in IDENTITY_KEYS:
            out += [normalize(x) for x in atoms(value)]
    return out


def content_values(params):
    out = []
    for key, value in (params or {}).items():
        if "contains" in key or key in ("value", "text", "fields", "cells"):
            out += [normalize(x) for x in atoms(value) if len(str(x)) >= 3]
    return out


def locate(calls, params):
    """Successful write calls that carry the check's identifiers and, when it names content, that content."""
    identity, content = identity_values(params), content_values(params)
    found = []
    for c in calls:
        if is_read(c["name"]) or c["error"] or not call_matches(c, identity):
            continue
        text = normalize(json.dumps(c["arguments"], ensure_ascii=False))
        if content and not any(v in text for v in content):
            continue
        found.append(c)
    return found


def call_matches(call, values):
    """A call whose arguments carry every identifying value (short values must match a whole argument)."""
    if not values:
        return False
    text = normalize(json.dumps(call["arguments"], ensure_ascii=False))
    whole = {normalize(x) for x in atoms(call["arguments"])}
    return all((v in whole) if len(v) <= 3 else (v in text) for v in values)


# ---------------------------------------------------------------- facts

def contract_for(task_name):
    """The task's manifest contract (populations only are used, for value overlap)."""
    try:
        from importlib import resources

        base = resources.files("automationbench_v1") / "contracts"
        catalog = json.loads((base / "catalog.json").read_text())
        relative = catalog["tasks"].get(task_name)
        return json.loads((base / relative).read_text()) if relative else None
    except (ImportError, FileNotFoundError, KeyError, ModuleNotFoundError):
        return None


def field_values(fields):
    """Literal values of a contract source's fields (a mapping or a list of fields)."""
    items = fields.values() if isinstance(fields, dict) else fields if isinstance(fields, list) else []
    return [item.get("value") if isinstance(item, dict) else item for item in items]


def check_literals(contract, signal):
    """Literal values of the population a manifest check is about."""
    if not contract:
        return []
    short = signal.split(".", 1)[-1].rsplit(".coverage", 1)[0].rsplit(".compliance", 1)[0]
    for check in contract.get("checks", []):
        if check.get("check_id") == signal or check.get("signal_id", "").split(".", 1)[-1] == short:
            source = contract.get("sources", {}).get(check.get("population"), {})
            return literals(field_values(source.get("fields")))
    return []


def build_facts(record, contract=None):
    """Ordered facts for one episode: dicts with kind, text, turn (or None) and calls (bad spans)."""
    trace = trace_of(record)
    messages = messages_of(record)
    corpus = observed_corpus(messages)
    calls, node_turns, turns = episode_calls(record)
    by_turn_call = {(c["turn"], c["call_index"]): c for c in calls}
    invocations, mapped = invocation_map(record, node_turns)
    findings = manifest_findings(record)
    assertions = (trace.get("info") or {}).get("automationbench", {}).get("assertions") or []
    failed = [a for a in assertions if not a.get("excluded") and not a.get("passed")]
    passed = [a for a in assertions if not a.get("excluded") and a.get("passed")]
    facts = []
    used_native = set()

    # 2: prohibited effects, located by the guard's witnessed invocation.
    guards = OrderedDict()
    for f in findings:
        if f["producer"].endswith("guards") and f["kind"] == "execution" and f["value"] == 1:
            guards.setdefault(f["check_id"] or f["signal"], []).append(f["invocation"])
    guard_rows = []
    for signal, occurrences in guards.items():
        located = []
        for invocation in dict.fromkeys(occurrences):
            link = invocations.get(invocation)
            call = by_turn_call.get(link) if link else None
            located.append(call)
        values = check_literals(contract, signal)
        native = [i for i, a in enumerate(failed)
                  if negative_family(a["type"]) and i not in used_native
                  and not any(m in a["type"] for m in UNCHANGED_MARKERS)
                  and ((values and set(literals(a.get("params"))) & set(values))
                       or any(c and call_matches(c, identity_values(a.get("params"))) for c in located))]
        used_native.update(native)
        guard_rows.append((signal, located, native))
    # Guards witnessed at the same single call site (same turn and tool) merge into one line.
    merged = OrderedDict()
    for signal, located, native in guard_rows:
        site = (located[0]["turn"], located[0]["name"]) if len(located) == 1 and located[0] else (signal,)
        merged.setdefault(site, []).append((signal, located, native))
    for site, members in merged.items():
        names = ", ".join(humanize(m[0].split(".", 1)[-1]) for m in members)
        located = [c for m in members for c in m[1]]
        native = [i for m in members for i in m[2]]
        if len(site) == 2 and len(members) > 1:
            where = f"turn {site[0]}: " + "; ".join(f"{c['name']}({destination(c, corpus)})" for c in located if c)
            text = f"Effects this task prohibits were performed ({names}) at {where}."
        else:
            where = "; ".join(f"turn {c['turn']}: {c['name']}({destination(c, corpus)})" if c else "an unmapped call"
                              for c in located)
            text = f"An effect this task prohibits was performed ({names}) at {where}."
        check = "; ".join(assertion_text(failed[i], corpus) for i in native)
        if check:
            text += f" Failed check: {check}."
        facts.append({"kind": "prohibited", "text": text, "turn": min((c["turn"] for c in located if c), default=None),
                      "calls": [c for c in located if c], "names": names, "check": check})
    # 2 (native only): failed prohibitions without a guard finding, one line per check type.
    by_type = OrderedDict()
    for i, a in enumerate(failed):
        if i in used_native or not negative_family(a["type"]) or any(m in a["type"] for m in UNCHANGED_MARKERS):
            continue
        used_native.add(i)
        by_type.setdefault(a["type"], []).append(a)
    for kind, group in by_type.items():
        located = []
        for a in group:
            located += [c for c in locate(calls, a.get("params")) if not any(c is d for d in located)]
        located.sort(key=lambda c: (c["turn"], c["call_index"]))
        where = "; ".join(f"turn {c['turn']}: {c['name']}({destination(c, corpus)})" for c in located)
        shown = ", ".join("(" + "; ".join(f"{k} {show(v, corpus)}" for k, v in (a.get("params") or {}).items()
                                          if k not in ("excluded", "scored")) + ")" for a in group)
        text = f"A prohibited effect was produced (failed check: {humanize(kind)} {shown})"
        text += f", caused at {where}." if where else "."
        facts.append({"kind": "prohibited", "text": text, "turn": located[0]["turn"] if located else None,
                      "calls": located, "names": None, "check": f"{humanize(kind)} {shown}"})
    # 3: records that had to stay unchanged.
    retained = [f for f in findings if "retained" in f["producer"] and f["kind"] == "trace"
                and f["value"] == 0 and not f["signal"].endswith(".coverage")]
    for i, a in enumerate(failed):
        if i in used_native or not any(m in a["type"] for m in UNCHANGED_MARKERS):
            continue
        used_native.add(i)
        located = locate(calls, a.get("params"))
        owner = next((f for f in facts if f["kind"] == "prohibited" and located
                      and all(any(c is d for d in f["calls"]) for c in located)), None)
        if owner is not None:
            owner["text"] = owner["text"].rstrip(".") + f"; failed check: {assertion_text(a, corpus)}."
            continue
        where = "; ".join(f"turn {c['turn']}: {c['name']}({destination(c, corpus)})" for c in located)
        text = f"A record that had to stay unchanged was changed: {assertion_text(a, corpus)} failed"
        text += f", changed at {where}." if where else "."
        facts.append({"kind": "changed", "text": text, "turn": located[0]["turn"] if located else None,
                      "calls": located, "check": assertion_text(a, corpus)})
    del retained  # retained-row verdicts also encode required final values; native checks state them
    # 4: tool errors, verbatim, with the bare identifier when the error lists the valid one.
    # Errors of one tool that differ only in the quoted name merge into one line listing the turns.
    groups = OrderedDict()
    for c in calls:
        if c["error"]:
            groups.setdefault((c["name"], re.sub(r"'[^']*'", "'...'", c["error"])), []).append(c)
    for (name, _), members in groups.items():
        first = members[0]
        valid = re.findall(r"\(([A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_\-]+)\)", first["error"])
        hint = f" The identifier to use is {', '.join(dict.fromkeys(valid))}." if valid else ""
        turns_text = ", ".join(str(t) for t in dict.fromkeys(c["turn"] for c in members))
        label = "Turn" if len(members) == 1 else "Turns"
        text = (f"{label} {turns_text}: {name}({destination(first, corpus)}) returned an error: "
                f"\"{first['error'][:240]}\"")
        if len(members) > 1:
            text += f" ({len(members) - 1} more call(s) of {name} failed the same way)"
        text += hint
        notes = {c["turn"]: f"{c['name']}({destination(c, corpus)}) returned an error: \"{c['error'][:240]}\"{hint}"
                 for c in reversed(members)}
        facts.append({"kind": "tool_error", "text": text, "turn": first["turn"], "calls": members, "notes": notes})
    # 6: turn budget, never phrased as a wrong decision.
    stop = trace.get("stop_condition")
    truncated = stop not in ("agent_completed", None)
    # 1: required effects missing by the end of the attempt.
    missing = defaultdict(list)
    for i, a in enumerate(failed):
        if i in used_native or negative_family(a["type"]):
            continue
        missing[a["type"]].append(a)
    missing_texts = [", ".join(assertion_text(a, corpus) for a in group) for group in missing.values()]
    failed_values = set(literals([a.get("params") for a in failed]))
    failed_atoms = [{normalize(v) for v in atoms(a.get("params"))} for a in failed]
    obligations = OrderedDict()
    for f in findings:
        if (f["producer"].endswith("obligations") and f["kind"] == "trace" and f["value"] == 0
                and f["semantics"] != "other" and not f["signal"].endswith(".coverage")):
            obligations.setdefault(f["check_id"] or f["signal"], []).append(f)
    for check, members in obligations.items():
        records, covered = [], 0
        for f in members:
            identity = [x for x in (f["candidate"] or []) if x not in ("int", "str")]
            if identity and any({normalize(x) for x in identity} <= group for group in failed_atoms):
                covered += 1
                continue
            values = check_literals(contract, f["signal"])
            if not identity and values and set(values) & failed_values:
                covered += 1
                continue
            label = (record_label(identity, corpus) if identity else
                     ", ".join(show(v, corpus) for v in values[:3]))
            if label.replace(f"({UNSEEN})", "").replace(",", "").strip():
                records.append(label)  # a label made only of unobserved values says nothing
        if covered and not records:
            continue
        name = humanize(check.split(".", 1)[-1])
        shown = "; ".join(dict.fromkeys(r for r in records if r))
        missing_texts.append(name + (f" for {shown}" if shown else ""))
    if truncated:
        pending = "; ".join(missing_texts) if missing_texts else "none recorded by the checks"
        facts.append({"kind": "budget", "turn": turns,
                      "text": f"The attempt ran out of budget ({stop}) at turn {turns} before finishing. "
                              f"Required effects not yet done: {pending}.", "calls": []})
    else:
        for text in missing_texts:
            facts.append({"kind": "missing", "turn": None, "calls": [], "detail": text,
                          "text": f"By the end of the attempt this required effect was missing: {text}."})
    # 5: a source the request names was never read.
    request = next((m.get("content") or "" for m in messages if m["role"] == "user"), "")
    named = [w for w in SOURCE_WORDS if re.search(rf"\b{re.escape(w)}\b", request, re.IGNORECASE)]
    results = normalize(" ".join(str(c["result"]) for c in calls if c["result"] and not c["error"]))
    unread = named if named and not any(w in results for w in named) else []
    if unread and failed:
        readers = [t for t in tool_names(record) if is_read(t)][:5]
        facts.append({"kind": "unread", "turn": None, "calls": [],
                      "text": f"The request refers to its {unread[0]}, but no tool result in this attempt "
                              f"contained one. Available tools that could find it: {', '.join(readers)}."})
    # 7: effects that were correct.
    good = [a for a in passed if not negative_family(a["type"])]
    if good:
        items = [assertion_text(a, corpus) for a in good]
        more = f"; and {len(items) - 6} more" if len(items) > 6 else ""
        facts.append({"kind": "correct", "turn": None, "calls": [],
                      "text": "Correct and should be kept: " + "; ".join(items[:6]) + more + "."})
    facts.sort(key=lambda f: KIND_ORDER.index(f["kind"]))
    return {"facts": facts, "mapped_all_invocations": mapped,
            "untagged": not mapped}


def tool_names(record):
    return [t.get("name") or (t.get("function") or {}).get("name") for t in trace_of(record).get("tools") or []]


def render_facts(facts, tags=True):
    """At most MAX_LINES lines; the last counts omitted facts per kind."""
    lines = []
    for f in facts:
        text = f["text"] if tags else re.sub(r"\bturns? [\d, ]+: ", "", f["text"], flags=re.IGNORECASE)
        lines.append(f"- {text}")
    if len(lines) > MAX_LINES:
        omitted = defaultdict(int)
        for f in facts[MAX_LINES - 1:]:
            omitted[f["kind"]] += 1
        lines = lines[:MAX_LINES - 1] + ["- Not listed: " + ", ".join(
            f"{n} more {KIND_LABEL[k]}" for k, n in omitted.items()) + "."]
    return lines


def problem_facts(facts):
    return [f for f in facts if f["kind"] != "correct"]


# ---------------------------------------------------------------- clean solution

def clean_sibling(record):
    """Qualifies when every native check passed, no manifest harm, no tool error after the last write."""
    from sdpo_ab_metrics import episode_outcome

    outcome = episode_outcome(record)
    if outcome["success"] != 1.0 or outcome["harm"] or outcome["preservation_break"]:
        return None
    calls, _, turns = episode_calls(record)
    writes = [i for i, c in enumerate(calls) if not is_read(c["name"]) and not c["error"]]
    last_write = writes[-1] if writes else -1
    if any(c["error"] for c in calls[last_write + 1:]):
        return None
    kept = [c for c in calls if not c["error"]]
    # drop writes a later identical write replaced
    final = []
    for i, c in enumerate(kept):
        if not is_read(c["name"]) and any(not is_read(d["name"]) and d["name"] == c["name"]
                                          and d["arguments"] == c["arguments"] for d in kept[i + 1:]):
            continue
        final.append(c)
    # drop repeated identical reads
    seen, deduped = set(), []
    for c in final:
        key = (c["name"], json.dumps(c["arguments"], sort_keys=True))
        if is_read(c["name"]) and key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    write_values = [v for c in deduped if not is_read(c["name"]) for v in literals(c["arguments"])]
    cleaned = [c for c in deduped if not is_read(c["name"])
               or any(v in normalize(c["result"] or "") for v in write_values)]
    assertions = trace_of(record)["info"]["automationbench"]["assertions"]
    scored = sum(not a.get("excluded") for a in assertions)
    return {"episode_id": record.get("id"), "calls": cleaned, "turns": turns,
            "writes": sum(not is_read(c["name"]) for c in cleaned), "scored": scored}


def render_solution(solution):
    lines = []
    for n, c in enumerate(solution["calls"], 1):
        args = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in c["arguments"].items())
        result = c["result"] if isinstance(c["result"], str) else json.dumps(c["result"])
        result = (result or "").replace("\n", " ")
        if len(result) > RESULT_CHARS:
            result = result[:RESULT_CHARS] + " [shortened]"
        lines.append(f"{n}. {c['name']}({args})\n   -> {result}")
    lines.append(f"Satisfied: {solution['scored']} of {solution['scored']} scored checks.")
    return "\n".join(lines)


def choose_solution(record, siblings):
    """Fewest kept writes, then fewest turns, then lowest sample index; never the episode itself."""
    options = []
    for index, sibling in enumerate(siblings):
        if sibling.get("id") == record.get("id"):
            continue
        cleaned = clean_sibling(sibling)
        if cleaned:
            options.append((cleaned["writes"], cleaned["turns"], index, cleaned))
    return min(options, key=lambda o: o[:3])[3] if options else None


# ---------------------------------------------------------------- teacher context

def first_user(record):
    return next(m["content"] for m in messages_of(record) if m["role"] == "user")


CORRECTIVE_LINES = 5
CORRECTIVE_ORDER = ("prohibited", "tool_error", "changed", "missing", "unread", "budget")


def error_correction(call):
    """The corrected argument for a tool error, when the error itself shows it (plan D21 variant C)."""
    error = call["error"] or ""
    valid = list(dict.fromkeys(re.findall(r"\(([A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_\-]+)\)", error)))
    missing = re.search(r"(\w+) '([^']*)' not found", error)
    if missing and valid:
        noun, bad = missing.group(1).lower(), normalize(missing.group(2))
        key = next((k for k, v in call["arguments"].items()
                    if isinstance(v, str) and bad and normalize(v) == bad), None)
        key = key or next((k for k in call["arguments"] if noun in k.lower()), noun)
        return f'{key}="{valid[0]}"' + (f" (other valid values: {', '.join(valid[1:4])})" if len(valid) > 1 else "")
    typed = re.search(r"(\w+)\.str Input should be a valid string \[type=string_type, input_value=([^,\]]+)", error)
    if typed:
        return f'{typed.group(1)}="{typed.group(2).strip()}" (a string, not a number)'
    return None


def corrective_facts(facts, corpus):
    """Variant C: each problem fact names the turn and tool and states the correct alternative when it
    is observable; otherwise the D5 wording is kept. At most CORRECTIVE_LINES lines."""
    out = []
    for f in facts:
        if f["kind"] == "correct":
            continue
        text, notes = f["text"], {}
        if f["kind"] == "tool_error":
            first = f["calls"][0]
            fix = error_correction(first)
            turns = ", ".join(str(t) for t in dict.fromkeys(c["turn"] for c in f["calls"]))
            label = "Turn" if len(f["calls"]) == 1 else "Turns"
            short = first["error"][:150]
            if fix:
                text = (f"{label} {turns}: {first['name']}({destination(first, corpus)}) failed "
                        f"(\"{short}\"). Correct call: {first['name']}({fix}).")
            else:
                text = f"{label} {turns}: {first['name']}({destination(first, corpus)}) failed: \"{short}\"."
            for c in f["calls"]:
                own = error_correction(c)
                notes.setdefault(c["turn"], []).append(f"{c['name']}({destination(c, corpus)}) fails here (\"{c['error'][:150]}\")."
                                    + (f" Correct call: {c['name']}({own})." if own else ""))
            notes = {t: " ".join(v) for t, v in notes.items()}
        elif f["kind"] == "prohibited" and f["calls"]:
            turns = sorted({c["turn"] for c in f["calls"]})
            label = "Turn" if len(turns) == 1 else "Turns"
            where = "; ".join(f"{c['name']}({destination(c, corpus)})" for c in f["calls"])
            rule = f" ({f['names']})" if f.get("names") else ""
            text = (f"{label} {', '.join(map(str, turns))}: do not make the call {where}; it performs an effect "
                    f"this task prohibits{rule}.")
            if f.get("check"):
                text += f" Failed check: {f['check']}."
            for c in f["calls"]:
                notes.setdefault(c["turn"], []).append(
                    f"do not make the call {c['name']}({destination(c, corpus)}); it performs an effect this task "
                    f"prohibits{rule}.")
            notes = {t: " ".join(v) for t, v in notes.items()}
        elif f["kind"] == "changed" and f["calls"]:
            turns = sorted({c["turn"] for c in f["calls"]})
            where = "; ".join(f"{c['name']}({destination(c, corpus)})" for c in f["calls"])
            text = (f"Turn {', '.join(map(str, turns))}: do not make the call {where}; it changed a record that "
                    f"must stay unchanged ({f['check']}).")
            for c in f["calls"]:
                notes.setdefault(c["turn"], []).append(
                    f"do not make the call {c['name']}({destination(c, corpus)}); the record must stay unchanged.")
            notes = {t: " ".join(v) for t, v in notes.items()}
        elif f["kind"] == "missing":
            text = f"Still required by the end: {f['detail']}."
        out.append({**f, "text": text, "notes": notes} if notes else {**f, "text": text})
    out.sort(key=lambda f: CORRECTIVE_ORDER.index(f["kind"]))
    return out


def render_corrective(facts):
    lines = [f"- {f['text']}" for f in facts]
    if len(lines) > CORRECTIVE_LINES:
        omitted = defaultdict(int)
        for f in facts[CORRECTIVE_LINES - 1:]:
            omitted[f["kind"]] += 1
        lines = lines[:CORRECTIVE_LINES - 1] + ["- Not listed: " + ", ".join(
            f"{n} more {KIND_LABEL[k]}" for k, n in omitted.items()) + "."]
    return lines


def build_context(record, siblings, arm, placement="both", contract=None, variant="F", *, exclusions=None,
                  verification="run", explain=False, goals=None):
    """Teacher first user message and per-turn notes for one episode.

    arm: "A" (facts only) or "B" (facts + cleaned sibling solution when one exists).
    placement: "first" (facts in the first message), "notes" (located facts as turn notes
    only), "both". Returns None when the episode contributes no loss.

    Variant "D" (plan D24) only: `exclusions` overrides the curated exclusion list;
    `verification` is "run" (replay unverified corrections and solutions in the frozen
    environment now; `prepare_verifications` batches them ahead) or "cache" (use only
    replays already run; anything unverified is not shown); `explain=True` returns the
    dict with its `skip_reason` instead of None when the episode contributes no loss.
    """
    if arm not in ("A", "B") or placement not in ("first", "notes", "both") or variant not in ("F", "C", "D"):
        raise ValueError("unknown arm, placement or variant")
    if variant == "D":
        if verification not in ("run", "cache"):
            raise ValueError("verification must be 'run' or 'cache'")
        context = build_context_d(record, siblings, arm, placement, contract, exclusions, verification, goals)
        return context if explain or context["skip_reason"] is None else None
    built = build_facts(record, contract)
    facts = built["facts"]
    if variant == "C":
        facts = corrective_facts(facts, observed_corpus(messages_of(record)))
    success = trace_of(record).get("metrics", {}).get("task_completed_correctly") == 1.0
    problems = problem_facts(facts)
    solution = choose_solution(record, siblings) if arm == "B" else None
    if success and not problems and solution is None:
        return None
    if not success and not problems:
        raise ValueError(f"failed attempt without a fact: {record.get('id')}")
    tags = not built["untagged"]
    parts = [first_user(record)]
    if solution is not None:
        parts.append(f"{SOLUTION_HEADING}\n\n{render_solution(solution)}")
    listed = facts if problems else []
    if listed and placement in ("first", "both"):
        heading = SUCCESS_HEADING if success else FAILED_HEADING
        rendered = render_corrective(listed) if variant == "C" else render_facts(listed, tags)
        parts.append(heading + "\n\n" + "\n".join(rendered))
    parts.append(CLOSING)
    notes = defaultdict(list)
    if placement in ("notes", "both") and tags:
        for f in problems:
            if f["kind"] not in ("prohibited", "changed", "tool_error"):
                continue
            if "notes" in f:
                per_turn = f["notes"]
            else:
                untagged = re.sub(r"\bturns? [\d, ]+: ", "", f["text"], flags=re.IGNORECASE)
                per_turn = {c["turn"]: untagged for c in f["calls"]}
            for turn, text in per_turn.items():
                if NOTE_PREFIX + text not in notes[turn]:
                    notes[turn].append(NOTE_PREFIX + text)
    return {"episode_id": record.get("id"), "task": trace_of(record)["task"]["data"]["name"], "arm": arm,
            "placement": placement, "variant": variant, "first_message": "\n\n".join(parts),
            "notes": {int(k): "\n".join(v) for k, v in notes.items()},
            "solution_episode": solution["episode_id"] if solution else None,
            "facts": [{k: f[k] for k in ("kind", "text", "turn")} for f in facts],
            "bad_spans": [{"kind": f["kind"], "turn": c["turn"], "call_index": c["call_index"], "name": c["name"]}
                          for f in problems if f["kind"] in ("prohibited", "changed", "tool_error")
                          for c in f["calls"]],
            "untagged": built["untagged"], "success": success}


def leaked_literals(context_text, record, contract=None):
    """Grader literals that appear in the context but were never observed by the attempt."""
    corpus = observed_corpus(messages_of(record))
    trace = trace_of(record)
    grader = set(literals([a.get("params") for a in trace["info"]["automationbench"]["assertions"]]))
    if contract:
        grader |= set(literals([field_values(s.get("fields")) for s in contract.get("sources", {}).values()]))
    text = normalize(context_text)
    names = normalize(" ".join(tool_names(record)))
    return sorted(v for v in grader if v not in corpus and v in text and len(v) >= 4 and v not in names)


# ======================================================================== variant D (plan D24)
#
# Every line comes from one failed native assertion, one witnessed manifest guard, or one
# tool error, and carries the id(s) it comes from. Prohibitions name the offending content
# (or, when the effect itself is forbidden, the addressee or record that makes it so); they
# never forbid an action as such. Calls are located exactly: the trace keeps a verified
# world snapshot after every tool-server invocation, so each native check is re-evaluated
# after every call and a prohibition is tied to the call that turned it from passing to
# failing. Corrections and solutions are shown only after they ran in the frozen
# environment (`sdpo_ab_execution`). Checks listed in `sdpo_ab_check_exclusions.json`
# never produce a line.

D_ORDER = ("prohibited", "changed", "missing", "tool_error")  # tool errors always after harms and gaps (B)
D_LINES = 8
D_HEADING_LIMIT = "The attempt ended at its turn limit."
EXCLUSIONS_PATH = __import__("pathlib").Path(__file__).with_name("sdpo_ab_check_exclusions.json")
ERROR_CHARS = 200


def _canonical(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def _cjson(value):
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def native_assertion(result):
    """The assertion as the task declares it, rebuilt from one recorded result."""
    return {"type": result["type"], **(result.get("params") or {})}


def native_check_id(index, result):
    """Stable id of a native check: position in the task's list, type and canonical parameters."""
    return f"native[{index}]:{result['type']}:{_cjson(result.get('params') or {})}"


def manifest_check_id(check_id):
    return f"manifest:{check_id}"


# ---------------------------------------------------------------- exclusions

def load_exclusions(path=None):
    """Curated checks that never produce feedback (hidden specs, broken goals)."""
    return json.loads(__import__("pathlib").Path(path or EXCLUSIONS_PATH).read_text())["entries"]


def excluded_native(task, result, exclusions):
    key = (result["type"], _cjson(result.get("params") or {}))
    for entry in exclusions:
        native = entry["check"].get("native")
        if entry["task"] == task and native and (native["type"], _cjson(native.get("params") or {})) == key:
            return entry
    return None


def excluded_manifest(task, check_id, exclusions):
    return next((e for e in exclusions if e["task"] == task and e["check"].get("manifest") == check_id), None)


def detect_broken_goals(records, threshold=0.95, minimum=3):
    """Candidate exclusions: manifest goals (and native checks) failing in >= threshold of the
    natively successful episodes of their task. Returned for review, never applied.

    A natively successful episode passes every scored native check by definition, so only
    manifest goals and guards can be flagged here; native checks are reported for completeness."""
    from sdpo_ab_metrics import episode_outcome, task_name

    stats = defaultdict(lambda: defaultdict(lambda: [0, 0, []]))
    successes = defaultdict(int)
    for record in records:
        outcome = episode_outcome(record)
        if outcome["success"] != 1.0:
            continue
        name = task_name(record)
        successes[name] += 1
        trace = trace_of(record)
        for index, result in enumerate(trace["info"]["automationbench"]["assertions"]):
            if result.get("excluded"):
                continue
            row = stats[name][("native", native_check_id(index, result))]
            row[1] += 1
            if not result.get("passed"):
                row[0] += 1
                row[2].append(record.get("id"))
        seen = set()
        for f in manifest_findings(record):
            if f["kind"] != "trace" or f["semantics"] == "other" or f["signal"].endswith(".coverage"):
                continue
            if not isinstance(f["value"], (int, float)) or f["status"] != "valid":
                continue
            role = "goal" if f["producer"].endswith("obligations") or "retained" in f["producer"] else (
                "guard" if f["producer"].endswith("guards") else None)
            if role is None:
                continue
            check = f["check_id"] or f["signal"]
            key = (role, check, f["signal"], json.dumps(f["candidate"]))
            if key in seen:
                continue
            seen.add(key)
            row = stats[name][("manifest", check)]
            row[1] += 1
            if f["value"] == 0:
                row[0] += 1
                row[2].append(record.get("id"))
    out = []
    for name, checks in sorted(stats.items()):
        for (source, check), (failed, total, examples) in sorted(checks.items()):
            if successes[name] >= minimum and total and failed / total >= threshold:
                out.append({"task": name, "check": {source: check}, "kind": "broken_goal",
                            "reason": f"fails in {failed} of {total} verdicts on {successes[name]} natively "
                                      f"successful episodes", "evidence": {"examples": sorted(set(examples))[:5]},
                            "status": "candidate"})
    return out


# ---------------------------------------------------------------- native families (R3)

def _aslist(value):
    return list(value) if isinstance(value, (list, tuple)) else [value]


def hidden_literals(record, contract, corpus):
    """D29: the grader literals this attempt never observed; a locator (subject, sender, name, id) that contains one
    of them is never shown."""
    trace = trace_of(record)
    grader = set(literals([a.get("params") for a in trace["info"]["automationbench"]["assertions"]]))
    if contract:
        grader |= set(literals([field_values(s.get("fields")) for s in contract.get("sources", {}).values()]))
    return {v for v in grader if len(v) >= 4 and not re.fullmatch(r"\d{1,3}", v) and not observable(v, corpus)}


_HOLD = re.compile(r"\b(suspended|on hold|legal hold|do not (?:close|merge|consolidate|update|send|contact|touch|"
                   r"process)|pending (?:\w+ )?dispute)\b", re.IGNORECASE)
_LIFTED = re.compile(r"\b(hold (?:was |has been )?(?:lifted|released|removed)|lifted|cleared to|no longer (?:on hold|"
                     r"suspended)|may (?:now )?(?:proceed|close|merge)|ok to (?:close|merge))\b", re.IGNORECASE)


def hold_note_pointer(say, ids):
    """D29-B: where the task data holds a note putting one of these records on hold (a sheet row or a message that
    names the record and says suspended / on hold / do not close ...). Quotes the note only if the attempt observed
    it. None when there is no such note, when several places hold one, or when another record lifts the hold."""
    if say.initial is None or not ids:
        return None
    world = say.initial.model_dump(mode="json")
    names = [normalize(i) for i in ids if i]

    def names_record(text):
        return any(re.search(r"(?<![\w-])" + re.escape(n) + r"(?![\w-])", normalize(text)) for n in names)

    found, lifted = [], False
    for row in (world.get("google_sheets") or {}).get("rows") or []:
        cells = row.get("cells") or {}
        text = " | ".join(str(v) for v in cells.values())
        if not names_record(text):
            continue
        note = next((str(v) for v in cells.values() if isinstance(v, str) and _HOLD.search(v)), None)
        if _LIFTED.search(text):
            lifted = True
        elif note:
            worksheet = row.get("worksheet_id")
            title = next((w.get("title") for w in (world.get("google_sheets") or {}).get("worksheets") or []
                          if isinstance(w, dict) and w.get("id") == worksheet), None) or worksheet
            sheet = row.get("spreadsheet_id")
            if _locator_ok(say, title) and _locator_ok(say, sheet):
                found.append((f"worksheet {json.dumps(title)} of spreadsheet {json.dumps(sheet)}", note))
    def sentences(text):
        return [x for x in re.split(r"(?<=[.!?])\s+|\n", text) if names_record(x)]

    for message in (world.get("gmail") or {}).get("messages") or []:
        tied = sentences(" ".join(str(message.get(k) or "") for k in ("subject", "body_plain", "body")))
        if any(_LIFTED.search(x) for x in tied):
            lifted = True
        elif any(_HOLD.search(x) for x in tied):  # the record and the hold in one sentence
            where = gmail_locator(message, say)
            if where:
                found.append((where, None))
    for message in (world.get("slack") or {}).get("messages") or []:
        tied = sentences(str(message.get("text") or ""))
        if tied:
            if any(_LIFTED.search(x) for x in tied):
                lifted = True
            elif any(_HOLD.search(x) for x in tied):
                channel = message.get("channel_id")
                if _locator_ok(say, channel):
                    found.append((f"a Slack message in channel {json.dumps(channel)}", None))
    places = list(dict.fromkeys(w for w, _ in found))
    if lifted or len(places) != 1:
        return None  # no note, a lifted or superseded hold, or several competing places: say nothing
    notes = [n for _, n in found if n]
    if len(notes) == 1 and observable(notes[0], say.corpus):
        return f"its hold note in {places[0]} says {json.dumps(notes[0])}"
    return f"its hold note is in {places[0]}"


def _is_own_sent(record):
    sender = str(record.get("from_") or record.get("from") or "").strip().lower()
    labels = [str(x).upper() for x in record.get("label_ids") or record.get("labels") or []]
    return sender in ("me", "") or "SENT" in labels


def _choose_source_hits(hits, world, say, needle):
    """D29b: the record to name when a value occurs in several records. The record whose own id/name IS the value
    first; else the record the task's checks bind as a source to read; else, when the candidates disagree (other
    sender or service), only a single candidate that is neither the user's own sent mail nor superseded; else
    none (the generic phrasing)."""
    if not hits:
        return hits
    def record_of(path):
        return world[path[0]][path[1]][path[2]] if isinstance(path[2], int) else {}
    first = hits[0]
    if normalize(first[1]) == needle and str(first[0][-1]) in ("id", "template_id", "name", "envelope_id", "email",
                                                               "title"):
        return [first]
    if "@" in needle:
        # an address in another email's sender field does not compete with an email whose body (or rule text)
        # names it: only the latter say who should receive something
        named = [h for h in hits if not (h[0][0] == "gmail" and len(h[0]) > 3
                                         and str(h[0][3]) in ("from_", "from", "sender"))]
        if named:
            hits = named
    keyed = OrderedDict()
    for h in hits:
        keyed.setdefault(tuple(h[0][:3]), h)
    candidates = list(keyed.values())
    if len(candidates) == 1:
        return candidates
    bound = getattr(say, "bound_ids", set())
    for h in candidates:
        record = record_of(h[0])
        if any(normalize(str(record.get(k) or "")) in bound for k in ("id", "message_id")):
            return [h]
    senders = {(h[0][0], str(record_of(h[0]).get("from_") or "")) for h in candidates}
    if len(senders) == 1:
        return candidates
    plausible = [h for h in candidates if not _is_own_sent(record_of(h[0]))
                 and not _SUPERSEDED.search(json.dumps(record_of(h[0])))]
    if len(plausible) == 1:
        return plausible
    # the current rule message among them ("Updated template selection rules" over an older reminder)
    rules = [h for h in plausible if h[0][0] == "gmail" and _RULE_SUBJECT.search(str(record_of(h[0]).get("subject")))]
    if len(rules) > 1:
        newer = [h for h in rules if re.search(r"\b(updated|new|revised|current)\b",
                                               str(record_of(h[0]).get("subject")), re.IGNORECASE)]
        rules = newer if len(newer) == 1 else rules
    if len(rules) == 1:
        return rules
    # an email over a note or record (the request refers to emails), when exactly one email holds the value
    emails = [h for h in plausible if h[0][0] == "gmail"]
    if len(emails) == 1:
        return emails
    return [((candidates[0][0][0], "_generic", None), "")]


_SHAPE_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+")


def _shape(token):
    return re.sub(r"[A-Za-z]", "a", re.sub(r"\d", "9", token.lower()))


def conflicting_literal(text, required):
    """D29b: a code in a locator with the same shape as a required literal but a different value (SCH-W02-2026
    against SCH-W05-2026, another envelope id): the locator would point at the wrong thing."""
    codes = [r for r in required if _SHAPE_TOKEN.fullmatch(r) and re.search(r"\d", r)]
    for token in _SHAPE_TOKEN.findall(str(text or "")):
        for code in codes:
            # same shape and the same leading segment (another code of the same kind), different value
            same_kind = re.split(r"[-_]", normalize(token))[0] == re.split(r"[-_]", normalize(code))[0]
            if same_kind and _shape(token) == _shape(code) and normalize(token) != normalize(code):
                return token
    return None


_RULE_SUBJECT = re.compile(r"\b(rules?|polic(?:y|ies)|guidelines?|selection|criteria|procedures?|sop)\b",
                           re.IGNORECASE)


def rule_message(say, noun, value):
    """D29b: the email stating the rule for choosing a record of this kind ("Updated template selection rules"),
    named by subject and sender, when exactly one current such email exists and its locator is safe."""
    if say.initial is None or not noun:
        return None
    messages = (say.initial.model_dump(mode="json").get("gmail") or {}).get("messages") or []
    stem = noun.lower()[:6]
    found = [m for m in messages if _RULE_SUBJECT.search(str(m.get("subject") or ""))
             and stem in str(m.get("subject") or "").lower() and not _is_own_sent(m)]
    if len(found) > 1:
        newer = [m for m in found if re.search(r"\b(updated|new|revised|current)\b", str(m.get("subject") or ""),
                                               re.IGNORECASE)]
        found = newer if len(newer) == 1 else found
    if len(found) != 1:
        return None
    where = gmail_locator(found[0], say, value)
    if not where:
        return None
    retrieved = observable(str(found[0].get("id") or "~"), say.corpus)
    return where + ("" if retrieved else ", which this attempt never opened")


def gmail_locator(message, say, value=None):
    """'the email "<subject>" from "<sender>"', dropping a part that carries a hidden value; None when both do."""
    subject, sender = message.get("subject"), message.get("from_") or message.get("from")
    subject_ok, sender_ok = _locator_ok(say, subject, value), _locator_ok(say, sender, value)
    sender_ok = sender_ok and "@" in str(sender)  # "me" (the user's own mailbox) does not locate anything
    if subject_ok and sender_ok:
        return f"the email {json.dumps(subject)} from {json.dumps(sender)}"
    if subject_ok:
        return f"the email {json.dumps(subject)}"
    if sender_ok:
        return f"an email from {json.dumps(sender)}"
    return None


def _locator_ok(say, text, value=None):
    """A locating field may be shown only if it carries neither the value it locates nor any other hidden literal."""
    if not isinstance(text, str) or not text.strip():
        return False
    plain = normalize(text)
    hidden = set(getattr(say, "hidden", ()) or ())
    if value is not None:
        hidden.add(normalize(str(value)))
    if any(h and (h in plain or (_is_number(h) and standalone_number(h, plain))) for h in hidden):
        return False
    # D29b: the literals of the same check as the value (its other required codes), never another check's
    needle = normalize(str(value)) if value is not None else None
    required = {needle} if needle else set()
    for group in getattr(say, "check_literals", ()) or ():
        if needle and needle in group:
            required |= group
    return conflicting_literal(text, required) is None


class _Say:
    """Value rule (R5): a literal is printed only if the attempt observed it."""

    def __init__(self, corpus, initial_world=None, placeholders=False):
        self.corpus = corpus
        self.initial = initial_world
        self.placeholders = placeholders  # variant D: unobserved values become tokens resolved at render time
        self.tokens = {}
        self.contested = set()
        self.recipient_variant = {}
        self.bound_ids, self.required, self.hidden, self.check_literals = set(), set(), set(), []

    def __call__(self, value):
        if self.placeholders and not isinstance(value, (list, tuple, dict)) and (
                not observable(value, self.corpus) or (_is_number(value)
                                                       and not standalone_number(value, self.corpus))):
            # a figure seen only inside a longer number ("15" of "$15,000") is not observed as itself (A)
            key = json.dumps(value, sort_keys=True)
            self.tokens.setdefault(key, value)
            return f"\x00{list(self.tokens).index(key)}\x00"
        return show(value, self.corpus)

    def all(self, values):
        return " and ".join(self(v) for v in _aslist(values))

    def where(self, spreadsheet, worksheet=None):
        text = f"spreadsheet {self(spreadsheet)}"
        return text + (f", worksheet {self(worksheet)}" if worksheet else "")

    def row(self, spreadsheet, row_id):
        """A row named by an observed cell of it (R7: never a bare row number)."""
        label = None
        if self.initial is not None:
            rows = [r for r in self.initial.google_sheets.rows if r.spreadsheet_id == spreadsheet]
            for row in rows:
                if str(row.row_id) == str(row_id):
                    # a readable cell that names this row alone (unique in its sheet), observed by the attempt
                    label = _human_label(row.cells, lambda v: observable(v, self.corpus), lambda column, value: sum(
                        r.cells.get(column) == value for r in rows if r.worksheet_id == row.worksheet_id) == 1)
                    break
        if label is None:
            return f"a row of spreadsheet {self(spreadsheet)} that this attempt never read"
        return f"the row for {self(label)} in spreadsheet {self(spreadsheet)}"


_LABEL_RANK = (("title", "name", "subject", "employee", "vendor", "event"), ("email",))


def _human_label(cells, is_seen, unique=lambda column, value: True, best_only=False):
    """The readable cell that names a record: a title, name or subject first, then an email; never a link, path
    or id when a readable cell was observed."""
    def rank(column):
        words_in = set(re.split(r"[^a-z]+", column.lower()))
        for position, words in enumerate(_LABEL_RANK):
            if words_in & set(words):
                return position
        return len(_LABEL_RANK)

    def plain(value):
        return not (value.startswith(("/", "http")) or re.fullmatch(r"[a-z]+_[\w-]+|\d+", value))

    usable = [(c, v) for c, v in cells.items() if isinstance(v, str) and len(v.strip()) >= 2 and is_seen(v)
              and unique(c, v)]
    usable.sort(key=lambda cv: (not plain(cv[1]), rank(cv[0])))
    if best_only:
        usable = [cv for cv in usable if rank(cv[0]) == 0 and plain(cv[1])]
    return usable[0][1] if usable else None


def _req(text, target=None, group=None):
    """A required end state; `group` = (head, clause) lets several requirements on one target share a line."""
    return {"mode": "require", "text": text, "target": target, "group": group}


def _forbid(text, offending, offending_kind, kind="prohibited", target=None, probe=None, absent=None):
    return {"mode": "forbid", "kind": kind, "text": text, "offending": [v for v in offending if v not in (None, "")],
            "offending_kind": offending_kind, "target": target, "probe": probe, "absent": absent}


def _subject(p, say, key="subject"):
    value = p.get(key) or p.get("subject_contains")
    return f" with {say(value)} in its subject" if value else ""


def _remove_text(values, say, channel, target_label, verb="remove"):
    return f"{verb} {say.all(values)} from the {channel} {target_label}"


SALESFORCE_OBJECTS = {"opportunities": "opportunity", "contacts": "contact", "leads": "lead", "accounts": "account",
                      "Opportunity": "opportunity", "Contact": "contact", "Lead": "lead", "Account": "account"}


def _sf_object(p):
    name = p.get("collection") or p.get("object_type")
    if name not in SALESFORCE_OBJECTS:
        raise ValueError(f"unknown Salesforce object {name!r}")
    return SALESFORCE_OBJECTS[name]


def _action(app, p, say, negative):
    key, params = p["action_key"], dict(p.get("params") or {})
    shapes = {("airtable", "add_comment"): ({"applicationId", "recordId", "tableName", "commentText_contains"},
                                            {"applicationId", "recordId", "tableName"}),
              ("asana", "create_task"): ({"name_contains"},),
              ("calendly", "book_meeting"): ({"name", "invitee_email", "duration"}, {"name"}),
              ("notion", "create_page"): ({"title"}, {"parent_page", "title"}),
              ("trello", "card"): ({"name"},),
              ("zoom", "create_meeting"): ({"topic", "duration"}, {"topic"})}
    if (app, key) not in shapes or set(params) not in shapes[(app, key)]:
        raise ValueError(f"unknown native family: {app}_action {key} {sorted(params)}")
    if app == "airtable":
        where = f"record {say(params['recordId'])} (base {say(params['applicationId'])}, table {say(params['tableName'])})"
        if negative:
            return _forbid(f"add no Airtable comment to {where}", [params["recordId"]], "addressee")
        return _req(f"an Airtable comment on {where} containing {say(params['commentText_contains'])}")
    if app == "asana":
        if negative:
            return _forbid(f"create no Asana task whose name contains {say(params['name_contains'])}",
                           [params["name_contains"]], "content")
        return _req(f"an Asana task whose name contains {say(params['name_contains'])}")
    if app == "calendly":
        if negative:
            return _forbid(f"book no Calendly meeting whose event name contains {say(params['name'])}",
                           [params["name"]], "content")
        return _req(f"a Calendly booking of an event whose name contains {say(params['name'])}, for invitee "
                    f"{say(params['invitee_email'])}, lasting {say(params['duration'])} minutes")
    if app == "notion":
        under = f" under page {say(params['parent_page'])}" if "parent_page" in params else ""
        return _req(f"a Notion page whose title contains {say(params['title'])}{under}")
    if app == "trello":
        return _req(f"a Trello card whose name contains {say(params['name'])}")
    if negative:
        return _forbid(f"create no Zoom meeting with the topic {say(params['topic'])}", [params["topic"]], "content")
    return _req(f"a Zoom meeting with the topic {say(params['topic'])} lasting {say(params['duration'])} minutes")


def _row_criteria(p, say):
    if "cell_contains" in p:
        value = p["cell_contains"]
        if isinstance(value, dict):
            return "whose " + ", ".join(f"{say(k)} contains {say(v)}" for k, v in value.items()), list(value.values())
        return f"containing {say.all(value)}", _aslist(value)
    parts, values = [], []
    if "column" in p:
        parts.append(f"{say(p['column'])} is {say(p['value'])}")
        values.append(p["value"])
    for k, v in (p.get("cells") or {}).items():
        parts.append(f"{say(k)} is {say(v)}")
        values.append(v)
    return "whose " + ", ".join(parts), values


def _sender(p, s):
    """A Gorgias message by sender type; an agent message is the attempt's own reply (the ticket tools post as
    the agent), so it is named by role rather than by the schema value."""
    if p.get("sender_type") == "agent":
        return "a reply"
    return f"a message from sender type {s(p['sender_type'])}"


def _t(keysets, fn):
    return {"keys": [frozenset(k) for k in keysets], "render": fn}


NATIVE_FAMILIES = {
    # e-mail
    "gmail_message_sent_to": _t([{"to"}, {"to", "subject"}], lambda p, s: _req(
        f"an email sent to {s(p['to'])}{_subject(p, s)}", ("email", p["to"]),
        (f"an email sent to {s(p['to'])}", _subject(p, s).strip()))),
    "gmail_message_sent_to_with_body_contains": _t(
        [{"to", "body_contains"}, {"to", "body_contains", "subject"}, {"to", "body_contains", "subject_contains"}],
        lambda p, s: _req(f"an email to {s(p['to'])}{_subject(p, s)} whose body contains {s.all(p['body_contains'])}",
                          ("email", p["to"]),
                          (f"an email sent to {s(p['to'])}",
                           f"{_subject(p, s).strip() + ' ' if _subject(p, s) else ''}whose body contains "
                           f"{s.all(p['body_contains'])}"))),
    "gmail_message_body_contains": _t([{"to", "body_contains"}], lambda p, s: _req(
        f"an email to {s(p['to'])} whose body contains {s(p['body_contains'])}", ("email", p["to"]),
        (f"an email sent to {s(p['to'])}", f"whose body contains {s(p['body_contains'])}"))),
    "gmail_message_sent": _t(
        [{"to", "subject_contains", "body_contains"}, {"to", "body_contains"}, {"body_contains"},
         {"to", "subject_contains"}, {"subject_contains"}],
        lambda p, s: _req("an email" + (f" to {s(p['to'])}" if p.get("to") else "")
                          + (f" with {s(p['subject_contains'])} in its subject" if p.get("subject_contains") else "")
                          + (f" whose body contains {s.all(p['body_contains'])}" if p.get("body_contains") else ""),
                          ("email", p["to"]) if p.get("to") else None,
                          (f"an email sent to {s(p['to'])}",
                           " ".join(x for x in ((f"with {s(p['subject_contains'])} in its subject"
                                                 if p.get("subject_contains") else ""),
                                                (f"whose body contains {s.all(p['body_contains'])}"
                                                 if p.get("body_contains") else "")) if x))
                          if p.get("to") else None)),
    "gmail_email_sent_count": _t([{"count"}], lambda p, s: _req(
        f"exactly {s(p['count'])} sent email(s) in total, no more and no fewer")),
    "gmail_message_sent_to_with_body_not_contains": _t([{"to", "body_not_contains"}], lambda p, s: _forbid(
        _remove_text([p["body_not_contains"]], s, "email", f"to {s(p['to'])}"), [p["body_not_contains"]], "content",
        target=("email", p["to"]),
        probe={"type": "gmail_message_not_sent_to_with_body_contains", "to": p["to"],
               "body_contains": [p["body_not_contains"]]},
        absent=({"type": "gmail_message_sent_to", "to": p["to"]}, f"an email sent to {s(p['to'])}"))),
    "gmail_message_not_sent_to_with_body_contains": _t([{"to", "body_contains"}], lambda p, s: _forbid(
        _remove_text(_aslist(p["body_contains"]), s, "email", f"to {s(p['to'])}"), _aslist(p["body_contains"]),
        "content", target=("email", p["to"]))),
    "gmail_message_not_sent_with_body": _t([{"to", "body_contains"}], lambda p, s: _forbid(
        _remove_text([p["body_contains"]], s, "email", f"to {s(p['to'])}"), [p["body_contains"]], "content",
        target=("email", p["to"]))),
    "gmail_message_not_sent": _t([{"to", "body_contains"}], lambda p, s: _forbid(
        _remove_text([p["body_contains"]], s, "email", f"to {s(p['to'])}"), [p["body_contains"]], "content",
        target=("email", p["to"]))),
    "gmail_message_not_sent_to": _t([{"to"}, {"to", "subject"}], lambda p, s: (
        _forbid(f"send {s(p['to'])} no email with {s(p['subject'])} in its subject", [p["subject"]], "content",
                target=("email", p["to"]))
        if p.get("subject") else
        _forbid(f"send nothing to {s(p['to'])}: this recipient must not receive any email in this task",
                [p["to"]], "addressee"))),
    "gmail_email_not_sent_to": _t([{"to"}], lambda p, s: _forbid(
        f"send nothing to {s(p['to'])}: this recipient must not receive any email in this task", [p["to"]],
        "addressee")),
    # Slack
    "slack_message_exists": _t([{"channel_name", "text_contains"}], lambda p, s: _req(
        f"a Slack message in channel {s(p['channel_name'])} containing {s.all(p['text_contains'])}",
        ("slack", p["channel_name"]), (f"a Slack message in channel {s(p['channel_name'])}",
                                       f"containing {s.all(p['text_contains'])}"))),
    "slack_message_in_channel": _t([{"channel", "text_contains"}, {"channel_name", "text_contains"}],
                                   lambda p, s: _req(
        f"a Slack message in channel {s(p.get('channel') or p.get('channel_name'))} containing "
        f"{s.all(p['text_contains'])}", ("slack", p.get("channel") or p.get("channel_name")),
        (f"a Slack message in channel {s(p.get('channel') or p.get('channel_name'))}",
         f"containing {s.all(p['text_contains'])}"))),
    "slack_message_sent_to_channel": _t([{"channel_name", "text_contains"}], lambda p, s: _req(
        f"a Slack message in channel {s(p['channel_name'])} containing {s.all(p['text_contains'])}",
        ("slack", p["channel_name"]), (f"a Slack message in channel {s(p['channel_name'])}",
                                       f"containing {s.all(p['text_contains'])}"))),
    "slack_message_not_exists": _t([{"channel_name", "text_contains"}], lambda p, s: _forbid(
        _remove_text([p["text_contains"]], s, "Slack message", f"in channel {s(p['channel_name'])}"),
        [p["text_contains"]], "content", target=("slack", p["channel_name"]))),
    "slack_message_not_in_channel": _t([{"channel_name", "text_contains"}], lambda p, s: _forbid(
        _remove_text([p["text_contains"]], s, "Slack message", f"in channel {s(p['channel_name'])}"),
        [p["text_contains"]], "content", target=("slack", p["channel_name"]))),
    "slack_channel_exists": _t([{"channel_name"}], lambda p, s: _req(
        f"a Slack channel named {s(p['channel_name'])}")),
    # Google Sheets
    "google_sheets_row_exists": _t(
        [{"cell_contains", "spreadsheet_id"}, {"cell_contains", "spreadsheet_id", "worksheet_id"},
         {"cells", "column", "spreadsheet_id", "value", "worksheet_id"}, {"cells", "spreadsheet_id", "worksheet_id"},
         {"column", "spreadsheet_id", "value", "worksheet_id"}],
        lambda p, s: _req(f"a row in {s.where(p['spreadsheet_id'], p.get('worksheet_id'))} "
                          f"{_row_criteria(p, s)[0]}", ("sheet", p["spreadsheet_id"]),
                          (f"rows in {s.where(p['spreadsheet_id'], p.get('worksheet_id'))}:",
                           f"one {_row_criteria(p, s)[0]}"))),
    "google_sheets_row_not_exists": _t(
        [{"cell_contains", "spreadsheet_id"}, {"cell_contains", "spreadsheet_id", "worksheet_id"},
         {"column", "spreadsheet_id", "value", "worksheet_id"}],
        lambda p, s: _forbid((f"do not log a row for {s(p['value'])} in " if "column" in p else
                              f"do not add a row for {s(p['cell_contains'])} in "
                              if isinstance(p.get("cell_contains"), str) else
                              f"do not put a row {_row_criteria(p, s)[0]} in ")
                             + f"{s.where(p['spreadsheet_id'], p.get('worksheet_id'))}", _row_criteria(p, s)[1],
                             "content", target=("sheet", p["spreadsheet_id"]))),
    "google_sheets_row_updated": _t([{"cell_contains", "row_id", "spreadsheet_id"}], lambda p, s: _req(
        f"{s.row(p['spreadsheet_id'], p['row_id'])} updated so that {_row_criteria(p, s)[0][len('whose '):]}")),
    "google_sheets_row_not_updated": _t([{"row_id", "spreadsheet_id"}, {"cell_contains", "row_id", "spreadsheet_id"}],
                                        lambda p, s: (
        _forbid(f"{s.row(p['spreadsheet_id'], p['row_id'])} must not be changed so that "
                f"{_row_criteria(p, s)[0][len('whose '):]}", _row_criteria(p, s)[1], "content", kind="changed")
        if "cell_contains" in p else
        _forbid(f"{s.row(p['spreadsheet_id'], p['row_id'])} must stay unchanged", [], "record", kind="changed"))),
    "google_sheets_row_count": _t([{"count", "spreadsheet_id", "worksheet_id"}], lambda p, s: _req(
        f"exactly {s(p['count'])} rows in {s.where(p['spreadsheet_id'], p['worksheet_id'])}")),
    # support desks
    "gorgias_ticket_exists": _t([{"status", "ticket_id"}], lambda p, s: _req(
        f"Gorgias ticket {s(p['ticket_id'])} with status {s(p['status'])}")),
    "gorgias_ticket_not_exists": _t([{"status", "ticket_id"}], lambda p, s: _forbid(
        f"Gorgias ticket {s(p['ticket_id'])} must not be set to status {s(p['status'])}", [p["ticket_id"]],
        "record")),
    "gorgias_ticket_has_message": _t([{"body_contains", "sender_type", "ticket_id"}], lambda p, s: _req(
        f"{_sender(p, s)} on Gorgias ticket {s(p['ticket_id'])} containing {s(p['body_contains'])}")),
    "gorgias_ticket_not_has_message": _t([{"body_contains", "sender_type", "ticket_id"}], lambda p, s: _forbid(
        f"post no {_sender(p, s)[2:]} containing {s(p['body_contains'])} on Gorgias ticket {s(p['ticket_id'])}",
        [p["body_contains"]], "content")),
    "reamaze_conversation_exists": _t([{"assignee_email", "conversation_id"}], lambda p, s: _req(
        f"Re:amaze conversation {s(p['conversation_id'])} assigned to {s(p['assignee_email'])}")),
    "reamaze_conversation_not_exists": _t([{"assignee_email", "conversation_id"}], lambda p, s: _forbid(
        f"Re:amaze conversation {s(p['conversation_id'])} must not be assigned to {s(p['assignee_email'])}",
        [p["assignee_email"]], "content")),
    "reamaze_conversation_has_message": _t([{"body_contains", "conversation_id"}], lambda p, s: _req(
        f"a message on Re:amaze conversation {s(p['conversation_id'])} containing the word "
        f"{s(p['body_contains'])}")),
    "reamaze_conversation_not_has_message": _t([{"body_contains", "conversation_id"}], lambda p, s: _forbid(
        f"post no message containing {s(p['body_contains'])} on Re:amaze conversation {s(p['conversation_id'])}",
        [p["body_contains"]], "content")),
    "reamaze_conversation_has_tag": _t([{"conversation_id", "tag"}], lambda p, s: _req(
        f"Re:amaze conversation {s(p['conversation_id'])} tagged {s(p['tag'])}")),
    "reamaze_conversation_not_has_tag": _t([{"conversation_id", "tag"}], lambda p, s: _forbid(
        f"Re:amaze conversation {s(p['conversation_id'])} must not be tagged {s(p['tag'])}", [p["tag"]], "content")),
    "zendesk_organization_field_contains": _t([{"field", "organization_id", "value_contains"}], lambda p, s: _req(
        f"Zendesk organization {s(p['organization_id'])} whose {s(p['field'])} contains {s(p['value_contains'])}")),
    "zendesk_organization_field_not_contains": _t([{"field", "organization_id", "value_contains"}],
                                                   lambda p, s: _forbid(
        f"Zendesk organization {s(p['organization_id'])}'s {s(p['field'])} must not contain "
        f"{s(p['value_contains'])}", [p["value_contains"]], "content")),
    "zendesk_organization_has_tag": _t([{"organization_id", "tag"}], lambda p, s: _req(
        f"Zendesk organization {s(p['organization_id'])} tagged {s(p['tag'])}")),
    "zendesk_organization_not_has_tag": _t([{"organization_id", "tag"}], lambda p, s: _forbid(
        f"Zendesk organization {s(p['organization_id'])} must not be tagged {s(p['tag'])}", [p["tag"]], "content")),
    "zendesk_ticket_exists": _t([{"group_id", "ticket_id"}], lambda p, s: _req(
        f"Zendesk ticket {s(p['ticket_id'])} in group {s(p['group_id'])}")),
    "zendesk_ticket_has_comment": _t([{"body_contains", "ticket_id"}], lambda p, s: _req(
        f"a comment on Zendesk ticket {s(p['ticket_id'])} containing {s(p['body_contains'])}")),
    "zendesk_ticket_not_has_comment": _t([{"body_contains", "ticket_id"}], lambda p, s: _forbid(
        f"add no comment containing {s(p['body_contains'])} to Zendesk ticket {s(p['ticket_id'])}",
        [p["body_contains"]], "content")),
    "zendesk_ticket_has_tag": _t([{"tag", "ticket_id"}], lambda p, s: _req(
        f"Zendesk ticket {s(p['ticket_id'])} tagged {s(p['tag'])}")),
    "zendesk_ticket_not_has_tag": _t([{"tag", "ticket_id"}], lambda p, s: _forbid(
        f"Zendesk ticket {s(p['ticket_id'])} must not be tagged {s(p['tag'])}", [p["tag"]], "content")),
    "zoho_desk_ticket_has_comment": _t([{"content_contains", "ticket_id"}], lambda p, s: _req(
        f"a comment on Zoho Desk ticket {s(p['ticket_id'])} containing {s(p['content_contains'])}")),
    "zoho_desk_ticket_not_has_comment": _t([{"content_contains", "ticket_id"}], lambda p, s: _forbid(
        f"add no comment containing {s(p['content_contains'])} to Zoho Desk ticket {s(p['ticket_id'])}",
        [p["content_contains"]], "content")),
    # CRM
    "hubspot_company_exists": _t([{"name"}, {"name", "domain"}], lambda p, s: _req(
        f"a HubSpot company named {s(p['name'])}" + (f" with domain {s(p['domain'])}" if p.get("domain") else ""))),
    "hubspot_company_not_exists": _t([{"name", "domain"}], lambda p, s: _forbid(
        f"create no HubSpot company named {s(p['name'])} with domain {s(p['domain'])}", [p["name"]], "addressee")),
    "hubspot_contact_count": _t([{"min_count"}], lambda p, s: _req(f"at least {s(p['min_count'])} HubSpot contacts")),
    "hubspot_contact_exists": _t([{"email"}], lambda p, s: _req(f"a HubSpot contact with email {s(p['email'])}")),
    "hubspot_contact_not_exists": _t([{"email"}], lambda p, s: _forbid(
        f"create no HubSpot contact for {s(p['email'])}", [p["email"]], "addressee")),
    "hubspot_contact_has_property": _t([{"email", "property", "value"}], lambda p, s: _req(
        f"HubSpot contact {s(p['email'])} with property {s(p['property'])} set to {s(p['value'])}")),
    "hubspot_ticket_exists": _t([{"subject_contains"}, {"subject_contains", "description_contains"}],
                                lambda p, s: _req(
        f"a HubSpot ticket whose subject contains {s(p['subject_contains'])}"
        + (f" and whose description contains {s(p['description_contains'])}" if p.get("description_contains") else ""),
        None, (f"a HubSpot ticket whose subject contains {s(p['subject_contains'])}",
               f"whose description contains {s(p['description_contains'])}") if p.get("description_contains") else None)),
    "salesforce_contact_exists_with_field": _t([{"additional_checks", "field", "value"}], lambda p, s: _req(
        f"a Salesforce contact with {s(p['field'])} {s(p['value'])}" + "".join(
            (f" whose {s(k[:-len('_contains')])} contains {s(v)}" if k.endswith("_contains") else f" with {s(k)} {s(v)}")
            for k, v in p["additional_checks"].items()))),
    "salesforce_contact_phone_equals": _t([{"contact_id", "phone"}], lambda p, s: _req(
        f"Salesforce contact {s(p['contact_id'])} with phone {s(p['phone'])}")),
    "salesforce_field_contains": _t([{"collection", "field", "record_id", "value"},
                                     {"field", "object_type", "record_id", "value"}], lambda p, s: _req(
        f"Salesforce {_sf_object(p)} {s(p['record_id'])} whose {s(p['field'])} contains {s(p['value'])}")),
    "salesforce_field_equals": _t([{"collection", "field", "record_id", "value"},
                                   {"field", "object_type", "record_id", "value"}], lambda p, s: _req(
        f"Salesforce {_sf_object(p)} {s(p['record_id'])} with {s(p['field'])} set to {s(p['value'])}")),
    "salesforce_field_not_changed": _t([{"collection", "expected_value", "field", "record_id"}], lambda p, s: _forbid(
        f"Salesforce {_sf_object(p)} {s(p['record_id'])}'s {s(p['field'])} must stay {s(p['expected_value'])}",
        [], "record", kind="changed")),
    "salesforce_lead_exists": _t([{"email"}], lambda p, s: _req(f"a Salesforce lead with email {s(p['email'])}")),
    "salesforce_lead_exists_with_field": _t([{"field", "value"}], lambda p, s: _req(
        f"a Salesforce lead with {s(p['field'])} {s(p['value'])}")),
    "salesforce_lead_field_contains": _t([{"email", "field", "value"}], lambda p, s: _req(
        f"Salesforce lead {s(p['email'])} whose {s(p['field'])} contains {s(p['value'])}")),
    "salesforce_lead_field_equals": _t([{"email", "field", "value"}], lambda p, s: _req(
        f"Salesforce lead {s(p['email'])} with {s(p['field'])} set to {s(p['value'])}")),
    "salesforce_lead_not_exists": _t([{"email"}], lambda p, s: _forbid(
        f"create no Salesforce lead for {s(p['email'])}", [p["email"]], "addressee")),
    "salesforce_note_exists": _t([{"body_contains", "parent_id"}], lambda p, s: _req(
        f"a Salesforce note on record {s(p['parent_id'])} containing {s(p['body_contains'])}")),
    "salesforce_note_exists_for_record": _t([{"body_contains", "parent_id"}], lambda p, s: _req(
        f"a Salesforce note on record {s(p['parent_id'])} containing {s(p['body_contains'])}")),
    "salesforce_note_not_exists": _t([{"parent_id"}], lambda p, s: _forbid(
        f"add no Salesforce note to record {s(p['parent_id'])}", [p["parent_id"]], "addressee")),
    "salesforce_task_exists_for_contact": _t([{"contact_id", "subject_contains"}], lambda p, s: _req(
        f"a Salesforce task for contact {s(p['contact_id'])} whose subject contains {s(p['subject_contains'])}")),
    "mailchimp_subscriber_exists": _t([{"email", "list_id"}], lambda p, s: _req(
        f"{s(p['email'])} subscribed to Mailchimp list {s(p['list_id'])}")),
    "mailchimp_subscriber_not_exists": _t([{"email", "list_id"}], lambda p, s: _forbid(
        f"do not subscribe {s(p['email'])} to Mailchimp list {s(p['list_id'])}", [p["email"]], "addressee")),
    "linkedin_invitation_message_contains": _t([{"profile_url", "text"}], lambda p, s: _req(
        f"a LinkedIn invitation to {s(p['profile_url'])} whose message contains {s(p['text'])}")),
    "linkedin_invitation_sent_to": _t([{"profile_id"}, {"recipient_profile_url"}], lambda p, s: _req(
        f"a LinkedIn invitation sent to {s(p.get('profile_id') or p.get('recipient_profile_url'))}")),
    "linkedin_invitation_not_sent_to": _t([{"profile_id"}, {"recipient_profile_url"}], lambda p, s: _forbid(
        f"send no LinkedIn invitation to {s(p.get('profile_id') or p.get('recipient_profile_url'))}",
        [p.get("profile_id") or p.get("recipient_profile_url")], "addressee")),
    # e-signature
    "docusign_cc_exists": _t([{"email"}], lambda p, s: _req(
        f"a DocuSign envelope with {s(p['email'])} as a CC recipient")),
    "docusign_envelope_from_template": _t([{"template_id"}], lambda p, s: _req(
        f"a DocuSign envelope made from template {s(p['template_id'])}")),
    "docusign_envelope_has_signer": _t([{"email"}, {"signer_email"}], lambda p, s: _req(
        f"a DocuSign envelope with {s(p.get('signer_email') or p.get('email'))} as a signer")),
    "docusign_envelope_not_has_signer": _t([{"signer_email"}], lambda p, s: _forbid(
        f"no DocuSign envelope may list {s(p['signer_email'])} as a signer", [p["signer_email"]], "addressee")),
    "docusign_envelope_voided": _t([{"envelope_id", "reason_contains"}], lambda p, s: _req(
        f"a void of DocuSign envelope {s(p['envelope_id'])} whose reason contains {s(p['reason_contains'])}")),
    "docusign_envelope_not_voided": _t([{"envelope_id"}], lambda p, s: _forbid(
        f"do not void DocuSign envelope {s(p['envelope_id'])}", [p["envelope_id"]], "addressee",
        kind="changed")),
    # SMS
    "twilio_sms_sent": _t([{"body_contains", "to_number"}], lambda p, s: _req(
        f"an SMS to {s(p['to_number'])} containing {s(p['body_contains'])}", ("sms", p["to_number"]),
        (f"an SMS to {s(p['to_number'])}", f"containing {s(p['body_contains'])}"))),
    "twilio_sms_not_sent": _t([{"body_contains", "to_number"}], lambda p, s: _forbid(
        _remove_text([p["body_contains"]], s, "SMS", f"to {s(p['to_number'])}"), [p["body_contains"]], "content",
        target=("sms", p["to_number"]))),
    # other apps
    "airtable_record_exists": _t([{"applicationId", "fields", "tableName"}], lambda p, s: _req(
        f"an Airtable record in table {s(p['tableName'])} (base {s(p['applicationId'])}) with "
        + ", ".join(f"{s(k)} = {s(v)}" for k, v in p["fields"].items()))),
}
for _app, _negative in (("airtable", True), ("airtable", False), ("asana", True), ("asana", False),
                        ("calendly", True), ("calendly", False), ("notion", False), ("trello", False),
                        ("zoom", True), ("zoom", False)):
    NATIVE_FAMILIES[f"{_app}_action_{'not_exists' if _negative else 'exists'}"] = _t(
        [{"action_key", "params"}], lambda p, s, _a=_app, _n=_negative: _action(_a, p, s, _n))

REQUIRED_NOTE = {"email": "the email itself is required, so keep sending it",
                 "slack": "the post in that channel is itself required, so keep posting it",
                 "sms": "the SMS itself is required, so keep sending it",
                 "sheet": "the other required rows still belong there"}


def native_spec(result, say):
    """Plain-language template for one native check; an unknown family raises ValueError (R3)."""
    family = NATIVE_FAMILIES.get(result["type"])
    params = result.get("params") or {}
    if family is None or frozenset(params) not in family["keys"]:
        raise ValueError(f"unknown native family: {result['type']} {sorted(params)}")
    spec = family["render"](params, say)
    spec["family"] = result["type"]
    return spec


# ---------------------------------------------------------------- manifest families (R3)

# Manifest checks occurring in fit-20 and transfer-20, by operator, population adapter and effect source.
# Guards are rendered when witnessed; goals are never rendered in variant D because they fail on
# natively successful episodes (S3 would be violated) and the native checks state the same goals.
GUARD_FAMILY_EFFECTS = {("gmail.messages@1", "send"): "send", ("slack.messages@1", "channel_message"): "post",
                        ("google_sheets.row_writes@1", "append"): "append",
                        ("google_sheets.row_writes@1", "update"): "row_update",
                        ("service.record_writes@1", "create"): "create",
                        ("service.record_writes@1", "update"): "update"}
GUARD_POPULATIONS = {"google_sheets.rows@1", "initial.records@1", "public.request@1"}
GOAL_FAMILIES = {
    ("effects.required_when@1", pop, src) for pop in GUARD_POPULATIONS
    for src in ("gmail.messages@1", "slack.messages@1", "google_sheets.row_writes@1", "service.record_writes@1",
                "gmail.message_reads@1", "google_sheets.reads@1", "linkedin.reads@1")} | {
    ("sheets.retained_when@1", "google_sheets.rows@1", "google_sheets.row_writes@1"),
    ("record.coverage@1", None, "salesforce.record@1"), ("record.fields_equal@1", None, "salesforce.record@1")}


def manifest_family(contract, check):
    sources = contract.get("sources", {})
    population = sources.get(check.get("population"), {}).get("adapter") if check.get("population") else None
    source = sources.get(check.get("source"), {})
    if check["operator"] == "effects.prohibited_when@1":
        if population in GUARD_POPULATIONS and (source.get("adapter"), source.get("kind")) in GUARD_FAMILY_EFFECTS:
            return ("guard", population, GUARD_FAMILY_EFFECTS[(source.get("adapter"), source.get("kind"))])
    elif (check["operator"], population, source.get("adapter")) in GOAL_FAMILIES:
        return ("goal", population, source.get("adapter"))
    raise ValueError(f"unknown manifest family: {check['operator']} {population} {source.get('adapter')} "
                     f"{source.get('kind')} ({check.get('check_id')})")


# ---------------------------------------------------------------- exact location (world after every call)

def _check(world, assertion):
    from automationbench.rubric.registry import AssertionRegistry

    try:
        return bool(AssertionRegistry.check(world, dict(assertion)))
    except Exception:  # noqa: BLE001 - an unevaluable check locates nothing
        return None


class WorldTimeline:
    """The verified world before and after every tool-server invocation of a recorded attempt."""

    def __init__(self, record):
        from automationbench_v1.capture import RAW_ACTION_KIND, trace_snapshot_store

        events = trace_of(record).get("tool_execution_events") or []
        self.store = trace_snapshot_store(events)
        self.steps = []
        for event in events:
            if event.get("source") != "tool_server" or event.get("phase") == "dispatch":
                continue
            receipt = json.loads(event["receipt_json"])
            for encoded in receipt.get("evidence_json") or ():
                envelope = json.loads(encoded)
                if isinstance(envelope, dict) and envelope.get("kind") == RAW_ACTION_KIND:
                    action = envelope["action"]
                    self.steps.append((receipt["invocation_id"], action["before_digest"], action["after_digest"]))
        self._worlds = {}

    def world(self, digest):
        if digest not in self._worlds:
            from automationbench_v1.world_codec import load_world

            text = self.store.text(digest)
            self._worlds[digest] = None if text is None else load_world(json.loads(text))
        return self._worlds[digest]

    @property
    def initial(self):
        return self.world(self.steps[0][1]) if self.steps else None

    @property
    def final(self):
        return self.world(self.steps[-1][2]) if self.steps else None

    def flips(self, assertion):
        """Invocations after which the assertion went from passing to failing; None when unknown."""
        out = []
        for invocation, before, after in self.steps:
            wb, wa = self.world(before), self.world(after)
            if wb is None or wa is None:
                return None
            if before == after:
                continue
            if _check(wb, assertion) is True and _check(wa, assertion) is False:
                out.append(invocation)
        return out


def _find_literal(arguments, literal):
    """The exact text of a literal inside a call's arguments (case-insensitive; digit grouping ignored)."""
    if literal in (None, ""):
        return None
    needle = str(literal)
    for value in atoms(arguments):
        text = str(value)
        match = re.search(re.escape(needle), text, re.IGNORECASE)
        if match:
            return match.group(0)
        if re.search(r"\d", needle):
            pattern = r",?".join(re.escape(ch) for ch in needle.replace(",", ""))
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
    spaced = r"\s+".join(re.escape(part) for part in needle.split())
    for value in atoms(arguments):
        match = re.search(spaced, str(value), re.IGNORECASE) if spaced else None
        if match:
            return match.group(0)
    raw = json.dumps(arguments, ensure_ascii=False)
    match = re.search(re.escape(needle), raw, re.IGNORECASE)
    return match.group(0) if match else None


# ---------------------------------------------------------------- tool errors and corrections (R4)

def plain_error(error):
    """A tool error in plain words: pydantic argument errors are restated; others are quoted."""
    fields = re.findall(r"(\w+)\.str\s+Input should be a valid string \[type=string_type, input_value=([^,\]]+), "
                        r"input_type=(\w+)\]", error)
    if fields:
        parts = list(dict.fromkeys(f"{name} must be a string (it was given the {kind} {value})"
                                   for name, value, kind in fields))
        return "; ".join(parts)
    missing = re.findall(r"(\w+)\s+Field required \[type=missing", error)
    if missing:
        return "missing required argument(s): " + ", ".join(dict.fromkeys(missing))
    typed = re.findall(r"(\w+)\s+Input should be a valid (list|integer|dictionary|number)", error)
    if typed:
        return "; ".join(dict.fromkeys(f"{name} must be a {kind}" for name, kind in typed))
    text = error if len(error) <= ERROR_CHARS else error[:ERROR_CHARS] + "..."
    return f"\"{text}\""


_LISTED = re.compile(r"'([^']*?) \(([^()']+)\)'")


def correction_candidates(call):
    """Mechanical fixes of a failed call that keep the attempt's own target (each must still run, R4).

    1. an argument rejected as "not a valid string" is passed as the same value written as a string;
    2. a "not found" error that lists the valid items: the attempt's value names exactly one listed item
       (by its title, id, or "title (id)"), and that item's id (or title for title/name arguments) is used;
    3. a row given as {"row_id": n}, "row 2", "row_2", "Row 2" or "\\"2\\"" is passed as the bare row id."""
    error, args = call["error"] or "", call["arguments"]
    out = []
    typed = {name for name, _, _ in re.findall(
        r"(\w+)\.str\s+Input should be a valid string \[type=string_type, input_value=([^,\]]+), input_type=(\w+)\]",
        error)}
    if typed and all(isinstance(args.get(k), (int, float)) and not isinstance(args.get(k), bool) for k in typed):
        out.append({k: json.dumps(args[k]) for k in sorted(typed)})
    missing = re.search(r"(?:Spreadsheet|Worksheet) '([^']*)' not found", error)
    if missing:
        listed = error[missing.end():]
        items = _LISTED.findall(listed)
        bad = normalize(missing.group(1))
        hits = [(title, ident) for title, ident in items
                if bad and bad in (normalize(title), normalize(ident), normalize(f"{title} ({ident})"))]
        keys = [k for k, v in args.items() if isinstance(v, str) and normalize(v) == bad]
        if len(hits) == 1 and len(keys) == 1:
            title, ident = hits[0]
            key = keys[0]
            order = (title, ident) if ("title" in key or "name" in key) else (ident, title)
            for value in order:
                if normalize(value) != bad:
                    out.append({key: value})
    for key in ("row_id", "row"):
        value = args.get(key)
        fixed = None
        if isinstance(value, dict) and set(value) == {"row_id"}:
            fixed = str(value["row_id"])
        elif isinstance(value, str):
            match = re.fullmatch(r"\s*(?:row[ _]?(?:id[ _])?)?\\?\"?(\d+)\\?\"?\s*", value, re.IGNORECASE)
            if match and match.group(1) != value:
                fixed = match.group(1)
        if fixed is not None and "not found" in error:
            out.append({key: fixed})
    unique = []
    for change in out:
        if change not in unique and any(args.get(k) != v for k, v in change.items()):
            unique.append(change)
    return unique


_VERIFIED = {}


def _correction_key(record, call_index, corrected_args):
    return ("correction", record.get("id"), call_index, _cjson(corrected_args))


def _solution_key(task, calls, digest):
    return ("solution", task, digest, _cjson([(c["turn"], c["name"], c["arguments"]) for c in calls]))


def _initial_digest(record):
    return ((trace_of(record).get("info") or {}).get("automationbench_capture") or {}).get("initial_digest")


def _correction_script(record, call_index, corrected_args):
    calls, _, _ = episode_calls(record)
    target = calls[call_index]
    turns = []
    for c in calls[:call_index]:
        if not turns or turns[-1][0] != c["turn"]:
            turns.append((c["turn"], []))
        turns[-1][1].append({"name": c["name"], "arguments": c["arguments"]})
    if not turns or turns[-1][0] != target["turn"]:
        turns.append((target["turn"], []))
    turns[-1][1].append({"name": target["name"], "arguments": corrected_args})
    # A recorded turn that made no tool call has no world effect; scripted turns always call tools.
    return {"key": _correction_key(record, call_index, corrected_args), "task": trace_of(record)["task"]["data"]["name"],
            "turns": [calls_ for _, calls_ in turns], "initial_digest": _initial_digest(record)}


def _negative_results(record):
    return [native_assertion(r) for r in trace_of(record)["info"]["automationbench"]["assertions"]
            if not r.get("excluded")]


def _judge_correction(record, call_index, corrected_args, replay):
    """Did the corrected call run, on the attempt's own world, without a new harm? (R4)"""
    if replay is None or "error" in replay:
        return {"ok": False, "reason": (replay or {}).get("error", "not run")}
    if _initial_digest(replay) != _initial_digest(record):
        return {"ok": False, "reason": "initial world differs from the attempt's"}
    original, _, _ = episode_calls(record)
    replayed, node_turns, _ = episode_calls(replay)
    if len(replayed) != call_index + 1:
        return {"ok": False, "reason": f"replay made {len(replayed)} calls, expected {call_index + 1}"}
    for a, b in zip(original[:call_index], replayed[:call_index], strict=True):
        if a["result"] != b["result"]:
            return {"ok": False, "reason": f"prefix result differs at turn {a['turn']} ({a['name']})"}
    fixed = replayed[call_index]
    if fixed["error"]:
        return {"ok": False, "reason": f"corrected call still fails: {fixed['error'][:160]}", "result": fixed["result"]}
    try:
        parsed = json.loads(fixed["result"]) if isinstance(fixed["result"], str) else fixed["result"]
    except ValueError:
        parsed = None
    if isinstance(parsed, dict) and (parsed.get("count") == 0 or any(
            isinstance(parsed.get(k), list) and not parsed.get(k) for k in ("results", "rows", "messages", "records"))):
        return {"ok": False, "reason": "corrected call ran but returned nothing", "result": fixed["result"]}
    invocations, _ = invocation_map(replay, node_turns)
    own = {inv for inv, link in invocations.items() if link == (fixed["turn"], fixed["call_index"])}
    timeline = WorldTimeline(replay)
    broken = []
    for assertion in _negative_results(replay):
        from automationbench.rubric.registry import AssertionRegistry

        if not AssertionRegistry.is_negative(assertion["type"]):
            continue
        flips = timeline.flips(assertion)
        if flips is None:
            return {"ok": False, "reason": "replayed worlds unavailable"}
        if own & set(flips):
            broken.append(assertion["type"])
    harms = [f["signal"] for f in _guard_witnesses(replay) if f["invocation"] in own]
    if broken or harms:
        return {"ok": False, "reason": f"corrected call causes a prohibited effect ({broken + harms})"}
    return {"ok": True, "result": fixed["result"]}


def _clean_error(error):
    error = re.sub(r"\s*For further information visit \S+", "", error or "")
    return re.sub(r"\s+", " ", error).strip()


def _world_before(record, call_index):
    """(world JSON, world revision) right before the attempt's call `call_index`, from its verified snapshots."""
    calls, node_turns, _ = episode_calls(record)
    invocations, _ = invocation_map(record, node_turns)
    site_order = {(c["turn"], c["call_index"]): i for i, c in enumerate(calls)}
    timeline = WorldTimeline(record)
    if not timeline.steps:
        return None, 0
    previous, revision = timeline.steps[0][1], 0
    for invocation, before, after in timeline.steps:
        position = site_order.get(invocations.get(invocation))
        if position is None:
            return None, 0
        if position >= call_index:
            previous = before
            break
        previous = after
        revision += before != after
    text = timeline.store.text(previous)
    return (json.loads(text) if text is not None else None), revision


def _judge_correction_in_process(record, call_index, corrected_args):
    """Run the attempt's failing call and the corrected call on the attempt's own world at that point."""
    from datetime import UTC, datetime

    from automationbench.rubric.registry import AssertionRegistry
    from automationbench_v1.world_codec import load_world
    from sdpo_ab_execution import InProcessTools

    trace = trace_of(record)
    calls, _, _ = episode_calls(record)
    original = calls[call_index]
    world, revision = _world_before(record, call_index)
    if world is None:
        return {"ok": False, "reason": "world before the call unavailable"}
    tools = InProcessTools(trace["task"]["data"]["name"])
    day = datetime.fromtimestamp(trace["timing"]["start"], UTC).isoformat() if trace.get("timing") else None
    clock = tools.clock(day)
    error, _, _ = tools.call(world, clock, revision, original["name"], original["arguments"])
    if _clean_error(error) != original["error"]:
        return {"ok": False, "reason": "the in-process call does not reproduce the recorded error",
                "recorded": original["error"][:200], "reproduced": (error or "")[:200]}
    error, result, after = tools.call(world, clock, revision, original["name"], corrected_args)
    if error:
        return {"ok": False, "reason": f"corrected call still fails: {_clean_error(error)[:160]}", "result": result}
    try:
        parsed = json.loads(result)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict) and (parsed.get("count") == 0 or any(
            isinstance(parsed.get(k), list) and not parsed.get(k) for k in ("results", "rows", "messages", "records"))):
        return {"ok": False, "reason": "corrected call ran but returned nothing", "result": result}
    broken = []
    if after is not world:
        before_world, after_world = load_world(world), load_world(after)
        for assertion in _negative_results(record):
            if (AssertionRegistry.is_negative(assertion["type"]) and _check(before_world, assertion) is True
                    and _check(after_world, assertion) is False):
                broken.append(assertion["type"])
    if broken:
        return {"ok": False, "reason": f"corrected call breaks a prohibition ({broken})"}
    return {"ok": True, "result": result, "mode": "in_process", "guards_checked": False}


def verify_corrections(items, concurrency=8, mode="in_process"):
    """Verify (record, call_index, corrected_args) items; results are cached.

    mode "in_process" (default): the failing call and the corrected call run through the task's own toolset and
    MCP tool manager on the attempt's verified world before the call; the failing call must reproduce its
    recorded error. mode "replay": the attempt's calls up to that point and the corrected call run as a full
    rollout in the frozen environment (also checks manifest guards; about a minute per item)."""
    todo = {}
    for record, call_index, corrected_args in items:
        key = _correction_key(record, call_index, corrected_args)
        if key not in _VERIFIED and key not in todo:
            todo[key] = (record, call_index, corrected_args)
    if todo and mode == "replay":
        from sdpo_ab_execution import run_scripts

        replays = run_scripts([_correction_script(*item) for item in todo.values()], concurrency=concurrency)
        for key, item in todo.items():
            _VERIFIED[key] = {**_judge_correction(*item, replays.get(key)), "mode": "replay"}
    else:
        for key, item in todo.items():
            _VERIFIED[key] = _judge_correction_in_process(*item)
    return [_VERIFIED[_correction_key(*item)] for item in items]


def verify_correction(record, call_index, corrected_args, mode="in_process"):
    """Is the corrected call executable at that point of the attempt? `call_index` is the call's position in
    `episode_calls(record)`. Returns {"ok", "reason"?, "result"?}. ok when the corrected call returns no error
    and a non-empty result on the attempt's own world, and makes no native prohibition fail. In "in_process"
    mode the attempt's own failing call must first reproduce its recorded error there (the check that the
    in-process path is the environment's); "replay" mode re-runs the attempt as a rollout and also requires
    no manifest guard to fire on the corrected call."""
    return verify_corrections([(record, call_index, corrected_args)], mode=mode)[0]


# ---------------------------------------------------------------- solutions (S1-S3)

def _solution_script(task, calls, digest):
    turns = []
    for c in calls:
        if not turns or turns[-1][0] != c["turn"]:
            turns.append((c["turn"], []))
        turns[-1][1].append({"name": c["name"], "arguments": c["arguments"]})
    return {"key": _solution_key(task, calls, digest), "task": task, "turns": [t for _, t in turns],
            "initial_digest": digest}


def _judge_solution(task, calls, digest, replay):
    from sdpo_ab_metrics import episode_outcome

    if replay is None or "error" in replay:
        return {"ok": False, "reason": (replay or {}).get("error", "not run")}
    if digest and _initial_digest(replay) != digest:
        return {"ok": False, "reason": "initial world differs from the sibling's"}
    outcome = episode_outcome(replay)
    replayed, _, _ = episode_calls(replay)
    errors = [c["name"] for c in replayed if c["error"]]
    results = trace_of(replay)["info"]["automationbench"]["assertions"]
    scored = [r for r in results if not r.get("excluded")]
    # S1 (I): the solution must also meet every verified goal of its task in its own replay
    verified = load_verified_goals()
    goals_failed = sorted({cid for cid in _goal_failures_strict(replay)
                           if goal_verdict(verified, task, cid) == "verified"})
    violations = solution_rule_violations(replay, calls)
    ok = (outcome["success"] == 1.0 and not outcome["harm"] and not outcome["preservation_break"]
          and not errors and len(replayed) == len(calls) and not goals_failed and not violations)
    return {"ok": ok, "reason": None if ok else
            f"success={outcome['success']} harms={outcome['manifest_harms']} breaks={outcome['preservation_breaks']} "
            f"errors={errors} calls={len(replayed)}/{len(calls)} verified_goals_failed={goals_failed} "
            f"rule_violations={violations[:3]}",
            "goals_failed": goals_failed,
            "calls": [{**c, "result": r["result"]} for c, r in zip(calls, replayed)] if ok else None,
            "assertions": [{"type": r["type"], "params": r.get("params"), "passed": r.get("passed"),
                            "excluded": r.get("excluded")} for r in results],
            "scored": len(scored), "passed": sum(bool(r.get("passed")) for r in scored)}


def verify_solutions(items, concurrency=8):
    from sdpo_ab_execution import run_scripts

    todo = {}
    for task, calls, digest in items:
        key = _solution_key(task, calls, digest)
        if key not in _VERIFIED and key not in todo:
            todo[key] = (task, calls, digest)
    if todo:
        replays = run_scripts([_solution_script(*item) for item in todo.values()], concurrency=concurrency)
        for key, item in todo.items():
            _VERIFIED[key] = _judge_solution(*item, replays.get(key))
    return [_VERIFIED[_solution_key(*item)] for item in items]


def verify_solution(task, solution_calls, initial_digest=None):
    """Replay `solution_calls` (dicts with turn, name, arguments; one assistant turn per distinct turn value)
    from the task's initial world in the frozen environment. ok when every scored native check passes, no
    manifest harm and no preservation break occur, no call errors, and every verified goal of the task is met. `initial_digest` pins which of the
    environment's two initial-world encodings to use (the sibling's own); None accepts either."""
    return verify_solutions([(task, list(solution_calls), initial_digest)])[0]


def solution_variants(sibling):
    """S2 order: the cleaned solution, then the successful attempt's non-errored calls in original order."""
    cleaned = clean_sibling(sibling)
    if cleaned is None:
        return []
    calls, _, _ = episode_calls(sibling)
    plain = [c for c in calls if not c["error"]]
    # 12: a solution keeps every successful read (the counts and checks it shows come from them); only errored
    # calls and exact repeats are dropped
    seen, kept = set(), []
    for c in plain:
        key = (c["name"], json.dumps(c["arguments"], sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        kept.append(c)
    variants = [("cleaned", kept)]
    if [(c["turn"], c["call_index"]) for c in plain] != [(c["turn"], c["call_index"]) for c in kept]:
        variants.append(("non_errored", plain))
    return variants


def _solution_options(record, siblings):
    options = []
    for index, sibling in enumerate(siblings):
        if sibling.get("id") == record.get("id"):
            continue
        cleaned = clean_sibling(sibling)
        if cleaned:
            options.append((cleaned["writes"], cleaned["turns"], index, sibling))
    return [o[3] for o in sorted(options, key=lambda o: o[:3])]


def choose_solution_d(record, siblings, verification="run"):
    """The first sibling (fewest writes, turns, index) whose cleaned or plain call list replays to full success."""
    excluded = load_solution_exclusions()
    for sibling in _solution_options(record, siblings):
        task = trace_of(sibling)["task"]["data"]["name"]
        if sibling.get("id") in (excluded.get(task) or {}):
            continue  # curated: breaks an explicit rule the checks do not score (D28)
        for label, calls in solution_variants(sibling):
            key = _solution_key(task, calls, _initial_digest(sibling))
            if key not in _VERIFIED and verification == "run":
                verify_solutions([(task, calls, _initial_digest(sibling))])
            verdict = _VERIFIED.get(key)
            if verdict and verdict["ok"]:
                return {"episode_id": sibling.get("id"), "variant": label, "calls": verdict["calls"],
                        "scored": verdict["scored"], "assertions": verdict["assertions"]}
    return None


def render_solution_d(solution):
    lines = []
    for n, c in enumerate(solution["calls"], 1):
        args = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in c["arguments"].items())
        result = c["result"] if isinstance(c["result"], str) else json.dumps(c["result"])
        result = (result or "").replace("\n", " ")
        if len(result) > RESULT_CHARS:
            result = result[:RESULT_CHARS] + " [shortened]"
        lines.append(f"{n}. {c['name']}({args})\n   -> {result}")
    # D33: no grader count after the solution (R7); the model read it as a verdict on its own attempt
    return "\n".join(lines)


def prepare_verifications(records, concurrency=8):
    """Run every correction and solution replay that variant D may show for these records, in one batch
    per kind (records of several tasks may be mixed; siblings are the records of the same task)."""
    by_task = defaultdict(list)
    for record in records:
        by_task[trace_of(record)["task"]["data"]["name"]].append(record)
    corrections, solutions = [], []
    for task, group in by_task.items():
        for record in group:
            corrections.extend(_correction_items(record))
            cleaned = clean_sibling(record)
            if cleaned:
                for _, calls in solution_variants(record):
                    solutions.append((task, calls, _initial_digest(record)))
    if corrections:
        verify_corrections(corrections, concurrency)
    if solutions:
        verify_solutions(solutions, concurrency)
    return {"corrections": len(corrections), "solutions": len(solutions)}


def _error_groups(calls):
    groups = OrderedDict()
    for index, c in enumerate(calls):
        if c["error"]:
            # identical texts only, after normalising ids/numbers and quote marks (G): differing quoted names
            # ('targets' vs 'DPO') are different errors
            shape = re.sub(r"\b(?=[\w-]*\d)[\w-]+\b", "ID", re.sub(r"[\"\u2018\u2019\u201c\u201d]", "'", c["error"]))
            groups.setdefault((c["name"], shape), []).append(index)
    return groups


def _correction_items(record):
    """Every failing call of each error group with distinct arguments, with each of its candidate corrections."""
    calls, _, _ = episode_calls(record)
    items = []
    for indices in _error_groups(calls).values():
        seen = set()
        for index in indices:
            key = _cjson(calls[index]["arguments"])
            if key in seen:
                continue
            seen.add(key)
            for change in correction_candidates(calls[index]):
                items.append((record, index, {**calls[index]["arguments"], **change}))
    return items


# ---------------------------------------------------------------- guard witnesses

def _guard_witnesses(record):
    """Witnessed prohibited effects: check id, signal, invocation and the population member (candidate)."""
    out = []
    for batch in trace_of(record).get("assessment_batches") or []:
        run = batch.get("run") or {}
        if run.get("status") != "complete" or not run.get("producer_id", "").endswith("manifest_guards"):
            continue
        config = json.loads(run.get("configuration_json") or "{}")
        case = config.get("case") or {}
        for assessment in batch.get("assessments") or []:
            subject = assessment.get("subject") or {}
            if subject.get("kind") != "execution" or assessment.get("value") != 1:
                continue
            out.append({"check_id": config.get("check_id"), "signal": assessment["signal"]["signal_id"],
                        "invocation": (subject.get("execution") or {}).get("invocation_id"),
                        "candidate": case.get("candidate_identity")})
    return out


def episode_contract(record, fallback=None):
    """The manifest contract this episode was assessed with (embedded in its assessment view), so a recorded
    episode is read against its own contract even after the installed environment's contracts are repaired;
    episodes without one fall back to the given or installed contract."""
    for view in trace_of(record).get("assessment_views") or []:
        if view.get("builder_revision") == "automationbench.manifest_inputs@1" and view.get("input_json"):
            contract = json.loads(view["input_json"]).get("contract")
            if contract:
                return contract
    return fallback if fallback is not None else contract_for(trace_of(record)["task"]["data"]["name"])


def _population_rows(record):
    views = trace_of(record).get("assessment_views") or []
    for view in views:
        if view.get("builder_revision") == "automationbench.manifest_inputs@1" and view.get("input_json"):
            material = json.loads(view["input_json"])
            return json.loads(material.get("table_evidence_json") or "{}")
    return {}


def _fields(node, roots, out):
    if isinstance(node, dict):
        if node.get("kind") == "field" and node.get("path") and node["path"][0] in roots and len(node["path"]) > 1:
            out.append(node["path"][1])
        for value in node.values():
            _fields(value, roots, out)
    elif isinstance(node, list):
        for value in node:
            _fields(value, roots, out)
    return out


MEMBER_ROOTS = ("request", "member", "candidate")


def _simple_conditions(predicate):
    """The guard's condition as (field, op, literal) when it is only equalities / mentions on the member's own
    fields joined by "all"; None otherwise (then no reason is given, because a partial one could mislead)."""
    if not isinstance(predicate, dict):
        return None
    op = predicate.get("op")
    if op == "all":
        parts = [_simple_conditions(arg) for arg in predicate.get("args") or []]
        return None if not parts or any(p is None for p in parts) else [c for p in parts for c in p]
    if op == "eq":
        left, right = predicate.get("left") or {}, predicate.get("right") or {}
        if (left.get("kind") == "field" and len(left.get("path") or []) == 2 and left["path"][0] in MEMBER_ROOTS
                and right.get("kind") == "literal" and isinstance(right.get("value"), str)):
            return [(left["path"][1], "eq", right["value"])]
    if op == "mentions":
        text, value = predicate.get("text") or {}, predicate.get("value") or {}
        if (text.get("kind") == "field" and len(text.get("path") or []) == 2 and text["path"][0] in MEMBER_ROOTS
                and value.get("kind") == "literal" and isinstance(value.get("value"), str)):
            return [(text["path"][1], "mentions", value["value"])]
    return None


def _effect_literals(node, negated=False, out=None):
    """String literals the guard's effect match looks for in the effect text (positive mentions only; nothing
    under "not" or "ne", and no equality/membership literal: those qualify the effect, e.g. a Recommendation
    value, and naming them would forbid allowed content)."""
    out = [] if out is None else out
    if isinstance(node, list):
        for item in node:
            _effect_literals(item, negated, out)
        return out
    if not isinstance(node, dict):
        return out
    op = node.get("op")
    if op in ("not", "ne"):
        return out
    if op == "mentions" and not negated:
        value = node.get("value") or {}
        if value.get("kind") == "literal":
            out += [v for v in _aslist(value.get("value")) if isinstance(v, str)]
    for key, value in node.items():
        if key not in ("excluding",):
            _effect_literals(value, negated, out)
    return out


ID_KEYS = re.compile(r"(^|_)(id|ids|to|cc|bcc|email|channel|record|recipient|to_number|phone)$", re.IGNORECASE)


def _find_in_args(arguments, literal):
    """(argument key, exact text) of a literal inside a call's top-level arguments, or None."""
    for key, value in arguments.items():
        found = _find_literal({key: value}, literal)
        if found:
            return key, found
    return None


VERIFIED_GOALS_PATH = __import__("pathlib").Path(__file__).with_name("sdpo_ab_verified_goals.json")


def _goal_findings(record):
    """Trace-level manifest goal verdicts: (check_id, candidate identity, value)."""
    out = []
    for f in manifest_findings(record):
        if (f["producer"].endswith("obligations") and f["kind"] == "trace" and f["semantics"] != "other"
                and not f["signal"].endswith(".coverage") and f["status"] == "valid"
                and isinstance(f["value"], (int, float))):
            out.append((f["check_id"] or f["signal"], f["candidate"], f["value"]))
    return out


SOLUTION_EXCLUSIONS_PATH = __import__("pathlib").Path(__file__).with_name("sdpo_ab_solution_exclusions.json")
_STOP = {"the", "and", "has", "have", "been", "for", "with", "your", "you", "our", "this", "that", "was", "are",
         "is", "be", "to", "of", "a", "an", "in", "on", "we", "will"}


def load_solution_exclusions(path=None):
    path = path or SOLUTION_EXCLUSIONS_PATH
    return json.loads(__import__("pathlib").Path(path).read_text()) if __import__("pathlib").Path(path).exists() else {}


def _sent_texts(calls):
    """(recipients, text) of everything a call list sends: emails, posts, messages, notes, comments."""
    out = []
    for c in calls:
        kind = _call_kind(c["name"])
        if kind not in ("send", "message", "create", "update") or c.get("error"):
            continue
        texts = [str(v) for k in ("subject", "body", "body_text", "body_html", "text", "message", "content",
                                  "note", "comment", "description") for v in _arg_values(c["arguments"], k)]
        if texts:
            recipients = " ".join(str(v) for k in ("to", "cc", "bcc", "email") for v in _arg_values(c["arguments"], k))
            out.append((recipients, "\n".join(texts)))
    return out


def _stems(phrase):
    return [w.lower()[:5] for w in re.findall(r"[A-Za-z]{3,}", phrase) if w.lower() not in _STOP]


def _initial_world_of(record):
    try:
        initial = WorldTimeline(record).initial
        return initial.model_dump(mode="json") if initial is not None else None
    except Exception:  # noqa: BLE001 - no world snapshots in this record
        return None


def _record_name_fields(item):
    fields = {k: v for k, v in item.items() if isinstance(v, str)}
    first, last = item.get("first_name") or item.get("firstName"), item.get("last_name") or item.get("lastName")
    if isinstance(first, str) and isinstance(last, str):
        fields["name"] = f"{first} {last}"
    return fields


def _record_identities(item):
    """The values that identify a record: its id, email, and name."""
    fields = _record_name_fields(item)
    return [v for k, v in fields.items() if len(v) >= 3 and (
        "@" in v or str(k).lower() in ("id", "email", "name", "full_name", "employee", "contact_name")
        or re.search(r"(^|[ _])id$", str(k).lower()))]


_EXCLUSION = re.compile(r"\b(exclud\w*|skip(?:s|ped|ping)?|omit\w*|not included|reject\w*|ineligible|"
                        r"not (?:yet )?eligible|no change|unchanged|remain(?:s|ed|ing)?)\b", re.IGNORECASE)


def protected_names(cells, sent, out):
    """Flag sent text naming a record (by name, email or id) that must stay unnamed, except to its own address."""
    names = [v for k, v in cells.items() if isinstance(v, str) and len(v) >= 4 and (
        "@" in v or str(k).lower() in ("name", "employee", "employee name", "full name", "vendor", "customer",
                                       "registrant", "contact", "title", "contact name")
        or re.search(r"(^|[ _])id$", str(k).lower()))]
    emails = [v for v in names if "@" in v]
    for recipients, text in sent:
        if any(normalize(e) in normalize(recipients) for e in emails):
            continue  # a message to that record itself
        for label in names:
            if re.search(r"(?<!\w)" + re.escape(normalize(label)) + r"(?!\w)", normalize(text)):
                message = f"names protected record {label!r}"
                if message not in out:
                    out.append(message)


def solution_rule_violations(replay, calls):
    """11: rules a solution must keep beyond the scored checks. (a) its sent text never names a record a guard of the
    task protects (held, excluded, ineligible) except to that record's own address; (b) it never states a phrase a
    check forbids, in any inflection or order within one sentence."""
    out = []
    sent = _sent_texts(calls)
    from automationbench_v1.contracts.predicates import _evaluate, parse_predicate

    try:
        material = _guard_material(replay)
    except Exception:  # noqa: BLE001 - no guard material: nothing protected can be named
        material = None
    contract = episode_contract(replay) or {}
    guards = [c for c in contract.get("checks", []) if c.get("role") == "harm" and c.get("prohibited_when")]
    # r5-3: records no applicable goal asks anything for (inapplicable everywhere: skipped, excluded, unchanged)
    statuses = defaultdict(set)
    for f in manifest_findings(replay):
        if f["producer"].endswith("obligations") and f["kind"] == "trace" and f.get("candidate") \
                and not f["signal"].endswith(".coverage"):
            statuses[json.dumps(list(f["candidate"]))].add(f["status"])
    untouched = {k for k, v in statuses.items() if v == {"inapplicable"}}
    try:
        goal_material = _goal_material(replay) if untouched else None
    except Exception:  # noqa: BLE001 - no obligation material
        goal_material = None
    for population in (goal_material[1].values() if goal_material else ()):
        for row in population.rows:
            if json.dumps(list(row.identity)) in untouched:
                protected_names(json.loads(row.cells_json), sent, out)
    # r5-3: records of the task's data that no check or request names, in a collection whose other records the
    # task does concern (the champion who did not move, the contacts left out): the solution does not name them
    # r6: no exclusion statement about task items (the system prompt forbids naming, enumerating or explaining
    # skipped, excluded or rejected items), except in a message to that person ("you are not yet eligible")
    world_people = {normalize(v) for _, v in _strings_in(((_initial_world_of(replay) or {}).get("google_sheets")) or {})
                    if isinstance(v, str) and "@" in v}
    for recipients, text in sent:
        addresses = [normalize(r) for r in re.split(r"[,;\s]+", recipients) if "@" in r]
        if len(addresses) == 1 and addresses[0] in world_people:
            continue  # a message to one person of the task's records (their own not-yet-eligible note)
        single = len(addresses) == 1
        for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
            m = _EXCLUSION.search(sentence)
            if m and not (single and re.search(r"\b(you|your)\b", sentence, re.IGNORECASE)):
                message = f"states an exclusion ({m.group(0)!r}): {sentence.strip()[:80]!r}"
                if message not in out:
                    out.append(message)
    results = trace_of(replay)["info"]["automationbench"]["assertions"]
    required_results = [r for r in results if not negative_family(r["type"])]
    referenced = {normalize(a) for a in atoms([r.get("params") for r in required_results]) if len(str(a)) >= 3}
    referenced |= {normalize(a) for a in atoms([c for c in contract.get("checks", []) if c.get("role") == "goal"])
                   if len(str(a)) >= 3}
    for population in (goal_material[1].values() if goal_material else ()):
        for row in population.rows:
            referenced |= {normalize(a) for a in atoms(json.loads(row.cells_json)) if len(str(a)) >= 3}
    world = _initial_world_of(replay)
    for service in (world or {}).values():
        for collection in (service.values() if isinstance(service, dict) else ()):
            items = [i for i in (collection if isinstance(collection, list) else ()) if isinstance(i, dict)]
            idents = [_record_identities(i) for i in items]
            if not any(set(map(normalize, ids)) & referenced for ids in idents):
                continue
            for item, ids in zip(items, idents):
                # people and accounts with an address only: a company or a ticket named in passing is not a record
                # the task left out
                if ids and any("@" in v for v in ids) and not set(map(normalize, ids)) & referenced:
                    protected_names({k: v for k, v in _record_name_fields(item).items()}, sent, out)
    if material:
        for name, population in material[1].items():
            conditions = []
            for check in guards:
                if check.get("population") == name:
                    try:
                        conditions.append(parse_predicate(check["prohibited_when"]))
                    except Exception:  # noqa: BLE001
                        continue
            if not conditions and not any(json.dumps(list(r.identity)) in untouched for r in population.rows):
                continue
            for row in population.rows:
                cells = json.loads(row.cells_json)
                context = {"request": cells, "member": cells}
                if json.dumps(list(row.identity)) in untouched:
                    protected_names(cells, sent, out)
                    continue
                protected = False
                for condition in conditions:
                    try:
                        protected = protected or _evaluate(condition, context).value is True
                    except Exception:  # noqa: BLE001 - a condition over other roots: not decidable here
                        continue
                # a record some goal still requires an action for is not skipped: only records the task holds,
                # excludes or skips entirely (a guard covers them, no goal requires anything) must stay unnamed
                for check in contract.get("checks", []):
                    if not protected:
                        break
                    if check.get("role") == "goal" and check.get("population") == name and check.get("required_when"):
                        try:
                            if _evaluate(parse_predicate(check["required_when"]), context).value is not False:
                                protected = False  # required, or undecidable here: not provably skipped
                        except Exception:  # noqa: BLE001 - undecidable: not provably skipped
                            protected = False
                if not protected:
                    continue
                protected_names(cells, sent, out)
    for result in trace_of(replay)["info"]["automationbench"]["assertions"]:
        try:
            spec = native_spec(result, _Say(""))
        except ValueError:
            continue
        if spec["mode"] != "forbid" or spec.get("offending_kind") != "content":
            continue
        for phrase in spec["offending"]:
            if not re.fullmatch(r"[A-Za-z][A-Za-z' ]+", str(phrase).strip()):
                continue  # an address, id or figure is the scored check's own business, not a phrase
            wanted = [w.lower() for w in re.findall(r"[A-Za-z]{3,}", str(phrase)) if w.lower() not in _STOP]
            if len(wanted) < 2:
                continue  # a single word is the scored check's own business
            def same_word(token, word):
                # the same word in another inflection, keeping its form: "processed" ~ "processed", not "processing"
                if not token.startswith(word[:5]):
                    return False
                for suffix in ("ed", "ing"):
                    if word.endswith(suffix) != token.endswith(suffix):
                        return False
                return True
            target = (result.get("params") or {}).get("to")
            for recipients, text in sent:
                if target and not _find_literal(recipients, target):
                    continue  # the check forbids the phrase towards its own addressee only
                for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
                    tokens = [w.lower() for w in re.findall(r"[A-Za-z]{3,}", sentence)]
                    # the phrase's words in order, a few words apart at most
                    for start in range(len(tokens)):
                        position, ok = start, True
                        for word in wanted:
                            hit = next((i for i in range(position, min(len(tokens), position + 4))
                                        if same_word(tokens[i], word)), None)
                            if hit is None:
                                ok = False
                                break
                            position = hit + 1
                        if ok:
                            out.append(f"states forbidden phrase {phrase!r}: {sentence.strip()[:80]!r}")
                            break
    return list(dict.fromkeys(out))


def _goal_failures_strict(record):
    """Goals without a VALID pass wherever they are required: a valid 0, or an abstained verdict (the effect's
    scope could not be judged) counts as a failure for qualifying a solution (S1/S2). Inapplicable rows are not
    required."""
    out = set()
    for f in manifest_findings(record):
        if not (f["producer"].endswith("obligations") and f["kind"] == "trace" and f["semantics"] != "other"
                and not f["signal"].endswith(".coverage")):
            continue
        cid = f["check_id"] or f["signal"]
        if f["status"] == "abstained" or (f["status"] == "valid" and f["value"] != 1):
            out.add(cid)
    return out


REPAIR_SOLUTIONS = "docs/research/evidence/sdpo-ab-task-repair-20261028/v1_solution_episodes.jsonl.gz"


GOAL_OVERRIDES_PATH = __import__("pathlib").Path(__file__).with_name("sdpo_ab_goal_overrides.json")


def verify_goals(episode_globs, out=VERIFIED_GOALS_PATH, solution_globs=(REPAIR_SOLUTIONS,), overrides=None):
    """{task: {goal_id: {"verdict": "verified" | "unverified" | "unverifiable", "evidence": [episode ids],
    "failed_in": [episode ids]}}} from recorded episodes.

    Evidence is every natively successful episode with no manifest harm of the task, from `episode_globs`
    plus the replay-verified constructed correct runs in `solution_globs` (one per repaired task, replayed in
    the repaired environment; they are held to the same test). A goal is verified when it has at least one
    valid verdict in that evidence and passes in all of it, or when every evidence episode it fails in is listed
    under `accepted_failures` in the overrides (`sdpo_ab_goal_overrides.json`: a natively successful attempt
    that truly violates the goal's intent); unverified when it fails in any other; unverifiable when no evidence
    gives it a verdict. Run it on episodes from the environment whose goals are to be rendered
    (the repaired one), never on pre-repair episodes."""
    import glob
    import gzip
    from pathlib import Path

    from sdpo_ab_metrics import episode_outcome, task_name

    goals = defaultdict(dict)
    paths = [p for pattern in [*_aslist(episode_globs), *_aslist(solution_globs or ())]
             for p in sorted(glob.glob(pattern, recursive=True))]
    solution_paths = {p for pattern in _aslist(solution_globs or ()) for p in glob.glob(pattern, recursive=True)}
    for path in dict.fromkeys(paths):
        for line in gzip.open(path, "rt") if path.endswith(".gz") else open(path):
            if not line.strip():
                continue
            record = json.loads(line)
            if "traces" not in record and isinstance(record.get("record"), dict):
                record = record["record"]  # replay-verified correct runs are wrapped as {key, task, record}
            name = task_name(record)
            contract = episode_contract(record)
            for check in (contract or {}).get("checks", []):
                if check.get("role") == "goal":
                    goals[name].setdefault(check["check_id"], {"evidence": [], "failed_in": []})
            outcome = episode_outcome(record)
            if outcome["success"] != 1.0 or outcome["harm"]:
                continue
            natural = path not in solution_paths
            seen = set()
            for check_id, _, value in _goal_findings(record):
                row = goals[name].setdefault(check_id, {"evidence": [], "failed_in": []})
                row.setdefault("natural", [])
                if record.get("id") not in row["evidence"]:
                    row["evidence"].append(record.get("id"))
                    if natural:
                        row["natural"].append(record.get("id"))
                if value != 1 and (check_id, record.get("id")) not in seen:
                    seen.add((check_id, record.get("id")))
                    row["failed_in"].append(record.get("id"))
    if overrides is None:
        overrides = json.loads(GOAL_OVERRIDES_PATH.read_text()) if GOAL_OVERRIDES_PATH.exists() else {}
    result = {}
    for name, checks in sorted(goals.items()):
        result[name] = {}
        for check_id, row in sorted(checks.items()):
            accepted = set(((overrides.get(name) or {}).get(check_id) or {}).get("accepted_failures") or [])
            open_failures = [e for e in row["failed_in"] if e not in accepted]
            # constructed correct runs can expose a failure but cannot verify a goal on their own
            verdict = ("unverified" if open_failures else "unverifiable" if not row.get("natural")
                       else "verified")
            result[name][check_id] = {"verdict": verdict, "evidence": row["evidence"], "failed_in": row["failed_in"],
                                      "natural_evidence": row.get("natural", []),
                                      "accepted_failures": sorted(set(row["failed_in"]) & accepted)}
    if out:
        Path(out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    return result


def goal_verdict(goals, task, check_id):
    value = (goals.get(task) or {}).get(check_id)
    return value.get("verdict") if isinstance(value, dict) else value


def load_verified_goals(path=None):
    path = __import__("pathlib").Path(path or VERIFIED_GOALS_PATH)
    return json.loads(path.read_text()) if path.exists() else {}


def _goal_terms(node, terms, where, under_choice=False):
    """Requirements stated by a goal's effect match, outside any "any"/"not" branch:
    terms (what the effect must mention) and where (recipient, channel)."""
    if isinstance(node, list):
        for item in node:
            _goal_terms(item, terms, where, under_choice)
        return
    if not isinstance(node, dict):
        return
    op = node.get("op")
    if op in ("any", "not", "ne"):
        return
    if op == "in":
        left, right = node.get("left") or {}, node.get("right") or {}
        path = (right.get("path") or [None, None]) if right.get("kind") == "field" else [None, None]
        if path[0] == "effect" and path[1] in ("recipients", "to"):
            where.setdefault("recipient", left)
            return
    if op == "eq":
        left, right = node.get("left") or {}, node.get("right") or {}
        for a, b in ((left, right), (right, left)):
            path = (a.get("path") or [None, None]) if a.get("kind") == "field" else [None, None]
            if path[0] == "effect" and path[1] in ("channel_id", "channel"):
                where.setdefault("channel", b)
                return
    if op == "mentions":
        terms.append([node.get("value") or {}])
    if op == "mentions_together":
        terms.append([(t or {}).get("value") or {} for t in node.get("terms") or []])
    for key in ("args", "arg"):
        if key in node:
            _goal_terms(node[key], terms, where, under_choice)


def _goal_value(value, member, say):
    if value.get("kind") == "literal" and isinstance(value.get("value"), (str, int, float)):
        return say(value["value"])
    if (value.get("kind") == "field" and len(value.get("path") or []) == 2 and value["path"][0] in MEMBER_ROOTS
            and member is not None and member.get(value["path"][1]) not in (None, "")):
        return say(member[value["path"][1]])
    return "a value computed from the source data"


GOAL_NOUNS = {("gmail.messages@1", "send"): "an email", ("slack.messages@1", "channel_message"): "a Slack message",
              ("google_sheets.row_writes@1", "append"): "a row added", ("google_sheets.row_writes@1", "update"):
              "a row update", ("service.record_writes@1", "create"): "a created record",
              ("service.record_writes@1", "update"): "a record update"}


def goal_spec(record, contract, check_id, candidate, say):
    """Plain-language requirement for one failed manifest goal (manifest-family template), or None when the
    family states no checkable requirement in plain words (reads, retained rows, coverage)."""
    check = next((c for c in contract.get("checks", []) if c.get("check_id") == check_id), None)
    if check is None:
        raise ValueError(f"goal {check_id} not in the episode's contract")
    kind, population, adapter = manifest_family(contract, check)
    if kind != "goal" or check["operator"] != "effects.required_when@1":
        return None
    tools = set(tool_names(record))
    options = [(check.get("source"), check.get("effect_match"))] + [
        (a.get("source"), a.get("effect_match")) for a in check.get("alternatives") or []]
    chosen = None
    for source_name, match in options:
        source = contract["sources"].get(source_name, {})
        prefix = {"gmail.messages@1": "gmail_", "slack.messages@1": "slack_",
                  "google_sheets.row_writes@1": "google_sheets_"}.get(source.get("adapter")) or (
            f"{source.get('service')}_" if source.get("service") else None)
        if prefix and any(t.startswith(prefix) for t in tools) and GOAL_NOUNS.get(
                (source.get("adapter"), source.get("kind"))):
            chosen = (source, match)
            break
    if chosen is None:
        return {"no_tool": True}
    source, effect_match = chosen
    adapter = source.get("adapter")
    noun = GOAL_NOUNS.get((adapter, source.get("kind")))
    if noun == "a created record" and source.get("service"):
        noun = f"a created {source['service'].replace('_', ' ')} {(source.get('collection') or ['record'])[0].rstrip('s')}"
    member = None
    if candidate is not None:
        material = {}
        for view in trace_of(record).get("assessment_views") or []:
            if view.get("builder_revision") == "automationbench.manifest_inputs@1" and view.get("input_json"):
                material = json.loads(view["input_json"])
        table = json.loads(material.get("obligation_population_evidence_json") or "{}").get(check["population"]) or {}
        member = next((json.loads(r["cells_json"]) for r in table.get("rows", [])
                       if _canonical(r["identity"]) == _canonical(candidate)), None)
    terms, where = [], {}
    _goal_terms(effect_match, terms, where)
    head = noun
    target = None
    if "recipient" in where:
        recipient = _goal_value(where["recipient"], member, say)
        head = f"an email sent to {recipient}" if noun == "an email" else f"{noun} to {recipient}"
        raw = where["recipient"].get("value") if where["recipient"].get("kind") == "literal" else (
            member or {}).get((where["recipient"].get("path") or [None, None])[1])
        target = ("email", raw) if noun == "an email" and isinstance(raw, str) else None
    elif "channel" in where:
        channel = _goal_value(where["channel"], member, say)
        head = f"a Slack message in channel {channel}"
    label = ""
    keys = contract["sources"].get(check.get("population"), {}).get("key_fields") or []
    if member is not None and population != "public.request@1" and keys and "recipient" not in where:
        label = " for " + ", ".join(say(member.get(k)) for k in keys if member.get(k) not in (None, ""))
    clauses = [("whose body mentions together " if noun == "an email" else "mentioning together ")
               + ", ".join(_goal_value(v, member, say) for v in group)
               if len(group) > 1 else ("whose body contains " if noun == "an email" else "containing ")
               + _goal_value(group[0], member, say) for group in terms if group]
    clause = "; ".join(clauses) if clauses else (label.strip() and f"about{label[4:]}")
    if not clauses and not label:
        return {"statement": head, "group": (head, ""), "target": target, "family": f"goal:{adapter}"}
    statement = f"{head}{label if not clauses else ''} {clause}".strip() if clauses else f"{head}{label}"
    return {"statement": statement, "group": (head, clause if clauses else label.strip()), "target": target,
            "family": f"goal:{adapter}"}


# ---------------------------------------------------------------- closest-effect diagnosis (G1, G2)

_GOAL_INPUTS = {}


def _goal_material(record):
    """(contract model, populations, effects) of the episode's own obligation material, verified by the env."""
    key = record.get("id")
    if key not in _GOAL_INPUTS:
        from automationbench_v1.capture import canonical_json
        from automationbench_v1.contracts.loader import load_contract
        from automationbench_v1.manifest_obligation_assessments import restore_obligation_inputs

        material = None
        for view in trace_of(record).get("assessment_views") or []:
            if view.get("builder_revision") == "automationbench.manifest_inputs@1" and view.get("input_json"):
                material = json.loads(view["input_json"])
        if material is None:
            _GOAL_INPUTS[key] = None
        else:
            contract = load_contract(canonical_json(material["contract"]))
            _GOAL_INPUTS[key] = (contract, *restore_obligation_inputs(material, contract))
        while len(_GOAL_INPUTS) > 16:
            _GOAL_INPUTS.pop(next(iter(_GOAL_INPUTS)))
    return _GOAL_INPUTS[key]


_IDENTITY_FIELDS = {"recipients", "to", "cc", "channel_id", "channel", "record_id", "native_record_id", "row_id",
                    "parent_id", "contact_id", "conversation_id", "ticket_id", "who_id", "to_number", "phone",
                    "recordId", "recordid", "envelope_id"}


def _identity_conjunct(node):
    """A condition that only says which addressee / record / channel the effect is about."""
    from automationbench_v1.contracts.predicates import Comparison

    if not isinstance(node, Comparison):
        return False
    for side in (node.left, node.right):
        for path in _operand_paths(side):
            if path[0] == "effect" and path[-1] in _IDENTITY_FIELDS:
                return True
    return False


class Unrenderable(Exception):
    pass


def _dedupe_choices(text):
    """'must be one of "sponsor" or "Sponsor" or "SPONSOR"': each choice once, case-insensitively."""
    def fix(m):
        values = _QUOTED.findall(m.group(1))
        seen, kept = set(), []
        for v in values:
            if normalize(v) not in seen:
                seen.add(normalize(v))
                kept.append(v)
        if len(kept) == 1:
            return f"must be {kept[0]}"
        return "must be one of " + ", ".join(kept[:-1]) + " or " + kept[-1]
    return re.sub(r'must be one of ((?:"[^"]*"(?:, | or |,? or )?)+)', fix, text)


def _goal_rule(node, context, say, call):
    return _dedupe_choices(_goal_rule_raw(node, context, say, call))


def _goal_rule_raw(node, context, say, call):
    """A failing goal condition stated exactly, or Unrenderable."""
    from automationbench_v1.contracts.predicates import Comparison, Mentions, MentionsTogether

    if isinstance(node, MentionsTogether):
        if node.excluding_values:
            raise Unrenderable("mentions_together with exclusions")
        field = json.dumps(_field_label(node.text.path))
        terms = [_value_text(t.value, context, say) for t in node.terms]
        unit = {"line": "on one line", "block": "in one paragraph", "text": "together"}[node.scope]
        return f"{field} must give {', '.join(terms[:-1])} and {terms[-1]} {unit}"
    from automationbench_v1.contracts.predicates import Junction

    if isinstance(node, Junction) and node.op == "any":
        # one of several alternatives: each stated faithfully, or the whole requirement is not rendered
        options = [_goal_rule(arg, context, say, None) for arg in node.args]
        fields = {o.split(" must ", 1)[0] for o in options}
        if len(fields) == 1 and all(" must contain " in o and "(" not in o for o in options):
            field = fields.pop()
            values = list(dict(((normalize(v), v) for v in reversed(
                [o.split(" must contain ", 1)[1] for o in options]))).values())[::-1]
            if len(values) == 1:
                return f"{field} must contain {values[0]}"
            return f"{field} must contain one of {', '.join(values[:-1])} or {values[-1]}"
        if len(fields) == 1 and all(" must be " in o and "(" not in o for o in options):
            field = fields.pop()
            values = list(dict(((normalize(v), v) for v in reversed(
                [o.split(" must be ", 1)[1] for o in options]))).values())[::-1]
            if len(values) == 1:
                return f"{field} must be {values[0]}"
            return f"{field} must be one of {', '.join(values[:-1])} or {values[-1]}"
        return "one of: " + "; or ".join(options)
    if isinstance(node, Comparison):
        # D34: "join.<alias> == matched": the effect had to come after reading a source (a read-before join)
        paths = [p for o in (node.left, node.right) for p in _operand_paths(o)]
        alias = next((p[1] for p in paths if p and p[0] == "join" and len(p) > 1), None)
        if alias is not None:
            where = _join_source(context, alias, say)
            if where is None:
                raise Unrenderable("read-before join on an unnamed source")
            return f"must come after reading {where}, which this attempt did not read first"
    if isinstance(node, Mentions) and node.mode == "amount" and node.value is not None:
        # D34: an amount the request computes (a total): stated as such, never as a hidden figure
        field = json.dumps(_field_label(node.text.path))
        literal = _resolve(node.value, context) if not _operand_paths(node.value) else None
        shown = say(literal) if literal not in (None, "") and observable(literal, say.corpus) else \
            "the amount computed from the source data as the request asks"
        return f"{field} must contain {shown}"
    if isinstance(node, (Mentions, Comparison)):
        if isinstance(node, Mentions) and (node.value is None or node.mode not in ("words", "verbatim", "prefix")):
            raise Unrenderable(f"mentions mode {node.mode}")
        value_side = node.value if isinstance(node, Mentions) else next(
            (o for o in (node.left, node.right) if not any(p[0] == "effect" for p in _operand_paths(o))), None)
        raw = _resolve(value_side, context) if value_side is not None else None
        for v in _aslist(raw):
            if isinstance(v, str) and v and not observable(v, say.corpus) and unobserved_source(v, say) is None:
                raise Unrenderable(f"value neither observed nor locatable: {v[:40]}")
        rule = _rule_for(node, False, context, say, call)
        if rule is None or isinstance(rule, tuple):
            raise Unrenderable(type(node).__name__)
        target = node.text if isinstance(node, Mentions) else next(
            (o for o in (node.left, node.right) if any(p[0] == "effect" for p in _operand_paths(o))), None)
        actual = _resolve(target, context) if target is not None else None
        if isinstance(actual, str) and actual and len(actual) <= 60 and call is not None and _find_literal(
                call["arguments"], actual):
            rule += f" (it is currently {json.dumps(actual)})"
        return rule
    raise Unrenderable(type(node).__name__)


def _join_alias(node):
    from automationbench_v1.contracts.predicates import Comparison

    if not isinstance(node, Comparison):
        return None
    paths = [p for o in (node.left, node.right) for p in _operand_paths(o)]
    return next((p[1] for p in paths if p and p[0] == "join" and len(p) > 1), None)


def _join_read_before(node, context, calls, index):
    """Did a successful read of the joined record come before the call at `index`? False when unknown."""
    alias = _join_alias(node)
    if alias is None or index is None:
        return False
    join = next((j for j in context.get("_effect_joins") or [] if j.get("alias") == alias), None)
    if join is None:
        return False
    wanted = [str(v) for v in _join_literals(join.get("where")).values() if isinstance(v, str) and len(str(v)) >= 3]
    if not wanted:
        return False
    for c in calls[:index]:
        if c["error"] or _call_kind(c["name"]) is not None:
            continue
        # a read that names the record (its id, or its spreadsheet) and whose result holds every literal
        if any(_find_literal(c["arguments"], w) for w in wanted) and all(
                _find_literal(c["arguments"], w) or _find_literal(c.get("result") or "", w) for w in wanted):
            return True
    return False


def _join_literals(where):
    literals = {}

    def walk(node):
        if isinstance(node, dict):
            if node.get("op") == "eq":
                left, right = node.get("left") or {}, node.get("right") or {}
                if left.get("kind") == "field" and right.get("kind") == "literal":
                    literals[str((left.get("path") or [""])[-1])] = right.get("value")
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(where)
    return literals


def _join_source(context, alias, say):
    """The record a read-before join names (an email by its locator, a worksheet by its title), or None."""
    joins = context.get("_effect_joins") or []
    join = next((j for j in joins if j.get("alias") == alias), None)
    if join is None or say.initial is None:
        return None
    literals = _join_literals(join.get("where"))
    world = say.initial.model_dump(mode="json")
    message_id = literals.get("message_id") or literals.get("native_record_id")
    if message_id:
        message = next((m for m in (world.get("gmail") or {}).get("messages") or [] if m.get("id") == message_id), None)
        return gmail_locator(message, say) if message else None
    if literals.get("spreadsheet_id"):
        sheet, worksheet = literals["spreadsheet_id"], literals.get("worksheet_id")
        title = next((w.get("title") for w in (world.get("google_sheets") or {}).get("worksheets") or []
                      if isinstance(w, dict) and w.get("id") == worksheet), None) or worksheet
        if worksheet and _locator_ok(say, title) and _locator_ok(say, sheet):
            return f"worksheet {json.dumps(title)} of spreadsheet {json.dumps(sheet)}"
        return f"spreadsheet {json.dumps(sheet)}" if _locator_ok(say, sheet) else None
    return None


def diagnose_goal(record, check_id, candidate, say, invocations, by_site, calls, forbidden=frozenset()):
    """Closest effect for one failed goal member: ("closest", text, call) with only the failing conditions,
    ("missing", None, None) when the attempt made no effect on that addressee/record, or ("unrenderable", why)."""
    from automationbench_v1.contracts.obligations import _context
    from automationbench_v1.contracts.predicates import Junction, Mentions, _evaluate

    restored = _goal_material(record)
    if restored is None:
        return ("unrenderable", "no obligation material", None)
    contract, populations, effects = restored
    check = next((c for c in contract.checks if c.check_id == check_id), None)
    if check is None:
        return ("unrenderable", "goal not in the episode's contract", None)
    row = next((r for r in populations[check.population].rows
                if _canonical(list(r.identity)) == _canonical(candidate)), None) if candidate is not None else None
    if row is None:
        rows = populations[check.population].rows
        row = rows[0] if len(rows) == 1 else None
    if row is None:
        return ("unrenderable", "population member not found", None)
    context = _context(check, row, populations)
    check_json = next((c for c in (episode_contract(record) or {}).get("checks", []) if c.get("check_id") == check_id),
                      {})
    context = context | {"_effect_joins": check_json.get("effect_joins") or []}
    options = [(check.source, check.effect_match)] + [(a.source, a.effect_match) for a in check.alternatives or ()]
    best = None
    for source, match in options:
        conjuncts = list(match.args) if isinstance(match, Junction) and match.op == "all" else [match]
        identity = [c for c in conjuncts if _identity_conjunct(c)]
        rest = [c for c in conjuncts if not _identity_conjunct(c)]
        source_effects = [e for e in (effects[source].effects if source in effects else ()) if e.params_json]
        only = None
        if not identity:
            # D34: no addressee/record conjunct: the conjuncts naming this member's own key (its name in the effect)
            # say which effect is about it; with none, only a single effect of that kind is unambiguous
            identity = [c for c in rest if _names_member(c)]
            rest = [c for c in rest if c not in identity]
            if not identity and len(source_effects) == 1:
                identity = [None]
            if not identity:
                # the one effect whose content names this member (its name or email): about this member
                names = [normalize(v) for k, v in (context.get("request") or {}).items() if isinstance(v, str)
                         and len(v) >= 4 and re.search(r"(name|email|employee|registrant|contact|title)", str(k).lower())]
                def mentions(e, values):
                    return sum(1 for n in values if re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)",
                                                              normalize(e.params_json)))
                about = [e for e in source_effects if names and mentions(e, names)]
                others = [normalize(v) for r in populations[check.population].rows if r is not row
                          for k, v in json.loads(r.cells_json).items() if isinstance(v, str) and len(v) >= 4
                          and re.search(r"(name|email|employee|registrant|contact|title)", str(k).lower())]
                if len(about) > 1:
                    # several effects name it: the one naming the fewest other members is about it alone
                    scores = sorted((mentions(e, others), i) for i, e in enumerate(about))
                    if len(scores) == 1 or scores[0][0] < scores[1][0]:
                        about = [about[scores[0][1]]]
                if len(about) == 1:
                    identity = [None]
                    only = about[0]
        for effect in effects[source].effects if source in effects else ():
            if effect.params_json is None:
                continue
            if only is not None and effect is not only:
                continue
            site = invocations.get(effect.invocation_id)
            if site in by_site and id(calls[by_site[site]]) in forbidden:
                continue  # a forbidden effect is never the closest effect of a requirement
            material = context | {"effect": json.loads(effect.params_json)}
            if not identity or any(c is not None and _evaluate(c, material).value is not True for c in identity):
                continue
            failing = [c for c in rest if _evaluate(c, material).value is not True]
            # D34: a read-before join is judged from the attempt's own calls (the material has no join results):
            # it holds when a successful read of that record came before the effect's call
            index = by_site.get(invocations.get(effect.invocation_id))
            failing = [c for c in failing if not _join_read_before(c, context, calls, index)]
            # E: among equally close effects, the one whose content names this record (its own cell values)
            text = normalize(effect.params_json)
            about = sum(1 for v in (context.get("request") or {}).values()
                        if isinstance(v, str) and len(v) >= 4 and not _is_number(v)
                        and re.search(r"(?<!\w)" + re.escape(normalize(v)) + r"(?!\w)", text))
            if best is None or (len(failing), -about) < (len(best[0]), -best[3]):
                best = (failing, material, effect, about)
    if best is None:
        return ("missing", None, None)
    failing, material, effect, _ = best
    if not failing:
        return ("unrenderable", "the closest effect meets every effect condition", None)
    site = invocations.get(effect.invocation_id)
    call = calls[by_site[site]] if site in by_site else None
    keep = _keep_state(record, check, failing, material, call, say)
    if keep is not None:
        return keep
    try:
        rules = [_goal_rule(c, material, say, call) for c in failing]
    except Unrenderable as error:
        return ("unrenderable", str(error), None)
    raws = []
    for node in failing:
        side = getattr(node, "value", None) if isinstance(node, Mentions) else next(
            (o for o in (getattr(node, "left", None), getattr(node, "right", None))
             if o is not None and not any(p[0] == "effect" for p in _operand_paths(o))), None)
        try:
            raws += [str(v) for v in _aslist(_resolve(side, material)) if v not in (None, "")] if side is not None else []
        except Exception:  # noqa: BLE001 - a value that does not resolve is simply not compared
            pass
    return ("closest", rules, call, raws)


_APP_TITLE = {"gorgias": "Gorgias", "zendesk": "Zendesk", "hubspot": "HubSpot", "salesforce": "Salesforce",
              "reamaze": "Re:amaze", "zoho": "Zoho Desk", "freshdesk": "Freshdesk", "docusign": "DocuSign",
              "airtable": "Airtable", "asana": "Asana", "trello": "Trello", "notion": "Notion", "calendly": "Calendly"}
_STATE_VERB = {"closed": "close", "resolved": "resolve", "cancelled": "cancel", "canceled": "cancel",
               "archived": "archive", "deleted": "delete", "voided": "void", "solved": "solve", "completed": "complete"}


def _keep_state(record, check, failing, material, call, say):
    """N: a record that must keep its state, changed by an update call, reads as 'Do not close Gorgias ticket
    "g_211"; it must stay "open"' with the record's own observed reason (or where that reason is), not as a tool
    tweak."""
    from automationbench_v1.contracts.predicates import Comparison

    if call is None or _call_kind(call["name"]) != "update" or len(failing) != 1:
        return None
    node = failing[0]
    if not isinstance(node, Comparison) or node.op != "eq":
        return None
    sides = [(o, any(p[0] == "effect" for p in _operand_paths(o))) for o in (node.left, node.right)]
    target = next((o for o, effect in sides if effect), None)
    value_side = next((o for o, effect in sides if not effect), None)
    if target is None or value_side is None:
        return None
    expected, actual = _resolve(value_side, material), _resolve(target, material)
    if not (isinstance(expected, str) and isinstance(actual, str)) or normalize(expected) == normalize(actual):
        return None
    app = call["name"].split("_")[0]
    obj = call["name"].rsplit("_", 1)[-1]
    ids = [v for k, v in call["arguments"].items() if str(k).lower().endswith("id") and isinstance(v, (str, int))]
    if not ids:
        return None
    noun = f"{_APP_TITLE.get(app, app.title())} {obj} {say(str(ids[0]))}"
    verb = _STATE_VERB.get(normalize(actual))
    text = (f"do not {verb} {noun}" if verb else f"do not set {noun} to {json.dumps(actual)}") + \
        f"; it must stay {say(expected)}"
    # the reason: the member's own fields on the satisfied path of the goal's condition, observed only
    cells = material.get("request") if isinstance(material.get("request"), dict) else {}
    fields = _satisfied_fields(episode_contract(record), check.population, cells, "require") or set()
    shown, unseen = [], []
    for field in sorted(fields):
        value = cells.get(field)
        if isinstance(value, str) and value.strip() and normalize(value) not in (normalize(expected),):
            (shown if observable(value, say.corpus) else unseen).append((field, value))
    if shown:
        text += " (" + "; ".join(f"{json.dumps(f)}: {say(v)}" for f, v in shown[:2]) + ")"
    elif unseen:
        identity = list(getattr(check, "population", "") and material.get("candidate", {}).get("identity") or [])
        sheet = next((x for x in identity if isinstance(x, str) and x.startswith("ss_")), None)
        if sheet:
            text += f" (its {json.dumps(unseen[0][0])} is in spreadsheet {say(sheet)})"
    return ("keep", text, call)


def _names_member(node):
    """A condition that the effect mentions the member's own key value (its name, email or id field)."""
    from automationbench_v1.contracts.predicates import Mentions

    from automationbench_v1.contracts.predicates import Comparison

    if isinstance(node, Comparison) and node.op == "eq":
        sides = [_operand_paths(o) for o in (node.left, node.right)]
        return (any(p and p[0][0] == "effect" for p in sides)
                and any(p and p[0][0] in MEMBER_ROOTS for p in sides))
    if not isinstance(node, Mentions) or node.value is None:
        return False
    paths = _operand_paths(node.value)
    return bool(paths) and paths[0][0] in MEMBER_ROOTS and any(p[0] == "effect" for p in _operand_paths(node.text))


def _subject_of(call, say):
    args = call["arguments"] if call else {}
    name = call["name"] if call else ""
    if name.startswith("gmail_"):
        return f"the email to {say(args.get('to'))}"
    if name.startswith("slack_"):
        return f"the Slack message in channel {say(args.get('channel') or args.get('channel_id'))}"
    if name.startswith("twilio_"):
        return f"the SMS to {say(args.get('to_number'))}"
    if name.startswith("google_sheets_add"):
        return "the row added here"
    if name.startswith("google_sheets_update"):
        return "the row updated here"
    if name:
        app = name.split("_")[0]
        obj = re.sub(r"^(create|add|update|send|post|new)_", "", "_".join(name.split("_")[1:])).replace("_", " ")
        return f"the {_APP_TITLE.get(app, app.title())} {obj} written here"
    return "the effect written here"


# native requirements: anchor (is there an effect on the same addressee/record?) and per-condition sub-checks
def _native_parts(result, say):
    """(anchor assertion, subject phrase, [(sub-assertion, rule text)]) for a decomposable native requirement,
    or None when the family has no anchor (then it stays 'Still missing')."""
    t, p = result["type"], result.get("params") or {}
    items = lambda v: [x for x in _aslist(v) if x]  # noqa: E731
    if t in ("gmail_message_sent_to_with_body_contains", "gmail_message_body_contains", "gmail_message_sent_to",
             "gmail_message_sent") and p.get("to"):
        subject = p.get("subject") or p.get("subject_contains")
        parts = []
        if subject:
            parts.append(({"type": "gmail_message_sent_to", "to": p["to"], "subject": subject},
                          f"must have {say(subject)} in its subject"))
        for item in items(p.get("body_contains")):
            parts.append(({"type": "gmail_message_body_contains", "to": p["to"], "body_contains": item},
                          f"must contain {say(item)}"))
        return {"type": "gmail_message_sent_to", "to": p["to"]}, f"the email to {say(p['to'])}", parts, "email"
    if t in ("slack_message_exists", "slack_message_in_channel", "slack_message_sent_to_channel"):
        channel = p.get("channel") or p.get("channel_name")
        anchor = {"type": "slack_message_in_channel", "channel": channel}
        parts = [({"type": "slack_message_in_channel", "channel": channel, "text_contains": item},
                  f"must contain {say(item)}") for item in items(p.get("text_contains"))]
        return anchor, f"the Slack message in channel {say(channel)}", parts, "slack message"
    if t == "twilio_sms_sent":
        return ({"type": "twilio_sms_sent", "to_number": p["to_number"]}, f"the SMS to {say(p['to_number'])}",
                [(dict(result.get("params"), type=t), f"must contain {say(p['body_contains'])}")], "SMS")
    if t == "reamaze_conversation_has_message":
        return ({"type": t, "conversation_id": p["conversation_id"]},
                f"the message on Re:amaze conversation {say(p['conversation_id'])}",
                [({"type": t, **p}, f"must contain the word {say(p['body_contains'])}")], "message")
    if t == "gorgias_ticket_has_message":
        return ({"type": t, "ticket_id": p["ticket_id"], "sender_type": p["sender_type"]},
                f"the reply on Gorgias ticket {say(p['ticket_id'])}",
                [({"type": t, **p}, f"must contain {say(p['body_contains'])}")], "reply")
    if t in ("zendesk_ticket_has_comment", "zoho_desk_ticket_has_comment"):
        key = "body_contains" if "body_contains" in p else "content_contains"
        return ({"type": t, "ticket_id": p["ticket_id"]}, f"the comment on ticket {say(p['ticket_id'])}",
                [({"type": t, **p}, f"must contain {say(p[key])}")], "comment")
    if t in ("salesforce_note_exists", "salesforce_note_exists_for_record"):
        return ({"type": "salesforce_note_exists", "parent_id": p["parent_id"]},
                f"the Salesforce note on record {say(p['parent_id'])}",
                [({"type": t, **p}, f"must contain {say(p['body_contains'])}")], "note")
    if t == "hubspot_contact_has_property":
        return ({"type": "hubspot_contact_exists", "email": p["email"]}, f"HubSpot contact {say(p['email'])}",
                [({"type": t, **p}, f"must have {say(p['property'])} set to {say(p['value'])}")], "contact")
    return None


# ---- G1 for record families: the closest effect must be of the SAME kind as the requirement
_READ_VERBS = re.compile(r"(find|get|list|lookup|search|query|read|fetch|describe|schema)")


def _call_kind(name):
    """The effect kind a successful call makes: send (email/Slack/SMS), message (a message, comment, note, reply
    or invitation on a record), void, draft, update (an existing record changed), create (a new record); None
    for reads."""
    n = name.lower()
    if _READ_VERBS.search(n) and not re.search(r"(create|add|update|send|post|void|book|new)", n):
        return None
    if "void" in n:
        return "void"
    if "draft" in n:
        return "draft"
    if n.startswith(("gmail_", "slack_", "twilio_")):
        return "send" if re.search(r"(send|reply|forward|post|message)", n) else None
    if re.search(r"(message|comment|note|reply|story|invitation)", n) and re.search(
            r"(create|add|send|post|reply|new|comment)", n):
        return "message"
    if re.search(r"(update|upsert|mark|cancel|tag|move|set_|assign|close|edit|signer|cc)", n):
        return "update"
    if re.search(r"(create|add|new|book|send|invite|subscribe|schedule)", n):
        return "create"
    return None


def _norm_key(key):
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _arg_values(arguments, field):
    """Values of `field` anywhere in a call's arguments (JSON-string objects included), by normalized key."""
    out = []

    def walk(value):
        if isinstance(value, str) and value.strip().startswith("{"):
            try:
                value = json.loads(value)
            except ValueError:
                return
        if isinstance(value, dict):
            for k, v in value.items():
                if _norm_key(k) == _norm_key(field):
                    out.append(v)
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    walk(arguments)
    return out


def _arg_has(arguments, field, value, equals=False):
    """Does the call give `value` for `field` (or anywhere, when field is None)? None when the field is absent."""
    if field is None:
        return _find_literal(arguments, value) is not None
    found = _arg_values(arguments, field)
    if not found:
        return None
    for v in found:
        if equals and normalize(str(v)) == normalize(str(value)):
            return True
        if equals and isinstance(value, (int, float)):
            try:
                if float(str(v).replace(",", "").replace("$", "")) == float(value):
                    return True
            except ValueError:
                pass
        if not equals and _find_literal(v, value) is not None:
            return True
    return False


def _cond(field, value, rule, clause, equals=False):
    return {"field": field, "value": value, "rule": rule, "clause": clause, "equals": equals}


def _record_family(result, say):
    """(app, kinds, identity, noun, conditions) for a native requirement on one record, or None.
    identity: (field, value) the closest call must carry, or None when the record is named by nothing but its
    conditions (then only a single same-kind call of the app counts). noun: how the record is named."""
    t, p, s = result["type"], result.get("params") or {}, say
    field_rule = lambda f, v: _cond(f, v, f"must have {s(v)} in its {s(f)}", f"its {s(f)} contains {s(v)}")  # noqa: E731
    field_set = lambda f, v: _cond(f, v, f"must have its {s(f)} set to {s(v)}", f"its {s(f)} is {s(v)}", True)  # noqa: E731
    if t in ("salesforce_field_contains", "salesforce_field_equals"):
        obj = _sf_object(p)
        make = field_rule if t.endswith("contains") else field_set
        return ("salesforce", {"create", "update"}, (None, p["record_id"]), f"Salesforce {obj} {s(p['record_id'])}",
                [make(p["field"], p["value"])])
    if t in ("salesforce_lead_field_contains", "salesforce_lead_field_equals"):
        make = field_rule if t.endswith("contains") else field_set
        return ("salesforce", {"create", "update"}, ("email", p["email"]), f"Salesforce lead {s(p['email'])}",
                [make(p["field"], p["value"])])
    if t == "salesforce_contact_phone_equals":
        return ("salesforce", {"create", "update"}, (None, p["contact_id"]),
                f"Salesforce contact {s(p['contact_id'])}", [field_set("phone", p["phone"])])
    if t in ("salesforce_note_exists", "salesforce_note_exists_for_record"):
        return ("salesforce", {"message", "create"}, (None, p["parent_id"]),
                f"Salesforce note on record {s(p['parent_id'])}",
                [_cond(None, p["body_contains"], f"must contain {s(p['body_contains'])}", "")], )
    if t == "hubspot_contact_has_property":
        return ("hubspot", {"create", "update"}, ("email", p["email"]), f"HubSpot contact {s(p['email'])}",
                [field_set(p["property"], p["value"])])
    if t == "hubspot_company_exists" and p.get("domain"):
        return ("hubspot", {"create"}, ("name", p["name"]), f"HubSpot company {s(p['name'])}",
                [field_set("domain", p["domain"])])
    if t == "hubspot_ticket_exists":
        conds = [field_rule("subject", p["subject_contains"])]
        if p.get("description_contains"):
            conds.append(field_rule("description", p["description_contains"]))
        return ("hubspot", {"create"}, None, "HubSpot ticket", conds)
    if t == "gorgias_ticket_exists":
        return ("gorgias", {"update", "create"}, ("ticket_id", p["ticket_id"]), f"Gorgias ticket {s(p['ticket_id'])}",
                [field_set("status", p["status"])])
    if t == "gorgias_ticket_has_message":
        return ("gorgias", {"message"}, ("ticket_id", p["ticket_id"]),
                f"{_sender(p, s)[2:]} on Gorgias ticket {s(p['ticket_id'])}",
                [_cond(None, p["body_contains"], f"must contain {s(p['body_contains'])}", "")])
    if t == "reamaze_conversation_has_message":
        return ("reamaze", {"message"}, ("conversation_id", p["conversation_id"]),
                f"message on Re:amaze conversation {s(p['conversation_id'])}",
                [_cond(None, p["body_contains"], f"must contain the word {s(p['body_contains'])}", "")])
    if t == "reamaze_conversation_exists":
        return ("reamaze", {"update"}, ("conversation_id", p["conversation_id"]),
                f"Re:amaze conversation {s(p['conversation_id'])}",
                [_cond("assignee_email", p["assignee_email"], f"must be assigned to {s(p['assignee_email'])}",
                       f"it is assigned to {s(p['assignee_email'])}", True)])
    if t == "reamaze_conversation_has_tag":
        return ("reamaze", {"update"}, ("conversation_id", p["conversation_id"]),
                f"Re:amaze conversation {s(p['conversation_id'])}",
                [_cond("tags", p["tag"], f"must be tagged {s(p['tag'])}", f"it is tagged {s(p['tag'])}")])
    if t in ("zendesk_ticket_has_comment", "zoho_desk_ticket_has_comment"):
        key = "body_contains" if "body_contains" in p else "content_contains"
        app = t.split("_ticket")[0]
        return (app, {"message"}, ("ticket_id", p["ticket_id"]),
                f"comment on {'Zendesk' if app == 'zendesk' else 'Zoho Desk'} ticket {s(p['ticket_id'])}",
                [_cond(None, p[key], f"must contain {s(p[key])}", "")])
    if t == "zendesk_ticket_exists":
        return ("zendesk", {"update", "create"}, ("ticket_id", p["ticket_id"]), f"Zendesk ticket {s(p['ticket_id'])}",
                [field_set("group_id", p["group_id"])])
    if t == "zendesk_ticket_has_tag":
        return ("zendesk", {"update", "create"}, ("ticket_id", p["ticket_id"]), f"Zendesk ticket {s(p['ticket_id'])}",
                [_cond("tags", p["tag"], f"must be tagged {s(p['tag'])}", f"it is tagged {s(p['tag'])}")])
    if t == "zendesk_organization_field_contains":
        return ("zendesk", {"update", "create"}, ("organization_id", p["organization_id"]),
                f"Zendesk organization {s(p['organization_id'])}", [field_rule(p["field"], p["value_contains"])])
    if t == "zendesk_organization_has_tag":
        return ("zendesk", {"update", "create"}, ("organization_id", p["organization_id"]),
                f"Zendesk organization {s(p['organization_id'])}",
                [_cond("tags", p["tag"], f"must be tagged {s(p['tag'])}", f"it is tagged {s(p['tag'])}")])
    if t == "docusign_envelope_voided":
        return ("docusign", {"void"}, (None, p["envelope_id"]), f"void of DocuSign envelope {s(p['envelope_id'])}",
                [_cond("reason", p["reason_contains"], f"must give a reason containing {s(p['reason_contains'])}", "")])
    if t == "docusign_envelope_from_template":
        return ("docusign", {"create"}, None, "DocuSign envelope",
                [_cond("template_id", p["template_id"], f"must be made from template {s(p['template_id'])}", "")])
    if t in ("docusign_envelope_has_signer", "docusign_cc_exists"):
        email = p.get("signer_email") or p.get("email")
        role = "a CC recipient" if t == "docusign_cc_exists" else "a signer"
        return ("docusign", {"create", "update"}, None, "DocuSign envelope",
                [_cond(None, email, f"must list {s(email)} as {role}", "")])
    if t == "mailchimp_subscriber_exists":
        return ("mailchimp", {"create", "update"}, ("email", p["email"]), f"Mailchimp subscriber {s(p['email'])}",
                [field_set("list_id", p["list_id"])])
    if t == "linkedin_invitation_message_contains":
        return ("linkedin", {"message", "create"}, (None, p["profile_url"]),
                f"LinkedIn invitation to {s(p['profile_url'])}",
                [_cond(None, p["text"], f"must contain {s(p['text'])} in its message", "")])
    if t == "airtable_record_exists":
        return ("airtable", {"create", "update"}, (None, p["tableName"]),
                f"Airtable record in table {s(p['tableName'])}",
                [field_set(k, v) for k, v in p["fields"].items()])
    if t == "google_sheets_row_updated" and isinstance(p.get("cell_contains"), dict):
        return ("google_sheets_update", {"update"}, ("row_id", p["row_id"]),
                s.row(p["spreadsheet_id"], p["row_id"]) + ":",
                [_cond(k, v, f"{s(k)} must be {s(v)}", f"{s(k)} is {s(v)}")
                 for k, v in p["cell_contains"].items()])
    if t.endswith("_action_exists"):
        app, q = t.split("_action")[0], (p.get("params") or {})
        if app == "calendly":
            return ("calendly", {"create"}, ("invitee_email", q.get("invitee_email")) if q.get("invitee_email") else None,
                    f"Calendly booking for {s(q['invitee_email'])}" if q.get("invitee_email") else "Calendly booking",
                    [_cond(None, q["name"], f"must be for the event {s(q['name'])}", ""),
                     _cond("duration", q["duration"], f"must last {s(q['duration'])} minutes", "", True)]
                    if "duration" in q else [_cond(None, q["name"], f"must be for the event {s(q['name'])}", "")])
        if app == "zoom" and "duration" in q:
            return ("zoom", {"create"}, ("topic", q["topic"]), f"Zoom meeting {s(q['topic'])}",
                    [_cond("duration", q["duration"], f"must last {s(q['duration'])} minutes", "", True)])
        if app == "notion" and "parent_page" in q:
            return ("notion", {"create"}, ("title", q["title"]), f"Notion page {s(q['title'])}",
                    [_cond(None, q["parent_page"], f"must be under page {s(q['parent_page'])}", "")])
        if app == "airtable" and "commentText_contains" in q:
            return ("airtable", {"message"}, (None, q["recordId"]), f"Airtable comment on record {s(q['recordId'])}",
                    [_cond(None, q["commentText_contains"], f"must contain {s(q['commentText_contains'])}", "")])
        if app == "asana" and "name_contains" in q:
            return ("asana", {"create", "update"}, None, "Asana task",
                    [_cond("name", q["name_contains"], f"must have {s(q['name_contains'])} in its name", "")])
        if app == "trello" and "name" in q:
            return ("trello", {"create", "update"}, None, "Trello card",
                    [_cond("name", q["name"], f"must have {s(q['name'])} in its name", "")])
    return None


_KIND_VERB = {"void": "voided", "update": "updated", "create": "created", "message": "posted on"}


def _other_record_note(same_kind, identity, kinds, say):
    """L: the same kind of effect made on a different record: 'this attempt voided "X" instead, at turn N'."""
    if identity is None or not same_kind or identity[0] == "row_id":
        return None
    # only effects on existing records (a void, an update): creating or posting on other records is normal in
    # tasks with several records and does not say the attempt chose the wrong one
    same_kind = [c for c in same_kind if _call_kind(c["name"]) in ("void", "update")]
    if not same_kind:
        return None
    call = same_kind[-1]
    field, _ = identity
    named = _arg_values(call["arguments"], field) if field else []
    if not named:
        named = [v for k, v in call["arguments"].items() if isinstance(v, (str, int)) and (
            str(k).lower().endswith("_id") or str(k).lower() in ("id", "email", "to", "name"))]
    if not named:
        return None
    verb = _KIND_VERB.get(_call_kind(call["name"]), "acted on")
    return f"this attempt {verb} {say(str(named[0]))} instead, at turn {call['turn']}"


def _app_of(name, app):
    if app == "google_sheets_update":
        return name.startswith("google_sheets_")
    return name.startswith(app + "_")


def _initial_has(initial, app, value):
    """Does the initial world hold a record whose own field equals `value` in this app's data?"""
    if initial is None or value in (None, ""):
        return False
    world = initial.model_dump(mode="json")
    service = world.get(app.replace("_update", "")) or {}
    return any(normalize(str(v)) == normalize(str(value)) for _, v in _strings_in(service))


def diagnose_record(result, say, calls, initial):
    """G1 for a record family: ("closest", [rules], call, subject) for the last same-kind call on the record,
    ("update", head, [clauses]) for a pre-existing record the attempt never changed, ("missing",) when there is
    no same-kind effect, ("unrenderable", why), or None for a family outside the table."""
    family = _record_family(result, say)
    if family is None:
        return None
    app, kinds, identity, noun, conditions = family
    candidates = [c for c in calls if not c["error"] and _app_of(c["name"], app) and _call_kind(c["name"]) in kinds]
    same_kind = list(candidates)
    if identity is not None:
        field, value = identity
        if field == "row_id":
            sheet = (result.get("params") or {}).get("spreadsheet_id")
            candidates = [c for c in candidates if any(str(v) == str(value) for v in _arg_values(c["arguments"], "row_id"))
                          and _find_literal(c["arguments"], sheet) is not None]
        else:
            names = [value, *identity_aliases(say, value)]
            candidates = [c for c in candidates if any(
                (_arg_has(c["arguments"], field, v, equals=True) if field else _find_literal(c["arguments"], v))
                for v in names)]
    elif len(candidates) != 1:
        candidates = []  # nothing names the record: only a single same-kind call is unambiguously the closest
    if not candidates:
        if identity is not None and identity[0] != "row_id" and kinds & {"update"} and \
                _initial_has(initial, app, identity[1]) and all(c["clause"] for c in conditions):
            other = _other_record_note(same_kind, identity, kinds, say)
            return ("update", f"update {noun} so that", [c["clause"] for c in conditions]) + ((other,) if other else ())
        other = _other_record_note(same_kind, identity, kinds, say)
        return ("missing", other) if other else ("missing",)
    call = candidates[-1]
    if len(conditions) == 1 and conditions[0]["equals"] and _norm_key(conditions[0]["field"] or "") in (
            "status", "state"):
        written = [str(v) for v in _arg_values(call["arguments"], conditions[0]["field"])]
        if written and _STATE_VERB.get(normalize(written[-1])) and _call_kind(call["name"]) == "update":
            return ("keep", f"do not {_STATE_VERB[normalize(written[-1])]} {noun}; it must stay "
                            f"{say(conditions[0]['value'])}", call)
    failing = []
    for c in conditions:
        has = _arg_has(call["arguments"], c["field"], c["value"], c["equals"])
        if has is not True:
            written = [v for v in (_arg_values(call["arguments"], c["field"]) if c["field"] else [])
                       if isinstance(v, (str, int, float)) and len(str(v)) <= 60]
            # the attempt wrote this value itself, so quoting it is observed (R5); it is shown as what the
            # record holds now, in parentheses, after the requirement
            now = f" (this attempt used {json.dumps(str(written[-1]))})" if written else ""
            counted = re.fullmatch(r"(\d[\d,.]*)\s+([A-Za-z]+)", str(c["value"]).strip())
            if counted and not now:
                text = " ".join(str(v) for v in (_arg_values(call["arguments"], c["field"]) if c["field"] else []))
                form = re.search(rf"(?<![\d.]){re.escape(counted.group(1))}\s*{counted.group(2)[0]}[A-Za-z]*", text,
                                 re.IGNORECASE)
                if form:
                    now = f" (this attempt wrote {json.dumps(form.group(0))})"
            failing.append((c["clause"] + now) if c["clause"] and kinds & {"create", "update"} else c["rule"] + now)
    if not failing:
        return ("unrenderable", "the closest same-kind call gives every condition")
    subject = noun.rstrip(":") if noun.startswith(("the ", "a ")) else f"the {noun}"
    if all(c["clause"] for c in conditions) and kinds & {"create", "update"}:
        # the attempt made this record: the line corrects that effect, naming what must hold (F)
        return ("closest", ["so that " + " and ".join(failing)], call, f"change {subject}")
    return ("closest", failing, call, subject)


def _is_record_name(value):
    """A value naming a record (a person, incident, vendor): letters, not a figure or a code."""
    text = str(value).strip()
    return bool(re.search(r"[A-Za-z]{2}", text)) and not _COMPUTABLE.match(text) and "_" not in text and \
        "@" not in text and not re.search(r"\d", text)


def _email_fields(call):
    args = call["arguments"]
    subject = " ".join(str(v) for v in _arg_values(args, "subject"))
    body = " ".join(str(v) for k in ("body", "body_text", "body_html", "message", "text")
                    for v in _arg_values(args, k))
    recipients = " ".join(str(v) for k in ("to", "cc", "bcc", "recipients") for v in _arg_values(args, k))
    return subject, body, recipients


_POLICY_WORDS = re.compile(r"\b(do not|don't|never|must not|not allowed|prohibit|policy|forbid|no longer)\b",
                           re.IGNORECASE)


def _rule_email(say, value, domain):
    """The email holding the rule against a recipient, named by what the attempt could see, when it was not read:
    'the email "Vendor policy" (not opened by this attempt)' or 'an email in Gmail (not opened by this attempt)'."""
    if say.initial is None:
        return None
    world = say.initial.model_dump(mode="json")
    for message in (world.get("gmail") or {}).get("messages") or []:
        text = json.dumps(message)
        if not _POLICY_WORDS.search(text) or not (_find_literal(text, value) or (
                domain and re.search(rf"\b{re.escape(domain)}\b", text, re.IGNORECASE))):
            continue
        body = str(message.get("body_plain") or message.get("body") or "")
        if body and observable(body[:120], say.corpus):
            return None  # the attempt read the rule
        where = gmail_locator(message, say) or "an email in Gmail"  # D29-A: named by subject and sender
        return f"{where} (not opened by this attempt)"
    return None


def _whole_value(say, value):
    """Is the value a whole field value of some source record (a full name, not a substring of one)?"""
    if say.initial is None:
        return False
    needle = normalize(str(value))
    return any(normalize(str(v)) == needle for _, v in _strings_in(say.initial.model_dump(mode="json")))


def _claim_noun(term, calls):
    """The claim a forbidden term makes, named by the email's own word for it: "enrolled" with a subject
    "Benefits Enrollment Confirmed" -> "enrollment"; otherwise the term itself, quoted."""
    stem = normalize(term)[:max(4, len(normalize(term)) - 3)]
    for c in calls:
        subject, body, _ = _email_fields(c)
        for word in re.findall(r"[A-Za-z]+", subject + " " + body[:300]):
            if normalize(word).startswith(stem) and normalize(word).endswith(("ment", "tion", "al", "ance")):
                return word.lower()
    return json.dumps(term)


def diagnose_email_about(result, say, world, calls):
    """E: a required email about a record (a person, incident, vendor) is anchored to the sent email whose content
    mentions that record, not to any email to the same recipient. None when the rule does not apply."""
    t, p = result["type"], result.get("params") or {}
    if t not in ("gmail_message_sent_to_with_body_contains", "gmail_message_body_contains") or not p.get("to"):
        return None
    values = [v for v in _aslist(p.get("body_contains")) if v]
    if world is None or not values or not all(_is_record_name(v) for v in values):
        return None
    failing = [v for v in values if _check(world, {"type": "gmail_message_body_contains", "to": p["to"],
                                                   "body_contains": v}) is False]
    if not failing:
        return None
    sent = [c for c in calls if not c["error"] and _call_kind(c["name"]) == "send" and c["name"].startswith("gmail_")]
    to_recipient = [c for c in sent if _find_literal(_email_fields(c)[2], p["to"]) is not None]
    # the subject names the record but the body does not: name the failing field
    # 6: per record, the email whose subject names it but whose body does not
    owners = {}
    for v in failing:
        hits = [c for c in to_recipient if _find_literal(_email_fields(c)[0], v)
                and not _find_literal(_email_fields(c)[1], v)]
        if hits:
            owners[v] = hits[-1]
    if owners and len(owners) == len(failing):
        distinct = list(dict.fromkeys(id(c) for c in owners.values()))
        call = list(owners.values())[0]
        if len(distinct) == 1:
            return ("closest", ["so that its body, not only its subject, names them"], call,
                    f"change the email about {say.all(failing)}")
        return ("closest", ["so that their bodies, not only their subjects, name them"], call,
                f"change the emails about {say.all(failing)}")
    about = [c for c in sent if c not in to_recipient
             and all(_find_literal(" ".join(_email_fields(c)[:2]), v) for v in failing)]
    if about:
        # a per-record email (the shortest one naming the record, to the fewest recipients) over a roster
        call = min(about, key=lambda c: (len(_email_fields(c)[1]), len(re.split(r"[,;]", _email_fields(c)[2]))))
        recipients = _email_fields(call)[2]
        required = getattr(say, "required_recipients", set())
        if not any(normalize(r.strip()) in required for r in re.split(r"[,;]", recipients) if r.strip()):
            # r5-1: its own recipient is not one the task requires: the email went to the wrong address
            return ("redirect", f"the email went to {say(recipients)}, which is the wrong recipient; "
                                f"it must go to {say(p['to'])}", call, say(p["to"]))
        if not all(_whole_value(say, v) for v in failing):
            # check substrings ("Mid-funnel") do not name an email; its turn does
            return ("closest", [f"so that it also goes to {say(p['to'])}"], call,
                    f"change the email at turn {call['turn']}")
        return ("closest", [f"so that it also goes to {say(p['to'])}"], call,
                f"change the email about {say.all(failing)}")
    if not any(_find_literal(" ".join(_email_fields(c)[:2]), v) for c in sent for v in failing) and \
            len(to_recipient) > 1:
        # one email per record to this recipient, none about this record: those emails are not its closest
        # effect; a single email to the recipient is (it is corrected at its turn). D34: say that emails to it
        # exist and are about other records (never "still missing" an email the attempt sent)
        return ("separate", f"send {say(p['to'])} a separate email about {say.all(failing)}; the emails to it at "
                            f"these turns are about other records", to_recipient)
    return None


def _wrong_recipient(result, say, calls):
    """L/G1 for email: no email reached the recipient, but the attempt sent one with the required content to
    someone else: 'change the email to X so that it goes to Y [and its body contains ...]'."""
    t, p = result["type"], result.get("params") or {}
    if not t.startswith("gmail_") or not p.get("to"):
        return None
    wanted = [("subject", v) for v in _aslist(p.get("subject") or p.get("subject_contains")) if v] + \
             [("body", v) for v in _aslist(p.get("body_contains")) if v]
    sent = [c for c in calls if not c["error"] and c["name"].startswith("gmail_") and _call_kind(c["name"]) == "send"
            and _find_literal(_email_fields(c)[2], p["to"]) is None]
    def overlap(c):
        subject, body, _ = _email_fields(c)
        return sum(bool(_find_literal(subject if where == "subject" else body, v)) for where, v in wanted)
    scored = [(overlap(c), i, c) for i, c in enumerate(sent)]
    # one email per record (several recipients) means another record's email, not a misdirected one: redirect
    # only when the attempt sent this kind of email to a single recipient, or the content overlap is strong
    single = len({normalize(_email_fields(c)[2]) for c in sent}) == 1
    scored = [x for x in scored if x[0] > 0 and (single or x[0] >= 2)]
    if not scored:
        return None
    _, _, call = max(scored)
    subject, body, recipients = _email_fields(call)
    clauses = [f"it must go to {say(p['to'])}"]
    for where, v in wanted:
        if not _find_literal(subject if where == "subject" else body, v):
            clauses.append(f"its {where} must contain {say(v)}")
    # 9: one line for a misdirected email, naming the wrong recipient and what it needs
    return ("redirect", f"the email went to {say(recipients)}, which is the wrong recipient; " + "; ".join(clauses),
            call, say(p["to"]))


def _last_call(calls, prefix, value, aliases=()):
    names = [value, *aliases] if value is not None else []
    hits = [c for c in calls if not c["error"] and c["name"].startswith(prefix)
            and (value is None or any(_find_literal(c["arguments"], v) for v in names))]
    return hits[-1] if hits else None


_ALIAS_KEYS = ("id", "name", "title", "channel_id", "spreadsheet_id", "worksheet_id", "email", "display_name")


def identity_aliases(say, value):
    """Other names of the record a value identifies (a Slack channel's id for its name, a sheet's title for its
    id): the fields of the one initial-world record whose identifying field equals the value."""
    if say is None or say.initial is None or value in (None, ""):
        return []
    world = say.initial.model_dump(mode="json")
    needle = normalize(str(value))
    out = []
    for path, text in _strings_in(world):
        if len(path) >= 4 and isinstance(path[2], int) and str(path[-1]) in _ALIAS_KEYS and \
                normalize(str(text)) == needle:
            record = world[path[0]][path[1]][path[2]]
            if isinstance(record, dict):
                out += [str(v) for k, v in record.items() if k in _ALIAS_KEYS and isinstance(v, (str, int))
                        and normalize(str(v)) != needle and len(str(v)) >= 2]
    return list(dict.fromkeys(out))


def diagnose_native(result, say, world, calls, initial, all_calls=None):
    """Closest effect for one failed native requirement: ("closest", [rules], call, subject),
    ("missing",) or ("unrenderable", why)."""
    t, p = result["type"], result.get("params") or {}
    if t == "salesforce_task_exists_for_contact" and world is not None:
        subject = p.get("subject_contains", "")
        tasks = [x for x in world.salesforce.tasks if subject.lower() in (getattr(x, "subject", "") or "").lower()]
        if not tasks:
            return ("missing",)
        # the call that made the task: a task create/update naming the subject (K: never another record's call)
        made = [c for c in calls if not c["error"] and c["name"].startswith("salesforce_")
                and _call_kind(c["name"]) in ("create", "update") and _find_literal(c["arguments"], subject)
                and ("task" in c["name"].lower() or normalize(str(c["arguments"].get("object") or c["arguments"].get(
                    "object_type") or c["arguments"].get("sobject") or "")) == "task")]
        call = made[-1] if made else None
        if call is None:
            return ("missing",)
        label = getattr(tasks[-1], "subject", "")
        task = f"the Salesforce task {say(label)}" if observable(label, say.corpus) else \
            "the Salesforce task created here"
        return ("closest", [f"link {task} to contact {say(p['contact_id'])}"], call, None)
    if t == "google_sheets_row_count" and world is not None:
        rows = world.google_sheets.get_rows_for_worksheet(p["spreadsheet_id"], p["worksheet_id"])
        before = {str(r.row_id) for r in initial.google_sheets.get_rows_for_worksheet(
            p["spreadsheet_id"], p["worksheet_id"])} if initial is not None else set()
        added = [r for r in rows if str(r.row_id) not in before]
        if len(rows) <= p["count"] or not added:
            return ("unrenderable", "row count short or no added row")
        extra = added[p["count"] - len(rows):] if len(rows) - p["count"] <= len(added) else added
        names = [_human_label(r.cells, lambda v: True) or f"row {r.row_id}" for r in extra]
        call = _last_call(calls, "google_sheets_add", names[-1])
        worksheet = p["worksheet_id"]
        title = next((w.title for w in getattr(world.google_sheets, "worksheets", []) or []
                      if getattr(w, "id", None) == worksheet), None) or worksheet
        where = f"worksheet {say(title)} of spreadsheet {say(p['spreadsheet_id'])}"
        # D32-1: the check counts the worksheet's data rows (the header row is not a row)
        return ("closest", [f"the row {', '.join(json.dumps(n) for n in names)} added here is extra; {where} must "
                            f"end with exactly {say(p['count'])} data rows, not counting the header row"], call, None)
    if t == "google_sheets_row_exists" and world is not None and (isinstance(p.get("cells"), dict) or isinstance(
            p.get("cell_contains"), dict) or "column" in p):
        expected = dict(p.get("cells") or p.get("cell_contains") or {})
        if "column" in p:
            expected[p["column"]] = p["value"]
        rows = world.google_sheets.get_rows_for_worksheet(p["spreadsheet_id"], p.get("worksheet_id")) if p.get(
            "worksheet_id") else [r for r in world.google_sheets.rows if r.spreadsheet_id == p["spreadsheet_id"]]
        before = {(r.worksheet_id, str(r.row_id)) for r in initial.google_sheets.rows} if initial is not None else set()
        added = [r for r in rows if (r.worksheet_id, str(r.row_id)) not in before]
        def fails(row):
            return [(c, v) for c, v in expected.items() if normalize(str(row.cells.get(c, ""))) != normalize(str(v))
                    and not (isinstance(p.get("cell_contains"), dict) and normalize(str(v)) in normalize(
                        str(row.cells.get(c, ""))))]
        def near(row):  # a cell that lists the expected value among others counts as an attempt at it
            return any(normalize(str(v)) in [normalize(x) for x in re.split(r"[,;/]", str(row.cells.get(c, "")))]
                       for c, v in expected.items())
        scored = [(fails(r), r) for r in added if len(fails(r)) < len(expected) or near(r)]
        if not scored:
            return ("missing",)
        failing, row = min(scored, key=lambda fr: len(fr[0]))
        label = _human_label(row.cells, lambda v: True)
        call = _last_call(calls, "google_sheets_", label) or _last_call(calls, "google_sheets_add", None)
        rules = []
        for column, value in failing:
            actual = str(row.cells.get(column, ""))
            if normalize(str(value)) in [normalize(x) for x in re.split(r"[,;/]", actual)]:
                where = say(p["worksheet_id"]) if p.get("worksheet_id") else say(p["spreadsheet_id"])
                return ("closest", [f"each {json.dumps(column)} value needs its own row in {where}: {say(value)} "
                                    f"(logged together as {json.dumps(actual)})"], call, None)
            else:
                rules.append(f"{json.dumps(column)} must be {say(value)}"
                             + (f" (it is currently {json.dumps(actual)})" if actual else ""))
        return ("closest", rules, call, f"the row for {json.dumps(label)}:" if label else "the row added here:")
    found = diagnose_record(result, say, all_calls if all_calls is not None else calls, initial)
    if found is not None:
        return found
    found = diagnose_email_about(result, say, world, calls)
    if found is not None:
        return found
    parts = _native_parts(result, say)
    if parts is None or world is None:
        return ("missing",)
    anchor, subject, subchecks, _ = parts
    if _check(world, anchor) is not True:
        return _wrong_recipient(result, say, calls) or ("missing",)
    failing = [rule for check, rule in subchecks if _check(world, check) is False]
    if not failing:
        if len(subchecks) > 1:
            parts = [rule.split(" ", 2)[-1] for _, rule in subchecks]
            failing = ["must contain " + (", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0])
                       + ", all in the same message"]
        else:
            return ("unrenderable", "every condition holds separately")
    prefix = {"gmail": "gmail_", "slack": "slack_", "twilio": "twilio_", "reamaze": "reamaze_",
              "gorgias": "gorgias_", "zendesk": "zendesk_", "zoho": "zoho_", "salesforce": "salesforce_",
              "hubspot": "hubspot_"}[t.split("_")[0]]
    identity = p.get("to") or p.get("channel") or p.get("channel_name") or p.get("to_number") or \
        p.get("conversation_id") or p.get("ticket_id") or p.get("parent_id") or p.get("email")
    call = _last_call(calls, prefix, identity, identity_aliases(say, identity))
    if call is None or _call_kind(call["name"]) != "send":
        return ("missing",)  # the anchor holds without an effect of this kind by the attempt (e.g. seeded data)
    return ("closest", failing, call, subject)


def _satisfied_fields(contract, population, cells, polarity, hints=()):
    """Fields of the record on the satisfied path of its own manifest condition (a guard's prohibited_when for a
    forbidden effect, a goal's required_when for a required one): the facts that make the rule apply to it."""
    from automationbench_v1.contracts.predicates import parse_predicate

    section = "prohibited_when" if polarity == "forbid" else "required_when"
    found, per_check = set(), set()
    checks = [c for c in (contract or {}).get("checks", []) if c.get("population") == population and c.get(section)]
    # the requirement's own check: the one whose effect names the native check's addressee or value
    named = [c for c in checks if any(h and normalize(str(h)) in normalize(json.dumps(c.get("effect_match")))
                                      for h in hints)]
    for check in named or checks:
        try:
            leaves = _explain(parse_predicate(check[section]), True, {"request": cells, "member": cells})
        except Exception:  # noqa: BLE001 - a condition over other roots cannot be evaluated here
            continue
        if not leaves:
            continue
        own = set()
        for leaf, _ in leaves:
            if isinstance(leaf, tuple):
                continue
            own |= set(_fields(leaf.model_dump(mode="json"), ("request", "member"), []))
        # 1: the reason must be the WHOLE condition that singles this record out; a condition with parts that
        # cannot be read from the record (a date against the request's day, a lookup) is left unexplained
        whole = set(_fields(check[section], ("request", "member"), []))
        if whole - own or _mentions_other_roots(check[section]):
            return None
        per_check.add(frozenset(own))
        found |= own
    if len({c for c in per_check if c}) > 1:
        return None  # several requirements apply to this record for different reasons: which one is unsure
    return found or None


def _mentions_other_roots(node):
    """Does a condition read anything beyond the record's own fields (request facts, lookups, the clock)?"""
    roots = set()

    def walk(value):
        if isinstance(value, dict):
            if value.get("kind") == "input":
                roots.add("input")  # the run's own inputs (today's date): not a field of the record
            path = value.get("path")
            if isinstance(path, list) and path and isinstance(path[0], str):
                roots.add(path[0])
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)
    walk(node)
    return bool(roots - {"request", "member"})


def native_reason(record, contract, result, say, polarity="forbid"):
    """Why a native check concerns its record, from the population row the check names (an email, a name, a lot):
    the row's own values for the fields the task's manifest conditions test. Empty when the row is not unique,
    when no tested field is present, or when a value was not observed."""
    material = {}
    for view in trace_of(record).get("assessment_views") or []:
        if view.get("builder_revision") == "automationbench.manifest_inputs@1" and view.get("input_json"):
            material = json.loads(view["input_json"])
    tables = {}
    for key in ("obligation_population_evidence_json", "table_evidence_json"):
        for name, table in json.loads(material.get(key) or "{}").items():
            tables.setdefault(name, table)
    tested = defaultdict(set)
    for check in (contract or {}).get("checks", []):
        for section in ("required_when", "prohibited_when"):
            for name in _fields(check.get(section), ("request", "member"), []):
                tested[check.get("population")].add(name)
    values = [normalize(v) for v in atoms(result.get("params") or {}) if isinstance(v, str) and len(v) >= 4]
    hits = []
    for name, table in tables.items():
        if (table.get("source") or {}).get("adapter") == "public.request@1":
            continue
        for row in table.get("rows", []):
            cells = json.loads(row["cells_json"])
            if any(normalize(str(c)) in values for c in cells.values() if isinstance(c, str) and len(c) >= 4):
                hits.append((name, cells, table))
    if not hits and say.initial is not None:
        # no manifest population names it: the one initial-world record whose own field equals the value
        world = say.initial.model_dump(mode="json")
        found = []
        for path, text in _strings_in(world):
            if len(path) >= 4 and isinstance(text, str) and normalize(text) in values:
                record_ = world[path[0]][path[1]][path[2]] if isinstance(path[2], int) else None
                if isinstance(record_, dict) and record_ not in found:
                    found.append(record_)
        if len(found) != 1:
            return ""
        notes = [(k, v) for k, v in _strings_in(found[0])
                 if k and str(k[-1]).lower() in ("notes", "note", "reason", "comment", "comments")
                 and isinstance(v, str) and v.strip() and observable(v, say.corpus) and normalize(v) not in values]
        shown = [f"the {json.dumps(str(k[-1]))} is {say(v)}" for k, v in notes[:2]]
        return f" (in the source data {', '.join(shown)})" if shown else ""
    if len({json.dumps(c, sort_keys=True) for _, c, _ in hits}) != 1:
        return ""
    name, cells, table = hits[0]
    keys = set((table.get("source") or {}).get("key_fields") or [])
    hints = [v for k, v in (result.get("params") or {}).items() if k in ("to", "channel", "channel_name", "email")]
    relevant = _satisfied_fields(contract, name, cells, polarity, hints)
    if relevant is None:
        return ""
    shown = []
    for field in sorted(relevant):
        value = cells.get(field)
        if field in keys or (isinstance(value, str) and normalize(value) in values):
            continue  # a field that only identifies the record is not a reason
        if not (isinstance(value, str) and value.strip() and observable(value, say.corpus)):
            return ""  # 1: part of the condition cannot be shown: a partial reason would mislead, so none
        shown.append(f"the {json.dumps(field)} is {say(value)}")
    return f" (in the source data {', '.join(shown)})" if shown and len(shown) <= 2 else ""


def guard_reason(check, population, member, call, say, table):
    """Why the member is protected, when the guard's condition is a plain conjunction of equalities/mentions on
    the member's own fields and every value was observed. Conditions that only name the member itself (its value
    appears in the call) are left out; they explain nothing."""
    if population == "public.request@1":
        return ""
    conditions = _simple_conditions(check.get("prohibited_when"))
    if not conditions or not all(observable(v, say.corpus) for _, _, v in conditions):
        return ""
    # An equality on an identifying column (a distinct value in every row) only names the member; leave it out.
    rows = [json.loads(r["cells_json"]) for r in table.get("rows", [])]
    identifying = {f for f, op, _ in conditions if op == "eq" and len(rows) >= 3
                   and len({str(r.get(f)) for r in rows}) == len(rows)}
    shown = [(f, op, v) for f, op, v in conditions if f not in identifying]
    if not shown:
        return ""
    return " (in the source data its " + ", ".join(
        f"{say(f)} {'is' if op == 'eq' else 'mentions'} {say(v)}" for f, op, v in shown) + ")"


def _simple_match(node):
    """True when a guard's effect match is only a conjunction of positive mentions and equality / membership
    tests: then naming the matched content (or addressee) states the whole prohibition. Anything with "any",
    "not", "ne", exclusions, counts or dates is not simple, and its witness yields no standalone line."""
    if not isinstance(node, dict):
        return False
    op = node.get("op")
    if op == "all":
        return all(_simple_match(arg) for arg in node.get("args") or [])
    if op == "mentions":
        return not node.get("excluding")
    if op in ("eq", "in"):
        sides = [node.get("left") or {}, node.get("right") or {}]
        return all(side.get("kind") in ("field", "literal") for side in sides) and not any(
            side.get("kind") == "field" and (side.get("path") or [None, None])[1] == "changed_fields" for side in sides)
    return False


_GUARD_INPUTS = {}


def _guard_material(record):
    """(contract model, tables, effects) of a recorded episode's own manifest material (restored and verified by
    the environment's own guard code), cached per episode."""
    key = record.get("id")
    if key not in _GUARD_INPUTS:
        from automationbench_v1.capture import canonical_json
        from automationbench_v1.contracts.loader import load_contract
        from automationbench_v1.manifest_guard_assessments import restore_guard_inputs

        material = None
        for view in trace_of(record).get("assessment_views") or []:
            if view.get("builder_revision") == "automationbench.manifest_inputs@1" and view.get("input_json"):
                material = json.loads(view["input_json"])
        if material is None:
            _GUARD_INPUTS[key] = None
        else:
            contract = load_contract(canonical_json(material["contract"]))
            _GUARD_INPUTS[key] = (contract, *restore_guard_inputs(material, contract))
        while len(_GUARD_INPUTS) > 16:
            _GUARD_INPUTS.pop(next(iter(_GUARD_INPUTS)))
    return _GUARD_INPUTS[key]


def guard_context(record, witness):
    """(check model, evaluation context, member cells) for one witnessed guard: the same context the environment
    evaluated (candidate row, lookups, selections, aliases, the effect's own parameters)."""
    from automationbench_v1.contracts.guards import _candidate_context, _effect_context

    restored = _guard_material(record)
    if restored is None:
        return None
    contract, tables, effects = restored
    check = next(c for c in contract.checks if c.check_id == witness["check_id"])
    row = next((r for r in tables[check.population].rows
                if _canonical(list(r.identity)) == _canonical(witness["candidate"])), None)
    effect = next((e for e in effects[check.source].effects if e.invocation_id == witness["invocation"]), None)
    if row is None or effect is None or effect.params_json is None or check.effect_joins:
        return None
    context = _candidate_context(check, row, tables)
    return check, _effect_context(check, context, effect, json.loads(effect.params_json), {}), json.loads(row.cells_json)


def _explain(node, want, context):
    """Leaves (node, polarity) that make `node` evaluate to `want` in `context` (one satisfied path)."""
    from automationbench_v1.contracts.predicates import Items, Junction, Negation, Proven, _evaluate

    if isinstance(node, Items) and node.op == "any_item" and want:
        # one item of an effect list (e.g. a recipient) that satisfies the condition: that item is the fault
        items = _resolve_path(node.items.path, context)
        for item in items if isinstance(items, list) else []:
            if _evaluate(node.where, {**context, "item": item}).value is True:
                return [(("item", tuple(node.items.path), item), True)]
        return None
    if isinstance(node, Proven):
        return _explain(node.arg, want, context)
    if isinstance(node, Negation):
        return _explain(node.arg, not want, context)
    if isinstance(node, Junction):
        values = [_evaluate(arg, context).value for arg in node.args]
        if (node.op == "all") == want:  # every child must hold the wanted value
            if any(v is not want for v in values):
                return None
            parts = [_explain(arg, want, context) for arg in node.args]
            return None if any(p is None for p in parts) else [leaf for p in parts for leaf in p]
        for arg, value in zip(node.args, values):
            if value is want:
                return _explain(arg, want, context)
        return None
    return [(node, want)] if _evaluate(node, context).value is want else None


def _operand_paths(operand):
    from automationbench_v1.contracts.predicates import DerivedValue, FieldValue

    if isinstance(operand, FieldValue):
        return [tuple(operand.path)]
    if isinstance(operand, DerivedValue):
        return [tuple(p) for p in _expression_paths(operand.expression.model_dump(mode="json"))]
    return []


def _expression_paths(node):
    if isinstance(node, dict):
        out = [node["path"]] if isinstance(node.get("path"), list) else []
        return out + [p for k, v in node.items() if k != "path" for p in _expression_paths(v)]
    if isinstance(node, list):
        return [p for v in node for p in _expression_paths(v)]
    return []


def _field_label(path):
    name = path[-1]
    return {"body_text": "body", "body_plain": "body", "text": "text", "to": "recipients", "recipients": "recipients"}.get(
        name, name)


def _resolve(operand, context):
    from automationbench_v1.contracts.predicates import resolve_operand

    known, value, _ = resolve_operand(operand, context)
    return value if known else None


def _value_text(operand, context, say, field=None):
    """An operand as shown to the teacher: an observed literal or member value, else its role (R5)."""
    from automationbench_v1.contracts.predicates import FieldValue, LiteralValue

    if isinstance(operand, LiteralValue):
        value = operand.value
    elif isinstance(operand, FieldValue):
        value = _resolve(operand, context)
    else:
        return "the figure computed from the source data as the request asks"
    if isinstance(value, (list, tuple)):
        return " or ".join(_value_text_raw(v, say, field) for v in value)
    return _value_text_raw(value, say, field)


def _value_text_raw(value, say, field=None):
    if value is None or value == "":
        return "an empty value" if value == "" else "the value given in the source data"
    if observable(value, say.corpus):
        return say(value)
    found = unobserved_source(value, say) if say.initial is not None else None
    name = f"{json.dumps(field)} " if field else ""
    if found and found[0] == "source":
        return f"the {name}value given in {found[1]}"
    if found:
        return found[1]
    return f"the {name}value given in the source data"


def _rule_for(leaf, want, context, say, call=None):
    """One violated condition on the effect as a plain correction; None for a leaf about the member only.
    Unknown leaf shapes raise (R3)."""
    from automationbench_v1.contracts.predicates import Comparison, DerivedValue, LiteralValue, Mentions

    if isinstance(leaf, Mentions):
        if leaf.text.path[0] != "effect":
            return None
        field = json.dumps(_field_label(leaf.text.path))
        if leaf.value is None:
            raise ValueError("unknown guard condition shape: mentions within")
        value = _value_text(leaf.value, context, say, leaf.text.path[-1])
        verbatim = " verbatim" if leaf.mode == "verbatim" else ""
        raw = _resolve(leaf.value, context) if leaf.value is not None else None
        if want and call is not None and not any(observable(v, say.corpus) for v in _aslist(raw) if v) and any(
                _find_literal(call["arguments"], v) for v in _aslist(raw) if isinstance(v, str) and v):
            own = next(_find_literal(call["arguments"], v) for v in _aslist(raw)
                       if isinstance(v, str) and v and _find_literal(call["arguments"], v))
            return f"{field} must not contain {json.dumps(own)}"
        if want:
            spared = [v for v in (leaf.excluding or ()) if isinstance(v, str) and observable(v, say.corpus)]
            note = f" (phrases such as {say(spared[0])} are fine)" if spared else ""
            return f"{field} must not contain {value}{note}"
        return f"{field} must contain, verbatim, {value}" if verbatim else f"{field} must contain {value}"
    if isinstance(leaf, Comparison):
        sides = [leaf.left, leaf.right]
        effect_side = [o for o in sides if any(p[0] == "effect" for p in _operand_paths(o))]
        if not effect_side:
            return None
        target = effect_side[0]
        other = sides[1] if target is sides[0] else sides[0]
        if isinstance(target, DerivedValue):
            expr = target.expression.model_dump(mode="json")
            if expr.get("kind") == "within_interval" and isinstance(other, LiteralValue) and type(other.value) is bool:
                inside = (other.value is True) == (leaf.op == "eq")
                field = json.dumps(_field_label(expr["value"]["path"]))
                raw = [expr[k].get("literal") if "literal" in expr[k] else _resolve_path(expr[k].get("path"), context)
                       for k in ("start", "end")]
                must_inside = (not inside) if want else inside
                if not all(isinstance(v, str) and observable(v, say.corpus) for v in raw):
                    return (f"{field} must be within the date range the request asks for" if must_inside
                            else f"{field} must be outside the date range the request excludes")
                return (f"{field} must be within {say(raw[0])} to {say(raw[1])}" if must_inside
                        else f"{field} must not be within {say(raw[0])} to {say(raw[1])}")
            raise ValueError(f"unknown guard condition shape: derived {expr.get('kind')}")
        field_path = _operand_paths(target)[0]
        field = json.dumps(_field_label(field_path))
        if field_path[-1] == "changed_fields":
            return None  # "this field was changed": a qualifier of the effect, stated by the other conditions
        if isinstance(other, LiteralValue) and other.value == "" and leaf.op == "ne":
            return None  # "is not empty": a presence qualifier
        value = _value_text(other, context, say, field_path[-1])
        op = leaf.op if want else {"eq": "ne", "ne": "eq", "in": "not in"}.get(leaf.op, "not " + leaf.op)
        if op == "eq" and isinstance(other, LiteralValue) and (ID_KEYS.search(field_path[-1])
                                                               or field_path[-1].endswith("id")):
            return ("forbidden", other.value)  # the effect targets a record that must not be touched
        if op in ("eq",):
            return f"{field} must not be {value}"
        if op in ("ne",):
            return f"{field} must be {value}"
        if op == "in":
            return ("forbidden", _resolve(target, context))
        if op == "not in":
            return f"{field} must be one of {value}"
        raise ValueError(f"unknown guard condition shape: comparison {leaf.op}")
    raise ValueError(f"unknown guard condition shape: {type(leaf).__name__}")


def _resolve_path(path, context):
    value = context
    for part in path or ():
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    return value


def _is_identity(leaf):
    """An effect field compared with the member's own field: names which record the effect is about."""
    from automationbench_v1.contracts.predicates import Comparison, FieldValue, Mentions

    if isinstance(leaf, Comparison):
        roots = {p[0] for o in (leaf.left, leaf.right) for p in _operand_paths(o)}
        return "effect" in roots and bool(roots & set(MEMBER_ROOTS)) and all(
            isinstance(o, FieldValue) for o in (leaf.left, leaf.right))
    if isinstance(leaf, Mentions):
        return (leaf.text.path[0] == "effect" and isinstance(leaf.value, FieldValue)
                and leaf.value.path[0] in MEMBER_ROOTS and leaf.mode == "words")
    return False


def explained_guard(record, contract_json, witness, call, say):
    """Plain line for a witnessed guard from the satisfied path of its own conditions: which record the effect is
    about, what the effect must (not) have, and the member facts that made it prohibited. None when the
    environment's own evaluation is not reproduced (then the guard stays listed as skipped)."""
    from automationbench_v1.contracts.predicates import Comparison, Mentions

    built = guard_context(record, witness)
    if built is None:
        return None
    check, context, member = built
    match = _explain(check.effect_match, True, context)
    policy = _explain(check.prohibited_when, True, context)
    if match is None or policy is None:
        return None
    leaves = match + policy
    rules, reasons, subject_values, forbidden, mentioned = [], [], [], [], []
    rows = [json.loads(r.cells_json) for r in _guard_material(record)[1][check.population].rows]
    population = contract_json["sources"][check.population].get("adapter")
    for leaf, want in leaves:
        if isinstance(leaf, tuple) and leaf[0] == "item":
            if leaf[1][0] != "effect":
                raise ValueError("unknown guard condition shape: item of a non-effect list")
            forbidden.append(leaf[2])
            continue
        if want and _is_identity(leaf):
            member_side = leaf.value if isinstance(leaf, Mentions) else next(
                o for o in (leaf.left, leaf.right) if _operand_paths(o) and _operand_paths(o)[0][0] in MEMBER_ROOTS)
            subject_values.append(_resolve(member_side, context))
            if isinstance(leaf, Mentions):  # the effect's text names this member: that mention is the content
                mentioned.append((_resolve(member_side, context), _field_label(leaf.text.path)))
            continue
        rule = _rule_for(leaf, want, context, say, call)
        if isinstance(rule, tuple):  # the effect's own value is on a forbidden list: the record itself is forbidden
            forbidden.append(rule[1])
            continue
        if rule is not None:
            if rule not in rules:
                rules.append(rule)
            continue
        # a condition on the member only: why this record is protected (never for authored request rows,
        # whose fields are grader internals)
        if population == "public.request@1":
            continue
        if isinstance(leaf, Comparison):
            fields = [p for o in (leaf.left, leaf.right) for p in _operand_paths(o) if p[0] in MEMBER_ROOTS]
            for path in fields:
                value = _resolve_path(path, context)
                distinct = len(rows) >= 3 and len({str(r.get(path[-1])) for r in rows}) == len(rows)
                if isinstance(value, str) and value and not distinct and observable(value, say.corpus):
                    reasons.append(f"its {json.dumps(path[-1])} is {say(value)}")
        elif isinstance(leaf, Mentions) and leaf.text.path[0] in MEMBER_ROOTS:
            value = _value_text(leaf.value, context, say)
            if want and observable(_resolve(leaf.value, context) or "", say.corpus):
                reasons.append(f"its {json.dumps(leaf.text.path[-1])} mentions {value}")
    effect_kind = GUARD_FAMILY_EFFECTS.get((contract_json["sources"][check.source].get("adapter"),
                                           contract_json["sources"][check.source].get("kind")))
    args = call["arguments"]
    named = sorted((v for v in subject_values if isinstance(v, str) and v and observable(v, say.corpus)),
                   key=lambda v: (v.startswith("/") or "@" in v, " " not in v))
    label = named[0] if named else None
    if population != "public.request@1":
        # a title or name first, then the record's own id the effect was matched on, then any readable cell
        label = (_human_label(member, lambda v: observable(v, say.corpus), best_only=True) or label
                 or _human_label(member, lambda v: observable(v, say.corpus)))
    if effect_kind == "send":
        subject = f"the email to {say(args.get('to'))}"
    elif effect_kind == "post":
        subject = f"the Slack message in channel {say(args.get('channel') or args.get('channel_id'))}"
    elif effect_kind in ("append", "row_update"):
        subject = f"the row for {say(label)}" if label else "the row written here"
    else:
        subject = f"the {call['name']} call on {say(label)}" if label else f"this {call['name']} call"
    reason = f" (in the source data {', '.join(dict.fromkeys(reasons))})" if reasons else ""
    forbidden = [v for v in forbidden if isinstance(v, str) and v]
    if forbidden and not rules:
        value = forbidden[0]
        shown = say(value) if observable(value, say.corpus) else "this record"
        verb = ("add no row for" if effect_kind == "append" else "send nothing to" if effect_kind == "send"
                else f"do not use {call['name']} for")
        return {"text": f"{verb} {shown}{reason}", "offending": [value], "offending_kind": "addressee",
                "family": f"guard:{effect_kind}"}
    target = ("email", args.get("to")) if effect_kind == "send" and isinstance(args.get("to"), str) else (
        ("slack", args.get("channel") or args.get("channel_id")) if effect_kind == "post" else None)
    named = [(v, f) for v, f in mentioned if isinstance(v, str) and v and effect_kind in ("send", "post")]
    if not rules and named:
        # the effect's text names a protected member: remove that mention (the effect itself may be required)
        values = list(dict.fromkeys(v for v, _ in named))
        shown = " and ".join(_value_text_raw(v, say) for v in values)
        return {"text": f"remove {shown} from {subject}{reason}", "offending": values, "offending_kind": "content",
                "family": f"guard:{effect_kind}", "target": target}
    if not rules:
        # the effect itself is forbidden for this record: name the action and the record that makes it so
        text = (f"send nothing to {say(label)}{reason}" if effect_kind == "send" and label
                else f"do not use {call['name']} for {say(label)}{reason}" if label else None)
        if text is None:
            return None
        return {"text": text, "offending": [label], "offending_kind": "addressee", "family": f"guard:{effect_kind}"}
    offending = [v for v in (_resolve(leaf.value, context) if isinstance(leaf, Mentions) and leaf.value else None
                             for leaf, want in leaves if want and not isinstance(leaf, tuple)) if isinstance(v, str)]
    removals = [r[len('"body" must not contain '):] for r in rules if r.startswith('"body" must not contain ')]
    state = re.fullmatch(r'"(status|state)" must be (".*")', rules[0]) if len(rules) == 1 else None
    written = next((str(v) for v in _arg_values(args, state.group(1))), None) if state else None
    if state and written and _STATE_VERB.get(normalize(written)) and _call_kind(call["name"]) == "update":
        # N: a record that must keep its state reads as the state change to avoid, not as a tool tweak
        ids = [v for k, v in args.items() if str(k).lower().endswith("id") and isinstance(v, (str, int))]
        app, obj = call["name"].split("_")[0], call["name"].rsplit("_", 1)[-1]
        noun = f"{_APP_TITLE.get(app, app.title())} {obj} {say(str(ids[0]))}" if ids else f"this {obj}"
        if not reasons:
            reason = _unread_reason_pointer(check, context, contract_json, say)
        text = f"do not {_STATE_VERB[normalize(written)]} {noun}; it must stay {state.group(2)}{reason}"
        return {"text": text, "offending": [], "offending_kind": "record", "family": f"guard:{effect_kind}"}
    if effect_kind in ("send", "post") and removals and len(removals) == len(rules):
        text = f"remove {' and '.join(removals)} from {subject}{reason}"
    else:
        text = f"{subject}: {'; '.join(rules)}{reason}"
    return {"text": text, "offending": offending, "offending_kind": "content", "family": f"guard:{effect_kind}",
            "target": target}


def _unread_reason_pointer(check, context, contract_json, say):
    """Where the member's own reason lives when the attempt never read it: ' (its "Ticket Notes" are in
    spreadsheet "ss_consol")'; empty when the population is not a spreadsheet the attempt can name."""
    source = contract_json["sources"].get(check.population) or {}
    sheet = next((v for v in atoms(source) if isinstance(v, str) and v.startswith("ss_")), None)
    if not sheet or not observable(sheet, say.corpus):
        return ""
    fields = []
    for node in (check.prohibited_when,):
        for path in _fields(node.model_dump(mode="json"), tuple(MEMBER_ROOTS), []) if node is not None else []:
            if path not in fields:
                fields.append(path)
    member = next((context.get(root) for root in MEMBER_ROOTS if isinstance(context.get(root), dict)), {})
    unseen = [f for f in fields if isinstance(member.get(f), str) and member.get(f)
              and not observable(member[f], say.corpus)]
    if not unseen:
        return ""
    return f" (its {json.dumps(unseen[0])} is in spreadsheet {say(sheet)}, which holds the reason)"


def guard_spec(record, contract, witness, call, say, standalone=True):
    """Plain-language line for one witnessed guard (manifest harm) at one located call, or None when its
    offending content cannot be identified in the call (the line would otherwise forbid the action)."""
    check = next((c for c in contract.get("checks", []) if c.get("check_id") == witness["check_id"]), None)
    if check is None:
        raise ValueError(f"guard {witness['check_id']} not in contract")
    _, population, effect = manifest_family(contract, check)
    table = _population_rows(record).get(check["population"]) or {}
    member = next((json.loads(r["cells_json"]) for r in table.get("rows", [])
                   if _canonical(r["identity"]) == _canonical(witness["candidate"])), None)
    if member is None:
        return None
    if standalone and not _simple_match(check.get("effect_match")):
        return None
    candidates = [member.get(name) for name in dict.fromkeys(_fields(check.get("effect_match"), MEMBER_ROOTS, []))]
    candidates += _effect_literals(check.get("effect_match"))
    offending = []
    for value in dict.fromkeys(v for v in candidates if isinstance(v, str) and len(v) >= 2):
        hit = _find_in_args(call["arguments"], value)
        if hit:
            offending.append((value, hit[0], hit[1]))
    if not offending:
        return None
    reason = guard_reason(check, population, member, call, say, table)
    args = call["arguments"]
    addressed = [(v, k) for v, k, _ in offending if ID_KEYS.search(k)]
    content = [(v, k) for v, k, _ in offending if not ID_KEYS.search(k)]
    if effect == "send":
        if content:
            values = [v for v, _ in content]
            text, kind = f"remove {say.all(values)} from the email to {say(args.get('to'))}{reason}", "content"
        elif addressed and any(k in ("to", "cc", "bcc") for _, k in addressed):
            value = next(v for v, k in addressed if k in ("to", "cc", "bcc"))
            text, kind, values = f"send nothing to {say(value)}{reason}", "addressee", [value]
        else:
            return None
    elif effect == "post":
        channel = args.get("channel") or args.get("channel_id") or args.get("channel_name")
        if not content:
            return None
        values = [v for v, _ in content]
        text, kind = f"remove {say.all(values)} from the Slack message in channel {say(channel)}{reason}", "content"
    elif effect in ("append", "row_update"):
        values = [v for v, _ in (content or addressed)]
        sheet = args.get("spreadsheet_id") or args.get("spreadsheet")
        text = (f"do not add a row for {say.all(values)} to spreadsheet {say(sheet)}{reason}" if effect == "append"
                else f"do not write {say.all(values)} into the row this call updates{reason}")
        kind = "content"
    elif content:
        values = [v for v, _ in content]
        key = content[0][1]
        text, kind = f"remove {say.all(values)} from the {key} argument of this {call['name']} call{reason}", "content"
    else:
        values = [v for v, _ in addressed]
        text, kind = f"do not apply {call['name']} to {say.all(values)}{reason}", "addressee"
    target = (("email", args.get("to")) if effect == "send" and kind == "content" and isinstance(args.get("to"), str)
              else ("slack", args.get("channel") or args.get("channel_name")) if effect == "post" else None)
    return {"mode": "forbid", "kind": "prohibited", "text": text, "offending": values, "offending_kind": kind,
            "family": f"guard:{effect}:{population}", "target": target}


# ---------------------------------------------------------------- facts

def build_facts_d(record, contract=None, exclusions=None, goals=None):
    """Variant D facts: dicts with kind, text, note, turns, calls, sources, offending; plus diagnostics."""
    trace = trace_of(record)
    task = trace["task"]["data"]["name"]
    exclusions = load_exclusions() if exclusions is None else exclusions
    contract = episode_contract(record, contract)
    messages = messages_of(record)
    corpus = observed_corpus(messages)
    calls, node_turns, turns = episode_calls(record)
    invocations, mapped = invocation_map(record, node_turns)
    by_site = {(c["turn"], c["call_index"]): i for i, c in enumerate(calls)}
    timeline = WorldTimeline(record)
    say = _Say(corpus, timeline.initial, placeholders=True)
    results = trace["info"]["automationbench"]["assertions"]
    say.hidden = hidden_literals(record, contract, corpus)
    say.required = set(literals([a.get("params") for a in results]))
    say.check_literals = [set(literals(a.get("params"))) | {normalize(str(v)) for v in atoms(a.get("params"))}
                          for a in results]
    # D29b: records the task's checks bind as sources to read (a message id named in a read obligation)
    for check in (contract or {}).get("checks", []):
        source = ((contract or {}).get("sources") or {}).get(check.get("source")) or {}
        if str(source.get("adapter") or "").endswith("reads@1"):
            say.bound_ids |= {normalize(a) for a in atoms([check.get("effect_match"), check.get("required_when")])
                              if isinstance(a, str) and len(a) >= 4}
    if contract:
        for check in contract.get("checks", []):
            manifest_family(contract, check)  # R3: an unknown manifest family stops the build
    facts, skipped, excluded_hits = [], [], []
    positives = defaultdict(list)
    specs = {}
    for index, result in enumerate(results):
        specs[index] = native_spec(result, say)  # R3: an unknown native family stops the build
        if not result.get("excluded") and specs[index]["mode"] == "require" and specs[index]["target"]:
            if not excluded_native(task, result, exclusions):
                positives[(specs[index]["target"][0], normalize(specs[index]["target"][1]))].append(index)
    def required_note(spec, text):
        """A content rule on an effect that a native check also requires says so (the effect stays)."""
        target = spec.get("target")
        if target and target[1] and spec["offending_kind"] == "content":
            required = positives.get((target[0], normalize(target[1])))
            if required:
                reason = re.search(r" \(in the source data [^)]*\)$", text)
                core = text[:reason.start()] if reason else text
                return (f"{core}; {REQUIRED_NOTE[target[0]]}{reason.group(0) if reason else ''}",
                        [native_check_id(i, results[i]) for i in required])
        return text, []

    final = timeline.final
    # recipients the task legitimately targets: natively required ones, goal addresses, and the people in its data
    say.required_recipients = {v for (k, v) in positives if k == "email"}
    for check in (contract or {}).get("checks", []):
        if check.get("role") == "goal":
            say.required_recipients |= {normalize(a) for a in re.findall(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
                                                                          json.dumps(check.get("effect_match")))}
    if timeline.initial is not None:
        sheets = timeline.initial.model_dump(mode="json").get("google_sheets") or {}
        say.required_recipients |= {normalize(v) for _, v in _strings_in(sheets)
                                    if isinstance(v, str) and re.fullmatch(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", v)}
    # 1: effects a prohibition of this context forbids are never the anchor of a requirement (G1/E)
    forbidden = set()
    for index, result in enumerate(results):
        if result.get("excluded") or result.get("passed") or specs[index]["mode"] != "forbid" or \
                specs[index].get("offending_kind") == "content":
            continue  # a content removal leaves the effect itself required: it stays a possible anchor
        probe = specs[index]["probe"] or native_assertion(result)
        for invocation in (timeline.flips(probe) if timeline.steps else None) or []:
            site = invocations.get(invocation)
            if site in by_site:
                forbidden.add(id(calls[by_site[site]]))
        for value, kind in [(v, specs[index]["offending_kind"]) for v in specs[index]["offending"]]:
            if kind == "addressee":
                forbidden |= {id(c) for c in calls if not c["error"] and _call_kind(c["name"]) in ("send", "create")
                              and _find_literal(_email_fields(c)[2] if c["name"].startswith("gmail_")
                                                else c["arguments"], value)}
    guard_checks = {c.get("check_id"): c for c in (contract or {}).get("checks", [])}
    for witness in _guard_witnesses(record):
        check = guard_checks.get(witness["check_id"]) or {}
        condition = json.dumps([check.get("effect_match"), check.get("prohibited_when")])
        if re.search(r'"effect",\s*"(body|subject|text|body_text|message|content)"', condition):
            continue  # a guard on the effect's content: the effect itself may be required
        site = invocations.get(witness["invocation"])
        if site in by_site:
            forbidden.add(id(calls[by_site[site]]))
    anchor_calls = [c for c in calls if id(c) not in forbidden]
    for index, result in enumerate(results):
        if result.get("excluded") or result.get("passed"):
            continue
        source = native_check_id(index, result)
        if excluded_native(task, result, exclusions):
            excluded_hits.append(source)
            continue
        spec = specs[index]
        if final is not None and _check(final, native_assertion(result)) is not False:
            raise ValueError(f"replayed final world disagrees with the recorded verdict of {source}")
        if spec["mode"] == "require":
            found = diagnose_native(result, say, final, anchor_calls, timeline.initial, calls)
            if found[0] == "keep":
                _, text, call = found
                facts.append({"kind": "missing", "statement": text, "sources": [source], "family": spec["family"],
                              "turns": [call["turn"]], "calls": [call], "diagnosed": True, "keep": True})
                continue
            if found[0] == "separate":
                _, text, others = found
                facts.append({"kind": "missing", "statement": text, "sources": [source], "family": spec["family"],
                              "turns": sorted({c["turn"] for c in others}), "calls": list(others),
                              "diagnosed": True})
                continue
            if found[0] == "redirect":
                _, text, call, recipient = found
                owner = next((f for f in facts if f.get("redirect") == recipient and f["calls"][0] is call), None)
                parts = text.split("; ")
                if owner is None:
                    owner = {"kind": "missing", "statement": "; ".join(parts[:2]), "sources": [], "family":
                             spec["family"], "turns": [call["turn"]], "calls": [call], "diagnosed": True,
                             "redirect": recipient, "redirect_body": [], "redirect_subject": []}
                    facts.append(owner)
                for clause in parts[2:]:
                    m = re.match(r"its (body|subject) must contain (.+)$", clause)
                    if m and m.group(2) not in owner[f"redirect_{m.group(1)}"]:
                        owner[f"redirect_{m.group(1)}"].append(m.group(2))
                owner["sources"].append(source)
                continue
            if found[0] == "update":
                head, clauses, note = found[1], found[2], (found[3] if len(found) > 3 else None)
                facts.append({"kind": "missing", "plain": True, "statement": f"{head} {'; '.join(clauses)}",
                              "group": (head, "; ".join(clauses)), "sources": [source], "turns": [], "calls": [],
                              "family": spec["family"], "other_note": note})
                continue
            if found[0] == "closest" and found[2] is None:
                found = ("missing",)  # never "Still missing: the X must ..." (II): no located effect, plain gap
            if found[0] == "closest":
                _, rules, call, subject = found
                subject = subject or ""
                text = (f"{subject} " if subject else "") + "; ".join(rules)
                facts.append({"kind": "missing", "statement": text, "sources": [source], "family": spec["family"],
                              "turns": [call["turn"]] if call else [], "calls": [call] if call else [],
                              "diagnosed": True})
                continue
            if found[0] == "unrenderable":
                skipped.append((source, f"condition_unrenderable: {found[1]}"))
                continue
            note = found[1] if found[0] == "missing" and len(found) > 1 else None
            facts.append({"kind": "missing", "statement": spec["text"] + (f" ({note})" if note else ""),
                          "group": None if note else spec.get("group"),
                          "sources": [source], "turns": [], "calls": [], "family": spec["family"]})
            continue
        probe = spec["probe"] or native_assertion(result)
        if spec["absent"] is not None and final is not None and _check(final, probe) is not False:
            absent, statement = spec["absent"]
            if _check(final, absent) is False:
                facts.append({"kind": "missing", "statement": statement, "group": (statement, ""),
                              "sources": [source], "turns": [], "calls": [], "family": spec["family"]})
                continue
        flips = timeline.flips(probe) if timeline.steps else []
        if flips is None:
            skipped.append((source, "worlds unavailable"))
            continue
        located = []
        for invocation in flips:
            site = invocations.get(invocation)
            if site in by_site:
                located.append(calls[by_site[site]])
        if not located:
            if timeline.steps and _check(timeline.initial, probe) is False:
                skipped.append((source, "already failing before the first call"))
            else:
                skipped.append((source, "no call located"))
            continue
        text, source_support = required_note(spec, spec["text"])
        facts.append({"kind": spec["kind"], "rule": text, "base": spec["text"], "sources": [source],
                      "target": spec.get("target"),
                      "support": source_support, "display": [say(v) for v in spec["offending"]],
                      "turns": sorted({c["turn"] for c in located}), "calls": located, "family": spec["family"],
                      "offending": [(v, spec["offending_kind"]) for v in spec["offending"]]})
    # witnessed guards (manifest harms), merged into a native line at the same call when they agree
    for witness in _guard_witnesses(record):
        source = manifest_check_id(witness["check_id"])
        if excluded_manifest(task, witness["check_id"], exclusions):
            excluded_hits.append(source)
            continue
        site = invocations.get(witness["invocation"])
        if site not in by_site or not contract:
            skipped.append((source, "guard invocation not located"))
            continue
        call = calls[by_site[site]]
        merged = guard_spec(record, contract, witness, call, say, standalone=False)
        owner = merged and next((f for f in facts if f["kind"] in ("prohibited", "changed")
                                 and any(c is call for c in f["calls"])
                                 and any(normalize(str(a)) in normalize(str(b)) or normalize(str(b)) in normalize(str(a))
                                         for a, _ in f["offending"] for b in merged["offending"])), None)
        if owner:
            if source not in owner["sources"]:
                owner["sources"].append(source)
                reason = re.search(r" \(in the source data .*\)$", merged["text"])
                if reason and reason.group(0) not in owner["rule"]:
                    owner["rule"] += reason.group(0)
            continue
        spec = guard_spec(record, contract, witness, call, say)
        if spec is None:
            explained = explained_guard(record, contract, witness, call, say)
            if explained is None:
                skipped.append((source, "guard condition not reproduced from the episode's evidence"))
                continue
            owner = next((f for f in facts if f["kind"] in ("prohibited", "changed")
                          and any(c is call for c in f["calls"])
                          and any(normalize(str(a)) == normalize(str(b)) for a, _ in f["offending"]
                                  for b in explained["offending"])), None)
            if owner is not None:
                if source not in owner["sources"]:
                    owner["sources"].append(source)
                continue
            same = next((f for f in facts if f.get("guard_text") == explained["text"]), None)
            if same is not None:
                if call not in same["calls"]:
                    same["calls"].append(call)
                    same["turns"] = sorted({c["turn"] for c in same["calls"]})
                if source not in same["sources"]:
                    same["sources"].append(source)
                continue
            rule, support = required_note(explained, explained["text"]) if explained["text"].startswith(
                "remove ") else (explained["text"], [])
            facts.append({"kind": "prohibited", "rule": rule, "base": explained["text"],
                          "guard_text": explained["text"], "target": explained.get("target"), "sources": [source],
                          "support": support,
                          "display": [], "turns": [call["turn"]], "calls": [call], "family": explained["family"],
                          "offending": [(v, explained["offending_kind"]) for v in explained["offending"]]})
            continue
        same = next((f for f in facts if f.get("guard_text") == spec["text"]), None)
        if same is not None:
            if call not in same["calls"]:
                same["calls"].append(call)
                same["turns"] = sorted({c["turn"] for c in same["calls"]})
            if source not in same["sources"]:
                same["sources"].append(source)
            continue
        rule, support = required_note(spec, spec["text"])
        facts.append({"kind": "prohibited", "rule": rule, "base": spec["text"], "guard_text": spec["text"],
                      "target": spec.get("target"),
                      "sources": [source], "support": support, "display": [say(v) for v in spec["offending"]],
                      "turns": [call["turn"]], "calls": [call], "family": spec["family"],
                      "offending": [(v, spec["offending_kind"]) for v in spec["offending"]]})
    # failed manifest goals: rendered only when on the verified-goals list (and not excluded)
    verified = load_verified_goals() if goals is None else goals
    failed_goals = OrderedDict()
    for check_id, candidate, value in _goal_findings(record):
        if value == 0:
            failed_goals.setdefault(check_id, []).append(candidate)
    for check_id, candidates in failed_goals.items():
        source = manifest_check_id(check_id)
        if excluded_manifest(task, check_id, exclusions):
            excluded_hits.append(source)
            continue
        if goal_verdict(verified, task, check_id) != "verified":
            skipped.append((source, "goal_not_verified"))
            continue
        for candidate in candidates:
            found = diagnose_goal(record, check_id, candidate, say, invocations, by_site, calls, forbidden)
            if found[0] == "keep":
                _, text, call = found
                facts.append({"kind": "missing", "statement": text, "sources": [source], "family": "goal",
                              "turns": [call["turn"]], "calls": [call], "diagnosed": True})
                continue
            if found[0] == "closest":
                rules, call = found[1], found[2]
                raws = found[3] if len(found) > 3 else []
                subject = _subject_of(call, say)
                per_rule = [re.findall(r'"((?:[^"\\]|\\.)*)"', r.split(" (it is currently ")[0])[1:] for r in rules]
                values = [v for vs in per_rule for v in vs] if all(per_rule) else []
                if call is not None and subject.startswith("the email to ") and values and all(
                        r.startswith('"body" must contain ') for r in rules) and all(
                        _find_literal(_email_fields(call)[0], v) and not _find_literal(_email_fields(call)[1], v)
                        for v in values):
                    named = " and ".join(json.dumps(v) for v in values)
                    text = f"change the email about {named} so that its body, not only its subject, names them"
                elif subject.startswith("the email to ") and all(r.startswith('"body" ') for r in rules):
                    text = f"{subject} {'; '.join(r[len(chr(34) + 'body' + chr(34) + ' '):] for r in rules)}"
                else:
                    text = f"{subject}: {'; '.join(rules)}"
                facts.append({"kind": "missing", "statement": text, "sources": [source], "family": "goal",
                              "turns": [call["turn"]] if call else [], "calls": [call] if call else [],
                              "diagnosed": True, "values_raw": raws})
                continue
            if found[0] == "unrenderable":
                skipped.append((source, f"condition_unrenderable: {found[1]}"))
                continue
            spec = goal_spec(record, contract, check_id, candidate, say)
            if spec is None:
                skipped.append((source, "goal family states no plain requirement"))
                continue
            if spec.get("no_tool"):
                skipped.append((source, "goal_effect_has_no_tool"))
                continue
            if spec["group"] and not spec["group"][1] and spec["statement"] == spec["group"][0] and \
                    not spec["statement"].startswith("an email sent to"):
                skipped.append((source, f"condition_unrenderable: only the effect kind ({spec['statement']})"))
                continue
            facts.append({"kind": "missing", "statement": spec["statement"], "group": spec["group"],
                          "sources": [source], "turns": [], "calls": [], "family": spec["family"]})
    # 1: a prohibition on a recipient lists every call that sent to it, not only the first that broke the check
    for f in facts:
        if f["kind"] != "prohibited" or not (str(f.get("family", "")).startswith("gmail")
                                              or f.get("family") == "guard:send"):
            continue
        for value, kind in f.get("offending", []):
            if kind != "addressee":
                continue
            more = [c for c in calls if not c["error"] and c["name"].startswith("gmail_")
                    and _call_kind(c["name"]) == "send" and _find_literal(_email_fields(c)[2], value)
                    and c not in f["calls"]]
            if more:
                f["calls"] = sorted(f["calls"] + more, key=lambda c: (c["turn"], c["call_index"]))
                f["turns"] = sorted({c["turn"] for c in f["calls"]})
    # several requirements correcting one email the attempt sent share one "change ... so that" line
    merged_changes, kept = {}, []
    for f in facts:
        statement = f.get("statement") or ""
        m = re.match(r"^(change .+?) so that (.+)$", statement) if f["kind"] == "missing" and f.get("calls") else None
        if m is None:
            kept.append(f)
            continue
        key = (m.group(1), id(f["calls"][0]))
        owner = merged_changes.get(key)
        if owner is None:
            merged_changes[key] = f
            f["clauses_changed"] = [m.group(2)]
            kept.append(f)
            continue
        for clause in re.split(r" and (?=it |its |\")", m.group(2)):
            if not any(clause in c for c in owner["clauses_changed"]):
                owner["clauses_changed"].append(clause)
        owner["sources"] += [x for x in f["sources"] if x not in owner["sources"]]
    for (head, _), f in merged_changes.items():
        f["statement"] = f"{head} so that {' and '.join(f['clauses_changed'])}"
    facts = kept
    # 2: a missing-row clause whose value a per-row line already asks for (or a date when a per-row line is about
    # the date) is that line's requirement, not a missing row
    located = [f for f in facts if f.get("turns") and f["kind"] in ("missing", "prohibited", "changed")]
    world_strings = list(_strings_in(timeline.initial.model_dump(mode="json"))) if timeline.initial is not None else []

    def columns_of(value):
        """The field a value belongs to in the source data: a record key holding it, or 'key: value' in text."""
        out = {"date"} if re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", value) else set()
        for path, text in world_strings:
            text = str(text)
            if normalize(text) == normalize(value) and path and isinstance(path[-1], str):
                out.add(path[-1])
            for m in re.finditer(r"([A-Za-z_][\w ]{1,30}?)\s*[:=]\s*" + re.escape(value), text):
                out.add(m.group(1).strip())
        return {c for c in out if c and c.lower() not in ("text", "body", "body_plain", "content", "notes", "value")}

    kept = []
    for f in facts:
        source = f["sources"][0] if f.get("sources") else ""
        if f["kind"] == "missing" and not f.get("turns") and f.get("family") == "google_sheets_row_exists" \
                and source.startswith("native["):
            params = results[int(source[7:source.index("]")])].get("params") or {}
            values = [str(v) for v in _aslist(params.get("cell_contains")) if not isinstance(v, dict)]
            def restates(o):
                text = o.get("statement") or o.get("rule") or ""
                if any(normalize(v) == normalize(r) for v in values for r in o.get("values_raw") or []):
                    return True
                if " row" not in text.lower():
                    return False
                # an unread value whose source is the very source a per-row line asks its field from
                for v in values:
                    if not observable(v, say.corpus):
                        found = unobserved_source(v, say)
                        if found and found[0] == "source" and f"given in {found[1]}" in text:
                            return True
                cols = set().union(*(columns_of(v) for v in values)) if values else set()
                if cols and any(json.dumps(c) in text for c in cols):
                    return True
                # the field the line is about is named next to the value in the source text ("batch ref SCH-...",
                # "post it on Instagram ... platform")
                fields = [q.strip('"') for q in _QUOTED.findall(text) if len(q) <= 24 and re.fullmatch(
                    r'"[a-z][a-z_ ]*"', q)]
                for field in fields:
                    pattern = re.escape(field).replace("_", "[ _]?").replace("\\ ", "[ _]?")
                    for v in values:
                        near = rf"{pattern}.{{0,60}}{re.escape(v)}|{re.escape(v)}.{{0,60}}{pattern}"
                        if any(re.search(near, str(t), re.IGNORECASE | re.DOTALL) for _, t in world_strings):
                            return True
                return False
            owner = next((o for o in located if values and restates(o)), None)
            if owner is not None:
                owner["sources"] += [x for x in f["sources"] if x not in owner["sources"]]
                continue
        kept.append(f)
    facts = kept
    # 5: values one source record holds together are ONE missing row: "a row containing A and B"
    records = []
    if timeline.initial is not None:
        world = timeline.initial.model_dump(mode="json")
        for service in world.values():
            for collection in (service.values() if isinstance(service, dict) else ()):
                for item in (collection if isinstance(collection, list) else ()):
                    if isinstance(item, dict):
                        records.append(normalize(json.dumps(item)))
    clusters, kept = [], []
    for f in facts:
        source = f["sources"][0] if f.get("sources") else ""
        if not (f["kind"] == "missing" and not f.get("turns") and f.get("family") == "google_sheets_row_exists"
                and source.startswith("native[")):
            kept.append(f)
            continue
        params = results[int(source[7:source.index("]")])].get("params") or {}
        values = [str(v) for v in _aslist(params.get("cell_contains")) if not isinstance(v, dict)]
        sheet = params.get("spreadsheet_id")
        home = next((c for c in clusters if c["sheet"] == sheet and values and any(
            all(normalize(v) in r for v in c["values"] + values) for r in records)), None)
        if home is None:
            clusters.append({"sheet": sheet, "values": list(values), "fact": f})
            kept.append(f)
            continue
        home["values"] += [v for v in values if v not in home["values"]]
        home["fact"]["sources"] += [x for x in f["sources"] if x not in home["fact"]["sources"]]
    for c in clusters:
        if len(c["values"]) > 1:
            f = c["fact"]
            head = f["group"][0] if f.get("group") else None
            shown = say.all(c["values"])
            f["statement"] = f"a row in {say.where(c['sheet'])} containing {shown}"
            f["group"] = None  # one record, one line: "a row ... containing A and B"
    facts = kept
    # 9: requirements on the email to the right recipient join the redirect line (never a second line)
    redirects = {f["redirect"]: f for f in facts if f.get("redirect")}
    kept = []
    for f in facts:
        st = f.get("statement") or ""
        owner = next((o for r, o in redirects.items() if f["kind"] == "missing" and not f.get("turns")
                      and st.startswith((f"an email sent to {r}", f"an email to {r}"))), None)
        if owner is None:
            kept.append(f)
            continue
        rest = st.split(next(r for r, o in redirects.items() if o is owner), 1)[1]
        call = owner["calls"][0]
        subject, body, _ = _email_fields(call)
        for m in re.finditer(r"with (.+?) in its subject", rest):
            if not _find_literal(subject, m.group(1).strip('"')) and m.group(1) not in owner["redirect_subject"]:
                owner["redirect_subject"].append(m.group(1))
        m = re.search(r"whose body contains (.+)$", rest)
        if m:
            for v in re.split(_VALUE_SEP, m.group(1)):
                if not (v.startswith('"') and _find_literal(body, json.loads(v))) and v not in owner["redirect_body"]:
                    owner["redirect_body"].append(v)
        owner["sources"] += [x for x in f["sources"] if x not in owner["sources"]]
    facts = kept
    joined = lambda vs: vs[0] if len(vs) == 1 else ", ".join(vs[:-1]) + " and " + vs[-1]  # noqa: E731
    for f in facts:
        if f.get("redirect"):
            if f.get("redirect_subject"):
                f["statement"] += f"; its subject must contain {joined(f['redirect_subject'])}"
            if f.get("redirect_body"):
                f["statement"] += f"; its body must contain {joined(f['redirect_body'])}"
    # r5-1: prohibitions on the misdirected email join its single line
    kept = []
    for f in facts:
        owner = next((o for o in facts if o.get("redirect") and f["kind"] == "prohibited" and o is not f
                      and any(c is o["calls"][0] for c in f.get("calls") or [])
                      and all(k == "content" for _, k in f.get("offending") or [("", "x")])), None)
        if owner is None:
            kept.append(f)
            continue
        if owner["kind"] != "prohibited":
            owner.update({"kind": "prohibited", "offending": [], "display": [], "support": [], "target": None,
                          "not_mention": []})
        owner["not_mention"] += [say(v) for v, _ in f["offending"] if say(v) not in owner["not_mention"]]
        owner["offending"] += [o for o in f["offending"] if o not in owner["offending"]]
        owner["sources"] += [x for x in f["sources"] if x not in owner["sources"]]
    for f in kept:
        if f.get("redirect") and f["kind"] == "prohibited":
            head, _, rest = f["statement"].partition("; it must go to ")
            rest_parts = rest.split("; ", 1)
            nm = f["not_mention"]
            nm_text = nm[0] if len(nm) == 1 else ", ".join(nm[:-1]) + " or " + nm[-1]
            f["statement"] = (f"{head}; it must go to {rest_parts[0]}; it must not mention {nm_text}"
                              + (f"; {rest_parts[1]}" if len(rest_parts) > 1 else ""))
            f["rule"] = f["base"] = f["statement"]
    facts = kept
    # 6: a change line whose records another change line at the same turn already covers is folded into it
    kept = []
    for f in facts:
        st = f.get("statement") or ""
        if f["kind"] == "missing" and f.get("turns") and st.startswith("change the email"):
            mine = set(_QUOTED.findall(st))
            owner = next((o for o in facts if o is not f and o["kind"] == "missing" and o.get("turns") == f["turns"]
                          and (o.get("statement") or "").startswith("change the email") and mine
                          and mine < set(_QUOTED.findall(o["statement"]))), None)
            if owner is not None:
                owner["sources"] += [x for x in f["sources"] if x not in owner["sources"]]
                continue
        kept.append(f)
    facts = kept
    # a requirement restated by a prohibition at the same call (N), or a goal line restated by a native line (E)
    kept = []
    strip = lambda t: normalize(re.sub(r" \([^)]*\)$", "", t or ""))  # noqa: E731
    for f in facts:
        if f["kind"] == "missing" and f.get("keep"):
            owner = next((o for o in facts if o["kind"] == "prohibited" and strip(o.get("rule")) == strip(
                f["statement"]) and any(c is d for c in o["calls"] for d in f["calls"])), None)
            if owner is not None:
                owner["sources"] += [x for x in f["sources"] if x not in owner["sources"]]
                continue
        if f["kind"] == "missing" and not f.get("turns"):
            # 3: "an email sent to R ..." restates a line that already redirects the email about it to R
            owner = None
            for o in facts:
                m = re.search(r" so that it also goes to (.+)$", o.get("statement") or "") if o is not f else None
                if not m or o["kind"] != "missing":
                    continue
                recipient = re.sub(r" \(in the source data.*$", "", m.group(1))
                if (f["statement"].startswith((f"an email sent to {recipient}", f"an email to {recipient}"))
                        and set(_QUOTED.findall(f["statement"])) - {recipient} <= set(_QUOTED.findall(o["statement"]))):
                    owner = o
                    break
            if owner is not None:
                owner["sources"] += [x for x in f["sources"] if x not in owner["sources"]]
                continue
        if f["kind"] == "missing" and all(x.startswith("manifest:") for x in f["sources"]):
            # D34: a goal line restating a native line (diagnosed or not) about the same values and pointers; a
            # record's other names (a channel's id for its name) count as the same value
            def quoted_with_aliases(text):
                out = set()
                for q in _QUOTED.findall(text):
                    out.add(q)
                    try:
                        out |= {json.dumps(a) for a in identity_aliases(say, json.loads(q))}
                    except ValueError:
                        pass
                return out
            mine = set(_QUOTED.findall(re.sub(r'"[^"]+"(?= must\b)', "", f["statement"] or "")))
            tokens = set(_TOKEN.findall(f["statement"] or ""))
            owner = next((o for o in facts if o is not f and o["kind"] == "missing"
                          and any(x.startswith("native[") for x in o["sources"])
                          and mine and mine <= quoted_with_aliases(o.get("statement") or "")
                          and tokens <= set(_TOKEN.findall(o.get("statement") or ""))
                          and (o.get("diagnosed") or not f.get("diagnosed"))),
                         None)
            if owner is not None:
                owner["sources"] += [x for x in f["sources"] if x not in owner["sources"]]
                continue
        kept.append(f)
    facts = kept
    # 3: a prohibition that overrides what the request asks for names the unread rule that forbids it
    request = normalize(next((m.get("content") or "" for m in messages if m["role"] == "user"), ""))
    for f in facts:
        if f["kind"] != "prohibited" or "forbids it" in f.get("rule", "") or "the request names it" in f.get(
                "rule", ""):
            continue
        addressees = [v for v, k in f.get("offending", []) if k == "addressee" and isinstance(v, str)]
        if not addressees:
            continue
        value = addressees[0]
        domain = value.split("@", 1)[1].split(".")[0] if "@" in value else ""
        if normalize(value) not in request and not (domain and re.search(rf"\b{re.escape(domain)}\b", request)):
            continue
        pointer = _rule_email(say, value, domain)
        named = normalize(value) in request  # D32-3: the request itself names this target
        if pointer or named:
            reason = re.search(r" \(in the source data [^)]*\)$", f["rule"])
            core = f["rule"][:reason.start()] if reason else f["rule"]
            override = ", even though the request names it" if named else ""
            f["rule"] = (f"{core}{override}" + (f"; {pointer} forbids it" if pointer else "")
                         + (reason.group(0) if reason else ""))
    # D29-B: a line forbidding to touch a record points to the record's hold note in the task data
    for f in facts:
        if f["kind"] not in ("prohibited", "changed") or "hold note" in f.get("rule", "") or \
                "(in the source data" in f.get("rule", ""):
            continue  # a line that already gives the record's own reason needs no pointer
        if f["kind"] == "prohibited" and f.get("offending") and all(k == "content" for _, k in f["offending"]):
            continue  # removing content from an effect is not touching a held record
        ids = [str(v) for v, _ in f.get("offending") or [] if isinstance(v, (str, int)) and len(str(v)) >= 3]
        ids += [json.loads(q) for q in _QUOTED.findall(f.get("rule") or "") if len(q) >= 5]
        pointer = hold_note_pointer(say, list(dict.fromkeys(ids)))
        if pointer:
            reason = re.search(r" \(in the source data [^)]*\)$", f["rule"])
            core = f["rule"][:reason.start()] if reason else f["rule"]
            f["rule"] = f"{core}; {pointer}" + (reason.group(0) if reason else "")
            f["base"] = f["rule"]
    # C: a forbidden mention that is the email's core claim (in its subject or opening sentence) is not fixed by
    # deleting the word; the email itself must change, while a different email to that recipient stays required
    for f in facts:
        target = f.get("target")
        if f["kind"] != "prohibited" or not f.get("support") or not target or target[0] != "email" or \
                not all(k == "content" for _, k in f.get("offending", [])) or len(f.get("offending", [])) != 1:
            continue
        term = f["offending"][0][0]
        places = []
        for c in f["calls"]:
            subject, body, _ = _email_fields(c)
            opening = re.split(r"(?<=[.!?])\s|\n\s*\n", body.strip(), maxsplit=1)[0] if body.strip() else ""
            if re.match(r"^(hi|hello|dear|hey)\b[^\n.!?]*[,\n]", opening, re.IGNORECASE):
                rest = body.strip()[len(opening):].strip()
                opening = re.split(r"(?<=[.!?])\s|\n", rest, maxsplit=1)[0] if rest else opening
            if _find_literal(subject, term):
                places.append("subject")
            elif _find_literal(opening, term):
                places.append("opening sentence")
        if not places or len(places) != len(f["calls"]):
            continue
        claim = _claim_noun(term, f["calls"])
        article = "an" if claim[0].lower() in "aeiou" else "a"
        f["rule"] = (f"the email to {say(target[1])} confirms their {claim}, which this task forbids for them; "
                     f"they still need an email, but not {article} {claim} confirmation")
        f["base"] = f["rule"]
        f["display"] = []  # one line per recipient: the claim, not a removable word
    # reasons for native lines: the tested fields of the one record the check names (G4)
    for f in facts:
        source = f["sources"][0] if f["sources"] else ""
        if not source.startswith("native["):
            continue
        key = "rule" if "rule" in f else "statement"
        if "(in the source data" in f.get(key, ""):
            continue
        result = results[int(source[7:source.index("]")])]
        reason = native_reason(record, contract, result, say,
                               "require" if f["kind"] == "missing" else "forbid")
        if reason:
            f[key] = f[key] + reason
    # A recipient who must receive nothing subsumes content rules about emails to that recipient.
    barred = {normalize(v): f for f in facts if f["kind"] == "prohibited"
              for v, k in f.get("offending", []) if k == "addressee" and f["family"].startswith("gmail")}
    kept = []
    for f in facts:
        target = f.get("target")
        owner = (barred.get(normalize(target[1])) if target and target[0] == "email" and target[1]
                 and all(k == "content" for _, k in f.get("offending", [])) else None)
        if owner is not None and owner is not f:
            owner["sources"] += [s for s in f["sources"] if s not in owner["sources"]]
            reason = re.search(r" \(in the source data [^)]*\)", f["rule"])
            if reason and "(in the source data" not in owner["rule"]:
                owner["rule"] += reason.group(0)
            continue
        kept.append(f)
    facts = kept
    facts = _merge_same_effect(facts, say)
    # tool errors
    for (name, _), indices in _error_groups(calls).items():
        members = [calls[i] for i in indices]
        facts.append({"kind": "tool_error", "sources": [f"tool_error:turn{c['turn']}:call{c['call_index']}"
                                                        for c in members],
                      "turns": sorted({c["turn"] for c in members}), "calls": members, "first_index": indices[0],
                      "error": plain_error(members[0]["error"]), "family": "tool_error"})
    return {"facts": facts, "skipped": skipped, "excluded": excluded_hits, "mapped": mapped, "say": say,
            "corpus": corpus, "calls": calls, "truncated": trace.get("stop_condition") not in ("agent_completed", None)}


def _merge_same_effect(facts, say):
    """Rules that differ only in the offending values and point at the same calls share one line:
    'Remove "Warehouse B" and "Logistics Hub" from the Slack message ...'."""
    out, groups = [], {}
    for f in facts:
        display = f.get("display") or []
        shown = " and ".join(display)
        if f["kind"] not in ("prohibited", "changed") or len(display) != 1 or f["rule"].count(shown) != 1:
            out.append(f)
            continue
        key = (f["kind"], f["rule"].replace(shown, "\x00"), tuple(sorted(id(c) for c in f["calls"])))
        owner = groups.get(key)
        if owner is None:
            groups[key] = {**f, "template": key[1], "values": list(display)}
            out.append(groups[key])
            continue
        if display[0] not in owner["values"]:
            owner["values"].append(display[0])
        owner["sources"] += [s for s in f["sources"] if s not in owner["sources"]]
        owner["support"] += [s for s in f.get("support", []) if s not in owner["support"]]
        owner["offending"] += [o for o in f["offending"] if o not in owner["offending"]]
    for f in out:
        if "template" in f:
            negative = _NEGATIVE.match(f["template"]) and len(f["values"]) > 1
            template = f["template"].replace("do not put a row ", "do not put rows ").replace(
                "do not log a row for ", "do not log rows for ").replace(
                "do not add a row for ", "do not add rows for ") if negative else f["template"]
            values = f["values"]
            conj = " or " if negative else " and "
            f["rule"] = template.replace("\x00", values[0] if len(values) == 1 else
                                         ", ".join(values[:-1]) + conj + values[-1])
    return out


def _turn_label(turns):
    return ("Turn " if len(turns) == 1 else "Turns ") + ", ".join(map(str, turns))


_VALUE_SEP = re.compile('(?:, | and )(?=["\x00])')
_ONE_FIELD = re.compile(r'^((?:its|whose) (?:"[^"]+"|body|description) contains) (.+)$')


def _coalesce(clauses):
    """Requirements on one field of one record share a clause: 'its "description" contains "A" and "B"'."""
    out, by_prefix = [], {}
    for clause in clauses:
        parts = [part.strip() for part in clause.split("; ")]
        for part in parts:
            m = _ONE_FIELD.match(part)
            if m and m.group(1) in by_prefix:
                values = by_prefix[m.group(1)]
                values += [v for v in re.split(_VALUE_SEP, m.group(2)) if v not in values]
                continue
            if m:
                by_prefix[m.group(1)] = list(dict.fromkeys(re.split(_VALUE_SEP, m.group(2))))
                out.append(m.group(1))
            else:
                out.append(part)
    def joined(values):
        return values[0] if len(values) == 1 else ", ".join(values[:-1]) + " and " + values[-1]
    return [f"{c} {joined(by_prefix[c])}" if c in by_prefix else c for c in out]


_COMPUTED_PAIR = re.compile(r'containing ("[^"]+"); containing a value computed from the source data')


def _collapse_computed(statement):
    """D34: 'containing "A"; containing a value computed ...; containing "B"; containing a value computed ...' is
    one clause: 'containing "A" and "B", each with its value computed from the source data'."""
    names = _COMPUTED_PAIR.findall(statement)
    if len(names) < 2:
        return statement
    first = _COMPUTED_PAIR.search(statement)
    rest = _COMPUTED_PAIR.sub("", statement[first.end():]).replace("; ;", ";")
    joined = ", ".join(names[:-1]) + " and " + names[-1]
    tail = re.sub(r"^(;\s*)+", "", rest).strip()
    return (statement[:first.start()] + f"containing {joined}, each with its value computed from the source data"
            + (f"; {tail}" if tail else ""))


def _merge_missing(facts):
    """Identical requirements merge; requirements on one target (one email address, channel, SMS number or
    sheet) share a line: "an email sent to X whose body contains A; whose body contains B"."""
    merged, by_text, by_head = [], {}, {}
    for f in facts:
        if f["kind"] != "missing":
            merged.append(f)
            continue
        key = normalize(f["statement"])
        if key in by_text:
            by_text[key]["sources"] += [s for s in f["sources"] if s not in by_text[key]["sources"]]
            continue
        group = f.get("group")
        if group and group[0] in by_head:
            owner = by_head[group[0]]
            if group[1] and group[1] not in owner["clauses"]:
                owner["clauses"].append(group[1])
            owner["sources"] += [s for s in f["sources"] if s not in owner["sources"]]
            by_text[key] = owner
            continue
        f = {**f, "clauses": [group[1]] if group and group[1] else []}
        if group:
            by_head[group[0]] = f
        by_text[key] = f
        merged.append(f)
    for f in merged:
        if f["kind"] == "missing" and f.get("group"):
            head = f["group"][0]
            # a clause that another clause extends ("one whose X is a" vs "one whose X is a, Y is b") is dropped
            clauses = [c for c in f["clauses"] if not any(o != c and o.startswith(c + ",") for o in f["clauses"])]
            f["statement"] = _collapse_computed(head + (" " + "; ".join(_coalesce(clauses)) if clauses else ""))
            if f.get("other_note"):
                f["statement"] += f" ({f['other_note']})"
    return merged


_TOKEN = re.compile("\x00(\\d+)\x00")
_COMPUTABLE = re.compile(r"^[\s$€£]*-?[\d,.]+\s*(%|days?|kwh|usd)?\s*$|^\d{4}-\d{2}(-\d{2})?", re.IGNORECASE)


def _strings_in(value, path=()):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _strings_in(v, (*path, k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _strings_in(v, (*path, i))
    elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
        yield path, value


def unobserved_source(value, say):
    """Where an unobserved required value comes from, named only by what the attempt could see (R5: the value
    itself is never shown): ("source", where) for a value found in the initial world, ("computed", phrase) for a
    figure or count the request asks the attempt to work out, or None when neither applies."""
    world = say.initial.model_dump(mode="json") if say.initial is not None else {}
    needle = normalize(value)
    seen = lambda v: isinstance(v, str) and v and observable(v, say.corpus)  # noqa: E731
    if needle:
        numeric = _is_number(value)
        hits = [(path, text) for path, text in _strings_in(world) if len(path) >= 3 and (
            (normalize(str(text)) == needle or standalone_number(value, text)) if numeric
            else needle in normalize(text))]
        if numeric:
            # never point to a source the data marks as superseded or outdated (the attempt cannot tell which applies)
            def stale(path):
                record = world[path[0]][path[1]][path[2]] if isinstance(path[2], int) else {}
                return bool(_SUPERSEDED.search(json.dumps(record)))
            hits = [h for h in hits if not stale(h[0])]
            # when the data also holds a revision or correction, the attempt cannot tell which applies: a figure
            # is then the computed one, never a pointer to one of the competing sources (A)
            revisions = re.compile(r"supersed|revised|revision|correction|corrected|replaces|effective immediately|"
                                   r"narrowed|no longer|instead of|updated [a-z ]*(range|target|policy|rate)",
                                   re.IGNORECASE)
            def contested(path):
                service = world.get(path[0]) or {}
                record = world[path[0]][path[1]][path[2]] if isinstance(path[2], int) else {}
                title = " ".join(str(record.get(k) or "") for k in ("subject", "title", "name"))
                stems = {w.lower()[:5] for w in re.findall(r"[A-Za-z]{3,}", title)} - {
                    "the", "and", "for", "from", "with", "updat", "new", "re", "fwd", "contr", "reque"}
                if not stems:
                    return False
                when = lambda r: str(r.get("date") or r.get("sent_at") or r.get("received_at") or r.get(  # noqa: E731
                    "timestamp") or r.get("created_at") or "")
                for other in service.get(path[1]) or []:
                    if not isinstance(other, dict) or other is record:
                        continue
                    text = json.dumps(other)
                    other_title = " ".join(str(other.get(k) or "") for k in ("subject", "title", "name"))
                    shared = {w.lower()[:5] for w in re.findall(r"[A-Za-z]{3,}", other_title)} & stems
                    # a NEWER record on the same subject that revises it (two shared title stems)
                    newer = not when(record) or not when(other) or when(other) > when(record)
                    if revisions.search(text) and len(shared) >= 2 and newer:
                        return True
                return revisions.search(json.dumps(record)) is not None and False
            for h in hits:
                if contested(h[0]):
                    say.contested.add(h[0][0])  # D: the unread app holds competing versions of this figure
            hits = [h for h in hits if not contested(h[0])]
        # the record whose own id or name IS the value first, then any record mentioning it
        # the record whose own id or name IS the value first; for a figure, a message stating it (the request
        # email) before a record that merely mentions it
        hits.sort(key=lambda h: (not (normalize(h[1]) == needle and str(h[0][-1]) in (
            "id", "template_id", "name", "envelope_id", "email", "title")), numeric and h[0][0] not in (
            "gmail", "slack")))
        hits = _choose_source_hits(hits, world, say, needle)
        for path, text in hits:
            service, collection, index = path[0], path[1], path[2]
            if collection == "_generic":
                # several disagreeing records hold the value and nothing binds one: name none of them
                generic = {"gmail": "an email in Gmail that this attempt did not read",
                           "slack": "a Slack message that this attempt did not read",
                           "google_sheets": "a Google Sheets spreadsheet, which this attempt never read"}
                label = service.replace("_", " ").replace("docusign", "DocuSign").replace("salesforce", "Salesforce")
                return ("source", generic.get(service, f"the {label} data, which this attempt never read"))
            record = world[service][collection][index] if isinstance(index, int) else {}
            # did the attempt retrieve this very record? (its id, or a distinctive field, is in a tool result)
            marks = [v for k, v in record.items() if isinstance(v, str) and len(v) >= 4
                     and (k in ("id", "message_id", "subject", "title", "name", "ts") or k.endswith("_id"))]
            retrieved = any(observable(v, say.corpus) for v in marks)
            if service == "gmail":
                tail = "" if retrieved else ", which this attempt never opened"
                field = path[3] if len(path) > 3 else None
                if str(field) in ("from_", "from", "sender"):
                    # D29b: the value IS the email's sender: name the field, by the subject only
                    subject = record.get("subject")
                    if _locator_ok(say, subject, value):
                        return ("whole", f"the sender of the email {json.dumps(subject)}{tail}")
                    return ("source", "an email in Gmail" + ("" if retrieved else " that this attempt did not read"))
                where = gmail_locator(record, say, value)
                if where:
                    body = str(field) in ("body", "body_plain", "body_html", "snippet")
                    full = (f"the body of {where}" if body else where) + tail
                    subject_only = gmail_locator({"subject": record.get("subject")}, say, value)
                    if subject_only:
                        # a recipient line never shows the sender (it could be read as the answer)
                        say.recipient_variant[full] = (f"the body of {subject_only}" if body else subject_only) + tail
                    return ("source", full)
                return ("source", "an email in Gmail" + ("" if retrieved else " that this attempt did not read"))
            if service == "google_sheets":
                sheet = record.get("spreadsheet_id")
                column = path[4] if len(path) > 4 and path[3] == "cells" else None
                cells = record.get("cells") or {}
                read = any(isinstance(v, str) and len(v) >= 4 and observable(v, say.corpus) for v in cells.values())
                worksheet = record.get("worksheet_id")
                title = next((w.get("title") for w in (world.get("google_sheets") or {}).get("worksheets") or []
                              if isinstance(w, dict) and w.get("id") == worksheet), None) or worksheet
                if _locator_ok(say, sheet, value):
                    where = (f"worksheet {json.dumps(title)} of spreadsheet {json.dumps(sheet)}"
                             if _locator_ok(say, title, value) else f"spreadsheet {json.dumps(sheet)}")
                else:
                    where = "a Google Sheets spreadsheet"
                col = column if column and _locator_ok(say, column, value) else None
                return ("source", f"{'the ' + json.dumps(col) + ' column of ' if col else ''}{where}"
                                  + ("" if read else ", which this attempt never read"))
            if service == "slack":
                channel = record.get("channel_id")
                name = next((c.get("name") for c in (world.get("slack") or {}).get("channels") or []
                             if isinstance(c, dict) and c.get("id") == channel), None)
                label = f"#{name}" if name and _locator_ok(say, name, value) else channel
                where = f" in channel {json.dumps(label)}" if _locator_ok(say, label, value) else ""
                return ("source", f"a Slack message{where}" + ("" if retrieved else " that this attempt did not read"))
            label = service.replace("_", " ").replace("docusign", "DocuSign").replace("salesforce", "Salesforce")
            # 10: name the item kind ("a Salesforce note", "a DocuSign template"), not "the data"
            collection = str(path[1]).replace("_", " ")
            noun = (collection[:-3] + "y" if collection.endswith("ies") else
                    collection[:-1] if collection.endswith("s") else collection)
            # D29-A: the record by its name (or id) when that does not carry the hidden value
            named = next((record.get(k) for k in ("name", "title", "subject", "id") if _locator_ok(
                say, record.get(k), value)), None)
            if named:
                return ("source", f"the {label} {noun} {json.dumps(named)}"
                                  + ("" if retrieved else ", which this attempt never read"))
            rule = rule_message(say, noun, value)
            if rule:
                return ("rule", rule)  # D29b: the choice cannot be named, its rule can
            return ("source", f"a {label} {noun}" + ("" if retrieved else ", which this attempt never read"))
    text = str(value).strip()
    counted = re.fullmatch(r"(\d[\d,.]*)\s+([A-Za-z][A-Za-z -]*)", text)
    if counted:
        word = counted.group(2)
        # P: the unit is printed only as a whole word the attempt observed, never as a grader stem; a stem whose
        # plural was observed ("registrant" -> "registrants") is stated in that form (it still contains the stem)
        plural = word + "s" if not word.endswith("s") else None
        if plural and seen(plural):
            word = plural
        elif not seen(word):
            return None  # the unit word is neither observed nor locatable: stating it vaguely is not faithful (G2)
        return ("computed", f"the value the request asks for, written as a number followed by the word {word}")
    if re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", text):
        return ("computed", "a date within the period the request asks for")  # 2c: a date is never a "figure"
    if isinstance(value, (int, float)) or _COMPUTABLE.match(text):
        # 4: a native literal is not "computed" (only a manifest expression is); a bare number that a source states
        # as the leading part of a thousands-formatted amount ("175" of "$175,000") is that amount
        if re.fullmatch(r"\d{3}", text):  # 1-2 digits are too short to identify an amount ("15" of "$15,000")
            pattern = re.compile(rf"(?<![\d.,]){text},\d{{3}}(?:\.\d+)?(?![\d,])")
            for _, candidate in _strings_in(world):
                m = pattern.search(str(candidate))
                if m:
                    inner = unobserved_source(m.group(0), say)
                    if inner and inner[0] == "source":
                        return ("amount", inner[1])
                    break
        return ("computed", "the figure the request calls for")
    return None


_ROLE_BEFORE = (
    (re.compile(r"(sent to|email to|SMS to|send nothing to|invitation to|invitation sent to|for invitee|go to|"
                r"goes to) $"),
     "the recipient named in "),
    (re.compile(r"(the row for|record|ticket|conversation|organization|contact|lead|envelope|channel) $"),
     "the one named in "),
)


_SLOT = re.compile(r"(?:\b(?!Update\b)[A-Z][\w:]*(?: [A-Z][\w:]*)? )?\b(template|opportunity|record|envelope|contact|lead|"
                   r"ticket|conversation|organization|note|channel|list|page|task|page) $")


def _describe(text, start, found, variants=None):
    """Phrase for an unobserved value at `start` of `text`, fitted to its role (addressee, record or value)."""
    kind, phrase = found
    if kind in ("computed", "quote", "whole"):
        return phrase
    if kind == "rule":
        comma = "," if re.search(r"(never opened|never read|did not read)$", phrase) else ""
        return f"the one {phrase}{comma} calls for"
    if kind == "amount":
        return "the amount given in " + phrase
    before = text[max(0, start - 40):start]
    for pattern, lead in _ROLE_BEFORE:
        if pattern.search(before):
            if lead.startswith("the recipient") and variants and phrase in variants:
                return lead + variants[phrase]  # the sender of the source is never shown next to a recipient
            return lead + phrase
    return "the value given in " + phrase


def _resolve_unobserved(facts, built):
    """Replace unobserved-value tokens by where the value comes from; drop a line whose value is nowhere."""
    say = built["say"]
    values = list(say.tokens.values())
    out = []
    for f in facts:
        texts = [f["text"], *(f.get("notes") or {}).values()]
        found = {int(m) for t in texts for m in _TOKEN.findall(t)}
        described = {}
        for i in found:
            hit = unobserved_source(values[i], say)
            own = next((_find_literal(c["arguments"], values[i]) for c in f.get("calls") or []
                        if _find_literal(c["arguments"], values[i])), None)
            if f["kind"] in ("prohibited", "changed") and own:
                hit = ("quote", json.dumps(own))  # the attempt wrote this text itself, so it observed it
            described[i] = hit
        if any(d is None for d in described.values()):
            for source in f["sources"]:
                built["skipped"].append((source, "condition_unrenderable: value neither observed nor locatable "
                                                 "(unobserved_value_unlocatable)"))
            continue

        def per_source(t):
            """M: several values from one source in one list name that source once."""
            def collapse(m):
                items = re.split('(?:, | and )(?=["\x00])', m.group(0))
                groups = OrderedDict()
                for item in items:
                    token = _TOKEN.fullmatch(item)
                    found = described[int(token.group(1))] if token else None
                    if found and (found[0] in ("computed", "amount") or (found[0] == "source" and _describe(
                            t, m.start(), found).startswith("the value given in "))):
                        groups.setdefault(found, []).append(item)
                plural = [found for found, members in groups.items() if len(members) > 1]
                if not plural:
                    return m.group(0)
                kept = [i for i in items if not (_TOKEN.fullmatch(i) and described[int(_TOKEN.fullmatch(i).group(1))]
                                                 in plural)]
                kept += [f"the values given in {where}" if kind == "source" else
                         f"the amounts given in {where}" if kind == "amount" else
                         where.replace("the figure the request", "the figures the request").replace(
                             "the value the request asks for, written", "the values the request asks for, written")
                         for kind, where in plural]
                return kept[0] if len(kept) == 1 else ", ".join(kept[:-1]) + " and " + kept[-1]
            return re.sub(r'(?:"[^"]*"|\x00\d+\x00)(?:(?:, | and )(?:"[^"]*"|\x00\d+\x00))+', collapse, t)

        def fix(t):
            t = per_source(t)
            parts, last = [], 0
            for m in _TOKEN.finditer(t):
                kind, where = described[int(m.group(1))]
                slot = _SLOT.search(t[last:m.start()]) if kind in ("source", "rule") else None
                if slot and kind == "rule":
                    # D29b: "made from the template that the email "Updated template selection rules" calls for"
                    comma = "," if re.search(r"(never opened|never read|did not read)$", where) else ""
                    parts.append(t[last:last + slot.start()] + f"the {slot.group(1)} that {where}{comma} calls for")
                    last = m.end()
                    continue
                if slot and re.match(rf"(?:a|the) [\w:]+(?: [\w:]+)? {re.escape(slot.group(1))}\b", where):
                    # the source IS such a record: "created from the DocuSign template "Premium Services", which this
                    # attempt never read" (D29: named by its own name or id)
                    parts.append(t[last:last + slot.start()] + where.replace(", which this attempt never read",
                                                                             " that this attempt never read")
                                 if where.startswith("a ") else t[last:last + slot.start()] + where)
                    last = m.end()
                    if where.startswith("the ") and "never" in where[-14:] and t[m.end():m.end() + 6].startswith(
                            (" whose", " with ", " conta", " so th")):
                        parts.append(",")
                    continue
                if slot:
                    # "created from template <value>" -> "created from the template named in <source>"
                    parts.append(t[last:last + slot.start()] + f"the {slot.group(0).strip()} named in {where}")
                    last = m.end()
                    follow = t[m.end():m.end() + 6]
                    if follow.startswith((" whose", " with ", " conta", " so th")):
                        parts.append(",")
                    continue
                phrase = _describe(t, m.start(), described[int(m.group(1))], say.recipient_variant)
                follow = t[m.end():m.end() + 6]
                # a pointer phrase ending in a relative clause is closed off before the next clause
                if follow.startswith((" whose", " with ", " conta", " lasti", " and ", " so th")) and (
                        "read" in phrase[-12:] or "opened" in phrase[-12:]):
                    phrase += ","
                parts.append(t[last:m.start()] + phrase)
                last = m.end()
            return "".join(parts) + t[last:]

        out.append({**f, "text": fix(f["text"]), "notes": {k: fix(v) for k, v in (f.get("notes") or {}).items()},
                    "value_tokens": found})
    return out


_QUOTED = re.compile(r'"(?:[^"\\]|\\.)*"')
_TURN_PREFIX = re.compile(r"^Turns? [\d, ]+: ")


_NEGATIVE = re.compile(r"^(Do not |Post no |Create no |Add no |Book no |Send no |Subscribe no |Make no )", re.IGNORECASE)


def _merge_lines(facts):
    """One line per fault (R6): lines that differ only in the record they name merge into one line naming every
    record and every turn; an end-state requirement that a turn line already states ("must be") merges into it."""
    reason_re = re.compile(r" \(in the source data [^)]*\)")

    def parts(f):
        body = reason_re.sub("", _TURN_PREFIX.sub("", f["text"]))
        literals = _QUOTED.findall(body)
        return _QUOTED.sub("\x01", body), literals

    own_row = re.compile(r'^(Turns? [\d, ]+): (Each ".*?" value needs its own row in ".*?"): (.*)\.$')
    rows, kept = OrderedDict(), []
    for f in facts:
        m = own_row.match(f["text"]) if f["kind"] == "missing" else None
        if m is None:
            kept.append(f)
            continue
        owner = rows.get(m.group(2))
        if owner is None:
            rows[m.group(2)] = f
            f["items"] = [m.group(3)]
            kept.append(f)
            continue
        owner["items"] += [i for i in [m.group(3)] if i not in owner["items"]]
        owner["turns"] = sorted(set(owner["turns"]) | set(f["turns"]))
        owner["sources"] += [x for x in f["sources"] if x not in owner["sources"]]
    # values a line forbids to log, and records a line keeps in their state (logging them as closed contradicts it)
    forbidden = {normalize(str(v)) for f in facts if f["kind"] == "prohibited"
                 and re.search(r"(?i)\bdo not (log|add|put) (a )?rows?\b", f.get("text") or "")
                 for v, _ in f.get("offending") or []}
    kept_open = {normalize(json.loads(q)) for f in facts if f["kind"] == "prohibited"
                 and re.search(r"(?i)\b(it|they) must stay\b", f.get("text") or "")
                 for q in _QUOTED.findall((f.get("text") or "").split(";")[0])}
    for key, f in rows.items():
        together = OrderedDict()
        for item in f["items"]:
            value, _, cell = item.partition(" (logged together as ")
            together.setdefault(cell, []).append(value)

        def segment(cell, values):
            if not cell:
                return " and ".join(values)
            # D32-4: a value a prohibition forbids to log is not part of the required row
            inner = cell.rstrip(")")
            try:
                tokens = [t.strip() for t in re.split(r"[,;/]", json.loads(inner))]
            except ValueError:
                tokens = []
            bad = [t for t in tokens if t and normalize(t) in forbidden | kept_open]
            if bad:
                shown = " and ".join(json.dumps(b) for b in bad)
                why = " (it stays open)" if all(normalize(b) in kept_open for b in bad) else ""
                return " and ".join(values) + f" in its own row, without {shown}, which must not be logged{why}"
            return " and ".join(values) + f" (logged together as {cell}"
        segments = [segment(cell, values) for cell, values in together.items()]
        joined = ("; " if any(" in its own row" in x for x in segments) else ", ").join(segments)
        f["text"] = f"{_turn_label(f['turns'])}: {key}: {joined}."
        f["notes"] = {t: f"{key}: {joined}." for t in f["turns"]}
    facts = kept
    located = [f for f in facts if f["kind"] in ("prohibited", "changed") or (f["kind"] == "missing" and f.get("turns"))]
    for f in located:
        f.setdefault("original_notes", dict(f.get("notes") or {}))
    for position in range(4):
        buckets = OrderedDict()
        for f in located:
            skeleton, literals = parts(f)
            if position < len(literals):
                key = (f["kind"], skeleton, tuple(v for i, v in enumerate(literals) if i != position))
                buckets.setdefault(key, []).append(f)
        for members in buckets.values():
            if len(members) < 2:
                continue
            owner = members[0]
            skeleton, literals = parts(owner)
            values = list(dict.fromkeys(parts(m)[1][position] for m in members))
            # a prohibition over several values forbids each: "Do not put rows containing A or B"
            negative = bool(_NEGATIVE.match(skeleton))
            conj = " or " if negative else " and "
            joined = ", ".join(values[:-1]) + conj + values[-1] if len(values) > 1 else values[0]
            literals[position] = joined
            if negative and len(values) > 1:
                skeleton = skeleton.replace("Do not put a row ", "Do not put rows ", 1).replace(
                    "Do not log a row for ", "Do not log rows for ", 1).replace(
                    "Do not add a row for ", "Do not add rows for ", 1)
            text = skeleton
            for literal in literals:
                text = text.replace("\x01", literal, 1)
            # a reason stays only when every merged value carries the same one; otherwise it is ambiguous.
            # Repeats of one value (several turns) pool their reasons: the reason belongs to the value.
            per_value = {}
            for m in members:
                per_value.setdefault(tuple(parts(m)[1]), set()).update(reason_re.findall(m["text"]))
            reasons = {frozenset(v) for v in per_value.values()}
            reasons = sorted(reasons.pop()) if len(reasons) == 1 else []
            if len(reasons) == 1:
                text = text.rstrip(".") + reasons[0] + "."
            text = text.replace("this recipient must not receive any email", "these recipients must not receive any email")
            if len(values) > 1:
                # several records on one line: plural nouns and pronouns
                text = re.sub(r"^(The|Change the) row for ", r"\1 rows for ", text)
                text = re.sub(r"\(in the source data its ", "(in the source data their ", text)
                if negative:
                    text = re.sub(r"^(Do not \w+ (?:[A-Z][\w:]* )+)(ticket|conversation|record|lead|envelope|"
                                  r"opportunity|contact)\b", r"\1\2s", text)
                    text = text.replace("; it must stay ", "; they must stay ")
                text = text.replace("; its hold note is in ", "; their hold notes are in ")
                text = re.sub(r"^Change the email about ", "Change the emails about ", text)
                text = text.replace(" so that it also goes to ", " so that they also go to ")
                text = text.replace("; the emails to it at these turns", "; the emails to them at these turns")
                text = text.replace("so that its body, not only its subject, names them",
                                    "so that their bodies, not only their subjects, name them")
            if len(values) > 1 and " confirms their " in text:
                text = text.replace("The email to ", "The emails to ", 1).replace(" confirms their ", " confirm their ", 1)
                text = text.replace("they still need an email", "each of them still needs an email")
            turns = sorted({t for m in members for t in m["turns"]})
            owner["turns"], owner["text"] = turns, f"{_turn_label(turns)}: {text}"
            for m in members[1:]:
                owner["sources"] += [x for x in m["sources"] if x not in owner["sources"]]
                owner["calls"] = owner["calls"] + [c for c in m["calls"] if c not in owner["calls"]]
                owner["spans"] = owner.get("spans", []) + m.get("spans", [])
                m["merged_into"] = owner
            # one note per turn: the call's own fault, or the merged line when several records share that turn
            per_turn = defaultdict(list)
            for m in members:
                for turn, note in (m.get("original_notes") or m.get("notes") or {}).items():
                    if note not in per_turn[turn]:
                        per_turn[turn].append(note)
            owner["notes"] = {turn: notes[0] if len(notes) == 1 else text[0].upper() + text[1:]
                              for turn, notes in per_turn.items()}
        located = [f for f in located if "merged_into" not in f]
    facts = [f for f in facts if "merged_into" not in f]
    combined, keyed = [], {}
    for f in facts:
        if f["kind"] == "missing" and f.get("diagnosed") and f.get("turns") and ": " in f["text"]:
            head, _, rest = f["text"].partition(": ")
            subject, colon, rule = rest.partition(": ") if rest.startswith("The ") and ": " in rest else ("", "", rest)
            key = (head, subject)
            if subject and key in keyed:
                owner = keyed[key]
                for part in rule.rstrip(".").split("; "):
                    if part not in owner["text"]:
                        owner["text"] = owner["text"].rstrip(".") + "; " + part + "."
                owner["sources"] += [x for x in f["sources"] if x not in owner["sources"]]
                owner["notes"] = {t: owner["text"].partition(": ")[2] for t in owner["turns"]}
                continue
            keyed[key] = f
        combined.append(f)
    facts = combined
    out = []
    pointer = re.compile(r'value given in [^;.,]+')
    for f in facts:
        if f["kind"] == "missing" and not f.get("turns"):
            phrases = set(pointer.findall(f["text"]))
            owner = next((o for o in facts if o is not f and o.get("turns") and " row" in o["text"].lower()
                          and "row" in f["text"] and phrases and all(ph in o["text"] for ph in phrases)
                          and len(_QUOTED.findall(f["text"])) <= 2), None)
            if owner is not None:
                owner["sources"] += [x for x in f["sources"] if x not in owner["sources"]]
                continue
        if f["kind"] == "missing" and not f.get("turns") and f.get("value_tokens") and \
                f.get("family") == "google_sheets_row_exists":
            # "a row containing <value>" restates a turn line that already asks the rows for that value
            owner = next((o for o in facts if o is not f and o.get("turns") and " row" in o["text"].lower()
                          and f["value_tokens"] <= set(o.get("value_tokens") or ())), None)
            if owner is not None:
                owner["sources"] += [x for x in f["sources"] if x not in owner["sources"]]
                continue
        if f["kind"] == "missing":
            mine = set(_QUOTED.findall(f["text"]))
            owner = next((o for o in facts if o["kind"] in ("prohibited", "changed") and " must be " in o["text"]
                          and len(mine) >= 2 and mine <= set(_QUOTED.findall(o["text"]))), None)
            if owner is not None:
                owner["sources"] += [x for x in f["sources"] if x not in owner["sources"]]
                continue
        out.append(f)
    return out


_ROW_POINTER = re.compile(r'^Still missing at the end of the attempt: the row for ("[^"]+") in spreadsheet ("[^"]+") '
                          r'updated so that ("[^"]+") contains (the value given in .+|"[^"]+")\.$')
_QUALIFIED = re.compile(r'(?<= in )(the (?:[\w:]+ ){1,2}data|a (?:Salesforce|DocuSign|hubspot|[a-z]+) [a-z]+|'
                        r'the email "[^"]+"(?: from "[^"]+")?|the (?:Salesforce|DocuSign|hubspot) [a-z]+ "[^"]+"|'
                        r'an email in Gmail|a Slack message(?: in channel "[^"]+")?|'
                        r'spreadsheet "[^"]+"|the email "[^"]+"|an email from "[^"]+")'
                        r'(, which this attempt never read| that this attempt did not read|, which this attempt never '
                        r'opened)')


def _qualify_once(text):
    """7: a source's "which this attempt never read" is said once per line."""
    seen = set()

    def once(m):
        if m.group(1) in seen:
            return m.group(1)
        seen.add(m.group(1))
        return m.group(0)
    return re.sub(r",,", ",", _QUALIFIED.sub(once, text))


def _merge_row_pointers(out):
    """4: rows of one spreadsheet missing one field whose values all come from one source share one line."""
    groups, kept = OrderedDict(), []
    for f in out:
        m = _ROW_POINTER.match(f.get("text") or "") if f["kind"] == "missing" and not f.get("turns") else None
        if m is None:
            kept.append(f)
            continue
        key = (m.group(2), m.group(3), m.group(4))
        if key not in groups:
            groups[key] = {**f, "rows": []}
            kept.append(groups[key])
        owner = groups[key]
        owner["rows"].append(m.group(1))
        if owner is not f:
            owner["sources"] = owner["sources"] + [x for x in f["sources"] if x not in owner["sources"]]
    for (sheet, field, where), f in groups.items():
        rows = f["rows"]
        if len(rows) == 1:
            continue
        names = ", ".join(rows[:-1]) + " and " + rows[-1]
        value = (where.replace("the value given in ", "the values given in ", 1)
                 if where.startswith("the value given in ") else where)
        f["text"] = f"The rows for {names} in spreadsheet {sheet} must have their {field} set to {value}."
    return kept


def _call_target(name, arguments):
    """The record a call acts on: (tool family, sheet, row) or (tool family, addressee/record id)."""
    family = name.split("_")[0]
    sheet = arguments.get("spreadsheet_id") or arguments.get("spreadsheet")
    row = arguments.get("row_id") or arguments.get("row")
    if sheet is not None and row is not None:
        return (family, normalize(str(sheet)), normalize(str(row)))
    for key in ("to", "ticket_id", "record_id", "conversation_id", "envelope_id", "email"):
        if arguments.get(key):
            return (family, normalize(str(arguments[key])))
    return None


_TARGET_KEYS = ("spreadsheet", "spreadsheet_id", "channel", "channel_id", "ticket_id", "conversation_id", "record_id",
                "envelope_id", "to", "object", "object_type", "table", "tableName", "base_id")


def recovered_error(fact, calls):
    """Did the attempt recover from this tool error by itself? A later successful call of the same tool on the
    same target (the same spreadsheet, channel, ticket ...), or a later successful call that uses one of the
    names the error message listed as valid."""
    if not fact.get("calls"):
        return False
    last = max(calls.index(c) for c in fact["calls"])
    later = [c for c in calls[last + 1:] if not c["error"]]
    name = fact["calls"][0]["name"]

    def target(c):
        return {k: normalize(str(c["arguments"].get(k))) for k in _TARGET_KEYS if c["arguments"].get(k) not in (None, "")}

    targets = [target(c) for c in fact["calls"]]
    for c in later:
        if c["name"] == name and any(t and all(target(c).get(k) == v for k, v in t.items()) for t in targets):
            return True
    listed = set()
    for c in fact["calls"]:
        for m in re.finditer(r"\[([^\]]*)\]", str(c["error"] or "")):
            for item in re.findall(r"'([^']+)'", m.group(1)):
                listed.add(item)
                listed |= set(re.findall(r"\(([^()]+)\)", item))
                listed.add(re.sub(r"\s*\([^()]*\)\s*$", "", item))
    listed = {x for x in listed if len(x) >= 3}
    return bool(listed) and any(_find_literal(c["arguments"], x) for c in later for x in listed)


def finish_facts_d(built, record, verification="run"):
    """Texts, notes and bad spans; corrections only when verified."""
    calls = built["calls"]
    forbidden_targets = {_call_target(c["name"], c["arguments"]) for f in built["facts"]
                         if f["kind"] == "changed" or (f["kind"] == "prohibited" and any(
                             k in ("addressee", "record") for _, k in f.get("offending") or []))
                         for c in f.get("calls") or []} - {None}
    out = []
    for f in _merge_missing(built["facts"]):
        if f["kind"] == "missing":
            if f.get("turns"):
                body = f["statement"][0].upper() + f["statement"][1:]
                out.append({**f, "text": f"{_turn_label(f['turns'])}: {body}.", "notes": {
                    t: body + "." for t in f["turns"]}, "spans": []})
                continue
            if f.get("plain"):
                out.append({**f, "text": f["statement"][0].upper() + f["statement"][1:] + ".", "notes": {},
                            "spans": []})
                continue
            prefix = (D_HEADING_LIMIT + " Still missing at the end: " if built["truncated"]
                      else "Still missing at the end of the attempt: ")
            out.append({**f, "text": prefix + f["statement"] + ".", "notes": {}, "spans": []})
            continue
        if f["kind"] == "tool_error":
            first = calls[f["first_index"]]
            fix, fixed = None, None
            for c in f["calls"]:
                index = next(i for i, d in enumerate(calls) if d is c)
                for change in correction_candidates(c):
                    corrected = {**c["arguments"], **change}
                    if _call_target(c["name"], corrected) in forbidden_targets:
                        continue  # 2: never suggest re-running a call another line forbids
                    key = _correction_key(record, index, corrected)
                    if key not in _VERIFIED and verification == "run":
                        verify_corrections([(record, index, corrected)])
                    verdict = _VERIFIED.get(key)
                    if verdict and verdict["ok"] and all(observable(v, built["corpus"]) for v in change.values()):
                        fix, fixed = change, c
                        break
                if fix:
                    break
            body = f"{first['name']} returned an error: {f['error']}."
            if len(f["calls"]) > len(f["turns"]):
                body += f" ({len(f['calls'])} calls.)"  # turns are listed once; the count is of calls, exact
            runs = ("The same call runs with " + ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}"
                                                            for k, v in fix.items()) + ".") if fix else ""
            fix_text = (f" {runs}" if fixed is first or len(f["turns"]) == 1 else f" At turn {fixed['turn']}, {runs[0].lower()}{runs[1:]}") if fix else ""
            if len({c["error"] for c in f["calls"]}) > 1:
                # 2: the calls passed different values: one generic line, no per-call correction
                generic = re.sub(r"\s*\((?:it was given|got|received)[^)]*\)", "", plain_error(first["error"]))
                body = f"{first['name']} returned an error in {len(f['calls'])} calls: {generic}."
                fix_text = (f" For example, the call at turn {fixed['turn']} runs with " + ", ".join(
                    f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in fix.items()) + ".") if fix else ""
                runs, fixed, fix = "", None, None
            notes = {}
            for c in f["calls"]:
                own = f" {runs}" if c is fixed else ""
                if c["turn"] in notes and own:
                    notes[c["turn"]] += own
                notes.setdefault(c["turn"], f"{c['name']} returns an error here: {plain_error(c['error'])}.{own}")
            spans = []
            for c in f["calls"]:
                span = {"kind": "tool_error", "turn": c["turn"], "call_index": c["call_index"], "name": c["name"]}
                if c is fixed:
                    key = next(iter(fix))
                    value = c["arguments"].get(key)
                    text = _find_literal({key: value}, value if isinstance(value, str) else json.dumps(value))
                    if text:
                        span["offending_text"], span["offending_kind"] = text, "argument"
                spans.append(span)
            out.append({**f, "text": f"{_turn_label(f['turns'])}: {body}{fix_text}", "notes": notes, "spans": spans,
                        "correction": {"turn": fixed["turn"], "call_index": fixed["call_index"], "change": fix}
                        if fix else None})
            continue
        spans = []
        for c in f["calls"]:
            span = {"kind": f["kind"], "turn": c["turn"], "call_index": c["call_index"], "name": c["name"]}
            for value, kind in f["offending"]:
                found = _find_literal(c["arguments"], value)
                if found:
                    span["offending_text"], span["offending_kind"] = found, kind
                    break
            spans.append(span)
        rule = f["rule"][0].upper() + f["rule"][1:]
        notes = {t: rule + "." for t in f["turns"]}
        out.append({**f, "text": f"{_turn_label(f['turns'])}: {rule}.", "notes": notes, "spans": spans})
    out = _resolve_unobserved(out, built)
    out = _merge_row_pointers(out)
    for f in out:
        f["text"] = _qualify_once(f["text"])
        f["notes"] = {k: _qualify_once(v) for k, v in (f.get("notes") or {}).items()}
    out = _merge_lines(out)
    for f in out:
        if f["kind"] == "tool_error":
            last = max(calls.index(c) for c in f["calls"]) if f["calls"] else -1
            f["recovered"] = any(not c["error"] and c["name"] == f["calls"][0]["name"] for c in calls[last + 1:])
    # D32-2: a tool error the attempt itself recovered from is not a lesson (it reads as "go find X")
    out = [f for f in out if not (f["kind"] == "tool_error" and recovered_error(f, calls))]
    # a tool error the attempt recovered from ranks below every other kind (R6 cap)
    source_line = unread_source_line(out, built, record)
    if source_line is not None:
        out.append(source_line)
    out.sort(key=lambda f: (len(D_ORDER) if f.get("recovered") else D_ORDER.index(f["kind"]),
                            f.get("source_note", False), min(f["turns"]) if f["turns"] else 10 ** 6))
    return out


_APP_NAMES = {"gmail": "Gmail", "slack": "Slack", "google_sheets": "Google Sheets", "salesforce": "Salesforce",
              "docusign": "DocuSign", "hubspot": "HubSpot", "google_drive": "Google Drive"}
_POINTER_APP = (("an email in Gmail", "gmail"), ("the email ", "gmail"), ("an email from ", "gmail"),
                ("a Slack message", "slack"), ("Google Sheets spreadsheet", "google_sheets"),
                ("the Salesforce data", "salesforce"), ("the DocuSign data", "docusign"),
                ("the hubspot data", "hubspot"), ("a Salesforce ", "salesforce"), ("a DocuSign ", "docusign"),
                ("a hubspot ", "hubspot"), ("the Salesforce ", "salesforce"), ("the DocuSign ", "docusign"),
                ("the hubspot ", "hubspot"), ("worksheet ", "google_sheets"), ("spreadsheet ", "google_sheets"))


_NAMED_SOURCE = re.compile(r'(the email "[^"]+"(?: from "[^"]+")?|an email from "[^"]+"|'
                           r'the (?:Salesforce|DocuSign|hubspot) [a-z]+ "[^"]+"|worksheet "[^"]+" of spreadsheet "[^"]+"|'
                           r'a Slack message in channel "[^"]+")(?:,? which this attempt never (?:opened|read)| that this '
                           r'attempt (?:did not read|never read))')


def unread_source_line(out, built, record):
    """D: ONE line naming the sources this attempt never retrieved that the failures depend on, by what the attempt
    could see (the app it never searched, or never opened an item in). Triggers: pointer phrases in the lines,
    and values of skipped checks that were neither observed nor located but sit in an app the attempt never used."""
    calls = built["calls"]
    # searched = read something there (sending email is not searching Gmail)
    used = {app for app in (*_APP_NAMES, "google_drive") if any(
        c["name"].startswith(app + "_") and _call_kind(c["name"]) is None for c in calls)}
    apps, sources = OrderedDict(), []
    # 6: the task's contract reads records of an app (a read obligation) the attempt never searched, and
    # manifest checks of the task failed in this episode: the lines depend on records there
    contract = episode_contract(record) or {}
    failed_manifest = [x for f in out for x in f.get("sources") or [] if x.startswith("manifest:")] or [
        f"manifest:{cid}" for cid, _, value in _goal_findings(record) if value == 0]
    for source in (contract.get("sources") or {}).values():
        adapter = str(source.get("adapter") or "")
        app = adapter.split(".", 1)[0]
        if adapter.endswith(("reads@1", "message_reads@1")) and app in _APP_NAMES and app not in used \
                and failed_manifest:
            apps.setdefault(app, "records")
            sources += [x for x in dict.fromkeys(failed_manifest) if x not in sources]
    for f in out:
        text = f.get("text") or ""
        if not re.search(r"(did not read|never read|never opened)", text):
            continue
        for phrase, app in _POINTER_APP:
            # "given in the email ...", "named in the body of the email ...", "the sender of the email ..."
            if re.search(r"((given|named) in (the body of )?|the sender of )" + re.escape(phrase), text):
                apps.setdefault(app, "unopened")
                sources += [x for x in f.get("sources") or [] if x not in sources]
    say = built["say"]
    if say.initial is not None:
        world = say.initial.model_dump(mode="json")
        for skipped_source, why in built.get("skipped") or []:
            m = re.search(r"value neither observed nor locatable: (.+)$", why)
            if not m:
                continue
            needle = normalize(m.group(1))
            for app in _APP_NAMES:
                if app in used or app in apps:
                    continue
                if any(needle in normalize(str(v)) for _, v in _strings_in(world.get(app) or {})):
                    apps.setdefault(app, "unsearched")
                    if skipped_source not in sources:
                        sources.append(skipped_source)
    for app in sorted(getattr(say, "contested", ())):
        if app in _APP_NAMES and app not in used and app not in apps:
            apps[app] = "unsearched"
            sources += [x for f in out if "the request calls for" in (f.get("text") or "")
                        for x in f.get("sources") or [] if x not in sources]
    if not apps:
        return None
    never = [_APP_NAMES[a] for a in apps if a not in used]
    unopened = [_APP_NAMES[a] for a in apps if a in used]
    parts = []
    if never:
        what = "records" if all(apps[a] == "records" for a in apps if a not in used) else "values and rules"
        parts.append(f"This attempt never searched {' or '.join(never)}; {what} these lines depend on are there")
    # D29-A: the sources the lines name, to open before acting; the generic sentence only for unnamed ones
    named = list(dict.fromkeys(m.group(1) for f in out for m in _NAMED_SOURCE.finditer(f.get("text") or "")))
    # one source once: 'the email "X"' is the same email as 'the email "X" from "Y"'
    named = [n for n in named if not any(o != n and o.startswith(n + " from ") for o in named)]
    generic = any(re.search(r"(an email in Gmail that this attempt did not read|a Slack message that this attempt "
                            r"did not read|a (?:Salesforce|DocuSign|hubspot) [a-z]+, which this attempt never read|"
                            r"a Google Sheets spreadsheet)", f.get("text") or "") for f in out)
    if named:
        joined = named[0] if len(named) == 1 else ", ".join(named[:-1]) + " and " + named[-1]
        parts.append(f"{'before' if never else 'Before'} acting, open {joined}; this attempt never opened "
                     f"{'it' if len(named) == 1 else 'them'}")
    if unopened and (generic or not named):
        kinds = {"gmail": "emails", "slack": "messages", "google_sheets": "rows", "google_drive": "files"}
        items = " and ".join(dict.fromkeys(kinds.get(a, "records") for a in apps if a in used))
        parts.append(f"{'it' if never or named else 'This attempt'} searched {' and '.join(unopened)} but never "
                     f"opened the {items} that hold these values")
    text = "; ".join(parts) + "."
    return {"kind": "missing", "text": text, "turns": [], "calls": [], "notes": {}, "spans": [],
            "sources": sources, "family": "unread_source", "source_note": True}


def render_facts_d(facts):
    lines = [f"- {f['text']}" for f in facts]
    if len(lines) > D_LINES:
        omitted = defaultdict(int)
        for f in facts[D_LINES - 1:]:
            omitted[f["kind"]] += 1
        lines = lines[:D_LINES - 1] + ["- Not listed: " + ", ".join(
            f"{n} more {KIND_LABEL[k]}" for k, n in omitted.items()) + "."]
    return lines


def _assert_no_grader_internals(text, record, contract):
    """R7: no check-type names, check ids or signal ids in the teacher text."""
    names = {r["type"] for r in trace_of(record)["info"]["automationbench"]["assertions"]}
    for check in (contract or {}).get("checks", []):
        names |= {check.get("check_id"), check.get("signal_id")}
    for name in filter(None, names):
        if re.search(rf"(?<![\w.-]){re.escape(name)}(?![\w.-])", text):
            raise ValueError(f"grader internal {name!r} in teacher text")


def build_context_d(record, siblings, arm, placement, contract=None, exclusions=None, verification="run",
                    goals=None):
    """Variant D context; always returns a dict. `skip_reason` is set when the episode contributes no loss."""
    trace = trace_of(record)
    task = trace["task"]["data"]["name"]
    contract = episode_contract(record, contract)
    if verification == "run":
        group = {r.get("id"): r for r in [record, *siblings]}
        prepare_verifications(list(group.values()))  # one batched replay per task group; cached afterwards
    built = build_facts_d(record, contract, exclusions, goals)
    facts = finish_facts_d(built, record, verification)
    success = trace.get("metrics", {}).get("task_completed_correctly") == 1.0
    solution = choose_solution_d(record, siblings, verification) if arm == "B" else None
    solution_dropped = None
    if solution is not None:
        # S3: every native check behind a fact passes in the solution's replay (true by S2; checked anyway)
        verdicts = {(a["type"], _cjson(a.get("params") or {})): a for a in solution["assertions"]}
        for f in facts:
            for source in f["sources"]:
                if source.startswith("native["):
                    result = trace["info"]["automationbench"]["assertions"][int(source[7:source.index("]")])]
                    verdict = verdicts.get((result["type"], _cjson(result.get("params") or {})))
                    if verdict is None or (not verdict["excluded"] and not verdict["passed"]):
                        solution_dropped = f"S3: solution {solution['episode_id']} does not satisfy {source}"
        if solution_dropped:
            solution = None
    base = {"episode_id": record.get("id"), "task": task, "arm": arm, "placement": placement, "variant": "D",
            "success": success, "untagged": not built["mapped"], "excluded_checks": built["excluded"],
            "skipped_checks": built["skipped"], "solution_dropped": solution_dropped}
    if not facts and solution is None:
        if success:
            reason = "success_without_feedback_or_solution"
        elif built["excluded"]:
            reason = "only_excluded_checks_failed"
        else:
            reason = "failed_without_locatable_fact"
        return {**base, "first_message": None, "notes": {}, "solution_episode": None, "facts": [], "bad_spans": [],
                "skip_reason": reason}
    parts = [first_user(record)]
    if solution is not None:
        parts.append(f"{SOLUTION_HEADING}\n\n{render_solution_d(solution)}")
    if facts and placement in ("first", "both"):
        heading = SUCCESS_HEADING if success else FAILED_HEADING
        parts.append(heading + "\n\n" + "\n".join(render_facts_d(facts)))
    parts.append(CLOSING)
    notes = defaultdict(list)
    if placement in ("notes", "both") and built["mapped"]:
        for f in facts:
            for turn, text in (f.get("notes") or {}).items():
                if NOTE_PREFIX + text not in notes[turn]:
                    notes[turn].append(NOTE_PREFIX + text)
    first_message = "\n\n".join(parts)
    _assert_no_grader_internals(first_message + "\n".join("\n".join(v) for v in notes.values()), record, contract)
    return {**base, "first_message": first_message, "notes": {int(k): "\n".join(v) for k, v in notes.items()},
            "solution_episode": solution["episode_id"] if solution else None,
            "solution_variant": solution["variant"] if solution else None,
            "facts": [{"kind": f["kind"], "text": f["text"], "turn": min(f["turns"]) if f["turns"] else None,
                       "turns": f["turns"], "sources": f["sources"], "support": f.get("support", []),
                       "family": f["family"], "correction": f.get("correction")} for f in facts],
            "bad_spans": [s for f in facts for s in f["spans"]], "skip_reason": None}
