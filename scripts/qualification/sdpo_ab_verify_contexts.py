"""Independent checker for SDPO teacher contexts (plan D24, requirements R1-R7 and S1-S3).

Builds the teacher context of recorded episodes through the builder's public entry point
(`sdpo_ab_context.build_context`, treated as a black box) and checks every context against
the recorded episodes and the frozen environment:

- C1 contradiction: a prohibition line forbids a call that a successful attempt of the same task
  makes (literal match on the arguments the line shows; then tool + addressee unless the line names
  content the successful call lacks), or a call of the arm-B solution shown in the same context.
- C2 correction executability: each suggested call is executed in the frozen environment at the
  point of the attempt (the attempt's earlier calls replayed first) and must not return an error.
- C3 solution validity (arm B): the shown solution's calls, replayed from the task's initial world,
  pass every native check (in-process) with no manifest harm (real harness with a scripted model).
- C4 leak: grader literals in the feedback that the attempt never observed (builder function and
  an independent implementation).
- C5 coverage: the build raises no error on a failed episode and gives it at least one fact.
- C6 grader internals: assertion type names (raw and humanised), check ids, 'row N' without a name.
- C7 length: at most 8 fact lines.
- C8 traceability: each fact references a failed native check or manifest finding of the episode.

Replay exactness: tool calls run in-process through the environment's own MCP tool server
(`AutomationBenchLimitedToolset` registered on an in-memory `MCPServer`, strict-argument middleware
and argument validation included), with the task loaded and set up by the environment's taskset and
the world JSON round-tripped through the state model between calls as the remote state channel does.
Every replayed prefix call is compared with the recorded tool result, so each C2 verdict carries
whether its prefix reproduced the recording exactly. Manifest harm needs the harness's execution
receipts, so C3 harm runs the real harness with a scripted model (as `sdpo_ab_replay.py` does).

Usage (CPU only):
    PYTHONPATH=scripts/qualification python scripts/qualification/sdpo_ab_verify_contexts.py \
        --variant C --arm A --placement both --tasks fit20 --out report.json
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import re
import sys
import time
import traceback
from collections import Counter, OrderedDict, defaultdict
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EVIDENCE = REPO / "docs/research/evidence"
DEFAULT_EPISODES = "sdpo-ab-2026102*/**/episodes.jsonl.gz"
TASK_SETS = {"fit20": EVIDENCE / "sdpo-ab-20261027-selection/fit20.json",
             "transfer20": EVIDENCE / "sdpo-ab-20261027-selection/transfer20.json"}
MAX_EXAMPLES = 10
MAX_FACT_LINES = 8
UNSEEN_TEXT = "a value this attempt never observed"

# Arguments that address or identify a target (who/where), as opposed to content.
ADDRESS_KEYS = {"to", "cc", "bcc", "recipient", "recipients", "email", "emails", "to_email", "channel",
                "channel_id", "channel_name", "user", "user_id", "users", "recipient_id", "to_number", "phone",
                "spreadsheet", "spreadsheet_id", "worksheet", "worksheet_id", "row", "row_id", "ticket_id",
                "conversation_id", "contact_id", "record_id", "id", "issue_key", "project", "project_id",
                "envelope_id", "profile_id", "board_id", "list_id", "card_id", "task_id", "deal_id",
                "opportunity_id", "account_id", "lead_id", "case_id", "event_id", "calendar_id", "file_id",
                "folder_id", "message_id", "thread_id", "employee_id", "candidate_id", "customer_id",
                "base_id", "table", "table_id", "table_name", "workspace", "workspace_id", "assignee",
                "assignee_email", "owner", "owner_id", "signer_email", "customer_email", "requester_email"}
# Arguments that name the item without being the offending content.
DESCRIPTOR_KEYS = {"subject", "title", "name", "summary", "topic", "label", "worksheet_title"}
PROHIBITION = re.compile(r"\b(do not|don't|must not|mustn't|never|should not|avoid)\b", re.IGNORECASE)
SUGGESTION = re.compile(r"(correct call|correct calls|instead,? call|instead use|use the call|should call|"
                        r"call instead|the call to make is|retry with|try)\s*:?\s*", re.IGNORECASE)
TOOL_CALL_START = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\(")
MINTED = re.compile(r"\b(?=[0-9a-f-]*\d)[0-9a-f]{8,}(?:-[0-9a-f]{4,})*\b|\b\d{8,}\b", re.IGNORECASE)


class Unseen:
    """The builder's placeholder for a value the attempt never observed (matches any value)."""

    def __repr__(self):
        return "UNSEEN"

    def __eq__(self, other):
        return isinstance(other, Unseen)

    def __hash__(self):
        return 0


UNSEEN = Unseen()


# ---------------------------------------------------------------- text helpers (pure)

def normalize(text):
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def value_text(value):
    """A comparable normal form of an argument value (numbers and numeric strings agree)."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in "[{":
            try:
                return value_text(json.loads(stripped))
            except ValueError:
                pass
        if re.fullmatch(r"-?\d+(\.0+)?", stripped):
            return str(int(float(stripped)))
        return normalize(value)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        return normalize(json.dumps({k: value_text(v) for k, v in sorted(value.items())}, ensure_ascii=False))
    if isinstance(value, (list, tuple)):
        return normalize(json.dumps([value_text(v) for v in value], ensure_ascii=False))
    return normalize(value)


def atoms(value):
    if isinstance(value, dict):
        return [x for v in value.values() for x in atoms(v)]
    if isinstance(value, (list, tuple)):
        return [x for v in value for x in atoms(v)]
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, float) and value.is_integer():
        return [str(int(value))]
    return [str(value)]


def args_text(arguments):
    """Everything inside a call's arguments as one normalised string (JSON strings decoded)."""
    def expand(value):
        if isinstance(value, str) and value.strip()[:1] in "[{":
            try:
                return expand(json.loads(value))
            except ValueError:
                return value
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        return value
    return normalize(" ".join(atoms(expand(arguments))))


def _skip_balanced(text, start):
    """Index after the bracketed group starting at text[start] (quotes respected)."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack, i, quote = [], start, None
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch == '"':
            quote = ch
        elif ch in pairs:
            stack.append(pairs[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
            if not stack:
                return i + 1
        i += 1
    return len(text)


def parse_kwargs(text):
    """`k="v", n=2, x=(a value this attempt never observed)` -> ordered dict of values.

    Values are JSON where possible; a parenthesised note after a value is ignored; the builder's
    unseen placeholder becomes UNSEEN; anything else is kept as its bare text."""
    out = OrderedDict()
    decoder = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        match = re.compile(r"\s*,?\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*").match(text, i)
        if not match:
            nxt = re.compile(r",\s*[A-Za-z_][A-Za-z0-9_]*\s*=").search(text, i + 1)
            if not nxt:
                break
            i = nxt.start()
            continue
        key, i = match.group(1), match.end()
        if i < n and text[i] == "(":
            end = _skip_balanced(text, i)
            inner = text[i + 1:end - 1].strip()
            out[key] = UNSEEN if inner == UNSEEN_TEXT else inner
            i = end
        else:
            try:
                value, i = decoder.raw_decode(text, i)
            except ValueError:
                bare = re.compile(r"[^,)]*").match(text, i)
                value, i = bare.group(0).strip(), bare.end()
            out[key] = value
        # skip a trailing note such as ` (a string, not a number)` up to the next `, key=`
        nxt = re.compile(r",\s*[A-Za-z_][A-Za-z0-9_]*\s*=").search(text, i)
        i = nxt.start() if nxt else n
    return out


def parse_calls(text, tool_names=None):
    """Tool calls written as `name(args)` in a line: list of dicts (name, args, start, end)."""
    calls = []
    for match in TOOL_CALL_START.finditer(text):
        name = match.group(1)
        if tool_names is not None and name not in tool_names:
            continue
        if any(c["start"] <= match.start() < c["end"] for c in calls):
            continue
        end = _skip_balanced(text, match.end() - 1)
        inner = text[match.end():end - 1]
        calls.append({"name": name, "args": parse_kwargs(inner), "start": match.start(), "end": end,
                      "text": text[match.start():end]})
    return calls


def masked(text, spans):
    chars = list(text)
    for start, end in spans:
        for k in range(start, min(end, len(chars))):
            chars[k] = "x" if not chars[k].isspace() else chars[k]
    return "".join(chars)


def sentence_end(text, start, spans=()):
    """End of the sentence that contains `start`, ignoring periods inside calls and quotes."""
    plain = masked(text, spans)
    plain = re.sub(r'"[^"]*"', lambda m: "x" * len(m.group(0)), plain)
    match = re.compile(r"\.(\s+[A-Z]|\s*$)").search(plain, start)
    return match.start() + 1 if match else len(text)


def prohibition_clauses(line, tool_names=None):
    """Each prohibition in a line: the clause text from the marker to the end of its sentence and the
    tool calls written inside it."""
    calls = parse_calls(line, tool_names)
    spans = [(c["start"], c["end"]) for c in calls]
    plain = masked(line, spans)
    plain = re.sub(r'"[^"]*"', lambda m: "x" * len(m.group(0)), plain)
    clauses = []
    for marker in PROHIBITION.finditer(plain):
        end = sentence_end(line, marker.start(), spans)
        inside = [c for c in calls if marker.start() <= c["start"] < end]
        clauses.append({"marker": marker.group(0), "start": marker.start(), "end": end,
                        "text": line[marker.start():end], "calls": inside})
    # merge clauses that share a sentence (e.g. "do not ... ; never ...")
    merged = []
    for clause in clauses:
        if merged and clause["start"] < merged[-1]["end"]:
            continue
        merged.append(clause)
    return merged


def quoted_literals(text):
    """Quoted strings of 3+ characters; quotes pair up in order, so an empty "" never shifts them."""
    out = []
    rest = text
    for pattern in (r'"([^"]*)"', "\u201c([^\u201d]*)\u201d", r"(?<![\w])'([^']*)'(?![\w])"):
        out += re.findall(pattern, rest)
        rest = re.sub(pattern, " ", rest)
    return [q for q in out if len(q) >= 3]


def suggestion_calls(line, tool_names=None):
    """Calls the line suggests making (after 'Correct call:' and similar), not inside a prohibition."""
    calls = parse_calls(line, tool_names)
    prohibited = [(c["start"], c["end"]) for c in prohibition_clauses(line, tool_names)]
    out = []
    for marker in SUGGESTION.finditer(line):
        if marker.group(1).lower() == "try" and not re.match(r"\s*\w+_\w+\(", line[marker.end():]):
            continue
        following = [c for c in calls if c["start"] >= marker.end() and not
                     any(s <= c["start"] < e for s, e in prohibited)]
        if following and following[0]["start"] - marker.end() <= 3:
            out.append(following[0])
            # consecutive calls joined by "and"/";"/","
            for prev, nxt in zip(following, following[1:]):
                gap = line[prev["end"]:nxt["start"]]
                if re.fullmatch(r"\s*(,|;|and|then|, then)\s*", gap):
                    out.append(nxt)
                else:
                    break
    seen, unique = set(), []
    for c in out:
        if c["start"] not in seen:
            seen.add(c["start"])
            unique.append(c)
    return unique


def line_turns(line):
    match = re.search(r"\bTurns? ((?:\d+(?:, | and |,)?)+)", line)
    return [int(x) for x in re.findall(r"\d+", match.group(1))] if match else []


def ids_only_difference(a, b):
    """Two tool results that differ only in minted identifiers (hex/digit runs, uuids)."""
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    return MINTED.sub("#", a) == MINTED.sub("#", b)


def is_error_result(content):
    """Whether a tool result is an error, by the forms the harness and the tools return."""
    if content is None:
        return True
    text = content if isinstance(content, str) else json.dumps(content)
    stripped = text.strip()
    lowered = stripped[:120].lower()
    if lowered.startswith(("error", "tool error", "invalid", "unknown tool", "unsupported arguments")):
        return True
    if stripped.startswith("{"):
        try:
            value = json.loads(stripped)
        except ValueError:
            return False
        if isinstance(value, dict):
            if value.get("success") is False:
                return True
            if "error" in value and value.get("error") and value.get("success") is not True:
                return True
    return False


# ---------------------------------------------------------------- episode helpers

def trace_of(record):
    traces = record.get("traces")
    return traces[0] if traces else record


def task_of(record):
    return trace_of(record)["task"]["data"]["name"]


def recorded_calls(record):
    """Ordered tool calls of an episode, read directly from the trace nodes."""
    nodes = trace_of(record)["nodes"]
    results = {}
    for node in nodes:
        message = node.get("message") or {}
        if message.get("role") == "tool":
            results[message.get("tool_call_id")] = message.get("content")
    calls, turn = [], 0
    for node in nodes:
        message = node.get("message") or {}
        if message.get("role") != "assistant":
            continue
        turn += 1
        for index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            name = call.get("name") or function.get("name")
            raw = call.get("arguments", function.get("arguments"))
            if isinstance(raw, dict):
                arguments, valid = raw, True
            else:
                try:
                    arguments = json.loads(raw or "{}")
                    valid = isinstance(arguments, dict)
                except ValueError:
                    arguments, valid = raw, False
            result = results.get(call.get("id"))
            calls.append({"turn": turn, "call_index": index, "name": name, "arguments": arguments,
                          "valid_json": valid, "raw": raw, "result": result,
                          "error": is_error_result(result)})
    return calls


def own_text(record):
    """What the attempt itself wrote in its tool calls (it knows these values)."""
    return normalize("\n".join(json.dumps(c["arguments"], ensure_ascii=False) if not isinstance(c["arguments"], str)
                               else c["arguments"] for c in recorded_calls(record)) + "\n" + args_text_all(record))


def args_text_all(record):
    return " ".join(args_text(c["arguments"]) for c in recorded_calls(record) if isinstance(c["arguments"], dict))


def observed_text(record):
    """What the attempt observed: its user messages, tool results and tool definitions."""
    trace = trace_of(record)
    parts = []
    for node in trace["nodes"]:
        message = node.get("message") or {}
        if message.get("role") in ("user", "tool", "system"):
            content = message.get("content")
            parts.append(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False))
    for tool in trace.get("tools") or []:
        parts.append(json.dumps(tool, ensure_ascii=False))
    return normalize("\n".join(parts))


def first_user_text(record):
    for node in trace_of(record)["nodes"]:
        message = node.get("message") or {}
        if message.get("role") == "user":
            return message.get("content") or ""
    return ""


def failed_assertions(record):
    assertions = (trace_of(record).get("info") or {}).get("automationbench", {}).get("assertions") or []
    return [(i, a) for i, a in enumerate(assertions) if not a.get("excluded") and not a.get("passed")]


def episode_day(record):
    timing = trace_of(record).get("timing") or {}
    start = (timing.get("setup") or {}).get("start") or timing.get("start")
    return datetime.fromtimestamp(start, UTC) if start else None


# ---------------------------------------------------------------- context anatomy

def context_parts(context, record, headings):
    """Split a built context into prompt, solution block, feedback lines and note lines."""
    first = context.get("first_message") or ""
    prompt = first_user_text(record)
    rest = first[len(prompt):] if prompt and first.startswith(prompt) else first
    solution_heading, feedback_headings, closing = headings
    solution, feedback = "", []
    blocks = rest.split("\n\n")
    mode = None
    for block in blocks:
        stripped = block.strip()
        if not stripped:
            continue
        if solution_heading and stripped.startswith(solution_heading):
            mode = "solution"
            solution += stripped[len(solution_heading):].strip() + "\n"
            continue
        if any(stripped.startswith(h) for h in feedback_headings if h):
            mode = "feedback"
            body = stripped
            for h in feedback_headings:
                if h and body.startswith(h):
                    body = body[len(h):]
            feedback += [line for line in body.strip().splitlines() if line.strip()]
            continue
        if closing and stripped == closing:
            mode = None
            continue
        if mode == "solution":
            solution += stripped + "\n"
        elif mode == "feedback":
            feedback += [line for line in stripped.splitlines() if line.strip()]
    notes = []
    for turn, text in sorted((context.get("notes") or {}).items(), key=lambda kv: int(kv[0])):
        for line in str(text).splitlines():
            if line.strip():
                notes.append((int(turn), line))
    return {"prompt": prompt, "solution": solution, "feedback": feedback, "notes": notes}


def solution_calls(solution_text, tool_names=None):
    """Calls of a rendered solution: numbered lines `n. tool(args)`."""
    calls = []
    for line in solution_text.splitlines():
        match = re.match(r"\s*\d+\.\s+(.*)$", line)
        if not match:
            continue
        parsed = parse_calls(match.group(1), tool_names)
        if parsed and parsed[0]["start"] == 0:
            calls.append({"name": parsed[0]["name"], "arguments": dict(parsed[0]["args"])})
    return calls


def context_lines(parts):
    """(where, turn, text) for every feedback line the teacher reads."""
    out = [("first_message", None, line) for line in parts["feedback"]]
    out += [("note", turn, line) for turn, line in parts["notes"]]
    return out


# ---------------------------------------------------------------- C1 contradiction

def literal_match(shown_args, call_args):
    """Every argument the line shows has the same value in the call (unseen values match anything)."""
    for key, value in shown_args.items():
        if value is UNSEEN or isinstance(value, Unseen):
            continue
        if key not in call_args:
            return False
        if value_text(value) != value_text(call_args[key]):
            return False
    return True


def addressees(arguments):
    out = []
    for key, value in (arguments or {}).items():
        if key.lower() in ADDRESS_KEYS or key.lower().endswith("_id") or key.lower().endswith("_email"):
            out += [normalize(v) for v in atoms(value) if len(str(v).strip()) >= 3]
    return out


def content_literals(clause, tool_names):
    """Literals a prohibition clause names beyond tool, addressee and item descriptors."""
    literals = []
    covered = []
    for call in clause["calls"]:
        covered.append((call["start"] - clause["start"], call["end"] - clause["start"]))
        for key, value in call["args"].items():
            if value is UNSEEN or key.lower() in ADDRESS_KEYS or key.lower() in DESCRIPTOR_KEYS \
                    or key.lower().endswith("_id") or key.lower().endswith("_email"):
                continue
            literals += [v for v in atoms(value) if len(str(v).strip()) >= 3]
    outside = masked(clause["text"], covered)
    literals += [q for q in quoted_literals(outside) if q.strip() != UNSEEN_TEXT]
    names = {normalize(t) for t in tool_names or ()}
    return [normalize(x) for x in literals if normalize(x) not in names]


def contradiction(line, reference_calls, tool_names):
    """Prohibition clauses of a line that forbid a reference (successful or solution) call.

    Returns a list of findings: kind 'literal' (same tool and same shown argument values) or
    'broad' (same tool and addressee, and the clause names no content that the call lacks)."""
    findings = []
    for clause in prohibition_clauses(line, tool_names):
        named_tools = {c["name"] for c in clause["calls"]}
        named_tools |= {t for t in tool_names or () if re.search(rf"\b{re.escape(t)}\b", clause["text"])}
        literal_hit = None
        for shown in clause["calls"]:
            for ref in reference_calls:
                if ref["name"] == shown["name"] and literal_match(shown["args"], ref["arguments"]):
                    literal_hit = (shown, ref)
                    break
            if literal_hit:
                break
        if literal_hit:
            findings.append({"kind": "literal", "clause": clause["text"], "call": literal_hit[0]["text"],
                             "reference": literal_hit[1]})
            continue
        content = content_literals(clause, tool_names)
        clause_text = normalize(clause["text"])
        for ref in reference_calls:
            if ref["name"] not in named_tools:
                continue
            if "_addressees" not in ref:
                ref["_addressees"] = addressees(ref["arguments"])
                ref["_text"] = args_text(ref["arguments"])
            shared = [a for a in ref["_addressees"] if a and a in clause_text]
            if not shared:
                continue
            ref_text = ref["_text"]
            if content and any(c not in ref_text for c in content):
                continue
            findings.append({"kind": "broad", "clause": clause["text"], "addressee": shared[0],
                             "reference": ref})
            break
    return findings


NEGATIVE = re.compile(r"(^|_)not(_|$)|_never_|_no_|_without_")


def same_service(tool, assertion_type):
    """A tool and an assertion about the same app (gmail_*, slack_*, google_sheets_*, ...)."""
    t, a = tool.split("_"), assertion_type.split("_")
    width = 2 if t[0] in ("google", "facebook", "linkedin", "zoho", "help") else 1
    return t[:width] == a[:width]


def required_effect_contradiction(line, assertions, tool_names):
    """Prohibition clauses that forbid an effect a positive native check of the task requires: the
    clause names a tool of the check's app and the check's addressee, and no content beyond what the
    check itself names (plan D24 R2: never forbid an action the task requires)."""
    findings = []
    for clause in prohibition_clauses(line, tool_names):
        named = {c["name"] for c in clause["calls"]} | {
            t for t in tool_names or () if re.search(rf"\b{re.escape(t)}\b", clause["text"])}
        clause_text = normalize(clause["text"])
        content = content_literals(clause, tool_names)
        for assertion in assertions:
            if assertion.get("excluded") or NEGATIVE.search(assertion["type"]):
                continue
            if not any(same_service(t, assertion["type"]) for t in named):
                continue
            params = assertion.get("params") or {}
            shared = [a for a in addressees(params) if a in clause_text]
            if not shared:
                continue
            required_text = args_text(params)
            if content and any(c not in required_text for c in content):
                continue
            findings.append({"kind": "required_effect", "clause": clause["text"], "addressee": shared[0],
                             "assertion": assertion["type"], "params": params})
            break
    return findings


ALIASES = {"spreadsheet_id": "spreadsheet", "worksheet_id": "worksheet", "row_id": "row", "channel_id": "channel",
           "channel_name": "channel"}


def target_identity(arguments):
    """The record or addressee a call acts on: its addressing arguments (aliases folded)."""
    out = {}
    for key, value in (arguments or {}).items():
        lowered = key.lower()
        if lowered in ADDRESS_KEYS or lowered.endswith("_id") or lowered.endswith("_email"):
            out[ALIASES.get(lowered, lowered)] = value_text(value)
    return out


def span_contradiction(span, attempt_call, reference_calls):
    """Record-aware C1 for a bad span with offending text: a reference call contradicts the line only
    when it makes the same effect (same tool) on the same record or addressee and carries the same
    offending content. Addressee spans: the reference sends to that addressee. Content spans: the
    reference's addressing arguments equal the attempt call's (where both have them) and its
    arguments contain the offending text."""
    text = normalize(span.get("offending_text") or "")
    if not text or attempt_call is None:
        return None
    mine = target_identity(attempt_call["arguments"] if isinstance(attempt_call["arguments"], dict) else {})
    for ref in reference_calls:
        if ref["name"] != span["name"]:
            continue
        if span.get("offending_kind") == "addressee":
            if text in addressees(ref["arguments"]):
                return ref
            continue
        theirs = target_identity(ref["arguments"])
        shared = set(mine) & set(theirs)
        if any(mine[k] != theirs[k] for k in shared):
            continue
        if text in args_text(ref["arguments"]):
            return ref
    return None


def native_index(source):
    match = re.match(r"native\[(\d+)\]", str(source))
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------- C12 requirement already satisfied

REQUIREMENT = re.compile(r"still missing|must contain|must have|is required|required by the end|should contain",
                         re.IGNORECASE)
FIELD_OF_PARAM = {"subject": ("subject", "title", "name"), "body": ("body", "text", "message", "content", "description",
                                                                     "notes", "comment"),
                  "text": ("text", "body", "message", "content"), "cell": ("cells", "fields", "fields_json", "values")}


def param_field(key):
    key = key.lower()
    for prefix, fields in FIELD_OF_PARAM.items():
        if key.startswith(prefix):
            return fields
    return None


def service_of(fact, assertions=None):
    """The app a requirement line is about: from its native source's type, or a goal family 'goal:<app>.x'."""
    for source in fact.get("sources") or []:
        parsed = parse_source(source)
        if parsed and parsed["kind"] == "native":
            parts = parsed["type"].split("_")
            return "_".join(parts[:2]) if parts[0] in ("google", "facebook", "linkedin") else parts[0]
    family = str(fact.get("family") or "")
    if family.startswith("goal:"):
        app = family[5:].split(".")[0]
        return None if app in ("service",) else app
    return None


def literal_fields(fact):
    """Quoted literal -> the call fields its native sub-condition is about (None: anywhere)."""
    out = {}
    for source in fact.get("sources") or []:
        parsed = parse_source(source)
        if not parsed or parsed["kind"] != "native":
            continue
        for key, value in (parsed["params"] or {}).items():
            fields = param_field(key)
            for atom in atoms(value):
                out[normalize(atom)] = fields
    return out


def field_text(arguments, fields):
    expanded = {}
    for key, value in (arguments or {}).items():
        if isinstance(value, str) and value.strip()[:1] in "[{":
            try:
                value = json.loads(value)
            except ValueError:
                pass
        expanded[key.lower()] = value
    if fields is None:
        return args_text(expanded)
    return args_text({k: v for k, v in expanded.items() if any(k.startswith(f) for f in fields)})


def requirement_line(fact):
    """A rendered line that states something required (not a prohibition or a tool error)."""
    text = fact.get("text") or ""
    if fact.get("kind") in ("prohibited", "tool_error") or re.match(
            r"(turns? [\d, ]+: )?(remove|send nothing|do not|don't|never)\b", text, re.IGNORECASE):
        return False  # prohibitions name content the attempt did write; that is their point
    if re.match(r"(turns? [\d, ]+: )?(link|change|set|move|rename|update|replace|correct)\b", text, re.IGNORECASE) \
            or re.search(r"\badded here\b|\bis extra\b", text, re.IGNORECASE):
        return False  # a correction of an effect the attempt made, named by its own content
    return fact.get("kind") == "missing" or bool(REQUIREMENT.search(text))


def requirement_satisfied(fact, line, calls, assertions=None):
    """C12 for one requirement line. Returns None, or a finding when the attempt's closest effect already
    satisfies a condition the line states as missing: 'all_stated_satisfied' when every stated condition
    holds (the line is false or omits the failing part), 'states_a_satisfied_condition' when some do."""
    # a parenthetical reason ("(in the source data the "Severity" is "Serious")") explains, it states no condition
    stated = re.sub(r"\([^()]*(?:\([^()]*\)[^()]*)*\)", " ", line)  # parentheticals explain; they state no condition
    quoted = [normalize(q) for q in quoted_literals(stated)]
    if not quoted:
        return None
    # D34: a line about an email to a recipient the checker cannot resolve (a pointer such as "the recipient named
    # in ..." or "the sender of ...") has no closest effect it can identify: not checkable here
    if re.search(r"\b(sent to|email to|go to|goes to) the (recipient named in|sender of)\b", stated):
        return None
    service = service_of(fact, assertions)
    # D34: a goal line naming its effect kind ("a created twilio sms_message", "a created hubspot ticket") is
    # about that app and object only
    made = re.search(r"\ba created (\w+) (\w+)", stated)
    nouns = []
    if made:
        service = service or made.group(1).lower()
        nouns = [n for n in made.group(2).lower().split("_") if len(n) >= 3]
    if service is None:
        # D34: a line about "the Airtable comment ...", "the email to ..." names its app
        named = re.search(r"\b(?:The|the) (Airtable|HubSpot|Gorgias|Zendesk|Salesforce|DocuSign|Slack|Calendly|Asana|"
                          r"Trello|Notion|Zoom|Mailchimp|Twilio)\b|\b(?:The|the) (email|SMS) to\b", stated)
        if named:
            service = (named.group(1) or {"email": "gmail", "SMS": "twilio"}[named.group(2)]).lower()
    writes = [c for c in calls if not c["error"] and isinstance(c["arguments"], dict)
              and not re.search(r"(find|search|get|list|read|lookup|retrieve|fetch|query|describe)", c["name"])
              and (service is None or c["name"].startswith(service))
              and (not nouns or any(n in c["name"] for n in nouns))]
    if not writes:
        return None
    noun = None
    for source in fact.get("sources") or []:
        parsed = parse_source(source)
        if parsed and parsed["kind"] == "native" and service:
            rest = parsed["type"][len(service):].strip("_").split("_")
            noun = rest[0] if rest and rest[0] else None
    same_kind = [c for c in writes if noun and noun in c["name"][len(service or ""):].split("_")]
    writes = same_kind or writes
    identity = set()
    for source in fact.get("sources") or []:
        parsed = parse_source(source)
        if parsed and parsed["kind"] == "native":
            identity |= {normalize(v) for k, v in (parsed["params"] or {}).items()
                         if not param_field(k) and isinstance(v, (str, int)) and len(str(v)) >= 3}
    addressed = [q for q in quoted if "@" in q or q in identity or q.startswith("#")]
    # D34: a quoted value that is a selector argument of a write (a channel, record, ticket, sheet or row id) names
    # WHERE the requirement applies, not WHAT it requires
    selectors = {normalize(str(v)) for c in writes for k, v in c["arguments"].items()
                 if isinstance(v, (str, int)) and re.search(r"(^|_)(channel|channel_id|record_?id|recordid|ticket_id|"
                                                            r"conversation_id|spreadsheet(_id)?|worksheet(_id)?|"
                                                            r"row(_id)?|envelope_id|object|table(name)?)$",
                                                            str(k).lower())}
    addressed += [q for q in quoted if q in selectors and q not in addressed]
    fields = literal_fields(fact)
    # D34: a goal line's own field binding ('"body" must contain "X"'): the literal is checked in that field only
    for field, rest_text in re.findall(r'"([A-Za-z_][\w ]*)" must contain(?:, verbatim,)? ((?:"[^"]*"(?:, | and | or )?)+)',
                                       stated):
        for q in quoted_literals(rest_text):
            fields.setdefault(normalize(q), (field.lower(),))

    def score(c):
        text = args_text(c["arguments"])
        return (sum(a in text for a in addressed), sum(q in text for q in quoted))
    addressed = set(addressed)
    closest = max(writes, key=score)
    record_differs = bool(addressed) and not any(a in args_text(closest["arguments"]) for a in addressed)
    # D34: a row requirement (native row_id) against a write to another row is about a different record
    rows = {str(v) for source in fact.get("sources") or [] for parsed in [parse_source(source)]
            if parsed and parsed["kind"] == "native" for k, v in (parsed["params"] or {}).items() if k == "row_id"}
    wrote = {str(closest["arguments"].get(k)) for k in ("row_id", "row") if closest["arguments"].get(k) is not None}
    if rows and wrote and not rows & wrote:
        return None
    if record_differs and not same_kind:
        return None  # no effect of that kind on that addressee or record
    content = [q for q in quoted if q not in addressed and q not in identity]
    if not content:
        return None
    satisfied = [q for q in content if q in field_text(closest["arguments"], fields.get(q))]
    if not satisfied:
        return None
    kind = ("all_stated_satisfied" if len(satisfied) == len(content) and not record_differs
            else "states_a_satisfied_condition")
    return {"kind": kind, "satisfied": satisfied, "not_satisfied": [q for q in content if q not in satisfied],
            "record_differs": record_differs, "closest_call": f"turn {closest['turn']}: {closest['name']}("
                            f"{json.dumps(closest['arguments'], ensure_ascii=False)[:400]})"}


# ---------------------------------------------------------------- C13 framing ("still missing" an effect that exists)

def world_aliases(record):
    """Normalised identity -> its alias group, from the task's initial world: Slack channel id/name (with and
    without '#'), spreadsheet and worksheet id/title, and any other record with an id and a name or title."""
    groups = {}

    def visit(value):
        if isinstance(value, dict):
            ident = value.get("id")
            label = value.get("name") or value.get("title")
            if isinstance(ident, str) and isinstance(label, str) and ident and label:
                group = {normalize(ident), normalize(label), normalize("#" + label.lstrip("#")),
                         normalize(label.lstrip("#"))}
                merged = set(group)
                for member in group:
                    merged |= groups.get(member, set())
                for member in merged:
                    groups[member] = merged
            for v in value.values():
                visit(v)
        elif isinstance(value, list):
            for v in value:
                visit(v)
    visit(((trace_of(record).get("task") or {}).get("data") or {}).get("initial_state") or {})
    return groups


def identity_forms(value, aliases):
    text = normalize(value)
    return aliases.get(text, set()) | {text, text.lstrip("#")}


CONTAINER_KEYS = {"spreadsheet", "spreadsheet_id", "worksheet", "worksheet_id", "base_id", "applicationid", "table",
                  "table_id", "table_name", "tablename", "project", "project_id", "board_id", "list_id", "workspace",
                  "workspace_id", "folder_id", "calendar_id"}
MESSAGE_WORDS = ("message", "reply", "comment", "note", "email", "post")


def framing_failure(fact, calls, aliases):
    """C13: a "Still missing ... <effect kind> to/on/in <identity>" line although the attempt made a successful
    effect of that kind on that identity (the line should say what that effect lacks, not that it is missing)."""
    text = fact.get("text") or ""
    if not re.search(r"\bstill missing\b", text, re.IGNORECASE) or not fact.get("sources"):
        return None
    service = service_of(fact)
    if not service:
        return None
    noun, messageish = None, False
    identities = set()
    for source in fact.get("sources") or []:
        parsed = parse_source(source)
        if parsed and parsed["kind"] == "native":
            rest = parsed["type"][len(service):].strip("_").split("_")
            noun = noun or (rest[0] if rest and rest[0] else None)
            messageish = messageish or any(w in rest for w in MESSAGE_WORDS)
            for key, value in (parsed["params"] or {}).items():
                if key.lower() in CONTAINER_KEYS:
                    continue  # a sheet or table holds the record; it is not the record
                if not param_field(key) and isinstance(value, (str, int)) and (
                        key.lower() in ADDRESS_KEYS or key.lower().endswith(("_id", "_name", "_email"))):
                    identities |= identity_forms(value, aliases)
    messageish = messageish or bool(re.search(r"\b(reply|message|comment|note|email)\b", text, re.IGNORECASE))
    for quoted in quoted_literals(text):
        if "@" in quoted or quoted.startswith("#"):
            identities |= identity_forms(quoted, aliases)
    identities = {i for i in identities if len(i) >= 2}
    if not identities:
        return None
    writes = [c for c in calls if not c["error"] and isinstance(c["arguments"], dict)
              and c["name"].startswith(service)
              and not re.search(r"(find|search|get|list|read|lookup|retrieve|fetch|query|describe)", c["name"])
              and (noun is None or noun in c["name"][len(service):].split("_")
                   or (noun == "message" and re.search(r"send|post|reply", c["name"])))
              and (not messageish or re.search(r"send|reply|message|comment|note|post", c["name"]))]
    for call in writes:
        mine = set()
        for key, value in call["arguments"].items():
            if key.lower() in CONTAINER_KEYS:
                continue
            if key.lower() in ADDRESS_KEYS or key.lower().endswith(("_id", "_name", "_email")):
                for atom in atoms(value):
                    for part in str(atom).split(","):
                        mine |= identity_forms(part.strip(), aliases)
        shared = identities & mine
        if shared:
            return {"identity": sorted(shared)[0],
                    "existing_effect": f"turn {call['turn']}: {call['name']}("
                                       f"{json.dumps(call['arguments'], ensure_ascii=False)[:300]})"}
    return None


# ---------------------------------------------------------------- C3 goal rule

def verified_goal_ids(task, path=None):
    """Goals the builder's verified-goals list marks verified for a task."""
    import sdpo_ab_context as builder

    path = Path(path or getattr(builder, "VERIFIED_GOALS_PATH", HERE / "sdpo_ab_verified_goals.json"))
    if not path.exists():
        return set()
    goals = json.loads(path.read_text()).get(task) or {}
    return {gid for gid, row in goals.items() if (row or {}).get("verdict") == "verified"}


def goal_failures(record, verified=None):
    """Verified goals that apply to an episode but are not satisfied: every applicable trace-level verdict
    (status other than 'inapplicable') must be status 'valid' with value 1; 'abstained' is a failure."""
    from sdpo_ab_metrics import manifest_findings

    verified = verified_goal_ids(task_of(record)) if verified is None else verified
    out = []
    for f in manifest_findings(record):
        if not f["producer"].endswith("obligations") or f["kind"] != "trace" or f["signal"].endswith(".coverage"):
            continue
        goal = f["check_id"] or f["signal"]
        if goal not in verified or f.get("status") == "inapplicable":
            continue
        if f.get("status") != "valid" or f.get("value") != 1:
            out.append({"goal": goal, "status": f.get("status"), "value": f.get("value"), "reason": f.get("reason")})
    return out


# ---------------------------------------------------------------- C6 grader internals

def grader_vocabulary(contracts):
    """Assertion type names (raw and humanised) and check ids (raw and humanised)."""
    from automationbench.rubric import assertions as _  # noqa: F401  (registers handlers)
    from automationbench.rubric.registry import AssertionRegistry

    types = sorted(AssertionRegistry._handlers)
    ids = set()
    for contract in contracts:
        for check in (contract or {}).get("checks", []):
            for key in ("check_id", "signal_id"):
                value = check.get(key)
                if value:
                    ids.add(value)
                    ids.add(value.split(".", 1)[-1])
    return types, sorted(ids)


@__import__("functools").lru_cache(maxsize=None)
def humanise(identifier):
    return re.sub(r"\s+", " ", re.sub(r"[._\-]", " ", identifier)).strip().lower()


# Humanised assertion names that are also ordinary wording (reviewed 2026-10-28: "the Slack message in
# channel X" in a variant-D line is plain language, not the grader's type name).
ORDINARY_PHRASES = {"slack message in channel"}


def internals_in(line, types, ids, row_names):
    hits = []
    lowered = line.lower()
    plain = normalize(line)
    for kind, names, need_underscore in (("assertion_type", types, False), ("check_id", ids, True)):
        for name in names:
            raw = name.lower()
            if (not need_underscore or "_" in raw) and raw in lowered and \
                    re.search(rf"(?<![a-z0-9_]){re.escape(raw)}(?![a-z0-9_])", lowered):
                hits.append((kind, name))
                continue
            human = humanise(name)
            if human in ORDINARY_PHRASES:
                continue
            if len(human.split()) >= 3 and human in plain and re.search(rf"\b{re.escape(human)}\b", plain):
                hits.append((f"{kind}_humanised", human))
    for match in re.finditer(r"\brows? (\d+)\b", line, re.IGNORECASE):
        names = row_names.get(int(match.group(1)), set())
        if not any(n in plain for n in names):
            hits.append(("row_without_name", match.group(0)))
    # keep the longest humanised match only when one contains another
    texts = [h[1] for h in hits]
    return [h for h in hits if not any(h[1] != t and h[1] in t for t in texts)]


def row_names_of(world_like):
    """row id -> non-numeric cell values of rows with that id (any sheet), for 'row N' naming."""
    out = defaultdict(set)

    def visit(value):
        if isinstance(value, dict):
            row_id = value.get("row_id", value.get("id")) if "cells" in value else None
            if row_id is not None and isinstance(value.get("cells"), dict):
                try:
                    key = int(row_id)
                except (TypeError, ValueError):
                    key = None
                if key is not None:
                    for cell in value["cells"].values():
                        text = normalize(cell)
                        if len(text) >= 3 and not re.fullmatch(r"[\d.,$%\- ]+", text):
                            out[key].add(text)
            for v in value.values():
                visit(v)
        elif isinstance(value, list):
            for v in value:
                visit(v)
    visit(world_like)
    return out


# ---------------------------------------------------------------- C4 leak (independent)

def grader_literals(record, contract):
    values = []
    for assertion in (trace_of(record).get("info") or {}).get("automationbench", {}).get("assertions") or []:
        values += atoms({k: v for k, v in (assertion.get("params") or {}).items() if k not in ("excluded", "scored")})
    for source in (contract or {}).get("sources", {}).values():
        fields = source.get("fields")
        items = fields.values() if isinstance(fields, dict) else fields if isinstance(fields, list) else []
        for item in items:
            values += atoms(item.get("value") if isinstance(item, dict) else item)
    out = {}
    for v in values:
        key = normalize(v)
        if (len(key) >= 3 and not re.fullmatch(r"[\d.,]+", key)) or re.fullmatch(r"\d{2,}([.,]\d+)?", key):
            out.setdefault(key, str(v).strip())
    return out  # normalised literal -> original spelling


def template_word_only(value, text, original=None):
    """A grader value spelled as a lowercase plain word (e.g. 'failed', 'pending') that the line uses
    only as its own wording, never inside a quoted value: not a leaked grader value."""
    if not re.fullmatch(r"[a-z]+(?: [a-z]+)?", original if original is not None else value):
        return False
    quoted = " ".join(normalize(q) for q in quoted_literals(text))
    return not re.search(rf"(?<![a-z0-9]){re.escape(value)}(?![a-z0-9])", quoted)


def request_and_results(record):
    """What R5 lets a line print: the request (user messages) and the tool results the attempt received."""
    parts = []
    for node in trace_of(record)["nodes"]:
        message = node.get("message") or {}
        if message.get("role") in ("user", "tool"):
            content = message.get("content")
            parts.append(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False))
    return normalize("\n".join(parts))


def observed_before_written(value, record):
    """`value` is a whole token of the request, or of a tool result received before the attempt itself first
    wrote it in a call's arguments (a result echoing the attempt's own write is not an observation)."""
    if token_in(value, normalize(first_user_text(record))):
        return True
    written = ""
    for call in recorded_calls(record):
        if isinstance(call["arguments"], dict):
            written += " " + args_text(call["arguments"])
        if token_in(value, written):
            return False
        result = call["result"]
        text = normalize(result if isinstance(result, str) else json.dumps(result, ensure_ascii=False))
        if token_in(value, text):
            return True
    return False


TURN_TAG = re.compile(r"\b(turns? [\d, ]+(?:and \d+)?|at turn \d+)", re.IGNORECASE)


def token_in(value, text):
    """`value` occurs in `text` as a whole token or phrase (word boundaries; both already normalised)."""
    return re.search(rf"(?<![a-z0-9]){re.escape(value)}(?![a-z0-9])", text) is not None


def quoted_own_offence(value, line, written):
    """A prohibition line may quote the attempt's own offending text: `value` is inside a quote of a
    prohibition line and is a whole token of the attempt's own call arguments."""
    if not re.match(r"\s*(-\s*)?(note on your next action:\s*)?(turns? [\d, ]+:\s*)?(remove|send nothing|do not|don't|"
                    r"never|must not)\b", line, re.IGNORECASE) and not PROHIBITION.search(line):
        return False
    return any(value in normalize(q) for q in quoted_literals(line)) and token_in(value, written)


def own_values(record):
    """Every whole value the attempt wrote in its call arguments (JSON strings expanded), normalised."""
    values = set()
    for call in recorded_calls(record):
        if not isinstance(call["arguments"], dict):
            continue
        for value in call["arguments"].values():
            if isinstance(value, str) and value.strip()[:1] in "[{":
                try:
                    value = json.loads(value)
                except ValueError:
                    pass
            values |= {normalize(a) for a in atoms(value)}
    return values


def independent_leaks(text, record, contract, observed=None):
    """Grader literals printed in `text` that the attempt never observed (R5).

    Observed means a whole-token match in the request, or in a tool result received before the attempt itself
    first wrote the value; the attempt's own call arguments do not count, except for offending text a
    prohibition line quotes back. Turn tags ("Turn 23:") are not values. A grader literal counts as printed
    when it is a whole token of the text, or part of a word inside a quote (a quoted stem such as "negotiat")."""
    written = args_text_all(record)
    mine = own_values(record)
    # D34: a field name the attempt itself used as an argument key ("rating") names where a value goes; it is
    # not a hidden value
    keys = {normalize(str(k)) for c in recorded_calls(record) if isinstance(c.get("arguments"), dict)
            for k in c["arguments"]}
    out = []
    for value, original in grader_literals(record, contract).items():
        if value in keys:
            continue
        if observed is not None and token_in(value, observed):
            continue
        if observed is None and observed_before_written(value, record):
            continue
        offending = []
        for line in text.splitlines():
            plain = normalize(TURN_TAG.sub(" ", line))
            quoted = [normalize(q) for q in quoted_literals(line)]
            if token_in(value, plain) or any(value in q for q in quoted):
                if template_word_only(value, line, original):
                    continue
                if quoted_own_offence(value, line, written):
                    continue
                containing = [q for q in quoted if value in q]
                if containing and not token_in(value, normalize(TURN_TAG.sub(" ", re.sub(r'"[^"]*"', " ", line)))) \
                        and all(q in mine for q in containing):
                    continue  # only inside a quote of a whole value the attempt itself wrote (its own record)
                offending.append(line)
        if offending:
            out.append(value)
    return sorted(out)


# ---------------------------------------------------------------- C8 traceability

REF_KEYS = ("refs", "references", "sources", "source", "checks", "check_ids", "check_id", "findings",
            "finding_ids", "finding_id", "trace", "evidence")


def fact_references(fact):
    for key in REF_KEYS:
        if key in fact and fact[key] not in (None, "", []):
            value = fact[key]
            return value if isinstance(value, list) else [value]
    return None


def failed_reference_ids(record):
    """Identifiers that really failed in this episode: native assertions and manifest findings."""
    from sdpo_ab_metrics import manifest_findings

    ids = set()
    for index, assertion in failed_assertions(record):
        ids |= {f"native:{index}", str(index), assertion["type"],
                f"{assertion['type']}:{json.dumps(assertion.get('params'), sort_keys=True)}"}
    for finding in manifest_findings(record):
        failed = finding["value"] == 0 if finding["kind"] == "trace" else finding["value"] == 1
        if failed and finding.get("status", "valid") == "valid":
            for key in ("check_id", "signal"):
                if finding.get(key):
                    ids.add(finding[key])
                    ids.add(finding[key].split(".", 1)[-1])
            if finding.get("invocation"):
                ids.add(finding["invocation"])
    return ids


def reference_resolves(ref, failed_ids, record=None):
    """A fact source that names a check which really failed in this episode."""
    tool_error = re.fullmatch(r"tool_error:turn(\d+):call(\d+)", str(ref))
    if tool_error and record is not None:
        position = (int(tool_error.group(1)), int(tool_error.group(2)))
        return any((c["turn"], c["call_index"]) == position and c["error"] for c in recorded_calls(record))
    parsed = parse_source(ref)
    if parsed is not None and record is not None:
        if parsed["kind"] == "native":
            assertions = (trace_of(record).get("info") or {}).get("automationbench", {}).get("assertions") or []
            index = parsed.get("index")
            candidates = [assertions[index]] if isinstance(index, int) and index < len(assertions) else assertions
            return any(a["type"] == parsed["type"] and canonical(a.get("params") or {}) == canonical(parsed["params"])
                       and not a.get("passed") and not a.get("excluded") for a in candidates)
        return parsed["check_id"] in failed_ids or parsed["check_id"].split(".", 1)[-1] in failed_ids
    if isinstance(ref, dict):
        candidates = [str(v) for k, v in ref.items() if k in ("id", "check_id", "signal", "signal_id", "type",
                                                               "index", "native", "finding", "invocation")]
    else:
        candidates = [str(ref)]
    return any(c in failed_ids or c.split(".", 1)[-1] in failed_ids for c in candidates)


# ---------------------------------------------------------------- C9 excluded checks

EXCLUSIONS = HERE / "sdpo_ab_check_exclusions.json"


def canonical(value):
    """Canonical JSON with integral floats as ints (the registry's matching rule)."""
    def norm(v):
        if isinstance(v, float) and v.is_integer():
            return int(v)
        if isinstance(v, dict):
            return {k: norm(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        return v
    return json.dumps(norm(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_exclusions(path):
    """The exclusion registry as matchable keys: ('native', task, type, canonical params) and
    ('manifest', task, check_id). None when the registry does not exist yet."""
    path = Path(path)
    if not path.exists():
        return None
    keys = set()
    for entry in json.loads(path.read_text()).get("entries", []):
        check = entry.get("check") or {}
        if check.get("native"):
            native = check["native"]
            keys.add(("native", entry["task"], native["type"], canonical(native.get("params") or {})))
        if check.get("manifest"):
            keys.add(("manifest", entry["task"], check["manifest"]))
    return keys


def parse_source(source):
    """A fact source id: 'native[i]:type:{params}' or 'manifest:check_id' (variant D's form), or a
    dict with type/params or check_id."""
    if isinstance(source, dict):
        if "check_id" in source:
            return {"kind": "manifest", "check_id": source["check_id"]}
        if "type" in source:
            return {"kind": "native", "index": source.get("index"), "type": source["type"],
                    "params": source.get("params") or {}}
        return None
    text = str(source)
    match = re.match(r"native\[(\d+)\]:([A-Za-z0-9_]+):(.*)$", text, re.DOTALL)
    if match:
        try:
            params = json.loads(match.group(3))
        except ValueError:
            return None
        return {"kind": "native", "index": int(match.group(1)), "type": match.group(2), "params": params}
    if text.startswith("manifest:"):
        return {"kind": "manifest", "check_id": text[len("manifest:"):]}
    return None


def source_excluded(source, task, keys):
    parsed = parse_source(source)
    if parsed is None:
        return False
    if parsed["kind"] == "native":
        return ("native", task, parsed["type"], canonical(parsed["params"])) in keys
    return ("manifest", task, parsed["check_id"]) in keys


def broken_goal_candidates(pool, allowed=None, threshold=0.95):
    """Manifest goals and native checks failing in >= threshold of the natively successful
    episodes of their task (independent of the registry)."""

    tally = GoalTally()
    for _, record in pool:
        tally.add(record)
    return tally.candidates(allowed, threshold)


class GoalTally:
    """Per task: natively successful episodes, and how many of them fail each goal or check."""

    def __init__(self):
        self.seen, self.counts, self.totals = set(), defaultdict(Counter), Counter()

    def add(self, record):
        from sdpo_ab_metrics import manifest_findings

        if record.get("id") in self.seen:
            return
        self.seen.add(record.get("id"))
        if (trace_of(record).get("metrics") or {}).get("task_completed_correctly") != 1.0:
            return
        name = task_of(record)
        self.totals[name] += 1
        failing = set()
        for finding in manifest_findings(record):
            if (finding["kind"] == "trace" and finding["value"] == 0 and finding.get("status", "valid") == "valid"
                    and not finding["signal"].endswith(".coverage") and "guards" not in finding["producer"]):
                failing.add(("manifest_goal", finding.get("check_id") or finding["signal"]))
        for assertion in (trace_of(record).get("info") or {}).get("automationbench", {}).get("assertions") or []:
            if not assertion.get("passed"):
                failing.add(("native_check" + ("_excluded" if assertion.get("excluded") else ""),
                             f"{assertion['type']} {json.dumps(assertion.get('params'), sort_keys=True)}"))
        for item in failing:
            self.counts[name][item] += 1

    def candidates(self, allowed=None, threshold=0.95, minimum=3):
        counts, totals = self.counts, self.totals
        out = []
        for name, counter in sorted(counts.items()):
            if allowed is not None and name not in allowed:
                continue
            for (kind, check), n in counter.most_common():
                if totals[name] >= minimum and n / totals[name] >= threshold:
                    out.append({"task": name, "kind": kind, "check": check, "failed_in": n,
                                "successful_episodes": totals[name]})
        return out


# ---------------------------------------------------------------- gating checks for one context (shared)

GATING_CHECKS = ("C1D", "C4", "C5", "C6", "C7", "C8", "C9", "C11", "C12", "C13", "C14", "C15")
_ASSERTION_TYPES = []


def assertion_types():
    if not _ASSERTION_TYPES:
        _ASSERTION_TYPES.extend(grader_vocabulary([])[0])
    return _ASSERTION_TYPES


def successful_references(records):
    """task -> distinct non-error calls of natively successful, harm-free records (C1D's references)."""
    from sdpo_ab_metrics import episode_outcome

    out = defaultdict(dict)
    for record in records:
        outcome = episode_outcome(record)
        if outcome["success"] != 1.0 or outcome["harm"]:
            continue
        pool = out[task_of(record)]
        for c in recorded_calls(record):
            key = (c["name"], json.dumps(c["arguments"], sort_keys=True, default=str))
            if not c["error"] and key not in pool:
                pool[key] = {"name": c["name"], "arguments": c["arguments"], "episode": record["id"]}
    return {task: list(pool.values()) for task, pool in out.items()}


def no_fact_failure(record, context):
    """C5 for a built context: a failed episode must have at least one fact."""
    failed = (trace_of(record).get("metrics") or {}).get("task_completed_correctly") != 1.0
    if failed and (context is None or not context.get("facts")):
        return {"check": "C5", "kind": "no_fact",
                "line": "(no context)" if context is None else
                f"(context without facts; skip_reason: {context.get('skip_reason')})"}
    return None


def check_context_gating(record, context, references, *, exclusions=None, contract=None, headings=None,
                         counts=None):
    """The gating checks C1D, C4, C5, C6, C7, C8, C9, C11 and C12 for one built context; no replay.

    `references`: successful calls of the same task (dicts with name, arguments and episode; see
    `successful_references`). `exclusions`: registry keys from `load_exclusions`, a registry path, or None for
    the committed registry (C9 is skipped when the registry does not exist). Returns the failures, each a dict
    with "check" and the offending "line". `counts` (a Counter) accumulates "checked:<id>" and "extra:<id>:<kind>"
    for reports, and "no_ids" when the facts carry no source ids (C8/C9 not checkable)."""
    import sdpo_ab_context as builder

    counts = counts if counts is not None else Counter()
    failures = []
    name = task_of(record)
    first = no_fact_failure(record, context)
    counts["checked:C5"] += 1 if (trace_of(record).get("metrics") or {}).get("task_completed_correctly") != 1.0 else 0
    if first:
        failures.append(first)
    if context is None or context.get("first_message") is None:
        return failures
    if exclusions is None or isinstance(exclusions, (str, Path)):
        exclusions = load_exclusions(exclusions or EXCLUSIONS)
    if contract is None:
        contract = builder.contract_for(name)
    headings = headings or (getattr(builder, "SOLUTION_HEADING", "Correct solution:"),
                            (getattr(builder, "FAILED_HEADING", ""), getattr(builder, "SUCCESS_HEADING", "")),
                            getattr(builder, "CLOSING", "Correctly solve the original question."))
    parts = context_parts(context, record, headings)
    lines = context_lines(parts)
    calls = recorded_calls(record)
    facts = context.get("facts") or []

    def fail(check, **detail):
        failures.append({"check": check, **detail})

    # C1D: record-aware contradiction of a bad span with a successful attempt.
    by_position = {(c["turn"], c["call_index"]): c for c in calls}
    for sp in context.get("bad_spans") or []:
        if not sp.get("offending_text") or sp.get("kind") not in ("prohibited", "changed"):
            continue
        counts["checked:C1D"] += 1
        attempt = by_position.get((sp["turn"], sp["call_index"]))
        ref = span_contradiction(sp, attempt, references)
        if ref is not None:
            fact = next((f for f in facts if f.get("kind") == sp.get("kind")
                         and sp["turn"] in (f.get("turns") or [f.get("turn")])), {})
            fail("C1D", line=short(fact.get("text", "")), offending_text=sp["offending_text"],
                 offending_kind=sp.get("offending_kind"),
                 attempt_call=short(json.dumps((attempt or {}).get("arguments"), ensure_ascii=False), 300),
                 successful_episode=ref.get("episode"),
                 successful_call=short(f"{ref['name']}({json.dumps(ref['arguments'], ensure_ascii=False)})", 300))

    # C4: leaks (builder function and independent version) in the feedback the teacher reads.
    feedback_text = "\n".join(line for _, _, line in lines)
    counts["checked:C4"] += 1
    raw_builder_leaks = builder.leaked_literals(feedback_text, record, contract)
    spelled = grader_literals(record, contract)
    builder_leaks = [v for v in raw_builder_leaks if not template_word_only(v, feedback_text, spelled.get(v))]
    if len(raw_builder_leaks) != len(builder_leaks):
        counts["extra:C4:builder_hits_that_are_template_words_only"] += 1
    written = args_text_all(record)
    lines_text = feedback_text.splitlines()
    quoted_back = [v for v in builder_leaks if all(quoted_own_offence(v, ln, written)
                                                   for ln in lines_text if v in normalize(ln))]
    if quoted_back:
        counts["extra:C4:builder_hits_quoting_the_attempts_own_offence"] += 1
    builder_leaks = [v for v in builder_leaks if v not in quoted_back]
    own_leaks = independent_leaks(feedback_text, record, contract)
    for kind, found in (("builder_leaked_literals", builder_leaks), ("independent", own_leaks)):
        if found:
            counts[f"extra:C4:{kind}"] += 1
    leaks = sorted(set(builder_leaks) | set(own_leaks))
    if leaks:
        offending = next((line for _, _, line in lines if any(v in normalize(line) for v in leaks)), "")
        fail("C4", line=short(offending), literals=leaks[:10],
             found_by=[k for k, v in (("builder", builder_leaks), ("independent", own_leaks)) if v])

    # C6: grader internals (text the attempt itself observed, e.g. a quoted tool error, is exempt).
    observed = observed_text(record)
    rows = row_names_of((trace_of(record)["task"]["data"] or {}).get("initial_state") or {})
    ids = grader_vocabulary([contract])[1]
    for where, turn, line in lines:
        counts["checked:C6"] += 1
        visible = line
        for quoted in quoted_literals(line):
            if normalize(quoted) in observed:
                visible = visible.replace(quoted, " ")
        hits = internals_in(visible, assertion_types(), ids, rows)
        if hits:
            for kind, _ in hits:
                counts[f"extra:C6:{kind}"] += 1
            fail("C6", where=where, turn=turn, line=short(line), terms=[f"{k}: {t}" for k, t in hits][:8])

    # C7: fact lines in the first message.
    counts["checked:C7"] += 1
    fact_lines = [line for line in parts["feedback"] if line.lstrip().startswith("-")] or parts["feedback"]
    if len(fact_lines) > MAX_FACT_LINES:
        fail("C7", line=f"{len(fact_lines)} fact lines", first=short(fact_lines[0], 200))

    # C8 traceability and C9 excluded checks.
    refs = [fact_references(f) for f in facts]
    if facts and all(r is None for r in refs):
        counts["no_ids"] += 1
    else:
        failed_ids = failed_reference_ids(record)
        for fact, ref in zip(facts, refs, strict=True):
            if exclusions is not None:
                counts["checked:C9"] += 1
                bad = [x for x in ref or [] if source_excluded(x, name, exclusions)]
                if bad:
                    fail("C9", line=short(fact.get("text")), excluded_references=bad)
            counts["checked:C8"] += 1
            if not ref or not all(reference_resolves(x, failed_ids, record) for x in ref):
                fail("C8", line=short(fact.get("text")), references=ref)

    # C11: placeholder lines; C12: requirement already satisfied.
    counts["checked:C11"] += len(lines)
    for where, turn, line in lines:
        if UNSEEN_TEXT in line:
            fail("C11", where=where, turn=turn, line=short(line))
    for fact in facts:
        if not requirement_line(fact) or not fact.get("sources"):
            continue
        counts["checked:C12"] += 1
        found = requirement_satisfied(fact, fact.get("text") or "", calls)
        if found:
            counts[f"extra:C12:{found['kind']}"] += 1
            fail("C12", line=short(fact.get("text") or ""), **found)

    # C13: a "still missing" line about an effect the attempt did make on that identity.
    aliases = world_aliases(record)
    for fact in facts:
        if not re.search(r"\bstill missing\b", fact.get("text") or "", re.IGNORECASE):
            continue
        counts["checked:C13"] += 1
        framed = framing_failure(fact, calls, aliases)
        if framed:
            fail("C13", line=short(fact.get("text") or ""), **framed)

    # C14: a source locator (email subject or sender, record name, worksheet, channel) carries a hidden value.
    hidden = [v for v in grader_literals(record, contract) if v not in observed]
    for where, turn, line in lines:
        counts["checked:C14"] += 1
        for locator in source_locators(line):
            carried = [v for v in hidden if v in normalize(locator)]
            if carried:
                fail("C14", where=where, turn=turn, line=short(line), locator=locator, literals=carried[:5])
    # C14 (D29b): a locator carrying a code shaped like a required literal of the SAME check, but different
    for fact in facts:
        own = set()
        for source in fact.get("sources") or []:
            parsed = parse_source(source)
            if parsed and parsed["kind"] == "native":
                own |= {normalize(str(v)) for v in atoms(parsed["params"] or {})}
        for locator in source_locators(fact.get("text") or ""):
            clash = same_shape_conflict(locator, sorted(own))
            if clash:
                counts["extra:C14:conflicting_same_shape_literal"] += 1
                fail("C14", line=short(fact.get("text") or ""), locator=locator, conflicting=clash)

    # C15: a hold-note pointer names a place that really holds a hold note tied to a record the line names.
    initial = ((trace_of(record).get("task") or {}).get("data") or {}).get("initial_state") or {}
    for where, turn, line in lines:
        for place in re.findall(r"hold notes? (?:(?:is|are) )?in (.*?)(?: says |;|\.$|$)", line):
            counts["checked:C15"] += 1
            if not hold_note_exists(initial, place, line):
                fail("C15", where=where, turn=turn, line=short(line), place=place)
    return failures


_LOCATOR = re.compile(r'(?:the email|an email from|from|worksheet|spreadsheet|channel|the (?:Salesforce|DocuSign|hubspot)'
                      r' [a-z]+) "([^"]+)"')
_HOLD_WORDS = re.compile(r"\b(suspended|on hold|legal hold|do not (?:close|merge|consolidate|update|send|contact|"
                         r"touch|process)|pending (?:\w+ )?dispute)\b", re.IGNORECASE)


def same_shape_conflict(locator, required):
    """A code in a locator shaped like a required literal but different (SCH-W02-2026 vs SCH-W05-2026)."""
    code = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)+")
    shape = lambda t: re.sub(r"[a-z]", "a", re.sub(r"[0-9]", "9", t))  # noqa: E731
    wanted = [r for r in required if code.fullmatch(r) and re.search(r"[0-9]", r)]
    for token in code.findall(normalize(locator)):
        for r in wanted:
            if shape(token) == shape(r) and token != r and re.split(r"[-_]", token)[0] == re.split(r"[-_]", r)[0]:
                return [token, r]
    return None


def source_locators(line):
    """The quoted locating fields of every named source in a line (independent of the builder's phrasing code)."""
    return [m.group(1) for m in _LOCATOR.finditer(line)]


def hold_note_exists(initial, place, line):
    """Is there, at the named place, a note with hold wording that names a record the line also names?"""
    quoted = [normalize(q) for q in re.findall(r'"([^"]+)"', line)]
    named = [q for q in quoted if q not in {normalize(x) for x in re.findall(r'"([^"]+)"', place)}]
    sheets = initial.get("google_sheets") or {}
    m = re.match(r'worksheet "([^"]+)" of spreadsheet "([^"]+)"', place)
    if m:
        title, sheet = m.group(1), m.group(2)
        ids = {w.get("id") for w in sheets.get("worksheets") or [] if isinstance(w, dict)
               and (w.get("title") == title or w.get("id") == title)} | {title}
        for row in sheets.get("rows") or []:
            if row.get("spreadsheet_id") != sheet or row.get("worksheet_id") not in ids:
                continue
            text = " | ".join(str(v) for v in (row.get("cells") or {}).values())
            if _HOLD_WORDS.search(text) and any(re.search(r"(?<![\w-])" + re.escape(n) + r"(?![\w-])",
                                                           normalize(text)) for n in named):
                return True
        return False
    m = re.match(r'the email "([^"]+)"', place) or re.match(r'an email from "([^"]+)"', place)
    if m:
        for message in (initial.get("gmail") or {}).get("messages") or []:
            if m.group(1) not in (message.get("subject"), message.get("from_"), message.get("from")):
                continue
            text = " ".join(str(message.get(k) or "") for k in ("subject", "body_plain", "body"))
            if _HOLD_WORDS.search(text) and any(n in normalize(text) for n in named):
                return True
        return False
    m = re.match(r'a Slack message in channel "([^"]+)"', place)
    if m:
        for message in (initial.get("slack") or {}).get("messages") or []:
            text = str(message.get("text") or "")
            if message.get("channel_id") == m.group(1) and _HOLD_WORDS.search(text) and any(
                    n in normalize(text) for n in named):
                return True
    return False


# ---------------------------------------------------------------- frozen environment replay

class Replayer:
    """In-process execution of tool calls through the environment's own MCP tool server."""

    def __init__(self):
        self.tasks, self.tool_cache, self.initials, self.servers, self.names = {}, {}, {}, {}, {}
        self.trajectories, self._adapter = {}, None

    def task(self, name):
        if name not in self.tasks:
            from automationbench_v1.taskset import AutomationBenchConfig, AutomationBenchTaskset
            from sdpo_frozen_environment import taskset_config

            [self.tasks[name]] = AutomationBenchTaskset(
                AutomationBenchConfig.model_validate(taskset_config([name]))).load()
        return self.tasks[name]

    def allowed_tools(self, name):
        return tuple(self.task(name).config.allowed_tools)

    def required_parameters(self, tool):
        if tool not in self.tool_cache:
            import inspect

            from automationbench.tools import ALL_TOOLS

            funcs = {f.__name__: f for f in ALL_TOOLS}
            func = funcs.get(tool)
            self.tool_cache[tool] = (None if func is None else
                                     [p for p, spec in inspect.signature(func).parameters.items()
                                      if p != "world" and spec.default is inspect.Parameter.empty])
        return self.tool_cache[tool]

    async def initial(self, name, day=None):
        """The task's initial state (JSON of the state model) as the environment's setup makes it.

        Not cached: a cached first setup did not reproduce the recorded results (setup is not a pure
        function of the task on the loaded task object), while a fresh setup per run reproduces
        every recorded result of a batch byte for byte."""
        key = (name, day.date().isoformat() if day else None)
        self.initials.pop(key, None)
        if key not in self.initials:
            import automationbench_v1.taskset as taskset_module
            from automationbench_v1.simulation import world_clock_base
            from automationbench_v1.tools import AutomationBenchState

            class Holder:
                state = AutomationBenchState()

            original = taskset_module.world_clock_base
            if day is not None:
                taskset_module.world_clock_base = lambda initial, today=None: world_clock_base(initial, today=day)
            try:
                await self.task(name).setup(Holder, None)
            finally:
                taskset_module.world_clock_base = original
            Holder.state.capture_actions = False  # capture only records evidence; it never changes results
            self.initials[key] = self.adapter().dump_json(Holder.state)
        return self.initials[key]

    def adapter(self):
        if self._adapter is None:
            from automationbench_v1.tools import AutomationBenchState
            from pydantic import TypeAdapter

            self._adapter = TypeAdapter(AutomationBenchState)
        return self._adapter

    def server(self, name):
        """One MCP server per task with the environment's toolset; state comes from `box`."""
        if name not in self.servers:
            from mcp.server.mcpserver import MCPServer
            from verifiers.v1.mcp.server import _call_state

            task = self.task(name)
            adapter = self.adapter()
            box = {"json": None}
            [toolset] = task.toolsets(task.config)

            async def pull():
                return adapter.validate_json(box["json"])

            async def push(before):
                box["json"] = adapter.dump_json(_call_state.get())

            toolset._pull_state = pull
            toolset._push_state = push
            server = MCPServer("automationbench-replay")
            toolset.register(server)
            self.servers[name] = (server, box)
        return self.servers[name]

    async def execute(self, name, calls, start):
        """Run calls in order from a state; returns (results, state JSON after each call)."""
        from mcp.client import Client
        from verifiers.v1.harnesses.utils.compaction import bound_tool_message
        from verifiers.v1.harnesses.utils.mcp import mcp_content_to_chat_content

        server, box = self.server(name)
        box["json"] = start
        results, states = [], []
        async with Client(server) as client:
            if name not in self.names:
                self.names[name] = {tool.name for tool in (await client.list_tools()).tools}
            names = self.names[name]
            for call in calls:
                arguments = call.get("arguments")
                if call.get("valid_json") is False or not isinstance(arguments, dict):
                    content = ("error: tool arguments must be a JSON object; resend as an object"
                               if call.get("valid_json") is not False else "error: invalid JSON in tool arguments")
                elif call["name"] not in names:
                    content = f"error: unknown tool {call['name']!r}"
                else:
                    try:
                        result = await client.call_tool(call["name"], arguments)
                        content = mcp_content_to_chat_content(result.content)
                    except Exception as error:  # noqa: BLE001 - a transport failure is an error result
                        content = f"error: replay transport failure: {type(error).__name__}: {error}"
                results.append(bound_tool_message({"content": content})["content"])
                states.append(box["json"])
        return results, states

    async def run(self, name, calls, day=None):
        """Set up the task's initial world, run calls in order; returns (results, final state)."""
        start = await self.initial(name, day)
        results, states = await self.execute(name, calls, start)
        return results, self.adapter().validate_json(states[-1] if states else start)

    async def trajectory(self, record):
        """An episode's recorded calls replayed once: (calls, results, state before each call + final)."""
        key = record.get("id")
        if key not in self.trajectories:
            name = task_of(record)
            calls = recorded_calls(record)
            start = await self.initial(name, episode_day(record))
            results, states = await self.execute(name, calls, start)
            self.trajectories = {key: (calls, results, [start] + states)}  # keep only the latest episode
        return self.trajectories[key]

    @staticmethod
    def score(state):
        from automationbench_v1.scoring import score_world

        return score_world(world=state.world, initial_state=state.initial_state, assertions=state.assertions)


def compare_prefix(recorded, replayed):
    exact = sum(1 for c, r in zip(recorded, replayed) if c["result"] == r)
    ids_only = sum(1 for c, r in zip(recorded, replayed) if c["result"] != r and ids_only_difference(c["result"], r))
    return {"calls": len(recorded), "identical": exact, "ids_only": ids_only,
            "different": len(recorded) - exact - ids_only}


# ---------------------------------------------------------------- C3 harness (manifest harm)

async def harness_replay(solutions, work_dir, port=18871, concurrency=8):
    """Run each scripted solution (one call per turn) through the real harness with manifest
    assessments; returns key -> outcome. `solutions`: list of (key, task_name, calls).

    One evaluation runs every task with as many rollouts as its most-used task has solutions. The
    scripted model hands each new rollout of a task the next unused script and recognises the
    rollout on later turns by the script number carried in its tool-call ids; spare rollouts end at
    once with a text reply and are ignored."""
    import asyncio as _asyncio

    from aiohttp import web
    from automationbench_v1.taskset import AutomationBenchConfig, AutomationBenchTaskset
    from sdpo_ab_metrics import episode_outcome
    from sdpo_ab_replay import environment
    from sdpo_frozen_environment import taskset_config
    from verifiers.v1.cli.eval.runner import run_eval
    from verifiers.v1.configs.cli.eval import EvalConfig

    if not solutions:
        return {}
    by_task = defaultdict(list)
    for key, name, calls in solutions:
        by_task[name].append((key, calls))
    names = sorted(by_task)
    pending, scripts, prompts = {}, {}, {}
    for task in AutomationBenchTaskset(AutomationBenchConfig.model_validate(taskset_config(names))).load():
        prompt = task.data.prompt
        user = next((m for m in prompt if (m.get("role") if isinstance(m, dict) else m.role) == "user"), None) \
            if isinstance(prompt, list) else None
        content = (user.get("content") if isinstance(user, dict) else getattr(user, "content", None)) \
            if user is not None else prompt
        prompts[json.dumps(content, sort_keys=True)] = task.data.task_name
    number = 0
    for name in names:
        pending[name] = []
        for key, calls in by_task[name]:
            scripts[number] = (key, name, calls)
            pending[name].append(number)
            number += 1
    lock = _asyncio.Lock()

    async def completions(request):
        body = await request.json()
        messages = body.get("messages", [])
        user = next(m for m in messages if m.get("role") == "user")
        name = prompts.get(json.dumps(user["content"], sort_keys=True))
        assistants = [m for m in messages if m.get("role") == "assistant"]
        script = None
        if assistants:
            first = (assistants[0].get("tool_calls") or [{}])[0].get("id", "")
            match = re.match(r"s(\d+)_t\d+$", first or "")
            script = int(match.group(1)) if match else None
        elif name is not None:
            async with lock:
                script = pending[name].pop(0) if pending[name] else None
        turn = len(assistants)
        calls = scripts[script][2] if script is not None else []
        if turn < len(calls):
            call = calls[turn]
            message = {"role": "assistant", "content": "", "tool_calls": [{
                "id": f"s{script}_t{turn}", "type": "function",
                "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}]}
            finish = "tool_calls"
        else:
            message, finish = {"role": "assistant", "content": "Done."}, "stop"
        return web.json_response({"id": f"verify-{time.time_ns()}", "object": "chat.completion",
                                  "created": int(time.time()), "model": "replay",
                                  "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                                  "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    os.environ.setdefault("VERIFY_REPLAY_KEY", "EMPTY")
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_post("/v1/chat/completions", completions)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    try:
        config = EvalConfig.model_validate({
            "env": environment(names), "model": "replay",
            "client": {"type": "eval", "base_url": f"http://127.0.0.1:{port}/v1", "api_key_var": "VERIFY_REPLAY_KEY"},
            "sampling": {"temperature": 0.0, "max_tokens": 4096}, "num_tasks": len(names), "task_keys": names,
            "num_rollouts": max(len(v) for v in by_task.values()), "max_concurrent": concurrency,
            "shuffle": False, "output_dir": str(Path(work_dir) / "harness"), "run": {"name": "verify", "dir": "."},
            "push": False, "rich": None, "serve": None})
        episodes = await run_eval(config)
    finally:
        await runner.cleanup()
    outcomes = {}
    for episode in episodes:
        record = episode.model_dump(mode="json")
        ids = [c.get("id", "") for n in trace_of(record)["nodes"]
               for c in ((n.get("message") or {}).get("tool_calls") or [])]
        match = re.match(r"s(\d+)_t\d+$", ids[0]) if ids else None
        if not match:
            continue  # a spare rollout
        key, name, calls = scripts[int(match.group(1))]
        outcome = episode_outcome(record)
        executed = recorded_calls(record)
        outcomes[key] = {"success": outcome["success"], "harm": outcome["harm"],
                         "manifest_harms": outcome["manifest_harms"],
                         "preservation_breaks": outcome["preservation_breaks"],
                         "tool_errors": sum(c["error"] for c in executed), "calls_executed": len(executed),
                         "failed_checks": [a["type"] for _, a in failed_assertions(record)],
                         "goal_failures": goal_failures(record)}
    return outcomes


# ---------------------------------------------------------------- report helpers

class Check:
    def __init__(self, check_id, title):
        self.id, self.title = check_id, title
        self.count, self.checked, self.examples, self.extra = 0, 0, [], Counter()
        self.not_checkable = None

    def hit(self, example):
        self.count += 1
        if len(self.examples) < MAX_EXAMPLES:
            self.examples.append(example)

    def as_dict(self):
        out = {"title": self.title, "count": self.count, "checked": self.checked, "examples": self.examples}
        if self.extra:
            out["breakdown"] = dict(self.extra)
        if self.not_checkable:
            out["not_checkable"] = self.not_checkable
        out["passed"] = self.count == 0 and not self.not_checkable
        return out


def episode_files(patterns):
    """Episode files matching globs: absolute, relative to the repository root, or relative to
    docs/research/evidence (e.g. 'sdpo-ab-2026102*/**/episodes.jsonl.gz')."""
    import glob

    found = set()
    for pattern in patterns:
        candidates = [pattern] if os.path.isabs(pattern) else [str(REPO / pattern), str(EVIDENCE / pattern)]
        for candidate in candidates:
            found |= {Path(p) for p in glob.glob(candidate, recursive=True) if Path(p).is_file()}
    return sorted(found)


def read_file(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def iter_episodes(patterns):
    """(path, record) one file at a time, so only one file's records are held in memory."""
    for path in episode_files(patterns):
        for record in read_file(path):
            yield path, record


def task_filter(name):
    if name == "all":
        return None
    return set(json.loads(TASK_SETS[name].read_text())["task_names"])


def rel(path):
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def short(text, limit=700):
    text = str(text)
    return text if len(text) <= limit else text[:limit] + " [...]"


# ---------------------------------------------------------------- main verification

async def verify(args):
    import sdpo_ab_context as builder
    from sdpo_ab_metrics import episode_outcome

    started = time.monotonic()
    headings = (getattr(builder, "SOLUTION_HEADING", "Correct solution:"),
                (getattr(builder, "FAILED_HEADING", ""), getattr(builder, "SUCCESS_HEADING", "")),
                getattr(builder, "CLOSING", "Correctly solve the original question."))
    allowed = task_filter(args.tasks)
    # Pass 1 over every recorded episode (streamed): successful references -- all native checks passed
    # and no manifest harm -- and the goal tally for broken-goal candidates.
    successes = defaultdict(dict)  # task -> distinct successful calls (first episode that made each)
    success_episodes = Counter()
    tally = GoalTally()
    reference_files = episode_files(args.references or [DEFAULT_EPISODES])
    working_calls = defaultdict(dict)  # task -> distinct calls that returned no error, in any episode
    for path, record in iter_episodes(args.references or [DEFAULT_EPISODES]):
        tally.add(record)
        for c in recorded_calls(record):
            key = (c["name"], json.dumps(c["arguments"], sort_keys=True, default=str))
            if not c["error"] and isinstance(c["arguments"], dict):
                working_calls[task_of(record)].setdefault(key, {"name": c["name"], "arguments": c["arguments"]})
        outcome = episode_outcome(record)
        if outcome["success"] == 1.0 and not outcome["harm"]:
            pool = successes[task_of(record)]
            success_episodes[task_of(record)] += 1
            for c in recorded_calls(record):
                key = (c["name"], json.dumps(c["arguments"], sort_keys=True, default=str))
                if not c["error"] and key not in pool:
                    pool[key] = {"name": c["name"], "arguments": c["arguments"], "episode": record["id"]}
    target_files = episode_files(args.episodes) if args.episodes else reference_files
    successful_calls = {task: list(pool.values()) for task, pool in working_calls.items()}

    replayer = Replayer()
    contracts = {}

    def contract(name):
        if name not in contracts:
            contracts[name] = builder.contract_for(name)
        return contracts[name]
    types, _ = grader_vocabulary([])
    task_ids = {}

    def check_ids(name):
        """The task's own check and signal ids (other tasks' ids cannot be printed from this one)."""
        if name not in task_ids:
            task_ids[name] = grader_vocabulary([contract(name)])[1]
        return task_ids[name]

    checks = OrderedDict((c.id, c) for c in [
        Check("C1", "Contradiction with a successful attempt of the same task (literal or tool+addressee) "
                    "or with the arm-B solution in the same context"),
        Check("C2", "Correction executability in the frozen environment at the point of the attempt"),
        Check("C2b", "Diagnostic (not a gate): suggested call never seen returning without error in any recorded "
                     "call of the task (the D24 count of 27/115 used this criterion)"),
        Check("C3", "Solution validity: replayed solution passes every native check with no manifest harm"),
        Check("C4", "Leak: grader literal the attempt never observed"),
        Check("C5", "Coverage: build error, or failed episode without a fact"),
        Check("C6", "Grader internals: assertion type names, check ids, 'row N' without a record name"),
        Check("C7", f"Length: more than {MAX_FACT_LINES} fact lines"),
        Check("C8", "Traceability: each fact references a check or finding that failed in the episode"),
        Check("C9", "Excluded checks: no fact derives from a check listed in sdpo_ab_check_exclusions.json"),
        Check("C1D", "Record-aware contradiction (variant D bad spans): a successful attempt makes the same effect "
                     "on the same record or addressee with the same offending content"),
        Check("C2D", "Variant D corrections (fact 'correction'): the corrected call, executed independently "
                     "in-process at that point of the attempt, returns no error"),
        Check("C10", "Coverage: every failed native check produces a line or is listed as excluded/skipped"),
        Check("C11", "Placeholder lines containing '(a value this attempt never observed)' (for review)"),
        Check("C12", "Requirement already satisfied: a requirement line states conditions the attempt's closest "
                     "effect on that addressee or record already meets"),
        Check("C13", "Framing: a 'still missing' line about an effect kind on an identity the attempt did produce "
                     "(ids and names normalised through the world)"),
        Check("C14", "Locator leak (D29): a named source's subject, sender, name, worksheet or channel carries a "
                     "grader literal the attempt never observed, or a code shaped like a required literal but "
                     "different (it would point at the wrong thing)"),
        Check("C15", "Hold pointer (D29): a 'hold note is in <place>' pointer names a place that holds a note with "
                     "hold wording tied to a record the line names"),
    ])
    skipped_by_task = defaultdict(Counter)
    skipped_examples = defaultdict(list)
    d_corrections = []
    excluded = load_exclusions(args.exclusions)
    stats = Counter()
    prohibition_lines = {"first_message": 0, "note": 0}
    prohibition_units = {}  # (task, where, line without turn tag) -> audit entry
    correction_units = {}
    fidelity = Counter()
    solution_jobs = {}
    fact_keys = set()

    async def check_context(path, record, context, siblings):
        name = task_of(record)
        episode = record.get("id")
        tools = set(replayer.allowed_tools(name)) | {t.get("name") or (t.get("function") or {}).get("name")
                                                     for t in trace_of(record).get("tools") or []}
        parts = context_parts(context, record, headings)
        lines = context_lines(parts)
        base = {"episode_id": episode, "task": name, "file": rel(path)}
        calls = recorded_calls(record)

        # C1: contradictions with successful attempts of the same task and with the shown solution.
        shown_solution = solution_calls(parts["solution"], tools) if parts["solution"] else []
        references = list(successes.get(name, {}).values())
        task_assertions = (trace_of(record).get("info") or {}).get("automationbench", {}).get("assertions") or []
        for where, turn, line in lines:
            clauses = prohibition_clauses(line, tools)
            if not clauses:
                continue
            checks["C1"].checked += 1
            prohibition_lines[where] += 1
            unit = (name, where, re.sub(r"^\s*-\s*|\bTurns? [\d, ]+:\s*", "", line).strip())
            entry = prohibition_units.setdefault(unit, {"task": name, "where": where, "line": unit[2],
                                                        "episodes": 0, "contradicts": None})
            entry["episodes"] += 1
            against_success = contradiction(line, references, tools)
            against_solution = contradiction(line, shown_solution, tools) if shown_solution else []
            against_required = required_effect_contradiction(line, task_assertions, tools)
            if not (against_success or against_solution or against_required):
                continue
            example = {**base, "where": where, "turn": turn, "line": short(line)}
            if against_success:
                f = against_success[0]
                checks["C1"].extra[f"successful_attempt_{f['kind']}_{where}"] += 1
                entry["contradicts"] = f"successful_attempt_{f['kind']}"
                entry["successful_episode"] = f["reference"]["episode"]
                entry["successful_call"] = short(f"{f['reference']['name']}("
                                                 f"{json.dumps(f['reference']['arguments'], ensure_ascii=False)})", 300)
                example.update({"kind": f"successful_attempt_{f['kind']}", "successful_episode": f["reference"]["episode"],
                                "successful_call": short(f"{f['reference']['name']}("
                                                         f"{json.dumps(f['reference']['arguments'], ensure_ascii=False)})", 400)})
            if against_required:
                f = against_required[0]
                checks["C1"].extra[f"required_effect_{where}"] += 1
                entry["contradicts_required_effect"] = f"{f['assertion']} {json.dumps(f['params'], ensure_ascii=False)}"
                example.update({"required_check": short(entry["contradicts_required_effect"], 300)})
                example.setdefault("kind", "required_effect")
            if against_solution:
                f = against_solution[0]
                checks["C1"].extra[f"solution_{f['kind']}_{where}"] += 1
                example.update({"solution_kind": f"solution_{f['kind']}", "solution_episode": context.get("solution_episode"),
                                "solution_call": short(f"{f['reference']['name']}("
                                                       f"{json.dumps(f['reference']['arguments'], ensure_ascii=False)})", 400)})
            checks["C1"].hit(example)

        # C2: every suggested call must execute without error at the point of the attempt.
        for where, turn, line in lines:
            suggestions = suggestion_calls(line, tools)
            if not suggestions:
                continue
            turns = [turn] if turn is not None else line_turns(line)
            for suggestion in suggestions:
                targets_ = [c for c in calls if c["name"] == suggestion["name"] and c["error"]
                            and (not turns or c["turn"] in turns)]
                if not targets_:
                    targets_ = [c for c in calls if c["name"] == suggestion["name"] and (not turns or c["turn"] in turns)]
                if not targets_:
                    targets_ = [None]
                for target in targets_[:1] if where == "first_message" else targets_:
                    shown = {k: v for k, v in suggestion["args"].items() if v is not UNSEEN}
                    if target is None:
                        prefix = [c for c in calls if turns and c["turn"] < min(turns)] if turns else calls
                        merged = shown
                    else:
                        prefix = [c for c in calls if (c["turn"], c["call_index"]) < (target["turn"], target["call_index"])]
                        merged = {**(target["arguments"] if isinstance(target["arguments"], dict) else {}), **shown}
                    key = (episode, suggestion["text"], json.dumps(merged, sort_keys=True, default=str), len(prefix))
                    if key not in correction_units:
                        recorded, replayed, states = await replayer.trajectory(record)
                        exactness = compare_prefix(prefix, replayed[:len(prefix)])
                        for k in ("calls", "identical", "ids_only", "different"):
                            fidelity[f"prefix_{k}"] += exactness[k]
                        # As shown (the gate) and merged onto the failed call's other arguments (diagnostic).
                        results, _ = await replayer.execute(name, [{"name": suggestion["name"], "arguments": shown},
                                                                   ], states[len(prefix)])
                        merged_results, _ = await replayer.execute(name, [{"name": suggestion["name"],
                                                                           "arguments": merged}], states[len(prefix)])
                        correction_units[key] = {
                            "result": results[-1], "error": is_error_result(results[-1]),
                            "merged_result": merged_results[-1], "merged_error": is_error_result(merged_results[-1]),
                            "prefix": exactness, "executed_call": {"name": suggestion["name"], "arguments": shown},
                            "merged_call": {"name": suggestion["name"], "arguments": merged},
                            "seen_working": any(c["name"] == suggestion["name"] and literal_match(shown, c["arguments"])
                                                for c in successful_calls.get(name, ())),
                            "suggested": suggestion["text"], "line": short(line, 400), "task": name}
                    verdict = correction_units[key]
                    checks["C2"].checked += 1
                    if verdict["prefix"]["different"]:
                        checks["C2"].extra["prefix_not_reproduced"] += 1
                    elif verdict["prefix"]["ids_only"]:
                        checks["C2"].extra["prefix_ids_only_difference"] += 1
                    checks["C2"].extra["as_shown_" + ("error" if verdict["error"] else "ok") + "__merged_"
                                       + ("error" if verdict["merged_error"] else "ok")] += 1
                    checks["C2b"].checked += 1
                    if not verdict["seen_working"]:
                        checks["C2b"].hit({**base, "where": where, "turn": turn, "line": short(line),
                                           "suggested_call": suggestion["text"],
                                           "environment_result": short(verdict["result"], 300)})
                    if verdict["error"]:
                        checks["C2"].extra[f"failed_{where}"] += 1
                        checks["C2"].hit({**base, "where": where, "turn": turn, "line": short(line),
                                          "suggested_call": suggestion["text"],
                                          "environment_result": short(verdict["result"], 400),
                                          "merged_call": short(json.dumps(verdict["merged_call"], ensure_ascii=False), 400),
                                          "merged_result": short(verdict["merged_result"], 300),
                                          "prefix_fidelity": verdict["prefix"]})

        # C3: the shown solution, from the initial world.
        if context.get("solution_episode"):
            source = next(({"id": s.get("id"), "day": episode_day(s)} for s in siblings[(path, name)]
                           if s.get("id") == context["solution_episode"]), None)
            job_key = hashlib.sha256(json.dumps([name, shown_solution], sort_keys=True, default=str).encode()).hexdigest()[:16]
            if job_key not in solution_jobs:
                solution_jobs[job_key] = {"task": name, "calls": shown_solution, "source": source,
                                          "contexts": []}
            solution_jobs[job_key]["contexts"].append({**base})

        # C1D / C2D / C10 / C11 (variant D structure; skipped when absent).
        by_position = {(c["turn"], c["call_index"]): c for c in calls}
        builder_calls = None
        for fact in context.get("facts") or []:
            fix = fact.get("correction")
            if not fix:
                continue
            target = by_position.get((fix["turn"], fix["call_index"]))
            if target is None or not isinstance(target["arguments"], dict):
                continue
            checks["C2D"].checked += 1
            corrected = {**target["arguments"], **fix["change"]}
            _, replayed, states = await replayer.trajectory(record)
            position = calls.index(target)
            results, _ = await replayer.execute(name, [{"name": target["name"], "arguments": corrected}],
                                                states[position])
            if builder_calls is None:
                builder_calls = builder.episode_calls(record)[0]
            builder_index = next((i for i, c in enumerate(builder_calls)
                                  if (c["turn"], c["call_index"]) == (fix["turn"], fix["call_index"])), None)
            d_corrections.append({"file": rel(path), "episode_id": episode, "task": name, "call_index": builder_index,
                                  "corrected": corrected, "independent_error": is_error_result(results[0])})
            if is_error_result(results[0]):
                checks["C2D"].hit({**base, "line": short(fact.get("text")), "corrected_call": short(
                    f"{target['name']}({json.dumps(corrected, ensure_ascii=False)})", 400),
                    "environment_result": short(results[0], 300)})
        if args.variant == "D":
            covered = {native_index(src) for f in context.get("facts") or [] for src in f.get("sources") or []}
            listed = {}
            for kind in ("excluded_checks", "skipped_checks"):
                for item in context.get(kind) or []:
                    if isinstance(item, (list, tuple)):
                        source, reason = item[0], item[1]
                    elif isinstance(item, dict):
                        source, reason = item.get("source") or item.get("check"), item.get("reason")
                    else:
                        source, reason = item, "listed in the exclusion registry"
                    listed[native_index(source)] = (kind, reason)
                    if kind == "skipped_checks":
                        skipped_by_task[name][str(source).split(":")[0] if "native[" in str(source)
                                              else str(source)] += 1
                        if len(skipped_examples[name]) < 3:
                            skipped_examples[name].append({"episode_id": episode, "check": short(source, 200),
                                                           "reason": reason})
            for index, assertion in failed_assertions(record):
                checks["C10"].checked += 1
                if index in covered:
                    continue
                if index in listed:
                    if not listed[index][1]:
                        checks["C10"].hit({**base, "line": f"native[{index}] {assertion['type']} listed without reason"})
                    continue
                checks["C10"].hit({**base, "line": f"native[{index}] {assertion['type']} "
                                                    f"{short(json.dumps(assertion.get('params'), ensure_ascii=False), 200)}"
                                                    " failed but has no line and is not excluded or skipped"})
        # C1D, C4, C6, C7, C8, C9, C11, C12: the shared gating implementation (C5 is counted at build time).
        tally = Counter()
        for failure in check_context_gating(record, context, references, exclusions=excluded,
                                            contract=contract(name), headings=headings, counts=tally):
            if failure["check"] == "C5":
                continue
            check = failure.pop("check")
            checks[check].hit({**base, **failure})
        for key, value in tally.items():
            if key.startswith("checked:") and key != "checked:C5":
                checks[key.split(":", 1)[1]].checked += value
            elif key.startswith("extra:"):
                _, cid, kind = key.split(":", 2)
                checks[cid].extra[kind] += value
        stats["c8_no_ids"] += tally["no_ids"]


    for path in target_files:
        records = [r for r in read_file(path) if (allowed is None or task_of(r) in allowed)
                   and task_of(r) not in set(args.exclude_tasks or ())]
        siblings = defaultdict(list)
        for record in records:
            siblings[(path, task_of(record))].append(record)
        for record in records:
            name = task_of(record)
            episode = record.get("id")
            outcome = episode_outcome(record)
            failed = outcome["success"] != 1.0
            stats["episodes"] += 1
            stats["failed_episodes" if failed else "successful_episodes"] += 1
            if not failed and args.arm != "B":
                continue  # arm A gives successes no solution; D24 asks for failed episodes and arm-B solutions
            try:
                if args.variant == "D":
                    builder.verify_corrections(builder._correction_items(record))  # in-process path only
                    mode = args.verification or ("run" if args.arm == "B" else "cache")
                    context = builder.build_context(record, siblings[(path, name)], args.arm, args.placement,
                                                    contract(name), variant="D", verification=mode, explain=True)
                else:
                    context = builder.build_context(record, siblings[(path, name)], args.arm, args.placement,
                                                    contract(name), variant=args.variant)
            except Exception as error:  # noqa: BLE001 - every build error is a C5 finding
                checks["C5"].checked += 1
                message = f"{type(error).__name__}: {error}"
                checks["C5"].extra["build_error_failed" if failed else "build_error_success"] += 1
                checks["C5"].hit({"episode_id": episode, "task": name, "file": rel(path), "line": short(message),
                                  "kind": "build_error", "traceback": traceback.format_exc(limit=3)[-600:]})
                stats["build_errors"] += 1
                continue
            if failed:
                checks["C5"].checked += 1
                missing = no_fact_failure(record, context)
                if missing:
                    checks["C5"].extra["failed_without_fact"] += 1
                    missing.pop("check")
                    checks["C5"].hit({"episode_id": episode, "task": name, "file": rel(path), **missing})
            if context is None or context.get("first_message") is None:
                stats["no_context"] += 1
                if context is not None and context.get("skip_reason"):
                    stats[f"skipped: {context['skip_reason']}"] += 1
                continue
            if not failed and context.get("solution_episode") is None and args.arm == "B":
                stats["success_contexts_without_solution"] += 1
            stats["contexts"] += 1
            fact_keys.update(k for f in context.get("facts") or [] for k in f)
            await check_context(path, record, context, siblings)
        # crash safety: checks so far after every file
        Path(str(args.out) + ".progress.json").write_text(json.dumps(
            {"after_file": rel(path), "stats": dict(stats), "checks": {cid: c.as_dict() for cid, c in checks.items()}},
            indent=1, ensure_ascii=False, default=str))

    if excluded is None:
        checks["C9"].not_checkable = f"exclusion registry not yet available ({rel(args.exclusions)})"
    elif stats["c8_no_ids"] and not checks["C9"].checked:
        checks["C9"].not_checkable = f"variant {args.variant} gives no source check ids on its facts"
    if stats["c8_no_ids"] and not checks["C8"].checked:
        checks["C8"].not_checkable = (f"variant {args.variant} gives no check or finding ids on its facts "
                                      f"({stats['c8_no_ids']} contexts); keys seen: "
                                      f"{sorted(fact_keys)}")

    # C3 execution: in-process native checks for every distinct solution, harness for manifest harm.
    harness = {}
    if solution_jobs and args.harness:
        jobs = [(key, job["task"], job["calls"]) for key, job in solution_jobs.items() if job["calls"]]
        work = Path(args.work_dir or (Path(args.out).parent / "verify-harness"))
        work.mkdir(parents=True, exist_ok=True)
        harness = await harness_replay(jobs, work, port=args.port, concurrency=args.concurrency)
    for key, job in solution_jobs.items():
        checks["C3"].checked += 1
        calls = job["calls"]
        source = job["source"]
        problems = []
        if not calls:
            problems.append("no calls could be parsed from the solution block")
            native = None
        else:
            day = source["day"] if source else None
            results, state = await replayer.run(job["task"], calls, day=day)
            snapshot = replayer.score(state)
            failed_checks = [a["type"] for a in snapshot.assertion_results if not a.get("excluded") and not a.get("passed")]
            errors = [f"{c['name']}: {short(r, 200)}" for c, r in zip(calls, results) if is_error_result(r)]
            native = {"task_completed_correctly": snapshot.task_completed_correctly,
                      "failed_checks": failed_checks, "tool_errors": errors}
            if snapshot.task_completed_correctly != 1.0:
                problems.append(f"native checks failed in-process: {failed_checks}")
            stats["c3_solution_calls"] += len(calls)
        verdict = harness.get(key)
        if args.harness and verdict is None and calls:
            problems.append("harness produced no episode")
        if verdict is not None:
            if verdict["calls_executed"] != len(calls):
                problems.append(f"harness executed {verdict['calls_executed']} of {len(calls)} calls")
            if verdict["success"] != 1.0:
                problems.append(f"native checks failed in the harness: {verdict['failed_checks']}")
            if verdict["harm"]:
                problems.append(f"manifest harm: {verdict['manifest_harms']}")
            if verdict.get("goal_failures"):
                problems.append(f"verified goals not satisfied: {verdict['goal_failures']}")
        job["result"] = {"native_in_process": native, "harness": verdict, "problems": problems}
        if problems:
            checks["C3"].extra["invalid_solutions"] += 1
            checks["C3"].extra["contexts_with_invalid_solution"] += len(job["contexts"])
            first = job["contexts"][0]
            checks["C3"].hit({**first, "line": "; ".join(problems), "solution_episode":
                              source.get("id") if source else None, "contexts_showing_it": len(job["contexts"]),
                              "solution_calls": len(calls)})
    if not args.harness and solution_jobs:
        checks["C3"].extra["harm_not_checked_harness_disabled"] = len(solution_jobs)

    distinct_corrections = {}
    for (episode, text, merged, _), verdict in correction_units.items():
        entry = distinct_corrections.setdefault((verdict["task"], text), {
            "call": text, "error": verdict["error"], "merged_error_somewhere": verdict["merged_error"],
            "seen_working": verdict["seen_working"],
            "suggested": verdict.get("suggested"), "line": verdict.get("line"), "task": verdict.get("task"),
            "result": short(verdict["result"], 300), "executions": 0, "failing_executions": 0})
        entry["executions"] += 1
        entry["failing_executions"] += verdict["error"]
        entry["error"] = entry["error"] or verdict["error"]
        entry["merged_error_somewhere"] = entry["merged_error_somewhere"] or verdict["merged_error"]
    report = {
        "variant": args.variant, "arm": args.arm, "placement": args.placement, "tasks": args.tasks,
        "episodes_source": args.episodes or [DEFAULT_EPISODES],
        "references_source": args.references or [DEFAULT_EPISODES], "excluded_tasks": args.exclude_tasks,
        "builder": rel(builder.__file__),
        "builder_sha256": hashlib.sha256(Path(builder.__file__).read_bytes()).hexdigest(),
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "runtime_seconds": round(time.monotonic() - started, 1),
        "stats": dict(stats),
        "successful_reference_attempts": dict(sorted(success_episodes.items())),
        "c1_prohibition_lines": {
            "first_message": prohibition_lines["first_message"], "notes": prohibition_lines["note"],
            **{f"distinct_{w}": sum(1 for u in prohibition_units if u[1] == w) for w in ("first_message", "note")},
            **{f"distinct_{w}_contradicting_successful_attempt": sum(
                1 for u, e in prohibition_units.items() if u[1] == w and e["contradicts"])
               for w in ("first_message", "note")},
            "distinct_note_or_line_texts": len({u[2].replace("Note on your next action: ", "") for u in prohibition_units}),
            "distinct_texts_contradicting": len({u[2].replace("Note on your next action: ", "")
                                                 for u, e in prohibition_units.items() if e["contradicts"]}),
            **{f"distinct_{w}_contradicting_any": sum(
                1 for u, e in prohibition_units.items() if u[1] == w and
                (e["contradicts"] or e.get("contradicts_required_effect")))
               for w in ("first_message", "note")},
            "units": sorted(prohibition_units.values(), key=lambda e: (e["task"], e["where"], e["line"]))},
        "c2_corrections": {"executions": len(correction_units),
                           "distinct_task_suggestions": len(distinct_corrections),
                           "distinct_task_suggestions_never_seen_working": sum(
                               not v["seen_working"] for v in distinct_corrections.values()),
                           "distinct_failing": sum(v["error"] for v in distinct_corrections.values()),
                           "distinct_suggested_texts": len({v["suggested"] for v in correction_units.values()}),
                           "distinct_suggested_texts_failing_somewhere": len(
                               {v["suggested"] for v in correction_units.values() if v["error"]}),
                           "distinct_suggested_texts_failing_everywhere": len(
                               {v["suggested"] for v in correction_units.values()} -
                               {v["suggested"] for v in correction_units.values() if not v["error"]}),
                           "distinct": sorted(distinct_corrections.values(), key=lambda v: (not v["error"], v["call"]))},
        "c2_replay_fidelity": dict(fidelity),
        "c3_solutions": {"distinct": len(solution_jobs), "harness": bool(args.harness),
                         "details": [{"key": k, "task": j["task"], "solution_episode": (j["source"] or {}).get("id"),
                                      "calls": len(j["calls"]), "contexts": len(j["contexts"]), **j.get("result", {})}
                                     for k, j in solution_jobs.items()]},
        "checks": {cid: c.as_dict() for cid, c in checks.items()},
        "c9_broken_goal_candidates": tally.candidates(allowed),
        "c10_skipped_by_task": {t: dict(c) for t, c in sorted(skipped_by_task.items())},
        "c10_skipped_examples": dict(skipped_examples),
        "c2d_corrections": d_corrections,
        "c9_registry": rel(args.exclusions) if excluded is not None else None,
    }
    if args.variant != "D":
        for cid in ("C1D", "C2D", "C10", "C12", "C13"):
            report["checks"][cid]["not_checkable"] = "variant D structure only (bad-span offending text, fact corrections, sources)"
            report["checks"][cid]["passed"] = False
    for cid in ("C2b", "C11"):
        report["checks"][cid]["gate"] = False
    report["passed"] = all(c["passed"] for cid, c in report["checks"].items()
                           if not c.get("not_checkable") and cid not in ("C2b", "C11"))
    return report


async def replay_fidelity(patterns, limit=None):
    """Replay recorded episodes in-process and compare every tool result and native verdict."""
    replayer = Replayer()
    totals = Counter()
    examples = []
    for number, (path, record) in enumerate(iter_episodes(patterns)):
        if limit is not None and number >= limit:
            break
        calls = recorded_calls(record)
        results, state = await replayer.run(task_of(record), calls, day=episode_day(record))
        exactness = compare_prefix(calls, results)
        for key in ("calls", "identical", "ids_only", "different"):
            totals[key] += exactness[key]
        snapshot = replayer.score(state)
        recorded = [(a["type"], bool(a.get("passed"))) for a in
                    trace_of(record)["info"]["automationbench"]["assertions"]]
        replayed = [(a["type"], bool(a.get("passed"))) for a in snapshot.assertion_results]
        totals["episodes"] += 1
        totals["native_verdicts_identical"] += recorded == replayed
        if exactness["different"] and len(examples) < MAX_EXAMPLES:
            bad = next(i for i, (c, r) in enumerate(zip(calls, results))
                       if c["result"] != r and not ids_only_difference(c["result"], r))
            examples.append({"episode_id": record["id"], "task": task_of(record), "call": calls[bad]["name"],
                             "recorded": short(calls[bad]["result"], 300), "replayed": short(results[bad], 300)})
    return {"totals": dict(totals), "examples": examples}


# ---------------------------------------------------------------- execution audit (variant D's verifiers)

def _alter(calls, rng):
    """Negative controls: (label, calls) with one required write dropped, and with one value altered."""
    import sdpo_ab_context as builder

    writes = [i for i, c in enumerate(calls) if not builder.is_read(c["name"])]
    if not writes:
        return []
    dropped = [c for i, c in enumerate(calls) if i != writes[-1]]
    target = dict(calls[writes[-1]])
    arguments = dict(target["arguments"])
    keys = [k for k, v in arguments.items() if isinstance(v, str) and v]
    key = next((k for k in keys if k.lower() in ADDRESS_KEYS or k.lower().endswith("_id")), keys[0] if keys else None)
    altered = None
    if key is not None:
        arguments[key] = arguments[key] + "-altered"
        target["arguments"] = arguments
        altered = calls[:writes[-1]] + [target] + calls[writes[-1] + 1:]
    out = [(f"dropped last write {calls[writes[-1]]['name']}", dropped)]
    if altered:
        out.append((f"altered {target['name']}.{key}", altered))
    return out


def execution_audit(args):
    """C2 sample replay and C3 verify_solution audit for variant D (harness runs, kept small)."""
    import random

    import sdpo_ab_context as builder

    rng = random.Random(20261028)
    out = {"generated": datetime.now(UTC).isoformat(timespec="seconds")}
    started = time.monotonic()
    replayer = Replayer()
    allowed = task_filter(args.tasks)
    excluded_tasks = set(args.exclude_tasks or ())
    records = {}
    for path, record in iter_episodes(args.episodes or [DEFAULT_EPISODES]):
        name = task_of(record)
        if (allowed is None or name in allowed) and name not in excluded_tasks:
            records[record["id"]] = (path, record)
    # C2: the builder's in-process verdicts against full replays, on a random sample of shown corrections.
    if args.c2_sample_from:
        shown = json.loads(Path(args.c2_sample_from).read_text()).get("c2d_corrections") or []
        unique = list({(c["episode_id"], c["call_index"], json.dumps(c["corrected"], sort_keys=True)): c
                       for c in shown if c["episode_id"] in records and c["call_index"] is not None}.values())
        sample = rng.sample(unique, min(args.c2_sample, len(unique)))
        items = [(records[c["episode_id"]][1], c["call_index"], c["corrected"]) for c in sample]
        in_process = builder.verify_corrections(items, mode="in_process")
        builder._VERIFIED.clear()
        replay = builder.verify_corrections(items, concurrency=args.concurrency, mode="replay")
        rows = []
        for c, a, b in zip(sample, in_process, replay):
            rows.append({"episode_id": c["episode_id"], "task": c["task"], "call_index": c["call_index"],
                         "corrected": c["corrected"], "in_process_ok": a.get("ok"), "replay_ok": b.get("ok"),
                         "checker_in_process_error": c.get("independent_error"),
                         "replay_reason": b.get("reason")})
        out["c2_sample"] = {"size": len(rows), "agree": sum(r["in_process_ok"] == r["replay_ok"] for r in rows),
                            "replay_ok": sum(bool(r["replay_ok"]) for r in rows), "rows": rows}
    # C3: shown-solution candidates from different tasks pass; negative controls are rejected.
    if args.c3_solutions:
        by_task = defaultdict(list)
        for path, record in records.values():
            if (trace_of(record).get("metrics") or {}).get("task_completed_correctly") == 1.0:
                by_task[task_of(record)].append(record)
        picks = []
        for wanted in args.c3_include or []:
            record = next((r for r, _ in ((rec, None) for rec in (x[1] for x in records.values()))
                           if r["id"].startswith(wanted)), None)
            variants = builder.solution_variants(record) if record else []
            if variants:
                picks.append((record, variants[0][1]))
            elif record:
                calls_ = [{"turn": c["turn"], "name": c["name"], "arguments": c["arguments"]}
                          for c in builder.episode_calls(record)[0] if not c["error"]]
                picks.append((record, calls_))
        tasks = sorted(by_task)
        while len(picks) < args.c3_solutions and any(by_task[t] for t in tasks):
            for t in tasks:
                if by_task[t] and len(picks) < args.c3_solutions:
                    record = by_task[t].pop(rng.randrange(len(by_task[t])))
                    variants = builder.solution_variants(record)
                    if variants:
                        picks.append((record, variants[0][1]))
        items = [(task_of(r), calls, builder._initial_digest(r)) for r, calls in picks]
        negatives = []
        for record, calls in picks[:args.c3_negatives]:
            for label, bad in _alter(list(calls), rng):
                negatives.append((record, label, (task_of(record), bad, builder._initial_digest(record))))
        verdicts = builder.verify_solutions(items + [n[2] for n in negatives], concurrency=args.concurrency)
        # independent: the same scripts through the real harness, judged with the verified-goal rule
        jobs = [(f"p{i}", task_of(r), calls) for i, (r, calls) in enumerate(picks)]
        jobs += [(f"n{i}", item[0], item[1]) for i, (_, _, item) in enumerate(negatives)]
        harness = asyncio.run(harness_replay(jobs, Path(args.work_dir or (Path(args.out).parent / "audit-harness")),
                                             port=args.port, concurrency=args.concurrency))

        def independent_ok(key, n_calls):
            h = harness.get(key) or {}
            return (h.get("success") == 1.0 and not h.get("harm") and not h.get("goal_failures")
                    and h.get("calls_executed") == n_calls)
        rows = []
        for index, ((record, calls), verdict) in enumerate(zip(picks, verdicts[:len(items)])):
            results, state = asyncio.run(replayer.run(task_of(record), calls, day=episode_day(record)))
            native = replayer.score(state).task_completed_correctly
            h = harness.get(f"p{index}") or {}
            rows.append({"task": task_of(record), "solution_episode": record["id"], "calls": len(calls),
                         "verify_solution_ok": verdict.get("ok"), "reason": verdict.get("reason"),
                         "harness_ok": independent_ok(f"p{index}", len(calls)),
                         "harness_goal_failures": h.get("goal_failures"), "harness_success": h.get("success"),
                         "passes": bool(verdict.get("ok")) and independent_ok(f"p{index}", len(calls)),
                         "checker_in_process_native": native,
                         "checker_in_process_errors": sum(is_error_result(r) for r in results)})
        negative_rows = []
        for index, ((record, label, item), verdict) in enumerate(zip(negatives, verdicts[len(items):])):
            results, state = asyncio.run(replayer.run(item[0], item[1], day=episode_day(record)))
            ok = bool(verdict.get("ok")) and independent_ok(f"n{index}", len(item[1]))
            negative_rows.append({"task": item[0], "solution_episode": record["id"], "control": label,
                                  "verify_solution_ok": verdict.get("ok"), "reason": verdict.get("reason"),
                                  "harness_goal_failures": (harness.get(f"n{index}") or {}).get("goal_failures"),
                                  "passes": ok,
                                  "checker_in_process_native": replayer.score(state).task_completed_correctly,
                                  "checker_in_process_errors": sum(is_error_result(r) for r in results)})
        out["c3"] = {"solutions": len(rows), "passed": sum(bool(r["passes"]) for r in rows),
                     "verify_solution_ok": sum(bool(r["verify_solution_ok"]) for r in rows),
                     "tasks": len({r["task"] for r in rows}), "rows": rows,
                     "negative_controls": len(negative_rows),
                     "negatives_rejected": sum(not r["passes"] for r in negative_rows),
                     "negative_rows": negative_rows}
    out["runtime_seconds"] = round(time.monotonic() - started, 1)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="D")
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument("--placement", choices=("first", "notes", "both"), default="both")
    parser.add_argument("--tasks", choices=("fit20", "transfer20", "all"), default="all")
    parser.add_argument("--episodes", nargs="*", help="episode file globs under docs/research/evidence "
                        "(default: every recorded episode); successful references always use all episodes")
    parser.add_argument("--references", nargs="*", help="episode globs for successful references, working "
                        "calls and broken-goal candidates (default: every recorded episode)")
    parser.add_argument("--exclude-tasks", nargs="*", default=[], help="task names to leave out of the targets")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-harness", dest="harness", action="store_false",
                        help="skip the harness replay of solutions (manifest harm then not checked)")
    parser.add_argument("--work-dir", help="folder for harness replay output (default: next to --out)")
    parser.add_argument("--port", type=int, default=18871)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--exclusions", default=str(EXCLUSIONS),
                        help="exclusion registry (C9); a missing file means 'not yet available'")
    parser.add_argument("--fidelity", action="store_true",
                        help="also replay every selected episode in-process and compare with the recording")
    parser.add_argument("--verification", choices=("run", "cache"),
                        help="variant D: builder verification mode (default: run for arm B solutions, cache for arm A)")
    parser.add_argument("--execution-audit", action="store_true",
                        help="variant D: replay a C2 sample (--c2-sample-from a report) and audit verify_solution")
    parser.add_argument("--c2-sample-from")
    parser.add_argument("--c2-sample", type=int, default=30)
    parser.add_argument("--c3-solutions", type=int, default=0)
    parser.add_argument("--c3-negatives", type=int, default=5)
    parser.add_argument("--c3-include", nargs="*", help="episode id prefixes whose solution the C3 audit must include")
    args = parser.parse_args(argv)
    if args.execution_audit:
        audit = execution_audit(args)
        Path(args.out).write_text(json.dumps(audit, indent=1, ensure_ascii=False, default=str) + "\n")
        print(json.dumps({k: {x: y for x, y in v.items() if not isinstance(y, list)} if isinstance(v, dict) else v
                          for k, v in audit.items()}, indent=1))
        return 0
    report = asyncio.run(verify(args))
    if args.fidelity:
        report["replay_fidelity"] = asyncio.run(replay_fidelity(args.episodes or [DEFAULT_EPISODES]))
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n")
    summary = {cid: (c["count"], c["checked"]) if not c.get("not_checkable") else "not checkable"
               for cid, c in report["checks"].items()}
    print(json.dumps({"out": args.out, "passed": report["passed"], "runtime_seconds": report["runtime_seconds"],
                      "checks (count, checked)": summary}, indent=1))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
