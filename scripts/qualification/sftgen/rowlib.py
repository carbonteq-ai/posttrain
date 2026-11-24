"""Shared pieces of the Route B SFT tooling: session -> row mapping, row walking, number grounding, word lists.

Pure Python (json, re, decimal, ast); importable from both the environment interpreter and the wt-sft .venv.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sdpo_ab_metrics import tool_error_text  # noqa: E402,F401  (re-exported: the runner's own error rule, D97 code)

ROW_SOURCE = "teacher-route-b"
CANONICAL_ID = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")  # posttrain.data adapters' _CANONICAL_ID
MAX_ROW_TOKENS = 32768
MAX_REPLY_TOKENS = 32768  # the runner's per-reply generation limit (eval-v2, plan v4; was 4096)
MAX_TURNS = 50
MAX_CALLS = 45
IDENTICAL_CALL_LIMIT = 5  # an identical call (name + sorted args) 5 or more times rejects the row

# Words that reveal the generation harness. A word is allowed when the task's own visible context (system prompt,
# user message, tool names, tool results) already uses it, e.g. "checks" in a request about bank checks. The verb
# "check" ("I'll check the sheet") is ordinary agent prose and is not listed (12 Nov: it was 31 of 41 hits on the 4B's
# own clean successes); the noun "checks" (the grader's checks) stays.
HARNESS_WORDS = ("abtool", "abtool2", "session", "sessions", "bash", "shell", "command", "commands", "json", "brief",
                 "checks", "assertion", "assertions", "grader", "graded", "grading",
                 "scoring", "harness", "transcript", "subagent", "claude", "sonnet", "opus", "teacher", "student",
                 "automationbench", "benchmark", "rubric", "stdout", "stderr")
HARNESS_LITERALS = ("--note", "--calls", "--text", "$PY", "$TOOL", "turns.jsonl", "meta.json", "smaller model")

# Final replies that name skipped/excluded items (system prompt: "list only items you acted on").
# Pattern of J-sft/research/final_reply_skips.py (audit A6); flagged for review, not a failure.
SKIP_PHRASES = re.compile(r"\b(skipped|skipping|excluded|excluding|exclusions?|excludes?|omitted|omitting|left out|"
                          r"not processed|ignored|were not (?:sent|included|updated|contacted)|"
                          r"no action (?:was )?(?:taken|required|needed))\b", re.I)
# 13 Nov (P7 acceptance reading): "**Exclusions applied:** TechRadar ...", "No Action Required: Brian Osei ...",
# "Exclude Archived/Draft content" slipped past the narrower pattern; each names or explains left-out items.
MARKUP = ("<tool_call>", "</tool_call>", "<function=", "<parameter=", "<tool_response>", "<|", "<think>", "</think>")


# ----------------------------------------------------------------------------- session -> row
def wrap_tool(tool: dict) -> dict:
    """The runner's tools entry (OpenAI function form, server order; the present exactness rule)."""
    return {"type": "function", "function": {"name": tool["name"], "description": tool["description"],
                                             "parameters": tool["parameters"]}}


def read_session(session: Path) -> dict:
    session = Path(session)
    meta = json.loads((session / "meta.json").read_text())
    prompt = json.loads((session / "prompt.json").read_text())
    turns = [json.loads(line) for line in (session / "turns.jsonl").read_text().splitlines() if line.strip()]
    reply_path = session / "reply.json"
    reply = json.loads(reply_path.read_text()) if reply_path.exists() else None
    score_path = session / "score.json"
    score = json.loads(score_path.read_text()) if score_path.exists() else None
    if len(turns) != meta["turns"] or sum(len(t["calls"]) for t in turns) != meta["calls"]:
        raise ValueError(f"{session}: turns.jsonl disagrees with meta.json")
    return {"meta": meta, "prompt": prompt, "turns": turns, "reply": reply, "score": score, "path": str(session)}


def row_id(model: str, task: str, prefix: str = "sft-route-b") -> str:
    identifier = f"{prefix}/{re.sub(r'[^a-z0-9._-]+', '-', model.lower())}/{task.lower()}"
    if not CANONICAL_ID.fullmatch(identifier):
        raise ValueError(f"not a canonical example id: {identifier!r}")
    return identifier


def session_row(data: dict, *, model: str, attempt: int, replay_result: dict | str = "not run",
                prefix: str = "sft-route-b") -> dict:
    """Recorded session -> framework SupervisedDataset row (key order of to_huggingface_sft_rows)."""
    meta, turns, reply = data["meta"], data["turns"], data["reply"]
    if reply is None:
        raise ValueError(f"{data['path']}: no final reply recorded (session not closed)")
    if not turns:
        raise ValueError(f"{data['path']}: no calling turn recorded")
    messages: list[dict[str, Any]] = [dict(m) for m in data["prompt"]["prompt"]]
    for t in turns:
        ids = [f"call_{t['turn']}_{c['index']}" for c in t["calls"]]
        messages.append({"role": "assistant", "content": t["note"], "tool_calls": [
            {"id": i, "type": "function", "function": {"name": c["tool"], "arguments": c["arguments"]}}
            for i, c in zip(ids, t["calls"])]})
        for i, c in zip(ids, t["calls"]):
            messages.append({"role": "tool", "tool_call_id": i, "name": c["tool"], "content": c["result"]})
    messages.append({"role": "assistant", "content": reply["text"]})
    trainable = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    metadata = {
        "task": meta["task"], "domain": meta["task"].split(".")[0], "source": ROW_SOURCE, "model": model,
        "attempt": attempt, "session": data["path"], "tool": meta.get("tool"), "tool_sha256": meta.get("tool_sha256"),
        "env_commit": meta["env_commit"], "world_date": meta["world_date"], "fresh_setup": meta["fresh_setup"],
        "world_clock": meta.get("world_clock"), "turns": meta["turns"], "calls": meta["calls"],
        "assistant_messages": len(trainable), "replay_result": replay_result,
        "finish_score": ({k: data["score"][k] for k in ("partial_credit", "task_completed_correctly",
                                                         "assertions_passed", "assertions_scored")}
                         if data.get("score") else "not run"),
        "synthetic_parts": {"tool_call_ids": "call_<turn>_<index>", "notes": "none (teacher's own)",
                            "final_reply": "none (teacher's own)"},
        "trained": "every assistant message (notes, calls, final reply)",
    }
    return {"id": row_id(model, meta["task"], prefix), "messages": messages,
            "tools": [wrap_tool(t) for t in data["prompt"]["tools"]], "trainable_message_indices": trainable,
            "metadata": metadata}


def row_calls(row: dict) -> list[dict]:
    """The row's calls in execution order, as abtool replay takes them."""
    return [{"tool": c["function"]["name"], "arguments": c["function"]["arguments"]}
            for m in row["messages"] if m["role"] == "assistant" for c in m.get("tool_calls") or []]


def serialize_row(row: dict) -> str:
    """One JSONL line exactly as the framework's materialization writes it (json.dumps defaults, key order kept)."""
    return json.dumps(row) + "\n"


# ----------------------------------------------------------------------------- numbers and grounding
NUMBER = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![\d])")
PERCENT_AFTER = re.compile(r"\s*%")
# An ordered-list marker at the start of a line ("1. ", "2) ", "- 3. "): layout, not a claim about the world.
LIST_MARKER = re.compile(r"(?m)^[ \t]*(?:[-*][ \t]+)?(\d{1,2})[.)][ \t]")
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def number_value(text: str) -> Decimal:
    return Decimal(text.replace(",", "")).normalize()


def numbers(text: str) -> list[str]:
    return NUMBER.findall(text or "")


def number_values(text: str) -> set[Decimal]:
    """Every number's value; a number written as a percentage also counts as its fraction ("6%" gives 6 and 0.06)."""
    out = set()
    text = text or ""
    for match in NUMBER.finditer(text):
        try:
            value = number_value(match.group(1))
        except InvalidOperation:
            continue
        out.add(value)
        if PERCENT_AFTER.match(text, match.end()):
            out.add((value / 100).normalize())
    return out


_EXPR_CHARS = set("0123456789.,$%()+-*/×x÷−– \t")
_EQUALS = re.compile(r"~=|≈|=")
_OPERATOR = re.compile(r"\d\s*%?\s*\)*\s*[-+*/×x÷−–]\s*\(*\s*\$?\d")
_OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b}


def _evaluate(node, steps: list[Decimal] | None = None):
    """Value of a +-*/ expression over numbers; `steps` collects every intermediate result (each operation's value)."""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, steps)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return Decimal(str(node.value))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        value = _OPS[type(node.op)](_evaluate(node.left, steps), _evaluate(node.right, steps))
        if steps is not None:
            steps.append(value)
        return value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _evaluate(node.operand, steps)
        return -value if isinstance(node.op, ast.USub) else value
    raise ValueError("not arithmetic")


def _to_python(expr: str) -> str:
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expr)
    expr = expr.replace("$", "").replace("×", "*").replace("÷", "/").replace("−", "-").replace("–", "-")
    expr = re.sub(r"(?<=[\d)\s])x(?=[\s\d(])", "*", expr)
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expr)
    return expr


def _side_value(candidate: str, steps: list[Decimal] | None = None) -> Decimal | None:
    """The value of one side of an equation, or None when it is not plain arithmetic over numbers."""
    if not candidate or not re.search(r"\d", candidate) or ISO_DATE.search(candidate):
        return None  # an ISO date is not a subtraction
    if candidate[0] in "+*/×÷)%.," or candidate[-1] in ".,+-*/×÷−–(":
        return None  # sentence punctuation or a dangling operator, not part of the arithmetic
    try:
        return _evaluate(ast.parse(_to_python(candidate), mode="eval"), steps)
    except (SyntaxError, ValueError, ZeroDivisionError, InvalidOperation, TypeError):
        return None


def _left_side(text: str, end: int) -> tuple[str, Decimal] | None:
    """The longest parseable arithmetic ending just before `end` (the equals sign)."""
    i = end
    while i > 0 and text[i - 1] in _EXPR_CHARS:
        i -= 1
    span = text[i:end]
    dates = list(ISO_DATE.finditer(span))
    if dates:
        span = span[dates[-1].end():]  # never reach into a date
    for k in range(len(span)):
        candidate = span[k:].strip()
        if not candidate or not re.search(r"\d", candidate):
            break
        value = _side_value(candidate)
        if value is not None:
            return candidate, value
    return None


def _right_side(text: str, start: int) -> tuple[str, Decimal] | None:
    """The longest parseable arithmetic starting just after `start` (the equals sign), up to the next equals sign."""
    j = start
    while j < len(text) and text[j] in _EXPR_CHARS:
        j += 1
    span = text[start:j]
    date = ISO_DATE.search(span)
    if date:
        span = span[:date.start()]
    for k in range(len(span), 0, -1):
        candidate = span[:k].strip()
        if not candidate:
            break
        value = _side_value(candidate)
        if value is not None:
            return candidate, value
    return None


def _decimals(text: str) -> int:
    return max((len(n.split(".")[1]) if "." in n else 0 for n in numbers(text)), default=0)


def expressions(text: str) -> list[dict]:
    """Every `<arithmetic> = <arithmetic or number>` written in the text, evaluated; chains `a = b = c` are checked
    pair by pair. At least one side must contain an operator between numbers. Each item: lhs, rhs, value (of the
    left side), ok, values (the numbers this equation establishes), rhs_number (when the right side is one number).
    Markdown bold markers are ignored; an ISO date is never read as a subtraction."""
    text = (text or "").replace("**", "")
    out = []
    for match in _EQUALS.finditer(text):
        left = _left_side(text, match.start())
        right = _right_side(text, match.end())
        if left is None or right is None:
            continue
        (lhs, value), (rhs, rhs_value) = left, right
        if not (_OPERATOR.search(lhs) or _OPERATOR.search(rhs)):
            continue
        steps: list[Decimal] = []
        _side_value(lhs, steps)
        _side_value(rhs, steps)
        rhs_display = rhs.replace("$", "").strip()
        single = NUMBER.fullmatch(rhs_display.lstrip("-").rstrip("%").strip())
        if single:
            percent = rhs_display.endswith("%")
            rhs_number = number_value(single.group(1)) * (-1 if rhs_display.startswith("-") else 1)
            tolerance = Decimal(5) * Decimal(10) ** (-(_decimals(rhs) + 1)) + Decimal("1e-9")
            candidates = [value, value * 100] if percent else [value]
            ok = any(abs(c - rhs_number) <= tolerance for c in candidates)
            values = candidates + [rhs_number] + (steps if ok else [])
        else:
            rhs_number = None
            tolerance = Decimal(5) * Decimal(10) ** (-(max(_decimals(rhs), _decimals(lhs)) + 1)) + Decimal("1e-9")
            ok = abs(value - rhs_value) <= tolerance
            # running-total notation "a = b + c [= d]": the right side starts with a's value and goes on from it
            # (e.g. "$50,000 x 9% = $4,500 + $500 bonus = $5,000"); b + c is computed here, so no wrong number passes
            first = NUMBER.search(rhs_display.lstrip("-"))
            if not ok and first and not rhs_display.lstrip("-")[first.end():].lstrip().startswith("%"):
                ok = abs(value - number_value(first.group(1))) <= tolerance
            values = [value, rhs_value] + (steps if ok else [])
        out.append({"lhs": lhs, "rhs": rhs_display, "value": str(value), "ok": ok, "values": values,
                    "rhs_number": rhs_number})
    return out


_DATE_EXPR = re.compile(r"(\d{4}-\d{2}-\d{2})\s*(minus|plus|-|\+|−)\s*(\d+)\s*(day|days|week|weeks)\s*"
                        r"(?:=|is|->|→)\s*(\d{4}-\d{2}-\d{2})", re.I)


def date_expressions(text: str) -> list[dict]:
    """Every `YYYY-MM-DD minus|plus N days|weeks = YYYY-MM-DD` written in the text, checked."""
    import datetime as dt

    out = []
    for m in _DATE_EXPR.finditer(text or ""):
        try:
            start, end = dt.date.fromisoformat(m.group(1)), dt.date.fromisoformat(m.group(5))
        except ValueError:
            continue
        days = int(m.group(3)) * (7 if m.group(4).lower().startswith("week") else 1)
        sign = 1 if m.group(2).lower() in ("plus", "+") else -1
        expected = start + dt.timedelta(days=sign * days)
        out.append({"text": m.group(0), "ok": expected == end, "expected": expected.isoformat(),
                    "values": [Decimal(end.year), Decimal(end.month), Decimal(end.day)]})
    return out


def grounding(text: str, context_values: set[Decimal]) -> dict:
    """Every number in `text` is in the prior context, or results from arithmetic written in `text` itself.

    A number is computed when it is a side of a correct equation, or equals a computed value rounded to the
    number's own decimal places (e.g. "(24,000 - 2,000) / 60 = 366.67, rounded to 367"), or is part of the checked
    result of a date step ("2026-01-28 minus 30 days = 2025-12-29"). A percentage counts as its fraction in both
    directions ("6%" in the context grounds 0.06, and 0.06 in the context grounds "6%"). Ordered-list markers at
    the start of a line ("1. ", "2) ") are layout and are not checked."""
    text = text or ""
    exprs = expressions(text)
    dates = date_expressions(text)
    computed: list[Decimal] = []
    for e in exprs + dates:
        if e["ok"]:
            computed += e["values"]
    wrong = [f"{e['lhs']} = {e['rhs']} (evaluates to {e['value']})" for e in exprs if not e["ok"]]
    wrong += [f"{d['text']} (is {d['expected']})" for d in dates if not d["ok"]]
    wrong_rhs = {abs(e["rhs_number"]) for e in exprs if not e["ok"] and e["rhs_number"] is not None}
    markers = {m.start(1) for m in LIST_MARKER.finditer(text)}
    ungrounded = []
    for match in NUMBER.finditer(text):
        if match.start(1) in markers:
            continue
        n = match.group(1)
        v = number_value(n)
        if v in wrong_rhs or v in context_values:
            continue
        if PERCENT_AFTER.match(text, match.end()) and (v / 100).normalize() in context_values:
            continue
        decimals = len(n.split(".")[1]) if "." in n else 0
        tolerance = Decimal(5) * Decimal(10) ** (-(decimals + 1)) + Decimal("1e-9")
        if any(abs(abs(c) - v) <= tolerance for c in computed):
            continue
        ungrounded.append(n)
    return {"ok": not ungrounded and not wrong, "ungrounded": sorted(set(ungrounded)), "wrong_expressions": wrong,
            "expressions": len(exprs)}


def argument_strings(value: Any, key: str = "") -> list[tuple[str, str]]:
    """(key, text) for every string or number inside a call's arguments."""
    out = []
    if isinstance(value, dict):
        for k, v in value.items():
            out += argument_strings(v, str(k))
    elif isinstance(value, list):
        for v in value:
            out += argument_strings(v, key)
    elif isinstance(value, bool) or value is None:
        pass
    elif isinstance(value, (int, float)):
        out.append((key, str(value)))
    else:
        out.append((key, str(value)))
    return out


LIMIT_KEYS = re.compile(r"max|limit|page|offset|top_k|size", re.I)


def words_in(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9]*", text or "")}


def harness_hits(text: str, allowed: set[str]) -> list[str]:
    found = words_in(text)
    hits = [w for w in HARNESS_WORDS if w in found and w not in allowed]
    hits += [s for s in HARNESS_LITERALS if s.lower() in (text or "").lower()]
    return hits


def content_text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
