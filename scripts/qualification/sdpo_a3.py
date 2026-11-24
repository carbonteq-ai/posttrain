"""Phase A3 (SDPO plan v4, runbook C3): can the frozen start model, given the engine's feedback, produce a verified fix
from the point where its own attempt went wrong? Measurement only: nothing here trains.

Pieces (each tested in test_sdpo_a3.py):
  ENGINE_OPTIONS / make_engine(arm)  the two teacher contexts: T-A2 = failed checks and tool facts, values hidden, no
                                     solution, no hints (arm A); T-A1 = values shown, the model's own verified sibling
                                     solution, call_fix hints (arm B, R0's teacher). Both with the final "not done" note.
  point_turn(context)                the continuation point: the earliest turn the feedback speaks about (a note located
                                     at a turn, or the final note); turn 1 means a full re-attempt with the feedback.
  calls_before(record, turn)         answered recorded calls made before that turn (the world snapshot index).
  context_digest(context)            sha256 of the teacher's text (first message, notes, final note), so the GPU phase
                                     proves it rebuilt exactly the context audited on the CPU.
  continue_episode(...)              the teacher continues to the end of the episode: every call executed in the forked
                                     world, no keep rule, limits as evaluation; returns the turns and the final grade.
  phase_a3(rt, args)                 the runner phase (`--phase a3-continue`).
  to_row(...) / leak_findings(...)   CPU: the student-visible trajectory (student prefix + teacher continuation) as an SFT
                                     row for validate_row.py --thinking, and the feedback-leak check.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

ENGINE_OPTIONS = {"line_budget": None, "char_budget": None, "not_done_source": "benchmark", "final_note": True}
ARMS = {"T-A2": {"expected_values": False, "pool": False, "fix_hints": [], "arm": "A"},
        "T-A1": {"expected_values": True, "pool": False, "fix_hints": ["call_fix"], "arm": "B"},
        # 14 Nov (added measurement arm, not pre-registered): T-A2's context plus QUIET; the smoke run showed both
        # pre-registered arms narrating the feedback in their reasoning ("as per the feedback from the earlier
        # attempt"), which a student would learn to claim without having it
        "T-A2q": {"expected_values": False, "pool": False, "fix_hints": [], "arm": "A", "base": "T-A2"},
        # 15 Nov (engine usefulness test, north star section 2 B; approved): the no-feedback control. Same points and
        # item digests as T-A2 (its engine context is built only for the digest check), but the teacher continues from
        # the student's own tokens with no feedback at all, so its fixes measure what resampling alone gives.
        "T-0": {"expected_values": False, "pool": False, "fix_hints": [], "arm": "A", "base": "T-A2",
                "no_feedback": True},
        # 15 Nov (Ali's engine requirements; Ali: move ahead on my judgement): feedback versions N and P, rewritten on
        # the CPU from T-A2's own context (weekend/engine-usefulness/rewrite.py) and read here by T-A2's digest. N = the
        # same lines once in the first message under a plain heading, no notes in the conversation; P = N with the
        # "never observed" placeholders replaced by verified locations (no values) and the missed-write lines added.
        "T-N": {"expected_values": False, "pool": False, "fix_hints": [], "arm": "A", "base": "T-A2", "rewrite": "N"},
        "T-P": {"expected_values": False, "pool": False, "fix_hints": [], "arm": "A", "base": "T-A2", "rewrite": "P"},
        # 16 Nov (overnight, Ali: improve the work on my judgement; measurement only): version Q = P + one guidance
        # line for the "this task forbids" lines (feedback_versions.q_text), aimed at requirements 4 and 5
        "T-Q": {"expected_values": False, "pool": False, "fix_hints": [], "arm": "A", "base": "T-A2", "rewrite": "Q"},
        # 16 Nov (plan approved by Ali, Step 3 change 1): version V1 = the old feedback's layout (lines in the first
        # message and again as notes at the mistake turns, final note) with P's pointers and missed-write lines
        # (feedback_versions.version_v1); its rewrite is a whole context (first message, notes, final note)
        "T-V1": {"expected_values": False, "pool": False, "fix_hints": [], "arm": "A", "base": "T-A2", "rewrite": "V1"},
        # 16 Nov (Step 3 change 2): V2 = V1 + for each failed prohibition whose target is one record of the initial
        # world, where that record is and to check it first (feedback_versions.version_v2; forbid_locate.py)
        "T-V2": {"expected_values": False, "pool": False, "fix_hints": [], "arm": "A", "base": "T-A2", "rewrite": "V2"},
        # 16-17 Nov (Step 3 change 3, widened by Ali): V3 = base + the engine's call-fix lines + one what-to-check
        # sentence per not-found tool error (feedback_versions.add_call_fixes / add_error_checks); on V1 or V2
        "T-V3-1": {"expected_values": False, "pool": False, "fix_hints": [], "arm": "A", "base": "T-A2", "rewrite": "V3-1"},
        "T-V3-2": {"expected_values": False, "pool": False, "fix_hints": [], "arm": "A", "base": "T-A2", "rewrite": "V3-2"},
        # Step 3 change 4: V4 = calm, short wording (feedback_versions.version_calm) on V1, V2, V3-1 or V3-2
        "T-V4-1": {"expected_values": False, "pool": False, "fix_hints": [], "arm": "A", "base": "T-A2", "rewrite": "V4-1"},
        "T-V4-2": {"expected_values": False, "pool": False, "fix_hints": [], "arm": "A", "base": "T-A2", "rewrite": "V4-2"},
        "T-V4-31": {"expected_values": False, "pool": False, "fix_hints": [], "arm": "A", "base": "T-A2", "rewrite": "V4-31"},
        "T-V4-32": {"expected_values": False, "pool": False, "fix_hints": [], "arm": "A", "base": "T-A2", "rewrite": "V4-32"},
        # 17 Nov (V4 diagnosis): V5 = V4-1's lines with V1's heading, closing line and note prefix (version_frame)
        "T-V5": {"expected_values": False, "pool": False, "fix_hints": [], "arm": "A", "base": "T-A2", "rewrite": "V5"}}
# v2 (16 Nov): v1's N/P/Q texts unchanged for its 268 digests, plus V1 (a whole context) for every W6 failed attempt
# v3 (16 Nov): v2 unchanged, plus V2 for every W6 failed attempt
# v4 (17 Nov): v3 unchanged, plus V3-1, V3-2, V4-1, V4-2, V4-31, V4-32 (weekend/engine-req/make_rewrites_v4.py)
# 17 Nov night: v5 = v4 with the V4 candidates rewritten without stand-in values (feedback_versions.say_unread)
# 17 Nov night: v6 = v5 with the V4 candidates tidied (feedback_versions.tidy: ",. ", the from/to of, qualifier repeats)
# 17 Nov morning: v7 = v6 unchanged, plus V5 = version_frame(V4-1) for every W6 failed attempt
REWRITES_FILE = Path(__file__).resolve().parents[2] / "docs/research/sdpo-automationbench/a3-rewrites-v7.json"
# on the GPU only scripts/qualification is shipped; the committed rewrites arrive as a staged, checksummed input
REWRITES_STAGED = Path("/stage") / REWRITES_FILE.name
_REWRITES = None
QUIET = ("When you work on this task now, reason only from the task request and the data your tools return. Do not "
         "mention these notes, any feedback, any check, or any earlier attempt in your reasoning, messages or replies.")


def teacher_context(arm, context):
    """The context the teacher reads: the engine's context, with QUIET appended to the first message for T-A2q.
    The digest is always taken on the engine's context (T-A2q items carry T-A2's digest)."""
    version = ARMS[arm].get("rewrite")
    if version:
        global _REWRITES
        if _REWRITES is None:
            _REWRITES = json.loads((REWRITES_STAGED if REWRITES_STAGED.exists() else REWRITES_FILE).read_text())
        rewritten = _REWRITES[context_digest(context)][version]  # a point without a rewrite is refused (KeyError)
        if isinstance(rewritten, dict):  # a whole context (V1): note turns back to integers (JSON keys are strings)
            # round 6 (U): a rewrite may say how its notes are given (note_role "user": own user message; read by
            # sdpo_ab_runner.teacher_sequence); without the key the notes stay in the tool result, as before
            return {**context, "first_message": rewritten["first_message"],
                    "notes": {int(k): v for k, v in (rewritten.get("notes") or {}).items()},
                    "final_note": rewritten.get("final_note"),
                    **({"note_role": rewritten["note_role"]} if rewritten.get("note_role") else {})}
        return {**context, "first_message": rewritten, "notes": {}, "final_note": None}
    if arm != "T-A2q":
        return context
    return {**context, "first_message": context["first_message"] + "\n\n" + QUIET}


def student_prefix(chain, position):
    """The student's own tokens before the action at `position` (every segment's recorded ids, exactly as sampled):
    the no-feedback control's prefix."""
    return [token for segment in chain["segments"][:position] for token in segment["ids"]]


def make_engine(arm, engine_class=None):
    """The teacher's engine for one arm. T-A1's solution comes from the attempt's own successful siblings (arm B reads
    the siblings given to the call), never from another model."""
    if engine_class is None:
        from sdpo_ab_engine import EngineContext as engine_class
    spec = ARMS[arm]
    return engine_class(expected_values=spec["expected_values"], pool=spec["pool"], fix_hints=list(spec["fix_hints"]),
                        options=dict(ENGINE_OPTIONS), withhold_untrue=True)


def point_turn(context):
    """Earliest turn (1-based action index) the feedback speaks about, or None when it says nothing located."""
    turns = [int(t) for t in (context.get("notes") or {})]
    final = (context.get("final_note") or {}).get("turn")
    if final is not None:
        turns.append(int(final))
    return min(turns) if turns else None


def context_digest(context):
    payload = {"first_message": context.get("first_message"),
               "notes": {str(k): v for k, v in sorted((context.get("notes") or {}).items(), key=lambda kv: int(kv[0]))},
               "final_note": context.get("final_note"), "solution": bool(context.get("solution_episode"))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def assistant_path(record):
    from sdpo_ab_world import path_nodes

    (trace,) = record["traces"]
    return [n["message"] for n in path_nodes(trace["nodes"])]


def calls_before(record, turn):
    """Answered recorded calls made in assistant turns 1..turn-1 (the index `rebuild_with_snapshots` snapshots at)."""
    path = assistant_path(record)
    answered = {m["tool_call_id"] for m in path if m.get("role") == "tool"}
    count, seen = 0, 0
    for m in path:
        if m.get("role") != "assistant":
            continue
        seen += 1
        if seen >= turn:
            break
        count += sum(1 for c in m.get("tool_calls") or [] if c["id"] in answered)
    return count


async def continue_episode(prefix, *, tokenizer, generate, parse, run_one, render, limits, turns_before, output_before,
                           visible=None):
    """The teacher continues from `prefix` (its own prompt: the student's turns before the point, the feedback notes up
    to the point) until it replies without a call or hits a limit. Every parsed call is executed in order in the forked
    world; its result is rendered back for the next turn. Returns {"turns", "stop", "tokens"}; each turn holds its
    generated ids, text, finish and executed calls ({id, name, arguments, status, result})."""
    tokens, turns, output, stop = list(prefix), [], output_before, None
    eos = tokenizer.eos_token_id
    while stop is None:
        if turns_before + len(turns) >= limits["max_turns"]:
            stop = "max_turns"
            break
        budget = min(limits["reply"], limits["context"] - len(tokens))
        if budget < 1:
            stop = "context_limit"
            break
        ids, finish = await generate(tokens, budget, len(turns))
        complete = finish == "stop"
        raw = tokenizer.decode(ids, skip_special_tokens=True)
        shown = visible(raw) if visible else raw
        calls = parse(shown) if complete else []
        tokens += ids
        if ids and ids[-1] != eos:
            tokens.append(eos)
        output += len(ids)
        turn = {"ids": list(ids), "text": raw, "finish": finish, "calls": []}
        turns.append(turn)
        if not complete:
            stop = "reply_limit" if budget == limits["reply"] else "context_limit"
            break
        if not calls:
            stop = "final_reply"
            break
        messages = []
        for call in calls:
            name, arguments = call["function"]["name"], call["function"]["arguments"]
            content, status = await run_one(name, arguments)
            turn["calls"].append({"id": call["id"], "name": name, "arguments": arguments, "status": status,
                                  "result": content})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content, "name": name})
        tokens += render(messages)
    return {"turns": turns, "stop": stop, "tokens": len(tokens)}


# ------------------------------------------------------------------ CPU: rows for the SFT gates, and the leak check

THINK_CLOSE = "</think>"
# must equal validate_row.A3_SOURCE / A3_PREFIX_REASON (wt-sft bb000aca); checked in test_sdpo_a3.py
A3_SOURCE = "sdpo-a3-continuation"
A3_PREFIX_REASON = "student turn before the continuation point (SDPO A3, not trained)"


def split_reasoning(raw):
    """(reasoning, visible text) of one generated reply (the generation prompt opened `<think>`)."""
    if THINK_CLOSE not in raw:
        return "", raw.strip()
    reasoning, visible = raw.split(THINK_CLOSE, 1)
    return reasoning.strip(), visible.strip()


def _call_message(call):
    arguments = call["arguments"]
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return {"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": arguments}}


def to_row(record, chain, turn, continuation, *, arm, item_id):
    """The student-visible trajectory as an SFT row: system and user from the episode, the student's own turns before
    the point (reasoning from its chain, calls and results from its trace), then the teacher's continuation. Only the
    continuation's assistant messages are trainable; the feedback is nowhere in the row."""
    path = assistant_path(record)
    head = [m for m in path if m.get("role") in ("system", "user")]
    messages = [{"role": m["role"], "content": m["content"]} for m in head[:2]]
    actions = [s for s in chain["segments"] if s["kind"] == "action"]
    seen = 0
    results = {m["tool_call_id"]: m for m in path if m.get("role") == "tool"}
    for m in path:
        if m.get("role") != "assistant":
            continue
        seen += 1
        if seen >= turn:
            break
        reasoning, _ = split_reasoning(actions[seen - 1].get("raw") or "")
        calls = [{"id": c["id"], "type": "function",
                  "function": {"name": c["name"], "arguments": json.loads(c["arguments"]) if isinstance(
                      c["arguments"], str) else c["arguments"]}} for c in m.get("tool_calls") or []]
        messages.append({"role": "assistant", "content": m.get("content") or "", "reasoning_content": reasoning,
                         **({"tool_calls": calls} if calls else {})})
        for c in m.get("tool_calls") or []:
            r = results[c["id"]]
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": r["content"], "name": c["name"]})
    prefix_turns = [k for k, x in enumerate(messages) if x["role"] == "assistant"]
    trainable = []
    for t in continuation["turns"]:
        reasoning, visible = split_reasoning(t["text"])
        trainable.append(len(messages))
        calls = [_call_message(c) for c in t["calls"]]
        messages.append({"role": "assistant", "content": _strip_calls(visible), "reasoning_content": reasoning,
                         **({"tool_calls": calls} if calls else {})})
        for c in t["calls"]:
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": c["result"], "name": c["name"]})
    task = record["traces"][0]["task"]["data"]["task_name"]
    return {"id": f"sdpo-a3/{arm}/{item_id}", "messages": messages, "tools": chain["tools"],
            "trainable_message_indices": trainable,
            "metadata": {"task": task, "episode_id": record["id"], "source": A3_SOURCE, "arm": arm,
                         "point_turn": turn, "stop": continuation["stop"], "masked_turns": prefix_turns,
                         "masked_reasons": {str(k): A3_PREFIX_REASON for k in prefix_turns}}}


_TOOL_CALL = re.compile(r"<tool_call>.*?</tool_call>", re.S)


def _strip_calls(visible):
    return _TOOL_CALL.sub("", visible).strip()


# Words by which a continuation shows it read the feedback (the student never sees it). Each pattern is tested on
# feedback-reading sentences and on ordinary task sentences (test_sdpo_a3.py), both ways.
FEEDBACK_PATTERNS = [
    r"\bfeedback\b", r"\bthe notes?\b(?! (field|column|section|text|body|parameter|property))", r"\bhints?\b",
    r"\bgrader\b", r"\bexpected value", r"\bstill not done\b", r"\bnot done at this point\b",
    r"\b(reference|verified|sibling) solution\b", r"\bcorrectly solve the original question\b",
    r"\bthis task forbids\b", r"\bnote on (your|my) next action\b", r"\bthe (check|assertion)s? (says?|failed|wants?)\b",
    r"\b(previous|earlier|prior|last|first) (unsuccessful |failed )?(attempt|try|run)\b",
]
_FEEDBACK = re.compile("|".join(FEEDBACK_PATTERNS), re.I)
_VALUE = re.compile(r"[A-Za-z0-9_.@:$%,/-]*\d[A-Za-z0-9_.@:$%,/-]*|[\w.+-]+@[\w-]+\.[\w.-]+")


def _values(text):
    return {v.strip(".,:;") for v in _VALUE.findall(text or "") if len(v.strip(".,:;")) >= 3}


_EXPRESSIONS = None


def _sft_expressions():
    """The SFT gates' own arithmetic reader (wt-sft rowlib.expressions), loaded without leaving anything behind:
    rowlib puts wt-sft's qualification folder on sys.path, and a later `import sdpo_ab_engine` (or any sibling) would
    then load wt-sft's older copy. sys.path and every module other than this one function are restored."""
    global _EXPRESSIONS
    if _EXPRESSIONS is None:
        import sys
        from pathlib import Path

        sftgen = Path(__file__).resolve().parents[3] / "wt-sft/scripts/qualification/sftgen"
        saved_path, before = list(sys.path), set(sys.modules)
        try:
            sys.path.insert(0, str(sftgen))
            import rowlib

            _EXPRESSIONS = rowlib.expressions
        finally:
            sys.path[:] = saved_path
            for name in set(sys.modules) - before:
                del sys.modules[name]
    return _EXPRESSIONS


def computed_values(text):
    """Numbers that correct arithmetic written in `text` produces (validate_row's expressions; each written result and
    the value of each correct chain). A privileged value equal (to its written precision) to one of them was worked
    out, not copied."""
    from decimal import Decimal

    out = set()
    for e in _sft_expressions()(text or ""):
        if e["ok"]:
            out |= {abs(Decimal(str(v))) for v in e["values"]}
    return out


def _worked_out(value, computed):
    from decimal import Decimal, InvalidOperation

    try:
        v = Decimal(value.replace(",", "").rstrip("%").lstrip("$"))
    except InvalidOperation:
        return False
    step = Decimal(1).scaleb(v.as_tuple().exponent) / 2 if v.as_tuple().exponent < 0 else Decimal("0.5")
    return any(abs(abs(v) - c) <= step for c in computed)


def _number(text):
    from decimal import Decimal, InvalidOperation

    try:
        return Decimal(text.replace(",", "").replace("$", "").rstrip(".…").strip())
    except InvalidOperation:
        return None


def _evaluate(expression):
    """The value of a plain arithmetic expression (numbers, + - * / × ÷, parentheses; '$', ',', '...' ignored), or
    None. Evaluated by walking the parsed tree, never by eval()."""
    import ast
    import operator
    from decimal import Decimal

    text = (expression.replace("$", "").replace(",", "").replace("…", "").replace("...", "")
            .replace("×", "*").replace("÷", "/"))
    if not re.fullmatch(r"[\d.+\-*/() ]+", text.strip() or "x"):
        return None
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Decimal(str(node.value))
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        raise ValueError("not arithmetic")

    try:
        return walk(ast.parse(text.strip(), mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, ArithmeticError):
        return None


def derived_on_a_line(value, text):
    """True when `value` directly follows '=' or '≈' on a line of `text` and the step checks out: after '≈' the number
    just before it rounds to `value` ("15.333... ≈ 15.3"); after '=' the expression since the previous '=' or '≈' (or
    the line start, after a label) evaluates to `value` to its written precision ("(3,000 / 18,000) * 92 = 15.33").
    A bare statement, or a value after a wrong step ("= 17.0 so 15.3", "2 * 3 = 15.3"), is not derived."""
    target = _number(value)
    if target is None:
        return False
    for line in (text or "").splitlines():
        at = line.find(value)
        while at >= 0:
            before = line[:at].rstrip()
            if before.endswith("≈"):
                prev = re.search(r"([\d$][\d,.…$]*)\s*≈$", before)
                n = _number(prev.group(1)) if prev else None
                if n is not None and _worked_out(value, {abs(n)}):
                    return True
            elif before.endswith("="):
                lhs = re.split(r"[=≈:]", before[:-1])[-1]
                n = _evaluate(lhs)
                if n is not None and _worked_out(value, {abs(n)}):
                    return True
            at = line.find(value, at + 1)
    return False


def leak_findings(row, context):
    """Feedback references and privileged values in the trainable turns: (a) a FEEDBACK_PATTERNS phrase in reasoning or
    visible text; (b) a value (number, id, email; >= 3 characters) that appears in the teacher's feedback or solution
    but nowhere in the student-visible context before that turn (system, user, earlier turns and tool results)."""
    privileged = _values(context.get("first_message")) | set().union(
        *[_values(v) for v in (context.get("notes") or {}).values()]) | _values(
        json.dumps(context.get("final_note") or {}))
    findings = []
    seen_text = ""
    for i, m in enumerate(row["messages"]):
        body = " ".join(str(m.get(k) or "") for k in ("content", "reasoning_content"))
        if m.get("tool_calls"):
            body += " " + json.dumps([c["function"]["arguments"] for c in m["tool_calls"]], ensure_ascii=False)
        if i in row["trainable_message_indices"]:
            for hit in _FEEDBACK.finditer(" ".join(str(m.get(k) or "") for k in ("content", "reasoning_content"))):
                findings.append({"message": i, "kind": "feedback_phrase", "text": hit.group(0)})
            worked = computed_values(m.get("reasoning_content") or "") | computed_values(m.get("content") or "")
            for value in sorted(_values(body) & privileged):
                reasoning = m.get("reasoning_content") or ""
                if value not in seen_text and not _worked_out(value, worked) \
                        and not derived_on_a_line(value, reasoning):
                    findings.append({"message": i, "kind": "privileged_value", "text": value})
        seen_text += "\n" + body
    return findings


# Leak check v2 (15 Nov): the v1 phrase list fired on ordinary task words ("the note(s)" of a Notes column, customer
# "feedback", "hints"): 149 of 554 no-feedback continuations, which cannot leak, were flagged. v2 keeps v1's privileged
# values and replaces the phrases by (a) wording only the feedback uses or a reference to it, and (b) any run of
# FEEDBACK_SHARED_WORDS words shared with the point's feedback text that the student-visible input does not hold.
# Calibration on the same runs: no-feedback 8 / 554 flagged; feedback rows 811 / 831 (v1: 812).
FEEDBACK_ONLY = re.compile(
    r"\bthis task forbids\b|\bnote on (your|my) next action\b|\bstill not done\b|\bnot done at this point\b|"
    r"\bcorrectly solve the original question\b|\b(previous|earlier|prior) (unsuccessful |failed )?attempts?\b|"
    r"\bexpected value|\bgrader\b|\b(reference|verified|sibling) solution\b|\bfrom your unsuccessful\b|"
    r"\bnever observed\b|\bthe (check|assertion)s? (says?|failed|wants?)\b|"
    r"\b(the|earlier|previous|this) feedback (says?|said|mentions?|mentioned|notes?|noted|indicates?|states?|shows?|tells?|warns?|points?)\b|"
    r"\b(per|based on|according to|from|following) (the )?(earlier |previous )?feedback\b|\b(earlier|previous|prior) feedback\b|"
    r"\bthere was (a |some )?feedback\b", re.I)
FEEDBACK_SHARED_WORDS = 6
_WORD = re.compile(r"[a-z0-9@._'-]+")


def _word_runs(text, n=FEEDBACK_SHARED_WORDS):
    words = _WORD.findall((text or "").lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def feedback_text(context):
    """The feedback the teacher read: the first message from the feedback heading on, every note, the final note."""
    first = context.get("first_message") or ""
    # the old heading, P/N's heading and the calm heading (V4); without one, the whole first message (17 Nov: the calm
    # heading was missing, so a V4 context's task text would have counted as feedback)
    start = max(first.find(h) for h in ("The following is feedback", "Results of an earlier attempt", "Notes on an earlier attempt"))
    parts = [first[start:] if start >= 0 else first] + [str(v) for v in (context.get("notes") or {}).values()]
    final = context.get("final_note")
    parts.append((final.get("line") or "") if isinstance(final, dict) else str(final or ""))  # a note may have no line
    return "\n".join(parts)


def leak_findings_v2(row, context):
    """v1's privileged values, plus "feedback_only_phrase" (FEEDBACK_ONLY in reasoning or visible text) and
    "feedback_words" (a run of FEEDBACK_SHARED_WORDS words shared with the feedback text, absent from every message
    the student saw, i.e. every non-trainable message)."""
    findings = [f for f in leak_findings(row, context) if f["kind"] != "feedback_phrase"]
    trainable = set(row["trainable_message_indices"])
    visible = " ".join(str(m.get("content") or "") for i, m in enumerate(row["messages"]) if i not in trainable)
    only_feedback = _word_runs(feedback_text(context)) - _word_runs(visible)
    for i in sorted(trainable):
        m = row["messages"][i]
        text = " ".join(str(m.get(k) or "") for k in ("reasoning_content", "content"))
        findings += [{"message": i, "kind": "feedback_only_phrase", "text": hit.group(0)} for hit in FEEDBACK_ONLY.finditer(text)]
        findings += [{"message": i, "kind": "feedback_words", "text": " ".join(run)} for run in sorted(_word_runs(text) & only_feedback)[:3]]
    return findings


# ------------------------------------------------------------------ GPU: the runner phase `--phase a3-continue`

def action_position(chain, turn):
    """Index in chain["segments"] of the student's `turn`-th action (1-based)."""
    actions = [i for i, s in enumerate(chain["segments"]) if s["kind"] == "action"]
    return actions[turn - 1]


async def phase_a3(rt, args, runner):
    """For each item of `--a3-items` (item_id, episode_id, arm, turn, digest): rebuild the attempt's teacher context
    with the engine and refuse it unless its digest equals the one audited on the CPU; rebuild the world just before
    the point; let the start model (the `--adapter`, i.e. P9's SFT model) read the teacher prompt and continue to the
    end, card-thinking sampling, every call executed; grade the final world. Writes a3-continuations.jsonl.gz and
    a3-summary.json. Nothing is trained. `runner` is the running runner module itself (`sys.modules["__main__"]` when
    the runner is a script): a fresh `import sdpo_ab_runner` would be a second copy whose THINKING is still off, and
    every prompt it renders would silently be thinking-off."""
    import asyncio
    import concurrent.futures
    import datetime as dt
    import gzip
    import time
    from collections import Counter, defaultdict
    from pathlib import Path

    from peft import load_peft_weights, set_peft_model_state_dict

    import sdpo_ab_world as world
    from sdpo_ab_execution import world_date

    if not runner.THINKING:
        raise ValueError("--phase a3-continue needs --thinking on")
    if not args.adapter:
        raise ValueError("--phase a3-continue needs --adapter (the start model)")
    set_peft_model_state_dict(rt.model, load_peft_weights(args.adapter), adapter_name="default")
    rt.model.set_adapter("default")
    await rt.rollout.suspend_for_update()
    await rt.rollout.refresh()
    after = await rt.rollout.score_probe(rt.model, *rt.probe)
    check = runner.adapter_effect(rt.startup_probe, after)
    (rt.out / "adapter-check.json").write_text(json.dumps({"adapter": args.adapter, **check}, indent=1))
    runner.log(stage="adapter-check", **check)

    items = [json.loads(line) for line in Path(args.a3_items).read_text().splitlines() if line.strip()]
    run = Path(args.a3_episodes)
    tasks = {i["task"] for i in items}  # the items' tasks only (their siblings feed T-A1's solution)
    records, chains = {}, {}
    with gzip.open(run / "episodes.jsonl.gz", "rt") as stream:
        for line in stream:
            r = json.loads(line)
            if world.task_name(r) in tasks:
                records[r["id"]] = r
    with gzip.open(run / "chains.jsonl.gz", "rt") as stream:
        for line in stream:
            c = json.loads(line)
            if c["episode_id"] in records:
                chains[c["episode_id"]] = c
    by_task = defaultdict(list)
    for record in records.values():
        by_task[world.task_name(record)].append(record)
    engines = {arm: make_engine(arm) for arm in sorted({i["arm"] for i in items})}
    sampling = runner.collect_sampling("card-thinking")
    rt.rollout.temperature, rt.rollout.top_p = sampling["temperature"], sampling["top_p"]
    rt.rollout.extra_sampling = {k: v for k, v in sampling.items() if k not in ("temperature", "top_p")}
    turns_limit, context_limit, reply_limit, _ = runner.episode_limits(args)
    limits = {"max_turns": turns_limit, "context": context_limit, "reply": reply_limit}
    day = dt.date.fromisoformat(world_date())
    loop = asyncio.get_running_loop()
    worker = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="a3-world")
    gate = asyncio.Semaphore(max(1, args.concurrency))
    counts, out_rows = Counter(), []
    started = time.monotonic()

    async def in_world(function, *arguments):
        return await loop.run_in_executor(worker, function, *arguments)

    async def one(item):
        """One item; any unexpected error is recorded as that item's drop (with its traceback) and never stops the
        run (a world rebuild or generation failure on one attempt must not cost the other items)."""
        import traceback

        try:
            await one_item(item)
        except Exception as error:  # noqa: BLE001
            counts["drop:error"] += 1
            out_rows.append({**{k: item[k] for k in ("item_id", "episode_id", "arm", "turn")},
                             "dropped": f"error: {type(error).__name__}: {error}",
                             "traceback": traceback.format_exc()[-3000:]})

    async def one_item(item):
        base = {k: item[k] for k in ("item_id", "episode_id", "arm", "turn")}
        record, chain = records.get(item["episode_id"]), chains.get(item["episode_id"])
        if record is None or chain is None or chain.get("mismatch"):
            counts["drop:missing_or_mismatched_chain"] += 1
            out_rows.append({**base, "dropped": "missing_or_mismatched_chain"})
            return
        task = world.task_name(record)
        try:
            context = await in_world(engines[item["arm"]], record, by_task[task], ARMS[item["arm"]]["arm"], "both")
        except Exception as error:  # noqa: BLE001 - an engine refusal or crash excludes the item, never stops the run
            counts["drop:engine"] += 1
            out_rows.append({**base, "dropped": f"engine: {type(error).__name__}: {error}"})
            return
        if context is None or context_digest(context) != item["digest"]:
            counts["drop:digest_mismatch"] += 1
            out_rows.append({**base, "dropped": "digest_mismatch"})
            return
        position = action_position(chain, item["turn"])
        if ARMS[item["arm"]].get("no_feedback"):
            prefix = student_prefix(chain, position)
        else:
            prefix = runner.engine_teacher_prefix(rt.tokenizer, chain, position, teacher_context(item["arm"], context))
        k = calls_before(record, item["turn"])
        built, report, snapshots = await in_world(world.rebuild_with_snapshots, record, day, {k})
        if report["variant"] == "unmatched" or not report["verified"]:
            counts["drop:replay_mismatch"] += 1
            out_rows.append({**base, "dropped": "replay_mismatch", "replay": report["replayed"]})
            return
        start = await in_world(world.fork, built, snapshots[k])
        native, parser = runner.tool_call_parser(rt.tokenizer, chain["messages0"], chain["tools"])

        def parse(text):
            return runner.parsed_calls(parser.extract_tool_calls(text, native))

        async def generate(tokens, budget, index):
            async with gate:
                ids, _logps, finish = await rt.rollout.sample(
                    list(tokens), budget, seed=runner.request_seed(f"{args.seed_salt}|a3|{item['item_id']}", index))
            return ids, finish

        async def run_one(name, arguments):
            ((content, status),) = await in_world(world.execute, start, [(name, arguments)])
            return content, status

        actions = [s for s in chain["segments"][:position] if s["kind"] == "action"]
        result = await continue_episode(
            prefix, tokenizer=rt.tokenizer, generate=generate, parse=parse, run_one=run_one,
            render=lambda messages: runner.render_delta(rt.tokenizer, messages), limits=limits,
            turns_before=item["turn"] - 1, output_before=sum(len(a["ids"]) for a in actions),
            visible=lambda raw: runner.visible_reply(raw)[1])
        final = await in_world(world.grade, start)
        success = final["task_completed_correctly"] == 1.0 and all(a.get("passed") for a in final["assertion_results"])
        counts["continued"] += 1
        counts["all_checks_pass"] += success
        out_rows.append({**base, "task": task, "digest": item["digest"], "prefix_tokens": len(prefix),
                         "stop": result["stop"], "tokens": result["tokens"], "turns": result["turns"],
                         "task_completed_correctly": final["task_completed_correctly"],
                         "partial_credit": final["partial_credit"], "assertion_results": final["assertion_results"],
                         "all_checks_pass": success})
        runner.log(stage="a3", item=item["item_id"], arm=item["arm"], stop=result["stop"], success=success,
                   turns=len(result["turns"]))

    await asyncio.gather(*(one(item) for item in items))
    out_rows.sort(key=lambda r: r["item_id"])
    with gzip.open(rt.out / "a3-continuations.jsonl.gz", "wt") as stream:
        for row in out_rows:
            stream.write(json.dumps(row) + "\n")
    runner.write(rt.out / "a3-summary.json", {"items": len(items), "counts": dict(counts), "limits": limits,
                                              "sampling": sampling, "elapsed_s": time.monotonic() - started})


# ------------------------------------------------------------------ A4: where the teacher and the student differ

def token_classes(targets, turns, think_close, opener, closer, newline_ids=()):
    """Per target token of the student's actions: "reasoning" (before `</think>`), "decision" (the first token after
    `</think>` that is not a newline: call opener or text), "call" (inside <tool_call>...</tool_call>, both tags
    included), "text" (other visible tokens). A turn without `</think>` is all "reasoning" (cut inside the thinking)."""
    classes, turn, state = [], None, None
    for token, t in zip(targets, turns):
        if t != turn:
            turn, state = t, "reasoning"
        if state == "reasoning":
            classes.append("reasoning")
            if token == think_close:
                state = "await_decision"
            continue
        if state == "await_decision":
            if token in newline_ids:
                classes.append("text")
                continue
            classes.append("decision")
            state = "call" if token == opener else "text"
            continue
        if token == opener:
            state = "call"
        classes.append("call" if state == "call" else "text")
        if state == "call" and token == closer:
            state = "text"
    return classes


async def phase_a4(rt, args, runner):
    """For the first `--a4-limit` items of `--a3-items`: on the student's own tokens from the item's turn to the end,
    the teacher's log-probability (engine context, notes) minus the student's (its own prompt), per token class
    (token_classes). Forward passes only; the start model (`--adapter`) plays both roles. Writes a4-signal.json."""
    import gzip
    import statistics
    from collections import defaultdict
    from pathlib import Path

    from peft import load_peft_weights, set_peft_model_state_dict

    import sdpo_ab_world as world

    if not runner.THINKING or not args.adapter:
        raise ValueError("--phase a4-signal needs --thinking on and --adapter")
    set_peft_model_state_dict(rt.model, load_peft_weights(args.adapter), adapter_name="default")
    rt.model.set_adapter("default")
    items = [json.loads(line) for line in Path(args.a3_items).read_text().splitlines() if line.strip()]
    items = items[:args.a4_limit]
    tasks = {i["task"] for i in items}
    records, chains = {}, {}
    run = Path(args.a3_episodes)
    with gzip.open(run / "episodes.jsonl.gz", "rt") as stream:
        for line in stream:
            r = json.loads(line)
            if world.task_name(r) in tasks:
                records[r["id"]] = r
    with gzip.open(run / "chains.jsonl.gz", "rt") as stream:
        for line in stream:
            c = json.loads(line)
            if c["episode_id"] in records:
                chains[c["episode_id"]] = c
    by_task = defaultdict(list)
    for r in records.values():
        by_task[world.task_name(r)].append(r)
    engines = {arm: make_engine(arm) for arm in sorted({i["arm"] for i in items})}
    tok = rt.tokenizer
    think_close = tok.convert_tokens_to_ids("</think>")
    opener, closer = tok.convert_tokens_to_ids("<tool_call>"), tok.convert_tokens_to_ids("</tool_call>")
    newlines = set(tok.encode("\n", add_special_tokens=False) + tok.encode("\n\n", add_special_tokens=False))
    diffs = defaultdict(list)
    done, skipped = 0, defaultdict(int)
    for item in items:
        record, chain = records.get(item["episode_id"]), chains.get(item["episode_id"])
        try:
            context = engines[item["arm"]](record, by_task[item["task"]], ARMS[item["arm"]]["arm"], "both")
        except Exception:  # noqa: BLE001
            skipped["engine"] += 1
            continue
        try:
            measured = _a4_item(item, chain, context, runner, tok, think_close, opener, closer, newlines, rt)
        except Exception as error:  # noqa: BLE001 - one item's failure never stops the phase
            skipped[f"error:{type(error).__name__}"] += 1
            continue
        if isinstance(measured, str):
            skipped[measured] += 1
            continue
        for key, values in measured.items():
            diffs[key] += values
        done += 1
    summary = {}
    for (arm, c), values in sorted(diffs.items()):
        summary[f"{arm}:{c}"] = {"tokens": len(values), "mean_teacher_minus_student": statistics.fmean(values),
                                 "share_teacher_higher": sum(v > 0 for v in values) / len(values),
                                 "sum_abs": sum(abs(v) for v in values)}
    runner.write(rt.out / "a4-signal.json", {"items": len(items), "measured": done, "skipped": dict(skipped),
                                              "classes": summary})


def _a4_item(item, chain, context, runner, tok, think_close, opener, closer, newlines, rt):
    """One A4 item: {(arm, class): [teacher minus student log-prob]} or a skip reason."""
    from collections import defaultdict

    if context is None or context_digest(context) != item["digest"]:
        return "digest"
    student = runner.student_sequence(chain)
    teacher = runner.teacher_sequence(tok, chain, teacher_context(item["arm"], context))
    if len(teacher["targets"]) != len(student["targets"]) or teacher["targets"] != student["targets"]:
        return "target_mismatch"
    keep = [k for k, t in enumerate(student["turns"]) if t >= item["turn"]]
    if not keep:
        return "empty"
    ls = runner.token_logprobs(rt.model, student["ids"], [student["positions"][k] for k in keep],
                               [student["targets"][k] for k in keep])
    lt = runner.token_logprobs(rt.model, teacher["ids"], [teacher["positions"][k] for k in keep],
                               [teacher["targets"][k] for k in keep])
    classes = token_classes([student["targets"][k] for k in keep], [student["turns"][k] for k in keep],
                            think_close, opener, closer, newlines)
    out = defaultdict(list)
    for c, a, b in zip(classes, lt, ls):
        out[(item["arm"], c)].append(a - b)
    return out


# ------------------------------------------------------------------ signal check (plan of 16 Nov, Step 2)

def action_tokens(ids, think_close):
    """Indices of a turn's action tokens: everything after its first `</think>` (the call blocks or the visible reply,
    and the end-of-turn token), or None when the turn never closed its thinking."""
    try:
        start = ids.index(think_close) + 1
    except ValueError:
        return None
    return list(range(start, len(ids))) or None


def signal_value(teacher_correct, student_correct, teacher_wrong, student_wrong):
    """How much more the teacher than the student prefers the correct turn's action over the student's own (wrong)
    action: (teacher - student log-likelihood of the correct action) - (teacher - student log-likelihood of the wrong
    action). Positive = the teacher pushes towards the correct action; 0 = no signal (e.g. no feedback)."""
    return (teacher_correct - student_correct) - (teacher_wrong - student_wrong)


def signal_item(item, chain, context, runner, tok, think_close, rt):
    """One signal item: the dict of sums, or a skip reason. The student reads its own prefix; the teacher reads the
    arm's feedback context (`engine_teacher_prefix`, notes of turns up to the point). Both score the student's own
    action at the point (wrong) and the first turn of a verified correct continuation from the same point (correct),
    on action tokens only (the correct turn's reasoning was written by a teacher that may have read feedback)."""
    if context is None or context_digest(context) != item["digest"]:
        return "digest"
    position = action_position(chain, item["turn"])
    wrong, correct = list(chain["segments"][position]["ids"]), list(item["correct_ids"])
    if wrong == correct:
        return "same_turn"
    sw, sc = action_tokens(wrong, think_close), action_tokens(correct, think_close)
    if sw is None or sc is None:
        return "unclosed_thinking"
    student_prefix_ids = student_prefix(chain, position)
    teacher_prefix_ids = runner.engine_teacher_prefix(tok, chain, position, teacher_context(item["arm"], context))
    sums = {}
    for name, turn, keep in (("wrong", wrong, sw), ("correct", correct, sc)):
        for who, prefix in (("student", student_prefix_ids), ("teacher", teacher_prefix_ids)):
            ids = prefix + turn
            values = runner.token_logprobs(rt.model, ids, [len(prefix) + k - 1 for k in keep], [turn[k] for k in keep])
            sums[f"{who}_{name}"] = sum(values)
        sums[f"tokens_{name}"] = len(keep)
    sums["signal"] = signal_value(sums["teacher_correct"], sums["student_correct"], sums["teacher_wrong"],
                                  sums["student_wrong"])
    return sums


async def phase_signal(rt, args, runner):
    """`--phase signal-check`: for each item of `--a3-items` (a point, an arm = the feedback version, the point's
    verified correct first turn `correct_ids`), forward passes only with the start model (`--adapter`) in both roles.
    Writes signal.jsonl (one row per item) and signal-summary.json. Nothing trains."""
    import gzip
    import statistics
    from collections import Counter, defaultdict
    from pathlib import Path

    from peft import load_peft_weights, set_peft_model_state_dict

    import sdpo_ab_world as world

    if not runner.THINKING or not args.adapter:
        raise ValueError("--phase signal-check needs --thinking on and --adapter")
    set_peft_model_state_dict(rt.model, load_peft_weights(args.adapter), adapter_name="default")
    rt.model.set_adapter("default")
    items = [json.loads(line) for line in Path(args.a3_items).read_text().splitlines() if line.strip()]
    tasks = {i["task"] for i in items}
    records, chains = {}, {}
    run = Path(args.a3_episodes)
    with gzip.open(run / "episodes.jsonl.gz", "rt") as stream:
        for line in stream:
            r = json.loads(line)
            if world.task_name(r) in tasks:
                records[r["id"]] = r
    with gzip.open(run / "chains.jsonl.gz", "rt") as stream:
        for line in stream:
            c = json.loads(line)
            if c["episode_id"] in records:
                chains[c["episode_id"]] = c
    by_task = defaultdict(list)
    for r in records.values():
        by_task[world.task_name(r)].append(r)
    engines = {arm: make_engine(arm) for arm in sorted({i["arm"] for i in items})}
    tok = rt.tokenizer
    think_close = tok.convert_tokens_to_ids("</think>")
    rows, counts = [], Counter()
    for item in items:
        base = {k: item.get(k) for k in ("item_id", "episode_id", "task", "arm", "turn", "a1_class", "correct_source")}
        record, chain = records.get(item["episode_id"]), chains.get(item["episode_id"])
        if record is None or chain is None or chain.get("mismatch"):
            counts["skip:chain"] += 1
            rows.append({**base, "skipped": "chain"})
            continue
        try:
            context = engines[item["arm"]](record, by_task[item["task"]], ARMS[item["arm"]]["arm"], "both")
            measured = signal_item(item, chain, context, runner, tok, think_close, rt)
        except Exception as error:  # noqa: BLE001 - one item's failure never stops the phase
            measured = f"error:{type(error).__name__}"
        if isinstance(measured, str):
            counts[f"skip:{measured}"] += 1
            rows.append({**base, "skipped": measured})
            continue
        counts["measured"] += 1
        rows.append({**base, **measured})
    with open(rt.out / "signal.jsonl", "w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    by_arm = defaultdict(list)
    for row in rows:
        if "signal" in row:
            by_arm[row["arm"]].append(row["signal"])
    runner.write(rt.out / "signal-summary.json", {
        "items": len(items), "counts": dict(counts),
        "arms": {arm: {"measured": len(v), "mean_signal": statistics.fmean(v), "share_positive": sum(x > 0 for x in v) / len(v)}
                 for arm, v in sorted(by_arm.items())}})
