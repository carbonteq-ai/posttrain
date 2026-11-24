"""SDPO v6 (19 Nov 2026): on-policy SDPO from P9. Ali: the teacher scores fresh student rollouts; sufficient steps; fix
what was wrong. Plan and gates: research/sdpo-work/weekend/sdpo-v6/PLAN.md (written before any v6 GPU job).

Phases (runner `--phase`):
  design-audit  P1, forward passes and reflections only, nothing trains. On recorded attempts (`--a3-episodes`, the fit
                tasks): per teacher design, M1-M4 of PLAN.md, and the pre-registered pick (`pick_design`).
  online-train  P2/P3. `--online-rounds` rounds. Each round:
                1. fresh rollouts of the current student on the `--fit` tasks (`--rollouts-per-task`, `--sampling`);
                2. teacher contexts in `--online-design`;
                3. training on failures (turns up to their first located wrong decision) and on successes (all turns),
                   reasoning+calls tokens, `--classic-batch` attempts per optimizer step;
                4. the EMA teacher (`--online-ema`) after every step;
                5. a checkpoint per round and the PLAN.md monitors.

Teacher designs (`DESIGNS`); every design gives a success the same outcome line:
  V5    the full3 feedback, built live from the engine's T-A2 context (`v5_live`, identical to the shipped rewrites).
  OUT   outcome only: "it failed; N of the task's M required checks were not met".
  HINT  the policy's own per-step hindsight hints (HINT, arXiv 2606.11559). The policy writes them from the task, its
        attempt, the outcome and the engine's findings. Each hint sits just before the step it is about, and is
        stripped of values the student never saw and of references to later steps.
"""
from __future__ import annotations

import json
import random
import re

DESIGNS = ("V5", "OUT", "HINT", "W1", "W2", "W3", "W4", "Z3C", "Z4C", "Z4CP", "Y1", "Y2", "Y1P", "Y2P", "Y1PQ", "Y2PQ",
           "S1", "S2", "S1P", "S2P", "S1PQ", "S2PQ", "S1Q", "S2Q", "S1PR", "S1R", "S2PR", "S2R",
           "S1PU", "S1PRU", "S1PUH", "S1FH", "S1FTH", "S1PTH", "S1FUH", "S1FTUH", "S1PTUH",
           "S1FTUHA", "S1FTUHM", "S1FUHA", "S1FUHM", "S1PTUHA", "S1PTUHM",
           "S1FUHW", "S1FUHAW", "S1FUHWN", "S1FUHWNL", "S1FUHWNLK", "S1FUHWNLE", "S1FUHWNLD", "S1FUHWNLJ", "S1FUHWNLEJ", "S1FUHWNLDJ", "S1FUHWNLG", "S1FUHWNLGO", "S1FUHWNLGJ", "S1FUHWNLGOJ", "S1FUHWNLO", "S1FUHWNLOJ", "S1FUHWNLGOJY", "S1FUHWNLOJY", "S1FUHWNLGOJYIV", "S1FUHWNLGOJYIVC")  # W1-W4: r3 rewrites of V5 (feedback_versions,
# PLAN.md "r3"); Z3C/Z4C: SDPO closure round 3 (feedback_c.zc_live; identical to the measured rewrites v10 on all
# 1,087 recorded W6 contexts, weekend/sdpo-close/tools/check_zc_live.py); Y1/Y2/Y1P/Y2P: round 4 (D1-10, feedback_c.yc_live:
# R-c2 on Z4, true pointer notes, P framing; identical to rewrites v13, check_zc_live.py with DESIGNS=Y1,Y2,Y1P,Y2P);
# S1/S2 (+P, PQ): round 5 (feedback_c.sc_live: R-d names every value with where the attempt read it, R-n says the facts
# about the cut turn once, in a note at that turn; identical to rewrites v14, check_zc_live.py with DESIGNS=S1,...);
# S1PU/S1PRU: round 6 (the S1P/S1PR texts, the note at the cut turn as its own user message: note_role "user");
# S1PUH: round 7 candidate (S1PU with a value the attempt has not seen said as one "no tool has shown");
# S1FH ... S1PTUH: round 7 (D1-15; feedback_c.xc_live: F = V5's framing with "results" for "feedback", T = the cut note
# first says the earlier attempt went wrong at this step, U, H)
HEAD = "The following is feedback from your earlier attempt at this task:"
TAIL = "Correctly solve the original question."
SUCCESS_LINE = "it succeeded; every check of the task passed."
HINT_PREFIX = "Hint for your next step: "
REFLECT_MAX_TOKENS = 2048
REFLECT_STEP_CHARS = 700  # per step, tool results are cut to this many characters in the reflection prompt
REFLECT_CALL_CHARS = 400
HINT_MAX_CHARS = 400
THINK_CLOSE = "</think>"
REASONING_CAP = 2.0  # round monitor: mean reasoning chars per attempt against round 1 (20 Nov, was 1.3)
INFRA_LIMIT = 0.05  # 20 Nov: a round with more infrastructure failures than this measures the engine, not the model
TRAINER_RESERVED_LIMIT_GIB = 16  # 20 Nov: round-1 collection ran with the trainer holding ~9.7 GiB


# ------------------------------------------------------------------ attempts as text (reflection input)

def decode_steps(tokenizer, chain):
    """[(visible text, [call strings], tool results text)] per assistant turn of a joined chain."""
    steps, segments = [], chain["segments"]
    for index, segment in enumerate(segments):
        if segment["kind"] != "action":
            continue
        text = tokenizer.decode(segment["ids"], skip_special_tokens=False)
        rest = text.split(THINK_CLOSE, 1)[1] if THINK_CLOSE in text else text
        calls = []
        for name, body in re.findall(r"<function=([^>\n]+)>(.*?)</function>", rest, re.S):
            params = re.findall(r"<parameter=([^>\n]+)>(.*?)</parameter>", body, re.S)
            args = ", ".join(f"{k.strip()}={re.sub(r'\s+', ' ', v.strip())}" for k, v in params)
            calls.append(f"{name.strip()}({args})"[:REFLECT_CALL_CHARS])
        visible = re.sub(r"<tool_call>.*?</tool_call>", "", rest, flags=re.S)
        visible = re.sub(r"<\|[^|]+\|>", "", visible).strip()
        results = ""
        if index + 1 < len(segments) and segments[index + 1]["kind"] != "action":
            following = tokenizer.decode(segments[index + 1]["ids"], skip_special_tokens=False)
            results = " ".join(re.sub(r"\s+", " ", x.strip())
                               for x in re.findall(r"<tool_response>(.*?)</tool_response>", following, re.S))
        steps.append((visible, calls, results))
    return steps


def task_text(chain):
    return next(m["content"] for m in chain["messages0"] if m["role"] == "user")


def visible_text(tokenizer, chain):
    """Everything the student saw or wrote in the attempt: task, every assistant token, every tool result."""
    parts = [task_text(chain)]
    for segment in chain["segments"][1:]:
        parts.append(tokenizer.decode(segment["ids"], skip_special_tokens=False))
    return "\n".join(parts)


def reflection_messages(task, steps, success, findings):
    """The reflection request (HINT's reflector: the policy itself, from the task, the attempt and the outcome; here
    also the engine's findings, which state what failed with unseen values hidden)."""
    lines = []
    for number, (visible, calls, results) in enumerate(steps, start=1):
        did = "; ".join(calls) if calls else ("final reply: " + visible[:REFLECT_CALL_CHARS] if visible else "no action")
        lines.append(f"Step {number}: {did}")
        if results:
            lines.append(f"  tools returned: {results[:REFLECT_STEP_CHARS]}")
    outcome = ("The attempt succeeded: every check of the task passed." if success else
               "The attempt failed: some of the task's checks were not met.")
    body = [f"TASK:\n{task}", "ATTEMPT (each step: what the agent did, then what the tools returned):\n" + "\n".join(lines),
            f"OUTCOME: {outcome}"]
    if findings and not success:
        body.append("GRADER FINDINGS (what went wrong; values the agent never saw are hidden):\n" + findings)
    body.append(
        "Write one hint per step as a JSON object {\"step1\": \"...\", \"step2\": \"...\", ...} with every step.\n"
        "Rules:\n"
        "- A hint says whether that step's action was right and, if not, what the agent should have checked or done at "
        "that point and why, using the task text and the tool results it had already seen.\n"
        "- Use an empty string for a step with nothing useful to say.\n"
        "- Do not quote values the agent never saw. Do not describe what happened in later steps.\n"
        "- At most two short sentences per hint.\n"
        "Return only the JSON object.")
    return [{"role": "system", "content": "You review an AI agent's attempt at a workplace automation task and write "
                                          "short, specific hints that would help it act correctly at each step."},
            {"role": "user", "content": "\n\n".join(body)}]


def parse_reflection(text, steps):
    """{step: hint} for 1..steps from the reflector's reply; unparseable -> {}."""
    body = text.split(THINK_CLOSE, 1)[1] if THINK_CLOSE in text else text
    start, end = body.find("{"), body.rfind("}")
    hints = {}
    if start != -1 and end > start:
        try:
            raw = json.loads(body[start:end + 1])
        except json.JSONDecodeError:
            raw = {f"step{n}": v for n, v in re.findall(r'"step\s*(\d+)"\s*:\s*"((?:[^"\\]|\\.)*)"',
                                                        body[start:end + 1])}
        for key, value in (raw.items() if isinstance(raw, dict) else []):
            match = re.fullmatch(r"\s*step\s*(\d+)\s*", str(key), re.I)
            if not match:
                continue
            if isinstance(value, dict):
                value = value.get("hint") or value.get("diagnosis") or ""
            if isinstance(value, str) and value.strip() and 1 <= int(match.group(1)) <= steps:
                hints[int(match.group(1))] = value.strip()
    return hints


_VALUE = re.compile(r'"([^"\n]{3,})"|\'([^\'\n]{3,})\'|([\w.+-]+@[\w-]+\.[\w.-]+)|(\$?\d[\d,]*(?:\.\d+)?%?)')


def _norm(text):
    return re.sub(r"[\s,$%]", "", text.lower())


def clean_hint(hint, step, seen):
    """Drop sentences that quote a value the student never saw (a quoted string, an email, a number with two or more
    digits) or that point to a later step; cap the length. `seen` = _norm(visible_text(...))."""
    kept = []
    for sentence in re.split(r"(?<=[.!?])\s+", hint.strip()):
        later = any(int(n) > step for n in re.findall(r"\bsteps?\s+(\d+)", sentence, re.I))
        unseen = False
        for groups in _VALUE.findall(sentence):
            value = next(g for g in groups if g).rstrip(".,;:!?)")
            digits = re.sub(r"\D", "", value)
            if groups[3] and len(digits) < 2:
                continue
            if _norm(value) not in seen:
                unseen = True
                break
        if sentence and not later and not unseen:
            kept.append(sentence)
    return " ".join(kept)[:HINT_MAX_CHARS].strip()


_NEGATIVE = re.compile(r"\b(should(?:n'?t| not)?(?: have)?|instead|wrong|incorrect|mistake|missed|forgot|failed|"
                       r"did not|didn'?t|not (?:correct|right|allowed|needed|required)|avoid|do not|don'?t|never|"
                       r"without (?:checking|reading|verifying)|error)\b", re.I)
_POSITIVE = re.compile(r"\b(correct(?:ly)?|good|right|appropriate|proper(?:ly)?|as required|well)\b", re.I)
_ERROR_CLAIM = re.compile(r"\b(wrong|incorrect|mistake|error|failed)\b", re.I)
HINT_STATS = {"kept": 0, "endorses_located_mistake": 0, "error_claim_on_success": 0}  # written by the phases


def truth_filter(hints, success, last_turn):
    """Ali's engine rules (never state anything false): drop model-written hints that contradict the grader.
    - A failed attempt's first located wrong decision (last_turn): a hint that only endorses that step goes
      (the checks say it was wrong).
    - A successful attempt: a hint that claims a wrong step, a mistake or an error goes (every check passed).
    Hints elsewhere are kept: the grader cannot locate their truth (e.g. a missing read). Returns (kept, dropped
    counts)."""
    kept, dropped = {}, {"endorses_located_mistake": 0, "error_claim_on_success": 0}
    for turn, hint in hints.items():
        if success and _ERROR_CLAIM.search(hint):
            dropped["error_claim_on_success"] += 1
            continue
        if not success and last_turn is not None and turn == last_turn and _POSITIVE.search(hint) \
                and not _NEGATIVE.search(hint):
            dropped["endorses_located_mistake"] += 1
            continue
        kept[turn] = hint
    for key, value in dropped.items():
        HINT_STATS[key] += value
    HINT_STATS["kept"] += len(kept)
    return kept, dropped


# ------------------------------------------------------------------ teacher contexts

def outcome_context(task, success, outcome=None):
    if success:
        line = SUCCESS_LINE
    else:
        passed, total = (outcome or {}).get("required_passed"), (outcome or {}).get("required_total")
        if passed is not None and total:
            line = f"it failed; {int(total - passed)} of the task's {int(total)} required checks were not met."
        else:
            line = "it failed; some of the task's checks were not met."
    return {"first_message": f"{task}\n\n{HEAD} {line}\n\n{TAIL}", "notes": {}, "final_note": None}


def hint_context(task, success, hints, last_turn=None):
    """HINT design: the outcome line and, before each step's action, that step's hint (only steps <= last_turn)."""
    line = SUCCESS_LINE if success else "it failed; some of the task's checks were not met."
    notes = {t: HINT_PREFIX + h for t, h in sorted(hints.items()) if h and (last_turn is None or t <= last_turn)}
    return {"first_message": f"{task}\n\n{HEAD} {line} Hints for some steps appear in the conversation where they "
                             f"apply.\n\n{TAIL}", "notes": notes, "final_note": None}


def v5_live(context, shown_context):
    """The full3 V5 text built live, step for step as the shipped rewrites were made (weekend/engine-req):
    V1 = version_v1 (v2/v3 build), V4-1 = version_calm(V1) with V1's final note (v4), then version_unread and
    version_tidy on the first message and notes (v6), then version_frame (v7). check_v5_live.py proves it equals the
    shipped V5 on recorded attempts."""
    import feedback_versions as F

    v1, _counts = F.version_v1(context, shown_context)
    calm = F.version_calm(v1)
    v4 = {"first_message": calm["first_message"], "notes": calm["notes"], "final_note": v1.get("final_note")}
    tidy = F.version_tidy(F.version_unread({"first_message": v4["first_message"], "notes": v4["notes"] or {}}))
    v5 = F.version_frame({**v4, "first_message": tidy["first_message"], "notes": tidy["notes"]})
    return {**context, "first_message": v5["first_message"],
            "notes": {int(k): v for k, v in (v5.get("notes") or {}).items()}, "final_note": v5.get("final_note")}


def first_wrong_turn(engine_context):
    """The first located wrong decision: the earliest turn the engine's feedback speaks about (sdpo_a3.point_turn)."""
    from sdpo_a3 import point_turn

    return point_turn(engine_context) if engine_context else None


def truncate(sequences, last_turn):
    """Keep only the trained tokens of turns <= last_turn (student and teacher stay aligned index by index). The
    runner runs as __main__, so its `pick` is not imported here (that would load the runner module a second time)."""
    def pick(sequence, keep, keys):  # = sdpo_ab_runner.pick
        return {**sequence, **{k: [sequence[k][i] for i in keep] for k in keys}}

    if last_turn is None:
        return sequences
    keep = [i for i, t in enumerate(sequences["student"]["turns"]) if t <= last_turn]
    return {**sequences, "student": pick(sequences["student"], keep, ("positions", "targets", "old", "turns")),
            "teacher": pick(sequences["teacher"], keep, ("positions", "targets"))}


# r3 (Ali 20 Nov, "Both together"; weekend/sdpo-v6/PLAN.md "r3"): fix A. The values of free-text parameters inside tool
# calls are not trained: every SDPO run lost guard10 tasks to wrong text written inside calls. A parameter is free text
# when its value is prose (5+ words) in >= 80% of P9's own W6 calls (weekend/sdpo-v6/r3/call-keys.json), record-row
# fields (cells, row_data, data) excluded. Tool names, parameter names, the tags and every other value stay trained.
FREE_TEXT_KEYS = frozenset({"body", "text", "subject", "description", "Description", "summary", "content", "comment",
                            "prompt", "message", "body_text", "email_body", "email_subject"})
_PARAMETER = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def call_text_indices(tokenizer, targets, turns):
    """Indices (into `targets`) of the tokens inside the value of a free-text parameter of a tool call. A token counts
    when its text overlaps the value and holds no '<' or '>' (so the tags stay trained). Calls are found between
    `<tool_call>` and `</tool_call>` within one turn; each token is decoded on its own (the call markup is ASCII)."""
    opener = tokenizer.convert_tokens_to_ids("<tool_call>")
    closer = tokenizer.convert_tokens_to_ids("</tool_call>")
    out = set()

    def mark(region):
        pieces = [tokenizer.decode([targets[i]], skip_special_tokens=False) for i in region]
        starts, text = [], ""
        for piece in pieces:
            starts.append(len(text))
            text += piece
        for match in _PARAMETER.finditer(text):
            if match.group(1).strip() not in FREE_TEXT_KEYS:
                continue
            a, b = match.span(2)
            for index, start, piece in zip(region, starts, pieces):
                if start + len(piece) > a and start < b and "<" not in piece and ">" not in piece:
                    out.add(index)

    region, turn = None, None
    for index, (token, t) in enumerate(zip(targets, turns)):
        if t != turn:
            if region:
                mark(region)  # a call left open at the end of a turn
            region, turn = None, t
        if token == opener:
            if region:
                mark(region)
            region = [index]
            continue
        if region is not None:
            region.append(index)
            if token == closer:
                mark(region)
                region = None
    if region:
        mark(region)
    return out


def token_classes(tokenizer, targets, turns):
    """One class per token of an "all"-token sequence: reasoning (up to and including `</think>`), decision (the first
    token after it that is not a newline: `<tool_call>` or the reply's first token), call (tool-call markup and values
    that fix A trains), call_text (free-text values, not trained under fix A), text (other visible text, never trained)."""
    close = tokenizer.convert_tokens_to_ids("</think>")
    opener = tokenizer.convert_tokens_to_ids("<tool_call>")
    closer = tokenizer.convert_tokens_to_ids("</tool_call>")
    newlines = set(tokenizer.encode("\n", add_special_tokens=False) + tokenizer.encode("\n\n", add_special_tokens=False))
    free = call_text_indices(tokenizer, targets, turns)
    classes, turn, state = [], None, None
    for index, (token, t) in enumerate(zip(targets, turns)):
        if t != turn:
            turn, state = t, "reasoning"
        if state == "reasoning":
            classes.append("reasoning")
            if token == close:
                state = "await_decision"
            continue
        if state == "await_decision":
            if token in newlines:
                classes.append("text")
                continue
            classes.append("decision")
            state = "call" if token == opener else "text"
            continue
        if token == opener:
            state = "call"
        if state == "call":
            classes.append("call_text" if index in free else "call")
            if token == closer:
                state = "text"
            continue
        classes.append("text")
    return classes


def without_call_text(tokenizer, sequences):
    """Fix A: drop the free-text value tokens of tool calls from an already selected sequence (student and teacher stay
    aligned index by index)."""
    student = sequences["student"]
    drop = call_text_indices(tokenizer, student["targets"], student["turns"])
    if not drop:
        return sequences
    keep = [i for i in range(len(student["targets"])) if i not in drop]

    def pick(sequence, keys):
        return {**sequence, **{k: [sequence[k][i] for i in keep] for k in keys}}

    return {**sequences, "student": pick(student, ("positions", "targets", "old", "turns")),
            "teacher": pick(sequences["teacher"], ("positions", "targets"))}


def with_decision_tokens(tokenizer, sequences):
    """Ali 13 Nov ("SDPO must reach missed actions"): from an "all"-token sequence keep the reasoning and tool-call
    tokens (exactly reasoning+calls) plus, in every turn, the first token after `</think>` and its newlines: the
    decision to call a tool or to reply. That one token is where a stop-instead-of-act mistake is made; the rest of
    the visible text stays untrained (full3 trained it and its doubt grew)."""
    student = sequences["student"]
    close = tokenizer.convert_tokens_to_ids("</think>")
    opener = tokenizer.convert_tokens_to_ids("<tool_call>")
    closer = tokenizer.convert_tokens_to_ids("</tool_call>")
    newlines = set(tokenizer.encode("\n", add_special_tokens=False) + tokenizer.encode("\n\n", add_special_tokens=False))
    keep, turn, state = [], None, None
    for index, (token, t) in enumerate(zip(student["targets"], student["turns"])):
        if t != turn:
            turn, state = t, "reasoning"
        if state == "reasoning":
            keep.append(index)
            if token == close:
                state = "await_decision"
            continue
        if state == "await_decision":
            if token in newlines:
                continue
            keep.append(index)  # the decision token: <tool_call> or the reply's first token
            state = "call" if token == opener else "text"
            if state == "call":
                continue
            continue
        if token == opener:
            state = "call"
        if state == "call":
            keep.append(index)
            if token == closer:
                state = "text"

    def pick(sequence, keys):
        return {**sequence, **{k: [sequence[k][i] for i in keep] for k in keys}}

    return {**sequences, "student": pick(student, ("positions", "targets", "old", "turns")),
            "teacher": pick(sequences["teacher"], ("positions", "targets"))}


def note_kind(engine_context, turn):
    """The mistake type of the engine's note at the first located wrong decision (P1 per-type report)."""
    if not engine_context or turn is None:
        return "none"
    text = (engine_context.get("notes") or {}).get(turn) or ((engine_context.get("final_note") or {}).get("line") or "")
    if "forbids" in text:
        return "forbidden"
    if re.search(r"error|not found|failed|invalid", text, re.I):
        return "tool error"
    if re.search(r"still needed|still missing|still not done|exists, but none", text, re.I):
        return "missed or wrong content"
    return "other"


# ------------------------------------------------------------------ P1 metrics and the pre-registered pick

def post_think_indices(tokenizer, student, turn):
    """Indices (into the trained tokens) of `turn`'s tokens after its `</think>` (the visible decision: calls, text)."""
    close = tokenizer.convert_tokens_to_ids(THINK_CLOSE)
    out, seen = [], False
    for i, (token, t) in enumerate(zip(student["targets"], student["turns"])):
        if t != turn:
            continue
        if seen:
            out.append(i)
        if token == close:
            seen = True
    return out


def audit_row(student_lp, teacher_lp, decision, success):
    """One attempt's numbers. student_lp/teacher_lp: log-probs of the trained tokens; decision: indices of the first
    wrong action's visible tokens (failures)."""
    n = len(student_lp)
    if not n:
        return None
    gap = [t - s for s, t in zip(student_lp, teacher_lp)]
    if success:
        return {"success": True, "kl": -sum(gap) / n}
    inside = set(decision or [])
    m1 = sum(gap[i] for i in inside) / len(inside) if inside else None
    others = [g for i, g in enumerate(gap) if i not in inside]
    return {"success": False, "m1": m1, "drift": -sum(others) / len(others) if others else None, "decision_tokens": len(inside)}


def summarise(rows):
    """Per design: M1 (mean decision gap, failures), M2 (drift on the failures' other tokens), M3 (KL on successes),
    M4 (M1 on failures whose located check is a forbidden one)."""
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    out = {}
    for design in sorted({r["design"] for r in rows}):
        rs = [r for r in rows if r["design"] == design]
        fails = [r for r in rs if not r["success"]]
        out[design] = {"failures": len(fails), "successes": len(rs) - len(fails),
                       "M1": mean(r["m1"] for r in fails), "M2": mean(r["drift"] for r in fails),
                       "M3": mean(r["kl"] for r in rs if r["success"]),
                       "M4": mean(r["m1"] for r in fails if r.get("forbidden")),
                       "M4_points": sum(1 for r in fails if r.get("forbidden") and r["m1"] is not None),
                       "M1_negative_share": mean([1.0 if r["m1"] < 0 else 0.0 for r in fails if r["m1"] is not None]),
                       "M1_by_kind": {k: {"points": sum(1 for r in fails if r.get("kind") == k and r["m1"] is not None),
                                          "M1": mean(r["m1"] for r in fails if r.get("kind") == k)}
                                      for k in sorted({r.get("kind") for r in fails if r.get("kind")})}}
    return out


def pick_design(summary):
    """PLAN.md rule: most negative M1 among designs with M4 < 0, M2 <= V5's M2 and M3 <= 2 x OUT's M3; else the best
    M1/M2 among designs with M4 < 0; else None (stop: fix the forbidden phrasing first)."""
    def ok(name):
        s = summary[name]
        return s["M1"] is not None and s["M4"] is not None and s["M4"] < 0

    v5_m2 = summary.get("V5", {}).get("M2")
    out_m3 = summary.get("OUT", {}).get("M3")
    strict = [d for d in summary if ok(d) and (v5_m2 is None or (summary[d]["M2"] or 0) <= v5_m2)
              and (out_m3 is None or (summary[d]["M3"] or 0) <= 2 * out_m3)]
    if strict:
        return {"design": min(strict, key=lambda d: summary[d]["M1"]), "rule": "strict"}
    loose = [d for d in summary if ok(d) and summary[d]["M2"]]
    if loose:
        return {"design": min(loose, key=lambda d: summary[d]["M1"] / summary[d]["M2"]), "rule": "concentration"}
    return {"design": None, "rule": "none: no design pushes the forbidden actions down"}


# ------------------------------------------------------------------ r3 scoring (PLAN.md "r3", written before results)
TRAINED_A = ("reasoning", "decision", "call")  # the classes fix A trains (call_text and other text are not trained)
CLASSES = ("reasoning", "decision", "call", "call_text", "text")


def audit_row_r3(student_lp, teacher_lp, decision, success, classes):
    """audit_row plus the r3 numbers: m1A (M1 on the decision tokens fix A trains), driftA (drift on the other tokens fix
    A trains), and the absolute teacher-student gap and token count per class (where the training pull sits)."""
    row = audit_row(student_lp, teacher_lp, decision, success)
    if row is None:
        return None
    gap = [t - s for s, t in zip(student_lp, teacher_lp)]
    row["class_abs"] = {c: sum(abs(g) for g, k in zip(gap, classes) if k == c) for c in CLASSES}
    row["class_n"] = {c: sum(1 for k in classes if k == c) for c in CLASSES}
    if success:
        trained = [g for g, k in zip(gap, classes) if k in TRAINED_A]
        row["klA"] = -sum(trained) / len(trained) if trained else None
        return row
    inside = {i for i in (decision or []) if classes[i] in ("decision", "call")}
    row["m1A"] = sum(gap[i] for i in inside) / len(inside) if inside else None
    others = [g for i, (g, k) in enumerate(zip(gap, classes)) if k in TRAINED_A and i not in inside]
    row["driftA"] = -sum(others) / len(others) if others else None
    row["decision_tokens_A"] = len(inside)
    return row


def task_bootstrap(diffs_by_task, draws=10_000, seed=20261027):
    """(mean, [2.5%, 97.5%]) of the mean per-attempt difference, resampling tasks with replacement (each task keeps all of
    its attempts). diffs_by_task: {task: [differences]}."""
    tasks = [t for t, d in diffs_by_task.items() if d]
    if not tasks:
        return None, None
    flat = [x for t in tasks for x in diffs_by_task[t]]
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        pick = [diffs_by_task[tasks[rng.randrange(len(tasks))]] for _ in tasks]
        n = sum(len(p) for p in pick)
        means.append(sum(sum(p) for p in pick) / n)
    means.sort()
    return sum(flat) / len(flat), [means[int(0.025 * draws)], means[int(0.975 * draws) - 1]]


def summarise_r3(rows, base="V5"):
    """Per design: M1, M1A, M2A, M4, M1A by mistake type, the class shares of the gap, and, against `base` on the same
    failed attempts, the paired M1A difference (all points and missed-or-wrong-content points) with task-bootstrap
    ranges."""
    from collections import defaultdict

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    by = defaultdict(dict)
    for r in rows:
        by[r["design"]][r["episode_id"]] = r
    out = {}
    for design, rs in sorted(by.items()):
        fails = [r for r in rs.values() if not r["success"]]
        total = {c: sum(r["class_abs"][c] for r in rs.values()) for c in CLASSES}
        whole = sum(total.values()) or 1.0
        entry = {"failures": len(fails), "successes": len(rs) - len(fails),
                 "M1": mean(r["m1"] for r in fails), "M1A": mean(r.get("m1A") for r in fails),
                 "M2A": mean(r.get("driftA") for r in fails), "M3A": mean(r.get("klA") for r in rs.values() if r["success"]),
                 "M4": mean(r["m1"] for r in fails if r.get("forbidden")),
                 "M4_points": sum(1 for r in fails if r.get("forbidden") and r["m1"] is not None),
                 "M1A_by_kind": {k: {"points": sum(1 for r in fails if r.get("kind") == k and r.get("m1A") is not None),
                                     "M1A": mean(r.get("m1A") for r in fails if r.get("kind") == k)}
                                 for k in sorted({r.get("kind") for r in fails if r.get("kind")})},
                 "gap_share": {c: total[c] / whole for c in CLASSES},
                 "tokens": {c: sum(r["class_n"][c] for r in rs.values()) for c in CLASSES}}
        if design != base and base in by:
            for label, keep in (("all", lambda r: True), ("missed_or_content", lambda r: r.get("kind") == "missed or wrong content")):
                diffs = defaultdict(list)
                for eid, r in rs.items():
                    b = by[base].get(eid)
                    if r["success"] or not b or not keep(r) or r.get("m1A") is None or b.get("m1A") is None:
                        continue
                    diffs[r.get("task")].append(r["m1A"] - b["m1A"])
                m, rng_ = task_bootstrap(diffs)
                entry[f"vs_{base}_{label}"] = {"points": sum(len(d) for d in diffs.values()), "mean": m, "range95": rng_}
        out[design] = entry
    return out


def pick_design_r3(summary, base="V5"):
    """PLAN.md "r3" pick rule (written before results). A rewrite replaces V5 only if: (a) its M1A is below V5's with
    the paired range below 0; (b) its M4 < 0; (c) its M1A on missed-or-wrong-content points is not measured worse
    (range not entirely above 0); (d) its M2A <= 1.25 x V5's. Several -> the lowest M1A; none -> V5."""
    b = summary.get(base) or {}
    ok = []
    for d, s in summary.items():
        if d == base or not d.startswith("W"):
            continue
        a = s.get(f"vs_{base}_all") or {}
        c = s.get(f"vs_{base}_missed_or_content") or {}
        reasons = []
        if not (a.get("range95") and a["range95"][1] < 0):
            reasons.append("a: M1A not measured below V5")
        if not (s.get("M4") is not None and s["M4"] < 0):
            reasons.append("b: M4 not below 0")
        if c.get("range95") and c["range95"][0] > 0:
            reasons.append("c: missed-or-content M1A measured worse")
        if not (s.get("M2A") is not None and b.get("M2A") is not None and s["M2A"] <= 1.25 * b["M2A"]):
            reasons.append("d: M2A above 1.25 x V5")
        s["pick_reasons"] = reasons
        if not reasons:
            ok.append(d)
    if ok:
        return {"design": min(ok, key=lambda d: summary[d]["M1A"]), "rule": "r3: measured improvement", "qualified": ok}
    return {"design": base, "rule": "r3: no rewrite qualified", "qualified": []}


# ------------------------------------------------------------------ round monitors (PLAN.md)

def reasoning_chars(tokenizer, chain):
    total = 0
    for segment in chain["segments"]:
        if segment["kind"] == "action":
            text = tokenizer.decode(segment["ids"], skip_special_tokens=False)
            total += len(text.split(THINK_CLOSE, 1)[0]) if THINK_CLOSE in text else len(text)
    return total


def monitor(first, current):
    """Stop reasons of round `current` against round 1 (dicts with success, reasoning, runaway)."""
    reasons = []
    # 8 points (19 Nov, before any run): with 384 pilot attempts per round the round-to-round noise SD is ~3.5 points,
    # so 5 points would stop a healthy run ~14% of the time over two checks; 8 points is ~1% per check
    if current["success"] < first["success"] - 0.08:
        reasons.append("online success down by more than 8 points")
    # 20 Nov: 2.0x, was 1.3x (my mis-specified bar): arm 2, the only recipe with measured transfer, ran 1.44x on H2 and
    # full3 1.64x on test120; 1.3x would stop every recipe that has shown a gain (weekend/sdpo-v6/PLAN.md, 20 Nov)
    if first["reasoning"] and current["reasoning"] > REASONING_CAP * first["reasoning"]:
        reasons.append(f"reasoning chars above {REASONING_CAP}x round 1")
    if current["runaway"] > max(2 * first["runaway"], 0.03):
        reasons.append("runaway share above 2x round 1 (and 3%)")
    return reasons


def infra_check(outcomes, round_number):
    """20 Nov (pilot r1): a dead rollout engine made all 384 round-2 attempts infrastructure failures, and the round
    reading counted them as success 0.0 (a false "success down" stop). More than INFRA_LIMIT of a round's attempts
    failing on infrastructure is an engine failure: raise. Returns the count."""
    n = len(outcomes)
    bad = sum(1 for o in outcomes.values() if o.get("infrastructure_failure"))
    if n == 0 or bad > INFRA_LIMIT * n:
        raise RuntimeError(f"round {round_number}: {bad} of {n} attempts are infrastructure failures (rollout engine); "
                           "the round would not measure the model")
    return bad


def release_training_memory(log):
    """20 Nov (pilot r1): free the trainer's cached GPU blocks before the rollout engine wakes. After round-1 training
    (peak 25.2 GB) the trainer kept 36.4 GiB reserved; the engine woke with 58.4 GiB of 95 and ran out of memory on its
    first prefill (EngineDeadError). Returns the trainer's reserved GiB; raises above TRAINER_RESERVED_LIMIT_GIB."""
    import gc

    import torch

    gc.collect()
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.empty_cache()
    reserved = torch.cuda.memory_reserved() / 2**30
    allocated = torch.cuda.memory_allocated() / 2**30
    log(stage="trainer_memory_before_refresh", reserved_gib=round(reserved, 2), allocated_gib=round(allocated, 2))
    if reserved > TRAINER_RESERVED_LIMIT_GIB:
        raise RuntimeError(f"trainer still holds {reserved:.1f} GiB after training (limit {TRAINER_RESERVED_LIMIT_GIB}); "
                           "the rollout engine would run out of memory")
    return reserved


# ------------------------------------------------------------------ GPU phases

def prompt_ids(tokenizer, messages):
    """The reflection request as token ids (thinking off). return_dict=True: transformers 5 returns a BatchEncoding
    (not a dict subclass) for tokenize=True, whose list() would be its keys (found by the CPU dry run, 19 Nov)."""
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
                                            return_dict=True)
    ids = encoded["input_ids"]
    return list(ids[0] if ids and isinstance(ids[0], list) else ids)


async def reflect_all(rt, items, temperature=0.7, top_p=0.8):
    """items: [(key, messages)] -> {key: reply text}; the currently loaded student writes every reflection (thinking
    off, so the reply is the JSON object)."""
    import asyncio

    saved = (rt.rollout.temperature, rt.rollout.top_p, dict(rt.rollout.extra_sampling))
    rt.rollout.temperature, rt.rollout.top_p, rt.rollout.extra_sampling = temperature, top_p, {}

    async def one(key, messages):
        prompt = prompt_ids(rt.tokenizer, messages)
        ids, _logps, _finish = await rt.rollout.sample(prompt, REFLECT_MAX_TOKENS)
        return key, rt.tokenizer.decode(ids, skip_special_tokens=True)

    try:
        replies = await asyncio.gather(*(one(k, m) for k, m in items))
    finally:
        rt.rollout.temperature, rt.rollout.top_p, rt.rollout.extra_sampling = saved
    return dict(replies)


def engines():
    """(hidden, shown): the engine's T-A2 context (values hidden) and the values-shown context V1's pointers use."""
    import feedback_versions as fv
    from sdpo_a3 import ENGINE_OPTIONS, make_engine
    from sdpo_ab_engine import EngineContext

    shown = EngineContext(expected_values=True, pool=False, fix_hints=list(fv.SHOWN_ENGINE["fix_hints"]),
                          options=dict(ENGINE_OPTIONS), withhold_untrue=True)
    return make_engine("T-A2"), shown


def build_contexts(rt, design, record, siblings, chain, outcome, hidden, shown, hints):
    """(teacher context, last trained turn or None, located-forbidden flag, why-not reason)."""
    success = bool(outcome and outcome.get("success") == 1)
    task = task_text(chain)
    if success:
        if design == "HINT":
            return hint_context(task, True, truth_filter(hints or {}, True, None)[0]), None, False, None
        return outcome_context(task, True), None, False, None
    engine = hidden(record, siblings, "A", "both")
    if engine is None:
        return None, None, False, "no_engine_feedback"
    last = first_wrong_turn(engine)
    forbidden = "forbids" in (engine.get("notes") or {}).get(last, "") if last else False
    if design == "V5":
        return v5_live(engine, shown(record, siblings, "A", "both")), last, forbidden, None
    if design in ("Z3C", "Z4C", "Z4CP"):  # SDPO closure (21 Nov): the version that passes Phase 1, built live
        import feedback_c

        return (feedback_c.zc_live(design, engine, shown(record, siblings, "A", "both"), record, chain), last, forbidden,
                None)
    if design in ("Y1", "Y2", "Y1P", "Y2P", "Y1PQ", "Y2PQ"):  # SDPO closure round 4 (21 Nov, D1-10): R-c2, pointer
        # notes, P framing, the quiet instruction
        import feedback_c

        return (feedback_c.yc_live(design, engine, shown(record, siblings, "A", "both"), record, chain), last, forbidden,
                None)
    if design in ("S1", "S2", "S1P", "S2P", "S1PQ", "S2PQ", "S1Q", "S2Q", "S1PR", "S1R", "S2PR", "S2R",
                  "S1PU", "S1PRU", "S1PUH", "S1FH", "S1FTH", "S1PTH", "S1FUH", "S1FTUH", "S1PTUH",
                  "S1FTUHA", "S1FTUHM", "S1FUHA", "S1FUHM", "S1PTUHA", "S1PTUHM",
                  "S1FUHW", "S1FUHAW", "S1FUHWN", "S1FUHWNL", "S1FUHWNLK", "S1FUHWNLE", "S1FUHWNLD", "S1FUHWNLJ", "S1FUHWNLEJ", "S1FUHWNLDJ", "S1FUHWNLG", "S1FUHWNLGO", "S1FUHWNLGJ", "S1FUHWNLGOJ", "S1FUHWNLO", "S1FUHWNLOJ", "S1FUHWNLGOJY", "S1FUHWNLOJY", "S1FUHWNLGOJYIV", "S1FUHWNLGOJYIVC"):  # round 5 (21 Nov): R-d, R-n, P framing, the quiet instruction; round 6: U (the
        # note at the cut turn as its own user message)
        import feedback_c

        return (feedback_c.sc_live(design, engine, shown(record, siblings, "A", "both"), record, chain), last, forbidden,
                None)
    if design in ("W1", "W2", "W3", "W4"):  # r3: one change each from V5
        import feedback_versions as fv

        shown_context = shown(record, siblings, "A", "both")
        v5 = v5_live(engine, shown_context)
        if design == "W4":
            return fv.rewrite_w4(v5, engine, shown_context), last, forbidden, None
        return {"W1": fv.rewrite_w1, "W2": fv.rewrite_w2, "W3": fv.rewrite_w3}[design](v5), last, forbidden, None
    if design == "OUT":
        return outcome_context(task, False, outcome), last, forbidden, None
    return hint_context(task, False, truth_filter(hints or {}, False, last)[0], last), last, forbidden, None


async def make_hints(rt, records, chains, outcomes, hidden, folder):
    """HINT design: reflections for every joined attempt (written by the currently loaded student); saved to
    folder/reflections.jsonl.gz. Returns {episode_id: {step: cleaned hint}}."""
    import gzip
    from collections import defaultdict

    import feedback_versions as fv
    import sdpo_ab_world as world

    by_task = defaultdict(list)
    for r in records.values():
        by_task[world.task_name(r)].append(r)
    items, meta = [], {}
    for episode_id, chain in chains.items():
        record, outcome = records[episode_id], outcomes.get(episode_id) or {}
        success = outcome.get("success") == 1
        findings = None
        if not success:
            try:
                engine = hidden(record, by_task[world.task_name(record)], "A", "both")
                findings = fv.split_feedback(engine["first_message"])[1] if engine else None
            except Exception:  # noqa: BLE001 - no findings: the reflector still sees the outcome
                findings = None
        steps = decode_steps(rt.tokenizer, chain)
        items.append((episode_id, reflection_messages(task_text(chain), steps, success, findings)))
        meta[episode_id] = len(steps)
    replies = await reflect_all(rt, items)
    hints, rows = {}, []
    for episode_id, reply in replies.items():
        seen = _norm(visible_text(rt.tokenizer, chains[episode_id]))
        raw = parse_reflection(reply, meta[episode_id])
        clean = {t: clean_hint(h, t, seen) for t, h in raw.items()}
        hints[episode_id] = {t: h for t, h in clean.items() if h}
        rows.append({"episode_id": episode_id, "steps": meta[episode_id], "reply": reply, "parsed": len(raw),
                     "kept": len(hints[episode_id])})
    with gzip.open(folder / "reflections.jsonl.gz", "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return hints


async def phase_design_audit(rt, args, runner):
    """P1: on recorded attempts of the fit tasks (W6 = P9's own), M1-M4 per design and the pick (PLAN.md)."""
    import gzip
    from collections import defaultdict
    from pathlib import Path

    from peft import load_peft_weights, set_peft_model_state_dict

    import sdpo_ab_world as world

    start = load_peft_weights(args.adapter)
    for name in ("default", "sdpo_teacher"):
        set_peft_model_state_dict(rt.model, start, adapter_name=name)
    rt.model.set_adapter("default")
    await rt.rollout.suspend_for_update()
    await rt.rollout.refresh()
    fit = frozenset(json.loads(Path(args.fit).read_text())["task_names"])
    run = Path(args.a3_episodes)
    outcomes = {o["episode_id"]: o for o in json.loads((run / "outcomes.json").read_text())}
    records, chains = {}, {}
    with gzip.open(run / "episodes.jsonl.gz", "rt") as stream:
        for line in stream:
            r = json.loads(line)
            if world.task_name(r) in fit:
                records[r["id"]] = r
    with gzip.open(run / "chains.jsonl.gz", "rt") as stream:
        for line in stream:
            c = json.loads(line)
            if c["episode_id"] in records and not c.get("mismatch"):
                chains[c["episode_id"]] = c
    ok = [e for e in chains if outcomes.get(e) and not outcomes[e].get("infrastructure_failure")]
    rng = random.Random(20261119)
    fails = sorted(e for e in ok if outcomes[e]["success"] != 1)
    wins = sorted(e for e in ok if outcomes[e]["success"] == 1)
    rng.shuffle(fails)
    rng.shuffle(wins)
    chosen = fails[: args.audit_failures] + wins[: args.audit_successes]
    chosen_chains = {e: chains[e] for e in chosen}
    hidden, shown = engines()
    folder = rt.out / "design-audit"
    folder.mkdir(parents=True, exist_ok=True)
    designs = [d.strip() for d in args.audit_designs.split(",")]
    hints = await make_hints(rt, records, chosen_chains, outcomes, hidden, folder) if "HINT" in designs else {}
    await rt.rollout.suspend_for_update()
    by_task = defaultdict(list)
    for r in records.values():
        by_task[world.task_name(r)].append(r)
    rows, kinds = [], {}
    for episode_id in chosen:
        record, chain, outcome = records[episode_id], chains[episode_id], outcomes[episode_id]
        siblings = by_task[world.task_name(record)]
        student_lp = None
        for design in designs:
            try:
                context, last, forbidden, why = build_contexts(rt, design, record, siblings, chain, outcome, hidden,
                                                               shown, hints.get(episode_id))
            except Exception as error:  # noqa: BLE001 - recorded, never stops the audit
                rows.append({"episode_id": episode_id, "design": design, "error": f"{type(error).__name__}: {error}"[:300]})
                continue
            if context is None:
                rows.append({"episode_id": episode_id, "design": design, "skipped": why})
                continue
            # every assistant token, so a final reply's visible text counts as the decision too (training itself uses
            # --loss-tokens); M2/M3 then also cover the visible text
            sequences = runner.prepare_sequences(rt.tokenizer, chain, context, "all")
            if isinstance(sequences, str):
                rows.append({"episode_id": episode_id, "design": design, "skipped": sequences})
                continue
            sequences = truncate(sequences, last)
            student, teacher = sequences["student"], sequences["teacher"]
            if not student["targets"]:
                continue
            if student_lp is None or len(student_lp) != len(student["targets"]):
                student_lp = runner.token_logprobs(rt.model, student["ids"], student["positions"], student["targets"],
                                                   adapter="default")
            teacher_lp = runner.token_logprobs(rt.model, teacher["ids"], teacher["positions"], teacher["targets"],
                                               adapter="sdpo_teacher")
            success = outcome["success"] == 1
            decision = None if success else post_think_indices(rt.tokenizer, student, last)
            classes = token_classes(rt.tokenizer, student["targets"], student["turns"])  # r3: fix A view, gap shares
            row = audit_row_r3(student_lp, teacher_lp, decision, success, classes)
            if row:
                kind = None if success else kinds.setdefault(episode_id, note_kind(
                    hidden(record, siblings, "A", "both"), last))
                rows.append({"episode_id": episode_id, "task": world.task_name(record), "design": design,
                             "forbidden": forbidden, "kind": kind, "last_turn": last, "tokens": len(student_lp), **row})
    rt.model.set_adapter("default")
    with open(folder / "rows.jsonl", "w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    fair = {}
    if getattr(args, "audit_points", None):  # r3, report-only: the fair decision test on the H1 divergence points
        fair = fair_test(rt, runner, Path(args.audit_points), records, chains, outcomes, by_task, designs, hidden, shown,
                         folder)
    summary = summarise([r for r in rows if "success" in r])
    decision = pick_design(summary)
    if any(d.startswith("W") for d in designs):  # r3: its own summary and the pick rule written before results
        summary_r3 = summarise_r3([r for r in rows if "success" in r])
        decision = pick_design_r3(summary_r3)
        (folder / "summary-r3.json").write_text(json.dumps({"summary": summary_r3, "pick": decision, "fair": fair,
                                                            "designs": designs}, indent=1))
    hint_counts = {"attempts_with_hints": sum(1 for h in hints.values() if h),
                   "hints_after_cleaning": sum(len(h) for h in hints.values()), **HINT_STATS}
    (folder / "summary.json").write_text(json.dumps({"summary": summary, "pick": decision, "designs": designs,
                                                     "failures": args.audit_failures,
                                                     "successes": args.audit_successes,
                                                     "hints": hint_counts}, indent=1))
    runner.log(stage="design_audit", pick=decision, **{d: {k: summary[d][k] for k in ("M1", "M2", "M3", "M4")}
                                                       for d in summary})


def fair_test(rt, runner, points_path, records, chains, outcomes, by_task, designs, hidden, shown, folder):
    """r3 report-only (PLAN.md "r3"): at each divergence point (a failed and a successful W6 attempt of one task made the
    same calls and saw the same results, then chose differently), signal = (teacher - student log-likelihood of the
    success's action) - (the same for the failure's own action), on action tokens, per design; signalA leaves out the
    free-text call values (fix A). The student reads its own prefix; the teacher reads the design's feedback (notes up
    to the point). Writes fair.jsonl; returns per-design means."""
    from statistics import fmean

    from sdpo_a3 import action_position, action_tokens, signal_value, student_prefix

    tok = rt.tokenizer
    close = tok.convert_tokens_to_ids(THINK_CLOSE)
    points = [json.loads(line) for line in points_path.read_text().splitlines() if line.strip()]
    rows = []
    for point in points:
        chain, good = chains.get(point["episode_id"]), chains.get(point["correct_source"])
        record = records.get(point["episode_id"])
        if chain is None or good is None or record is None:
            rows.append({**point, "skipped": "chain"})
            continue
        position = action_position(chain, point["turn"])
        wrong = list(chain["segments"][position]["ids"])
        correct = list(good["segments"][action_position(good, point["turn"])]["ids"])
        sw, sc = action_tokens(wrong, close), action_tokens(correct, close)
        if wrong == correct or sw is None or sc is None:
            rows.append({**point, "skipped": "same_or_unclosed"})
            continue
        free_w = call_text_indices(tok, wrong, [1] * len(wrong))
        free_c = call_text_indices(tok, correct, [1] * len(correct))
        s_prefix = student_prefix(chain, position)
        student = {}
        for design in designs:
            try:
                context = build_contexts(rt, design, record, by_task[world_task(record)], chain,
                                         outcomes.get(point["episode_id"]), hidden, shown, None)[0]
            except Exception as error:  # noqa: BLE001 - recorded, never stops the test
                rows.append({**point, "design": design, "skipped": f"{type(error).__name__}"})
                continue
            if context is None:
                rows.append({**point, "design": design, "skipped": "no_context"})
                continue
            t_prefix = runner.engine_teacher_prefix(tok, chain, position, context)
            sums = {}
            for name, turn, keep, free in (("wrong", wrong, sw, free_w), ("correct", correct, sc, free_c)):
                for who, prefix in (("student", s_prefix), ("teacher", t_prefix)):
                    if who == "student" and (name, "lp") in student:
                        values = student[(name, "lp")]
                    else:
                        values = runner.token_logprobs(rt.model, prefix + turn, [len(prefix) + k - 1 for k in keep],
                                                       [turn[k] for k in keep], adapter="default")
                        if who == "student":
                            student[(name, "lp")] = values
                    sums[f"{who}_{name}"] = sum(values)
                    sums[f"{who}_{name}_A"] = sum(v for k, v in zip(keep, values) if k not in free)
            rows.append({**{k: point.get(k) for k in ("episode_id", "task", "turn", "kind", "correct_source")},
                         "design": design,
                         "signal": signal_value(sums["teacher_correct"], sums["student_correct"],
                                                sums["teacher_wrong"], sums["student_wrong"]),
                         "signalA": signal_value(sums["teacher_correct_A"], sums["student_correct_A"],
                                                 sums["teacher_wrong_A"], sums["student_wrong_A"])})
    with open(folder / "fair.jsonl", "w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    out = {}
    for design in designs:
        rs = [r for r in rows if r.get("design") == design and "signal" in r]
        if rs:
            out[design] = {"points": len(rs), "mean_signal": fmean(r["signal"] for r in rs),
                           "share_positive": sum(r["signal"] > 0 for r in rs) / len(rs),
                           "mean_signalA": fmean(r["signalA"] for r in rs),
                           "share_positiveA": sum(r["signalA"] > 0 for r in rs) / len(rs)}
    out["skipped"] = sum(1 for r in rows if "skipped" in r)
    return out


def world_task(record):
    import sdpo_ab_world as world

    return world.task_name(record)


async def phase_online(rt, args, runner):
    """P2/P3: on-policy rounds (PLAN.md)."""
    import gzip
    import time
    from collections import Counter, defaultdict
    from pathlib import Path

    import torch
    from peft import get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict

    import sdpo_ab_world as world
    from sdpo_classic import train_pass

    if args.online_design not in DESIGNS:
        raise ValueError(f"--online-design must be one of {DESIGNS}")
    start = load_peft_weights(args.adapter)
    for name in ("default", "sdpo_teacher"):
        set_peft_model_state_dict(rt.model, start, adapter_name=name)
    rt.model.set_adapter("default")
    await rt.rollout.suspend_for_update()
    await rt.rollout.refresh()
    after = await rt.rollout.score_probe(rt.model, *rt.probe)
    check = runner.adapter_effect(rt.startup_probe, after)
    (rt.out / "adapter-check.json").write_text(json.dumps({"adapter": args.adapter, **check}, indent=1))
    runner.log(stage="adapter-check", **check)
    fit = json.loads(Path(args.fit).read_text())["task_names"]
    sampling = runner.collect_sampling(args.sampling)
    hidden, shown = engines()
    parameters = [p for _, p in rt.model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)  # one AdamW across the rounds
    history, first, checkpoints, verified = [], None, [], None
    for round_number in range(1, args.online_rounds + 1):
        label = f"online/round-{round_number}"
        records_list, joined, outcome_list, summary = await rt.collect(label, fit, args.rollouts_per_task,
                                                                       sampling["temperature"], sampling["top_p"],
                                                                       sampling=sampling)
        records = {r["id"]: r for r in records_list}
        outcomes = {o["episode_id"]: o for o in outcome_list}
        chains = {e: c for e, c in joined.items() if c and not c.get("mismatch")}
        folder = rt.out / label
        # the monitors read this round's fresh rollouts, i.e. the model trained so far (PLAN.md), before any training
        reasoning = [reasoning_chars(rt.tokenizer, c) for c in chains.values()]
        scored = [o for o in outcomes.values() if o.get("success") is not None and not o.get("infrastructure_failure")]
        infra = infra_check(outcomes, round_number)
        reading = {"round": round_number, "success": summary["mean_task_success"],
                   "reasoning": sum(reasoning) / max(1, len(reasoning)),
                   "runaway": sum(1 for o in scored if o.get("runaway")) / max(1, len(scored)),
                   "infrastructure_failures": infra}
        first = first or reading
        reading["stop"] = monitor(first, reading) if round_number > 1 else []
        if reading["stop"]:
            history.append(reading)
            (rt.out / "online").mkdir(parents=True, exist_ok=True)
            (rt.out / "online" / "history.json").write_text(json.dumps(history, indent=1))
            runner.log(stage="online_stop", **reading)
            break
        if checkpoints:  # the previous checkpoint produced rollouts that passed: it is verified
            verified = checkpoints[-1]
        hints = {}
        if args.online_design == "HINT":
            hints = await make_hints(rt, records, chains, outcomes, hidden, folder)
        await rt.rollout.suspend_for_update()
        by_task = defaultdict(list)
        for r in records.values():
            by_task[world.task_name(r)].append(r)
        prepared, admission = [], []
        for episode_id, chain in chains.items():
            outcome = outcomes.get(episode_id) or {}
            entry = {"episode_id": episode_id, "task": world.task_name(records[episode_id]),
                     "success": outcome.get("success")}
            if not outcome or outcome.get("infrastructure_failure") or outcome.get("success") is None:
                admission.append({**entry, "admitted": False, "reason": "not_scored"})
                continue
            try:
                context, last, forbidden, why = build_contexts(rt, args.online_design, records[episode_id],
                                                               by_task[entry["task"]], chain, outcome, hidden, shown,
                                                               hints.get(episode_id))
            except Exception as error:  # noqa: BLE001 - an engine refusal excludes the attempt, never stops
                admission.append({**entry, "admitted": False, "reason": f"engine: {type(error).__name__}: {error}"[:300]})
                continue
            if context is None:
                admission.append({**entry, "admitted": False, "reason": why})
                continue
            if args.online_decision_token:  # reasoning + calls + each turn's decision token (Ali 13 Nov)
                sequences = runner.prepare_sequences(rt.tokenizer, chain, context, "all")
                if not isinstance(sequences, str):
                    sequences = with_decision_tokens(rt.tokenizer, sequences)
            else:
                sequences = runner.prepare_sequences(rt.tokenizer, chain, context, args.loss_tokens)
            if isinstance(sequences, str):
                admission.append({**entry, "admitted": False, "reason": sequences})
                continue
            call_text_tokens = 0
            if getattr(args, "online_call_text", "train") == "skip":  # r3 fix A: free-text call values not trained
                before = len(sequences["student"]["targets"])
                sequences = without_call_text(rt.tokenizer, sequences)
                call_text_tokens = before - len(sequences["student"]["targets"])
            sequences = truncate(sequences, last)
            if not sequences["student"]["targets"]:
                admission.append({**entry, "admitted": False, "reason": "nothing_before_first_wrong_turn"})
                continue
            admission.append({**entry, "admitted": True, "last_turn": last, "forbidden": forbidden,
                              "tokens": len(sequences["student"]["targets"]), "call_text_dropped": call_text_tokens,
                              "teacher_length": len(sequences["teacher"]["ids"])})
            prepared.append((entry, sequences))
        random.Random(20261119 + round_number).shuffle(prepared)
        with gzip.open(folder / "admission.jsonl.gz", "wt") as stream:
            for a in admission:
                stream.write(json.dumps(a) + "\n")
        counts = Counter("admitted" if a["admitted"] else "excluded:" + a["reason"].split(":")[0] for a in admission)
        runner.log(stage="online_prepared", round=round_number, design=args.online_design, **counts)

        def ema():
            student_state = get_peft_model_state_dict(rt.model, adapter_name="default")
            teacher_state = get_peft_model_state_dict(rt.model, adapter_name="sdpo_teacher")
            set_peft_model_state_dict(rt.model, {k: v.lerp(student_state[k].to(v.device), args.online_ema)
                                                 for k, v in teacher_state.items()}, adapter_name="sdpo_teacher")
            rt.model.set_adapter("default")

        started = time.monotonic()
        steps, losses, oom, peak = train_pass(rt.model, prepared, runner.episode_loss, args.lr,
                                              max(1, args.classic_batch), folder / "steps.jsonl", runner.log,
                                              after_step=ema, optimizer=optimizer)
        checkpoint = rt.out / "online" / f"checkpoint-{round_number}"
        rt.model.save_pretrained(checkpoint, selected_adapters=["default"], safe_serialization=True)
        checkpoints.append(str(checkpoint))
        reading.update({"hint_stats": dict(HINT_STATS), "attempts_with_hints": sum(1 for h in hints.values() if h),
                        "admitted": counts.get("admitted", 0), "steps": len(steps), "oom": oom, "peak_mib": peak,
                        "first_loss": steps[0]["loss"] if steps else None,
                        "last_loss": steps[-1]["loss"] if steps else None,
                        "mean_ratio": (sum(x["ratio"] for x in steps if x.get("ratio") is not None)
                                       / max(1, sum(1 for x in steps if x.get("ratio") is not None))),
                        "train_seconds": round(time.monotonic() - started, 1), "checkpoint": str(checkpoint)})
        history.append(reading)
        (rt.out / "online" / "history.json").write_text(json.dumps(history, indent=1))
        runner.log(stage="online_round", **{k: v for k, v in reading.items() if k != "checkpoint"})
        del prepared
        release_training_memory(runner.log)  # 20 Nov: before the engine wakes (pilot r1 out-of-memory)
        await rt.rollout.refresh()  # the next round's rollouts come from the updated student
    stopped = bool(history and history[-1]["stop"])
    # finished: the last checkpoint (it is read by the held-out evaluation); stopped at round r: the last checkpoint
    # whose own rollouts passed the monitors (checkpoint r-2), else the start adapter (no trained model is kept)
    final = (verified if stopped else (checkpoints[-1] if checkpoints else None))
    (rt.out / "online-summary.json").write_text(json.dumps({"design": args.online_design, "rounds": len(history),
                                                            "stopped": stopped, "final_checkpoint": final,
                                                            "start_adapter": args.adapter, "history": history,
                                                            "lr": args.lr, "batch": args.classic_batch,
                                                            "ema": args.online_ema, "fit": args.fit,
                                                            "decision_token": bool(args.online_decision_token),
                                                            "call_text": getattr(args, "online_call_text", "train"),
                                                            "rollouts_per_task": args.rollouts_per_task}, indent=1))
    runner.log(stage="online_done", final=final, rounds=len(history))
