"""SDPO closure, round-2 feedback designs Z1-Z4 built LIVE from the engine (decision D1-4, 20 Nov 2026). Each design is a
general rule on V5f; nothing is written for a particular task or attempt. On recorded attempts the texts are identical to
weekend/sdpo-close/tools/build_candidates_z.py (rewrites v9); check_z_live.py proves it.

  V5f  V5 with tidy2: "Still needed" facts that differ only in a trailing 'for "X"' are kept (V5's tidy merged them and
       lost X, e.g. three DocuSign signers -> "a docusign envelope signer").
  Z1   no notes: V5f's first message only.
  Z2   V5f with true pointer notes (each checked against the summary; a note with no true pointer raises ValueError).
  Z3   Z1, and every tool-error line (both formats) shows the tool's error text without the failed calls' own argument
       values, plus the what-to-check sentence (feedback_versions.ERROR_CHECKS; the spreadsheet/worksheet sentence names
       a worksheet argument only when the tool has one).
  Z4   Z3, and a "This task forbids it." line whose engine check is scoped ends
       "This task forbids that exact action, not every <tool> call."
Inputs (all live in sdpo_online.build_contexts): the engine's T-A2 context (`facts`), the shown context, the attempt's
record (its checks and tool list) and its chain (each turn's raw text and the tool schemas)."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

import feedback_versions as F

DESIGNS_Z = ("Z1", "Z2", "Z3", "Z4")
NOTE_HEAD = "Note on your next action: "
STILL = ("Still needed:", "Still not done at this point:")
TURNS = re.compile(r"^At turns? (?P<t>\d+(?:(?:, and |, | and )\d+)*), ")
ERR1 = re.compile(r"^At turns? (?P<turns>\d+(?:(?:, and |, | and )\d+)*), (?P<tool>[a-z0-9_]+) "
                  r"(?:returned an error|failed): (?P<error>.*)$")
ERR2 = re.compile(r"^(?P<tool>[a-z0-9_]+) failed (?P<n>\d+) times \(turns (?P<a>\d+)-(?P<b>\d+)\), first: (?P<error>.*)$")
SUFFIX = re.compile(r" \(and \d+ other [a-z0-9_]+ errors?(?: at turns? [0-9, and]+)?\)$")
FORBIDS_IT = re.compile(r"^(?P<lead>At turns? [0-9, and]+, (?P<tool>[a-z0-9_]+) .*?)\. This task forbids it\.$")
SHEET_SLASH = (r"^google_sheets_", r"^Spreadsheet '[^'/]+/[^']+' not found",
               "The spreadsheet argument takes only the spreadsheet; give the worksheet in its own argument.")
SHEET_ONLY = "The spreadsheet argument takes only the spreadsheet."
X3_SCOPE = "This task forbids that exact action, not every {tool} call."
P_TURN = "Note on your next action: this is the step where the problem listed above for turn {t} happened."
P_TURNS = "Note on your next action: this is the step where the problems listed above for turn {t} happened."
P_STILL = "Note on your next action: the work listed above as still needed is still missing."
P_RANGE = "Note on your next action: {tool} fails at this step, as listed above."
P_AGAIN = "Note on your next action: {tool} fails again at this step, as listed above."
ERR_SENT = re.compile(r"^(?P<tool>[a-z0-9_]+) (?:returned an error|failed): ")
QUOTE = re.compile(r"'[^'\n]{1,300}'|\"[^\"\n]{1,300}\"|\{[^{}\n]{1,300}\}")
CALL = re.compile(r"<tool_call>\s*<function=(?P<name>[^>\s]+)>(?P<body>.*?)</function>\s*</tool_call>", re.S)
PARAM = re.compile(r"<parameter=(?P<key>[^>\s]+)>\n?(?P<value>.*?)\n?</parameter>", re.S)


def split(first_message):
    prompt, block = F.split_feedback(first_message)
    return prompt, block.split("\n")


def join(prompt, lines):
    return f"{prompt}{F.OLD_HEAD}\n\n" + "\n".join(lines) + f"\n\n{F.OLD_TAIL}"


# ------------------------------------------------------------------ V5f
def tidy2(text):
    """tidy, except that a "Still needed" fact merges only with an identical one (its 'for "X"' kept)."""
    facts = F._facts(F._plain_words(text))
    plain = [F._fact_text([F.QUALIFIER.sub("", s) for s in st], sp) for _, st, sp in facts]
    full = [F._fact_text(st, sp) for _, st, sp in facts]
    still = [st[0].lstrip().startswith(STILL) for _, st, _ in facts]
    key = [(fu if s else pl) for fu, pl, s in zip(full, plain, still)]
    key = [k[len(F.NOTE_NEW):] if k.startswith(F.NOTE_NEW) else k for k in key]
    count = Counter(key)
    out, said, newline = [], set(), False
    for (sep, st, sp), pl, fu, k, s in zip(facts, plain, full, key, still):
        body = fu
        if count[k] > 1 and body.strip():
            if k in said:
                newline = newline or sep == "\n"
                continue
            said.add(k)
            body = fu if s else pl
        out.append(("\n" if newline and out else sep if out else "") + body)
        newline = False
    return "".join(out)


def v5_and_v5f(engine_context, shown_context):
    """(V5 as sdpo_online.v5_live builds it, V5f's first message)."""
    v1, _counts = F.version_v1(engine_context, shown_context)
    calm = F.version_calm(v1)
    v4 = {"first_message": calm["first_message"], "notes": calm["notes"], "final_note": v1.get("final_note")}
    unread = F.version_unread({"first_message": v4["first_message"], "notes": v4["notes"] or {}})
    tidy = F.version_tidy(unread)
    v5 = F.version_frame({**v4, "first_message": tidy["first_message"], "notes": tidy["notes"]})
    v5f_first = F.version_frame({**v4, "first_message": tidy2(unread["first_message"]), "notes": tidy["notes"]})["first_message"]
    return v5, v5f_first


# ------------------------------------------------------------------ Z2 pointers
def note_kind(body):
    if body.startswith(STILL):
        return "still"
    sents = [s for s in re.split(r"(?<=[.'\"])\s+(?=[a-z0-9]+_[a-z0-9_]+ )", body) if s.strip()]
    tools = {ERR_SENT.match(s)["tool"] for s in sents if ERR_SENT.match(s)}
    if sents and len(tools) == 1 and all(ERR_SENT.match(s) for s in sents):
        return "error:" + tools.pop()
    return "other"


def z2_notes(first_lines, v5_notes):
    listed = defaultdict(int)
    for ln in first_lines:
        m = TURNS.match(ln)
        if m:
            for t in re.findall(r"\d+", m["t"]):
                listed[int(t)] += 1
    ranges, err_tools = defaultdict(list), set()
    for ln in first_lines:
        m1, m2 = ERR1.match(ln), ERR2.match(ln)
        if m1:
            err_tools.add(m1["tool"])
        if m2:
            ranges[m2["tool"]].append((int(m2["a"]), int(m2["b"])))
            err_tools.add(m2["tool"])
    has_still = any(l.startswith("Still needed:") for l in first_lines)
    notes = {}
    for t, note in v5_notes.items():
        body = note[len(NOTE_HEAD):] if note.startswith(NOTE_HEAD) else note
        kind, ti = note_kind(body), int(t)
        if kind == "still":
            if has_still:
                notes[t] = P_STILL
                continue
        elif kind.startswith("error:"):
            tool = kind[6:]
            if any(a <= ti <= b for a, b in ranges.get(tool, [])):
                notes[t] = P_RANGE.format(tool=tool)
                continue
            if ti in listed:
                notes[t] = (P_TURNS if listed[ti] > 1 else P_TURN).format(t=t)
                continue
            if tool in err_tools:
                notes[t] = P_AGAIN.format(tool=tool)
                continue
        elif ti in listed:
            notes[t] = (P_TURNS if listed[ti] > 1 else P_TURN).format(t=t)
            continue
        raise ValueError(f"Z2: no true pointer for the {kind} note at turn {t}")
    return notes


# ------------------------------------------------------------------ Z3 tool-error lines
def call_values(raw, tool):
    vals = set()
    for m in CALL.finditer(raw or ""):
        if m["name"] != tool:
            continue
        for p in PARAM.finditer(m["body"]):
            v = p["value"].strip()
            vals.add(v)
            try:
                j = json.loads(v)
            except ValueError:
                continue
            stack = [j]
            while stack:
                x = stack.pop()
                if isinstance(x, dict):
                    stack.extend(x.values())
                    vals.add(json.dumps(x))
                elif isinstance(x, list):
                    stack.extend(x)
                elif x is not None and not isinstance(x, bool):
                    vals.add(str(x))
    return {v for v in vals if v.strip()}


def _token_from_call(token, values):
    inner = token[1:-1] if token[0] in "'\"" else token
    if not inner.strip():
        return False
    for v in values:
        if inner == v or (len(inner) >= 2 and inner in v):
            return True
        if token[0] == "{":
            try:
                if json.loads(token) == json.loads(v):
                    return True
            except ValueError:
                pass
    return False


# v12 (21 Nov): the tool's own list of allowed values ("Input should be 'queue', 'schedule', ... or 'draft'") is never
# masked: removing a call value that is also an allowed option made the list false (1 context, marketing.seasonal_campaign)
ENUM = re.compile(r"""(['"])[^'"\n]{1,40}\1(?:, (['"])[^'"\n]{1,40}\2)+(?:,? or (['"])[^'"\n]{1,40}\3)?""")


def _in_enum(spans, a, b):
    return any(s <= a and b <= e for s, e in spans)


def mask_error(error, values):
    removed = 0
    out, last = [], 0
    enums = [(m.start(), m.end()) for m in ENUM.finditer(error)]
    for m in QUOTE.finditer(error):
        if not _token_from_call(m[0], values) or _in_enum(enums, m.start(), m.end()):
            continue
        before = error[last:m.start()]
        cut = re.search(r"(?:,? but found|,? found) ?$", before)
        if cut:
            before = before[: cut.start()]
        elif re.search(r"\bUnexpected ?$", before):
            before = before.rstrip() + " text"
        out.append(before.rstrip(" "))
        last, removed = m.end(), removed + 1
    text = "".join(out) + error[last:]
    for v in sorted((v for v in values if len(v) >= 3 and "\n" not in v), key=len, reverse=True):
        pat = re.compile(r"(?:(?:,? but found|, found|(?<!\bnot) found) )?(?<![\w@.\-])" + re.escape(v) + r"(?![\w@\-])")
        enums = [(m.start(), m.end()) for m in ENUM.finditer(text)]
        n = len([m for m in pat.finditer(text) if not _in_enum(enums, m.start(), m.end())])
        text = pat.sub(lambda m: m.group(0) if _in_enum(enums, m.start(), m.end()) else "", text)
        removed += n
    if not removed:
        return error, 0
    text = re.sub(r" +([.,;:)])", r"\1", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"  +", " ", text).strip()
    return text, removed


def check_sentence(tool, error, task_tools, params):
    first = F._first_error(error).rstrip(".")
    if re.search(SHEET_SLASH[0], tool) and re.search(SHEET_SLASH[1], first):
        return SHEET_SLASH[2] if "worksheet" in params.get(tool, set()) else SHEET_ONLY
    return F.error_check(tool, error, task_tools)


def error_lines(lines, raws, task_tools, params):
    out = []
    for ln in lines:
        m1, m2 = ERR1.match(ln), ERR2.match(ln)
        m = m1 or m2
        if m:
            turns = [int(x) for x in re.findall(r"\d+", m1["turns"])] if m1 else list(range(int(m2["a"]), int(m2["b"]) + 1))
            err, suffix = m["error"], ""
            sm = SUFFIX.search(err)
            if sm:
                err, suffix = err[: sm.start()], err[sm.start():]
            values = set().union(*(call_values(raws[t - 1] if 0 < t <= len(raws) else "", m["tool"]) for t in turns))
            masked, _n = mask_error(err, values)
            ln = ln[: m.start("error")] + masked + suffix
            s = check_sentence(m["tool"], m["error"], task_tools, params)
            if s and s not in ln:
                ln = ln.rstrip()
                ln = f"{ln if ln.endswith(('.', '!', '?')) else ln + '.'} {s}"
        out.append(ln)
    return out


# ------------------------------------------------------------------ Z4 forbid scope
def scoped(facts, asserts, tool, turn):
    located = [f for f in facts if f["kind"] == "prohibited" and f.get("text") and f.get("name") == tool
               and f.get("turn") == turn]
    if not located:
        return False
    for f in located:
        if f["id"].startswith("native:"):
            a = asserts[int(f["id"].split(":")[1])]
            if not {k for k in a["params"] if k != "excluded"}:
                return False
        elif not str(f.get("type") or "").startswith("effects.prohibited_when"):
            return False
    return True


def forbid_lines(lines, facts, asserts):
    out = []
    for ln in lines:
        m = FORBIDS_IT.match(ln)
        if m and scoped(facts, asserts, m["tool"], int(re.findall(r"\d+", m["lead"])[0])):
            ln = f"{m['lead']}. " + X3_SCOPE.format(tool=m["tool"])
        out.append(ln)
    return out


# ------------------------------------------------------------------ live entry point
def attempt_inputs(record, chain):
    """(each turn's raw text, the task's tool names, {tool: parameter names}, the task's checks)."""
    raws = [s.get("raw") or "" for s in chain["segments"] if s["kind"] == "action"]
    trace = record["traces"][0]
    task_tools = {x.get("name") or (x.get("function") or {}).get("name") for x in (trace.get("tools") or [])}
    params = {}
    for t in chain.get("tools") or []:
        f = t.get("function") or t
        if f.get("name") and f["name"] not in params:
            params[f["name"]] = set(((f.get("parameters") or {}).get("properties") or {}).keys())
    return raws, task_tools, params, trace["info"]["automationbench"]["assertions"]


def z_live(design, engine_context, shown_context, record, chain):
    """The teacher context of design Z1-Z4 for one failed attempt (the engine's context with the version's text)."""
    if design not in DESIGNS_Z:
        raise ValueError(design)
    v5, first = v5_and_v5f(engine_context, shown_context)
    if design == "Z1":
        return {**engine_context, "first_message": first, "notes": {}, "final_note": None}
    prompt, lines = split(first)
    if design == "Z2":
        notes = z2_notes(lines, {str(k): v for k, v in (v5.get("notes") or {}).items()})
        return {**engine_context, "first_message": first, "notes": {int(k): v for k, v in notes.items()},
                "final_note": v5.get("final_note")}
    raws, task_tools, params, asserts = attempt_inputs(record, chain)
    lines = error_lines(lines, raws, task_tools, params)
    if design == "Z4":
        lines = forbid_lines(lines, engine_context.get("facts") or [], asserts)
    return {**engine_context, "first_message": join(prompt, lines), "notes": {}, "final_note": None}
