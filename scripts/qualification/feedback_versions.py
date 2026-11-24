"""Feedback versions for the 4B self-teacher (Ali's engine requirements, 15 Nov), as functions of the engine's own
contexts, so the measured versions (a3-rewrites-v1.json, built by weekend/engine-usefulness/rewrite.py) and the
contexts used in training are the same text. Generic: no task-specific wording.

N (framing): the current feedback lines, once, in the first message, under a plain heading; no notes inside the
  conversation, no final note.
P (N + pointers): every "a value this attempt never observed" replaced by the engine's verified location of that
  value without the value ("the email of salesforce contact "Fran Finance" (salesforce_query returns it)"; a value
  that is itself a record's name -> "the name of a <app> <collection> (<tool> lists them)"), or, with no single
  location, by "a value not found in anything this attempt read"; then the engine's verified "No call wrote a ..."
  lines. `shown` is the same attempt's context built with expected values shown and the value-location and
  missed-write hints (SHOWN_ENGINE); only its location lines and facts are read, never shown to the teacher.
Q (P + one guidance line, 16 Nov): when the feedback has a "this task forbids" line, one line before the closing
  sentence tells the teacher to find the reason in the task's instructions or the records and act on it, and that
  changing only the wording or swapping in a similar value repeats the mistake (P's seeded reading: 3 of 20 fixes
  dodged a forbidden string that way; P's forbidden-type gain over no feedback was not shown, +4.2, -7.8..+16.1).
  No new fact and no new prohibition: it only refers to the lines above it.
"""
from __future__ import annotations

import json
import re
from collections import Counter

PLACEHOLDER = "a value this attempt never observed"
OLD_HEAD = "The following is feedback from your unsuccessful earlier attempt:"
OLD_TAIL = "Correctly solve the original question."
NEW_HEAD = "Results of an earlier attempt at this same task (that attempt failed):"
NEW_TAIL = "Do the task correctly now."
NO_PLACE = "a value not found in anything this attempt read"
FORBIDS = "this task forbids"
FORBID_GUIDE = ("For each thing above that this task forbids: find the reason in the task's instructions or in the "
                "records your tools return, and act on that reason. Changing only the wording, or swapping in a "
                "similar value, repeats the mistake.")
# the values-shown engine the pointers were measured with (weekend/engine-usefulness/probe_shown.py)
SHOWN_ENGINE = {"expected_values": True, "fix_hints": ["value_location", "wrong_value", "missed_write",
                                                       "unrequested_change", "call_fix", "why_clause", "solution_note"]}
FIELD = re.compile(r'^In the initial world, "(?P<v>.+?)" is the (?P<field>.+?) of (?P<where>.+?)'
                   r'(?: \((?P<tools>.+?) returns it\))?\.$')
NAMED = re.compile(r'^In the initial world, there is an? (?P<what>.+?) named "(?P<v>.+?)"(?: \(id ".*?"\))?'
                   r'(?: \((?P<tools>.+?) returns it\))?\.$')


def pointer(line):
    """(kind, words) for one value-location line, or None."""
    m = FIELD.match(line)
    if m:
        tools = f' ({m["tools"]} returns it)' if m["tools"] else ""
        return "field", f'the {m["field"]} of {m["where"]}{tools}'
    m = NAMED.match(line)
    if m:
        tools = f' ({m["tools"]} lists them)' if m["tools"] else ""
        return "name", f'the name of a {m["what"]}{tools}'
    return None


def values_in(hidden, shown):
    """The values that fill the placeholders of `hidden`, read from the same line rendered with values shown."""
    parts = hidden.split(PLACEHOLDER)
    pattern = "".join(re.escape(p) + ('"(.+?)"' if i < len(parts) - 1 else "") for i, p in enumerate(parts))
    m = re.match(pattern.rstrip("\\."), shown)
    return list(m.groups()) if m else None


def split_feedback(first_message):
    """(task prompt with its trailing blank lines, the feedback lines block)."""
    if first_message.count(OLD_HEAD) != 1 or not first_message.rstrip().endswith(OLD_TAIL):
        raise ValueError("not an engine feedback message")
    prompt, rest = first_message.split(OLD_HEAD)
    return prompt, rest.rstrip()[: -len(OLD_TAIL)].strip()


def version_n(context):
    prompt, lines = split_feedback(context["first_message"])
    return {**context, "first_message": f"{prompt}{NEW_HEAD}\n\n{lines}\n\n{NEW_TAIL}", "notes": {}, "final_note": None}


def version_p(context, shown):
    """(context, counts). The counts say how each placeholder was replaced."""
    from collections import Counter

    prompt, lines = split_feedback(context["first_message"])
    hidden_facts = {f["text"]: f["id"] for f in context.get("facts") or [] if f.get("text")}
    shown_facts = {f["id"]: f["text"] for f in (shown or {}).get("facts") or [] if f.get("text")}
    hints = ((shown or {}).get("fix_hints") or {}).get("lines", [])
    places = {}
    for x in hints:
        if x.get("kind") == "value_location" and "value" in x:
            p = pointer(x["line"])
            if p:
                places.setdefault(x["value"], p)
    counts, new_lines = Counter(), []
    for ln in lines.split("\n"):
        if PLACEHOLDER not in ln:
            new_lines.append(ln)
            continue
        fid = hidden_facts.get(ln) or hidden_facts.get(ln.rstrip("."))
        vals = values_in(ln, shown_facts[fid]) if fid and fid in shown_facts else None
        pieces = ln.split(PLACEHOLDER)
        rebuilt = pieces[0]
        for i, piece in enumerate(pieces[1:]):
            v = vals[i] if vals and i < len(vals) else None
            p = places.get(v) if v is not None else None
            counts["placeholders"] += 1
            if p:
                counts["pointer_" + p[0]] += 1
                rebuilt += p[1] + piece
            else:
                counts["no_place" if vals else "unaligned"] += 1
                rebuilt += NO_PLACE + piece
        new_lines.append(rebuilt)
    for x in hints:
        if x.get("kind") == "missed_write" and x.get("line") and x["line"] not in new_lines:
            new_lines.append(x["line"])
            counts["missed_write"] += 1
    text = f"{prompt}{NEW_HEAD}\n\n" + "\n".join(new_lines) + f"\n\n{NEW_TAIL}"
    for v in places:  # a hidden value never reaches the teacher (a value the feedback already showed may stay)
        if v not in context["first_message"] and v in text:
            raise ValueError(f"version P would show the hidden value {v!r}")
    return {**context, "first_message": text, "notes": {}, "final_note": None}, dict(counts)


def q_text(p_first_message):
    """Version Q's first message from version P's: FORBID_GUIDE before the closing sentence when a line forbids."""
    if not p_first_message.endswith("\n\n" + NEW_TAIL) or NEW_HEAD not in p_first_message:
        raise ValueError("not a version P message")
    if FORBIDS not in p_first_message.split(NEW_HEAD, 1)[1]:
        return p_first_message
    return p_first_message[: -len(NEW_TAIL)] + FORBID_GUIDE + "\n\n" + NEW_TAIL


def version_q(context, shown):
    """(context, counts) as version_p, with FORBID_GUIDE added (q_text)."""
    p, counts = version_p(context, shown)
    q = q_text(p["first_message"])
    return {**p, "first_message": q}, {**counts, "forbid_guide": int(q != p["first_message"])}


# ------------------------------------------------------------------ version V1 (plan of 16 Nov, Step 3, change 1)
# V1 = the old feedback (T-A2: every line in the first message AND each line again as a note at the turn where the
# mistake happened, plus the final note), with only P's information changes: every placeholder replaced by P's pointer
# (or by NO_PLACE when there is no single location) and P's verified missed-write lines added after the lines. Head,
# tail, note framing and placement are the old feedback's, unchanged. Measured reason (16 Nov review): the old
# feedback, which places notes at the turns, solves 37.2% of points against 29.1% for the same lines shown once (N),
# and 30.7% for P. The first-message lines are P's (version_p itself) except where a line's hidden wording is shared by
# facts with different values (P can then name the other fact's record): NO_PLACE there; inside the notes, each fact the engine
# repeats there is replaced by the same fact rebuilt with P's rule (facts carry ids, so the value comes from the same
# fact rendered with values shown); a note placeholder no fact matches becomes NO_PLACE, P's own fallback.


def _places(shown):
    places = {}
    for x in ((shown or {}).get("fix_hints") or {}).get("lines", []):
        if x.get("kind") == "value_location" and "value" in x:
            p = pointer(x["line"])
            if p:
                places.setdefault(x["value"], p)
    return places


def _core(text):
    """A fact or line text without its line prefix ("Not done: ", "At turn N, ") and final full stop: the form in which
    the engine repeats it inside notes."""
    return re.sub(r"^(Not done: |At turn \d+, )", "", text).rstrip(".")


def _fact_pairs(context, shown, places, counts):
    """[(hidden core, rebuilt core)] for every fact with a placeholder, longest first, rebuilt by P's rule (the value
    read from the same fact rendered with values shown -> its pointer, else NO_PLACE). A hidden core shared by facts
    whose rebuilt cores differ (two facts, same wording, different hidden values) is ambiguous inside a note: it is left
    out, so the note says NO_PLACE there rather than possibly the other fact's pointer (counted)."""
    shown_facts = {f["id"]: f["text"] for f in (shown or {}).get("facts") or [] if f.get("text")}
    by_core = {}
    for f in context.get("facts") or []:
        text = f.get("text") or ""
        if PLACEHOLDER not in text:
            continue
        vals = values_in(text, shown_facts[f["id"]]) if f["id"] in shown_facts else None
        pieces = text.split(PLACEHOLDER)
        rebuilt = pieces[0]
        for i, piece in enumerate(pieces[1:]):
            v = vals[i] if vals and i < len(vals) else None
            p = places.get(v) if v is not None else None
            rebuilt += (p[1] if p else NO_PLACE) + piece
        by_core.setdefault(_core(text), set()).add(_core(rebuilt))
    pairs = []
    for old, news in by_core.items():
        if len(news) == 1:
            pairs.append((old, next(iter(news))))
        else:
            counts["note_ambiguous_core"] += 1
    return sorted(pairs, key=lambda x: (-len(x[0]), x[0]))


def _fill(text, pairs, counts, tag):
    """A note (or final-note) text with each fact's core replaced by its rebuilt core; a placeholder left after that
    (no fact matched) becomes NO_PLACE (counted)."""
    if PLACEHOLDER not in text:
        return text
    n = text.count(PLACEHOLDER)
    for old, new in pairs:
        text = text.replace(old, new)
    left = text.count(PLACEHOLDER)
    counts[f"{tag}_placeholders"] += n
    counts[f"{tag}_by_fact"] += n - left
    counts[f"{tag}_no_fact"] += left
    return text.replace(PLACEHOLDER, NO_PLACE)


def version_v1(context, shown):
    """(context, counts). The old feedback's layout with P's pointers and missed-write lines (see above)."""
    from collections import Counter

    p, counts = version_p(context, shown)
    counts = Counter(counts)
    prompt, old_block = split_feedback(context["first_message"])
    _, p_block = p["first_message"].split(NEW_HEAD, 1)
    p_lines = p_block.rstrip()[: -len(NEW_TAIL)].strip().split("\n")
    old_lines = old_block.split("\n")
    # P matches a line to its fact by the line's text, so two facts with the same hidden wording but different values
    # get one fact's value on both lines; V1 says NO_PLACE on such a line instead (never a pointer to the other record)
    shown_facts = {f["id"]: f["text"] for f in (shown or {}).get("facts") or [] if f.get("text")}
    variants = {}
    for f in context.get("facts") or []:
        t = f.get("text") or ""
        if PLACEHOLDER in t:
            variants.setdefault(t.rstrip("."), set()).add(shown_facts.get(f["id"]))
    for i, ln in enumerate(old_lines):
        if PLACEHOLDER in ln and (len(variants.get(ln.rstrip("."), ())) > 1 or old_lines.count(ln) > 1):
            p_lines[i] = ln.replace(PLACEHOLDER, NO_PLACE)
            counts["line_ambiguous"] += 1
    first = f"{prompt}{OLD_HEAD}\n\n" + "\n".join(p_lines) + f"\n\n{OLD_TAIL}"
    places = _places(shown)
    pairs = _fact_pairs(context, shown, places, counts)
    notes = {k: _fill(v, pairs, counts, "note") for k, v in (context.get("notes") or {}).items()}
    final = context.get("final_note")
    if final:  # the final note is metadata (its text is also a note); its own fields get the same rule
        final = dict(final)
        if final.get("line"):
            final["line"] = _fill(final["line"], pairs, Counter(), "final")
        if "texts" in final:
            final["texts"] = [_fill(t, pairs, Counter(), "final") for t in final["texts"] or []]

    def visible(f, n, fn):
        return f + "\n" + "\n".join(n.values()) + "\n" + json.dumps(fn or {}, ensure_ascii=False)

    before, after = visible(context["first_message"], context.get("notes") or {}, context.get("final_note")), \
        visible(first, notes, final)
    for v in places:  # a hidden value never reaches the teacher (a value the feedback already showed may stay)
        if v not in before and v in after:
            raise ValueError(f"version V1 would show the hidden value {v!r}")
    if PLACEHOLDER in after:
        raise ValueError("version V1 left a placeholder")
    return {**context, "first_message": first, "notes": notes, "final_note": final}, dict(counts)


# ------------------------------------------------------------------ version V2 (plan of 16 Nov, Step 3, change 2)
# V2 = V1 + for each failed prohibition whose target is a single record of the initial world (forbid_locate.py: the
# engine's own WorldIndex and value_line; the value is that record's own name, email or id; containers skipped), one
# sentence on its line and on its note at the turn of the action: where that record is, and to check it first. It says
# nothing about why the action is forbidden (no reason is claimed), only where the target's record is.

FORBID_CHECK = "check that record before acting on it."


def forbid_sentence(value_line):
    """One plain sentence from an engine value line, or None."""
    m = FIELD.match(value_line)
    if m:
        tools = f' ({m["tools"]} returns it)' if m["tools"] else ""
        return f'"{m["v"]}" is the {m["field"]} of {m["where"]}{tools}; {FORBID_CHECK}'
    m = NAMED.match(value_line)
    if m:
        tools = f' ({m["tools"]} returns it)' if m["tools"] else ""
        return f'"{m["v"]}" is the name of a {m["what"]}{tools}; {FORBID_CHECK}'
    return None


def version_v2(context, shown, locations):
    """(context, counts). V1 with the forbidden targets' record sentences (`locations`: {item id: [value lines]})."""
    from collections import Counter

    v1, counts = version_v1(context, shown)
    counts = Counter(counts)
    prompt, _ = split_feedback(v1["first_message"])
    _, old_lines = split_feedback(context["first_message"])
    old_lines = old_lines.split("\n")
    _, v1_lines = split_feedback(v1["first_message"])
    lines = v1_lines.split("\n")
    notes = dict(v1["notes"])
    facts = {f["id"]: f for f in context.get("facts") or [] if f.get("id")}
    places = _places(shown)
    seen_text = context["first_message"] + "\n" + "\n".join((context.get("notes") or {}).values())
    hidden = [v for v in places if v not in seen_text]
    for item_id in sorted(locations):
        sentences = []
        for vl in locations[item_id]:
            s = forbid_sentence(vl)
            if s and any(v in s for v in hidden):
                counts["forbid_dropped_hidden_value"] += 1  # a sentence that would show a hidden value is not said
                continue
            if s and s not in sentences:
                sentences.append(s)
        fact = facts.get(item_id)
        if not sentences or fact is None or FORBIDS not in (fact.get("text") or ""):
            counts["forbid_unplaced"] += 1
            continue
        text = fact["text"]
        index = next((i for i, ln in enumerate(old_lines) if ln == text or _core(text) in ln), None)
        if index is not None and index < len(lines):
            added = [s for s in sentences if s not in lines[index]]
            if added:
                lines[index] = lines[index] + " " + " ".join(added)
                counts["forbid_line"] += 1
        turn = fact.get("turn")
        if turn is not None and int(turn) in notes:
            added = [s for s in sentences if s not in notes[int(turn)]]
            if added:
                notes[int(turn)] = notes[int(turn)] + " " + " ".join(added)
                counts["forbid_note"] += 1
    first = f"{prompt}{OLD_HEAD}\n\n" + "\n".join(lines) + f"\n\n{OLD_TAIL}"
    before = context["first_message"] + "\n" + "\n".join((context.get("notes") or {}).values())
    after = first + "\n" + "\n".join(notes.values())
    for v in places:  # a hidden value never reaches the teacher
        if v not in before and v in after:
            raise ValueError(f"version V2 would show the hidden value {v!r}")
    if PLACEHOLDER in after:
        raise ValueError("version V2 left a placeholder")
    return {**v1, "first_message": first, "notes": notes}, dict(counts)


# ------------------------------------------------------------------ change 3 (plan of 16 Nov, Step 3): tool errors
# On top of a base version (V1 or V2, whichever the solve test keeps): each of the engine's proven call-fix lines
# (abce.fix.call_fix, kind "call_fix": proved by the initial world or by the error the attempt received, e.g. "The
# value "ws_engineers" given at turn 1 is the id of google sheets worksheet ... in the initial world, not a
# spreadsheet; google_sheets_find_worksheet accepted ... at turn 2.") is added to that turn's tool-error line and to
# the note of that turn. A line that would show a hidden value is not added.

def _first_error(message):
    """The first of the error messages a tool-error line or note states ("... (and 2 other ... errors at turn 4)")."""
    return re.sub(r" \(and \d+ other [a-z0-9_]+ errors? at turns? [0-9, ]+\)\.?$", "", message.strip())


# Change 3, widened (Ali, 16 Nov 22:2x): for a tool error that names an object it could not find, one plain sentence
# on what to check. Each rule: (failing tools, error pattern, sentence, lookup tool and what it does). The lookup tool
# is named only when it is in the task's own tool list and is not the failing tool; what it does is said in the words
# of its own description (environment v1.1). Syntax and argument errors get no sentence: their own text says what to
# change. No sentence holds a value.
ERROR_CHECKS = (
    (r"^slack_", r"^Channel '.*' not found", "Use an exact channel name or ID.",
     ("slack_list_channels", "lists all channels")),
    (r"^google_sheets_", r"^Worksheet '.*' not found in spreadsheet", "Use an exact worksheet title or ID.",
     ("google_sheets_get_spreadsheet_by_id", "returns a spreadsheet with its worksheets")),
    (r"^google_sheets_", r"^Spreadsheet '.*' not found", "Use an exact spreadsheet title or ID.",
     ("google_drive_find_multiple_files", "finds spreadsheets by name")),
    (r"^google_sheets_update_row$", r"^Row .* not found in worksheet",
     "The row argument takes the row's own row_id, as the rows read from the sheet show it.", None),
    (r"^jira_", r"^jira_project_not_found", "Use an exact project id, key or name.", ("jira_project", "finds a project")),
    (r"^zoom_find_meeting$", r"^No matching meeting found", "Check the meeting first.", ("zoom_list_meetings", "lists the meetings")),
    (r"^docusign_", r"^Template not found", "Use an exact template name or ID.",
     ("docusign_find_template", "finds a template by name or ID")),
    (r"^google_ads_", r"^Campaign not found in this account", "The campaign is in another account. Check the mainAccountId value.", None),
    (r"^google_ads_", r"^Campaign not found$", "Use an exact campaign name or ID.",
     ("google_ads_find_campaign_by_name", "finds a campaign by name")),
    (r"^slack_send_direct_message$", r"^User '.*' not found", "slack_send_direct_message takes a user ID, username or email.", None),
)
TOOL_ERROR_LINE = re.compile(r"^At turns? (?P<turns>[0-9, ]+?), (?P<tool>[a-z0-9_]+) failed: (?P<error>.*)$")


def error_check(tool, error, task_tools):
    """The sentence for one tool error, or None."""
    first = _first_error(error).rstrip(".")
    for tools, pattern, sentence, lookup in ERROR_CHECKS:
        if re.search(tools, tool) and re.search(pattern, first):
            if lookup is not None and lookup[0] in task_tools and lookup[0] != tool:
                return f"{sentence} {lookup[0]} {lookup[1]}."
            return sentence
    return None


def add_error_checks(base, task_tools, shown=None):
    """(context, counts): `base` with each tool-error line's check sentence appended to that line and to the note of
    each of its turns. A line that already carries a call-fix line for that tool and turn is left alone (the call-fix
    line is more specific)."""
    from collections import Counter

    counts = Counter()
    fixed = {(int(h["turn"]), h.get("tool")) for h in ((shown or {}).get("fix_hints") or {}).get("lines", [])
             if h.get("kind") == "call_fix" and h.get("turn") is not None}
    prompt, block = split_feedback(base["first_message"])
    lines = block.split("\n")
    notes = dict(base["notes"])
    for i, ln in enumerate(list(lines)):
        m = TOOL_ERROR_LINE.match(ln.split(" The value ")[0].split(" The error at turn ")[0])
        if not m:
            continue
        tool = m["tool"]
        turns = [int(t) for t in re.findall(r"\d+", m["turns"])]
        if any((t, tool) in fixed for t in turns):
            counts["error_check_skipped_call_fix"] += 1
            continue
        sentence = error_check(tool, m["error"], task_tools)
        if sentence is None:
            counts["error_check_none"] += 1
            continue
        if sentence not in lines[i]:
            lines[i] = lines[i] + " " + sentence
            counts["error_check_line"] += 1
        for t in turns:
            key = next((k for k in notes if int(k) == t), None)
            if key is None or f"{tool} failed:" not in notes[key] or sentence in notes[key]:
                continue
            note = notes[key]
            start = note.index(f"{tool} failed:") + len(f"{tool} failed:")
            nxt = re.search(r" [a-z0-9_]+ failed: ", note[start:])  # the next tool's error in the same note
            at = start + nxt.start() if nxt else len(note)
            notes[key] = note[:at] + " " + sentence + note[at:]
            counts["error_check_note"] += 1
    first = f"{prompt}{OLD_HEAD}\n\n" + "\n".join(lines) + f"\n\n{OLD_TAIL}"
    return {**base, "first_message": first, "notes": notes}, dict(counts)


def add_call_fixes(base, context, shown):
    """(context, counts): `base` (a version context built from `context`) with the call-fix lines added."""
    from collections import Counter

    counts = Counter()
    places = _places(shown)
    seen_text = context["first_message"] + "\n" + "\n".join((context.get("notes") or {}).values())
    hidden = [v for v in places if v not in seen_text]
    prompt, block = split_feedback(base["first_message"])
    lines = block.split("\n")
    notes = dict(base["notes"])
    for hint in ((shown or {}).get("fix_hints") or {}).get("lines", []):
        if hint.get("kind") != "call_fix" or not hint.get("line") or hint.get("turn") is None:
            continue
        line, turn = hint["line"], int(hint["turn"])
        if any(v in line for v in hidden):
            counts["call_fix_dropped_hidden_value"] += 1
            continue
        tool = hint.get("tool") or ""
        index = next((i for i, ln in enumerate(lines) if f"At turn {turn}, {tool} failed:" in ln
                      or (ln.startswith("At turns ") and f"{tool} failed:" in ln and f" {turn}," in ln)), None)
        if index is not None and line not in lines[index]:
            lines[index] = lines[index] + " " + line
            counts["call_fix_line"] += 1
        if turn in notes and line not in notes[turn]:
            notes[turn] = notes[turn] + " " + line
            counts["call_fix_note"] += 1
        if index is None and turn not in notes:
            counts["call_fix_unplaced"] += 1
    first = f"{prompt}{OLD_HEAD}\n\n" + "\n".join(lines) + f"\n\n{OLD_TAIL}"
    return {**base, "first_message": first, "notes": notes}, dict(counts)

# ------------------------------------------------------------------ change 4 (plan of 16 Nov, Step 3): calm, short wording
# Fixed rewrites of the engine's own templates, meaning kept (anything else is left exactly as it is): head and tail
# without "unsuccessful", "Note:" instead of "Note on your next action:", "Still needed:" instead of "Not done:" /
# "Still not done at this point:", "X exists, but none with Y" / "X exists, but has no Y" -> "X with Y (still
# missing)", "no X exists; one is required" -> "X", "; this task forbids" and "; their F had to stay" as their own
# sentence, "failed:" -> "returned an error:", P's "a value not found in anything this attempt read" -> "an unseen
# value", a full stop between statements a note runs together, and a line or note sentence said twice kept once.
# calm_check: every word of the old text outside the old framing is still there, and nothing is added but the new
# framing (on the first message and every note). Measured on all 1,087 V1 contexts: 0 problems; placeholders 0, double
# negatives 887 -> 0, repeated lines 4 -> 0, repeated note sentences 265 -> 100 (weekend/engine-req/calm_proto.py).

CALM_HEAD = "Notes on an earlier attempt at this task:"
CALM_TAIL = "Now do the task."
NOTE_OLD, NOTE_NEW = "Note on your next action: ", "Note: "
NO_PLACE_OLD, NO_PLACE_NEW = "a value not found in anything this attempt read", "an unseen value"
EXISTS = re.compile(r"^(?P<lead>Not done: |\(\d+\) |\d+\. )?(?P<x>.+?) exists, but (?:none with|has no) (?P<y>.+?)(?P<end>[.;]?)$")


REQUIRED = re.compile(r"^(?P<lead>Not done: |\(\d+\) |\d+\. )?no (?P<x>.+?) exists; one is required(?P<end>[.;]?)$")


def calm_item(text):
    """One line or list item."""
    m = REQUIRED.match(text)
    if m:
        lead = "Still needed: " if m["lead"] == "Not done: " else (m["lead"] or "")
        return f"{lead}{m['x']}{m['end'] or '.'}"
    m = EXISTS.match(text)
    if m:
        lead = "Still needed: " if m["lead"] == "Not done: " else (m["lead"] or "")
        return f"{lead}{m['x']} with {m['y']} (still missing){m['end'] or '.'}"
    if text.startswith("Not done: "):
        return "Still needed: " + text[len("Not done: "):]
    return text


def calm_text(text):
    text = text.replace(NO_PLACE_OLD, NO_PLACE_NEW).replace(" failed: ", " returned an error: ")
    # notes join several statements with a space only; each gets its own sentence
    text = re.sub(r"(?<=[^.\s;:]) (?=[a-z0-9_]+ returned an error: |Still needed: |No call wrote )", ". ", text)
    text = re.sub(r"; this task forbids ", ". This task forbids ", text)
    text = re.sub(r"; their (\S+) had to stay ", r". Its \1 had to stay ", text)
    return text


def calm_note(note):
    body = note[len(NOTE_OLD):] if note.startswith(NOTE_OLD) else note
    body = body.replace("Still not done at this point: ", "Still needed: ")
    parts = re.split(r"(\(\d+\) |\n\d+\. )", body)
    out = parts[0]
    for i in range(1, len(parts), 2):
        item = parts[i + 1]
        tail = item.rstrip()
        sep = tail[-1] if tail[-1:] in ";." else ""
        core = tail[:-1] if sep else tail
        r = REQUIRED.match(core)
        m = EXISTS.match(core)
        core = r["x"] if r else (f"{m['x']} with {m['y']} (still missing)" if m else core)
        out += parts[i] + core + ("." if sep else "") + item[len(tail):]
    out = calm_text(out)
    seen, kept = set(), []
    for sentence in re.split(r"(?<=\.)\s+", out):
        if sentence not in seen:
            seen.add(sentence)
            kept.append(sentence)
    return NOTE_NEW + " ".join(kept)


# Short, single statements (16-17 Nov night, after the B5 audit of the first calm version; meaning kept, calm_check):
#   R1 2+ record sentences "...; check that record before acting on it." in one text: each record sentence kept, the
#      instruction said once at the end ("For each of these, check that record before acting on it.")
#   R2 the engine's call-fix line "The value V given at turn N is the D in the initial world, not a K; T accepted ..."
#      -> "The value V given at turn N is not a K. It is the D in the initial world. T accepted ..."
#   R3 'with cell "F" = not "A", with cell "F" = not "B"' -> 'with cell "F" = not any of "A", "B"'
#   R4 'with X containing "A" or with X containing "B"' -> 'with X containing "A" or "B"'
#   R5 a full stop after a tool's error text that runs into an appended sentence (punctuation only)
#   R6 '... worksheet "W" with cell ...' -> '... worksheet "W". Its cell ...', only when it adds no repeated sentence
FORBID_TAIL = "; " + FORBID_CHECK
FORBID_ONCE = "For each of these, " + FORBID_CHECK
CALL_FIX_LINE = re.compile(r'The value (?P<v>"[^"]*") given at turn (?P<t>\d+) is the (?P<desc>.+?) in the initial world, '
                           r'not (?P<k>an? [a-z][a-z ]*?)(?P<rest>(?:; [A-Za-z0-9_]+ accepted .*?)?)\.')
NOT_RUN = re.compile(r'(?P<lead>with|Its) cell (?P<f>"[^"]*") = not (?P<a>"[^"]*")(?P<more>(?:, with cell (?P=f) = not "[^"]*")+)')
OR_RUN = re.compile(r'(?P<pre>with (?P<what>[a-z ]+?) containing )(?P<a>"[^"]*")(?P<more>(?: or with (?P=what) containing "[^"]*")+)')
APPENDED_START = re.compile(r'(?<=[^.\s]) (?=The value "|The error at turn \d|Use an exact |The row argument |The campaign is |'
                            r'Check the meeting |slack_send_direct_message takes )')
CELLS = re.compile(r'(worksheet "[^"]*") with cell (?=")')


def _short(text):
    text = CALL_FIX_LINE.sub(lambda m: (f'The value {m["v"]} given at turn {m["t"]} is not {m["k"]}. It is the {m["desc"]} '
                                        f'in the initial world.' + (f' {m["rest"][2:]}.' if m["rest"] else "")), text)
    text = APPENDED_START.sub(". ", text)

    def repeats(s):
        parts = [p[len(NOTE_NEW):] if p.startswith(NOTE_NEW) else p for p in re.split(r"(?<=[.;])\s+|\n+", s) if p.strip()]
        return len(parts) - len(set(parts))
    split = CELLS.sub(r"\1. Its cell ", text)
    if repeats(split) <= repeats(text):
        text = split
    text = NOT_RUN.sub(lambda m: f'{m["lead"]} cell {m["f"]} = not any of '
                       + ", ".join([m["a"]] + re.findall(r'= not ("[^"]*")', m["more"])), text)
    text = OR_RUN.sub(lambda m: m["pre"] + " or ".join([m["a"]] + re.findall(r'containing ("[^"]*")', m["more"])), text)
    if text.count(FORBID_TAIL) >= 2:
        text = text.replace(FORBID_TAIL, ".") + " " + FORBID_ONCE
    return text


def version_calm(ctx):
    first = ctx["first_message"]
    prompt, rest = first.split(OLD_HEAD, 1)
    lines = rest.rstrip()[: -len(OLD_TAIL)].strip().split("\n")
    new_lines = []
    for ln in lines:  # the engine sometimes states a line twice; once is enough
        calm = _short(calm_text(calm_item(ln)))
        if calm not in new_lines:
            new_lines.append(calm)
    out_first = f"{prompt}{CALM_HEAD}\n\n" + "\n".join(new_lines) + f"\n\n{CALM_TAIL}"
    notes = {k: _short(calm_note(v)) for k, v in (ctx.get("notes") or {}).items()}
    return {**ctx, "first_message": out_first, "notes": notes}


FRAMING_OLD = set(re.findall(r"[a-z0-9']+", " ".join([OLD_HEAD, OLD_TAIL, NOTE_OLD, NO_PLACE_OLD,
    "Not done exists but none with has no failed Still not done at this point this task forbids their had to stay one is required"]).lower()))
FRAMING_NEW = set(re.findall(r"[a-z0-9']+", " ".join([CALM_HEAD, CALM_TAIL, NOTE_NEW, NO_PLACE_NEW,
    "Still needed with still missing returned an error This task forbids Its had to stay",
    "For each of these any It"]).lower()))


def tokens(text):
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def calm_check(old, new):
    """Problems: a word of the old text (outside the old framing) missing from the new text, or a new word that is not
    part of the new framing. Checked on the first message and on every note."""
    problems = []
    pairs = [("first message", old["first_message"], new["first_message"])] + \
        [(f"note {k}", old["notes"][k], new["notes"].get(k, "")) for k in old.get("notes") or {}]
    for where, a, b in pairs:
        lost = tokens(a) - tokens(b) - FRAMING_OLD
        added = tokens(b) - tokens(a) - FRAMING_NEW
        if lost or added:
            problems.append(f"{where}: lost {sorted(lost)[:5]} added {sorted(added)[:5]}")
    return problems


# ------------------------------------------------------------------ change 4, part 2 (17 Nov night): no stand-in values
# The calm text still wrote a hidden value as the stand-in "an unseen value" (P's "a value not found in anything this
# attempt read"): 1,811 times in 458 of 1,087 contexts. Ali's requirement 3 says no placeholders, and the B5 audit had
# matched only the oldest wording. Seeded reading of V2's solves: 3 of 20 misread the stand-in (one took it as "write
# something found in what you read"). say_unread removes the value slot and says, in its own clause, which field needs a
# value the student has not read ("..., whose body needs a value you have not read"); true at the continuation point,
# since the student has read only part of what the whole earlier attempt read. The earlier attempt's own calls say
# "held". Identity slots with no field ("a docusign envelope for X") keep plain words in place. unread_check proves the
# rewrite: no stand-in left, every quoted value kept in order, words lost only from UNREAD_LOST, added only from
# UNREAD_ADDED.

NOT_READ = "a value you have not read"
UNREAD_PAST = re.compile(r"^\s*(?:Note: )?(?:At turns? [0-9, and]+, )?[a-z0-9]+_[a-z0-9_]+ "
                         r"(?:produced|added|updated|created|changed|sent|posted|wrote|moved|deleted|archived)\b")
_U = NO_PLACE_NEW
UNREAD_RULES = [  # (pattern on the masked item, kind); group f names the field, the match is the removed clause
    (re.compile(r",? with (?P<f>[a-z][a-z ]{0,30}?) (?P<op>containing )?(?P<vals>(?:\x00|" + _U + r")"
                r"(?:(?:, and |, | and )(?:\x00|" + _U + r"))+)"), "list"),
    (re.compile(r",? with (?P<f>[a-z][a-z ]{0,30}?) matching the pattern " + _U), "match"),
    (re.compile(r",? with additional checks (?P<f>[a-z][a-z ]{0,30}?) " + _U), "is"),
    (re.compile(r" on record " + _U), " on a record you have not read"),
    (re.compile(r"(?<=message) " + _U + r"(?= is required)"), " you have not read"),
    (re.compile(r" (?P<f>phone) " + _U), "is"),
    (re.compile(r" for contact " + _U), " for a contact you have not read"),
    (re.compile(r"(?<=\x00) (?P<f>phone|email|title|name|date|amount|status) " + _U), "is"),
    (re.compile(r"(?:,? with |; |, | and )(?P<f>[a-z][a-z ]{0,30}?) not containing " + _U), "leave_out"),
    (re.compile(r",? with (?P<f>[a-z][a-z ]{0,30}?) containing " + _U), "contain"),
    (re.compile(r"(?<! not) containing " + _U), "contain"),  # never after "not": that is a must-leave-out
    (re.compile(r" to " + _U), "to"),
    (re.compile(r"(?:; |, | and )(?P<f>[A-Za-z][A-Za-z0-9 /_-]{0,30}?) = " + _U), "is"),
    (re.compile(r"(?:; |, | and )(?P<f>[a-z][a-z ]{0,20}?) " + _U), "is"),
    (re.compile(r" with cells (?:containing )?(?P<f>[A-Za-z][A-Za-z0-9 /_-]{0,30}?) = " + _U + r"(?P<next>; )?"), "cell"),
    (re.compile(r",? with (?P<f>[a-z][a-z ]{0,20}?) " + _U), "is"),
    (re.compile(r" in (?P<f>list|spreadsheet|calendar|channel|project|board|folder|account|worksheet) " + _U), "in"),
    (re.compile(r" label " + _U), " with a label you have not read"),
    (re.compile(r" (?P<f>[a-z][a-z ]{0,20}?) " + _U + r"(?P<next>; )"), "is_first"),
]
UNREAD_LOST = {"an", "unseen", "value", "with", "containing", "to", "cells", "and", "additional", "checks", "the",
               "matching", "on", "record", "for", "contact", "label"}
UNREAD_ADDED = {"whose", "needs", "need", "held", "is", "was", "has", "had", "one", "a", "value", "values", "you", "have", "not",
                "read", "also", "and", "recipient", "cell", "text", "match", "to", "pattern", "with", "on", "record", "for",
                "contact", "label", "in", "leave", "out", "unknown"}


def _unread_item(item):
    """One masked item (a line, a note sentence or a list item) -> the item without stand-ins."""
    if _U not in item:
        return item
    past = bool(UNREAD_PAST.match(item))
    needs = []
    for _ in range(12):
        if _U not in item:
            break
        for pat, kind in UNREAD_RULES:
            m = pat.search(item)
            if not m or (kind == "list" and _U not in m["vals"]):
                continue
            f = re.sub(r"^with ", "", m.groupdict().get("f") or "")
            new = ""
            if kind == "list":
                vals = re.split(r", and |, | and ", m["vals"])
                shown = [v for v in vals if v == "\x00"]
                if shown:
                    new = f' with {f} {m["op"] or ""}' + (shown[0] if len(shown) == 1 else
                                                         ", ".join(shown[:-1]) + " and " + shown[-1])
                n = len(vals) - len(shown)
                needs.append(f"whose {f} {'also ' if shown else ''}{'held' if past else 'needs'} "
                             f"{'a value' if n == 1 else 'values'} you have not read")
            elif kind == "contain":
                verb = ("held" if past else "need" if f == "cells" else "needs")  # "whose cells need"
                needs.append(f"whose {f or 'text'} {verb} {NOT_READ}")
            elif kind == "leave_out":  # "with body not containing X": the body must leave X out
                needs.append(f"whose {f} {'had' if past else 'has'} to leave out {NOT_READ}")
            elif kind == "match":
                needs.append(f"whose {f} {'had' if past else 'has'} to match a pattern you have not read")
            elif kind in ("is", "to", "cell", "is_first"):
                field = "recipient" if kind == "to" else " ".join(f.split()) + (" cell" if kind == "cell" else "")
                needs.append(f"whose {field} {'was' if past else 'is'} one you have not read")
                new = " with cells " if kind == "cell" and m["next"] else (" " if kind == "is_first" else "")
            elif kind == "in":
                new = f" in a {f} unknown to you"  # these lines often say "created no ...": no second negation
            else:
                new = kind  # a fixed in-place phrase
            item = item[: m.start()] + new + item[m.end():]
            break
        else:
            break
    item = item.replace(_U, NOT_READ)  # identity slots no rule names a field for ("a docusign envelope for X")
    if needs:
        tail = re.search(r"(?: for \x00)?(?: \(still missing\))?[.;]?\s*$", item)
        item = item[: tail.start()] + ", " + " and ".join(dict.fromkeys(needs)) + item[tail.start():]
    return item


def say_unread(text):
    quoted = re.findall(r'"[^"]*"', text)
    pieces = re.split(r"((?<=\.) |\n|\(\d+\) |(?<=\n)\d+\. )", re.sub(r'"[^"]*"', "\x00", text))
    out = "".join(_unread_item(p) for p in pieces).split("\x00")
    return out[0] + "".join(q + piece for q, piece in zip(quoted, out[1:]))


def version_unread(ctx):
    """V4 = version_calm then say_unread on the first message and every note."""
    return {**ctx, "first_message": say_unread(ctx["first_message"]),
            "notes": {k: say_unread(v) for k, v in (ctx.get("notes") or {}).items()}}


def unread_check(calm, new):
    """Problems of say_unread on one context: a stand-in left, a quoted value changed or reordered, a word (counted with
    repeats) lost outside UNREAD_LOST (a "not" only where "not containing" became "has to leave out"), or a word not in
    the calm text added outside UNREAD_ADDED."""
    problems = []
    pairs = [("first message", calm["first_message"], new["first_message"])] + \
        [(f"note {k}", calm["notes"][k], new["notes"].get(k, "")) for k in calm.get("notes") or {}]
    for where, a, b in pairs:
        if _U in b:
            problems.append(f"{where}: stand-in left")
        if re.findall(r'"[^"]*"', a) != re.findall(r'"[^"]*"', b):
            problems.append(f"{where}: quoted values changed")
        plain = lambda s: Counter(re.findall(r"[a-z0-9']+", re.sub(r'"[^"]*"', " ", s.lower())))
        gone = plain(a) - plain(b.replace("you have not read", "you have read"))  # the clause's own "not" never hides a lost one
        gone["not"] -= len(re.findall(r"\bto leave out\b", b)) - len(re.findall(r"\bto leave out\b", a))  # "not containing"
        lost = {w for w, n in gone.items() if n > 0} - UNREAD_LOST
        added = set(plain(b)) - set(plain(a)) - UNREAD_ADDED  # a field name said again in its clause is not new
        if lost or added:
            problems.append(f"{where}: lost {sorted(lost)[:5]} added {sorted(added)[:5]}")
    return problems


# ------------------------------------------------------------------ change 4, part 3 (17 Nov night): tidy
# Found by my Step 4 reading of one V4 context per task, before any V4 run: (1) calm_text's sentence split put a full
# stop after the comma of "At turn N," ("At turn 7,. google_... returned an error"; 253 times in 214 of 1,087 contexts);
# (2) the engine's field words "the from of gmail message X" / "the to of gmail message X" (168 times); (3) statements
# that differ only in a trailing 'for "X"' (which check they come from), said up to 5 times (113 extra lines in 22
# contexts). tidy: ", " for ",. ", "the sender of" / "the recipient of", and such statements said once without the
# trailing qualifier. tidy_check: every statement after tidy is a statement before it (or that statement without its
# qualifier), every statement before it is still said, quoted values only from before.

JARGON = {"from": "sender", "to": "recipient"}
QUALIFIER = re.compile(r' for "[^"]*"(?=(?: \(still missing\))?\.$)')


def _outside_quotes(pattern, repl, text):
    """re.sub applied only where the match lies outside quoted values (found on a same-length mask)."""
    masked = re.sub(r'"[^"]*"', lambda m: '"' + "\x00" * (len(m[0]) - 2) + '"', text)
    out, last = [], 0
    for m in re.finditer(pattern, masked):
        out.append(text[last:m.start()])
        out.append(repl(m) if callable(repl) else repl)
        last = m.end()
    return "".join(out) + text[last:]


def _statements(text):
    """(separator, statement) pairs: first-message lines and a note's sentences, split outside quoted values."""
    masked = re.sub(r'"[^"]*"', lambda m: '"' + "\x00" * (len(m[0]) - 2) + '"', text)
    pairs, last, sep = [], 0, ""
    for m in re.finditer(r"\n|(?<=\.) (?=[A-Z(]|[a-z0-9]+_[a-z0-9_]+ )", masked):  # notes start sentences with tool names
        pairs.append((sep, text[last:m.start()]))
        sep, last = m[0], m.end()
    pairs.append((sep, text[last:]))
    return pairs


def _plain_words(text):
    text = _outside_quotes(r",\. ", ", ", text)
    return _outside_quotes(r"\bthe (from|to) of (?=gmail message \")", lambda m: f"the {JARGON[m[1]]} of ", text)


DEPENDENT = re.compile(r"^(This task |These |Its |It |For each of these|The value |The error |The row |The campaign |"
                       r"Use an exact |Check the |[a-z0-9]+_[a-z0-9_]+ (?:lists|returns|finds|takes) )")


def _facts(text):
    """Statements grouped into facts: a statement plus the statements after it that lean on it ("This task forbids it",
    "Its cell ...", "It is the ..."); a line break always starts a new fact. Returns (separator, [statements], [seps])."""
    facts = []
    for sep, s in _statements(text):
        if facts and sep != "\n" and DEPENDENT.match(s):
            facts[-1][1].append(s)
            facts[-1][2].append(sep)
        else:
            facts.append((sep, [s], [""]))
    return facts


def _fact_text(statements, seps):
    return "".join(sp + s for sp, s in zip(seps, statements))


def tidy(text):
    """Merge whole facts said more than once (also when they differ only in a trailing 'for "X"' qualifier): the first
    is kept without its qualifiers, later ones are dropped with all their dependent statements."""
    facts = _facts(_plain_words(text))
    plain = [_fact_text([QUALIFIER.sub("", s) for s in st], sp) for _, st, sp in facts]
    key = [pl[len(NOTE_NEW):] if pl.startswith(NOTE_NEW) else pl for pl in plain]  # a note's first fact starts "Note: "
    count = Counter(key)
    out, said, newline = [], set(), False
    for (sep, st, sp), pl, k in zip(facts, plain, key):
        body = _fact_text(st, sp)
        if count[k] > 1 and body.strip():
            if k in said:
                newline = newline or sep == "\n"  # a dropped line start passes its line break on
                continue
            said.add(k)
            body = pl
        out.append(("\n" if newline and out else sep if out else "") + body)
        newline = False
    return "".join(out)


def version_tidy(ctx):
    return {**ctx, "first_message": tidy(ctx["first_message"]),
            "notes": {k: tidy(v) for k, v in (ctx.get("notes") or {}).items()}}


def tidy_check(before, after):
    """Problems of tidy on one context: a fact after tidy that is not a fact before it (or one without its trailing
    qualifiers), a fact before it no longer said, a fact said twice, or punctuation / field words left untidied."""
    problems = []
    pairs = [("first message", before["first_message"], after["first_message"])] + \
        [(f"note {k}", before["notes"][k], after["notes"].get(k, "")) for k in before.get("notes") or {}]
    for where, a, b in pairs:
        old = [(_fact_text(st, sp), _fact_text([QUALIFIER.sub("", s) for s in st], sp)) for _, st, sp in _facts(_plain_words(a))]
        new = [_fact_text(st, sp) for _, st, sp in _facts(b)]
        allowed = {f for pair in old for f in pair}
        extra = [f for f in new if f not in allowed]
        if extra:
            problems.append(f"{where}: new fact {extra[0][:80]!r}")
        bare = lambda f: f[len(NOTE_NEW):] if f.startswith(NOTE_NEW) else f
        said = {bare(f) for f in new}
        if any(bare(f) not in said and bare(pl) not in said for f, pl in old):
            problems.append(f"{where}: fact lost")
        repeated = [f for f, n in Counter(f[len(NOTE_NEW):] if f.startswith(NOTE_NEW) else f for f in new).items()
                    if n > 1 and f.strip()]
        if repeated:
            problems.append(f"{where}: fact said twice {repeated[0][:80]!r}")
        if _plain_words(b) != b:
            problems.append(f"{where}: not tidied")
    return problems


# ------------------------------------------------------------------ V5 (17 Nov morning): V4's lines, V1's framing
# V4 diagnosis (measured on the 398 points, paired, task bootstrap; plan entry 17 Nov 08:xx): V4 - V1 on the 246 points
# whose V4 text has no unread clause (only calm/tidy and the framing differ there): solve -5.1 (-9.9..-0.2), forbidden
# -7.2 (-13.2..-1.6); on the 153 points with an unread clause V4 >= V1 (content +7.4, +0.6..+14.4). 87% of V1's forbid
# lines are in V4 word for word (only "; this task forbids" -> ". This task forbids"). My reading of 20 V4 solves: 4 ask
# whether the "Notes on an earlier attempt" are mistakes to avoid or a record of what happened. V5 puts back V1's
# heading, closing line and note prefix (the earlier attempt was unsuccessful; the note is about the next action) and
# keeps every V4 line. frame_check proves that nothing else changes.

def version_frame(ctx):
    first = ctx["first_message"]
    if first.count(CALM_HEAD) != 1 or not first.rstrip().endswith(CALM_TAIL):
        raise ValueError("not a calm (V4) first message")
    body = first.rstrip()
    first = body[: -len(CALM_TAIL)].replace(CALM_HEAD, OLD_HEAD) + OLD_TAIL + first[len(body):]
    notes = {k: (NOTE_OLD + v[len(NOTE_NEW):] if v.startswith(NOTE_NEW) else v) for k, v in (ctx.get("notes") or {}).items()}
    return {**ctx, "first_message": first, "notes": notes}


def frame_check(v4, v5):
    """Problems: V5 differs from V4 anywhere except the heading, the closing line and the note prefix."""
    problems = []
    back = v5["first_message"].replace(OLD_HEAD, CALM_HEAD)
    back = back.rstrip()[: -len(OLD_TAIL)] + CALM_TAIL + back[len(back.rstrip()):] if back.rstrip().endswith(OLD_TAIL) else back
    if back != v4["first_message"] or v5["first_message"].count(OLD_HEAD) != 1:
        problems.append("first message: more than the framing changed")
    if set(v5.get("notes") or {}) != set(v4.get("notes") or {}):
        problems.append("note turns differ")
    for k, v in (v5.get("notes") or {}).items():
        w = v4["notes"][k]
        if not (v.startswith(NOTE_OLD) and w.startswith(NOTE_NEW) and v[len(NOTE_OLD):] == w[len(NOTE_NEW):]) and v != w:
            problems.append(f"note {k}: more than the prefix changed")
    for key in set(v4) | set(v5):
        if key not in ("first_message", "notes") and v4.get(key) != v5.get(key):
            problems.append(f"{key} differs")
    return problems


# ------------------------------------------------------------------ r3 (20 Nov, Ali "Both together"): four rewrites of V5
# weekend/sdpo-v6/PLAN.md "r3". Each is ONE change from V5 (the full3/v6 feedback, sdpo_online.v5_live), adds no fact
# about the attempt and keeps Ali's two hard rules (never false; never forbid what the task requires).
#   W1 reason pointer: FORBID_GUIDE before the closing sentence when a line says this task forbids something (the engine
#      can state no reason: 0 of 57 forbidden lines on 60 H1 failures carry one, weekend/sdpo-v6/r3/engine-probe.jsonl).
#   W2 no restating before the action: in a note, the facts that say this task forbids something become one sentence
#      that names no action and no value (V5's notes never mix forbidden facts with still-needed lists: 914 / 884 / 396
#      other on 1,087 contexts).
#   W3 no "still not done": no final note; notes that list still-needed work go. The first message keeps its lines.
#   W4 where the unread value lives: "needs a value you have not read" -> "needs a value that is not written in any record
#      (it has to be worked out)", only in contexts where every hidden value is aligned to its fact and the engine has no
#      location line for it anywhere in the initial world (w4_guard); otherwise V5 unchanged.

W2_SENTENCE = "this step breaks a rule of this task (see the feedback above)."
W4_ONE = "a value that is not written in any record (it has to be worked out)"
W4_MANY = "values that are not written in any record (they have to be worked out)"
STILL_NOTE = re.compile(r"^\s*Still (?:needed|not done)")
# V5 says a prohibition two ways: "This task forbids ..." (3,331 times on 1,087 contexts) and "the task requires one with
# <field> not containing ..." (248); both count as a forbidden fact for W1 and W2
MUST_NOT = re.compile(r"requires one with [a-z ]+ not containing")


def prohibits(text):
    return FORBIDS in text.lower() or bool(MUST_NOT.search(text))


def _v5_parts(first):
    """(text up to and including the heading, the lines block, the closing sentence with what follows) of V5."""
    if first.count(OLD_HEAD) != 1 or not first.rstrip().endswith(OLD_TAIL):
        raise ValueError("not a V5 first message")
    head, rest = first.split(OLD_HEAD)
    stripped = rest.rstrip()
    return head + OLD_HEAD, stripped[: -len(OLD_TAIL)], OLD_TAIL + rest[len(stripped):]


def rewrite_w1(ctx):
    head, body, tail = _v5_parts(ctx["first_message"])
    if not prohibits(body):
        return ctx
    return {**ctx, "first_message": head + body.rstrip() + "\n\n" + FORBID_GUIDE + "\n\n" + tail}


def _note_without(note, drop):
    """(the note without the facts any of whose statements `drop` matches, number of facts dropped)."""
    prefix = NOTE_OLD if note.startswith(NOTE_OLD) else ""
    kept, dropped = [], 0
    for sep, statements, seps in _facts(note[len(prefix):]):
        if any(drop(s) for s in statements):
            dropped += 1
            continue
        kept.append((sep, _fact_text(statements, seps)))
    return prefix + "".join((sep if i else "") + text for i, (sep, text) in enumerate(kept)), dropped


def rewrite_w2(ctx):
    notes = {}
    for turn, note in (ctx.get("notes") or {}).items():
        rest, dropped = _note_without(note, prohibits)
        if dropped:
            left = rest[len(NOTE_OLD):].strip() if rest.startswith(NOTE_OLD) else rest.strip()
            rest = NOTE_OLD + W2_SENTENCE + (" " + left if left else "")
        notes[turn] = rest
    return {**ctx, "notes": notes}


def rewrite_w3(ctx):
    notes = {t: n for t, n in (ctx.get("notes") or {}).items()
             if not STILL_NOTE.match(n[len(NOTE_OLD):] if n.startswith(NOTE_OLD) else n)}
    return {**ctx, "notes": notes, "final_note": None}


def w4_guard(hidden, shown):
    """True when every placeholder of the engine's hidden context is aligned to its value (the same fact rendered with
    values shown) and the shown context has no value-location line for that value: the engine found it in no record of
    the initial world, so "not written in any record" is the engine's own finding."""
    shown_facts = {f["id"]: f["text"] for f in (shown or {}).get("facts") or [] if f.get("text")}
    located = {x.get("value") for x in ((shown or {}).get("fix_hints") or {}).get("lines", [])
               if x.get("kind") == "value_location"}
    seen = 0
    for f in hidden.get("facts") or []:
        text = f.get("text") or ""
        if PLACEHOLDER not in text:
            continue
        vals = values_in(text, shown_facts[f["id"]]) if f.get("id") in shown_facts else None
        if not vals or len(vals) != text.count(PLACEHOLDER):
            return False
        if any(v in located for v in vals):
            return False
        seen += 1
    return seen > 0


def rewrite_w4(ctx, hidden, shown):
    if not w4_guard(hidden, shown):
        return ctx

    def say(text):
        return (text.replace("needs " + NOT_READ, "needs " + W4_ONE).replace("need " + NOT_READ, "need " + W4_ONE)
                .replace("needs values you have not read", "needs " + W4_MANY)
                .replace("need values you have not read", "need " + W4_MANY))

    return {**ctx, "first_message": say(ctx["first_message"]),
            "notes": {t: say(n) for t, n in (ctx.get("notes") or {}).items()}}


def r3_check(v5, new, design):
    """Problems of one rewrite against V5 (mechanical; the r3 hard-rule sweep). Every rewrite: no quoted value that V5 did
    not already say, and the task prompt unchanged. W1: only FORBID_GUIDE added. W2: no forbidden fact left in a note, and
    no quoted value inside a rewritten note. W3: no still-needed note, no final note, other notes unchanged. W4: only the
    unread phrases changed."""
    problems = []
    quotes = lambda t: set(re.findall(r'"[^"]*"', t))  # noqa: E731
    v5_all = v5["first_message"] + "\n".join((v5.get("notes") or {}).values())
    new_all = new["first_message"] + "\n".join((new.get("notes") or {}).values())
    if quotes(new_all) - quotes(v5_all):
        problems.append("new quoted value")
    if new["first_message"].split(OLD_HEAD)[0] != v5["first_message"].split(OLD_HEAD)[0]:
        problems.append("task prompt changed")
    if design == "W1":
        if new["first_message"].replace("\n\n" + FORBID_GUIDE, "") != v5["first_message"] or new.get("notes") != v5.get("notes"):
            problems.append("W1 changed more than the guide line")
    elif design == "W2":
        if new["first_message"] != v5["first_message"]:
            problems.append("W2 changed the first message")
        for t, n in (new.get("notes") or {}).items():
            if prohibits(n.replace(W2_SENTENCE, "")):
                problems.append(f"W2 note {t} still states a prohibition")
            if W2_SENTENCE in n and quotes(n) - quotes(_note_without(v5["notes"][t], prohibits)[0]):
                problems.append(f"W2 note {t} quotes a forbidden value")
    elif design == "W3":
        if new["first_message"] != v5["first_message"] or new.get("final_note") is not None:
            problems.append("W3 changed the first message or kept the final note")
        for t, n in (new.get("notes") or {}).items():
            if n != v5["notes"].get(t) or STILL_NOTE.match(n[len(NOTE_OLD):]):
                problems.append(f"W3 note {t} changed or still lists work")
    elif design == "W4":
        undo = lambda t: (t.replace(W4_ONE, NOT_READ).replace(W4_MANY, "values you have not read"))  # noqa: E731
        if undo(new["first_message"]) != v5["first_message"] or {k: undo(v) for k, v in (new.get("notes") or {}).items()} != v5.get("notes"):
            problems.append("W4 changed more than the unread phrases")
    return problems
