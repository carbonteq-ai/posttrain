"""SDPO closure R-c (21 Nov 2026): per-point visibility as a general TEXT rule on the engine's finished feedback, so the
engine's truth logic stays untouched. A quoted value the student had not seen before turn t (the engine's own test:
observed in the request or a tool result, or written in its own calls, before that turn, normalised as the engine
normalises) is replaced by the engine's stand-in, and the existing unread wording (feedback_versions.say_unread) says
it the way V4/V5 already say values never seen at all. Forbid sentences get the engine's own hidden-value wording
("This task forbids the text shown in that slack message"); a trailing 'for "X"' (which check a line comes from) goes
when X is hidden. Still-needed lines that become identical are said once with their count.
redact(first_message, seen) -> (text, counts, problems); seen(value) -> bool.
v13 (21 Nov, D1-10): R-c2 (redact(..., required_only=True)) hides an unseen value only when the feedback also needs it
written (a still-needed value); designs Y1/Y2(+P) in yc_live."""
from __future__ import annotations

import re
from collections import Counter

import feedback_versions as F

QUOTED = re.compile(r'"([^"\n]{1,300})"')
U = F._U  # "an unseen value"
FORBID_FIELD = re.compile(r"This task forbids (?P<f>[a-z][a-z ]{0,30}?) (?:containing |= )?" + re.escape(U)
                          + r"(?:(?:, | and | or )" + re.escape(U) + r")*(?P<tail> in (?:that|those|these|each of these) [^.;]+)")
FORBID_FIELD_QUOTED = re.compile(r'This task forbids (?P<f>[a-z][a-z ]{0,30}?) (?:containing |= )?(?:"[^"]*"(?:, | and | or )?)*'
                                 + re.escape(U) + r'(?:(?:, | and | or )(?:"[^"]*"|' + re.escape(U) + r'))*'
                                 r"(?P<tail> in (?:that|those|these|each of these) [^.;]+)")
QUALIFIER_U = re.compile(r" for " + re.escape(U) + r"(?=(?: \(still missing\))?[.;]|$)")
STILL = "Still needed: "


def visible_messages(messages, before_turn):
    if before_turn is None:
        return messages
    out, turn = [], 0
    for m in messages:
        if (m or {}).get("role") == "assistant":
            turn += 1
            if turn >= before_turn:
                break
        out.append(m)
    return out


def tool_doc_strings(tool_defs):
    """Every key and scalar of the attempt's tool definitions (names, descriptions, parameter names, enum values): text
    the attempt has in its prompt at every turn."""
    if isinstance(tool_defs, dict):
        return [s for k, v in tool_defs.items() for s in [str(k)] + tool_doc_strings(v)]
    if isinstance(tool_defs, (list, tuple)):
        return [s for v in tool_defs for s in tool_doc_strings(v)]
    if isinstance(tool_defs, (str, int, float)) and not isinstance(tool_defs, bool):
        return [str(tool_defs)]
    return []


def seen_test(episode_messages, tool_names, before_turn, tool_defs=None):
    """The engine's observed/written test (abce Corpus), on what the attempt had seen before `before_turn`. v12 (21 Nov):
    the attempt's tool definitions count as seen too (the engine's Corpus takes only the tool names), so a value the
    tool documentation states (an enum such as "sent", a schema example such as {"Name": "John"}) is not hidden."""
    from abce.record.observed import Corpus

    docs = [{"role": "user", "content": "\n".join(tool_doc_strings(tool_defs))}] if tool_defs else []
    corpus = Corpus(docs + list(visible_messages(episode_messages, before_turn)), tuple(tool_names or ()))
    return lambda v: corpus.is_observed(v) or corpus.is_written(v)


# a quoted name right after cell/column/field/property/key/header, or a Salesforce object collection ('salesforce field
# in "accounts"', 'salesforce records in "tasks"'; the only ' in "X"' slots the engine writes), is a schema name
SCHEMA_SLOT = re.compile(r'\b(?:cells?|columns?|fields?|propert(?:y|ies)|keys?|headers?|(?:field|records?) in) '
                         r'"([^"\n]{1,300})"')


NEG = re.compile(r"\b(no|not|none|never|without|nothing|nobody|neither|nor)\b|n't\b")
CLAUSE = re.compile(r"[;,:]|\band\b|\bbut\b|\bor\b")
GARBLE = re.compile(r"= in |= ,|=\s*$|\bwhose not\b|\bnot held\b|= not a value you have not read|not any of a value you have"
                    r"|not any of values you have")
FORBID_SOME = "This task forbids some of what that call wrote."  # the engine's forbid lines quote only values the call wrote


def _trim_head(head):
    """The line's opening words, ended cleanly before its first hidden value: trailing punctuation, a dangling "not" /
    "containing", and a trailing slot phrase ("on record", "in group", "with recipient profile id", "in") go; a bare
    noun never does."""
    h = head.rstrip()
    while True:
        before = h
        h = re.sub(r"[\s,;(=:.]+$", "", h)
        h = re.sub(r"\s+(?:not any of|any of|not|containing|lacking|whose)$", "", h)
        h = re.sub(r"\s+(?:in|on|for|to|with|from|at)(?:\s+(?:the\s+)?[a-z_]+){0,3}$", "", h)
        if h == before:
            return h


def _hard(sentence):
    """A sentence a 4B model should not get: two negations in one clause (the B5 audit's own rule) or a garbled slot."""
    plain = re.sub(r'"[^"]*"', '"Q"', sentence.lower())
    return bool(GARBLE.search(sentence)) or any(len(NEG.findall(c)) >= 2 for c in CLAUSE.split(plain))


def _plain_line(line, hidden):
    """Tier 2: the line's own true opening words up to its first hidden value (or grouped list), then one clause that the
    details were not read; a following sentence of the line stays only if it names no hidden value (a forbid sentence
    in the engine's hidden form, "This task forbids the X shown in that Y", names none)."""
    sents = re.split(r"(?<=\.) (?=[A-Z])", line)
    first, rest = sents[0], sents[1:]
    cut = len(first)
    for q in hidden:
        i = first.find(f'"{q}"')
        if 0 <= i < cut:
            cut = i
    j = first.find(" (")
    if 0 <= j < cut:
        cut = j
    head = _trim_head(first[:cut])
    kept, forbid_lost = [], False
    for x in rest:
        if not x.startswith("This task forbids"):
            continue  # v12: the plain form keeps only the forbid sentence; other details are summed up by PLAIN_TAIL
        y = x
        for q in sorted(set(hidden), key=len, reverse=True):
            y = y.replace(f'"{q}"', U)
        y = _field_words(_forbid_rules(y))
        if U not in y and not _hard(y) and not unreadable(y):
            kept.append(y)
        else:
            forbid_lost = True
    if forbid_lost and not any(k.startswith("This task forbids") for k in kept):
        kept.append(FORBID_SOME)
    return " ".join([f"{head}; {PLAIN_TAIL}"] + kept)


# ------------------------------------------------------------------ v12 (21 Nov): readable hidden-value wording
# Found by reading every distinct v11 line shape (weekend/sdpo-close/phase1/readability/): say_unread was built for V4/V5,
# where hidden values sat in field slots; per-point hiding also hides identity values ("the recipient of gmail message
# X", "on ticket X", "worksheet X", "for X"), lists that mix shown and hidden values, and forbid sentences with cells or
# several fields. say_unread then left the plain words "a value you have not read" in place: 316 of 5,715 lines per
# version read garbled ("the recipient of gmail message a value you have not read"). The rules below say an identity
# value as "a <thing> you have not read", count hidden list items, and name the forbidden field; `unreadable` then
# accepts the hidden-value words only in their own clause, and any other line goes to the plain form (tier 2).
PLAIN_TAIL = "some details are values you have not read."
UQ = re.escape(U)
ENT = r"(?:[a-z0-9]+ ){0,3}[a-z]+"
ENTITY_NOUNS = (r"(?:ticket|thread|group|account|customer|template|signer|profile|user|channel|spreadsheet|contact|list|"
                r"board|folder|project|calendar|audience|opportunity|message|deal|card|form|page|database|team|workspace|"
                r"organization|company|campaign|segment|pipeline|queue|inbox|mailbox|conversation|order|product|invoice)")


def _a(noun):
    return ("an " if noun[:1] in "aeiou" and not re.match(r"u(?:s|ni)", noun) else "a ") + noun


def _or_chain(m):
    items = re.findall(r'"[^"\n]*"|' + UQ, m.group(0)[len(f"with {m['f']} "):])
    shown = [x for x in items if x != U]
    hidden = len(items) - len(shown)
    if not hidden:
        return m.group(0)
    if not shown:
        return f"with {m['f']} {U}"
    return (f"with {m['f']} " + " or ".join(shown) + " or "
            + ("a value you have not read" if hidden == 1 else f"{hidden} values you have not read"))


C_RULES = [
    # alternatives for one field ('with text "A" or with text X or with text "B"'): the shown ones, then the hidden count
    (re.compile(r'\bwith (?P<f>[a-z][a-z ]{0,30}?) (?:"[^"\n]*"|' + UQ + r')(?: or with (?P=f) (?:"[^"\n]*"|' + UQ + r'))+'),
     _or_chain),
    # 'in spreadsheet the id of google sheets spreadsheet X' -> 'in a google sheets spreadsheet you have not read'
    (re.compile(r"\b(?P<prep>in|on|from|for) (?P<slot>[a-z]+(?: [a-z]+)?) the (?:[a-z]+ )?id of (?P<ent>" + ENT + r") "
                + UQ), lambda m: f"{m['prep']} {_a(m['ent'])} you have not read"),
    # 'a gmail message the id of gmail message X' -> 'a gmail message you have not read'
    (re.compile(r"\b(?:a|an) (?P<ent>" + ENT + r") the (?:[a-z]+ )?id of (?P=ent) " + UQ),
     lambda m: f"{_a(m['ent'])} you have not read"),
    # 'worksheet the id of google sheets worksheet X (... returns it)' -> ', in a google sheets worksheet you have not read (...),'
    (re.compile(r" worksheet the id of (?P<ent>" + ENT + r") " + UQ + r"(?P<par> \([^()]*\))?(?P<col>(?= column| required))?"),
     lambda m: f", in {_a(m['ent'])} you have not read{m['par'] or ''}" + ("," if m["col"] is not None else "")),
    # 'the recipient of gmail message X' -> 'the recipient of a gmail message you have not read'
    (re.compile(r"\bthe (?P<field>[a-z]+(?: [a-z]+)?) of (?P<ent>" + ENT + r") " + UQ),
     lambda m: f"the {m['field']} of {_a(m['ent'])} you have not read"),
    (re.compile(r" worksheet " + UQ + r"(?= column| required)"), ", in a worksheet you have not read,"),
    (re.compile(r" worksheet " + UQ), ", in a worksheet you have not read"),
    (re.compile(r" labels message " + UQ), " labels on a message you have not read"),
    # identity slots: 'on ticket X', 'in thread X', 'for account X', 'to profile X', '(in channel X'
    (re.compile(r"\b(?P<prep>on|in|for|to|from) (?P<ent>(?:[a-z0-9]+ ){0,2}" + ENTITY_NOUNS + r") " + UQ),
     lambda m: f"{m['prep']} {_a(m['ent'])} you have not read"),
    (re.compile(r" company id " + UQ), " for a company you have not read"),
    (re.compile(r"\b(?P<ent>customer tag) for " + UQ), lambda m: f"{m['ent']} for a customer you have not read"),
    (re.compile(r"\b(?P<ent>salesforce lead|calendly invitee|docusign envelope signer) for " + UQ),
     lambda m: f"{m['ent']} you have not read"),
    (re.compile(r"\bdocusign envelope for " + UQ), "docusign envelope for a recipient you have not read"),
    (re.compile(r" row (?P<n>\d+) \(" + UQ + r"\)"), lambda m: f" row {m['n']}"),
    # the engine's must-stay check: 'its group had to stay X' -> 'its group had to stay unchanged'
    (re.compile(r"; its (?P<f>[a-z][a-z ]{0,25}?) had to stay " + UQ), lambda m: f"; its {m['f']} had to stay unchanged"),
    (re.compile(r'\bIts cell "(?P<f>[^"]+)" = ' + UQ), lambda m: f'Its cell "{m["f"]}" held a value you have not read'),
    (re.compile(r"; the task requires one with (?P<f>[a-z][a-z ]{0,20}?) not containing " + UQ),
     lambda m: f"; the task requires one whose {m['f']} leaves out a value you have not read"),
    (re.compile(r"; (?P<f>[a-z][a-z ]{0,20}?) containing " + UQ), lambda m: f", with {m['f']} containing {U}"),
    # the recipient written as 'with to X': said as 'to X' (say_unread's recipient rule)
    (re.compile(r" with to " + UQ), " to " + U),
    # a parenthesised 'with <field> X' clause becomes a plain clause (say_unread then words it)
    (re.compile(r" \((?P<c>with [a-z][a-z ]{0,30}? (?:containing )?" + UQ + r")\)"), lambda m: f", {m['c']}"),
]
# 'with body containing "X"; the task requires one with body not containing "X"' (the same value): said as the field
# holding a value the attempt has not read, and the task requiring one whose field leaves that value out
REQ_LEAVE_SAME = re.compile(r'(?:,? with |, )(?P<f>[a-z][a-z ]{0,20}?) containing "(?P<v>[^"\n]+)"; the task requires one with '
                            r'(?P=f) not containing "(?P=v)"')
TAIL = r"(?P<tail> in (?:that|those|these|each of these) [^.;]+)"
FORBID_C = [
    (re.compile(r"This task forbids cells containing (?P<f>[A-Za-z][A-Za-z0-9 %/_-]{0,30}?) = " + UQ + TAIL),
     lambda m: f"This task forbids the {m['f']} cell shown{m['tail']}"),
    (re.compile(r'This task forbids cell "(?P<f>[^"]+)" = ' + UQ + TAIL),
     lambda m: f'This task forbids the cell "{m["f"]}" shown{m["tail"]}'),
]
FVAL = r'(?:(?:containing |= )?(?:"[^"\n]*"|' + UQ + r"))"
FORBID_MULTI = re.compile(r"This task forbids (?P<items>[a-z][a-z ]{0,30}? " + FVAL + r"(?:(?:, | and )[a-z][a-z ]{0,30}? "
                          + FVAL + r")+)" + TAIL)
SEP = r"(?:, or |, and |, | or | and )"
ITEM = r'(?:"[^"\n]*"|' + UQ + r")(?: \(\d+ [a-z]+\))?"
LIST = re.compile(r"(?P<pre>\b(?:without|not any of|none of|any of) )?(?P<body>" + ITEM + r"(?:" + SEP + ITEM + r")+)")
FIELD_WORDS = {"body plain": "body", "in channel": "channel", "for email": "email", "due on": "due date", "due": "due date",
               "description containing": "description", "to": "recipient", "from": "sender"}
PLURAL_FIELDS = ("tags", "labels", "attendees", "values", "recipients")


def _forbid_multi(m):
    fields = re.findall(r"(?:^|, | and )([a-z][a-z ]{0,30}?) " + FVAL, m["items"])
    if U not in m["items"] or not fields:
        return m.group(0)
    return f"This task forbids the {' and '.join(dict.fromkeys(f.strip() for f in fields))} shown{m['tail']}"


def _list(m):
    pre, body = m["pre"] or "", m["body"]
    items = re.findall(ITEM, body)
    hidden = sum(1 for it in items if it.startswith(U))
    if not hidden:
        return m.group(0)
    shown = [it for it in items if not it.startswith(U)]
    if not shown:
        return pre + U  # several hidden values in one slot: one clause says a value the attempt has not read
    if re.search(r"whose [a-z ]+ also (?:needs|need|held) values? you have not read", m.string[m.end():]):
        return pre + (", ".join(shown[:-1]) + " and " + shown[-1] if len(shown) > 1 else shown[0])
    seps = re.findall(SEP, body)
    # the list's own joining word; a comma-only list joins with "or" after a negation ("without", "not any of") and
    # with "and" otherwise ("containing A, B" needs both), so the count never reads as a name for the item before it
    last = (" or " if any("or" in s for s in seps) else " and " if any("and" in s for s in seps)
            else " or " if pre else " and ")
    return pre + ", ".join(shown) + last + ("a value you have not read" if hidden == 1
                                            else f"{hidden} values you have not read")


def _forbid_rules(text):
    """Forbid sentences that hide a value say which field is forbidden, as the engine's hidden form does ("This task
    forbids the text shown in that slack message"): cells, a quoted cell, several fields, then FORBID_FIELD(_QUOTED)."""
    before = text
    for pat, repl in FORBID_C:
        text = pat.sub(repl, text)
    text = FORBID_MULTI.sub(_forbid_multi, text)
    if text == before:
        text = FORBID_FIELD.sub(lambda m: f"This task forbids the {m['f'].strip()} shown{m['tail']}", text)
    if text == before:
        text = FORBID_FIELD_QUOTED.sub(lambda m: f"This task forbids the {m['f'].strip()} shown{m['tail']}", text)
    return text.replace("This task forbids the containing shown", "This task forbids the text shown")


def _c_rules(text):
    text = LIST.sub(_list, text)
    for pat, repl in C_RULES:
        text = pat.sub(repl, text)
    return re.sub(r",(?=\s*(?:[,.;]|\(still missing\)|$))", "", text)


def _field_words(text):
    for a, b in FIELD_WORDS.items():
        text = re.sub(rf"\bwhose {a} (?=(?:also )?(?:needs|need|held|is|was|has|had)\b)", f"whose {b} ", text)
        text = re.sub(rf"(?<=\bthe )((?:[a-z ]+? and )*){a}(?=(?: and [a-z ]+?)* shown\b)", rf"\g<1>{b}", text)
        text = re.sub(rf"(?<= and ){a}(?=(?: and [a-z ]+?)* shown\b)", b, text)
    for f in PLURAL_FIELDS:
        text = re.sub(rf"\bwhose {f} (is|was) one you have not read",
                      lambda m: f"whose {f} {'are' if m[1] == 'is' else 'were'} ones you have not read", text)
    return text


OK_LEFT = re.compile(r'(?:whose (?P<f>[A-Za-z0-9 %/_"-]{1,40}?) (?:also )?(?:needs|need|held|is|was|were|are|has to leave out'
                     r'|had to leave out|leaves out)'
                     r'|(?:the|some) details are|\bIts cell "[^"\n]+" held'
                     r'|"[^"\n]*"(?:, | or | and ))\s*$')
BAD_FIELD = re.compile(r"^(?:in|for|to|on|with|from|at|by|of|its|the)\b|^cells [A-Z]|\b(?:containing|plain|had|stay|its|task|requires)\b|=")
GARBLE_C = re.compile(r"\(\s*\)|\bor\)|\(with [a-z ]+ or\b|,\s*,|\{[^}]*you have not read|\bwith \(|\bwhose its\b"
                      r"|\bor or\b|\band and\b|\.;|; [a-z]+(?: [a-z]+)?, whose\b|\bwhose the\b|\bwhose cells [A-Z]")
PHRASE_C = re.compile(r"\b(?:a value|\d+ values|values|ones?) you have not read")


def unreadable(text):
    """The problems of a finished line (empty when readable): a hidden-value phrase outside its own clause, a field slot
    with non-field words, or a known garble."""
    out = [f"garble: {m.group(0)}" for m in GARBLE_C.finditer(text)]
    for m in PHRASE_C.finditer(text):
        left = OK_LEFT.search(text[max(0, m.start() - 80):m.start()])
        if not left:
            out.append(f"in place: ...{text[max(0, m.start() - 40):m.end()]}")
        elif left.group("f") and BAD_FIELD.search(left.group("f")):
            out.append(f"field words: whose {left.group('f')}")
    return out


def _redact_line(line, seen, counts, required=None):
    """Only values are redacted. A quoted name right after cell/column/field/property/key/header is a schema name, not a
    value (the engine's leak detector skips schema-naming keys the same way), and stays. `required` (R-c2, v13): when
    given, only values in it can be hidden (values the feedback also needs written); None = R-c (every unseen value)."""
    schema = {m.group(1) for m in SCHEMA_SLOT.finditer(line)}
    hidden = [q for q in QUOTED.findall(line) if q not in schema and not seen(q) and (required is None or q in required)]
    if not hidden:
        return line
    counts["lines_redacted"] += 1
    counts["values_redacted"] += len(hidden)
    new = REQ_LEAVE_SAME.sub(lambda m: (f", whose {m['f']} held a value you have not read; the task requires one whose "
                                        f"{m['f']} leaves that value out") if m["v"] in hidden else m.group(0), line)
    for q in sorted(set(hidden), key=len, reverse=True):
        new = new.replace(f'"{q}"', U)
    new = QUALIFIER_U.sub("", new)
    before = new
    new = _forbid_rules(new)
    counts["forbid_sentences_reworded"] += new != before
    before_c = new
    new = _c_rules(new)
    counts["lines_c_rules"] += new != before_c
    new = _post(_field_words(F.say_unread(new)))
    if U in new or any(_hard(x) for x in re.split(r"(?<=[.;])\s+", new)) or unreadable(new):
        counts["plain_lines"] += 1  # tier 2: no clean wording for this line
        return _plain_line(line, hidden)
    return new


POST_RULES = [  # V4's own in-place identity slots (values the whole attempt never saw), said as the C rules say them
    (re.compile(r"\bwhose cells (?P<f>[A-Z][A-Za-z0-9 %/_-]{0,30}?) (?P<v>is|was|needs|need|held)\b"),
     lambda m: f"whose {m['f']} cell {'needs' if m['v'] == 'need' else m['v']}"),
    (re.compile(r"\b(?P<ent>salesforce lead|calendly invitee|docusign envelope signer) for a value you have not read"),
     lambda m: f"{m['ent']} you have not read"),
    (re.compile(r"\bdocusign envelope for a value you have not read"), "docusign envelope for a recipient you have not read"),
    (re.compile(r"\bcustomer tag for a value you have not read"), "customer tag for a customer you have not read"),
]


def _post(line):
    for pat, repl in POST_RULES:
        line = pat.sub(repl, line)
    return line


# ------------------------------------------------------------------ v13 (21 Nov, decisions D1-9/D1-10): R-c2
# zcc1 (Z4C on H1, v11) solved 30.7% against V5's 36.6%. Where V5 showed a note at the point and R-c hid values, Z4C
# lost 30 points (37 points); R-c had hidden the content of the very action being corrected (the student's own forbidden
# write at the point: "This task forbids the text shown in that slack message"), which a correct continuation never
# writes. V5's only copies (7 of 291 solves) were target values that the feedback also asks to be written ("Still
# needed: a slack message in channel X"). R-c2: a value unseen at the point is hidden only when the feedback also needs it
# written, i.e. it appears in a still-needed item outside a negation; every other value (forbidden values, kept values,
# the rest of what the student did) is said as the engine says it (observed or written somewhere in the attempt, F14.6).
STILL_PREFIXES = ("Still needed", "Still not done at this point")
NEGATED_VALUES = re.compile(r'\b(?:not containing|without|not any of|none of|leaves? out|not)\s+((?:"[^"\n]*"(?:,? (?:or|and) |, )?)+)')


def required_values(block):
    """Quoted values the feedback needs written: those of still-needed lines, except values only under a negation there
    ('with body not containing "X"' names a value to leave out). Schema names are not values (as in _redact_line)."""
    out = set()
    for ln in block.split("\n"):
        if not ln.startswith(STILL_PREFIXES):
            continue
        schema = {m.group(1) for m in SCHEMA_SLOT.finditer(ln)}
        negated = [(m.start(1), m.end(1)) for m in NEGATED_VALUES.finditer(ln)]
        for m in QUOTED.finditer(ln):
            if m.group(1) in schema or any(a <= m.start() and m.end() <= b for a, b in negated):
                continue
            out.add(m.group(1))
    return out


# R-c2 names the content next to a hidden target ('in a channel you have not read with text containing "X"'); a comma
# keeps the clauses apart for a 4B reader (F14.3). R-c2 designs only, so the measured v12 texts stay byte-identical.
UNREAD_WITH = re.compile(r"(\b(?:a|an) [a-z0-9 ]{1,60}? you have not read) (with )")


def redact(first_message, seen, required_only=False):
    """R-c (default) or R-c2 (`required_only`: only values the feedback also needs written can be hidden)."""
    prompt, block = F.split_feedback(first_message)
    required = required_values(block) if required_only else None
    counts, problems = Counter(), []
    lines = []
    for ln in block.split("\n"):
        new = _post(_redact_line(ln, seen, counts, required)) if ln.strip() else ln
        if required_only and new.strip():
            new = UNREAD_WITH.sub(r"\1, \2", new)
        if U in new:
            problems.append(f"stand-in left: {new[:200]}")
        bad = unreadable(new) if new.strip() else []
        if bad:
            problems.append(f"unreadable ({bad[0]}): {new[:200]}")
        lines.append(new)
    # lines made identical by the redaction: a still-needed line (each one a separate required item) is said once with
    # its count; any other line (the same message or call, its hidden values now said as "shown") is said once
    count = Counter(l for l in lines if l.strip())
    out, said = [], set()
    for l in lines:
        if l.strip() and count[l] > 1:
            if l in said:
                continue
            said.add(l)
            if l.startswith(STILL):
                counts["still_lines_merged_with_count"] += count[l]
                l = f"Still needed, {count[l]} of these: " + l[len(STILL):]
            else:
                counts["identical_lines_said_once"] += count[l]
        out.append(l)
    return f"{prompt}{F.OLD_HEAD}\n\n" + "\n".join(out) + f"\n\n{F.OLD_TAIL}", counts, problems


# ------------------------------------------------------------------ live use (training): the cut and the per-point text
READ_VERBS = {"find", "get", "list", "lookup", "query", "search", "project", "fetch", "read", "retrieve"}
MULTI_APPS = ("google_sheets", "google_drive", "google_ads", "google_calendar", "zoho_desk", "linkedin_leadgen_forms")
NEGATED = re.compile(r"(^|_)not(_|$)")


def app_of(name):  # = weekend/phaseA/a1_anatomy.app_of
    for multi in MULTI_APPS:
        if name.startswith(multi + "_"):
            return multi
    return name.split("_", 1)[0]


def is_write(tool):  # = a1_anatomy.is_write
    return not (set(tool.split("_")[len(app_of(tool).split("_")):]) & READ_VERBS)


def is_must_not(check_type):  # = a1_anatomy.is_must_not
    return bool(NEGATED.search(check_type)) and "with_body_not_contains" not in check_type


def work_start_turn(record):
    """= weekend/phaseA/a3_prepare.work_start_turn: the first assistant turn that wrote to the app of a failed required
    check, or None."""
    from sdpo_a3 import assistant_path

    checks = record["traces"][0]["info"]["automationbench"]["assertions"]
    apps = {app_of(c["type"]) for c in checks if not c.get("excluded") and not c.get("passed")
            and not is_must_not(c["type"])}
    turn = 0
    for m in assistant_path(record):
        if m.get("role") != "assistant":
            continue
        turn += 1
        if any(app_of(c["name"]) in apps and is_write(c["name"]) for c in m.get("tool_calls") or []):
            return turn
    return None


def visibility_turn(engine_context, record):
    """The R-c cut: min(the engine's point, the work start), or None (whole attempt) when neither exists."""
    from sdpo_a3 import point_turn

    located = [x for x in (point_turn(engine_context), work_start_turn(record)) if x is not None]
    return min(located) if located else None


def p_frame(first_message):
    """P's framing (15 Nov versions N/P: the lowest feedback-mention rate measured): V5's heading and closing replaced by
    P's, and nothing else (checked)."""
    if first_message.count(F.OLD_HEAD) != 1 or not first_message.rstrip().endswith(F.OLD_TAIL):
        raise ValueError("not a V5-framed first message")
    body = first_message.rstrip()
    out = body[: -len(F.OLD_TAIL)].replace(F.OLD_HEAD, F.NEW_HEAD) + F.NEW_TAIL + first_message[len(body):]
    back = out.replace(F.NEW_HEAD, F.OLD_HEAD)
    back = back.rstrip()[: -len(F.NEW_TAIL)] + F.OLD_TAIL + back[len(back.rstrip()):]
    if back != first_message:
        raise ValueError("P framing changed more than the heading and closing")
    return out


def zc_live(design, engine_context, shown_context, record, chain):
    """Z3C / Z4C (and Z4CP = Z4C with P's framing) for one failed attempt, live: feedback_z's Z3/Z4 text, then redact()
    with the engine's own observed / written test on what the attempt had seen before the cut."""
    import feedback_z
    from abce.record.episode import Episode

    if design not in ("Z3C", "Z4C", "Z4CP"):
        raise ValueError(design)
    if design == "Z4CP":
        z = zc_live("Z4C", engine_context, shown_context, record, chain)
        return {**z, "first_message": p_frame(z["first_message"])}
    base = feedback_z.z_live(design[:2], engine_context, shown_context, record, chain)
    episode = Episode(record=record)
    tool_defs = record["traces"][0].get("tools")
    if not tool_defs:
        raise ValueError("R-c: the attempt's tool definitions are missing (the seen test needs them)")
    seen = seen_test(episode.messages, episode.task_data.get("zapier_tools"), visibility_turn(engine_context, record),
                     tool_defs)
    text, _counts, problems = redact(base["first_message"], seen)
    if problems:
        raise ValueError(f"R-c: {problems[0]}")
    return {**base, "first_message": text}


DESIGNS_Y = ("Y1", "Y2", "Y1P", "Y2P", "Y1PQ", "Y2PQ")
STILL_MERGED = re.compile(r"^Still needed, \d+ of these: ")


def y_pointer_notes(first_message, v5_notes):
    """Z2's true pointer notes (feedback_z.z2_notes: V5's note turns, each pointer checked against the summary), computed
    on the finished Y1 lines, so every pointer is true of the text the teacher reads (a merged still-needed line still
    lists still-needed work)."""
    import feedback_z

    lines = [STILL_MERGED.sub("Still needed: ", ln) for ln in feedback_z.split(first_message)[1]]
    return {int(k): v for k, v in feedback_z.z2_notes(lines, {str(k): v for k, v in (v5_notes or {}).items()}).items()}


def yc_live(design, engine_context, shown_context, record, chain):
    """v13 (decision D1-10), live: Y1 = Z4 with R-c2 (a value unseen at the point is hidden only when the feedback also
    needs it written); Y2 = Y1 with true pointer notes at V5's note turns (said once: a note points to its summary line
    and repeats nothing); P = P's framing (heading and closing only); a trailing Q = the measured quiet instruction
    (sdpo_a3.QUIET, T-A2q 14 Nov: feedback phrases 516 -> 340) after the closing line, as T-A2q appends it."""
    import feedback_z
    from abce.record.episode import Episode

    if design not in DESIGNS_Y:
        raise ValueError(design)
    base = feedback_z.z_live("Z4", engine_context, shown_context, record, chain)
    episode = Episode(record=record)
    tool_defs = record["traces"][0].get("tools")
    if not tool_defs:
        raise ValueError("R-c2: the attempt's tool definitions are missing (the seen test needs them)")
    seen = seen_test(episode.messages, episode.task_data.get("zapier_tools"), visibility_turn(engine_context, record),
                     tool_defs)
    text, _counts, problems = redact(base["first_message"], seen, required_only=True)
    if problems:
        raise ValueError(f"R-c2: {problems[0]}")
    out = {**base, "first_message": text}
    if design[:2] == "Y2":
        v5, _first = feedback_z.v5_and_v5f(engine_context, shown_context)
        out["notes"] = y_pointer_notes(text, v5.get("notes"))
    if "P" in design[2:]:
        out["first_message"] = p_frame(out["first_message"])
    if design.endswith("Q"):
        from sdpo_a3 import QUIET

        out["first_message"] = out["first_message"] + "\n\n" + QUIET
    return out


# ------------------------------------------------------------------ v14 (21 Nov, round 5): R-d and R-n (designs S1, S2)
# Round 4 (x4 ye1: Y2PQ on H1) solved 25.8% against V5's 36.6% and the old feedback's 33.5%. Per point
# (weekend/sdpo-close/phase1/v13/attrib/attrib-v5-y2pq.json): where R-c2 hid a target (the record, channel, recipient
# or message to act on), 41 points fell from 43.9% (V5) to 17.1%; where it hid only content values, 16 points fell about
# 3 points. R-d keeps every value the engine names (observed or written in the attempt, F14.6), and for a value the
# feedback needs written that the student had not read at the point it says once, at its first mention, where the
# attempt read it: '"X" (read it with <tool> first)'. The teacher then knows what to act on and where the data is, and a
# continuation that writes X without reading it (B4's copy) is told not to. R-n (F6, F14.2): the facts about the cut
# turn (the summary lines "At turn t, ...") are said once, in a note at that turn worded as V5 words its notes, and leave
# the summary; every other fact stays in the summary, once. Only that note is ever read: training stops at the engine's
# point and a continuation starts at the cut, so a note at a later turn is never in the teacher's context.
POINTER = " (read it with {tool} first)"
ENGINE_POINTER = re.compile(r"^ \([a-z0-9_]+(?: or [a-z0-9_]+)* (?:returns|lists) (?:it|them)\)")
NOTE_HEAD = "Note on your next action: "
ONLY_NOTE = "Its mistake is noted at the step where it happened."


def where_shown(episode_messages, tool_names, tool_defs=None):
    """R-d: where(value, from_turn) -> the name of the reading tool (is_write False) whose result first showed the value
    at a turn >= from_turn (the engine's observed test, message by message, then the plain text of the results), or None
    when no reading tool's result did (the attempt only produced it itself) or the value is under 4 characters."""
    import json

    from abce.record.observed import Corpus, normalize, value_forms

    docs = [{"role": "user", "content": "\n".join(tool_doc_strings(tool_defs))}] if tool_defs else []
    names = tuple(tool_names or ())
    msgs = list(episode_messages)
    starts = [i for i, m in enumerate(msgs) if (m or {}).get("role") == "assistant"]
    cache = {}

    def corpus(upto):
        if upto not in cache:
            cache[upto] = Corpus(docs + msgs[:upto], names)
        return cache[upto]

    def results(from_turn):  # results of tools that read data; a write tool's result shows what the attempt wrote
        for s in range(max(1, from_turn or 1), len(starts) + 1):
            end = starts[s] if s < len(starts) else len(msgs)
            for j in range(starts[s - 1] + 1, end):
                if (msgs[j] or {}).get("role") == "tool" and msgs[j].get("name") and not is_write(msgs[j]["name"]):
                    yield j

    def text_of(j):
        content = msgs[j].get("content") or ""
        return normalize(content if isinstance(content, str) else json.dumps(content))

    def shows(j, forms):
        """the result's text holds the value (the engine's own test: anywhere in the normalised text)"""
        text = text_of(j)
        return any(f in text for f in forms)

    def where(value, from_turn):
        # a value under 4 characters ("3", "hot", "VP") turns up inside names, ids, counts and dates of almost any result,
        # so where it first appears says nothing about where it comes from: it gets no pointer (it stays hidden)
        forms = [f for f in value_forms(value) if len(f) >= 4]
        if not forms:
            return None
        for j in results(from_turn):
            if corpus(j + 1).is_observed(value) and not corpus(j).is_observed(value):
                return msgs[j].get("name") or "a tool"
        # the engine drops from a result any text the attempt itself wrote earlier (its echo rule), so a value the attempt
        # wrote in a call of the same or an earlier turn is never 'observed' there; the result that shows it is then the
        # first one whose text holds it
        for j in results(from_turn):
            if shows(j, forms):
                return msgs[j].get("name") or "a tool"
        return None
    return where


def said_plainly(line):
    """The wording redact() gives every line it keeps (R-c2): V4's in-place identity slots said as the C rules say them
    (_post), and a comma after 'a <thing> you have not read' before 'with'."""
    return UNREAD_WITH.sub(r"\1, \2", _post(line)) if line.strip() else line


def name_where(first_message, seen, where, cut):
    """R-d on a V5-framed first message. Each value the feedback needs written (required_values) that was unseen at the
    cut is said as the engine says it and gets, once, at its first mention in a still-needed line, where the attempt
    read it (one pointer per tool in a line: a second value of the same tool there is read by the same call); a value
    right before the engine's own pointer ('(X returns it)') keeps that pointer only. A needed value that no tool
    result ever showed (the attempt only produced it itself, later) and that the engine does not point to has nothing
    to point to: it stays hidden as R-c2 hides it (redact(..., required_only=True) on exactly those values), so the
    teacher has to produce it as the student did. -> (text, counts, problems)."""
    prompt, block = F.split_feedback(first_message)
    counts, problems, said = Counter(), [], set()
    required = required_values(block)
    pointed = {m.group(1) for ln in block.split("\n") for m in QUOTED.finditer(ln) if ENGINE_POINTER.match(ln[m.end():])}
    alone = {q for q in required if not seen(q) and q not in pointed and where(q, cut) is None}
    if alone:
        first_message, c0, p0 = redact(first_message, lambda v: seen(v) or v not in alone, required_only=True)
        counts.update({f"hidden_{k}": v for k, v in c0.items()})
        counts["values_written_only_hidden"] += len(alone)
        problems += p0
        prompt, block = F.split_feedback(first_message)
        required = required_values(block)
    out = []
    for ln in block.split("\n"):
        ln = said_plainly(ln)
        if not ln.startswith(STILL_PREFIXES):  # where to read a value is said where the value is needed written
            out.append(ln)
            continue
        schema = {m.group(1) for m in SCHEMA_SLOT.finditer(ln)}
        negated = [(m.start(1), m.end(1)) for m in NEGATED_VALUES.finditer(ln)]
        pieces, last, tools_here = [], 0, set()
        for m in QUOTED.finditer(ln):
            q = m.group(1)
            if q in schema or q not in required or q in said or seen(q) \
                    or any(a <= m.start() and m.end() <= b for a, b in negated):
                continue
            said.add(q)
            counts["values_unseen_named"] += 1
            if ENGINE_POINTER.match(ln[m.end():]):
                counts["engine_pointer_kept"] += 1
                continue
            tool = where(q, cut)
            if tool is None:
                problems.append(f"R-d: no tool result shows {q!r} after hiding: {ln[:160]}")
                continue
            if tool in tools_here:  # one pointer per tool in a line (the line already says to read that tool)
                counts["values_under_a_pointer_of_the_line"] += 1
                continue
            tools_here.add(tool)
            pieces.append(ln[last:m.end()] + POINTER.format(tool=tool))
            last = m.end()
            counts["pointers"] += 1
        out.append("".join(pieces) + ln[last:])
    return f"{prompt}{F.OLD_HEAD}\n\n" + "\n".join(out) + f"\n\n{F.OLD_TAIL}", counts, problems


def _sentences(text):
    """Sentences outside quoted values (= engine-req/audit_text.sentences, B5's own split)."""
    quoted = re.findall(r'"[^"]*"', text)
    masked = re.sub(r'"[^"]*"', "\x00", text)
    out = []
    for piece in re.split(r"(?<=[.;])\s+|\n+|\(\d+\)\s*", masked):
        if piece.strip():
            n = piece.count("\x00")
            restored = piece
            for q in quoted[:n]:
                restored = restored.replace("\x00", q, 1)
            quoted = quoted[n:]
            out.append(restored.strip())
    return out


def _norm(text):
    return re.sub(r"\s+", " ", text).strip().rstrip(".;").lower()


TURN_HEAD = re.compile(r"^At turn \d+, ")


def note_repeats(note, first_message):
    """B5's two repeat tests on a note (= audit_text 'repeated sentence in a note' and audit_repeats 'a note repeating
    the summary': a part of the note that equals or is contained in a line of the first message)."""
    body = note[len(NOTE_HEAD):] if note.startswith(NOTE_HEAD) else note
    if any(n > 1 for n in Counter(_sentences(note)).values()):
        return True
    lines = {_norm(TURN_HEAD.sub("", ln)) for ln in first_message.splitlines() if ln.strip()}
    parts = [_norm(x) for x in re.split(r"(?<=[.;])\s+", body) if x.strip()]
    return any(x in lines or any(x and x in ln for ln in lines) for x in parts)


def at_turn_note(first_message, cut):
    """R-n: the summary lines about the cut turn ('At turn t, ...') leave the summary and are said once, in a note at
    turn t ('Note on your next action: ...', V5's note wording). When that note would still repeat something (a sentence
    twice inside it, or a part that a summary line also says: two facts of one record at two turns), nothing moves, so
    nothing is ever said twice (B5). -> (first_message, notes)."""
    if cut is None:
        return first_message, {}
    prompt, block = F.split_feedback(first_message)
    head = f"At turn {cut}, "
    lines = block.split("\n")
    moved = [ln[len(head):] for ln in lines if ln.startswith(head)]
    if not moved:
        return first_message, {}
    kept = [ln for ln in lines if not ln.startswith(head)]
    body = "\n".join(kept).strip("\n") or ONLY_NOTE
    text = f"{prompt}{F.OLD_HEAD}\n\n{body}\n\n{F.OLD_TAIL}"
    note = NOTE_HEAD + " ".join(moved)
    if note_repeats(note, text):
        return first_message, {}
    return text, {int(cut): note}


DESIGNS_S = ("S1", "S1P", "S1PQ", "S1Q", "S2", "S2P", "S2PQ", "S2Q", "S1PR", "S1R", "S2PR", "S2R")
# U (round 6, 21 Nov): the same texts as the design without U, its note at the cut turn given as its own user message
# after that turn's tool results (note_role "user", read by sdpo_ab_runner.teacher_sequence). Inside the tool result
# S1PR's teacher read the note-only fact as the tool's own output (x5 s5a: "it says helpscout_add_note was already
# called... That's odd because I didn't call that function yet"). Only S1 designs have a note.
DESIGNS_U = ("S1PU", "S1PRU")
# H (round 7 candidate, 21 Nov): a value the attempt has not seen is said as one "no tool has shown" instead of one "you
# have not read". Every design's excess doubt over the old feedback sits at points with that phrase (x5: S1PR 12.31 vs
# 11.96, S1R 12.87 vs 11.96; elsewhere at or below the old feedback), and the teachers ask what it means ("Let me think
# about what 'text needs a value you have not read (still missing)' means"). Same length and the same one negation, so
# sentence lengths and the double-negative audit are unchanged; the old phrase never sits inside a quoted value and the
# new one appears nowhere else, so the swap is exact and reversible (each H text maps back to its base text).
UNSHOWN_OLD, UNSHOWN_NEW = "you have not read", "no tool has shown"
DESIGNS_H = ("S1PUH",)


def say_unshown(text):
    """H: the hidden-value phrase said as what no tool has shown (an exact swap outside quoted values)."""
    if text is None:
        return None
    if UNSHOWN_NEW in text:
        raise ValueError("H: the text already holds the new phrase; the swap would not be reversible")
    if any(UNSHOWN_OLD in q for q in re.findall(r'"[^"\n]*"', text)):
        raise ValueError("H: the hidden-value phrase sits inside a quoted value")
    return text.replace(UNSHOWN_OLD, UNSHOWN_NEW)
# R (round 5): the quiet instruction without "reason only from the task request and the data your tools return", which
# made Y2PQ's teacher set the feedback aside (x4 ye1, finance.xero_bill_entry: "The earlier notes about 'still missing'
# messages might be from a different interpretation or test case", then only the summary the task text names; on the 88
# points where only the framing differed from V5, Y2PQ solved 38.1% against V5's 43.2%, its continuations ending
# earlier). It keeps Q's "do not mention" sentence and asks the teacher to act on the feedback.
QUIET_R_P = ("Act on these results. Do not mention them, any feedback, any check or any earlier attempt in your "
             "reasoning, messages or replies.")
QUIET_R_V5 = "Act on this feedback. Do not mention it, any check or any earlier attempt in your reasoning, messages or replies."
# Round 7 (21 Nov, decision D1-15), on the S1 text with no quiet instruction (R raised mentions with P's framing and did
# not lower them with V5's: S1PR 3.22, S1R 4.49 vs V5 4.36):
# F = V5's framing with one noun changed, "feedback" -> "results". With V5's framing half of all mentions cite the
#   noun ("the feedback says/mentions/...", S1R 2.4 of 4.36 per 1,000 words, in first and later replies alike); P's
#   heading noun "results" is cited about 0.03 per 1,000. Every citation of F's own words is already counted by B4
#   ("from your unsuccessful", "earlier attempt", "the results say/show", "correctly solve the original question").
# T = the cut note first says that the earlier attempt went wrong at this step. The S1 note reports that attempt's action
#   in the past tense at a step the teacher has not taken yet, and teachers then work out whose action it was ("The note
#   says '... produced ...' - this suggests I DID call it before"; S1PU first replies at noted points: 5.16 mentions per
#   1,000 words vs 3.25 without a note). Every one of the 620 S1 notes states a mistake of that step (a forbidden or
#   changed value, or a tool error), so the sentence is true wherever it is added.
# H and U as above. Each text maps back exactly to its S1 base (F's heading, T's sentence and H's phrase reversed).
F_HEAD = "The following are results from your unsuccessful earlier attempt:"
T_LEAD = "at this step, the earlier attempt went wrong. "
DESIGNS_X = ("S1FH", "S1FTH", "S1PTH", "S1FUH", "S1FTUH", "S1PTUH")


def f_frame(first_message):
    """F's framing: V5's heading with "feedback" said as "results"; the closing kept; nothing else changes (checked)."""
    if first_message.count(F.OLD_HEAD) != 1 or not first_message.rstrip().endswith(F.OLD_TAIL):
        raise ValueError("not a V5-framed first message")
    if F_HEAD in first_message:
        raise ValueError("F: the heading is already present")
    out = first_message.replace(F.OLD_HEAD, F_HEAD)
    if out.replace(F_HEAD, F.OLD_HEAD) != first_message:
        raise ValueError("F framing changed more than the heading")
    return out


def t_note(note):
    """T: an R-n note first says that the earlier attempt went wrong at this step; its facts follow unchanged."""
    if not note.startswith(NOTE_HEAD) or T_LEAD in note:
        raise ValueError("T: not a plain R-n note")
    return NOTE_HEAD + T_LEAD + note[len(NOTE_HEAD):]


def note_says_a_sentence_twice(note):
    """R-n's own rule (a note that would say a sentence twice is not made; nothing moves), tested on the note's sentences
    without its heading. note_repeats counted the heading into the first sentence, so two cut-turn lines opening with
    the same sentence passed (5 contexts, one task's turn 2, found 21 Nov when T put the heading in a sentence of its
    own). Kept apart from note_repeats so every earlier design's text stays byte-identical."""
    body = note[len(NOTE_HEAD):] if note.startswith(NOTE_HEAD) else note
    return any(n > 1 for n in Counter(_sentences(body)).values())


def xc_live(design, engine_context, shown_context, record, chain):
    """Round 7, live: S1 (V5's framing, no quiet instruction; S2, the same text with nothing moved, where R-n's note
    would say a sentence twice), then T on its cut note (if T), F's or P's framing, H on every text, and note_role
    "user" (if U). Names: S1 + P|F + [T] + [U] + H."""
    if design not in DESIGNS_X:
        raise ValueError(design)
    flags = design[2:]
    base = sc_live("S1", engine_context, shown_context, record, chain)
    if any(note_says_a_sentence_twice(v) for v in (base.get("notes") or {}).values()):
        base = sc_live("S2", engine_context, shown_context, record, chain)
    notes = {t: (t_note(v) if "T" in flags else v) for t, v in (base.get("notes") or {}).items()}
    framed = p_frame(base["first_message"]) if flags[0] == "P" else f_frame(base["first_message"])
    out = {**base, "first_message": say_unshown(framed), "notes": {t: say_unshown(v) for t, v in notes.items()}}
    if "U" in flags:
        out["note_role"] = "user"
    return out


# Round 8 (21 Nov, after x7 s8a): one closing line after a round-7 text. Measured cause of the doubt bar's misses [V]:
# every V5-framed design's excess over the old feedback sits in the later replies at points with a hidden value (S1FTUH
# 12.19 vs the old feedback's 11.49 per 1,000 words; 11.69 to 12.73 across six designs; first replies and points without
# a hidden value are at or below the old feedback), where the teachers search longer (S1FTUH 2,673 words per point vs
# 2,261). R's acting instruction brought that slice to the old feedback's length and doubt (S1PRU 2,293 words and 11.69 vs
# S1PU 2,960 and 12.32, the same texts otherwise), but its second sentence names "any feedback, any check or any earlier
# attempt", and the teachers echoed it ("earlier attempt" 0.74 -> 1.15 per 1,000 words). A = R's acting sentence alone;
# M = the same request with the ban said once, naming only "these results" (the heading's own noun). Names: a round-7
# name + A|M; each text is its round-7 text plus "\n\n" + the line (checked, reversible).
ACT_A = "Act on these results."
ACT_M = "Act on these results without mentioning them in your reasoning, messages or replies."
DESIGNS_X8 = tuple(d + s for d in ("S1FTUH", "S1FUH", "S1PTUH") for s in ("A", "M"))


def add_act(first_message, line):
    """A/M: the line after the closing sentence (V5's/F's or P's), once; nothing else changes (checked)."""
    if line not in (ACT_A, ACT_M):
        raise ValueError("not a round-8 line")
    if first_message != first_message.rstrip() or not first_message.endswith((F.OLD_TAIL, F.NEW_TAIL)):
        raise ValueError("A/M: the text does not end with a closing sentence")
    if ACT_A in first_message:  # ACT_M starts with ACT_A's request; either one already there
        raise ValueError("A/M: the line is already present")
    out = first_message + "\n\n" + line
    if out[: -len("\n\n" + line)] != first_message:
        raise ValueError("A/M changed more than the closing line")
    return out


def x8_live(design, engine_context, shown_context, record, chain):
    """Round 8, live: the round-7 text (xc_live) plus A's or M's line after its closing sentence."""
    if design not in DESIGNS_X8:
        raise ValueError(design)
    out = xc_live(design[:-1], engine_context, shown_context, record, chain)
    return {**out, "first_message": add_act(out["first_message"], ACT_A if design.endswith("A") else ACT_M)}


# Round 9 candidate W (22 Nov 00:xx, after x8 s9a). Measured cause [V, weekend/sdpo-close/phase1/v18/w4/
# w4-split-all-designs.txt]: at the 97 H1 points where the engine proves that no record of the task's initial world
# holds the hidden value (feedback_versions.w4_guard), all eight new designs that say it as one "you have not read" or "no
# tool has shown" solve less than the old feedback (12.4-17.5% vs 19.6%) and seven doubt more (12.49-13.54 vs 12.05 per 1,000
# reasoning words; S1PRU 12.08): the teachers search the tools for a value no record holds ("Wait - maybe the 'missing
# value' is something else entirely", finance.xero_vendor_onboard). At the 70 hidden points where a record holds the value
# the same phrase helps (24.3-30.0% vs 17.4%). W says what the engine's finding means for the teacher, where it holds:
# "a value you have to work out" (no record has it written, so it cannot be read; r3's W4 meaning). r3's longer words
# ("a value that is not written in any record ...") pushed 313 more sentences past 25 words on v19's first build (473 vs
# 160; F14.3 short sentences); these are as long as the H phrase they replace and carry no negation. It applies only when
# w_guard holds for the whole context; elsewhere the text is unchanged. Names: a round-7 or round-8 name + W.
W_ONE = "a value you have to work out"
W_MANY = "values you have to work out"
W_PAIRS = ((" needs a value " + UNSHOWN_NEW, " needs " + W_ONE), (" need a value " + UNSHOWN_NEW, " need " + W_ONE),
           (" needs values " + UNSHOWN_NEW, " needs " + W_MANY), (" need values " + UNSHOWN_NEW, " need " + W_MANY))
DESIGNS_X9 = ("S1FUHW", "S1FUHAW")


def say_worked_out(text):
    """W: an H phrase for a needed value said as the engine's finding (an exact swap outside quoted values, reversible)."""
    if text is None:
        return None
    if W_ONE in text or W_MANY in text:
        raise ValueError("W: the text already holds the new phrase")
    if any(UNSHOWN_NEW in q for q in QUOTED.findall(text)):
        raise ValueError("W: the hidden-value phrase sits inside a quoted value")
    out = text
    for old, new in W_PAIRS:
        out = out.replace(old, new)
    back = out
    for old, new in W_PAIRS:
        back = back.replace(new, old)
    if back != text:
        raise ValueError("W: the swap is not reversible")
    return out


def _quoted_all(context):
    return set(QUOTED.findall(context["first_message"] + "\n" + "\n".join((context.get("notes") or {}).values())))


def w_guard(engine_context, shown_context, z4, text):
    """W's statement holds for every value the text hides: the engine's W4 guard (every hidden value of the engine
    context is aligned to its fact and has no value-location line, so the engine found it in no record of the initial
    world) and no value R-d hid (quoted in Z4, the text R-d starts from, and gone from the text) has a value-location
    line either."""
    if not F.w4_guard(engine_context, shown_context):
        return False
    located = {x.get("value") for x in ((shown_context or {}).get("fix_hints") or {}).get("lines", [])
               if x.get("kind") == "value_location"}
    return not ((_quoted_all(z4) - _quoted_all(text)) & located)


def x9_live(design, engine_context, shown_context, record, chain):
    """Round 9 candidate, live: the round-7 or round-8 text, then W on every text where w_guard holds."""
    import feedback_z

    if design not in DESIGNS_X9:
        raise ValueError(design)
    name = design[:-1]
    out = (x8_live if name in DESIGNS_X8 else xc_live)(name, engine_context, shown_context, record, chain)
    z4 = feedback_z.z_live("Z4", engine_context, shown_context, record, chain)
    if not w_guard(engine_context, shown_context, z4, out):
        return out
    return {**out, "first_message": say_worked_out(out["first_message"]),
            "notes": {t: say_worked_out(v) for t, v in (out.get("notes") or {}).items()}}


# N (round 9, 22 Nov 01:xx, decision D1-18): the cut-turn note says the task's rule for the next action in the present
# tense. R-n's note reports, in the past tense, an action the teacher has not taken yet ("gmail_send_email produced ...
# This task forbids that exact action, not every gmail_send_email call."), and S1FUH's teachers stop on it ("Wait, this is
# confusing. Let me re-read the note", hr.asana_compliance_tasks; "But I haven't actually called xero_create_contact yet",
# finance.xero_vendor_onboard). T's sentence ("at this step, the earlier attempt went wrong") lowered first-reply doubt
# there (11.0-11.2 vs 11.6 per 1,000 words) but its words were echoed (mentions 3.02 vs 2.54). N names no attempt: each
# (action, rule) pair of the note becomes one rule sentence with the same tool, verb, values and scope:
#   "<tool> <verb> <thing> with <field>. This task forbids <field> in that <type>[ for V]."
#       -> "This task forbids <tool> <verb-ing> <thing> with <field>[ for V]."
#   "<tool> <verb> <thing>. This task forbids that exact action, not every <tool> call."
#       -> "This task forbids <tool> <verb-ing> <thing>, not every <tool> call."
#   "<tool> <verb> N <things> ... (one each). This task forbids them."
#       -> "This task forbids <tool> <verb-ing> <things> ..."
#   "<tool> <verb> <thing>. This task forbids it[ for V]." -> "This task forbids <tool> <verb-ing> <thing>[ for V]."
# Any other action sentence keeps its words and its place and only leaves the past tense: "<tool> <verb> <thing>." ->
# "<tool> can <verb-base> <thing>." (the rule sentences after it unchanged), and "had to stay" -> "has to stay". A first
# build merged these pairs into one "When <tool> <verb-s> <thing>, this task forbids ..." sentence: 68 sentences went past
# 25 words (up to 51), so it was replaced before any measurement (F14.3).
#   "<tool> <verb> <thing> with <field>; the task requires one with <field'>."
#       -> "This task requires <thing> with <field'>."
# Every other sentence (tool errors, "it for V" facts, unparsed pairs) is kept as it is. present_check proves each rewrite:
# the same quoted values, one rule sentence per rewritten pair, no past-tense action left in it.
N_GERUND = {"produced": "producing", "added": "adding", "updated": "updating", "created": "creating",
            "changed": "changing", "sent": "sending", "posted": "posting", "deleted": "deleting", "removed": "removing"}
N_BASE = {"produced": "produce", "added": "add", "updated": "update", "created": "create", "changed": "change",
          "sent": "send", "posted": "post", "deleted": "delete", "removed": "remove"}
N_TOOL = r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+"
N_ACTION = re.compile(r"^(?P<tool>" + N_TOOL + r") (?P<verb>" + "|".join(N_GERUND) + r") (?P<obj>.+)\.$")
N_REQUIRES = re.compile(r"^(?P<tool>" + N_TOOL + r") (?P<verb>" + "|".join(N_GERUND)
                        + r") (?P<ident>.+?) with (?P<field>[^;]+); the task requires one with (?P<need>.+)\.$")
N_FIELD = re.compile(r'^This task forbids (?P<field>.+) in that (?P<type>[a-z][a-z ]*?)(?P<for> for "[^"\n]*")?\.$')
N_COUNT = re.compile(r"^(?P<n>\d+) (?P<rest>.+) \(one each\)$")
DESIGNS_XN = ("S1FUHWN",)


def _note_sentences(body):
    """Sentences of a note body, split after a full stop outside quoted values; joined back with one space each."""
    quoted = re.findall(r'"[^"\n]*"', body)
    masked = re.sub(r'"[^"\n]*"', "\x00", body)
    out, k = [], 0
    for piece in re.split(r"(?<=\.)\s+", masked):
        n = piece.count("\x00")
        restored = piece
        for q in quoted[k:k + n]:
            restored = restored.replace("\x00", q, 1)
        k += n
        if restored:
            out.append(restored)
    if " ".join(out) != body:
        raise ValueError("N: the note does not split into sentences joined by one space")
    return out


def _present_pair(action, rule):
    """One rule sentence for an (action, rule) pair, or None when the pair is not one of N's shapes."""
    m = N_ACTION.match(action)
    if m:
        tool, verb, obj = m["tool"], m["verb"], m["obj"]
        head = f"This task forbids {tool} {N_GERUND[verb]} "
        if rule == f"This task forbids that exact action, not every {tool} call.":
            return head + obj + f", not every {tool} call."
        if rule == "This task forbids them.":
            c = N_COUNT.match(obj)
            return head + c["rest"] + "." if c else None
        it = re.match(r'^This task forbids it(?P<for> for "[^"\n]*")?\.$', rule)
        if it:
            return head + obj + (it["for"] or "") + "."
        f = N_FIELD.match(rule)
        if f and obj.endswith(" with " + f["field"]) and re.match(r"^an? " + re.escape(f["type"]) + r"\b", obj):
            return head + obj + (f["for"] or "") + "."
    return None


N_QUOTED = re.compile(r'"([^"\n]*)"')  # empty values allowed (an engine line can say 'with value ""'); QUOTED needs one


def _same_values(new, *old):
    return set(N_QUOTED.findall(new)) == set(N_QUOTED.findall(" ".join(old)))


def present_rules(note):
    """N on one R-n note (with or without NOTE_HEAD). A pair is rewritten only when its one rule sentence holds exactly
    the pair's quoted values; anything else is kept as it is, so a note N cannot parse comes back unchanged."""
    if note is None:
        return None
    head = NOTE_HEAD if note.startswith(NOTE_HEAD) else ""
    sents = _note_sentences(note[len(head):])
    out, i = [], 0
    while i < len(sents):
        r = N_REQUIRES.match(sents[i])
        if r:
            new = f"This task requires {r['ident']} with {r['need']}."
            if _same_values(new, sents[i]):
                out.append(new)
                i += 1
                continue
        new = _present_pair(sents[i], sents[i + 1]) if i + 1 < len(sents) else None
        if new and _same_values(new, sents[i], sents[i + 1]):
            out.append(new)
            i += 2
            continue
        out.append(_present_can(sents[i]))
        i += 1
    return head + " ".join(out)


def _present_can(sentence):
    """An action sentence N's merges did not take, out of the past tense with its words kept: "<tool> <verb> ..." ->
    "<tool> can <verb-base> ...", and "had to stay" -> "has to stay"; any other sentence unchanged."""
    m = N_ACTION.match(sentence) or re.match(r"^(?P<tool>" + N_TOOL + r") (?P<verb>" + "|".join(N_GERUND) + r") (?P<obj>.+)$",
                                             sentence)
    if m:
        sentence = f"{m['tool']} can {N_BASE[m['verb']]} " + sentence[len(m["tool"]) + 1 + len(m["verb"]) + 1:]
    return sentence.replace(" had to stay ", " has to stay ")


def _back_can(sentence):
    m = re.match(r"^(?P<tool>" + N_TOOL + r") can (?P<base>" + "|".join(N_BASE.values()) + r") ", sentence)
    if m:
        past = {b: v for v, b in N_BASE.items()}[m["base"]]
        sentence = f"{m['tool']} {past} " + sentence[m.end():]
    return sentence.replace(" has to stay ", " had to stay ")


def present_check(old, new):
    """Problems of one N rewrite: the heading and the quoted values (as a set) kept; every sentence of the new note is
    either a sentence of the old note, unchanged, or a present-tense rule ("This task forbids/requires ...") with no
    past-tense action verb; the number of rules kept."""
    problems = []
    head = NOTE_HEAD if old.startswith(NOTE_HEAD) else ""
    if not new.startswith(head):
        problems.append("heading changed")
    if set(N_QUOTED.findall(old)) != set(N_QUOTED.findall(new)):
        problems.append("quoted values differ")
    old_s = _note_sentences(old[len(head):])
    for s in _note_sentences(new[len(head):]):
        if s in old_s:
            continue
        if s.startswith(("This task forbids ", "This task requires ")):
            if re.search(N_TOOL + r" (?:" + "|".join(N_GERUND) + r")\b", s):
                problems.append(f"past-tense action left: {s[:120]}")
        elif _back_can(s) not in old_s:
            problems.append(f"changed sentence that is neither a rule nor the same words in the present: {s[:120]}")
    rules_old = len(re.findall(r"this task forbids|the task requires one with|had to stay", old, re.I))
    rules_new = len(re.findall(r"this task forbids|this task requires|the task requires one with|had to stay|has to stay",
                               new, re.I))
    if re.search(r"(?:^|\. )" + N_TOOL + r" (?:" + "|".join(N_GERUND) + r")\b", new[len(head):]):
        problems.append("a past-tense action sentence is left")
    if rules_old != rules_new:
        problems.append(f"rules {rules_old} -> {rules_new}")
    return problems


def xn_live(design, engine_context, shown_context, record, chain):
    """Round 9, live: the W text (x9_live), then N on its cut-turn note."""
    if design not in DESIGNS_XN:
        raise ValueError(design)
    out = x9_live(design[:-1], engine_context, shown_context, record, chain)
    notes = {}
    for t, v in (out.get("notes") or {}).items():
        new = present_rules(v)
        problems = present_check(v, new)
        if problems:
            raise ValueError(f"N: {problems[0]}")
        notes[t] = new
    return {**out, "notes": notes}


# L (round 10, 22 Nov, decision D1-21): where the deciding data is. My reading of 20 seeded S1FUHWN solves found 4 that
# act on a line without its reason (a forbidden word renamed, the still-needed list executed with no data, an invented
# reason, a draft called sent); in all 4 the reason sits in records the failed attempt never opened. L names those
# records, from the attempt's own tool results and the task's initial world, without any value the lines do not
# already quote:
#   L2 an app none of whose records the attempt opened: "Never opened: the task's gmail messages (read them with
#      gmail_find_email first)."
#   L1 in an app the attempt opened in part, the unopened records that mention a value the lines quote: "Never opened:
#      3 calendly invitees that mention "EnterpriseCo" or "StandardCo" (read them with calendly_list_invitees first)."
# A record is opened when its longest own text (12 characters or more; else an id-like field) occurs in the request or
# in a tool result of the whole failed attempt (tool results decoded from JSON first, so text matches as stored).
# 22 Nov 04:4x (before any measurement): each line's head is the one that is exactly true of the records it names:
# "Never opened: " when no part of them was ever shown (no own text of 12+ characters, no id-like field), else "Never
# read in full: " (a subject in a listing, or a record queried for its Id and Name, was shown but never its longest text).
DESIGNS_XL = ("S1FUHWNL",)
L_CLOSE = "\n\nCorrectly solve the original question."
L_HEAD = "Never opened: "
L_HEAD_PART = "Never read in full: "
L_HEADS = (L_HEAD, L_HEAD_PART)
L_LONG = 12
L_IDS = ("id", "row_id", "uri", "message_id", "event_id", "ticket_id", "email", "key")
L_MAX_SHARE = 5  # a value that more unopened records of one collection mention says nothing about where the reason is
L_MAX_VALUES = 3
L_MAX_LINES = 3  # at most 2 L2 lines and 2 L1 lines, 3 in all
L_VERBS = ("find", "search", "list", "lookup", "query", "get", "fetch", "read", "retrieve", "project")


def _l_records(state, app=None, path=()):
    """(app, collection, record) for every dict inside a list of the initial world, at any depth ('meta' left out)."""
    if isinstance(state, dict):
        for k, v in state.items():
            if app is None and k == "meta":
                continue
            yield from _l_records(v, app or k, path + (k,))
    elif isinstance(state, list):
        for x in state:
            if isinstance(x, dict):
                yield app, (path[-1] if len(path) > 1 else "records"), x
                yield from _l_records(x, app, path)


def _l_own(rec):
    """A record's own scalar texts (nested dicts included; nested lists are records of their own)."""
    out = []
    for v in rec.values():
        if isinstance(v, dict):
            out += _l_own(v)
        elif isinstance(v, (str, int, float)) and not isinstance(v, bool):
            out.append(str(v))
    return out


def _l_keys(state, acc=None):
    acc = set() if acc is None else acc
    if isinstance(state, dict):
        for k, v in state.items():
            acc.add(str(k).casefold())
            _l_keys(v, acc)
    elif isinstance(state, list):
        for x in state:
            _l_keys(x, acc)
    return acc


def _l_decoded(text, depth=0):
    """A tool result's text plus every string inside it once decoded from JSON (twice at most: a JSON string that holds
    JSON), so a stored text with newlines or quotes matches what the result showed."""
    import json

    out = [text]
    if depth >= 2 or not text.strip().startswith(("{", "[")):
        return out
    try:
        obj = json.loads(text)
    except ValueError:
        return out
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            stack.extend(o.values())
        elif isinstance(o, list):
            stack.extend(o)
        elif isinstance(o, str):
            out += _l_decoded(o, depth + 1)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            out.append(str(o))
    return out


def l_observed(messages):
    """The request and every tool result of the whole failed attempt, decoded and normalised as the engine normalises."""
    import json

    from abce.record.observed import normalize

    parts = []
    for m in messages:
        if (m or {}).get("role") not in ("user", "tool"):
            continue
        c = m.get("content") or ""
        if not isinstance(c, str):
            c = json.dumps(c)
        parts += _l_decoded(c)
    return normalize("\n".join(parts))


def l_opened(rec, observed):
    from abce.record.observed import Corpus, normalize

    texts = sorted((s for s in _l_own(rec) if len(s.strip()) >= L_LONG), key=lambda s: (-len(s), s))
    if texts:
        return normalize(texts[0]) in observed
    ids = [str(rec[k]) for k in L_IDS if isinstance(rec.get(k), (str, int)) and not isinstance(rec.get(k), bool)
           and len(str(rec[k]).strip()) >= 4]
    return any(Corpus._form_in(normalize(i), observed) for i in ids)


def l_part_seen(rec, observed):
    """Some part of a record was shown: an own text of 12+ characters, or an id-like field, occurs in what the attempt
    saw (the lenient reading; a record never opened by l_opened can still have its subject or id shown)."""
    from abce.record.observed import Corpus, normalize

    if any(normalize(s) in observed for s in _l_own(rec) if len(s.strip()) >= L_LONG):
        return True
    ids = [str(rec[k]) for k in L_IDS if isinstance(rec.get(k), (str, int)) and not isinstance(rec.get(k), bool)
           and len(str(rec[k]).strip()) >= 4]
    return any(Corpus._form_in(normalize(i), observed) for i in ids)


def _l_singular(word):
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("sses", "shes", "ches", "xes")):
        return word[:-2]
    return word[:-1] if word.endswith("s") and not word.endswith("ss") else word


def l_noun(app, collection, n):
    """'gmail messages', 'google sheets row', 'calendly invitees': the app's words and the collection's, singular for 1."""
    coll = collection.replace("_", " ").strip() or "records"
    if n == 1:
        coll = " ".join(coll.split()[:-1] + [_l_singular(coll.split()[-1])])
    return f"{app.replace('_', ' ')} {coll}"


def l_tool(app, collection, tool_names):
    """The read tool the line names: a reading tool of the app among the task's tools; one that names the collection
    first, then one that needs no id, then by verb (find, search, list, ...), then the shortest; None when the task has
    none (the student could not read the records)."""
    one = _l_singular(collection.split("_")[-1])
    cands = [t for t in tool_names or () if app_of(t) == app and not is_write(t)]

    def rank(t):
        words = t.split("_")
        verb = min((L_VERBS.index(w) for w in words if w in L_VERBS), default=len(L_VERBS))
        return (0 if (one in words or collection in words) else 1, 1 if "id" in words else 0, verb, len(t), t)
    return min(cands, key=rank) if cands else None


def _l_values(text, notes, state):
    """The quoted values the lines give that can say where a record is: 4 characters or more with a letter; not a schema
    name (a SCHEMA_SLOT quote or any key of the initial world); not one lowercase word (a status or a common word such as
    "sent" names no record: it turns up in many texts). A value the attempt wrote itself stays: writing a name
    is not reading the records that decide the task (the SLA alert named "EnterpriseCo" without opening its invitee)."""
    keys = _l_keys(state)
    out = []
    for t in [text] + list((notes or {}).values()):
        schema = set(SCHEMA_SLOT.findall(t))
        for v in QUOTED.findall(t):
            if (len(v.strip()) >= 4 and re.search(r"[A-Za-z]", v) and v not in schema and v.casefold() not in keys
                    and not re.fullmatch(r"[a-z]+", v) and v not in out):
                out.append(v)
    return out


def l_lines(text, notes, state, messages, tool_names):
    """L's lines for one failed attempt (see the L comment above), in a fixed order: L2 by app (most unopened records
    first), then L1 by collection (most values first); at most L_MAX_LINES."""
    from abce.record.observed import Corpus, normalize

    observed = l_observed(messages)
    by_app = {}
    for app, coll, rec in _l_records(state):
        if not any(len(s.strip()) >= 4 for s in _l_own(rec)):
            continue
        # a container counts as opened when a record inside it was (a spreadsheet whose rows the attempt read)
        is_open = l_opened(rec, observed) or any(l_opened(sub, observed) for _a, _c, sub in _l_records(rec, app, ("x",)))
        by_app.setdefault(app, []).append((coll, rec, is_open))
    l2, l1 = [], []
    for app, recs in sorted(by_app.items()):
        shut = [(c, r) for c, r, o in recs if not o and any(len(s.strip()) >= L_LONG for s in _l_own(r))]
        if not any(o for _c, _r, o in recs) and shut:
            colls = Counter(c for c, _r in shut)
            coll = min(colls, key=lambda c: (-colls[c], c))
            tool = l_tool(app, coll, tool_names)
            if tool:
                part = any(l_part_seen(r, observed) for c, r, _o in recs if c == coll)
                l2.append((-len(shut), app, coll, colls[coll], tool, L_HEAD_PART if part else L_HEAD))
    covered = {x[1] for x in l2}
    values = _l_values(text, notes, state)
    for app, recs in sorted(by_app.items()):
        if app in covered:
            continue
        by_coll, recs_by_id = {}, {}
        for coll, rec, is_open in recs:
            if is_open:
                continue
            body = normalize(" ".join(_l_own(rec)))
            for v in values:
                if Corpus._form_in(normalize(v), body):
                    by_coll.setdefault(coll, {}).setdefault(v, []).append(id(rec))
                    recs_by_id[id(rec)] = rec
        for coll, hits in sorted(by_coll.items()):
            kept = [v for v in values if 0 < len(hits.get(v, [])) <= L_MAX_SHARE][:L_MAX_VALUES]
            if not kept:
                continue
            tool = l_tool(app, coll, tool_names)
            if tool:
                named = {r for v in kept for r in hits[v]}
                part = any(l_part_seen(recs_by_id[r], observed) for r in named)
                l1.append((-len(kept), -len(named), app, coll, len(named), kept, tool, L_HEAD_PART if part else L_HEAD))
    lines = []
    for _k, app, coll, n, tool, head in sorted(l2)[:2]:
        lines.append(f"{head}the task's {l_noun(app, coll, n)} (read {'it' if n == 1 else 'them'} with {tool} first).")
    for _k, _n, app, coll, n, kept, tool, head in sorted(l1)[:2]:
        if len(lines) >= L_MAX_LINES:
            break
        quoted = [f'"{v}"' for v in kept]
        said = quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " or " + quoted[-1]
        lines.append(f"{head}{n} {l_noun(app, coll, n)} that {'mentions' if n == 1 else 'mention'} {said} "
                     f"(read {'it' if n == 1 else 'them'} with {tool} first).")
    return lines[:L_MAX_LINES]


def say_never_opened(first_message, lines):
    """L's lines on their own lines right before the closing line; the text unchanged when there are none."""
    if not lines:
        return first_message
    if not first_message.endswith(L_CLOSE) or first_message.count(L_CLOSE) != 1:
        raise ValueError("L: the first message does not end with the closing line once")
    return first_message[:-len(L_CLOSE)] + "\n" + "\n".join(lines) + L_CLOSE


def xl_live(design, engine_context, shown_context, record, chain):
    """Round 10, live: the S1FUHWN text (xn_live), then L's lines in its first message (notes unchanged)."""
    from abce.record.episode import Episode

    if design not in DESIGNS_XL:
        raise ValueError(design)
    out = xn_live(design[:-1], engine_context, shown_context, record, chain)
    episode = Episode(record=record)
    tools = episode.task_data.get("zapier_tools") or [t.get("name") for t in (record["traces"][0].get("tools") or [])]
    lines = l_lines(out["first_message"], out.get("notes"), episode.task_data.get("initial_state") or {},
                    episode.messages, tools)
    return {**out, "first_message": say_never_opened(out["first_message"], lines)}


# --- Round 11 (22 Nov, D1-24): K = "already read: where the reason is". My reading of 20 S1FUHWNL solves found 3
# dodges, all at points without L lines, where the deciding data sat in a record the teacher HAD opened (ticket fd_510's
# do-not-auto-reply tag, the privacy email naming #company-announcements, the price sheet's hold note on "Website
# Design"); the forbid or keep line names the item and the string without where the reason is, and the teacher reworded
# the string or invented a reason. K adds, to each text (the first message, a note) with a forbid or keep line, the
# records the attempt had read that are about the line's item:
#   the item = a record of the task's initial world that holds no records of its own (a ticket, a contact, a product, a
#   channel; not a spreadsheet or worksheet) one of whose identity fields (id-like, name, title, subject) is a value the
#   line quotes; its names = its name, title and subject values ("wp_002" -> "Website Design"; never an id: an account id
#   inside contacts or a group id inside tickets says nothing about where the reason is);
#   K names (a) the item's own record when it holds a text of 20+ characters other than its identity values (a
#   description, a note) and (b) the other records that mention one of its names in their own fields (a name more than 5
#   records mention says nothing about where the reason is) and hold a text of 20+ characters; each named record was
#   opened BEFORE the point the teacher continues from (L's opened test on the messages before the R-c cut, the
#   containers rule as L): what the attempt opened later is not yet read there; each line names a read tool of the task.
#     Already read: the freshdesk ticket "fd_510" (read it again with freshdesk_get_tickets first).
#     Already read: 1 gmail message that mentions "company-announcements" (read it again with gmail_find_email first).
# At most 2 lines per text and 3 per context, no record named twice; first-message lines go after L's, right before the
# closing line; a note's lines go on their own lines at its end.
DESIGNS_XK = ("S1FUHWNLK",)
K_HEAD = "Already read: "
K_LONG = 20
K_NAME_KEYS = ("name", "title", "subject", "account_name", "full_name", "display_name")
K_ID_KEYS = L_IDS + K_NAME_KEYS
K_TRIGGER = re.compile(r"\bforbids\b|\bto stay\b|\bnot containing\b")
K_MAX_PER_TEXT = 2
K_MAX_LINES = 3
# where a write went, not what it is about: a spreadsheet, worksheet, file or board is named in a line as the place
# ("rows in spreadsheet X worksheet Y"); a record of these collections is never an item
K_PLACES = ("spreadsheets", "worksheets", "files", "folders", "drives", "boards", "bases", "tables", "databases")
K_LABEL_MAX = 40


def _k_values(rec, keys):
    """A record's values of the given fields (4+ characters with a letter), in field order."""
    out = []
    for k in keys:
        v = rec.get(k)
        if isinstance(v, str) and len(v.strip()) >= 4 and re.search(r"[A-Za-z]", v) and v.strip() not in out:
            out.append(v.strip())
    return out


def _k_keys(rec, keys):
    """The identity fields of a record: a person's "title" (a record with an email) is a job title, not a name."""
    return tuple(k for k in keys if not (k == "title" and "email" in rec))


def _k_nested(rec):
    return [sub for _a, _c, sub in _l_records(rec, "x", ("x",))]


def _k_is_open(rec, observed):
    return l_opened(rec, observed) or any(l_opened(sub, observed) for sub in _k_nested(rec))


def k_lines_for(text, state, observed, tool_names, named):
    """K's lines for one text (the first message or one note) that has a forbid or keep line. observed: what the
    attempt had seen before the cut; named: ids of the records already named in this context (updated)."""
    from abce.record.observed import Corpus, normalize

    units = [u for u in text.split("\n") if K_TRIGGER.search(u)]
    if not units:
        return []
    values = _l_values("\n".join(units), {}, state)
    want = {v.strip().casefold(): v for v in values}
    everything = list(_l_records(state))
    selves, mentions = {}, {}  # (app, coll) -> {record id: (record, label)}
    for app, coll, subj in everything:
        ids = _k_values(subj, _k_keys(subj, K_ID_KEYS))
        hit = next((want[i.casefold()] for i in ids if i.casefold() in want), None)
        if hit is None or coll in K_PLACES or _k_nested(subj):
            continue
        names = _k_values(subj, _k_keys(subj, K_NAME_KEYS))
        label = hit if len(hit) <= K_LABEL_MAX or not names else names[0]
        if (any(len(s.strip()) >= K_LONG and s.strip() not in ids for s in _l_own(subj))
                and _k_is_open(subj, observed) and len(label) <= K_LABEL_MAX):
            selves.setdefault((app, coll), {})[id(subj)] = (subj, label)
        for name in names:
            hits = []
            for app2, coll2, rec2 in everything:
                if rec2 is subj or any(x is subj for x in _k_nested(rec2)):
                    continue
                if Corpus._form_in(normalize(name), normalize(" ".join(_l_own(rec2)))):
                    hits.append((app2, coll2, rec2))
            if not hits or len(hits) > L_MAX_SHARE:
                continue
            for app2, coll2, rec2 in hits:
                if any(len(s.strip()) >= K_LONG for s in _l_own(rec2)) and _k_is_open(rec2, observed):
                    mentions.setdefault((app2, coll2), {})[id(rec2)] = (rec2, name)
    lines = []
    for (app, coll), recs in sorted(selves.items()):
        fresh = [(r, v) for k, (r, v) in recs.items() if k not in named][:L_MAX_VALUES]
        tool = l_tool(app, coll, tool_names)
        labels = list(dict.fromkeys(f'"{v}"' for _r, v in fresh))
        if not fresh or not tool or len(lines) >= K_MAX_PER_TEXT or len(labels) < len(fresh):
            continue  # a label two records share would not say which record
        said = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]
        n = len(fresh)
        lines.append(f"{K_HEAD}the {l_noun(app, coll, n)} {said} (read {'it' if n == 1 else 'them'} again with {tool} first).")
        named.update(id(r) for r, _v in fresh)
    for (app, coll), recs in sorted(mentions.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        fresh = [(r, v) for k, (r, v) in recs.items() if k not in named]
        tool = l_tool(app, coll, tool_names)
        if not fresh or not tool or len(lines) >= K_MAX_PER_TEXT:
            continue
        labels = list(dict.fromkeys(f'"{v}"' for _r, v in fresh))[:L_MAX_VALUES]
        said = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " or " + labels[-1]
        n = len(fresh)
        lines.append(f"{K_HEAD}{n} {l_noun(app, coll, n)} that {'mentions' if n == 1 else 'mention'} {said} "
                     f"(read {'it' if n == 1 else 'them'} again with {tool} first).")
        named.update(id(r) for r, _v in fresh)
    return lines


def k_lines(first_message, notes, state, messages, tool_names, cut):
    """K's lines for one context: (first-message lines, {turn: note lines}); 3 in all at most, first message first.
    cut: the R-c cut (visibility_turn); only what the attempt had seen before it counts as read."""
    observed = l_observed(visible_messages(messages, cut))
    named = set()
    first = k_lines_for(first_message, state, observed, tool_names, named)[:K_MAX_LINES]
    left = K_MAX_LINES - len(first)
    out_notes = {}
    for turn in sorted(notes or {}, key=lambda t: int(t) if str(t).isdigit() else 0):
        if left <= 0:
            break
        got = k_lines_for(notes[turn], state, observed, tool_names, named)[:left]
        if got:
            out_notes[turn] = got
            left -= len(got)
    return first, out_notes


def say_already_read(first_message, notes, first, by_note):
    """K's first-message lines right before the closing line (after L's); a note's lines on their own lines at its end."""
    if first:
        if not first_message.endswith(L_CLOSE) or first_message.count(L_CLOSE) != 1:
            raise ValueError("K: the first message does not end with the closing line once")
        first_message = first_message[:-len(L_CLOSE)] + "\n" + "\n".join(first) + L_CLOSE
    if by_note:
        notes = {t: (x + "\n" + "\n".join(by_note[t]) if t in by_note else x) for t, x in (notes or {}).items()}
    return first_message, notes


def xk_live(design, engine_context, shown_context, record, chain):
    """Round 11, live: the S1FUHWNL text (xl_live), then K's lines in its first message and notes."""
    from abce.record.episode import Episode

    if design not in DESIGNS_XK:
        raise ValueError(design)
    out = xl_live(design[:-1], engine_context, shown_context, record, chain)
    episode = Episode(record=record)
    tools = episode.task_data.get("zapier_tools") or [t.get("name") for t in (record["traces"][0].get("tools") or [])]
    first, by_note = k_lines(out["first_message"], out.get("notes"), episode.task_data.get("initial_state") or {},
                             episode.messages, tools, visibility_turn(engine_context, record))
    fm, notes = say_already_read(out["first_message"], out.get("notes"), first, by_note)
    return {**out, "first_message": fm, "notes": notes}


# --- Round 12 (22 Nov, D1-29): E = "the forbidden thing, not the string". Reading 60 solves over rounds 9-11 found 9
# dodges at forbid or keep lines: the teacher keeps the content a line forbids in other words ('leave balance' or
# 'Time Off' for "PTO", 'No match' for "No FAQ match") or states a reason the task's data does not give ('constrained by
# system requirement', 'Your hire date falls outside ...'). The benchmark's forbidden words and values mark content to
# keep out of that output (45 of its 364 content checks read: excluded or held items, names, decoy values, status
# claims; none where other wording is the fix), and pointing at records the attempt had read (K) did not change it.
# E adds one line, once per context, to the first text (the first message's results, else the note) with a forbid or
# keep line: when a line forbids content, that output must not say the same thing in other words; for S1FUHWNLE also,
# when a line forbids or keeps anything, any reason must come from the task's data. S1FUHWNLD has the first sentence.
#   Where a line forbids a word or value, that output must not say the same thing in other words. Any reason you give
#   must come from the task's data.
# First-message line right before the closing line (after L's); a note's line on its own line at its end.
DESIGNS_XE = ("S1FUHWNLE", "S1FUHWNLD")
E_CONTENT = re.compile(r"\bforbids\b[^\n]*\b(?:containing|text|body|cells?|value|summary|subject)\b|\bnot containing\b")
E_SAME = "Where a line forbids a word or value, that output must not say the same thing in other words."
E_REASON = "Any reason you give must come from the task's data."


def e_results(first_message):
    """The first message's results (after F_HEAD); '' when it has none (the task's own words never count)."""
    i = first_message.find(F_HEAD)
    return first_message[i + len(F_HEAD):] if i >= 0 else ""


def e_note_order(notes):
    return sorted(notes or {}, key=lambda t: int(t) if str(t).isdigit() else 0)


def e_line(design, first_message, notes):
    """E's line for one context: '' when no text has a forbid or keep line (or, for S1FUHWNLD, none forbids content)."""
    if design not in DESIGNS_XE:
        raise ValueError(design)
    texts = [e_results(first_message)] + [notes[t] for t in e_note_order(notes)]
    lines = [x for t in texts for x in t.split("\n") if K_TRIGGER.search(x)]
    parts = [E_SAME] if any(E_CONTENT.search(x) for x in lines) else []
    if design == "S1FUHWNLE" and lines:
        parts.append(E_REASON)
    return " ".join(parts)


def say_e(first_message, notes, line):
    """E's line in the first text with a forbid or keep line: right before the first message's closing line, else on
    its own line at that note's end; the texts unchanged when the line is empty."""
    if not line:
        return first_message, notes
    if any(K_TRIGGER.search(x) for x in e_results(first_message).split("\n")):
        if not first_message.endswith(L_CLOSE) or first_message.count(L_CLOSE) != 1:
            raise ValueError("E: the first message does not end with the closing line once")
        return first_message[:-len(L_CLOSE)] + "\n" + line + L_CLOSE, notes
    for t in e_note_order(notes):
        if any(K_TRIGGER.search(x) for x in notes[t].split("\n")):
            return first_message, {**notes, t: notes[t] + "\n" + line}
    raise ValueError("E: a line without a text that holds a forbid or keep line")


def xe_live(design, engine_context, shown_context, record, chain):
    """Round 12, live: the S1FUHWNL text (xl_live), then E's line once (first message or note)."""
    if design not in DESIGNS_XE:
        raise ValueError(design)
    out = xl_live("S1FUHWNL", engine_context, shown_context, record, chain)
    fm, notes = say_e(out["first_message"], out.get("notes"), e_line(design, out["first_message"], out.get("notes")))
    return {**out, "first_message": fm, "notes": notes}


# Round 13 (22 Nov): J. A forbid of several strings together reads as several forbidden strings: 'This task forbids
# body containing "depreciation", "FA-003" in that gmail message for "FA-003".' The check forbids one output that holds
# them all (_forbid_multi: "containing A, B" needs both), and on round 12's screen the teacher then avoided the word
# "depreciation" itself in a depreciation report. J says the list as one condition: 'containing both "depreciation" and
# "FA-003"' (3 or more: 'containing all of "A", "B" and "C"'). Only a quoted list right after "containing" inside a
# "This task forbids" sentence changes; lists of separate targets ('gmail messages to "a", "b" and "c"', 'assignee id
# "x", "y"') and every other sentence stay as they are.
J_LIST = re.compile(r'(?<=\bcontaining )(?<!\bnot containing )("[^"\n]+"(?:, "[^"\n]+")+)(?!,| and | or )')


def j_forbid_spans(line):
    """(start, end) of each 'This task forbids' sentence in one line; a sentence ends at the first '.' outside quotes
    that ends the line or comes before a space."""
    spans, i = [], 0
    while (j := line.find("This task forbids", i)) >= 0:
        k, quoted = j, False
        while k < len(line):
            if line[k] == '"':
                quoted = not quoted
            elif line[k] == "." and not quoted and (k + 1 == len(line) or line[k + 1] == " "):
                break
            k += 1
        spans.append((j, min(k + 1, len(line))))
        i = k + 1
    return spans


def j_list(m):
    items = re.findall(r'"[^"\n]+"', m.group(1))
    if len(items) == 2:
        return f"both {items[0]} and {items[1]}"
    return "all of " + ", ".join(items[:-1]) + " and " + items[-1]


def say_joint(text):
    """J on one text: every joint 'containing' list inside a forbid sentence said as one condition."""
    out = []
    for line in text.split("\n"):
        for a, b in reversed(j_forbid_spans(line)):
            line = line[:a] + J_LIST.sub(j_list, line[a:b]) + line[b:]
        out.append(line)
    return "\n".join(out)


# Round 13 candidate (22 Nov, draft): G. Round 12's E sentence ("... that output must not say the same thing in other
# words.") left the conflict it meets unresolved: where the task's words ask for an item a line forbids ("include each
# affected person"), the teacher deliberates ("This is a genuine conflict"; doubt and conflict talk rise where E acts)
# and some solves still keep the item in other words. G says which side wins and what to do, once per context, right
# after the last forbid or keep line of the first text that has one: the line is right, leave that item out, in any
# wording. S1FUHWNLGO adds O where any line forbids or keeps something: no message or record refers to the results or
# notes (round 12's borderline dodges wrote "per system constraint(s)" in an email and a Slack post; O does not say
# "these lines", which the teacher could read as the values the lines quote, some of which an output needs).
DESIGNS_XG = ("S1FUHWNLG", "S1FUHWNLGO", "S1FUHWNLO")  # S1FUHWNLO: O alone (round 13 fallback: no G sentence)
G_RIGHT = "Each forbid line is right, even where the task's words seem to ask otherwise: leave that item out, in any wording."
O_QUIET = "Never refer to these results or notes in any message or record you write."


def g_line(design, first_message, notes):
    """G's line for one context: G where a line forbids content (not S1FUHWNLO), plus O (S1FUHWNLGO, S1FUHWNLO) where
    any line forbids or keeps."""
    if design not in DESIGNS_XG:
        raise ValueError(design)
    texts = [e_results(first_message)] + [notes[t] for t in e_note_order(notes)]
    lines = [x for t in texts for x in t.split("\n") if K_TRIGGER.search(x)]
    parts = [G_RIGHT] if design != "S1FUHWNLO" and any(E_CONTENT.search(x) for x in lines) else []
    if design in ("S1FUHWNLGO", "S1FUHWNLO") and lines:
        parts.append(O_QUIET)
    return " ".join(parts)


def say_g(first_message, notes, line):
    """G's line on its own line right after the last forbid or keep line of the first text that has one (the first
    message's results, else the earliest such note); the texts unchanged when the line is empty."""
    if not line:
        return first_message, notes
    i = first_message.find(F_HEAD)
    if i >= 0:
        head, res = first_message[:i + len(F_HEAD)], first_message[i + len(F_HEAD):]
        rl = res.split("\n")
        hits = [k for k, x in enumerate(rl) if K_TRIGGER.search(x)]
        if hits:
            rl.insert(hits[-1] + 1, line)
            return head + "\n".join(rl), notes
    for t in e_note_order(notes):
        nl = notes[t].split("\n")
        hits = [k for k, x in enumerate(nl) if K_TRIGGER.search(x)]
        if hits:
            nl.insert(hits[-1] + 1, line)
            return first_message, {**notes, t: "\n".join(nl)}
    raise ValueError("G: a line without a text that holds a forbid or keep line")


def xg_live(design, engine_context, shown_context, record, chain):
    """Round 13 candidate, live: the S1FUHWNL text (xl_live), then G's line once (first message or note)."""
    if design not in DESIGNS_XG:
        raise ValueError(design)
    out = xl_live("S1FUHWNL", engine_context, shown_context, record, chain)
    fm, notes = say_g(out["first_message"], out.get("notes"), g_line(design, out["first_message"], out.get("notes")))
    return {**out, "first_message": fm, "notes": notes}


# A trailing J = the base design's text with J on its first message and every note (nothing else changes).
DESIGNS_J = ("S1FUHWNLJ", "S1FUHWNLEJ", "S1FUHWNLDJ", "S1FUHWNLGJ", "S1FUHWNLGOJ", "S1FUHWNLOJ")


def xj_live(design, engine_context, shown_context, record, chain):
    if design not in DESIGNS_J:
        raise ValueError(design)
    out = sc_live(design[:-1], engine_context, shown_context, record, chain)
    return {**out, "first_message": say_joint(out["first_message"]),
            "notes": {t: say_joint(v) for t, v in (out.get("notes") or {}).items()}}


# Round 14 (22 Nov): Y, said once and plainly. Round 13's reading and a scan of all 398 continuations found two wording
# faults in every version's text: (1) a value said twice in one sentence, when the trailing 'for "X"' (which check a
# line comes from) names a value the same clause already quotes ('with body containing "Diana Parks" for "Diana Parks"';
# 195 of 1,087 contexts); the teacher repeats the phrase and stops on it ("The phrase 'for Diana Parks' is a bit
# ambiguous"); (2) forbid sentences that lost their field word or keep engine field names ('This task forbids containing
# "SSO"', 'forbids in group "grp_t2"', 'forbids to "a@b" and body containing ...', 'the to shown', 'body plain', 'the
# values text shown'); where they occur the teacher quoted them on 12 of 14 points and called them confusing on 7. Y
# drops such a tail and says those sentences in the engine's own plain forms ('forbids text containing', as the hidden
# form's "the text shown"; 'recipients "a" or "b" and body containing', as the engine's other recipient forbids). Only
# words outside quoted values change, and no quoted value is added or lost.
DESIGNS_XY = ("S1FUHWNLGOJY", "S1FUHWNLOJY")  # DESIGNS_Y is round 4's Y1/Y2 (yc_live)
Y_TAIL = re.compile(r' for "([^"\n]+)"')
Y_TAIL_END = re.compile(r'(?: \(still missing\))?(?:[.;),]|$)')
Y_PLAIN = (
    (re.compile(r"This task forbids containing (?=\")"), "This task forbids text containing "),
    (re.compile(r"This task forbids in ([a-z]+) (?=\")"), r"This task forbids \1 "),
    (re.compile(r'This task forbids to (?="[^"\n]+"(?: or | and |, )")'), "This task forbids recipients "),
    (re.compile(r"This task forbids to (?=\")"), "This task forbids recipient "),
    (re.compile(r"\bthe to\b(?=(?: and [a-z ]+?)? shown\b)"), "the recipient"),
    (re.compile(r"\bthe from\b(?=(?: and [a-z ]+?)? shown\b)"), "the sender"),
    (re.compile(r"\bbody plain\b"), "body"),
    (re.compile(r"\bthe values text shown\b"), "the text shown"),
    (re.compile(r"\bwith to (?=\")"), "to "),
    (re.compile(r"\bwith values containing (?=\")"), "with text containing "),
)


def y_scope_start(text, end):
    """Where the clause holding text[end] starts: after the last sentence end ('. '), ';' or unclosed '(' outside
    quoted values before it."""
    start, quoted, opened = 0, False, []
    for k in range(end):
        ch = text[k]
        if ch == '"':
            quoted = not quoted
        elif quoted:
            continue
        elif ch == "(":
            opened.append(k)
        elif ch == ")" and opened:
            opened.pop()
        elif ch == ";" or (ch == "." and text[k + 1:k + 2] == " "):
            start = max(start, k + 1)
    return max([start] + [k + 1 for k in opened])


Y_LIST_BEFORE = re.compile(r'(?:"(?:, | and | or )|\bboth |\ball of )$')
Y_LIST_AFTER = re.compile(r'(?:, | and | or )"')


def y_said_alone(scope, value):
    """True when the clause quotes the value on its own at least once, not as one item of a quoted list (there the
    tail says which item the line is about: 'containing "Hector Ruiz", "Amy Liu" for "Hector Ruiz"')."""
    q = '"' + value + '"'
    i = scope.find(q)
    while i >= 0:
        if scope[:i].count('"') % 2 == 0 and not Y_LIST_BEFORE.search(scope[:i]) and not Y_LIST_AFTER.match(scope, i + len(q)):
            return True
        i = scope.find(q, i + 1)
    return False


def y_once(line):
    """Drop each trailing ' for "X"' whose X the same clause already quotes on its own (rightmost first)."""
    for m in reversed(list(Y_TAIL.finditer(line))):
        if line[:m.start()].count('"') % 2 or not Y_TAIL_END.match(line, m.end()):
            continue
        if y_said_alone(line[y_scope_start(line, m.start()):m.start()], m.group(1)):
            line = line[:m.start()] + line[m.end():]
    return line


def say_once(text):
    """Y on one text: the plain forms first (outside quoted values), then each value said once per clause."""
    out = []
    for line in text.split("\n"):
        for rx, repl in Y_PLAIN:
            line = F._outside_quotes(rx.pattern, (lambda m, repl=repl: m.expand(repl)), line)
        out.append(y_once(line))
    return "\n".join(out)


def xy_live(design, engine_context, shown_context, record, chain):
    if design not in DESIGNS_XY:
        raise ValueError(design)
    out = sc_live(design[:-1], engine_context, shown_context, record, chain)
    return {**out, "first_message": say_once(out["first_message"]),
            "notes": {t: say_once(v) for t, v in (out.get("notes") or {}).items()}}


# Round 15 (22 Nov, decisions D1-38, D1-39): I, V and C on S1FUHWNLGOJY.
# I (D1-39): a pointer may name only a read tool that can return the records it points to. L's tool choice took any read
# tool of the app, so 36 of 1,087 contexts pointed at records with a tool that cannot show them ("the task's slack
# messages (read them with slack_list_channels first)": that tool returns channel names and topics; google calendar
# events with google_calendar_find_calendars; buffer posts with buffer_list_channels; zoho desk accounts with
# zoho_desk_get_tickets), and the teacher said "I don't have a function to list or search slack messages by content".
# A tool can return a collection's records when its name (the words after the app) names that collection, or names no
# collection of the app at all (gmail_find_email for gmail messages; salesforce_find_records). An L line whose tool
# cannot is given the best tool that can (L's ranking), or dropped when the task has none.
# V: my full reading of 20 S1FUHWNLGOJY solves and a scan of all its 154 solves outside the listed task found the reason
# shape where a keep line holds a value ('its field "stage_name" has to stay "Demo"'): the teacher keeps the value and
# writes a reason no record gives ('per account override kept stage as Demo'; 'Price maintained at $4,500 per system
# constraint'), 2 of the 6 keep-line solves. O already stands once in every context with a keep line; V makes it say
# that too. No keep-line context asks for a reason in a still-needed line (0 of 11 in v28), so V meets no check.
# C: at finance.gl_variance_analysis the only email holding the rule was opened by the attempt after the point the teacher
# continues from, so L's whole-attempt test gave it no pointer, and L1 named a planted email that merely mentions the
# forbidden word. C is L2 said at that point and true there: for each app whose records the attempt opened, but none
# before the point (the R-c cut, as K), one line at the end of the note at the cut ("Not read yet: the task's gmail
# messages (read them with gmail_find_email first)."; "Not read in full yet: " when some part of them was shown before
# the point), at most 2, each with a tool I allows; the first message's L1 lines about the same app and collection are
# dropped (C names all their records). Only a note that stands exactly at the cut gets C's lines; without one, C adds nothing.
DESIGNS_XV = ("S1FUHWNLGOJYIV", "S1FUHWNLGOJYIVC")
V_QUIET = "Never refer to these results or notes, or say why a value is left as it is, in any message or record you write."
# (not "has to stay": K_TRIGGER and the keep-line scans read "to stay" as a keep line)
V_KEEP = re.compile(r"\bhas to stay\b")
C_HEAD = "Not read yet: "
C_HEAD_PART = "Not read in full yet: "
C_MAX_LINES = 2
I_TOOL = re.compile(r"\(read (?:it|them) with (\S+) first\)\.$")


def _i_names(tool, coll):
    """The tool's own words (after its app) name the collection: all its words in order, singular ('scheduled_events' ->
    'scheduled event': calendly_list_scheduled_events names it, calendly_list_event_types does not)."""
    words = [_l_singular(w) for w in tool.split("_")[len(app_of(tool).split("_")):]]
    want = [_l_singular(w) for w in coll.split("_")]
    return any(words[i:i + len(want)] == want for i in range(len(words) - len(want) + 1))


_I_RECORDS = None


def _i_app_records(app):
    """The app's record kinds in the pinned environment's world schema (i_app_records.json, made by weekend/sdpo-close/
    tools/make_i_app_records.py from env v1.1): a tool named for one of them stands for it even in a task whose world
    holds none of it (google_calendar_find_calendars in a task with events only)."""
    global _I_RECORDS
    if _I_RECORDS is None:
        import json
        import os

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "i_app_records.json")) as f:
            _I_RECORDS = {k: set(v) for k, v in json.load(f)["apps"].items()}
    return _I_RECORDS.get(app, set())


def i_can_read(tool, coll, colls):
    """I: the tool can return the collection's records: it names that collection, or no record kind of the app (the
    task's collections and the schema's)."""
    kinds = set(colls) | _i_app_records(app_of(tool))
    return _i_names(tool, coll) or not any(_i_names(tool, c) for c in kinds)


def i_tool(app, coll, tool_names, colls):
    """L's tool choice among the task's tools that I allows for the collection; None when there is none."""
    return l_tool(app, coll, [t for t in tool_names or () if i_can_read(t, coll, colls)])


def _app_colls(state):
    out = {}
    for app, coll, _rec in _l_records(state):
        out.setdefault(app, set()).add(coll)
    return out


def say_i(first_message, state, tool_names):
    """I on the first message's L lines: (text, [(line, new line or None)]). A line whose app and collection cannot be
    told from its words is left as it is."""
    colls = _app_colls(state)
    nouns = sorted(((l_noun(a, c, n), a, c) for a, cs in colls.items() for c in cs for n in (1, 2)), key=lambda x: -len(x[0]))
    out, changed = [], []
    for ln in first_message.split("\n"):
        m = I_TOOL.search(ln) if ln.startswith(L_HEADS) else None
        hit = None
        if m:
            hit = next(((a, c) for noun, a, c in nouns if f"the task's {noun} (read" in ln
                        or re.search(r": \d+ " + re.escape(noun) + r" that mention", ln)), None)
        if hit and not i_can_read(m.group(1), hit[1], colls[hit[0]]):
            alt = i_tool(hit[0], hit[1], tool_names, colls[hit[0]])
            new = None if alt is None else ln[:m.start(1)] + alt + ln[m.end(1):]
            changed.append((ln, new))
            if new is not None:
                out.append(new)
            continue
        out.append(ln)
    return "\n".join(out), changed


def say_v(first_message, notes):
    """V on one context: where any text has a keep line, O's sentence (it stands once) becomes V's; else unchanged."""
    notes = notes or {}
    texts = [first_message] + list(notes.values())
    if not any(V_KEEP.search(t) for t in texts):
        return first_message, notes
    if sum(t.count(O_QUIET) for t in texts) != 1:
        raise ValueError("V: O must stand exactly once in a context with a keep line")
    return first_message.replace(O_QUIET, V_QUIET), {k: v.replace(O_QUIET, V_QUIET) for k, v in notes.items()}


def c_lines(state, messages, tool_names, cut):
    """C's lines and the (app, collection) pairs they name: apps of the initial world whose records the attempt opened (L's
    test on the whole attempt) but none before the cut; the collection named is the one with most records unopened at
    the cut; the tool is one I allows."""
    if cut is None:
        return [], set()
    whole, before = l_observed(messages), l_observed(visible_messages(messages, cut))
    colls_of = _app_colls(state)
    by_app = {}
    for app, coll, rec in _l_records(state):
        if not any(len(s.strip()) >= 4 for s in _l_own(rec)):
            continue
        subs = [sub for _a, _c, sub in _l_records(rec, app, ("x",))]
        opened_whole = l_opened(rec, whole) or any(l_opened(s, whole) for s in subs)
        opened_before = l_opened(rec, before) or any(l_opened(s, before) for s in subs)
        by_app.setdefault(app, []).append((coll, rec, opened_whole, opened_before))
    found = []
    for app, recs in sorted(by_app.items()):
        if not any(w for _c, _r, w, _b in recs) or any(b for _c, _r, _w, b in recs):
            continue
        shut = [(c, r) for c, r, _w, b in recs if not b and any(len(s.strip()) >= L_LONG for s in _l_own(r))]
        if not shut:
            continue
        colls = Counter(c for c, _r in shut)
        coll = min(colls, key=lambda c: (-colls[c], c))
        tool = i_tool(app, coll, tool_names, colls_of[app])
        if tool:
            part = any(l_part_seen(r, before) for c, r, _w, _b in recs if c == coll)
            found.append((-len(shut), app, coll, colls[coll], tool, C_HEAD_PART if part else C_HEAD))
    lines, covered = [], set()
    for _k, app, coll, n, tool, head in sorted(found)[:C_MAX_LINES]:
        lines.append(f"{head}the task's {l_noun(app, coll, n)} (read {'it' if n == 1 else 'them'} with {tool} first).")
        covered.add((app, coll))
    return lines, covered


def say_c(first_message, notes, lines, covered, cut):
    """C's lines on their own lines at the end of the note at the cut, and the first message's L1 lines about the same
    app and collection dropped (C names all their records); the texts unchanged when there are no lines or no note
    stands at the cut."""
    notes = notes or {}
    key = next((k for k in notes if str(k).isdigit() and int(k) == cut), None)
    if not lines or key is None:
        return first_message, notes
    nouns = [l_noun(a, c, n) for a, c in covered for n in (1, 2)]
    keep = []
    for ln in first_message.split("\n"):
        if ln.startswith(L_HEADS) and any(re.search(r": \d+ " + re.escape(x) + r" that mention", ln) for x in nouns):
            continue
        keep.append(ln)
    return "\n".join(keep), {**notes, key: notes[key] + "\n" + "\n".join(lines)}


def xv_live(design, engine_context, shown_context, record, chain):
    """Round 15, live: the S1FUHWNLGOJY text (xy_live), then I on its L lines, then V; S1FUHWNLGOJYIVC adds C."""
    from abce.record.episode import Episode

    if design not in DESIGNS_XV:
        raise ValueError(design)
    out = xy_live("S1FUHWNLGOJY", engine_context, shown_context, record, chain)
    episode = Episode(record=record)
    tools = episode.task_data.get("zapier_tools") or [t.get("name") for t in (record["traces"][0].get("tools") or [])]
    state = episode.task_data.get("initial_state") or {}
    fm, _changed = say_i(out["first_message"], state, tools)
    fm, notes = say_v(fm, out.get("notes"))
    if design == "S1FUHWNLGOJYIVC":
        cut = visibility_turn(engine_context, record)
        lines, covered = c_lines(state, episode.messages, tools, cut)
        fm, notes = say_c(fm, notes, lines, covered, cut)
    return {**out, "first_message": fm, "notes": notes}


def sc_live(design, engine_context, shown_context, record, chain):
    """v14 (round 5), live: S2 = Z4 with R-d (nothing hidden; a needed value unseen at the point is named with where the
    attempt read it) and no notes; S1 = S2 with R-n (the facts about the cut turn said once, in a note at that turn);
    P = P's framing; a trailing Q = the quiet instruction (sdpo_a3.QUIET) after the closing line (S1Q: V5's framing with
    it); a trailing R = the acting quiet instruction (QUIET_R_P with P's framing, QUIET_R_V5 with V5's)."""
    import feedback_z
    from abce.record.episode import Episode

    if design in DESIGNS_XV:
        return xv_live(design, engine_context, shown_context, record, chain)
    if design in DESIGNS_XY:
        return xy_live(design, engine_context, shown_context, record, chain)
    if design in DESIGNS_J:
        return xj_live(design, engine_context, shown_context, record, chain)
    if design in DESIGNS_XG:
        return xg_live(design, engine_context, shown_context, record, chain)
    if design in DESIGNS_XE:
        return xe_live(design, engine_context, shown_context, record, chain)
    if design in DESIGNS_XK:
        return xk_live(design, engine_context, shown_context, record, chain)
    if design in DESIGNS_XL:
        return xl_live(design, engine_context, shown_context, record, chain)
    if design in DESIGNS_XN:
        return xn_live(design, engine_context, shown_context, record, chain)
    if design in DESIGNS_X9:
        return x9_live(design, engine_context, shown_context, record, chain)
    if design in DESIGNS_X8:
        return x8_live(design, engine_context, shown_context, record, chain)
    if design in DESIGNS_X:
        return xc_live(design, engine_context, shown_context, record, chain)
    if design in DESIGNS_H:
        out = sc_live(design[:-1], engine_context, shown_context, record, chain)
        return {**out, "first_message": say_unshown(out["first_message"]),
                "notes": {t: say_unshown(v) for t, v in (out.get("notes") or {}).items()}}
    if design in DESIGNS_U:
        return {**sc_live(design[:-1], engine_context, shown_context, record, chain), "note_role": "user"}
    if design not in DESIGNS_S:
        raise ValueError(design)
    base = feedback_z.z_live("Z4", engine_context, shown_context, record, chain)
    episode = Episode(record=record)
    tool_defs = record["traces"][0].get("tools")
    if not tool_defs:
        raise ValueError("R-d: the attempt's tool definitions are missing (the seen test needs them)")
    tools = episode.task_data.get("zapier_tools")
    cut = visibility_turn(engine_context, record)
    seen = seen_test(episode.messages, tools, cut, tool_defs)
    text, _counts, problems = name_where(base["first_message"], seen, where_shown(episode.messages, tools, tool_defs), cut)
    if problems:
        raise ValueError(f"R-d: {problems[0]}")
    out = {**base, "first_message": text, "notes": {}, "final_note": None}
    if design.startswith("S1"):
        out["first_message"], out["notes"] = at_turn_note(text, cut)
    if "P" in design[2:]:
        out["first_message"] = p_frame(out["first_message"])
    if design.endswith("Q"):
        from sdpo_a3 import QUIET

        out["first_message"] = out["first_message"] + "\n\n" + QUIET
    if design.endswith("R"):
        out["first_message"] += "\n\n" + (QUIET_R_P if "P" in design[2:] else QUIET_R_V5)
    return out
