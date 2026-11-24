"""Kind ids of the plain lines (engine-final specs, section D, order R1): every body line and note fragment of a
version-(c) context gets one kind, plus flags for the N1 clauses and the joined H2 suffix. The truth checker
cross-checks the classifier against its own attribution (`kind_attribution_mismatch`)."""

from __future__ import annotations

import re

_T = r"(?:At turns? [\d, ]+, )?"
# D89 (kind_attribution_mismatch): a tool name as the attempt called it, which may hold spaces ("salesforce Lead create",
# an unknown tool); the record's own tool error states it so
_TOOL = r"(?:\S+|[A-Za-z]\w*(?: [A-Za-z]\w*)+)"
WHY_FLAGS = (
    ("why_effect", re.compile(r" \(also required: | \(the task also requires: |; this task forbids it when ")),
    ("why_rule", re.compile(r" \(the task's rule: (?:forbidden|required) when ")),
)
RULES = [
    ("turn_limit", re.compile(r"^The attempt ended at its turn limit\.$")),
    ("not_listed", re.compile(r"^Not listed: \d+ more\.$")),
    ("final_not_done", re.compile(r"^Still not done at this point:(?: \(1\) .+\.|\n1\. .+)$", re.DOTALL)),  # correction r0: the note before the final reply
    ("hint_call_fix", re.compile(r'^The value ".*?" given at turn \d+ is the |^The error at turn \d+ lists ".*", the only listed entry ')),
    ("hint_value", re.compile(r'^In the initial world, ".*?" is the ')),
    ("hint_named", re.compile(r"^In the initial world, there is ")),
    ("hint_wrong_value", re.compile(r'^At turn \d+, \S+ wrote ".*" as .*; the task needed "')),
    ("hint_content_at_call", re.compile(r'^At turn \d+, \S+ (?:sent an email to|posted a message to|wrote) .*(?: without ".*" '
                                        r'in .*; the task requires .* to contain "|; the task needs ".*" as ")')),
    ("hint_must_not_at_call", re.compile(r'^At turn \d+, \S+ (?:with .+ )?broke this rule of the task: no .+\.$')),
    ("hint_rule_population", re.compile(r"^This applies to the rows of worksheet ")),
    ("cause_diff", re.compile(r"^(?:A successful attempt at this task|The correct solution above) called \S+ ")),
    ("unrequested_change", re.compile(r"^At turn \d+, \S+ changed .* of .*; no requirement of this task names that record\.$")),
    ("solution_note", re.compile(r"^Note: this solution ")),
    ("hint_missed_write", re.compile(r"^No call wrote an? .+\.$")),
    ("tool_error_repeated", re.compile(r"^" + _TOOL + r" failed \d+ times \(turns \d+-\d+\), first: ")),
    ("tool_error_others", re.compile(r"^" + _T + _TOOL + r" failed: .* \(and \d+ other " + _TOOL + r" errors?")),
    ("tool_error", re.compile(r"^" + _T + _TOOL + r" failed: ")),
    ("harm_collapsed", re.compile(r"^" + _T + r"\S+ (?:produced|updated|added|changed) \d+ |; this task forbids them|; they had to stay as they were"
                                  r"|; their .+ had to stay")),
    ("harm_required_content", re.compile(r"^" + _T + r"\S+ produced .*; the task requires one ")),
    ("harm_as_made", re.compile(r"; this task forbids it as made \(a further condition the task states\)")),
    # engine-fix 7: the keep tail names the protected field ("; its FX Rate had to stay "1.08"")
    ("changed", re.compile(r"^" + _T + r"\S+ changed .*; (?:it had to stay as it was|its .+ had to stay)")),
    # engine-fix 6: the attribute tail "; this task forbids ATTRS in|on that NOUN" (a collapsed line is matched by its count)
    ("harm", re.compile(r"^" + _T + r"(?:\S+|The attempt) (?:produced|updated|added) .*(?:; this task forbids it\b|; what this task forbids is |; this task forbids sending to that recipient\b|; this task forbids that "
                        r"|; this task forbids (?!it\b|them\b).+? (?:in|on) (?:that|these|those|each of these) [A-Za-z0-9])")),
    ("absent", re.compile(r"^Not done: no .* exists; one is required")),
    ("exists_but_none", re.compile(r"^Not done: .* exists, but none ")),
    ("exists_but_no_thing", re.compile(r"^Not done: .* exists, but has no ")),
    ("produced_nothing", re.compile(r"^Not done: the attempt (?:sent|created) no ")),
    ("never_read", re.compile(r"^Not done: the attempt never read ")),
    ("count", re.compile(r"^Not done: (?:exactly|at least) \d+ .* required; \d+ found")),
    ("not_met", re.compile(r"^Not met: ")),
    ("not_done_update", re.compile(r"^Not done: .* was to be updated")),
    ("required_effect", re.compile(r"^Not done: .* is required ")),
    # engine-fix 7 (judge-final3 71): a negative attribute check on an existing object asks for a removal
    ("required_removal", re.compile(r"^Not done: .* (?:had to be removed; |must not be there; |must not have )")),
    ("not_done", re.compile(r"^Not done: ")),
]


def _strip_suffixes(line: str) -> tuple[str, list[str]]:
    from .why_parse import balanced_segment

    flags = []
    if line.startswith("No call wrote "):
        return line, flags
    if " No call wrote " in line and re.search(r" No call wrote an? .+\.$", line):
        flags.append("missed_write")
        line = line[:line.rindex(" No call wrote ")]
    for name, pattern in WHY_FLAGS:
        if pattern.search(line):
            flags.append(name)
    for opener in (" (the task's rule: ", " (also required: ", " (the task also requires: "):
        seg = balanced_segment(line, opener)
        if seg is not None:
            line = line[:seg[0]] + line[seg[1]:]
    if "; this task forbids it when " in line:
        line = line.split("; this task forbids it when ", 1)[0] + "; this task forbids it."
    return line, flags


def classify(line: str) -> tuple[str, list[str]]:
    text, flags = _strip_suffixes(line.strip())
    for kind, pattern in RULES:
        if pattern.search(text):
            return kind, flags
    return "unknown", flags


STATEMENT_KINDS = {
    "prohibited": {"harm", "harm_as_made", "harm_collapsed"},
    "changed": {"changed", "harm_collapsed"},
    "tool_error": {"tool_error", "tool_error_others"},
    "missing": {"harm_required_content", "absent", "exists_but_none", "exists_but_no_thing", "produced_nothing",
                "never_read", "count", "not_met", "not_done_update", "required_effect", "required_removal", "not_done"},
}
HINT_KINDS = {"value_location": {"hint_value", "hint_named"}, "wrong_value": {"hint_wrong_value"},
              "rule_source": {"hint_rule_population"}, "cause_diff": {"cause_diff"},
              "unrequested_change": {"unrequested_change"}, "missed_write": {"not_done", "hint_missed_write"},
              "call_fix": {"hint_call_fix"}, "content_at_call": {"hint_content_at_call", "hint_must_not_at_call"}}


# line kinds and flags of the classifier that belong to a selectable fix-hint kind (abce.fix FIX_HINT_KINDS)
SELECTABLE = {"hint_value": "value_location", "hint_named": "value_location", "hint_wrong_value": "wrong_value",
              "hint_missed_write": "missed_write", "missed_write": "missed_write",
              "unrequested_change": "unrequested_change", "hint_call_fix": "call_fix",
              "hint_content_at_call": "content_at_call", "hint_must_not_at_call": "content_at_call",
              "why_effect": "why_clause", "why_rule": "why_clause", "solution_note": "solution_note"}


def compatible(kind: str, attributed: str) -> bool:
    """Whether the classifier's kind agrees with the checker's attribution ("statement:<item kind>",
    "hint:<meta kind>", "collapsed", "repeated", "turn_limit", "not_listed", "final_not_done")."""
    if attributed.startswith("statement:"):
        return kind in STATEMENT_KINDS.get(attributed.split(":", 1)[1], set())
    if attributed.startswith("hint:"):
        return kind in HINT_KINDS.get(attributed.split(":", 1)[1], set())
    return {"collapsed": {"harm_collapsed", "harm", "harm_as_made", "changed"}, "repeated": {"tool_error_repeated"},
            "turn_limit": {"turn_limit"}, "not_listed": {"not_listed"},
            "final_not_done": {"final_not_done"}}.get(attributed, set()).__contains__(kind)
