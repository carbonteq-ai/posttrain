"""Compact reference/protection hints for fresh native solver execution."""

import json
import re

LUNA_SUMMARY_PROMPT = """Summarize a workflow procedure for a solver.
Return JSON {procedure:[string],pitfalls:[string]}. Use at most6 procedure steps
and3 pitfalls,140 words total. Describe useful retrieval, decision, action and
verification operations. The example lists tool names and argument FIELD NAMES,
not correct argument values. Do not add names, IDs, addresses, dates or outcomes.
Do not discuss teachers, datasets, annotations or formatting. No preamble.
"""

GOAL_HINT_PROMPT = """Turn the task's goal and preservation declarations into a
short operational checklist. Return JSON {checks:[string]}, at most6 items and
100 words. Focus on what to read, verify, achieve or leave unchanged. Use only
facts in the user request; other declarations may suggest checks, not secret
values. Do not invent entities, IDs, dates or results. No commentary or preamble.
"""


def text_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(item.get("text", "") for item in content if isinstance(item, dict))
    raise ValueError("unexpected message content shape")


def user_key(messages):
    return json.dumps([text_content(m["content"]) for m in messages if m["role"] == "user"], ensure_ascii=False)


def compact_reference(actions):
    return [
        {
            "tool": a["call"]["name"],
            "argument_fields": sorted(a["call"].get("arguments", {})),
            "observed_success": a["result"].get("observed_result", {}).get("success")
            if isinstance(a["result"].get("observed_result"), dict)
            else None,
            "had_error": bool(a["result"].get("observed_error")),
        }
        for a in actions
    ]


def compact_goals(contract):
    return [
        {
            "id": c["check_id"],
            "role": c["role"],
            "operator": c["operator"],
            "source_kind": contract.get("sources", {}).get(c.get("source"), {}).get("kind"),
            "source_adapter": contract.get("sources", {}).get(c.get("source"), {}).get("adapter"),
        }
        for c in contract["checks"]
        if c["role"] in ("goal", "harm")
    ]


def validate_hints(value, fields, max_items, word_limit):
    if not isinstance(value, dict):
        raise ValueError("hints must be an object")
    lines = []
    for field, limit in zip(fields, max_items, strict=True):
        items = value.get(field)
        if (
            not isinstance(items, list)
            or len(items) > limit
            or any(not isinstance(x, str) or not x.strip() for x in items)
        ):
            raise ValueError("invalid hint list")
        lines.extend(items)
    if not lines or len(" ".join(lines).split()) > word_limit:
        raise ValueError("empty or overlong hints")
    return lines


def check_hint_literals(lines, public_prompt, reference):
    """Conservative syntactic screen; semantic review remains required."""
    text = "\n".join(lines)

    # No copied historical argument/result values that were absent from the
    # normal task prompt. This is a review gate, not proof of causal grounding.
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for x in value.values():
                yield from strings(x)
        elif isinstance(value, list):
            for x in value:
                yield from strings(x)

    def public_inbox_query(value):
        # A search operator is not a world-state fact. Permit only this exact
        # syntax plus alphabetic words already present in the public request;
        # addresses, IDs, numbers, hostnames and unseen names remain rejected.
        if not re.fullmatch(r"in:inbox(?: [A-Za-z]+){2,}", value):
            return False
        public_words = set(re.findall(r"[A-Za-z]+", public_prompt.casefold()))
        return all(word.casefold() in public_words for word in value.split()[1:])

    copied = sorted({s for s in strings(reference)
                     if len(s) >= 5 and s in text and s not in public_prompt and not public_inbox_query(s)})
    novel = [
        s
        for s in re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b(?:msg_|ws_|ss_|CPART|hv_|hu_)[\w-]+", text)
        if s not in public_prompt
    ]
    novel += [
        s for s in re.findall(r"\b(?:[A-Za-z0-9-]+\.)+(?:com|org|net|io|co|ai|edu|gov|example)\b", text)
        if s not in public_prompt
    ]
    if copied or novel:
        raise ValueError(f"nonpublic literal in hints: {copied + novel}")
    if re.search(r"\b(teacher|annotation|dataset|privileged|luna)\b", text, re.I):
        raise ValueError("generation bookkeeping leaked into solver instructions")


def guided_messages(messages, hint_lines):
    """Add only compressed instructions, preserving a separate clean input."""
    copied = [{**message} for message in messages]
    if not copied or copied[0]["role"] != "system":
        raise ValueError("native system message expected")
    copied[0]["content"] = (
        text_content(copied[0]["content"])
        + "\n\nOperational checklist:\n"
        + "\n".join("- " + line for line in hint_lines)
    )
    return copied
