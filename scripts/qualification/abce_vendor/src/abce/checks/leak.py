"""Leak test (docs/design.md §5.3): with expected values hidden, no rendered text may contain a
value of a failed item that the attempt never observed.

Two independent implementations: (1) every `Value` of a failed item with `observed=False` is
searched for in the rendered text by its normalised forms; (2) every literal in the task's own
assertion parameters that is absent from the attempt's observed corpus is searched for as well
(so a leak through a path other than `Value` is caught too).
"""

from __future__ import annotations

from typing import Any

from ..record.model import Record
from ..record.observed import Corpus, normalize, value_forms, atoms


def rendered_text(context: dict) -> str:
    return "\n".join([context.get("first_message") or "", *context.get("notes", {}).values()])


def _template_text() -> list[str]:
    """The engine's fixed wording (headings, closings, statement templates, key labels): it carries
    no value, so a hidden value that happens to be an English word of the grammar ("follow" in
    "The following is feedback", "done" in "Not done:") is not a leak."""
    from ..render import common, plain, raw

    fixed = [common.FAILED_HEADING, common.SUCCESS_HEADING, common.CLOSING, common.NOTE_PREFIX,
             raw.HEADING, raw.CLOSING, raw.SOLUTION_HEADING, raw.SOLUTION_CLOSING,
             plain.MATCHED_HEADING, plain.MATCHED_CLOSING, plain.MATCHED_SOLUTION_HEADING, plain.MATCHED_SOLUTION_CLOSING,
             "A previous attempt on this same task satisfied all native checks. Use it as a reference, not as a new "
             "observation in your history.", "Correct solution:", "Reconsider your next action at this point and",
             "Not listed:", "The attempt ended at its turn limit.", "Not done:", "exists, but none", "exists, but has no",
             "this task forbids it", "is required", "the task requires one", "it had to stay as it was", "was to be updated",
             "plus a further condition the task states", "on one line", "At turn", "At turns", "The attempt produced",
             "Not met:", "a further condition the task states on", "the attempt never read", "the task names",
             "forbids it as made", "lacking a phrase the task states", "lacking the value the task states",
             "not the value the task states", "a phrase the task states", "without", "on that line", "in one paragraph",
             "in that paragraph", "the attempt sent no", "the attempt created no", "this task forbids them",
             "they had to stay as they were", "is required", "or another effect the task allows", "named by",
             "more like them", "one each", "failed:", "failed",
             "The attempt updated", "The attempt added", "The attempt changed", "required;", "at least", "no such", "one is",
             "Native checks that failed:", "Observed tool responses to this action:", "Every native check of this attempt passed."]
    fixed += list(plain.ADDRESS_LABELS.values()) + list(plain.CONTENT_LABELS.values())
    return sorted({normalize(f) for f in fixed if f and normalize(f)}, key=len, reverse=True)


_TEMPLATE_PATTERN = None


def scanned_text(context: dict) -> str:
    """The rendered text, normalised, with the engine's fixed wording removed as whole tokens (the label
    "in" is never cut out of "routing", which would leave "rout" for the detector to find)."""
    global _TEMPLATE_PATTERN
    import re

    if _TEMPLATE_PATTERN is None:
        phrases = "|".join(re.escape(p) for p in _template_text())
        _TEMPLATE_PATTERN = re.compile(r"(?<![a-z0-9])(?:" + phrases + r")(?![a-z0-9])")
    text = normalize(rendered_text(context))
    # the engine's own numbers (turn numbers, counts of objects, errors and dropped lines) are not values;
    # stripped before the fixed wording, which would otherwise take the "at turns" cue away
    text = re.sub(r"\bat turns? [\d, ]+\d", " ", text)
    text = re.sub(r"\b(?:and|produced|updated|added|changed|exactly|at least|not listed:) \d+\b", " ", text)
    text = re.sub(r"\b\d+ (?:found|more like them|more|other)\b", " ", text)
    return _TEMPLATE_PATTERN.sub(" ", text)


def _request_text(context: dict, episode_messages: list[dict] | None) -> str:
    for message in episode_messages or []:
        if message.get("role") == "user":
            content = message.get("content") or ""
            return normalize(content if isinstance(content, str) else str(content))
    return ""


def leaks(record: Record, context: dict, *, episode_messages: list[dict] | None = None,
          task_assertions: list[dict] | None = None, tool_names: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """`tool_names`: the task's tools (known to the attempt). Schema-naming keys (field, property, column,
    action_key, …) are not values and are skipped by the assertion-literal detector, as in the Record."""
    from ..record.native import SCHEMA_KEYS
    text = scanned_text(context)
    request = _request_text(context, episode_messages)
    found = []
    for item in record.items:
        if item.outcome != "failed":
            continue
        for key, value in {**item.address, **item.content}.items():
            if value.observed or value.written:
                continue
            raw = value.raw
            if isinstance(raw, (list, dict)):
                # a compound value leaks only through the elements the attempt neither observed nor
                # wrote (a list mixing seen and written elements has none)
                hidden = set(value.hidden_elements)
                elements = ([v for i, v in enumerate(raw) if str(i) in hidden] if isinstance(raw, list)
                            else [v for k, v in raw.items() if k in hidden])
                forms = [f for v in elements for f in value_forms(v)]
            else:
                forms = list(value.forms)
            for form in forms:
                if (len(form) >= 3 or (len(form) == 2 and form.isdigit())) and Corpus._form_in(form, text) and not Corpus._form_in(form, request):
                    found.append({"item": item.id, "key": key, "form": form, "via": "value"})
                    break
    if task_assertions and episode_messages is not None:
        corpus = Corpus(episode_messages, tool_names)
        for assertion in task_assertions:
            for atom in atoms({k: v for k, v in assertion.items() if k != "type" and k not in SCHEMA_KEYS}):
                if isinstance(atom, bool) or atom is None:
                    continue
                for form in value_forms(atom):
                    if (len(form) < 4 and not (len(form) >= 2 and form.isdigit())) or corpus.is_observed(atom) or corpus.is_written(atom):
                        continue
                    if Corpus._form_in(form, text) and not Corpus._form_in(form, request):
                        found.append({"item": None, "key": assertion["type"], "form": form, "via": "assertion"})
                        break
    return found
