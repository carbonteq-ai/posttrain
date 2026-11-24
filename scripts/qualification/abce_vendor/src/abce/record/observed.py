"""What the attempt observed, and whether a value is observed (docs/design.md L0, leak test).

Observed = the request (user messages) and every tool result. Text the attempt wrote itself in
tool arguments is tracked separately as `written`, so a value that only appears in a tool result
because the attempt put it there is not counted as observed (the predecessor's rule J).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_WS = re.compile(r"\s+")
_NUM = re.compile(r"^\$?-?\d[\d,]*(\.\d+)?%?$")


def normalize(text: str) -> str:
    text = re.sub(r"(\d),(\d)", r"\1\2", text)  # 1,234 -> 1234 (handler leniency)
    text = re.sub(r"(\d+)\.0+%", r"\1%", text)
    text = re.sub(r"(\d)\.0+(?!\d)", r"\1", text)  # 31000.0 -> 31000 (observability, not grading)
    return _WS.sub(" ", text.lower()).strip()


def value_forms(value: Any) -> tuple[str, ...]:
    if isinstance(value, bool):
        return (str(value).lower(),)
    if isinstance(value, (int, float)):
        forms = [str(value)]
        if float(value).is_integer():
            forms += [str(int(value)), f"{int(value):,}"]
        else:
            forms += [f"{value:,.2f}", f"{value:.2f}"]
        return tuple(dict.fromkeys(normalize(f) for f in forms))
    if value is None:
        return ()
    if isinstance(value, (dict, list)):
        return tuple(dict.fromkeys(f for v in atoms(value) for f in value_forms(v)))
    return (normalize(str(value)),)


def atoms(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [a for v in value.values() for a in atoms(v)]
    if isinstance(value, (list, tuple)):
        return [a for v in value for a in atoms(v)]
    return [value]


def _strings(value: Any) -> list[str]:
    return [str(a) for a in atoms(value) if isinstance(a, (str, int, float)) and not isinstance(a, bool)]


class Corpus:
    """Observed text and written text of one episode."""

    def __init__(self, messages: list[dict], tool_names: list[str] | tuple[str, ...] = ()):
        observed_parts: list[str] = []
        written_parts: list[str] = []
        # The attempt knows its tools: their names and name parts count as observed.
        for name in tool_names:
            observed_parts.append(name + " " + " ".join(name.split("_")) + " " + "_".join(name.split("_")[1:]))
        for message in messages:
            role = message.get("role")
            if role == "assistant":
                for call in message.get("tool_calls") or []:
                    function = call.get("function") or {}
                    raw = call.get("arguments", function.get("arguments"))
                    try:
                        args = json.loads(raw) if isinstance(raw, str) else raw
                    except ValueError:
                        args = raw
                    written_parts.extend(_strings(args))
                continue
            if role not in ("user", "tool"):
                continue
            content = message.get("content") or ""
            if not isinstance(content, str):
                content = json.dumps(content)
            if role == "tool":
                # Drop echoes of the attempt's own free text before counting a result as observed.
                for text in written_parts:
                    if len(text) >= 4:
                        content = content.replace(text, " ").replace(json.dumps(text)[1:-1], " ")
            observed_parts.append(content)
        self.observed = normalize("\n".join(observed_parts))
        self.written = normalize("\n".join(written_parts))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.observed.encode()).hexdigest()

    def is_observed(self, value: Any) -> bool:
        """A scalar is observed when any of its forms occurs; a list/dict when every element is."""
        if isinstance(value, (list, tuple)):
            return all(self.is_observed(v) for v in value)
        if isinstance(value, dict):
            return all(self.is_observed(v) for v in value.values())
        forms = value_forms(value)
        return not forms or any(self._form_in(f, self.observed) for f in forms)

    def is_written(self, value: Any) -> bool:
        if isinstance(value, (list, tuple)):
            return all(self.is_written(v) for v in value)
        if isinstance(value, dict):
            return all(self.is_written(v) for v in value.values())
        return any(self._form_in(f, self.written) for f in value_forms(value))

    def hidden_elements(self, value: Any) -> tuple[str, ...]:
        """Elements of a compound value that were neither observed nor written."""
        if isinstance(value, (list, tuple)):
            return tuple(str(i) for i, v in enumerate(value) if not (self.is_observed(v) or self.is_written(v)))
        if isinstance(value, dict):
            return tuple(k for k, v in value.items() if not (self.is_observed(v) or self.is_written(v)))
        return ()

    @staticmethod
    def _form_in(form: str, text: str) -> bool:
        """A form occurs in the text as whole tokens: a number not inside a longer number or id, a word or
        phrase not inside a longer word ("new" is not in "renewed", "li@x" not in "ali@x")."""
        if not form:
            return True
        if _NUM.match(form.replace(" ", "")):
            # a bare number must stand alone, not be a substring of a longer number/id
            return re.search(r"(?<!\d)(?<!\d\.)" + re.escape(form) + r"(?!\.?\d)", text) is not None
        if form not in text:
            return False
        # a JSON-escaped newline or tab ("\\n") next to the form is a boundary too (texts are often json.dumps output);
        # a phrase's last word may carry an inflection ("8 registrant" is in "8 registrants")
        tail = r"(?:(?![a-z0-9])|(?=\\n)|(?=\\t)|(?=\\r)|(?=(?:s|es|ed|ing)(?![a-z0-9])))" if " " in form else r"(?:(?![a-z0-9])|(?=\\n)|(?=\\t)|(?=\\r))"
        return re.search(r"(?:(?<![a-z0-9])|(?<=\\n)|(?<=\\t)|(?<=\\r))" + re.escape(form) + tail, text) is not None
