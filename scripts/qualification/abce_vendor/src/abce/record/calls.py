"""Tool calls of an episode, with results, errors and the tool-server invocation that ran them.

The invocation -> turn mapping is the environment's own (`manifest_step_credit.invocation_turns`)
when importable; the call index within the turn comes from the harness event's
`emitted_call_index`. Captured-action digests come from the raw-action envelopes.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .episode import Episode
from .model import Call

READ_VERBS = ("find", "get", "list", "search", "lookup", "query", "read", "fetch", "retrieve", "describe")
_FURTHER = re.compile(r"\s*For further information visit \S+")


def tool_error_text(content: Any) -> str | None:
    """The error string of a tool result, or None (same rule as the environment's turn rewards)."""
    text = content if isinstance(content, str) else json.dumps(content)
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            value = json.loads(stripped)
        except ValueError:
            value = None
        if isinstance(value, dict) and (
            value.get("success") is False or (set(value) <= {"error", "success", "details"} and "error" in value)
        ):
            return str(value.get("error") or value.get("details") or stripped)
    lowered = stripped[:80].lower()
    if lowered.startswith(("error", "tool error", "invalid", "unknown tool")):
        return stripped
    if lowered.startswith("unsupported arguments for") and "no action was executed" in stripped.lower():
        # the harness rejected the call before dispatch (undeclared argument names): nothing ran
        return stripped
    return None


# A suggestion clause a tool appends after its own message: a short label ("Known spreadsheets",
# "Worksheets", "Available channels", "Did you mean"), a colon, and a bracketed list of the objects it
# knows, or a cue word with a comma-separated list running to the end. Detected by shape, not by tool.
# The clause must follow a finished sentence of the tool's own message, so a message that *starts*
# with "<label>: [...]" (a Python exception rendered as "Error executing tool x: [<class ...>]") keeps it.
_CUE = r"(?:known|available|existing|valid|possible|allowed|accepted|options|choose from|try one of|must be one of)"
_LABEL = r"[A-Za-z][\w'/-]*(?: [A-Za-z][\w'/-]*){0,2}"
_SUGGESTION = re.compile(
    r"(?<=[.!?;])\s+(?:"
    r"(?:" + _CUE + r"(?: " + _LABEL + r")?|" + _LABEL + r")\s*(?:are|is)?\s*:\s*[\[({].*"  # label + bracketed list
    r"|" + _CUE + r"(?: " + _LABEL + r")?\s*(?:are|is)?\s*:\s*[^\s].*"  # cue word + list to the end
    r"|did you mean\b.*"
    r")$",
    re.IGNORECASE | re.DOTALL,
)


def error_message(error: str) -> str:
    """The tool's error without the trailing suggestion or enumeration of known objects it may append
    ("Spreadsheet 'Visa' not found. Known spreadsheets: ['Visa Tracking (ss_visa)']" -> "Spreadsheet
    'Visa' not found."). The error's own message is kept exactly; only the list is dropped, so a line
    never hands the teacher the title it should have looked up."""
    text = re.sub(r"\s+", " ", error).strip()
    return _SUGGESTION.sub("", text).rstrip()


def result_is_empty(content: Any) -> bool:
    text = content if isinstance(content, str) else json.dumps(content)
    stripped = text.strip()
    if stripped in ("[]", "{}", "", "null"):
        return True
    try:
        value = json.loads(stripped)
    except ValueError:
        return False
    if isinstance(value, dict):
        for key in ("count", "total_count", "results", "items", "records", "rows", "messages", "data"):
            if key in value:
                v = value[key]
                return v in (0, [], None, {})
    return value == []


def parse_arguments(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except (ValueError, TypeError):
        return {"_raw": raw}
    return value if isinstance(value, dict) else {"_value": value}


def app_of(tool: str) -> str:
    for prefix in ("google_sheets", "google_calendar", "google_drive", "google_ads", "zoho_desk",
                   "facebook_pages", "facebook_lead_ads", "facebook_conversions", "linkedin_ads",
                   "linkedin_conversions", "linkedin_leadgen_forms"):
        if tool.startswith(prefix + "_"):
            return prefix
    return tool.split("_", 1)[0]


def raw_actions(episode: Episode) -> dict[str, dict]:
    """Captured action per invocation id (tool, arguments, before/after digests, status)."""
    actions: dict[str, dict] = {}
    for event in episode.events:
        if event.get("source") != "tool_server" or event.get("phase") not in ("returned", "raised"):
            continue
        receipt = json.loads(event["receipt_json"])
        for encoded in receipt.get("evidence_json") or ():
            envelope = json.loads(encoded)
            if isinstance(envelope, dict) and isinstance(envelope.get("action"), dict):
                actions[event["invocation_id"]] = dict(envelope["action"])
    return actions


def invocation_links(episode: Episode, node_turns: dict[int, int]) -> tuple[dict[str, tuple[int, int]], bool]:
    """invocation id -> (turn, call index) through the issuing harness execution; (mapping, complete)."""
    executions: dict[str, tuple[int, int]] = {}
    dispatches = []
    for event in episode.events:
        if event.get("source") in ("harness", "interceptor") and isinstance(event.get("node_index"), int):
            executions.setdefault(str(event.get("execution_id")), (event["node_index"], event.get("emitted_call_index") or 0))
        elif event.get("source") == "tool_server" and event.get("phase") == "dispatch":
            receipt = json.loads(event["receipt_json"])
            dispatches.append((event["invocation_id"], receipt.get("parent_execution_id")))
    mapping: dict[str, tuple[int, int]] = {}
    complete = True
    for invocation, parent in dispatches:
        link = executions.get(str(parent))
        if link is None or link[0] not in node_turns:
            complete = False
            continue
        mapping[invocation] = (node_turns[link[0]], link[1])
    try:
        from automationbench_v1.manifest_step_credit import invocation_turns

        native = invocation_turns([], episode.events, node_turns)
        if native is None:
            complete = False
        else:
            for key, value in mapping.items():
                if native.get(key) != value[0]:
                    raise ValueError("invocation turn mapping disagrees with the environment")
    except ImportError:
        pass
    return mapping, complete


def episode_calls(episode: Episode) -> tuple[list[Call], dict[int, int], int, bool]:
    """(calls, node_index -> turn, number of assistant turns, invocation mapping complete)."""
    results: dict[str, Any] = {}
    result_names: dict[str, Any] = {}
    for node in episode.nodes:
        message = node.get("message") or {}
        if message.get("role") == "tool":
            results[message.get("tool_call_id")] = message.get("content")
            result_names[message.get("tool_call_id")] = message.get("name")
    node_turns: dict[int, int] = {}
    turn = 0
    raw: list[dict] = []
    for index, node in enumerate(episode.nodes):
        message = node.get("message") or {}
        if message.get("role") != "assistant":
            continue
        turn += 1
        node_turns[index] = turn
        for position, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            name = call.get("name") or function.get("name") or ""
            arguments = call.get("arguments", function.get("arguments"))
            result = results.get(call.get("id"))
            error = tool_error_text(result) if result is not None else None
            if error:
                error = re.sub(r"\s+", " ", _FURTHER.sub("", error)).strip()
            raw.append({"turn": turn, "index": position, "id": call.get("id"), "tool": name,
                        "args": parse_arguments(arguments), "result": result,
                        "result_name": result_names.get(call.get("id")), "error": error})
    links, complete = invocation_links(episode, node_turns)
    actions = raw_actions(episode)
    by_site = {v: k for k, v in links.items()}
    calls = []
    for item in raw:
        invocation = by_site.get((item["turn"], item["index"]))
        action = actions.get(invocation) if invocation else None
        empty = item["error"] is None and item["result"] is not None and result_is_empty(item["result"]) and any(
            v in item["tool"] for v in READ_VERBS)
        result_name = item["result_name"]
        calls.append(Call(
            turn=item["turn"], index=item["index"], id=item["id"], tool=item["tool"], app=app_of(item["tool"]),
            args=item["args"], result=item["result"], result_name=result_name, error=item["error"], empty=empty,
            invocation_id=invocation,
            before_digest=action.get("before_digest") if action else None,
            after_digest=action.get("after_digest") if action else None,
            mutated=(action.get("before_digest") != action.get("after_digest")) if action else None,
        ))
    return calls, node_turns, turn, complete
