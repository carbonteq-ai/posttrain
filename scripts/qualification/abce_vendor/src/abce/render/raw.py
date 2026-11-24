"""Raw style: the handover's feedback format (979c8fe0 `feedback_text`), rendered from the Record.

Text = fixed heading + `json.dumps({"native_checks": [{type, passed, excluded, params}],
"observed_tool_responses": [...]}, sort_keys=True)` + fixed closing; or, with a successful
sibling, the "Correct solution:" block. Differences from the handover are switches, not edits:
`expected_values=False` replaces parameter values the attempt never observed with a placeholder,
and `placement` decides where the text goes. With `expected_values=True`, `placement="decision_point"`
and no manifest items, the text is byte-for-byte the handover's.
"""

from __future__ import annotations

import json
from typing import Any

from ..record.model import Item, Record, Value
from . import common

HEADING = ("The following is native environment feedback from this earlier attempt. "
           "These are hindsight observations, not actions already executed at the original decision point.\n")
CLOSING = ("\nCorrectly continue the original task from the original decision point. "
           "Do not repeat a search unchanged when its response supplies no matching evidence. "
           "Do not invent identifiers or claim completion without evidence.")
SOLUTION_HEADING = ("A previous attempt on this same task satisfied all native checks. "
                    "Use it as a reference, not as a new observation in the student's history.\n"
                    "Correct solution:\n")
SOLUTION_CLOSING = "\nReconsider the next action at the original decision point."


def _params(item: Item, expected_values: bool) -> dict[str, Any]:
    """The recorded parameters verbatim; with expected values hidden, each value the attempt neither
    observed nor wrote is replaced by a placeholder."""
    recorded = item.semantics.get("params")
    values = {**item.address, **item.content}
    if not isinstance(recorded, dict):
        recorded = {k.split("#")[0]: v.raw for k, v in values.items()}
        recorded.update(item.cardinality or {})
        recorded.update(item.control)
    params: dict[str, Any] = {}
    for key, raw in recorded.items():
        value = values.get(key)
        params[key] = raw if value is None else _hide(value, expected_values)
    return params


def _hide(value: Value, expected_values: bool) -> Any:
    if common.visible(value, expected_values):
        return value.raw
    raw = value.raw
    hidden = set(value.hidden_elements)
    if isinstance(raw, list):
        return [f"[{common.UNSEEN}]" if str(i) in hidden else v for i, v in enumerate(raw)]
    if isinstance(raw, dict):
        return {k: (f"[{common.UNSEEN}]" if k in hidden else v) for k, v in raw.items()}
    return f"[{common.UNSEEN}]"


def native_checks(record: Record, expected_values: bool) -> list[dict]:
    rows = []
    for item in record.items:
        if item.origin != "native" or item.type == "tool_error":
            continue
        passed = item.semantics.get("passed", item.outcome == "passed")
        rows.append({"type": item.type, "passed": passed, "excluded": item.outcome == "excluded",
                     "params": _params(item, expected_values)})
    return rows


def _candidate_label(item: Item, expected_values: bool) -> Any:
    if not item.candidate:
        return None
    if expected_values or item.candidate.get("observed", True):
        return item.candidate["label"]
    return None  # an unobserved candidate identity is a value too


def manifest_checks(record: Record, expected_values: bool = True) -> list[dict]:
    rows = []
    for index, item in enumerate(i for i in record.items if i.origin == "manifest" and i.outcome != "abstained"
                                 and not i.semantics.get("native_disagrees")):
        # check ids are grader internals and often contain expected-value words: not shown.
        rows.append({"check": index, "role": "harm" if item.polarity == "forbid" else "goal",
                     "passed": item.outcome == "passed", "turn": item.location.turn,
                     "candidate": _candidate_label(item, expected_values)})
    return rows


def tool_responses(record: Record, before_turn: int, episode_messages: list[dict] | None) -> list[dict]:
    """The tool responses of the turn before `before_turn` (first-message / notes placements)."""
    if before_turn <= 1:
        return []
    return [{"tool": c.result_name, "response": c.result} for c in record.calls if c.turn == before_turn - 1 and c.result is not None]


def action_observations(record: Record, turn: int) -> list[dict]:
    """The handover's `observations` for the action at assistant turn `turn` (1-based): the responses
    to that turn's own tool calls; for a turn without calls, the last tool response before it."""
    calls = [c for c in record.calls if c.turn == turn]
    if calls:  # a turn with calls shows their responses (none if the episode ended before they came back)
        return [{"tool": c.result_name, "response": c.result} for c in calls if c.result is not None]
    earlier = [c for c in record.calls if c.turn < turn and c.result is not None]
    return [{"tool": earlier[-1].result_name, "response": earlier[-1].result}] if earlier else []


def feedback_for_action(record: Record, turn: int, *, expected_values: bool, solution: dict | None,
                        include_manifest: bool = False) -> str:
    """The handover's feedback text for one action (`feedback_text`), from the Record."""
    if solution is not None:
        return SOLUTION_HEADING + solution["text"] + SOLUTION_CLOSING
    payload: dict[str, Any] = {"native_checks": native_checks(record, expected_values),
                               "observed_tool_responses": action_observations(record, turn)}
    if include_manifest:
        rows = manifest_checks(record, expected_values)
        if rows:
            payload["manifest_checks"] = rows
    return HEADING + json.dumps(payload, sort_keys=True) + CLOSING


def feedback(record: Record, *, expected_values: bool, before_turn: int, episode_messages: list[dict] | None,
             include_manifest: bool, solution: dict | None) -> str:
    if solution is not None:
        return SOLUTION_HEADING + solution["text"] + SOLUTION_CLOSING
    payload: dict[str, Any] = {"native_checks": native_checks(record, expected_values),
                               "observed_tool_responses": tool_responses(record, before_turn, episode_messages)}
    if include_manifest:
        rows = manifest_checks(record, expected_values)
        if rows:
            payload["manifest_checks"] = rows
    return HEADING + json.dumps(payload, sort_keys=True) + CLOSING


def render(record: Record, *, placement: str, expected_values: bool, episode_messages: list[dict] | None = None,
           solution: dict | None = None, include_manifest: bool = True) -> dict:
    items = common.facts(record)
    reason = common.skip_reason(record, items, solution)
    request = common.request_text(record, episode_messages)
    notes: dict[int, str] = {}
    body_first = None
    turn = common.decision_turn(record, items)
    if placement in ("first", "both"):
        body_first = feedback(record, expected_values=expected_values, before_turn=1, episode_messages=episode_messages,
                              include_manifest=include_manifest, solution=solution)
    if placement == "decision_point":
        notes[turn] = feedback(record, expected_values=expected_values, before_turn=turn, episode_messages=episode_messages,
                               include_manifest=include_manifest, solution=solution)
    if placement in ("notes", "both"):
        for t in sorted({t for t in (common.located_turn(i) for i in items) if t is not None}):
            notes[t] = feedback(record, expected_values=expected_values, before_turn=t, episode_messages=episode_messages,
                                include_manifest=include_manifest, solution=solution)
        if placement == "notes" and not notes:
            notes[turn] = feedback(record, expected_values=expected_values, before_turn=turn,
                                   episode_messages=episode_messages, include_manifest=include_manifest, solution=solution)
    return common.assemble(record, request, body_first, notes, style="raw", placement=placement,
                           expected_values=expected_values, items=items,
                           solution_episode=solution["episode_id"] if solution else None, reason=reason)
