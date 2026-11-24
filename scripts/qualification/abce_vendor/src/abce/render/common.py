"""Shared rendering pieces: value visibility, item selection, placement, the context dict.

Both renderers (raw, plain) consume the same Record and the same switches; nothing here depends
on the style. The context dict shape is the training runner's: `first_message` (string) and
`notes` (turn number -> text appended to the tool/user message before that assistant turn).
"""

from __future__ import annotations

from typing import Any

from ..record.model import Item, Record, Value

FAILED_HEADING = "The following is feedback from your unsuccessful earlier attempt:"
SUCCESS_HEADING = "The following is feedback on your earlier attempt:"
CLOSING = "Correctly solve the original question."
NOTE_PREFIX = "Note on your next action: "
UNSEEN = "a value this attempt never observed"
PLACEMENTS = ("first", "decision_point", "notes", "both")
STYLES = ("raw", "plain", "plain_matched")
KIND_ORDER = {"prohibited": 0, "changed": 1, "tool_error": 2, "missing": 3, "budget": 4}


def visible(value: Value, expected_values: bool) -> bool:
    """A value may be shown when the attempt saw it (request or tool result) or wrote it itself
    (its own arguments: showing it back reveals nothing new), or when expected values are shown."""
    return expected_values or value.observed or value.written


def shown_raw(value: Value, expected_values: bool) -> Any:
    """The raw-mode form of a parameter value: itself, or a placeholder when hidden."""
    return value.raw if visible(value, expected_values) else f"[{UNSEEN}]"


def item_kind(item: Item) -> str:
    if item.type == "tool_error":
        return "tool_error"
    if item.polarity == "forbid":
        return "prohibited"
    if item.polarity == "keep":
        return "changed"
    return "missing"


def facts(record: Record) -> list[Item]:
    """Items a teacher context is about: failed items, in a stable order: harms first, then within a
    kind the located items (by turn) before the unlocated, native evidence before manifest."""
    def ordinal(item_id: str) -> tuple:
        head, _, rest = item_id.partition(":")
        first = rest.split(":")[0]
        return (head, int(first) if first.isdigit() else 10**6, rest)  # native:3 before native:10

    chosen = [i for i in record.items if i.outcome == "failed"]
    chosen.sort(key=lambda i: (KIND_ORDER[item_kind(i)], 0 if i.location.kind == "at_call" else 1,
                               i.location.turn or 10**6, 0 if i.origin == "native" else 1, ordinal(i.id)))
    return chosen


def located_call(record: Record, item: Item):
    if item.location.kind != "at_call" or item.location.call is None:
        return None
    ref = item.location.call
    return next((c for c in record.calls if c.turn == ref.turn and c.index == ref.index), None)


def fact_dicts(record: Record, items: list[Item], texts: dict[str, str | None] | None = None) -> list[dict]:
    """The runner-facing fact list: one dict per failed item with at least `id` and `kind`; located
    facts carry `turn`, `call` (index within the turn) and `name` (the tool); `text` is the plain
    line rendered for it (None when the renderer withheld it)."""
    out = []
    for item in items:
        call = located_call(record, item)
        out.append({"id": item.id, "kind": item_kind(item), "origin": item.origin, "type": item.type,
                    "check_id": item.check_id, "turn": item.location.turn if item.location.kind == "at_call" else None,
                    "call": call.index if call else None, "name": call.tool if call else None,
                    "location": item.location.kind, "text": (texts or {}).get(item.id)})
    return out


def located_turn(item: Item) -> int | None:
    if item.location.kind in ("at_call",) and item.location.turn is not None:
        return item.location.turn
    return None


def decision_turn(record: Record, items: list[Item]) -> int:
    """The first turn with a located failure, else turn 1 (the handover placed feedback before an action)."""
    turns = [t for t in (located_turn(i) for i in items) if t is not None]
    return min(turns) if turns else 1


def final_reply_turn(episode_messages: list[dict] | None, record: Record) -> int | None:
    """The assistant-turn number of the attempt's final reply: the last assistant message, when it carries no tool call
    and an earlier assistant message did (the reply after the last tool call). None when there is no such reply (the
    attempt was cut off with a call pending, made no call at all, or the messages do not match the Record's turns)."""
    assistants = [m for m in episode_messages or [] if m.get("role") == "assistant"]
    if not assistants or len(assistants) != record.turns:
        return None
    if assistants[-1].get("tool_calls") or not any(m.get("tool_calls") for m in assistants[:-1]):
        return None
    if any(c.turn == len(assistants) for c in record.calls):
        return None
    return len(assistants)


def request_text(record: Record, episode_messages: list[dict] | None) -> str:
    if episode_messages:
        for message in episode_messages:
            if message.get("role") == "user":
                content = message.get("content") or ""
                return content if isinstance(content, str) else str(content)
    return ""


def skip_reason(record: Record, items: list[Item], solution: dict | None) -> str | None:
    if not items and record.success and solution is None:
        return "success_without_facts"
    if not items and not record.success:
        return "failed_without_facts"
    return None


def bad_spans(items: list[Item], record: Record | None = None) -> list[dict]:
    """Located harms, broken records and tool errors: {turn, call, name, kind, item} (+ `call_index`
    for existing callers)."""
    spans = []
    for item in items:
        if item.location.kind == "at_call" and item.location.call is not None and item_kind(item) in ("prohibited", "changed", "tool_error"):
            call = located_call(record, item) if record is not None else None
            spans.append({"kind": item_kind(item), "turn": item.location.call.turn, "call": item.location.call.index,
                          "call_index": item.location.call.index, "name": call.tool if call else None, "item": item.id})
    return spans


def assemble(record: Record, request: str, body_first: str | None, notes: dict[int, str], *, style: str,
             placement: str, expected_values: bool, items: list[Item], solution_episode: str | None,
             reason: str | None, texts: dict[str, str | None] | None = None, coverage: dict | None = None) -> dict:
    first = request
    if body_first:
        first = (request + "\n\n" + body_first) if request else body_first
    return {
        "episode_id": record.episode_id, "task": record.task, "style": style, "placement": placement,
        "expected_values": expected_values, "first_message": first, "notes": {int(k): v for k, v in notes.items()},
        "skip_reason": reason, "facts": fact_dicts(record, items, texts), "bad_spans": bad_spans(items, record),
        "solution_episode": solution_episode, "success": record.success,
        "coverage": coverage or {"rendered": len(items), "suppressed": [], "dropped": []},
    }
