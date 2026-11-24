"""The Record: render-independent account of one episode (docs/design.md §2).

Plain pydantic models, JSON round-trippable. Nothing in here is prose.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer

Polarity = Literal["require", "forbid", "keep"]
Outcome = Literal["passed", "failed", "excluded", "abstained", "inapplicable"]
Origin = Literal["native", "manifest"]
LocationKind = Literal["at_call", "after_turn", "end_of_attempt", "unknown"]


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Value(Frozen):
    raw: Any
    observed: bool  # the attempt saw this value in the request or a tool result
    written: bool = False  # the attempt wrote this value in one of its own arguments
    forms: tuple[str, ...] = ()  # normalised text forms a handler would accept
    hidden_elements: tuple[str, ...] = ()  # list indexes / dict keys of a compound value the attempt never saw or wrote


class CallRef(Frozen):
    turn: int  # 1-based assistant turn ordinal
    index: int  # position among that turn's tool calls


class Call(Frozen):
    turn: int
    index: int
    id: str | None
    tool: str
    app: str
    args: dict[str, Any]
    result: Any = None  # the tool message content verbatim (usually a string)
    result_name: str | None = None  # the tool message's `name`
    error: str | None = None  # None when the result is not an error
    empty: bool = False  # a read that returned nothing
    invocation_id: str | None = None
    before_digest: str | None = None
    after_digest: str | None = None
    mutated: bool | None = None  # before != after; None when no snapshot


class Location(Frozen):
    kind: LocationKind
    turn: int | None = None
    call: CallRef | None = None
    source: str | None = None  # how it was located: flip | guard_occurrence | witness | cause | none


class ProbeResult(Frozen):
    status: Literal["pending", "full", "partial", "none"] = "pending"
    reason: str | None = None
    failing_clauses: tuple[str, ...] = ()
    required_keys: tuple[str, ...] = ()
    flips: tuple[str, ...] = ()  # invocation ids after which the result changed
    flip_direction: Literal["pass_to_fail", "fail_to_pass"] | None = None
    offending_object: dict[str, Any] | None = None  # {collection, index, fields}
    witness_object: dict[str, Any] | None = None
    initial_result: bool | None = None
    final_result: bool | None = None
    found_count: int | None = None
    # Whether an object with the item's identity (address) exists on the final world: True when the
    # grader said so on an address-only / content-neutral variant, False when nothing the attempt
    # added or called carries the address, None when unsure (then no existence claim is rendered).
    identity_present: bool | None = None
    # Whether an object the attempt itself added carries the address (True), or nothing the attempt
    # added or successfully called does and no container the address names is referred to by an
    # added object (False: "the attempt sent no X <address>" is safe), or unsure (None).
    identity_added: bool | None = None
    offending_invocation: str | None = None  # the call after which a required effect carried forbidden content
    # For a failing clause whose value is a list: the indexes of the elements that fail on their own
    # (the others are present), so a line names only what is missing.
    missing_elements: dict[str, tuple[int, ...]] = Field(default_factory=dict)
    # judge 37/69/70/121/136: the record the check names by its id exists on the final world with other values for the
    # check's other address keys (its current group): {key: current value}; None when not probed so
    id_record_fields: dict[str, Any] | None = None
    # engine-fix 9 (pattern 2): the grader's count of a cardinality check's objects on the initial world (the SENT messages
    # the task started with); None when not probed or the grader could not say
    initial_count: int | None = None

    @model_serializer(mode="wrap")
    def _without_unset_id_record(self, handler):
        """A probe without `id_record_fields` / `initial_count` serializes exactly as before the fields existed (records
        written by this engine stay readable by earlier ones: the byte-for-byte tests replay them on older sources)."""
        data = handler(self)
        if isinstance(data, dict):
            for key in ("id_record_fields", "initial_count"):
                if data.get(key) is None:
                    data.pop(key, None)
        return data


class EffectRef(Frozen):
    app: str
    object: str
    kind: str  # send | post | exist | update | field | has | count | flag | read | action | other


class Item(Frozen):
    id: str
    origin: Origin
    type: str  # native type or manifest operator
    check_id: str | None = None
    outcome: Outcome
    polarity: Polarity
    polarity_source: str  # registry_marker | name | manifest_role | probe
    effect: EffectRef
    address: dict[str, Value] = Field(default_factory=dict)
    content: dict[str, Value] = Field(default_factory=dict)
    cardinality: dict[str, Any] | None = None
    control: dict[str, Any] = Field(default_factory=dict)  # scored / excluded flags
    unclassified_keys: tuple[str, ...] = ()
    candidate: dict[str, Any] | None = None
    location: Location = Location(kind="unknown")
    probe: ProbeResult = ProbeResult()
    semantics: dict[str, Any] = Field(default_factory=dict)
    evidence: tuple[str, ...] = ()
    merged_from: tuple[str, ...] = ()


class Record(Frozen):
    schema_version: Literal[1] = 1
    task: str
    episode_id: str
    success: bool
    score: dict[str, float]
    turns: int
    calls: tuple[Call, ...]
    items: tuple[Item, ...]
    observed_digest: str
    budget: dict[str, Any]
    siblings: tuple[dict[str, Any], ...] = ()
    notes: tuple[str, ...] = ()  # builder diagnostics (never rendered)

    def failed_items(self) -> list[Item]:
        return [i for i in self.items if i.outcome == "failed"]
