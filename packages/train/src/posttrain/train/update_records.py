"""Immutable references for policy updates; native traces remain replay authority.

These records contain no token arrays or backend tensors. A native adapter must
verify their retained references before scoring; validation here establishes
internal coordinate and membership consistency, not native evidence authenticity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from typing import Literal


class InvalidPolicyUpdate(ValueError):
    """A policy update cannot preserve its declared evidence contract."""


def require_identity(*values: str) -> None:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise InvalidPolicyUpdate("non-empty immutable identities are required")


def record_digest(record: object) -> str:
    """Canonical structural identity, independent of Python's randomized hash."""
    if not is_dataclass(record) or isinstance(record, type):
        raise TypeError("digest requires a dataclass record")
    payload = json.dumps(asdict(record), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class ActionRef:
    episode_id: str
    branch_id: str
    turn_id: str
    token_index: int

    def __post_init__(self) -> None:
        require_identity(self.episode_id, self.branch_id, self.turn_id)
        if type(self.token_index) is not int or self.token_index < 0:
            raise InvalidPolicyUpdate("action requires an original non-negative token index")


@dataclass(frozen=True, slots=True)
class ConditioningView:
    id: str
    native_ref: str
    token_ids_ref: str
    attention_ref: str
    positions_ref: str
    template_revision: str
    digest: str
    context_tokens: int

    def __post_init__(self) -> None:
        require_identity(
            self.id, self.native_ref, self.token_ids_ref, self.attention_ref,
            self.positions_ref, self.template_revision, self.digest,
        )
        if type(self.context_tokens) is not int or self.context_tokens < 1:
            raise InvalidPolicyUpdate("conditioning view requires its actual context size")


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """Eligible sampled assistant action and its exact original conditioning view."""

    action: ActionRef
    conditioning_id: str
    native_ref: str

    def __post_init__(self) -> None:
        require_identity(self.conditioning_id, self.native_ref)


@dataclass(frozen=True, slots=True)
class ActionInterval:
    """Half-open interval in one original episode/branch/turn coordinate system."""

    start: ActionRef
    end: int

    def __post_init__(self) -> None:
        if type(self.end) is not int or self.end <= self.start.token_index:
            raise InvalidPolicyUpdate("action interval must be non-empty and half-open")

    def actions(self) -> tuple[ActionRef, ...]:
        return tuple(
            ActionRef(self.start.episode_id, self.start.branch_id, self.start.turn_id, index)
            for index in range(self.start.token_index, self.end)
        )


@dataclass(frozen=True, slots=True)
class SemanticSpan:
    id: str
    role: str
    projection_revision: str
    action_intervals: tuple[ActionInterval, ...]

    def __post_init__(self) -> None:
        require_identity(self.id, self.role, self.projection_revision)
        if not self.action_intervals:
            raise InvalidPolicyUpdate("semantic span requires original action intervals")

    def actions(self) -> tuple[ActionRef, ...]:
        # Set selection is independent of credit combination. Overlap is legal.
        return tuple(sorted({action for interval in self.action_intervals for action in interval.actions()}))


@dataclass(frozen=True, slots=True)
class PopulationRelation:
    """Named estimator membership; expected members prove declared completeness."""

    id: str
    kind: str
    members: tuple[ActionRef, ...]
    completeness: Literal["complete", "partial"]
    expected_members: tuple[ActionRef, ...]

    def __post_init__(self) -> None:
        require_identity(self.id, self.kind)
        if self.completeness not in {"complete", "partial"}:
            raise InvalidPolicyUpdate("unknown population completeness")
        if not self.members or not self.expected_members:
            raise InvalidPolicyUpdate("population membership must be non-empty")
        if len(set(self.members)) != len(self.members) or len(set(self.expected_members)) != len(self.expected_members):
            raise InvalidPolicyUpdate("population membership cannot duplicate actions")
        if not set(self.members) <= set(self.expected_members):
            raise InvalidPolicyUpdate("population members exceed declared expected membership")
        if self.completeness == "complete" and set(self.members) != set(self.expected_members):
            raise InvalidPolicyUpdate("complete population lacks expected members")


@dataclass(frozen=True, slots=True)
class PolicyVersions:
    sampler: str
    old_score: str
    current: str
    reference: str | None

    def __post_init__(self) -> None:
        require_identity(self.sampler, self.old_score, self.current)
        if self.reference is not None:
            require_identity(self.reference)


@dataclass(frozen=True, slots=True)
class PopulationSnapshot:
    id: str
    native_evidence_ref: str
    native_evidence_digest: str
    actions: tuple[ActionRecord, ...]
    conditioning: tuple[ConditioningView, ...]
    spans: tuple[SemanticSpan, ...]
    relations: tuple[PopulationRelation, ...]
    versions: PolicyVersions
    selector_digest: str

    def __post_init__(self) -> None:
        require_identity(self.id, self.native_evidence_ref, self.native_evidence_digest, self.selector_digest)
        eligible = {record.action for record in self.actions}
        if not eligible or len(eligible) != len(self.actions):
            raise InvalidPolicyUpdate("snapshot requires unique eligible original actions")
        contexts = {view.id for view in self.conditioning}
        if len(contexts) != len(self.conditioning):
            raise InvalidPolicyUpdate("conditioning identities must be unique")
        if any(record.conditioning_id not in contexts for record in self.actions):
            raise InvalidPolicyUpdate("action has no retained actual conditioning view")
        for records in (self.spans, self.relations):
            if len({record.id for record in records}) != len(records):
                raise InvalidPolicyUpdate("annotation and relation identities must be unique")
        if any(not set(span.actions()) <= eligible for span in self.spans):
            raise InvalidPolicyUpdate("span addresses absent or ineligible sampled actions")
        if any(not set(relation.members) <= eligible for relation in self.relations):
            raise InvalidPolicyUpdate("relation addresses absent sampled actions")

    @property
    def digest(self) -> str:
        return record_digest(self)

    def select_roles(self, roles: tuple[str, ...]) -> tuple[ActionRef, ...]:
        """Union every supplied annotation of the named roles (population-independent selection)."""
        return tuple(sorted({action for span in self.spans if span.role in roles for action in span.actions()}))

    def select_spans(self, span_ids: tuple[str, ...]) -> tuple[ActionRef, ...]:
        """Union selected annotations without duplicating loss contributions."""
        by_id = {span.id: span for span in self.spans}
        unknown = set(span_ids) - by_id.keys()
        if unknown:
            raise InvalidPolicyUpdate(f"unknown semantic spans: {sorted(unknown)}")
        return tuple(sorted({action for span_id in span_ids for action in by_id[span_id].actions()}))


@dataclass(frozen=True, slots=True)
class ActionSelection:
    """Population-frozen union of supplied spans, or all eligible actions.

    ``spans`` names population-specific annotation IDs; ``roles`` names span
    roles (for example ``reasoning`` or ``answer``) so a catalog selection is
    stable across populations whose span IDs differ.
    """

    mode: Literal["all", "spans", "roles"] = "all"
    span_ids: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (self.mode not in {"all", "spans", "roles"}
                or (self.mode != "spans" and self.span_ids) or (self.mode != "roles" and self.roles)
                or (self.mode == "roles" and not self.roles)):
            raise InvalidPolicyUpdate("selector must declare all actions, supplied spans or span roles")
        if len(set(self.span_ids)) != len(self.span_ids) or len(set(self.roles)) != len(self.roles):
            raise InvalidPolicyUpdate("selector span identities and roles must be unique")
        if any(not isinstance(role, str) or not role.strip() for role in self.roles):
            raise InvalidPolicyUpdate("selector roles must be nonempty names")

    def resolve(self, snapshot: PopulationSnapshot) -> tuple[ActionRef, ...]:
        if self.mode == "all":
            return tuple(record.action for record in snapshot.actions)
        if self.mode == "roles":
            return snapshot.select_roles(self.roles)
        return snapshot.select_spans(self.span_ids)
