"""Immutable references for policy updates; native traces remain replay authority.

A population is a table of conditioning views (one per sampled assistant turn)
and a canonical order of their sampled tokens: views in admission order, and
within a view the native token indices in ascending order. A "position" is an
index into that order. Per-token values elsewhere in the engine are arrays
indexed by position; no record exists per token.

These records contain no backend tensors. A native adapter must verify their
retained references before scoring; validation here establishes internal
coordinate and membership consistency, not native evidence authenticity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np


class InvalidPolicyUpdate(ValueError):
    """A policy update cannot preserve its declared evidence contract."""


def require_identity(*values: str) -> None:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise InvalidPolicyUpdate("non-empty immutable identities are required")


def payload_digest(payload: object) -> str:
    """SHA-256 of canonical JSON (sorted keys, compact, finite numbers only)."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def record_digest(record: object) -> str:
    """Canonical structural identity, independent of Python's randomized hash."""
    if not is_dataclass(record) or isinstance(record, type):
        raise TypeError("digest requires a dataclass record")
    return payload_digest(asdict(record))


def frozen_array(values: Any, dtype: Any) -> np.ndarray:
    """A read-only contiguous copy, so records holding it stay immutable."""
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def array_digest(array: np.ndarray) -> str:
    """Identity of an array's dtype, shape and little-endian bytes."""
    data = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
    hasher = hashlib.sha256(f"{data.dtype.str}:{data.shape}".encode())
    hasher.update(data.tobytes())
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class ActionRef:
    """Address of one sampled token: a boundary type for spans, messages and tests."""

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
    """One sampled assistant turn: its exact original context and its sampled tokens."""

    id: str
    native_ref: str
    token_ids_ref: str
    attention_ref: str
    positions_ref: str
    template_revision: str
    digest: str
    context_tokens: int
    episode_id: str
    branch_id: str
    sampled: tuple[int, ...]

    def __post_init__(self) -> None:
        require_identity(
            self.id,
            self.native_ref,
            self.token_ids_ref,
            self.attention_ref,
            self.positions_ref,
            self.template_revision,
            self.digest,
            self.episode_id,
            self.branch_id,
        )
        if type(self.context_tokens) is not int or self.context_tokens < 1:
            raise InvalidPolicyUpdate("conditioning view requires its actual context size")
        if (
            not self.sampled
            or any(type(index) is not int or index < 0 for index in self.sampled)
            or any(left >= right for left, right in zip(self.sampled, self.sampled[1:], strict=False))
        ):
            raise InvalidPolicyUpdate("conditioning view requires ascending unique sampled token indices")


@dataclass(frozen=True, slots=True)
class ActionInterval:
    """Half-open interval in one original episode/branch/turn coordinate system."""

    start: ActionRef
    end: int

    def __post_init__(self) -> None:
        if type(self.end) is not int or self.end <= self.start.token_index:
            raise InvalidPolicyUpdate("action interval must be non-empty and half-open")


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


@dataclass(frozen=True, slots=True)
class PopulationRelation:
    """Named estimator membership over whole turns; expected members prove declared completeness."""

    id: str
    kind: str
    members: tuple[str, ...]
    completeness: Literal["complete", "partial"]
    expected_members: tuple[str, ...]

    def __post_init__(self) -> None:
        require_identity(self.id, self.kind)
        if self.completeness not in {"complete", "partial"}:
            raise InvalidPolicyUpdate("unknown population completeness")
        if not self.members or not self.expected_members:
            raise InvalidPolicyUpdate("population membership must be non-empty")
        if len(set(self.members)) != len(self.members) or len(set(self.expected_members)) != len(self.expected_members):
            raise InvalidPolicyUpdate("population membership cannot duplicate turns")
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


_VIEW_FIELDS = tuple(item.name for item in fields(ConditioningView))


@dataclass(frozen=True, slots=True, eq=False)
class PopulationSnapshot:
    """Frozen admitted population: conditioning views and the canonical token order.

    Identity is ``digest``, computed once from every field. Records holding a
    snapshot compare it by digest, never by structural equality.
    """

    id: str
    native_evidence_ref: str
    native_evidence_digest: str
    conditioning: tuple[ConditioningView, ...]
    spans: tuple[SemanticSpan, ...]
    relations: tuple[PopulationRelation, ...]
    versions: PolicyVersions
    selector_digest: str
    offsets: np.ndarray = field(init=False, repr=False)
    view_of: np.ndarray = field(init=False, repr=False)
    episode_of: np.ndarray = field(init=False, repr=False)
    episodes: tuple[tuple[str, str], ...] = field(init=False, repr=False)
    digest: str = field(init=False)
    _views: MappingProxyType[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        require_identity(self.id, self.native_evidence_ref, self.native_evidence_digest, self.selector_digest)
        if not self.conditioning:
            raise InvalidPolicyUpdate("snapshot requires sampled original turns")
        views = {view.id: index for index, view in enumerate(self.conditioning)}
        if len(views) != len(self.conditioning):
            raise InvalidPolicyUpdate("conditioning identities must be unique")
        if any(view.native_ref != self.native_evidence_ref for view in self.conditioning):
            raise InvalidPolicyUpdate("conditioning view references different population evidence")
        for records in (self.spans, self.relations):
            if len({record.id for record in records}) != len(records):
                raise InvalidPolicyUpdate("annotation and relation identities must be unique")
        if any(not set(relation.members) <= views.keys() for relation in self.relations):
            raise InvalidPolicyUpdate("relation addresses absent sampled turns")
        episodes = tuple(dict.fromkeys((view.episode_id, view.branch_id) for view in self.conditioning))
        episode_index = {episode: index for index, episode in enumerate(episodes)}
        lengths = np.fromiter((len(view.sampled) for view in self.conditioning), dtype=np.int64)
        offsets = np.zeros(len(self.conditioning) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        view_episode = np.fromiter(
            (episode_index[(view.episode_id, view.branch_id)] for view in self.conditioning), dtype=np.int32
        )
        set_ = object.__setattr__
        set_(self, "_views", MappingProxyType(views))
        set_(self, "offsets", frozen_array(offsets, np.int64))
        set_(
            self,
            "view_of",
            frozen_array(np.repeat(np.arange(len(self.conditioning), dtype=np.int32), lengths), np.int32),
        )
        set_(self, "episode_of", frozen_array(np.repeat(view_episode, lengths), np.int32))
        set_(self, "episodes", episodes)
        for span in self.spans:
            self.span_positions(span)
        set_(self, "digest", payload_digest(self._identity_payload()))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": "posttrain.population-snapshot.v2",
            "id": self.id,
            "native_evidence_ref": self.native_evidence_ref,
            "native_evidence_digest": self.native_evidence_digest,
            "conditioning": [{name: getattr(view, name) for name in _VIEW_FIELDS} for view in self.conditioning],
            "spans": [asdict(span) for span in self.spans],
            "relations": [asdict(relation) for relation in self.relations],
            "versions": asdict(self.versions),
            "selector_digest": self.selector_digest,
        }

    @property
    def size(self) -> int:
        """Number of sampled tokens (positions)."""
        return int(self.offsets[-1])

    def view_index(self, view_id: str) -> int:
        try:
            return self._views[view_id]
        except KeyError as error:
            raise InvalidPolicyUpdate(f"population has no conditioning view {view_id!r}") from error

    def view_positions(self, view: int) -> slice:
        return slice(int(self.offsets[view]), int(self.offsets[view + 1]))

    def views_mask(self, views: tuple[int, ...] | np.ndarray) -> np.ndarray:
        """Positions owned by the given views."""
        selected = np.zeros(len(self.conditioning), dtype=bool)
        selected[np.asarray(views, dtype=np.int64)] = True
        return selected[self.view_of]

    def action(self, position: int) -> ActionRef:
        view = self.conditioning[int(self.view_of[position])]
        local = position - int(self.offsets[self.view_of[position]])
        return ActionRef(view.episode_id, view.branch_id, view.id, view.sampled[local])

    def actions(self) -> tuple[ActionRef, ...]:
        """Every position's address; for boundaries and tests, never for storage."""
        return tuple(
            ActionRef(view.episode_id, view.branch_id, view.id, index)
            for view in self.conditioning
            for index in view.sampled
        )

    def positions(self, actions: tuple[ActionRef, ...] | list[ActionRef]) -> np.ndarray:
        """Positions of the given admitted actions, in the given order."""
        result = np.empty(len(actions), dtype=np.int64)
        for slot, action in enumerate(actions):
            view_index = self.view_index(action.turn_id)
            view = self.conditioning[view_index]
            local = int(np.searchsorted(view.sampled, action.token_index))
            if (
                (action.episode_id, action.branch_id) != (view.episode_id, view.branch_id)
                or local >= len(view.sampled)
                or view.sampled[local] != action.token_index
            ):
                raise InvalidPolicyUpdate("action addresses an absent or ineligible sampled token")
            result[slot] = int(self.offsets[view_index]) + local
        return result

    def span_positions(self, span: SemanticSpan) -> np.ndarray:
        """Sorted unique positions a span covers; every covered token must be sampled."""
        chunks = []
        for interval in span.action_intervals:
            start = interval.start
            view_index = self.view_index(start.turn_id)
            view = self.conditioning[view_index]
            if (start.episode_id, start.branch_id) != (view.episode_id, view.branch_id):
                raise InvalidPolicyUpdate("span addresses absent or ineligible sampled actions")
            sampled = np.asarray(view.sampled, dtype=np.int64)
            low, high = np.searchsorted(sampled, [start.token_index, interval.end])
            if high - low != interval.end - start.token_index:
                raise InvalidPolicyUpdate("span addresses absent or ineligible sampled actions")
            chunks.append(np.arange(low, high, dtype=np.int64) + int(self.offsets[view_index]))
        return np.unique(np.concatenate(chunks))

    def select_roles(self, roles: tuple[str, ...]) -> np.ndarray:
        """Union every supplied annotation of the named roles (population-independent selection)."""
        mask = np.zeros(self.size, dtype=bool)
        for span in self.spans:
            if span.role in roles:
                mask[self.span_positions(span)] = True
        return mask

    def select_spans(self, span_ids: tuple[str, ...]) -> np.ndarray:
        """Union selected annotations without duplicating loss contributions."""
        by_id = {span.id: span for span in self.spans}
        unknown = set(span_ids) - by_id.keys()
        if unknown:
            raise InvalidPolicyUpdate(f"unknown semantic spans: {sorted(unknown)}")
        mask = np.zeros(self.size, dtype=bool)
        for span_id in span_ids:
            mask[self.span_positions(by_id[span_id])] = True
        return mask


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
        if (
            self.mode not in {"all", "spans", "roles"}
            or (self.mode != "spans" and self.span_ids)
            or (self.mode != "roles" and self.roles)
            or (self.mode == "roles" and not self.roles)
        ):
            raise InvalidPolicyUpdate("selector must declare all actions, supplied spans or span roles")
        if len(set(self.span_ids)) != len(self.span_ids) or len(set(self.roles)) != len(self.roles):
            raise InvalidPolicyUpdate("selector span identities and roles must be unique")
        if any(not isinstance(role, str) or not role.strip() for role in self.roles):
            raise InvalidPolicyUpdate("selector roles must be nonempty names")

    def resolve(self, snapshot: PopulationSnapshot) -> np.ndarray:
        """Boolean mask over the snapshot's positions."""
        if self.mode == "all":
            return np.ones(snapshot.size, dtype=bool)
        if self.mode == "roles":
            return snapshot.select_roles(self.roles)
        return snapshot.select_spans(self.span_ids)
