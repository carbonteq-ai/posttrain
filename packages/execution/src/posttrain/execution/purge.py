"""Immutable cross-plane purge plans and their machine-scoped journal.

This module deliberately contains no provider or registry implementation.  A
planner describes exact resources and an adapter later validates and mutates
those resources.  The plan store lives outside project/run state so its audit
record survives the deletion it describes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol, cast

from posttrain.common import ContractError, JsonValue

PurgeMode = Literal["run", "project"]
PurgePlane = Literal["provider", "registry", "tracking", "local"]
PurgeTombstoneStatus = Literal["applying", "partial", "purged"]
PurgeTombstonePlaneOutcome = Literal["not-applicable", "pending", "applying", "completed", "deferred", "failed"]

_PLAN_SCHEMA_V1 = "posttrain.execution-purge-plan.v1"
_PLAN_SCHEMA_V2 = "posttrain.execution-purge-plan.v2"
_RECEIPT_SCHEMA = "posttrain.execution-purge-receipt.v1"
_JOURNAL_SCHEMA = "posttrain.execution-purge-journal.v1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_PURGE_ID = re.compile(r"^purge-[0-9a-f]{16}$")
_ACTION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_REASON_CATEGORY = re.compile(r"^[a-z][a-z0-9-]{0,63}$")

type PurgePlanSchema = Literal[
    "posttrain.execution-purge-plan.v1",
    "posttrain.execution-purge-plan.v2",
]


def _mapping(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return MappingProxyType(dict(value))


def _require_text(value: str, label: str) -> str:
    if not value.strip() or "\x00" in value:
        raise ContractError(f"purge {label} cannot be empty")
    return value


def _timestamp(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ContractError(f"purge {label} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class PurgeReason:
    """Safe, non-secret authorization context bound into a v2 purge plan."""

    category: str
    note: str | None = None
    actor: str | None = None

    def __post_init__(self) -> None:
        if not _REASON_CATEGORY.fullmatch(self.category):
            raise ContractError("purge reason category must be a lowercase slug")
        for label, value, maximum in (
            ("reason note", self.note, 280),
            ("reason actor", self.actor, 128),
        ):
            if value is None:
                continue
            if not value.strip() or "\x00" in value or "\n" in value or "\r" in value:
                raise ContractError(f"purge {label} must be one non-empty line")
            if len(value) > maximum:
                raise ContractError(f"purge {label} exceeds {maximum} characters")

    def payload(self) -> dict[str, str | None]:
        return {"category": self.category, "note": self.note, "actor": self.actor}


@dataclass(frozen=True, slots=True)
class PurgeTombstone:
    """Privacy-bounded audit state retained after cross-plane erasure."""

    purge_id: str
    plan_digest: str
    mode: PurgeMode
    project_id: str
    run_ids: tuple[str, ...]
    reason: PurgeReason
    status: PurgeTombstoneStatus
    plane_outcomes: Mapping[PurgePlane, PurgeTombstonePlaneOutcome]
    updated_at: datetime

    def __post_init__(self) -> None:
        if not _PURGE_ID.fullmatch(self.purge_id):
            raise ContractError("purge tombstone id is not valid")
        if not _DIGEST.fullmatch(self.plan_digest):
            raise ContractError("purge tombstone digest must be SHA-256")
        if self.mode not in {"run", "project"}:
            raise ContractError("purge tombstone mode is invalid")
        _require_text(self.project_id, "tombstone project id")
        if not self.run_ids or len(set(self.run_ids)) != len(self.run_ids):
            raise ContractError("purge tombstone run ids must be non-empty and unique")
        if self.status not in {"applying", "partial", "purged"}:
            raise ContractError("purge tombstone status is invalid")
        expected_planes = {"provider", "registry", "tracking", "local"}
        outcomes = dict(self.plane_outcomes)
        if set(outcomes) != expected_planes:
            raise ContractError("purge tombstone must state every plane outcome")
        valid_outcomes = {"not-applicable", "pending", "applying", "completed", "deferred", "failed"}
        if any(outcome not in valid_outcomes for outcome in outcomes.values()):
            raise ContractError("purge tombstone plane outcome is invalid")
        if self.status == "purged" and any(
            outcome not in {"not-applicable", "completed"} for outcome in outcomes.values()
        ):
            raise ContractError("purged tombstone cannot retain pending plane outcomes")
        object.__setattr__(self, "plane_outcomes", MappingProxyType(outcomes))
        _timestamp(self.updated_at, "tombstone update time")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "posttrain.execution-purge-tombstone.v1",
            "purge_id": self.purge_id,
            "plan_digest": self.plan_digest,
            "mode": self.mode,
            "project_id": self.project_id,
            "run_ids": list(self.run_ids),
            "reason": self.reason.payload(),
            "status": self.status,
            "plane_outcomes": dict(self.plane_outcomes),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_plan(
        cls,
        plan: PurgePlan,
        events: Sequence[Mapping[str, object]],
        *,
        updated_at: datetime | None = None,
    ) -> PurgeTombstone:
        if plan.reason is None:
            raise ContractError("legacy purge plan cannot create a tombstone")
        action_statuses: dict[str, str] = {}
        for event in events:
            action_id = event.get("action_id")
            event_status = event.get("status")
            if isinstance(action_id, str) and isinstance(event_status, str):
                action_statuses[action_id] = event_status
        outcomes: dict[PurgePlane, PurgeTombstonePlaneOutcome] = {}
        for plane in ("provider", "registry", "tracking", "local"):
            actions = tuple(action for action in plan.actions if action.plane == plane)
            statuses = tuple(action_statuses.get(action.action_id) for action in actions)
            if not actions:
                outcome: PurgeTombstonePlaneOutcome = "not-applicable"
            elif all(status in {"completed", "skipped"} for status in statuses):
                outcome = "completed"
            elif any(status == "failed" for status in statuses):
                outcome = "failed"
            elif any(status == "deferred" for status in statuses):
                outcome = "deferred"
            elif any(status == "started" for status in statuses):
                outcome = "applying"
            else:
                outcome = "pending"
            outcomes[plane] = outcome
        status: PurgeTombstoneStatus
        if all(outcome in {"not-applicable", "completed"} for outcome in outcomes.values()):
            status = cast(PurgeTombstoneStatus, "purged")
        elif any(outcome in {"failed", "deferred"} for outcome in outcomes.values()):
            status = cast(PurgeTombstoneStatus, "partial")
        else:
            status = cast(PurgeTombstoneStatus, "applying")
        return cls(
            purge_id=plan.purge_id,
            plan_digest=plan.digest,
            mode=plan.mode,
            project_id=plan.project_id,
            run_ids=plan.run_ids,
            reason=plan.reason,
            status=status,
            plane_outcomes=outcomes,
            updated_at=updated_at or datetime.now(UTC),
        )


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ContractError("purge plan contains a non-JSON value") from error


def _digest(payload: Mapping[str, object]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class PurgeAction:
    """One exact deletion action and the precondition it must recheck."""

    action_id: str
    plane: PurgePlane
    kind: str
    target: Mapping[str, JsonValue]
    depends_on: tuple[str, ...] = ()
    precondition: Mapping[str, JsonValue] = MappingProxyType({})
    logical_bytes: int = 0

    def __post_init__(self) -> None:
        if not _ACTION_ID.fullmatch(self.action_id):
            raise ContractError("purge action id is not path-safe")
        _require_text(self.kind, "action kind")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ContractError("purge action dependencies must be unique")
        if self.action_id in self.depends_on:
            raise ContractError("purge action cannot depend on itself")
        if self.plane not in {"provider", "registry", "tracking", "local"}:
            raise ContractError("purge action plane is invalid")
        if self.logical_bytes < 0:
            raise ContractError("purge action logical bytes cannot be negative")
        object.__setattr__(self, "target", _mapping(self.target))
        object.__setattr__(self, "precondition", _mapping(self.precondition))

    def payload(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "plane": self.plane,
            "kind": self.kind,
            "target": dict(self.target),
            "depends_on": list(self.depends_on),
            "precondition": dict(self.precondition),
            "logical_bytes": self.logical_bytes,
        }


@dataclass(frozen=True, slots=True)
class PurgePlan:
    """A complete, immutable preview of one run or project purge."""

    purge_id: str
    mode: PurgeMode
    project_id: str
    run_ids: tuple[str, ...]
    root_run_id: str | None
    dependency_edges: tuple[tuple[str, str], ...]
    tracking_actions: tuple[PurgeAction, ...]
    provider_actions: tuple[PurgeAction, ...]
    registry_actions: tuple[PurgeAction, ...]
    local_actions: tuple[PurgeAction, ...]
    warnings: tuple[str, ...]
    blockers: tuple[str, ...]
    reason: PurgeReason | None
    digest: str
    created_at: datetime
    schema: PurgePlanSchema = _PLAN_SCHEMA_V2

    def __post_init__(self) -> None:
        if not _PURGE_ID.fullmatch(self.purge_id):
            raise ContractError("purge id is not valid")
        if self.schema not in {_PLAN_SCHEMA_V1, _PLAN_SCHEMA_V2}:
            raise ContractError("purge plan schema is invalid")
        if self.mode not in {"run", "project"}:
            raise ContractError("purge mode is invalid")
        if self.schema == _PLAN_SCHEMA_V2 and self.reason is None:
            raise ContractError("purge plan reason is required")
        if self.schema == _PLAN_SCHEMA_V1 and self.reason is not None:
            raise ContractError("legacy purge plan cannot contain a reason")
        _require_text(self.project_id, "project id")
        if not self.run_ids and self.mode == "run":
            raise ContractError("run purge plan must select at least one run")
        if len(set(self.run_ids)) != len(self.run_ids):
            raise ContractError("purge plan run ids must be unique")
        if self.mode == "run" and self.root_run_id is None:
            raise ContractError("run purge plan must name a root run")
        if self.root_run_id is not None and self.root_run_id not in self.run_ids:
            raise ContractError("purge root run must be selected")
        if len(set(self.dependency_edges)) != len(self.dependency_edges):
            raise ContractError("purge dependency edges must be unique")
        selected_runs = set(self.run_ids)
        for producer, consumer in self.dependency_edges:
            if producer not in selected_runs or consumer not in selected_runs:
                raise ContractError("purge dependency edges must stay inside the selected run closure")
        action_groups = (
            self.provider_actions,
            self.registry_actions,
            self.tracking_actions,
            self.local_actions,
        )
        actions = tuple(action for group in action_groups for action in group)
        action_ids = [action.action_id for action in actions]
        if len(set(action_ids)) != len(action_ids):
            raise ContractError("purge action ids must be globally unique")
        known = set(action_ids)
        for action in actions:
            unknown = set(action.depends_on) - known
            if unknown:
                raise ContractError("purge action depends on unknown action(s): " + ", ".join(sorted(unknown)))
            target_run_id = action.target.get("run_id")
            if action.kind == "tracking.delete_project":
                if self.mode != "project":
                    raise ContractError("run purge cannot contain a project deletion action")
                continue
            if isinstance(target_run_id, str) and target_run_id not in selected_runs:
                raise ContractError("purge action target must belong to the selected run closure")
        _timestamp(self.created_at, "creation time")
        if not _DIGEST.fullmatch(self.digest):
            raise ContractError("purge plan digest must be SHA-256")
        if self.digest != self.computed_digest():
            raise ContractError("purge plan digest does not match its immutable contents")

    @property
    def actions(self) -> tuple[PurgeAction, ...]:
        """Return actions in their intentional apply order."""

        return (
            *self.provider_actions,
            *self.registry_actions,
            *self.tracking_actions,
            *self.local_actions,
        )

    def semantic_payload(self) -> dict[str, object]:
        """Return the digest-covered fields, excluding timestamps and labels."""

        return {
            "schema": self.schema,
            "mode": self.mode,
            "project_id": self.project_id,
            "run_ids": list(self.run_ids),
            "root_run_id": self.root_run_id,
            "dependency_edges": [list(edge) for edge in self.dependency_edges],
            "provider_actions": [action.payload() for action in self.provider_actions],
            "registry_actions": [action.payload() for action in self.registry_actions],
            "tracking_actions": [action.payload() for action in self.tracking_actions],
            "local_actions": [action.payload() for action in self.local_actions],
            "warnings": list(self.warnings),
            "blockers": list(self.blockers),
            **({"reason": self.reason.payload()} if self.reason is not None else {}),
        }

    def computed_digest(self) -> str:
        return _digest(self.semantic_payload())

    @classmethod
    def build(
        cls,
        *,
        mode: PurgeMode,
        project_id: str,
        run_ids: Sequence[str],
        root_run_id: str | None,
        dependency_edges: Sequence[tuple[str, str]] = (),
        tracking_actions: Sequence[PurgeAction] = (),
        provider_actions: Sequence[PurgeAction] = (),
        registry_actions: Sequence[PurgeAction] = (),
        local_actions: Sequence[PurgeAction] = (),
        warnings: Sequence[str] = (),
        blockers: Sequence[str] = (),
        reason: PurgeReason,
        created_at: datetime | None = None,
    ) -> PurgePlan:
        """Build a plan and derive its collision-checked content address."""

        creation = created_at or datetime.now(UTC)
        payload = {
            "schema": _PLAN_SCHEMA_V2,
            "mode": mode,
            "project_id": project_id,
            "run_ids": list(run_ids),
            "root_run_id": root_run_id,
            "dependency_edges": [list(edge) for edge in dependency_edges],
            "provider_actions": [action.payload() for action in provider_actions],
            "registry_actions": [action.payload() for action in registry_actions],
            "tracking_actions": [action.payload() for action in tracking_actions],
            "local_actions": [action.payload() for action in local_actions],
            "warnings": list(warnings),
            "blockers": list(blockers),
            "reason": reason.payload(),
        }
        digest = _digest(payload)
        return cls(
            purge_id=PurgeStore.purge_id_for_digest(digest),
            mode=mode,
            project_id=project_id,
            run_ids=tuple(run_ids),
            root_run_id=root_run_id,
            dependency_edges=tuple(dependency_edges),
            tracking_actions=tuple(tracking_actions),
            provider_actions=tuple(provider_actions),
            registry_actions=tuple(registry_actions),
            local_actions=tuple(local_actions),
            warnings=tuple(warnings),
            blockers=tuple(blockers),
            reason=reason,
            digest=digest,
            created_at=creation,
            schema=_PLAN_SCHEMA_V2,
        )


@dataclass(frozen=True, slots=True)
class PurgeReceipt:
    """Durable result of applying one immutable purge plan."""

    purge_id: str
    plan_digest: str
    completed_actions: tuple[str, ...]
    skipped_actions: tuple[str, ...]
    failed_action: str | None
    completed_at: datetime

    def __post_init__(self) -> None:
        if not _PURGE_ID.fullmatch(self.purge_id):
            raise ContractError("purge receipt id is not valid")
        if not _DIGEST.fullmatch(self.plan_digest):
            raise ContractError("purge receipt digest must be SHA-256")
        if len(set(self.completed_actions)) != len(self.completed_actions):
            raise ContractError("purge receipt completed actions must be unique")
        if len(set(self.skipped_actions)) != len(self.skipped_actions):
            raise ContractError("purge receipt skipped actions must be unique")
        if set(self.completed_actions) & set(self.skipped_actions):
            raise ContractError("purge receipt cannot complete and skip one action")
        if self.failed_action in set(self.completed_actions) | set(self.skipped_actions):
            raise ContractError("purge receipt failed action cannot already be settled")
        _timestamp(self.completed_at, "completion time")


class PurgeActionExecutor(Protocol):
    """Provider adapter used by the journaled apply engine."""

    def revalidate(self, action: PurgeAction) -> None: ...

    def apply(self, action: PurgeAction) -> None: ...


class PurgeApplyError(RuntimeError):
    """One action failed after the immutable plan was accepted."""

    def __init__(self, action_id: str, cause: Exception) -> None:
        self.action_id = action_id
        self.cause = cause
        super().__init__(f"purge action {action_id!r} failed: {cause}")


class PurgeActionDeferred(RuntimeError):
    """One action is safe to retry but its external precondition is not ready."""


class PurgeApplyDeferred(RuntimeError):
    """An immutable purge stopped before mutation could safely continue."""

    def __init__(self, action_id: str, cause: PurgeActionDeferred) -> None:
        self.action_id = action_id
        self.cause = cause
        super().__init__(f"purge action {action_id!r} is deferred: {cause}")


class PurgeStore:
    """Durable machine-scoped plans, journals, and receipts."""

    def __init__(self, state_root: Path) -> None:
        if not state_root.is_absolute():
            raise ValueError("purge state root must be absolute")
        self._root = state_root.resolve() / "purges"

    @property
    def root(self) -> Path:
        return self._root

    @staticmethod
    def purge_id_for_digest(digest: str) -> str:
        if not _DIGEST.fullmatch(digest):
            raise ContractError("purge plan digest must be SHA-256")
        return "purge-" + digest.removeprefix("sha256:")[:16]

    def plan_path(self, purge_id: str) -> Path:
        self._validate_id(purge_id)
        return self._root / purge_id / "plan.json"

    def journal_path(self, purge_id: str) -> Path:
        self._validate_id(purge_id)
        return self._root / purge_id / "journal.jsonl"

    def receipt_path(self, purge_id: str) -> Path:
        self._validate_id(purge_id)
        return self._root / purge_id / "receipt.json"

    def tombstone_path(self, purge_id: str) -> Path:
        self._validate_id(purge_id)
        return self._root / purge_id / "tombstone.json"

    def save_plan(self, plan: PurgePlan) -> PurgePlan:
        """Persist an immutable plan, reusing an identical content address."""

        path = self.plan_path(plan.purge_id)
        if path.is_file():
            existing = self.load_plan(plan.purge_id)
            if existing.digest != plan.digest or existing.semantic_payload() != plan.semantic_payload():
                raise ContractError(f"purge id {plan.purge_id} is already bound to another plan")
            return existing
        self._write_json(path, self._plan_payload(plan))
        return plan

    def load_plan(self, purge_id: str) -> PurgePlan:
        payload = self._read_object(self.plan_path(purge_id), "purge plan")
        schema = payload.get("schema")
        if schema not in {_PLAN_SCHEMA_V1, _PLAN_SCHEMA_V2}:
            raise ContractError(f"purge plan {purge_id} has an unsupported schema")
        try:
            reason = self._reason(payload["reason"]) if schema == _PLAN_SCHEMA_V2 else None
            return PurgePlan(
                purge_id=purge_id,
                mode=cast(PurgeMode, payload["mode"]),
                project_id=str(payload["project_id"]),
                run_ids=tuple(str(value) for value in _sequence(payload["run_ids"], "run ids")),
                root_run_id=(str(payload["root_run_id"]) if payload.get("root_run_id") is not None else None),
                dependency_edges=tuple(
                    (str(edge[0]), str(edge[1])) for edge in _edge_sequence(payload["dependency_edges"])
                ),
                tracking_actions=tuple(
                    self._action(value) for value in _sequence(payload["tracking_actions"], "tracking actions")
                ),
                provider_actions=tuple(
                    self._action(value) for value in _sequence(payload["provider_actions"], "provider actions")
                ),
                registry_actions=tuple(
                    self._action(value) for value in _sequence(payload["registry_actions"], "registry actions")
                ),
                local_actions=tuple(
                    self._action(value) for value in _sequence(payload["local_actions"], "local actions")
                ),
                warnings=tuple(str(value) for value in _sequence(payload["warnings"], "warnings")),
                blockers=tuple(str(value) for value in _sequence(payload["blockers"], "blockers")),
                reason=reason,
                digest=str(payload["digest"]),
                created_at=datetime.fromisoformat(str(payload["created_at"])),
                schema=cast(PurgePlanSchema, schema),
            )
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise ContractError(f"purge plan {purge_id} is invalid") from error

    def append_journal(self, purge_id: str, *, action_id: str, status: str, detail: str | None = None) -> None:
        """Append and fsync one action event without rewriting prior events."""

        _require_text(action_id, "journal action id")
        _require_text(status, "journal status")
        payload: dict[str, object] = {
            "schema": _JOURNAL_SCHEMA,
            "action_id": action_id,
            "status": status,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        if detail is not None:
            payload["detail"] = detail
        path = self.journal_path(purge_id)
        self._append_json(path, payload)

    def journal(self, purge_id: str) -> tuple[dict[str, object], ...]:
        path = self.journal_path(purge_id)
        if not path.is_file():
            return ()
        events: list[dict[str, object]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ContractError(f"purge journal is invalid at line {line_number}") from error
            if not isinstance(payload, dict) or payload.get("schema") != _JOURNAL_SCHEMA:
                raise ContractError(f"purge journal is invalid at line {line_number}")
            events.append(payload)
        return tuple(events)

    def save_receipt(self, receipt: PurgeReceipt) -> PurgeReceipt:
        plan = self.load_plan(receipt.purge_id)
        if plan.digest != receipt.plan_digest:
            raise ContractError(f"purge receipt {receipt.purge_id} does not match its plan")
        path = self.receipt_path(receipt.purge_id)
        if path.is_file():
            existing = self.load_receipt(receipt.purge_id)
            if existing != receipt:
                raise ContractError(f"purge receipt {receipt.purge_id} conflicts with the existing receipt")
            return existing
        self._write_json(path, self._receipt_payload(receipt))
        return receipt

    def load_receipt(self, purge_id: str) -> PurgeReceipt:
        payload = self._read_object(self.receipt_path(purge_id), "purge receipt")
        if payload.get("schema") != _RECEIPT_SCHEMA:
            raise ContractError(f"purge receipt {purge_id} has an unsupported schema")
        try:
            return PurgeReceipt(
                purge_id=purge_id,
                plan_digest=str(payload["plan_digest"]),
                completed_actions=tuple(
                    str(value) for value in _sequence(payload["completed_actions"], "completed actions")
                ),
                skipped_actions=tuple(str(value) for value in _sequence(payload["skipped_actions"], "skipped actions")),
                failed_action=(str(payload["failed_action"]) if payload.get("failed_action") is not None else None),
                completed_at=datetime.fromisoformat(str(payload["completed_at"])),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError(f"purge receipt {purge_id} is invalid") from error

    def save_tombstone(self, tombstone: PurgeTombstone) -> PurgeTombstone:
        plan = self.load_plan(tombstone.purge_id)
        if plan.reason is None:
            raise ContractError("legacy purge plan cannot create a tombstone")
        if (
            tombstone.plan_digest != plan.digest
            or tombstone.mode != plan.mode
            or tombstone.project_id != plan.project_id
            or tombstone.run_ids != plan.run_ids
            or tombstone.reason != plan.reason
        ):
            raise ContractError(f"purge tombstone {tombstone.purge_id} does not match its plan")
        self._write_json(self.tombstone_path(tombstone.purge_id), tombstone.payload())
        return tombstone

    def update_tombstone(self, purge_id: str) -> PurgeTombstone:
        plan = self.load_plan(purge_id)
        tombstone = PurgeTombstone.from_plan(plan, self.journal(purge_id))
        return self.save_tombstone(tombstone)

    def load_tombstone(self, purge_id: str) -> PurgeTombstone:
        payload = self._read_object(self.tombstone_path(purge_id), "purge tombstone")
        if payload.get("schema") != "posttrain.execution-purge-tombstone.v1":
            raise ContractError(f"purge tombstone {purge_id} has an unsupported schema")
        try:
            outcomes = _object(payload["plane_outcomes"], "tombstone plane outcomes")
            return PurgeTombstone(
                purge_id=purge_id,
                plan_digest=str(payload["plan_digest"]),
                mode=cast(PurgeMode, payload["mode"]),
                project_id=str(payload["project_id"]),
                run_ids=tuple(str(value) for value in _sequence(payload["run_ids"], "tombstone run ids")),
                reason=self._reason(payload["reason"]),
                status=cast(PurgeTombstoneStatus, payload["status"]),
                plane_outcomes={
                    cast(PurgePlane, plane): cast(PurgeTombstonePlaneOutcome, outcome)
                    for plane, outcome in outcomes.items()
                },
                updated_at=datetime.fromisoformat(str(payload["updated_at"])),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError(f"purge tombstone {purge_id} is invalid") from error

    @staticmethod
    def _validate_id(purge_id: str) -> None:
        if not _PURGE_ID.fullmatch(purge_id):
            raise ContractError("purge id is not valid")

    @staticmethod
    def _plan_payload(plan: PurgePlan) -> dict[str, object]:
        return {
            **plan.semantic_payload(),
            "purge_id": plan.purge_id,
            "digest": plan.digest,
            "created_at": plan.created_at.isoformat(),
        }

    @staticmethod
    def _receipt_payload(receipt: PurgeReceipt) -> dict[str, object]:
        return {
            "schema": _RECEIPT_SCHEMA,
            "purge_id": receipt.purge_id,
            "plan_digest": receipt.plan_digest,
            "completed_actions": list(receipt.completed_actions),
            "skipped_actions": list(receipt.skipped_actions),
            "failed_action": receipt.failed_action,
            "completed_at": receipt.completed_at.isoformat(),
        }

    @staticmethod
    def _action(value: object) -> PurgeAction:
        if not isinstance(value, dict):
            raise ContractError("purge action must be an object")
        try:
            return PurgeAction(
                action_id=str(value["action_id"]),
                plane=cast(PurgePlane, value["plane"]),
                kind=str(value["kind"]),
                target=dict(_object(value["target"], "action target")),
                depends_on=tuple(str(item) for item in _sequence(value["depends_on"], "action dependencies")),
                precondition=dict(_object(value["precondition"], "action precondition")),
                logical_bytes=int(value["logical_bytes"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("purge action is invalid") from error

    @staticmethod
    def _reason(value: object) -> PurgeReason:
        if not isinstance(value, dict):
            raise ContractError("purge reason must be an object")
        try:
            category = value["category"]
            note = value.get("note")
            actor = value.get("actor")
            if not isinstance(category, str) or note is not None and not isinstance(note, str) or actor is not None and not isinstance(actor, str):
                raise TypeError
            return PurgeReason(category=category, note=note, actor=actor)
        except (KeyError, TypeError) as error:
            raise ContractError("purge reason is invalid") from error

    @staticmethod
    def _read_object(path: Path, label: str) -> dict[str, object]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ContractError(f"{label} is missing: {path}") from error
        except json.JSONDecodeError as error:
            raise ContractError(f"{label} is invalid: {path}") from error
        if not isinstance(payload, dict):
            raise ContractError(f"{label} must be an object: {path}")
        return payload

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f".{path.name}-{uuid.uuid4().hex}.tmp"
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            encoded = _canonical_bytes(payload) + b"\n"
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)

    @staticmethod
    def _append_json(path: Path, payload: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, _canonical_bytes(payload) + b"\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ContractError(f"purge {label} must be a list")
    return value


def _edge_sequence(value: object) -> list[list[object]]:
    edges = _sequence(value, "dependency edges")
    result: list[list[object]] = []
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2:
            raise ContractError("purge dependency edge must contain two run ids")
        result.append(edge)
    return result


def _object(value: object, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ContractError(f"purge {label} must be an object")
    return value


def apply_purge_plan(
    store: PurgeStore,
    purge_id: str,
    executors: Mapping[PurgePlane, PurgeActionExecutor],
) -> PurgeReceipt:
    """Resume and apply one immutable plan in dependency order.

    A completed journal event is the only state used to skip an action. A
    started or failed event is retried after revalidation, so an interrupted
    process cannot widen the plan or silently substitute a newer resource.
    """

    plan = store.load_plan(purge_id)
    if plan.reason is not None:
        store.update_tombstone(purge_id)
    if plan.blockers:
        raise ContractError("purge plan is blocked: " + "; ".join(plan.blockers))
    try:
        return store.load_receipt(purge_id)
    except ContractError as error:
        if "is missing" not in str(error):
            raise

    events = store.journal(purge_id)
    completed = {str(event["action_id"]) for event in events if event.get("status") in {"completed", "skipped"}}
    skipped = {str(event["action_id"]) for event in events if event.get("status") == "skipped"}
    for action in plan.actions:
        if action.action_id in completed:
            continue
        if not set(action.depends_on).issubset(completed):
            missing = sorted(set(action.depends_on) - completed)
            raise ContractError(
                f"purge action {action.action_id!r} depends on incomplete action(s): {', '.join(missing)}"
            )
        executor = executors.get(action.plane)
        if executor is None:
            raise ContractError(f"purge has no executor for {action.plane!r}")
        store.append_journal(purge_id, action_id=action.action_id, status="started")
        if plan.reason is not None:
            store.update_tombstone(purge_id)
        try:
            executor.revalidate(action)
            executor.apply(action)
        except PurgeActionDeferred as error:
            store.append_journal(
                purge_id,
                action_id=action.action_id,
                status="deferred",
                detail=str(error),
            )
            if plan.reason is not None:
                store.update_tombstone(purge_id)
            raise PurgeApplyDeferred(action.action_id, error) from error
        except Exception as error:
            store.append_journal(
                purge_id,
                action_id=action.action_id,
                status="failed",
                detail=f"{type(error).__name__}: {error}",
            )
            if plan.reason is not None:
                store.update_tombstone(purge_id)
            raise PurgeApplyError(action.action_id, error) from error
        store.append_journal(purge_id, action_id=action.action_id, status="completed")
        if plan.reason is not None:
            store.update_tombstone(purge_id)
        completed.add(action.action_id)

    receipt = PurgeReceipt(
        purge_id=plan.purge_id,
        plan_digest=plan.digest,
        completed_actions=tuple(action.action_id for action in plan.actions),
        skipped_actions=tuple(sorted(skipped)),
        failed_action=None,
        completed_at=datetime.now(UTC),
    )
    saved = store.save_receipt(receipt)
    if plan.reason is not None:
        store.update_tombstone(purge_id)
    return saved


__all__ = [
    "PurgeAction",
    "PurgeActionExecutor",
    "PurgeActionDeferred",
    "PurgeApplyDeferred",
    "PurgeApplyError",
    "apply_purge_plan",
    "PurgeMode",
    "PurgePlan",
    "PurgePlane",
    "PurgeReceipt",
    "PurgeReason",
    "PurgeTombstone",
    "PurgeTombstonePlaneOutcome",
    "PurgeTombstoneStatus",
    "PurgeStore",
]
