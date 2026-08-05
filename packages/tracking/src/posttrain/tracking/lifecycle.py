"""Optional destructive lifecycle contracts for tracking backends.

Tracking readers remain read-only by default.  A backend that does not expose
these administrator capabilities must make purge unavailable instead of
silently deleting only part of a run's evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from posttrain.common import ContractError

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def _text(value: str, label: str) -> None:
    if not value.strip() or "\x00" in value:
        raise ContractError(f"tracking {label} cannot be empty")


def _digest(value: str, label: str) -> None:
    if not _DIGEST.fullmatch(value):
        raise ContractError(f"tracking {label} must be SHA-256")


@dataclass(frozen=True, slots=True)
class TrackingArtifactPurge:
    """One artifact version the tracking server proposes to remove."""

    version_id: str
    name: str
    version: str
    digest: str | None
    logical_bytes: int
    consumer_run_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.version_id, "artifact version id")
        _text(self.name, "artifact name")
        _text(self.version, "artifact version")
        if self.digest is not None:
            _digest(self.digest, "artifact digest")
        if self.logical_bytes < 0:
            raise ContractError("tracking artifact logical bytes cannot be negative")
        if len(set(self.consumer_run_ids)) != len(self.consumer_run_ids):
            raise ContractError("tracking artifact consumers must be unique")


@dataclass(frozen=True, slots=True)
class TrackingPurgePlan:
    """Authenticated provider preview for exact run and artifact deletion."""

    provider: str
    project: str
    provider_run_ids: tuple[str, ...]
    run_ids: tuple[str, ...]
    artifacts: tuple[TrackingArtifactPurge, ...]
    blockers: tuple[str, ...]
    digest: str
    created_at: datetime

    def __post_init__(self) -> None:
        _text(self.provider, "provider")
        _text(self.project, "project")
        if not self.provider_run_ids:
            raise ContractError("tracking purge must select at least one provider run")
        if len(set(self.provider_run_ids)) != len(self.provider_run_ids):
            raise ContractError("tracking provider run ids must be unique")
        if len(set(self.run_ids)) != len(self.run_ids):
            raise ContractError("tracking run ids must be unique")
        if len({artifact.version_id for artifact in self.artifacts}) != len(self.artifacts):
            raise ContractError("tracking purge artifact versions must be unique")
        _digest(self.digest, "purge plan digest")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ContractError("tracking purge plan time must be timezone-aware")


@dataclass(frozen=True, slots=True)
class TrackingPurgeReceipt:
    """Receipt returned after the tracking backend applies a run purge."""

    provider: str
    project: str
    plan_digest: str
    deleted_provider_run_ids: tuple[str, ...]
    deleted_artifact_version_ids: tuple[str, ...]
    already_absent_provider_run_ids: tuple[str, ...]
    completed_at: datetime

    def __post_init__(self) -> None:
        _text(self.provider, "provider")
        _text(self.project, "project")
        _digest(self.plan_digest, "purge receipt digest")
        if self.completed_at.tzinfo is None or self.completed_at.utcoffset() is None:
            raise ContractError("tracking purge receipt time must be timezone-aware")


@dataclass(frozen=True, slots=True)
class TrackingProjectDeletePlan:
    """Authenticated project-scoped deletion preview."""

    provider: str
    project: str
    exists: bool
    runs: int
    artifacts: int
    artifact_versions: int
    logical_bytes: int
    storage_bytes: int
    blockers: tuple[str, ...]
    digest: str
    created_at: datetime

    def __post_init__(self) -> None:
        _text(self.provider, "provider")
        _text(self.project, "project")
        for label, value in (
            ("runs", self.runs),
            ("artifacts", self.artifacts),
            ("artifact versions", self.artifact_versions),
            ("logical bytes", self.logical_bytes),
            ("storage bytes", self.storage_bytes),
        ):
            if value < 0:
                raise ContractError(f"tracking project {label} cannot be negative")
        _digest(self.digest, "project delete plan digest")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ContractError("tracking project delete plan time must be timezone-aware")


@dataclass(frozen=True, slots=True)
class TrackingProjectDeleteReceipt:
    """Receipt returned after a project-scoped tracking deletion."""

    provider: str
    project: str
    plan_digest: str
    deleted: bool
    completed_at: datetime

    def __post_init__(self) -> None:
        _text(self.provider, "provider")
        _text(self.project, "project")
        _digest(self.plan_digest, "project delete receipt digest")
        if self.completed_at.tzinfo is None or self.completed_at.utcoffset() is None:
            raise ContractError("tracking project delete receipt time must be timezone-aware")


class TrackingLifecycleAdmin(Protocol):
    """Optional authenticated deletion surface owned by a tracking adapter."""

    def plan_run_purge(
        self,
        *,
        project: str,
        provider_run_ids: tuple[str, ...],
    ) -> TrackingPurgePlan: ...

    def apply_run_purge(self, plan: TrackingPurgePlan) -> TrackingPurgeReceipt: ...

    def project_delete_plan(self, *, project: str) -> TrackingProjectDeletePlan: ...

    def delete_project(self, plan: TrackingProjectDeletePlan) -> TrackingProjectDeleteReceipt: ...


__all__ = [
    "TrackingArtifactPurge",
    "TrackingLifecycleAdmin",
    "TrackingProjectDeletePlan",
    "TrackingProjectDeleteReceipt",
    "TrackingPurgePlan",
    "TrackingPurgeReceipt",
]
