"""Provider-neutral run purge closure planning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

from posttrain.common import ContractError, JsonValue

from .purge import PurgeAction, PurgeMode, PurgePlan, PurgePlane, PurgeReason
from .registry import RegistryManifestRef


@dataclass(frozen=True, slots=True)
class PurgeRunCandidate:
    """Immutable control/evidence snapshot supplied by a composition root."""

    run_id: str
    project_id: str
    provider: str
    provider_id: str
    state: str
    reconciled: bool
    evidence_provider: str
    evidence_project: str
    tracking_provider_run_id: str
    consumers: tuple[str, ...] = ()
    external_consumers: tuple[str, ...] = ()
    lineage_complete: bool = True
    lineage_blockers: tuple[str, ...] = ()
    image: RegistryManifestRef | None = None
    workspace: Path | None = None
    local_paths: tuple[Path, ...] = ()
    completed_planes: tuple[PurgePlane, ...] = ()
    evidence_retention: str = "standard"

    def __post_init__(self) -> None:
        for label, value in (
            ("run id", self.run_id),
            ("project id", self.project_id),
            ("provider", self.provider),
            ("provider id", self.provider_id),
            ("state", self.state),
            ("evidence provider", self.evidence_provider),
            ("evidence project", self.evidence_project),
            ("tracking provider run id", self.tracking_provider_run_id),
        ):
            if not value.strip() or "\x00" in value:
                raise ContractError(f"purge candidate {label} cannot be empty")
        if len(set(self.consumers)) != len(self.consumers):
            raise ContractError("purge candidate consumers must be unique")
        if len(set(self.external_consumers)) != len(self.external_consumers):
            raise ContractError("purge candidate external consumers must be unique")
        if len(set(self.lineage_blockers)) != len(self.lineage_blockers):
            raise ContractError("purge candidate lineage blockers must be unique")
        for path in self.local_paths:
            if not path.is_absolute():
                raise ContractError("purge candidate local paths must be absolute")
        if self.workspace is not None and not self.workspace.is_absolute():
            raise ContractError("purge candidate workspace must be absolute")
        if len(set(self.completed_planes)) != len(self.completed_planes):
            raise ContractError("purge candidate completed planes must be unique")
        if any(plane not in {"provider", "registry", "tracking", "local"} for plane in self.completed_planes):
            raise ContractError("purge candidate completed plane is invalid")
        if self.evidence_retention not in {"standard", "pinned"}:
            raise ContractError("purge candidate evidence retention is invalid")


ORPHAN_TRACKING_RUN_BASIS = "orphan-tracking-run"
DEFAULT_ORPHAN_STALE_AFTER = timedelta(hours=24)
_TERMINAL_TRACKING_STATUSES = frozenset({"succeeded", "failed", "cancelled", "lost"})


@dataclass(frozen=True, slots=True)
class OrphanProviderInventory:
    """Provider-side evidence that no execution is attributable to an orphan.

    ``checked`` names every provider inventory that was successfully queried
    (for example ``dstack project 'main'``). ``active`` lists executions that
    still reference the run id, and ``blockers`` lists inventories that could
    not be queried. An empty ``checked`` tuple is never proof of absence.
    """

    checked: tuple[str, ...] = ()
    active: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OrphanTrackingRun:
    """One tracking-plane run with no local submission receipt.

    A composition root builds this from the tracking backend, the provider
    inventories, and the machine registry-ownership inventory. The planner is
    the only place that turns that evidence into blockers, so preview and
    apply-time revalidation cannot disagree about the rules.
    """

    run_id: str
    project_id: str
    evidence_provider: str
    evidence_project: str
    tracking_provider_run_id: str
    tracking_status: str
    last_activity_at: datetime | None
    provider: OrphanProviderInventory
    recorded_provider: str | None = None
    recorded_job_image: str | None = None
    consumers: tuple[str, ...] = ()
    lineage_complete: bool = True
    lineage_blockers: tuple[str, ...] = ()
    tracked_artifacts: int = 0
    evidence_retention: str = "standard"
    admission: OrphanAdmissionEntry | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("run id", self.run_id),
            ("project id", self.project_id),
            ("evidence provider", self.evidence_provider),
            ("evidence project", self.evidence_project),
            ("tracking provider run id", self.tracking_provider_run_id),
            ("tracking status", self.tracking_status),
        ):
            if not value.strip() or "\x00" in value:
                raise ContractError(f"orphan tracking run {label} cannot be empty")
        if self.last_activity_at is not None and self.last_activity_at.utcoffset() is None:
            raise ContractError("orphan tracking run last activity must be timezone-aware")
        if self.tracked_artifacts < 0:
            raise ContractError("orphan tracking run artifact count cannot be negative")


@dataclass(frozen=True, slots=True)
class OrphanAdmissionEntry:
    """The machine admission-ledger entry that still names an orphaned run.

    ``control_store_status`` is ``absent`` when the recorded owning control
    store no longer exists, ``no-receipt`` when it exists without a submission
    receipt for the run, ``has-receipt`` when the owner still controls it, and
    ``unknown`` when no locator was recorded or it could not be inspected.
    ``provider_state`` is the neutral state of the provider execution the entry
    names, queried now (``lost`` means absent), or ``None`` if the query failed.
    """

    state: str
    admission_key: str
    provider: str
    provider_id: str | None
    control_store: str | None
    control_store_status: str
    provider_state: str | None
    provider_native_state: str | None = None
    provider_query_error: str | None = None
    job_image: str | None = None

    def __post_init__(self) -> None:
        if self.control_store_status not in {"absent", "no-receipt", "has-receipt", "unknown"}:
            raise ContractError("orphan admission control-store status is invalid")
        if not self.admission_key.strip() or not self.state.strip():
            raise ContractError("orphan admission entry identity is invalid")


class PurgeRunCatalog(Protocol):
    def get(self, run_id: str) -> PurgeRunCandidate | None: ...

    def list(self) -> tuple[PurgeRunCandidate, ...]: ...

    def registry_image_owners(self) -> Mapping[str, tuple[str, ...]]: ...

    def registry_inventory_blockers(self) -> tuple[str, ...]: ...


def build_run_purge_plan(
    catalog: PurgeRunCatalog,
    *,
    root_run_id: str,
    reason: PurgeReason,
    cascade: bool = False,
) -> PurgePlan:
    """Build a blocked-or-applicable run plan without mutating any adapter."""

    root = catalog.get(root_run_id)
    blockers: list[str] = []
    warnings: list[str] = []
    if root is None:
        return PurgePlan.build(
            mode="run",
            project_id="unknown",
            run_ids=(root_run_id,),
            root_run_id=root_run_id,
            reason=reason,
            blockers=(f"run {root_run_id!r} was not found",),
        )

    if root.evidence_retention == "pinned":
        warnings.append(f"run {root.run_id!r} is pinned; explicit run purge overrides its retention pin")

    selected: dict[str, PurgeRunCandidate] = {}
    visiting: set[str] = set()

    def visit(candidate: PurgeRunCandidate) -> None:
        if candidate.run_id in selected:
            return
        if candidate.run_id in visiting:
            blockers.append(f"artifact consumer graph contains a cycle at {candidate.run_id!r}")
            return
        visiting.add(candidate.run_id)
        _validate_candidate(root, candidate, blockers)
        if candidate.run_id != root.run_id and candidate.evidence_retention == "pinned":
            blockers.append(f"pinned run {candidate.run_id!r} cannot be added by a cascade purge")
        selected[candidate.run_id] = candidate
        consumers = tuple(candidate.consumers)
        if candidate.external_consumers:
            blockers.extend(
                f"run {candidate.run_id!r} has external consumer {consumer!r}"
                for consumer in candidate.external_consumers
            )
        if not cascade:
            blockers.extend(
                f"run {candidate.run_id!r} is consumed by unselected run {consumer!r}" for consumer in consumers
            )
        else:
            for consumer_id in consumers:
                consumer = catalog.get(consumer_id)
                if consumer is None:
                    blockers.append(f"consumer run {consumer_id!r} is missing from control state")
                else:
                    visit(consumer)
        visiting.remove(candidate.run_id)

    visit(root)
    if not cascade:
        selected = {root.run_id: root}
    return _assemble_plan(
        mode="run",
        project_id=root.project_id,
        root_run_id=root_run_id,
        selected=selected,
        registry_image_owners=catalog.registry_image_owners(),
        dependency_edges=tuple(
            (candidate.run_id, consumer)
            for candidate in selected.values()
            for consumer in candidate.consumers
            if consumer in selected
        ),
        warnings=warnings,
        blockers=[*blockers, *catalog.registry_inventory_blockers()],
        reason=reason,
    )


@dataclass(frozen=True, slots=True)
class _OrphanAssessment:
    blockers: tuple[str, ...]
    registry_deletions: tuple[str, ...]
    registry_retained: tuple[tuple[str, tuple[str, ...]], ...]
    settle_admission: bool
    attributed_images: tuple[str, ...]


_SETTLED_ADMISSION_STATES = frozenset({"completed", "cancelled"})
_TERMINAL_PROVIDER_STATES = frozenset({"succeeded", "failed", "cancelled", "lost"})


def _assess_orphan(
    orphan: OrphanTrackingRun,
    *,
    registry_image_owners: Mapping[str, tuple[str, ...]],
    registry_inventory_blockers: tuple[str, ...],
    now: datetime,
    stale_after: timedelta,
) -> _OrphanAssessment:
    run = orphan.run_id
    blockers: list[str] = []
    if orphan.evidence_provider != "trackio":
        blockers.append(f"orphan run {run!r} evidence provider is {orphan.evidence_provider!r}")
    if orphan.evidence_project != orphan.project_id:
        blockers.append(f"orphan run {run!r} evidence is in project {orphan.evidence_project!r}")

    # (a) provider attribution: live inventories, then the admission entry.
    if not orphan.provider.checked and not orphan.provider.blockers:
        blockers.append(f"orphan run {run!r} has no queryable provider inventory")
    blockers.extend(f"orphan run {run!r} provider check failed: {item}" for item in orphan.provider.blockers)
    blockers.extend(
        f"orphan run {run!r} is still attributable to provider execution state: {item}"
        for item in orphan.provider.active
    )
    settle_admission = False
    admission = orphan.admission
    if admission is not None and admission.state not in _SETTLED_ADMISSION_STATES:
        label = f"machine admission entry {admission.state}"
        before = len(blockers)
        if admission.state != "terminal_pending_evidence":
            blockers.append(
                f"orphan run {run!r} has an unsettled {label}; only terminal_pending_evidence can be settled"
            )
        if admission.control_store_status == "has-receipt":
            blockers.append(
                f"orphan run {run!r} {label} is still owned by control store {admission.control_store!r}, "
                "which holds its submission receipt; reconcile it from that project"
            )
        elif admission.control_store_status == "unknown":
            blockers.append(f"orphan run {run!r} {label} has no inspectable owning control store")
        execution = f"{admission.provider}:{admission.provider_id or 'unknown'}"
        if admission.provider_query_error is not None or admission.provider_state is None:
            blockers.append(
                f"orphan run {run!r} {label} names provider execution {execution} whose state could not be "
                f"queried ({admission.provider_query_error or 'no state'})"
            )
        elif admission.provider_state not in _TERMINAL_PROVIDER_STATES:
            blockers.append(
                f"orphan run {run!r} {label} names provider execution {execution} which is "
                f"{admission.provider_state} ({admission.provider_native_state or 'unknown'})"
            )
        settle_admission = len(blockers) == before

    # (b) registry attribution: only the image recorded by the run's own
    # admission entry is explained. It is deleted when the run is its only
    # owner and retained (as a normal purge retains it) when other runs share
    # it. Any other attribution may mean another project still controls the
    # run, so it blocks.
    blockers.extend(f"orphan run {run!r} registry check failed: {item}" for item in registry_inventory_blockers)
    attributed = tuple(_attributed_images(orphan, registry_image_owners))
    deletions: list[str] = []
    retained: list[tuple[str, tuple[str, ...]]] = []
    for reference in attributed:
        if admission is None or admission.job_image != reference:
            blockers.append(
                f"orphan run {run!r} is attributed registry image {reference!r} by an owner record other than "
                "its admission entry"
            )
            continue
        others = tuple(sorted(owner for owner in registry_image_owners.get(reference, ()) if owner != run))
        if others:
            retained.append((reference, others))
        else:
            deletions.append(reference)

    # (c) surviving consumers
    blockers.extend(
        f"orphan run {run!r} is consumed by surviving run {consumer!r}"
        for consumer in orphan.consumers
        if consumer not in {run, orphan.tracking_provider_run_id}
    )
    if not orphan.lineage_complete:
        blockers.append(f"orphan run {run!r} has incomplete tracking lineage discovery")
    blockers.extend(orphan.lineage_blockers)

    # (d) terminal, or running but stale
    if orphan.tracking_status == "running":
        if orphan.last_activity_at is None:
            blockers.append(f"orphan run {run!r} is recorded as running and has no activity timestamp")
        elif now - orphan.last_activity_at < stale_after:
            blockers.append(
                f"orphan run {run!r} is recorded as running and was active at "
                f"{orphan.last_activity_at.isoformat()} (within {_hours(stale_after)})"
            )
    elif orphan.tracking_status not in _TERMINAL_TRACKING_STATUSES:
        blockers.append(f"orphan run {run!r} tracking status {orphan.tracking_status!r} is not terminal")
    return _OrphanAssessment(
        blockers=tuple(dict.fromkeys(blockers)),
        registry_deletions=tuple(deletions),
        registry_retained=tuple(retained),
        settle_admission=settle_admission,
        attributed_images=attributed,
    )


def orphan_run_blockers(
    orphan: OrphanTrackingRun,
    *,
    registry_image_owners: Mapping[str, tuple[str, ...]],
    registry_inventory_blockers: tuple[str, ...],
    now: datetime,
    stale_after: timedelta = DEFAULT_ORPHAN_STALE_AFTER,
) -> tuple[str, ...]:
    """Return every reason an orphaned tracking run cannot be purged now.

    The four checks are deliberately independent: provider attribution
    (live inventories and any abandoned admission entry), registry
    attribution, surviving consumers, and terminal-or-stale state. Any
    unavailable inventory blocks because absence cannot be proved.
    """

    return _assess_orphan(
        orphan,
        registry_image_owners=registry_image_owners,
        registry_inventory_blockers=registry_inventory_blockers,
        now=now,
        stale_after=stale_after,
    ).blockers


def build_orphan_run_purge_plan(
    orphan: OrphanTrackingRun,
    *,
    reason: PurgeReason,
    registry_image_owners: Mapping[str, tuple[str, ...]],
    registry_inventory_blockers: tuple[str, ...],
    now: datetime,
    stale_after: timedelta = DEFAULT_ORPHAN_STALE_AFTER,
) -> PurgePlan:
    """Build an orphan plan: tracking deletion, plus proven registry/admission cleanup.

    The tracking run is always the only evidence deleted. When the run's
    abandoned admission entry is proven settleable, the plan also settles it
    (local plane) and deletes the actual-job image that entry exclusively
    owns (registry plane). Provider records are never cleaned without a
    submission receipt; their terminal state is recorded as evidence.
    """

    run = orphan.run_id
    assessment = _assess_orphan(
        orphan,
        registry_image_owners=registry_image_owners,
        registry_inventory_blockers=registry_inventory_blockers,
        now=now,
        stale_after=stale_after,
    )
    admission = orphan.admission
    warnings = [
        f"orphan purge: run {run!r} has no local submission receipt; provider records and workspaces are not cleaned"
    ]
    if orphan.evidence_retention == "pinned":
        warnings.append(f"run {run!r} is pinned; explicit run purge overrides its retention pin")
    warnings.extend(
        f"job image {reference!r} retained; referenced by unselected run(s): "
        + ", ".join(repr(owner) for owner in owners)
        for reference, owners in assessment.registry_retained
    )
    retained_references = {reference for reference, _owners in assessment.registry_retained}
    if (
        orphan.recorded_job_image
        and orphan.recorded_job_image not in assessment.registry_deletions
        and orphan.recorded_job_image not in retained_references
    ):
        warnings.append(f"registry manifest {orphan.recorded_job_image!r} recorded by the run is left untouched")
    if orphan.tracking_status == "running" and orphan.last_activity_at is not None:
        warnings.append(f"run {run!r} is recorded as running but stale since {orphan.last_activity_at.isoformat()}")
    registry_actions = tuple(
        PurgeAction(
            action_id=f"registry:{run}" if index == 0 else f"registry:{run}:{index}",
            plane="registry",
            kind="registry.delete_manifest",
            target={"reference": reference, "run_id": run},
            precondition={"orphan": True, "exclusive_owner": run},
        )
        for index, reference in enumerate(assessment.registry_deletions)
    )
    tracking_id = f"tracking:{run}"
    local_actions: tuple[PurgeAction, ...] = ()
    if assessment.settle_admission and admission is not None:
        local_actions = (
            PurgeAction(
                action_id=f"local:{run}:admission",
                plane="local",
                kind="local.settle_admission",
                target={
                    "run_id": run,
                    "admission_key": admission.admission_key,
                    "provider": admission.provider,
                    "provider_id": admission.provider_id,
                    "note": (
                        f"settled by orphan purge ({reason.category}); owning control store "
                        f"{admission.control_store_status}; provider execution {admission.provider_state}"
                    ),
                },
                depends_on=(tracking_id,),
                precondition={
                    "state": "terminal_pending_evidence",
                    "control_store": admission.control_store_status,
                    "provider_state": admission.provider_state,
                },
            ),
        )
    return PurgePlan.build(
        mode="run",
        project_id=orphan.project_id,
        run_ids=(run,),
        root_run_id=run,
        registry_actions=registry_actions,
        tracking_actions=(
            PurgeAction(
                action_id=tracking_id,
                plane="tracking",
                kind="tracking.delete_run",
                target={
                    "provider": orphan.evidence_provider,
                    "project": orphan.evidence_project,
                    "provider_run_id": orphan.tracking_provider_run_id,
                },
                depends_on=tuple(action.action_id for action in registry_actions),
                precondition={"orphan": True},
            ),
        ),
        local_actions=local_actions,
        warnings=tuple(warnings),
        blockers=assessment.blockers,
        reason=reason,
        basis=orphan_basis(
            orphan,
            stale_after=stale_after,
            attributed_images=list(assessment.attributed_images),
            registry_inventory_complete=not registry_inventory_blockers,
            registry_deletions=list(assessment.registry_deletions),
            registry_retained=[reference for reference, _owners in assessment.registry_retained],
            settle_admission=assessment.settle_admission,
        ),
    )


def _attributed_images(orphan: OrphanTrackingRun, owners: Mapping[str, tuple[str, ...]]) -> list[str]:
    return sorted(reference for reference, run_ids in owners.items() if orphan.run_id in run_ids)


def orphan_basis(
    orphan: OrphanTrackingRun,
    *,
    stale_after: timedelta,
    attributed_images: list[str],
    registry_inventory_complete: bool,
    registry_deletions: list[str] | None = None,
    registry_retained: list[str] | None = None,
    settle_admission: bool = False,
) -> dict[str, JsonValue]:
    """Return the secret-free, digest-bound evidence recorded in the tombstone."""

    admission = orphan.admission
    return {
        "kind": ORPHAN_TRACKING_RUN_BASIS,
        "local_receipt": "absent",
        "tracking_provider_run_id": orphan.tracking_provider_run_id,
        "tracking_status": orphan.tracking_status,
        "last_activity_at": orphan.last_activity_at.isoformat() if orphan.last_activity_at is not None else None,
        "stale_after_seconds": int(stale_after.total_seconds()),
        "recorded_provider": orphan.recorded_provider,
        "provider_inventories": list(orphan.provider.checked),
        "active_provider_executions": list(orphan.provider.active),
        "admission_state": admission.state if admission is not None else None,
        "admission_control_store": admission.control_store if admission is not None else None,
        "admission_control_store_status": admission.control_store_status if admission is not None else None,
        "admission_provider_execution": (
            f"{admission.provider}:{admission.provider_id or 'unknown'} "
            f"({admission.provider_state or 'unqueried'}, {admission.provider_native_state or 'unknown'})"
            if admission is not None
            else None
        ),
        "admission_settle": settle_admission,
        "registry_attributed_images": list(attributed_images),
        "registry_deletions": list(registry_deletions or ()),
        "registry_retained_shared": list(registry_retained or ()),
        "registry_inventory_complete": registry_inventory_complete,
        "tracked_artifacts": orphan.tracked_artifacts,
        "surviving_consumers": list(orphan.consumers),
    }


def _hours(value: timedelta) -> str:
    hours = value.total_seconds() / 3600
    return f"{hours:g} h"


def build_project_purge_plan(
    catalog: PurgeRunCatalog,
    *,
    project_id: str,
    reason: PurgeReason,
) -> PurgePlan:
    """Build a blocked-or-applicable plan for every known run in one project."""

    candidates = {candidate.run_id: candidate for candidate in catalog.list()}
    selected = {run_id: candidate for run_id, candidate in candidates.items() if candidate.project_id == project_id}
    blockers: list[str] = []
    warnings: list[str] = []
    if not selected:
        blockers.append(f"project {project_id!r} has no known execution runs")
    for candidate in candidates.values():
        if candidate.project_id != project_id:
            blockers.append(f"unmatched run {candidate.run_id!r} belongs to project {candidate.project_id!r}")
    if selected:
        root = next(iter(selected.values()))
        for candidate in selected.values():
            _validate_candidate(root, candidate, blockers)
            if candidate.evidence_retention == "pinned":
                blockers.append(
                    f"pinned run {candidate.run_id!r} cannot be included in project purge without explicit run selection"
                )
            for consumer in candidate.consumers:
                if consumer not in selected:
                    blockers.append(f"run {candidate.run_id!r} has unmatched consumer {consumer!r}")
            blockers.extend(
                f"run {candidate.run_id!r} has external consumer {consumer!r}"
                for consumer in candidate.external_consumers
            )
        return _assemble_plan(
            mode="project",
            project_id=project_id,
            root_run_id=None,
            selected=selected,
            registry_image_owners=catalog.registry_image_owners(),
            dependency_edges=tuple(
                (candidate.run_id, consumer)
                for candidate in selected.values()
                for consumer in candidate.consumers
                if consumer in selected
            ),
            warnings=warnings,
            blockers=[*blockers, *catalog.registry_inventory_blockers()],
            reason=reason,
        )
    return PurgePlan.build(
        mode="project",
        project_id=project_id,
        run_ids=(),
        root_run_id=None,
        warnings=tuple(warnings),
        blockers=tuple(dict.fromkeys(blockers)),
        reason=reason,
    )


def _assemble_plan(
    *,
    mode: PurgeMode,
    project_id: str,
    root_run_id: str | None,
    selected: Mapping[str, PurgeRunCandidate],
    registry_image_owners: Mapping[str, tuple[str, ...]],
    dependency_edges: tuple[tuple[str, str], ...],
    warnings: list[str],
    blockers: list[str],
    reason: PurgeReason,
) -> PurgePlan:
    run_ids = tuple(selected)
    provider_actions: list[PurgeAction] = []
    registry_actions: list[PurgeAction] = []
    tracking_actions: list[PurgeAction] = []
    local_actions: list[PurgeAction] = []

    for candidate in selected.values():
        provider_id = f"provider:{candidate.run_id}"
        if "provider" not in candidate.completed_planes:
            provider_actions.append(
                PurgeAction(
                    action_id=provider_id,
                    plane="provider",
                    kind="provider.cleanup",
                    target={
                        "provider": candidate.provider,
                        "provider_id": candidate.provider_id,
                        "run_id": candidate.run_id,
                    },
                    precondition={"state": "terminal", "reconciled": candidate.reconciled},
                )
            )

    registry_action_by_run: dict[str, str] = {}
    selected_image_owners: dict[str, list[PurgeRunCandidate]] = {}
    for candidate in selected.values():
        if "registry" in candidate.completed_planes:
            continue
        if candidate.image is None:
            blockers.append(f"run {candidate.run_id!r} has no digest-pinned actual-job image")
            continue
        selected_image_owners.setdefault(candidate.image.value, []).append(candidate)

    for reference, owners in selected_image_owners.items():
        external_owners = sorted(
            run_id for run_id in registry_image_owners.get(reference, ()) if run_id not in selected
        )
        if external_owners:
            warnings.append(
                f"job image {reference!r} retained; referenced by unselected "
                f"run(s): {', '.join(repr(run_id) for run_id in external_owners)}"
            )
            continue
        action_id = f"registry:{owners[0].run_id}"
        registry_actions.append(
            PurgeAction(
                action_id=action_id,
                plane="registry",
                kind="registry.delete_manifest",
                target={"reference": reference, "run_id": owners[0].run_id},
                depends_on=tuple(
                    f"provider:{owner.run_id}" for owner in owners if "provider" not in owner.completed_planes
                ),
            )
        )
        registry_action_by_run.update((owner.run_id, action_id) for owner in owners)

    ordered_tracking = _leaf_first(selected, root_run_id)
    for candidate in ordered_tracking:
        if "tracking" in candidate.completed_planes:
            warnings.append(f"run {candidate.run_id!r} tracking plane already completed; resuming remaining planes")
        else:
            consumer_dependencies = tuple(
                f"tracking:{consumer}" for consumer in candidate.consumers if consumer in selected
            )
            registry_id = registry_action_by_run.get(candidate.run_id)
            registry_dependencies = (registry_id,) if registry_id is not None else ()
            tracking_actions.append(
                PurgeAction(
                    action_id=f"tracking:{candidate.run_id}",
                    plane="tracking",
                    kind="tracking.delete_run",
                    target={
                        "provider": candidate.evidence_provider,
                        "project": candidate.evidence_project,
                        "provider_run_id": candidate.tracking_provider_run_id,
                    },
                    depends_on=(*consumer_dependencies, *registry_dependencies),
                )
            )
        paths = candidate.local_paths or ((candidate.workspace,) if candidate.workspace is not None else ())
        if not paths:
            warnings.append(f"run {candidate.run_id!r} has no local state target")
        for index, path in enumerate(paths):
            if path is None:
                continue
            tracking_id = f"tracking:{candidate.run_id}"
            local_dependencies = (tracking_id,) if "tracking" not in candidate.completed_planes else ()
            local_actions.append(
                PurgeAction(
                    action_id=f"local:{candidate.run_id}:{index}",
                    plane="local",
                    kind="local.remove_path",
                    target={"run_id": candidate.run_id, "path": str(path)},
                    depends_on=local_dependencies,
                )
            )

    if mode == "project" and selected:
        tracking_actions.append(
            PurgeAction(
                action_id="tracking:project",
                plane="tracking",
                kind="tracking.delete_project",
                target={"provider": "trackio", "project": project_id},
                depends_on=tuple(action.action_id for action in tracking_actions),
            )
        )

    return PurgePlan.build(
        mode=mode,
        project_id=project_id,
        run_ids=run_ids,
        root_run_id=root_run_id,
        dependency_edges=dependency_edges,
        provider_actions=tuple(provider_actions),
        registry_actions=tuple(registry_actions),
        tracking_actions=tuple(tracking_actions),
        local_actions=tuple(local_actions),
        warnings=tuple(dict.fromkeys(warnings)),
        blockers=tuple(dict.fromkeys(blockers)),
        reason=reason,
    )


def _validate_candidate(root: PurgeRunCandidate, candidate: PurgeRunCandidate, blockers: list[str]) -> None:
    if candidate.project_id != root.project_id:
        blockers.append(f"run {candidate.run_id!r} belongs to project {candidate.project_id!r}")
    if candidate.provider != candidate.provider.strip():
        blockers.append(f"run {candidate.run_id!r} has an invalid provider identity")
    if candidate.state not in {"succeeded", "failed", "cancelled", "lost"}:
        blockers.append(f"run {candidate.run_id!r} is not terminal ({candidate.state})")
    if not candidate.reconciled:
        blockers.append(f"run {candidate.run_id!r} is not reconciled")
    if candidate.evidence_provider != "trackio":
        blockers.append(f"run {candidate.run_id!r} evidence provider is {candidate.evidence_provider!r}")
    if candidate.evidence_project != root.project_id:
        blockers.append(f"run {candidate.run_id!r} evidence is in project {candidate.evidence_project!r}")
    if not candidate.lineage_complete:
        blockers.append(f"run {candidate.run_id!r} has incomplete tracking lineage discovery")
    blockers.extend(candidate.lineage_blockers)


def _leaf_first(selected: Mapping[str, PurgeRunCandidate], root_run_id: str | None) -> tuple[PurgeRunCandidate, ...]:
    ordered: list[PurgeRunCandidate] = []
    visited: set[str] = set()

    def visit(run_id: str) -> None:
        if run_id in visited:
            return
        visited.add(run_id)
        candidate = selected.get(run_id)
        if candidate is None:
            return
        for consumer in candidate.consumers:
            visit(consumer)
        ordered.append(candidate)

    if root_run_id is not None:
        visit(root_run_id)
    for run_id in selected:
        visit(run_id)
    return tuple(ordered)


__all__ = [
    "DEFAULT_ORPHAN_STALE_AFTER",
    "ORPHAN_TRACKING_RUN_BASIS",
    "OrphanAdmissionEntry",
    "OrphanProviderInventory",
    "OrphanTrackingRun",
    "PurgeRunCandidate",
    "PurgeRunCatalog",
    "build_orphan_run_purge_plan",
    "build_project_purge_plan",
    "build_run_purge_plan",
    "orphan_basis",
    "orphan_run_blockers",
]
