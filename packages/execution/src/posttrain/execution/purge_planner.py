"""Provider-neutral run purge closure planning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from posttrain.common import ContractError

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
    "PurgeRunCandidate",
    "PurgeRunCatalog",
    "build_project_purge_plan",
    "build_run_purge_plan",
]
