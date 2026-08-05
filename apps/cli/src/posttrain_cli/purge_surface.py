"""Shared purge planning, presentation, and guarded apply helpers."""

from __future__ import annotations

import importlib
import json
from typing import Any

from posttrain.execution import (
    ExecutionProviderPurgeExecutor,
    ExecutionSubmissionStore,
    LocalStatePurgeExecutor,
    PurgeActionExecutor,
    PurgePlan,
    PurgePlane,
    PurgeRunCandidate,
    PurgeStore,
    RegistryManifestRef,
    apply_purge_plan,
    build_project_purge_plan,
    build_run_purge_plan,
)

from .context import CliState
from .execution_config import load_local_execution_config
from .execution_provider import execution_service_for_run
from .state_layout import cache_path
from .tracking_config import project_tracking_environment


def candidate_catalog(layout: Any) -> dict[str, PurgeRunCandidate]:
    """Build a deliberately fail-closed local inventory for preview commands."""

    store = ExecutionSubmissionStore(layout.state)
    purge_store = PurgeStore(layout.state)
    candidates: dict[str, PurgeRunCandidate] = {}
    for submission in store.list_submissions():
        evidence = submission.evidence_source
        state = "unknown"
        reconciled = False
        tracking_provider_run_id: str | None = None
        try:
            cleaned = store.cleaned_result(submission.run_id)
        except Exception:
            cleaned = None
        if cleaned is not None:
            state = cleaned.record.state
        snapshot = store.run_root(submission.run_id) / "reconciliation.json"
        if snapshot.is_file():
            try:
                payload = json.loads(snapshot.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    reconciled = payload.get("state") == "consistent"
                    value = payload.get("tracking_provider_run_id")
                    tracking_provider_run_id = value if isinstance(value, str) else None
            except (OSError, json.JSONDecodeError):
                reconciled = False
        try:
            state = execution_service_for_run(layout, submission.run_id).status(submission.run_id).state
        except Exception:
            pass
        image = None
        try:
            image = RegistryManifestRef.parse(submission.job_image)
        except Exception:
            pass
        # The submission receipt records the provider's workspace path. For a
        # dstack run that path belongs to the remote worker and must not be
        # handed to the local-state executor. Always purge the local execution
        # receipt directory; add the workspace only for the local provider,
        # whose storage root is intentionally configured on this machine.
        local_paths = [store.run_root(submission.run_id)]
        workspace = submission.run_workspace if submission.provider == "local" else None
        if workspace is not None and workspace not in local_paths:
            local_paths.append(workspace)
        candidates[submission.run_id] = PurgeRunCandidate(
            run_id=submission.run_id,
            project_id=layout.project_id,
            provider=submission.provider,
            provider_id=submission.provider_id,
            state=state,
            reconciled=reconciled,
            evidence_provider=evidence.provider if evidence is not None else "unknown",
            evidence_project=evidence.project if evidence is not None else layout.project_id,
            tracking_provider_run_id=tracking_provider_run_id
            or (evidence.source_id if evidence is not None else submission.run_id),
            image=image,
            workspace=workspace,
            local_paths=tuple(local_paths),
            completed_planes=_completed_purge_planes(
                purge_store,
                run_id=submission.run_id,
                project_id=layout.project_id,
            ),
            lineage_complete=False,
        )
    _populate_trackio_lineage(layout, candidates)
    return candidates


def _completed_purge_planes(
    store: PurgeStore,
    *,
    run_id: str,
    project_id: str,
) -> tuple[PurgePlane, ...]:
    """Return planes already completed by an earlier immutable purge plan.

    A failed apply can leave provider, registry, and tracking resources gone
    while the final local action is still pending.  A new preview must resume
    from the journal without trusting the old plan's (possibly stale) local
    target.  Only completed/skipped journal events count; started/failed events
    remain actionable.
    """

    completed: set[PurgePlane] = set()
    if not store.root.is_dir():
        return ()
    for directory in store.root.iterdir():
        if not directory.is_dir():
            continue
        try:
            plan = store.load_plan(directory.name)
            if plan.project_id != project_id or run_id not in plan.run_ids:
                continue
            events = store.journal(directory.name)
        except Exception:
            continue
        settled = {
            str(event["action_id"])
            for event in events
            if event.get("status") in {"completed", "skipped"}
        }
        for action in plan.actions:
            if action.action_id in settled:
                completed.add(action.plane)
    return tuple(sorted(completed))


def _populate_trackio_lineage(layout: Any, candidates: dict[str, PurgeRunCandidate]) -> None:
    trackio_candidates = {
        run_id: candidate for run_id, candidate in candidates.items() if candidate.evidence_provider == "trackio"
    }
    if not trackio_candidates:
        return
    try:
        module = importlib.import_module("posttrain_tracking_trackio")
        admin_type = module.TrackioLifecycleAdmin
        environment = project_tracking_environment(layout)
        server_url = environment.get("POSTTRAIN_TRACKIO_SERVER_URL")
        if not server_url:
            raise RuntimeError("POSTTRAIN_TRACKIO_SERVER_URL is not configured")
        admin = admin_type(server_url, write_token=environment.get("TRACKIO_WRITE_TOKEN"))
    except Exception as error:
        for run_id, candidate in tuple(trackio_candidates.items()):
            candidates[run_id] = _lineage_failure(
                candidate,
                f"tracking lifecycle service unavailable ({type(error).__name__})",
            )
        return
    provider_to_run = {candidate.tracking_provider_run_id: run_id for run_id, candidate in trackio_candidates.items()}
    for run_id, candidate in tuple(trackio_candidates.items()):
        if "tracking" in candidate.completed_planes:
            candidates[run_id] = _replace_lineage(
                candidate,
                consumers=(),
                external_consumers=(),
                lineage_complete=True,
                lineage_blockers=(),
            )
            continue
        try:
            plan = admin.plan_run_purge(
                project=candidate.evidence_project,
                provider_run_ids=(candidate.tracking_provider_run_id,),
            )
            consumers = {
                provider_to_run[consumer]
                for artifact in plan.artifacts
                for consumer in artifact.consumer_run_ids
                if consumer in provider_to_run and consumer != candidate.tracking_provider_run_id
            }
            external = tuple(
                consumer
                for artifact in plan.artifacts
                for consumer in artifact.consumer_run_ids
                if consumer not in provider_to_run
            )
            candidates[run_id] = _replace_lineage(
                candidate,
                consumers=tuple(sorted(consumers)),
                external_consumers=tuple(sorted(set(external))),
                lineage_complete=not plan.blockers,
                lineage_blockers=plan.blockers,
            )
        except Exception as error:
            candidates[run_id] = _lineage_failure(
                candidate,
                f"tracking lineage preview failed ({type(error).__name__})",
            )


def _replace_lineage(candidate: PurgeRunCandidate, **values: Any) -> PurgeRunCandidate:
    from dataclasses import replace

    return replace(candidate, **values)


def _lineage_failure(candidate: PurgeRunCandidate, blocker: str) -> PurgeRunCandidate:
    return _replace_lineage(candidate, lineage_complete=False, lineage_blockers=(blocker,))


def plan_store(layout: Any) -> PurgeStore:
    return PurgeStore(layout.state)


def save_run_preview(layout: Any, run_id: str, *, cascade: bool) -> PurgePlan:
    candidates = candidate_catalog(layout)
    plan = build_run_purge_plan(
        _Catalog(candidates),
        root_run_id=run_id,
        cascade=cascade,
    )
    return plan_store(layout).save_plan(plan)


def save_project_preview(layout: Any) -> PurgePlan:
    candidates = candidate_catalog(layout)
    plan = build_project_purge_plan(_Catalog(candidates), project_id=layout.project_id)
    if not plan.blockers:
        plan = PurgePlan.build(
            mode=plan.mode,
            project_id=plan.project_id,
            run_ids=plan.run_ids,
            root_run_id=None,
            dependency_edges=plan.dependency_edges,
            provider_actions=plan.provider_actions,
            registry_actions=plan.registry_actions,
            tracking_actions=plan.tracking_actions,
            local_actions=plan.local_actions,
            warnings=plan.warnings,
            blockers=("project purge cross-plane inventory adapters are not configured",),
        )
    return plan_store(layout).save_plan(plan)


def render_plan(plan: PurgePlan) -> str:
    counts = {
        "provider": len(plan.provider_actions),
        "registry": len(plan.registry_actions),
        "tracking": len(plan.tracking_actions),
        "local": len(plan.local_actions),
    }
    lines = [
        "Purge preview — no changes made",
        f"Target: {plan.root_run_id or plan.project_id} (project: {plan.project_id})",
        f"Closure: {len(plan.run_ids)} runs, {len(plan.dependency_edges)} artifact-consumer edges",
        f"Provider: {counts['provider']} terminal records/workspaces",
        f"OCI: {counts['registry']} digest-pinned actual-job manifests",
        f"Trackio: {counts['tracking']} runs",
        f"Local: {counts['local']} execution targets",
        "Blockers: " + ("; ".join(plan.blockers) if plan.blockers else "none"),
        "Warnings: " + ("; ".join(plan.warnings) if plan.warnings else "none"),
        f"Plan: {plan.purge_id}",
        f"Digest: {plan.digest}",
    ]
    if plan.blockers:
        lines.append("Next: resolve blockers and create a new preview")
    else:
        lines.append(f"Next: posttrain purge apply {plan.purge_id}")
    return "\n".join(lines)


def apply_saved_plan(
    state: CliState,
    purge_id: str,
    *,
    expected_digest: str | None,
    assume_yes: bool,
) -> Any:
    layout = state.layout()
    store = plan_store(layout)
    plan = store.load_plan(purge_id)
    if expected_digest is not None and expected_digest != plan.digest:
        raise ValueError("purge plan digest does not match --expect-digest")
    if plan.blockers:
        raise ValueError("purge plan is blocked: " + "; ".join(plan.blockers))
    if not assume_yes:
        if not state.json_output:
            typed = input(f"Type {purge_id} to apply this purge: ")
            if typed != purge_id:
                raise ValueError("purge id confirmation did not match")
        else:
            raise ValueError("JSON purge apply requires --expect-digest and --yes")
    if assume_yes and expected_digest is None:
        raise ValueError("non-interactive purge apply requires --expect-digest")

    executors = _apply_executors(layout, plan)
    return apply_purge_plan(
        store,
        purge_id,
        executors,
    )


def _apply_executors(layout: Any, plan: PurgePlan) -> dict[PurgePlane, PurgeActionExecutor]:
    services = {
        str(action.target["run_id"]): execution_service_for_run(layout, str(action.target["run_id"]))
        for action in plan.provider_actions
    }
    local_roots = [layout.state, cache_path(layout, "runs")]
    try:
        local_binding = load_local_execution_config(layout).local
    except Exception:
        local_binding = None
    if local_binding is not None and local_binding.storage is not None:
        local_roots.append(local_binding.storage.run_root)
    executors: dict[PurgePlane, PurgeActionExecutor] = {
        "provider": ExecutionProviderPurgeExecutor(services),
        # Submission receipts live in the project state root, while the
        # local provider's run workspaces live in the configured machine
        # storage root. Both are exact, framework-owned state roots.
        "local": LocalStatePurgeExecutor(tuple(local_roots)),
    }
    if plan.registry_actions:
        try:
            module = importlib.import_module("posttrain_execution_buildkit")
            transport = module.UrllibDistributionTransport()
            admin = module.DistributionRegistryLifecycleAdmin(transport)
            executors["registry"] = module.RegistryPurgeActionExecutor(admin)
        except Exception as error:
            raise RuntimeError(f"OCI registry purge adapter is unavailable ({type(error).__name__})") from error
    if plan.tracking_actions:
        try:
            module = importlib.import_module("posttrain_tracking_trackio")
            environment = project_tracking_environment(layout)
            server_url = environment.get("POSTTRAIN_TRACKIO_SERVER_URL")
            if not server_url:
                raise RuntimeError("POSTTRAIN_TRACKIO_SERVER_URL is not configured")
            admin = module.TrackioLifecycleAdmin(
                server_url,
                write_token=environment.get("TRACKIO_WRITE_TOKEN"),
            )
            executors["tracking"] = module.TrackioPurgeActionExecutor(admin)
        except Exception as error:
            raise RuntimeError(f"Trackio purge adapter is unavailable ({type(error).__name__})") from error
    return executors


class _Catalog:
    def __init__(self, values: dict[str, PurgeRunCandidate]) -> None:
        self._values = values

    def get(self, run_id: str) -> PurgeRunCandidate | None:
        return self._values.get(run_id)

    def list(self) -> tuple[PurgeRunCandidate, ...]:
        return tuple(self._values.values())


__all__ = [
    "apply_saved_plan",
    "candidate_catalog",
    "plan_store",
    "render_plan",
    "save_project_preview",
    "save_run_preview",
]
