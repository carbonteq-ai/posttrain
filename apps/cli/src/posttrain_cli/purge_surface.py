"""Shared purge planning, presentation, and guarded apply helpers."""

from __future__ import annotations

import importlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from posttrain.catalog import load_project_layout
from posttrain.common import ContractError
from posttrain.execution import (
    DEFAULT_ORPHAN_STALE_AFTER,
    ORPHAN_TRACKING_RUN_BASIS,
    SETTLE_ADMISSION_KIND,
    AdmissionEntry,
    AdmissionSettlePurgeExecutor,
    ExecutionHandle,
    ExecutionProviderPurgeExecutor,
    ExecutionSubmissionStore,
    LocalStatePurgeExecutor,
    OrphanAdmissionEntry,
    OrphanProviderInventory,
    OrphanTrackingRun,
    PurgeAction,
    PurgeActionExecutor,
    PurgePlan,
    PurgePlane,
    PurgeReason,
    PurgeRunCandidate,
    PurgeStore,
    PurgeTombstone,
    RegistryManifestRef,
    apply_purge_plan,
    build_orphan_run_purge_plan,
    build_project_purge_plan,
    build_run_purge_plan,
    orphan_run_blockers,
)

from .context import CliState
from .execution_config import (
    load_local_execution_config,
    load_machine_config,
    resolve_admission_state_root,
)
from .execution_provider import execution_admission_service, execution_service_for_run
from .state_layout import cache_path
from .tracking_config import project_tracking_environment


def candidate_catalog(
    layout: Any,
    *,
    discover_lineage_for: tuple[str, ...] | None = None,
    refresh_status_for: tuple[str, ...] | None = None,
) -> dict[str, PurgeRunCandidate]:
    """Build a deliberately fail-closed local inventory for preview commands.

    Run previews still need the complete local identity map: a Trackio consumer
    might belong to another submitted framework run.  They do *not* need to
    issue one remote lineage request per historical run merely to preview one
    selected root.  ``discover_lineage_for`` limits remote discovery to that
    root (or roots); project preview deliberately leaves it unset. Provider
    status follows the same rule: historical receipts supply the identity map,
    while a run preview refreshes only the root whose terminality it must
    prove.
    """

    store = ExecutionSubmissionStore(layout.state)
    purge_stores = _plan_stores(layout)
    candidates: dict[str, PurgeRunCandidate] = {}
    submissions = store.list_submissions()
    for submission in submissions:
        evidence = submission.evidence_source
        state = "unknown"
        reconciled = False
        tracking_provider_run_id: str | None = None
        try:
            cleaned = store.cleaned_result(submission.run_id)
        except Exception:
            cleaned = None
        cleanup_evidence_state: str | None = None
        if cleaned is not None:
            try:
                cleanup_payload = json.loads(
                    (store.run_root(submission.run_id) / "cleanup.json").read_text(encoding="utf-8")
                )
                if isinstance(cleanup_payload, dict):
                    value = cleanup_payload.get("evidence_state")
                    cleanup_evidence_state = value if isinstance(value, str) else None
            except (OSError, json.JSONDecodeError):
                cleanup_evidence_state = None
        provider_terminal_without_tracking = bool(cleaned is not None and cleanup_evidence_state == "provider-terminal")
        if cleaned is not None:
            state = cleaned.record.state
        if provider_terminal_without_tracking:
            # Cleanup already performed the guarded remote lookup and retained
            # the fact that no tracking run exists. Treat that plane as
            # completed so purge does not turn the source id into a fictitious
            # provider run id.
            reconciled = True
        snapshot = store.run_root(submission.run_id) / "reconciliation.json"
        if snapshot.is_file():
            try:
                payload = json.loads(snapshot.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    reconciled = reconciled or _reconciliation_allows_purge(payload)
                    value = payload.get("tracking_provider_run_id")
                    tracking_provider_run_id = value if isinstance(value, str) else None
            except (OSError, json.JSONDecodeError):
                reconciled = False
        if refresh_status_for is None or submission.run_id in refresh_status_for:
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
        completed_planes = _completed_purge_planes(
            purge_stores,
            run_id=submission.run_id,
            project_id=layout.project_id,
        )
        if provider_terminal_without_tracking and "tracking" not in completed_planes:
            completed_planes = (*completed_planes, "tracking")
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
            completed_planes=completed_planes,
            lineage_complete=False,
            evidence_retention=submission.evidence_retention,
        )
    submission_ids = {submission.run_id for submission in submissions}
    retired_run_ids: set[str] = set()
    for purge_store in purge_stores:
        # ``_completed_purge_run_ids`` takes the state root that owns the
        # ``purges`` directory, not the store's own ``purges`` root.
        retired_run_ids.update(_completed_purge_run_ids(purge_store.root.parent))
    try:
        admission_entries = execution_admission_service(layout).list()
    except Exception:
        admission_entries = ()
    for entry in admission_entries:
        try:
            project_id = entry.plan.request.run_spec.project_id
        except AttributeError:
            continue
        if (
            project_id != layout.project_id
            or entry.run_id in submission_ids
            or entry.run_id in retired_run_ids
            or entry.state != "cancelled"
        ):
            continue
        # A cancelled admission without a submission receipt never reached a
        # provider or tracking backend. Preserve the image reference for the
        # registry ownership check, but mark those absent planes complete so
        # purge only creates the durable tombstone (and removes an unshared
        # job image when appropriate).
        settled_planes: set[PurgePlane] = set(
            _completed_purge_planes(
                purge_stores,
                run_id=entry.run_id,
                project_id=layout.project_id,
            )
        )
        settled_planes.update(("provider", "tracking"))
        completed_planes = tuple(sorted(settled_planes))
        candidates[entry.run_id] = PurgeRunCandidate(
            run_id=entry.run_id,
            project_id=layout.project_id,
            provider=entry.plan.provider,
            provider_id=entry.plan.native_plan_id or entry.run_id,
            state="cancelled",
            reconciled=True,
            evidence_provider="trackio",
            evidence_project=layout.project_id,
            tracking_provider_run_id=entry.run_id,
            image=RegistryManifestRef.parse(entry.plan.request.image.value),
            completed_planes=completed_planes,
            lineage_complete=True,
        )
    _populate_trackio_lineage(layout, candidates, discover_run_ids=discover_lineage_for)
    return candidates


def _reconciliation_allows_purge(payload: dict[str, Any]) -> bool:
    """Prove both run authorities are terminal without erasing diagnostics.

    Provider cleanup can report ``cancelled`` after Trackio has already
    finalized the workload as ``failed`` (or the inverse).  That outcome
    disagreement must remain visible in reconciliation, but it is not an
    in-flight-state ambiguity: both authorities are terminal and the exact
    Trackio run is known.  Purge may therefore proceed through its independent
    ownership and lineage gates.  Successful, lost, partial, or unidentified
    disagreements remain fail-closed.
    """

    if payload.get("state") == "consistent":
        return True
    provider_record = payload.get("provider_record")
    provider_state = provider_record.get("state") if isinstance(provider_record, dict) else None
    tracking_provider_run_id = payload.get("tracking_provider_run_id")
    return (
        payload.get("state") == "inconsistent"
        and provider_state in {"failed", "cancelled"}
        and payload.get("tracking_status") in {"failed", "cancelled"}
        and isinstance(tracking_provider_run_id, str)
        and bool(tracking_provider_run_id.strip())
    )


def _completed_purge_planes(
    stores: tuple[PurgeStore, ...],
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
    for store in stores:
        if not store.root.is_dir():
            continue
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
            settled = {str(event["action_id"]) for event in events if event.get("status") in {"completed", "skipped"}}
            for action in plan.actions:
                if action.action_id in settled:
                    completed.add(action.plane)
    return tuple(sorted(completed))


def _populate_trackio_lineage(
    layout: Any,
    candidates: dict[str, PurgeRunCandidate],
    *,
    discover_run_ids: tuple[str, ...] | None = None,
) -> None:
    trackio_candidates = {
        run_id: candidate
        for run_id, candidate in candidates.items()
        if candidate.evidence_provider == "trackio" and "tracking" not in candidate.completed_planes
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
        admin = admin_type(
            server_url,
            write_token=environment.get("TRACKIO_WRITE_TOKEN"),
            ca_bundle=_machine_trust_bundle(),
        )
    except Exception as error:
        for run_id, candidate in tuple(trackio_candidates.items()):
            candidates[run_id] = _lineage_failure(
                candidate,
                f"tracking lifecycle service unavailable ({type(error).__name__})",
            )
        return
    provider_to_run = {candidate.tracking_provider_run_id: run_id for run_id, candidate in trackio_candidates.items()}
    if discover_run_ids is not None:
        requested = set(discover_run_ids)
        trackio_candidates = {
            run_id: candidate for run_id, candidate in trackio_candidates.items() if run_id in requested
        }
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
    del layout
    return PurgeStore(resolve_admission_state_root())


def _legacy_plan_store(layout: Any) -> PurgeStore:
    return PurgeStore(layout.state)


def _plan_stores(layout: Any) -> tuple[PurgeStore, ...]:
    primary = plan_store(layout)
    legacy = _legacy_plan_store(layout)
    return (primary,) if primary.root == legacy.root else (primary, legacy)


def saved_plan_store(layout: Any, purge_id: str) -> PurgeStore:
    """Locate one plan across the machine store and legacy project store."""

    matches = tuple(store for store in _plan_stores(layout) if store.plan_path(purge_id).is_file())
    if not matches:
        return plan_store(layout)
    if len(matches) == 1:
        return matches[0]
    first = matches[0].load_plan(purge_id)
    for store in matches[1:]:
        candidate = store.load_plan(purge_id)
        if candidate.digest != first.digest or candidate.semantic_payload() != first.semantic_payload():
            raise ContractError(f"purge id {purge_id!r} conflicts across durable stores")
    return matches[0]


def load_saved_plan(layout: Any, purge_id: str) -> PurgePlan:
    return saved_plan_store(layout, purge_id).load_plan(purge_id)


def load_saved_tombstone(layout: Any, purge_id: str) -> PurgeTombstone | None:
    """Return the safe audit state when a v2 purge has begun."""

    store = saved_plan_store(layout, purge_id)
    if not store.tombstone_path(purge_id).is_file():
        return None
    return store.load_tombstone(purge_id)


def save_run_preview(
    layout: Any,
    run_id: str,
    *,
    cascade: bool,
    reason: PurgeReason,
    orphan: bool = False,
    stale_after: timedelta = DEFAULT_ORPHAN_STALE_AFTER,
) -> PurgePlan:
    if orphan:
        if cascade:
            raise ValueError("--orphan purges one tracking run and cannot be combined with --cascade")
        return plan_store(layout).save_plan(_orphan_preview(layout, run_id, reason=reason, stale_after=stale_after))
    candidates = candidate_catalog(
        layout,
        discover_lineage_for=(run_id,) if not cascade else None,
        refresh_status_for=(run_id,) if not cascade else None,
    )
    registry_owners, registry_blockers = _registry_image_inventory(layout, candidates)
    plan = build_run_purge_plan(
        _Catalog(
            candidates,
            registry_owners=registry_owners,
            registry_blockers=registry_blockers,
        ),
        root_run_id=run_id,
        reason=reason,
        cascade=cascade,
    )
    return plan_store(layout).save_plan(plan)


def _orphan_preview(
    layout: Any,
    run_id: str,
    *,
    reason: PurgeReason,
    stale_after: timedelta,
) -> PurgePlan:
    """Build an explicit tracking-only plan for a run with no local receipt."""

    candidates, local_blocker = _orphan_local_control_check(layout, run_id)
    orphan = _collect_orphan(layout, run_id) if local_blocker is None else local_blocker
    if isinstance(orphan, str):
        return PurgePlan.build(
            mode="run",
            project_id=layout.project_id,
            run_ids=(run_id,),
            root_run_id=run_id,
            reason=reason,
            blockers=(orphan,),
        )
    registry_owners, registry_blockers = _registry_image_inventory(layout, candidates)
    return build_orphan_run_purge_plan(
        orphan,
        reason=reason,
        registry_image_owners=registry_owners,
        registry_inventory_blockers=registry_blockers,
        now=datetime.now(UTC),
        stale_after=stale_after,
    )


def _orphan_local_control_check(layout: Any, run_id: str) -> tuple[dict[str, PurgeRunCandidate], str | None]:
    """Refuse orphan mode for a run that this machine still controls."""

    # No lineage discovery or provider refresh: the local inventory is only
    # the identity map and the registry-ownership seed here.
    candidates = candidate_catalog(layout, discover_lineage_for=(), refresh_status_for=())
    if run_id in candidates:
        return candidates, (
            f"run {run_id!r} has local control state on this machine; omit --orphan to use the normal purge"
        )
    return candidates, None


def _collect_orphan(layout: Any, run_id: str) -> OrphanTrackingRun | str:
    """Gather tracking, lineage, and provider evidence; return a blocker on failure."""

    if getattr(layout, "tracking", "trackio") != "trackio":
        return f"orphan purge requires Trackio evidence; project tracking is {layout.tracking!r}"
    project = layout.project_id
    try:
        module = importlib.import_module("posttrain_tracking_trackio")
        environment = project_tracking_environment(layout)
        server_url = environment.get("POSTTRAIN_TRACKIO_SERVER_URL")
        if not server_url:
            raise RuntimeError("POSTTRAIN_TRACKIO_SERVER_URL is not configured")
        activity = module.TrackioRunActivityLookup(project, server_url=server_url).find(run_id)
    except Exception as error:
        return f"tracking lookup for orphan run {run_id!r} failed ({type(error).__name__}: {_short(error)})"
    if activity is None:
        return f"run {run_id!r} was not found in tracking project {project!r}"

    consumers: tuple[str, ...] = ()
    lineage_blockers: tuple[str, ...] = ()
    lineage_complete = True
    tracked_artifacts = 0
    try:
        admin = module.TrackioLifecycleAdmin(
            server_url,
            write_token=environment.get("TRACKIO_WRITE_TOKEN"),
            ca_bundle=_machine_trust_bundle(),
        )
        lineage = admin.plan_run_purge(project=project, provider_run_ids=(activity.provider_run_id,))
        consumers = tuple(
            sorted(
                {
                    consumer
                    for artifact in lineage.artifacts
                    for consumer in artifact.consumer_run_ids
                    if consumer != activity.provider_run_id
                }
            )
        )
        lineage_blockers = tuple(lineage.blockers)
        lineage_complete = not lineage.blockers
        tracked_artifacts = len(lineage.artifacts)
    except Exception as error:
        lineage_complete = False
        lineage_blockers = (f"tracking lineage preview failed ({type(error).__name__})",)

    inventory, admission = _orphan_provider_inventory(layout, run_id, activity.recorded_provider)
    return OrphanTrackingRun(
        run_id=run_id,
        project_id=project,
        evidence_provider="trackio",
        evidence_project=project,
        tracking_provider_run_id=activity.provider_run_id,
        tracking_status=activity.status,
        last_activity_at=activity.last_activity_at,
        provider=inventory,
        admission=admission,
        recorded_provider=activity.recorded_provider,
        recorded_job_image=activity.recorded_job_image,
        consumers=consumers,
        lineage_complete=lineage_complete,
        lineage_blockers=lineage_blockers,
        tracked_artifacts=tracked_artifacts,
        evidence_retention=activity.evidence_retention,
    )


_PROVIDER_ALIASES = {"local": "local-docker", "local-docker": "local-docker", "dstack": "dstack"}
_SETTLED_ADMISSION_STATES = frozenset({"completed", "cancelled"})


def _orphan_provider_inventory(
    layout: Any,
    run_id: str,
    recorded_provider: str | None,
) -> tuple[OrphanProviderInventory, OrphanAdmissionEntry | None]:
    """Prove no provider execution still references an orphaned run id.

    A provider recorded in the run's tracking configuration must be queried
    successfully. Without one, every configured provider is queried. The
    machine admission ledger is read as well because it spans projects and
    removed checkouts on this machine; an entry for the run is returned with
    the evidence the planner needs to decide whether it may be settled.
    """

    checked: list[str] = []
    active: list[str] = []
    blockers: list[str] = []
    entry: AdmissionEntry | None = None
    try:
        entries = execution_admission_service(layout).list()
    except Exception as error:
        blockers.append(f"machine admission ledger is unavailable ({type(error).__name__})")
    else:
        checked.append("machine admission ledger")
        entry = next((item for item in entries if item.run_id == run_id), None)
    try:
        local_config = load_local_execution_config(layout, verify_published_locks=False, resolve_registry=False)
    except Exception as error:
        blockers.append(f"execution provider configuration is unavailable ({type(error).__name__})")
        admission = _orphan_admission(layout, None, entry) if entry is not None else None
        return OrphanProviderInventory(tuple(checked), tuple(active), tuple(blockers)), admission
    if recorded_provider is not None:
        name = _PROVIDER_ALIASES.get(recorded_provider)
        if name is None:
            blockers.append(f"recorded provider {recorded_provider!r} has no inventory adapter")
            names: tuple[str, ...] = ()
        else:
            names = (name,)
    else:
        names = tuple(
            name
            for name, binding in (("dstack", local_config.dstack), ("local-docker", local_config.local))
            if binding is not None
        )
    provider_checked = False
    for name in names:
        try:
            provider = _inventory_provider(layout, local_config, name)
            active.extend(provider.active_executions_for_run(run_id))
        except Exception as error:
            blockers.append(f"{name} inventory is unavailable ({type(error).__name__}: {_short(error)})")
            continue
        checked.append(str(provider.inventory_scope))
        provider_checked = True
    if not provider_checked and not blockers:
        blockers.append("no execution provider inventory is configured")
    admission = _orphan_admission(layout, local_config, entry) if entry is not None else None
    return OrphanProviderInventory(tuple(checked), tuple(active), tuple(blockers)), admission


def _orphan_admission(layout: Any, local_config: Any, entry: AdmissionEntry) -> OrphanAdmissionEntry:
    """Describe one ledger entry: owning control store and named execution now."""

    control_store: Path | None = None
    status = "unknown"
    try:
        if entry.control_locator is not None:
            control_store = entry.control_locator.control_store
        elif entry.control_store_uri is not None:
            parsed = urlparse(entry.control_store_uri)
            if parsed.scheme == "file" and parsed.path:
                control_store = Path(unquote(parsed.path)).resolve()
        if control_store is not None:
            if not control_store.exists():
                status = "absent"
            else:
                receipts = ExecutionSubmissionStore(control_store)
                status = "has-receipt" if receipts.run_root(entry.run_id).exists() else "no-receipt"
    except Exception:
        status = "unknown"
    provider_id = entry.plan.native_plan_id
    provider_state: str | None = None
    native_state: str | None = None
    query_error: str | None = None
    if entry.state not in _SETTLED_ADMISSION_STATES:
        if not provider_id:
            query_error = "the admission entry records no provider execution id"
        else:
            try:
                provider = _entry_provider(layout, local_config, entry)
                record = provider.status(
                    ExecutionHandle(entry.plan.provider, provider_id, entry.plan.request.idempotency_key)
                )
                provider_state = str(record.state)
                native_state = str(record.native_state) if record.native_state is not None else None
            except Exception as error:
                query_error = f"{type(error).__name__}: {_short(error)}"
    return OrphanAdmissionEntry(
        state=entry.state,
        admission_key=entry.admission_key or "",
        provider=entry.plan.provider,
        provider_id=provider_id,
        control_store=str(control_store) if control_store is not None else None,
        control_store_status=status,
        provider_state=provider_state,
        provider_native_state=native_state,
        provider_query_error=query_error,
        job_image=entry.plan.request.image.value,
    )


def _entry_provider(layout: Any, local_config: Any, entry: AdmissionEntry) -> Any:
    """Build the provider binding the entry recorded, else the configured one."""

    source = entry.provider_source
    if entry.plan.provider == "dstack" and source is not None and source.adapter_python is not None:
        module = importlib.import_module("posttrain_execution_dstack")
        return module.DstackExecutionProvider.from_sdk_environment(
            project=source.endpoint_scope,
            python=source.adapter_python,
            environment_file=source.credential_file,
        )
    if local_config is None:
        raise RuntimeError("execution provider configuration is unavailable")
    name = _PROVIDER_ALIASES.get(entry.plan.provider)
    if name is None:
        raise RuntimeError(f"admission provider {entry.plan.provider!r} has no adapter")
    return _inventory_provider(layout, local_config, name)


def _inventory_provider(layout: Any, local_config: Any, name: str) -> Any:
    if name == "dstack":
        binding = local_config.dstack
        if binding is None:
            raise RuntimeError("dstack provider binding is not configured")
        module = importlib.import_module("posttrain_execution_dstack")
        return module.DstackExecutionProvider.from_sdk_environment(
            project=binding.project,
            python=binding.python,
            environment_file=binding.environment_file,
        )
    if name == "local-docker":
        module = importlib.import_module("posttrain_execution_local")
        return module.LocalDockerExecutionProvider(state_root=layout.state)
    raise RuntimeError(f"unsupported provider {name!r}")


def _short(error: Exception) -> str:
    return " ".join(str(error).split())[:200]


def _revalidate_orphan(layout: Any, store: PurgeStore, plan: PurgePlan) -> None:
    """Re-prove every orphan check immediately before tracking deletion."""

    basis = plan.basis
    if basis is None or basis.get("kind") != ORPHAN_TRACKING_RUN_BASIS:
        return
    run_id = plan.root_run_id
    if run_id is None:
        raise RuntimeError("orphan purge plan has no root run")
    settled = {
        str(event.get("action_id"))
        for event in store.journal(plan.purge_id)
        if event.get("status") in {"completed", "skipped"}
    }
    # Once the tracking run is gone the orphan cannot be re-read; a resumed
    # apply relies on each remaining executor's own exact revalidation.
    if any(action.plane == "tracking" and action.action_id in settled for action in plan.actions):
        return
    candidates, local_blocker = _orphan_local_control_check(layout, run_id)
    orphan = _collect_orphan(layout, run_id) if local_blocker is None else local_blocker
    if isinstance(orphan, str):
        raise RuntimeError(f"orphan purge revalidation failed: {orphan}")
    if orphan.tracking_provider_run_id != basis.get("tracking_provider_run_id"):
        raise RuntimeError("orphan purge revalidation failed: the tracking run identity changed after preview")
    stale_seconds = basis.get("stale_after_seconds")
    if not isinstance(stale_seconds, int) or stale_seconds <= 0:
        raise RuntimeError("orphan purge plan has an invalid stale threshold")
    registry_owners, registry_blockers = _registry_image_inventory(layout, candidates)
    blockers = orphan_run_blockers(
        orphan,
        registry_image_owners=registry_owners,
        registry_inventory_blockers=registry_blockers,
        now=datetime.now(UTC),
        stale_after=timedelta(seconds=stale_seconds),
    )
    if blockers:
        raise RuntimeError("orphan purge revalidation failed: " + "; ".join(blockers))


def save_project_preview(layout: Any, *, reason: PurgeReason) -> PurgePlan:
    candidates = candidate_catalog(layout)
    registry_owners, registry_blockers = _registry_image_inventory(layout, candidates)
    plan = build_project_purge_plan(
        _Catalog(
            candidates,
            registry_owners=registry_owners,
            registry_blockers=registry_blockers,
        ),
        project_id=layout.project_id,
        reason=reason,
    )
    return plan_store(layout).save_plan(plan)


def render_plan(plan: PurgePlan, *, tombstone: PurgeTombstone | None = None) -> str:
    counts = {
        "provider": len(plan.provider_actions),
        "registry": len(plan.registry_actions),
        "tracking": len(plan.tracking_actions),
        "local": len(plan.local_actions),
    }
    lines = [
        "Purge preview — no changes made",
        f"Target: {plan.root_run_id or plan.project_id} (project: {plan.project_id})",
        f"Reason: {plan.reason.category if plan.reason is not None else 'legacy plan (no reason recorded)'}",
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
    lines[1:1] = _orphan_lines(plan)
    if plan.blockers:
        lines.append("Next: resolve blockers and create a new preview")
    else:
        lines.append(f"Next: posttrain purge apply {plan.purge_id} --expect-digest {plan.digest} --yes")
    if tombstone is not None:
        lines.extend(
            (
                f"Tombstone: {tombstone.status}",
                "Plane outcomes: "
                + ", ".join(f"{plane}={outcome}" for plane, outcome in tombstone.plane_outcomes.items()),
            )
        )
    return "\n".join(lines)


def _orphan_lines(plan: PurgePlan) -> list[str]:
    basis = plan.basis
    if basis is None or basis.get("kind") != ORPHAN_TRACKING_RUN_BASIS:
        return []

    def joined(key: str) -> str:
        value = basis.get(key)
        return ", ".join(str(item) for item in value) if isinstance(value, list) and value else "none"

    stale_seconds = basis.get("stale_after_seconds")
    threshold = f"{stale_seconds / 3600:g} h" if isinstance(stale_seconds, int) else "unknown"
    admission_state = basis.get("admission_state")
    if admission_state is None:
        admission = "admission entry: none"
    else:
        admission = (
            f"admission entry: {admission_state}, control store {basis.get('admission_control_store') or 'unknown'} "
            f"({basis.get('admission_control_store_status')}), provider execution "
            f"{basis.get('admission_provider_execution')}; "
            + ("settle to completed" if basis.get("admission_settle") else "not settled by this plan")
        )
    return [
        "Kind: orphan purge (no local submission receipt; deletes the tracking run, plus a proven-exclusive "
        "job image and abandoned admission entry; provider records and workspaces are not cleaned)",
        f"  (a) provider: recorded provider {basis.get('recorded_provider') or 'none'}; "
        f"inventories checked: {joined('provider_inventories')}; "
        f"active executions: {joined('active_provider_executions')}; {admission}",
        f"  (b) registry: attributed images: {joined('registry_attributed_images')}; "
        f"delete: {joined('registry_deletions')}; retain (shared): {joined('registry_retained_shared')}; "
        f"inventory {'complete' if basis.get('registry_inventory_complete') else 'incomplete'}",
        f"  (c) lineage: Trackio run {basis.get('tracking_provider_run_id')}; "
        f"tracked artifacts: {basis.get('tracked_artifacts')}; surviving consumers: {joined('surviving_consumers')}",
        f"  (d) state: tracking status {basis.get('tracking_status')}; "
        f"last activity {basis.get('last_activity_at') or 'unknown'}; stale threshold {threshold}",
    ]


def apply_saved_plan(
    state: CliState,
    purge_id: str,
    *,
    expected_digest: str | None,
    assume_yes: bool,
) -> Any:
    layout = state.layout()
    store = saved_plan_store(layout, purge_id)
    plan = store.load_plan(purge_id)
    if plan.reason is None:
        raise ValueError("legacy purge plans cannot be applied; create a new preview with --reason")
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

    _revalidate_registry_ownership(layout, plan)
    _revalidate_orphan(layout, store, plan)
    executors = _apply_executors(layout, plan)
    return apply_purge_plan(
        store,
        purge_id,
        executors,
    )


def _revalidate_registry_ownership(layout: Any, plan: PurgePlan) -> None:
    """Fail closed if a delayed plan's image acquired a new run owner."""

    if not plan.registry_actions:
        return
    owners, blockers = _registry_image_inventory(layout, {})
    if blockers:
        raise RuntimeError("registry ownership revalidation failed: " + "; ".join(blockers))
    selected = set(plan.run_ids)
    for action in plan.registry_actions:
        reference = str(action.target["reference"])
        external = tuple(run_id for run_id in owners.get(reference, ()) if run_id not in selected)
        if external:
            raise RuntimeError(
                f"registry image {reference!r} acquired unselected owner(s) after preview: "
                + ", ".join(repr(run_id) for run_id in external)
            )


class _LocalPlaneExecutor:
    """Dispatch local-plane actions by kind to the exact executor for each."""

    def __init__(self, remove: PurgeActionExecutor, settle: PurgeActionExecutor | None) -> None:
        self._remove = remove
        self._settle = settle

    def _executor(self, action: PurgeAction) -> PurgeActionExecutor:
        if action.kind == SETTLE_ADMISSION_KIND:
            if self._settle is None:
                raise ContractError("admission settle executor is unavailable")
            return self._settle
        return self._remove

    def revalidate(self, action: PurgeAction) -> None:
        self._executor(action).revalidate(action)

    def apply(self, action: PurgeAction) -> None:
        self._executor(action).apply(action)


def _apply_executors(layout: Any, plan: PurgePlan) -> dict[PurgePlane, PurgeActionExecutor]:
    services = {
        str(action.target["run_id"]): execution_service_for_run(layout, str(action.target["run_id"]))
        for action in plan.provider_actions
    }
    local_roots = [layout.state, cache_path(layout, "runs")]
    try:
        local_binding = load_local_execution_config(
            layout,
            verify_published_locks=False,
            resolve_registry=False,
        ).local
    except Exception:
        local_binding = None
    if local_binding is not None and local_binding.storage is not None:
        local_roots.append(local_binding.storage.run_root)
    executors: dict[PurgePlane, PurgeActionExecutor] = {
        "provider": ExecutionProviderPurgeExecutor(services),
        # Submission receipts live in the project state root, while the
        # local provider's run workspaces live in the configured machine
        # storage root. Both are exact, framework-owned state roots.
        "local": _LocalPlaneExecutor(
            LocalStatePurgeExecutor(tuple(local_roots)),
            (
                AdmissionSettlePurgeExecutor(execution_admission_service(layout))
                if any(action.kind == SETTLE_ADMISSION_KIND for action in plan.local_actions)
                else None
            ),
        ),
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
                ca_bundle=_machine_trust_bundle(),
            )
            executors["tracking"] = module.TrackioPurgeActionExecutor(admin)
        except Exception as error:
            raise RuntimeError(f"Trackio purge adapter is unavailable ({type(error).__name__})") from error
    return executors


def _machine_trust_bundle() -> Path | None:
    """Use the operator-owned trust root for private tracking ingress.

    Tracking destinations and tokens are project-scoped runtime choices; the
    CA anchor is machine infrastructure policy and must never be supplied from
    a project environment file or weakened with an insecure TLS override.
    """

    machine = load_machine_config()
    return machine.local.trust_bundle if machine is not None else None


def _registry_image_inventory(
    layout: Any,
    candidates: dict[str, PurgeRunCandidate],
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    """Inventory actual-job image owners across the machine control boundary.

    OCI manifest deletion is registry-global, while an opened project only
    owns one submission store. The machine admission ledger is the durable
    cross-project index for every admitted run and retains immutable job-image
    references even for legacy entries without a project locator. Registered
    project stores supplement that ledger for submissions created before
    machine admission. Any unreadable registered source blocks registry
    deletion instead of silently narrowing the ownership scope.
    """

    owners: dict[str, set[str]] = {}
    blockers: list[str] = []

    def add(run_id: str, value: str, *, source: str) -> None:
        try:
            reference = RegistryManifestRef.parse(value)
        except Exception as error:
            blockers.append(
                f"registry ownership source {source!r} has invalid image for run {run_id!r} ({type(error).__name__})"
            )
            return
        owners.setdefault(reference.value, set()).add(run_id)

    for candidate in candidates.values():
        if candidate.image is not None and "registry" not in candidate.completed_planes:
            owners.setdefault(candidate.image.value, set()).add(candidate.run_id)

    try:
        entries = execution_admission_service(layout).list()
    except Exception as error:
        entries = ()
        blockers.append(f"machine admission image inventory is unavailable ({type(error).__name__})")
    project_roots: set[Path] = {layout.root.resolve()}
    try:
        machine = load_machine_config()
    except Exception as error:
        machine = None
        blockers.append(f"registered project image inventory is unavailable ({type(error).__name__})")
    if machine is not None:
        project_roots.update(path.resolve() for path in machine.projects)

    project_layouts: list[Any] = []
    for root in sorted(project_roots):
        try:
            owner = load_project_layout(root)
        except Exception as error:
            blockers.append(f"registered project image inventory {str(root)!r} is unavailable ({type(error).__name__})")
            continue
        project_layouts.append(owner)

    retired_runs: set[str] = set()
    retired_runs.update(_completed_purge_run_ids(resolve_admission_state_root()))
    for owner in project_layouts:
        retired_runs.update(_completed_purge_run_ids(owner.state))

    for owner in project_layouts:
        try:
            submissions = ExecutionSubmissionStore(owner.state).list_submissions()
        except Exception as error:
            blockers.append(
                f"project {owner.project_id!r} submission image inventory is unavailable ({type(error).__name__})"
            )
            continue
        for submission in submissions:
            if submission.run_id in retired_runs:
                continue
            add(
                submission.run_id,
                submission.job_image,
                source=f"project {owner.project_id}",
            )

    for entry in entries:
        if entry.run_id not in retired_runs:
            add(entry.run_id, entry.plan.request.image.value, source="machine admission")

    archive_root = resolve_admission_state_root() / "admission" / "terminal"
    if archive_root.exists() and not archive_root.is_dir():
        blockers.append("machine admission terminal image inventory is not a directory")
    elif archive_root.is_dir():
        for path in sorted(archive_root.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                run_id = payload["run_id"]
                image = payload["job_image"]
                if not isinstance(run_id, str) or not isinstance(image, str):
                    raise ValueError("terminal admission image receipt fields are invalid")
            except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
                blockers.append(
                    f"machine admission terminal image receipt {path.name!r} is invalid ({type(error).__name__})"
                )
                continue
            if run_id not in retired_runs:
                add(run_id, image, source="machine admission terminal archive")

    return (
        {reference: tuple(sorted(run_ids)) for reference, run_ids in sorted(owners.items())},
        tuple(dict.fromkeys(blockers)),
    )


def _completed_purge_run_ids(state_root: Path) -> set[str]:
    """Return run owners retired by complete, unblocked purge receipts."""

    store = PurgeStore(state_root)
    if not store.root.is_dir():
        return set()
    retired: set[str] = set()
    for directory in store.root.iterdir():
        if not directory.is_dir() or not (directory / "receipt.json").is_file():
            continue
        try:
            plan = store.load_plan(directory.name)
            receipt = store.load_receipt(directory.name)
        except Exception:
            continue
        if not plan.blockers and receipt.failed_action is None:
            retired.update(plan.run_ids)
    return retired


class _Catalog:
    def __init__(
        self,
        values: dict[str, PurgeRunCandidate],
        *,
        registry_owners: dict[str, tuple[str, ...]],
        registry_blockers: tuple[str, ...],
    ) -> None:
        self._values = values
        self._registry_owners = registry_owners
        self._registry_blockers = registry_blockers

    def get(self, run_id: str) -> PurgeRunCandidate | None:
        return self._values.get(run_id)

    def list(self) -> tuple[PurgeRunCandidate, ...]:
        return tuple(self._values.values())

    def registry_image_owners(self) -> dict[str, tuple[str, ...]]:
        return dict(self._registry_owners)

    def registry_inventory_blockers(self) -> tuple[str, ...]:
        return self._registry_blockers


__all__ = [
    "apply_saved_plan",
    "candidate_catalog",
    "load_saved_plan",
    "load_saved_tombstone",
    "plan_store",
    "render_plan",
    "save_project_preview",
    "save_run_preview",
]
