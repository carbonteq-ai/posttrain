from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from posttrain.execution import (
    PurgeAction,
    PurgePlan,
    PurgeReason,
    PurgeRunCandidate,
    PurgeStore,
    PurgeTombstone,
    RegistryManifestRef,
)
from posttrain_cli import purge_surface


def _plan(reference: str) -> PurgePlan:
    return PurgePlan.build(
        mode="run",
        project_id="fixture",
        run_ids=("selected-run",),
        root_run_id="selected-run",
        registry_actions=(
            PurgeAction(
                action_id="registry:selected-run",
                plane="registry",
                kind="registry.delete_manifest",
                target={"reference": reference, "run_id": "selected-run"},
            ),
        ),
        reason=PurgeReason(category="disposable-fixture"),
    )


def test_provider_terminal_cleanup_marks_missing_tracking_plane_complete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    state = (tmp_path / "state").resolve()
    run_id = "startup-failed-before-tracking"
    run_root = state / "executions" / run_id
    run_root.mkdir(parents=True)
    (run_root / "cleanup.json").write_text(
        json.dumps({"evidence_state": "provider-terminal"}),
        encoding="utf-8",
    )
    submission = SimpleNamespace(
        run_id=run_id,
        evidence_source=SimpleNamespace(provider="trackio", project="fixture", source_id="trackio-fixture"),
        provider="dstack",
        provider_id="provider-1",
        job_image="not-an-image",
        run_workspace=None,
        evidence_retention="standard",
    )

    class Store:
        def list_submissions(self):
            return (submission,)

        def cleaned_result(self, _run_id):
            return SimpleNamespace(record=SimpleNamespace(state="failed"))

        def run_root(self, _run_id):
            return run_root

    monkeypatch.setattr(purge_surface, "ExecutionSubmissionStore", lambda _state: Store())
    monkeypatch.setattr(purge_surface, "_plan_stores", lambda _layout: ())
    monkeypatch.setattr(purge_surface, "_populate_trackio_lineage", lambda *_args, **_kwargs: None)

    candidate = purge_surface.candidate_catalog(SimpleNamespace(state=state, project_id="fixture"))[run_id]

    assert candidate.reconciled is True
    assert candidate.evidence_provider == "trackio"
    assert candidate.completed_planes == ("tracking",)


def test_cancelled_admission_without_submission_can_be_purged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    state = (tmp_path / "state").resolve()
    image = RegistryManifestRef("registry.lan/posttrain-job", "sha256:" + "a" * 64)
    request = SimpleNamespace(
        run_spec=SimpleNamespace(project_id="fixture"),
        image=image,
    )
    entry = SimpleNamespace(
        run_id="cancelled-before-submit",
        state="cancelled",
        plan=SimpleNamespace(provider="local-docker", native_plan_id=None, request=request),
    )

    class Store:
        def list_submissions(self):
            return ()

    monkeypatch.setattr(purge_surface, "ExecutionSubmissionStore", lambda _state: Store())
    monkeypatch.setattr(purge_surface, "_plan_stores", lambda _layout: ())
    monkeypatch.setattr(
        purge_surface,
        "execution_admission_service",
        lambda _layout: SimpleNamespace(list=lambda: (entry,)),
    )
    monkeypatch.setattr(
        purge_surface,
        "_populate_trackio_lineage",
        lambda _layout, candidates, **_kwargs: None,
    )

    candidate = purge_surface.candidate_catalog(SimpleNamespace(state=state, project_id="fixture"))[entry.run_id]

    assert candidate.state == "cancelled"
    assert candidate.reconciled is True
    assert candidate.image == image
    assert candidate.completed_planes == ("provider", "tracking")


def test_apply_time_registry_revalidation_allows_only_selected_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = "registry.lan/posttrain-job@sha256:" + "a" * 64
    monkeypatch.setattr(
        purge_surface,
        "_registry_image_inventory",
        lambda _layout, _candidates: ({reference: ("selected-run",)}, ()),
    )

    purge_surface._revalidate_registry_ownership(SimpleNamespace(), _plan(reference))


def test_apply_time_registry_revalidation_rejects_owner_added_after_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = "registry.lan/posttrain-job@sha256:" + "b" * 64
    monkeypatch.setattr(
        purge_surface,
        "_registry_image_inventory",
        lambda _layout, _candidates: (
            {reference: ("selected-run", "new-production-run")},
            (),
        ),
    )

    with pytest.raises(RuntimeError, match="acquired unselected owner"):
        purge_surface._revalidate_registry_ownership(SimpleNamespace(), _plan(reference))


def test_apply_time_registry_revalidation_fails_closed_on_incomplete_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = "registry.lan/posttrain-job@sha256:" + "c" * 64
    monkeypatch.setattr(
        purge_surface,
        "_registry_image_inventory",
        lambda _layout, _candidates: ({}, ("registered store is unreadable",)),
    )

    with pytest.raises(RuntimeError, match="registered store is unreadable"):
        purge_surface._revalidate_registry_ownership(SimpleNamespace(), _plan(reference))


def test_machine_plan_store_reads_legacy_project_plan_during_migration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    machine_root = (tmp_path / "machine").resolve()
    project_root = (tmp_path / "project-state").resolve()
    layout = SimpleNamespace(state=project_root)
    plan = _plan("registry.lan/posttrain-job@sha256:" + "d" * 64)
    PurgeStore(project_root).save_plan(plan)
    monkeypatch.setattr(purge_surface, "resolve_admission_state_root", lambda: machine_root)

    assert purge_surface.plan_store(layout).root == machine_root / "purges"
    assert purge_surface.saved_plan_store(layout, plan.purge_id).root == project_root / "purges"
    assert purge_surface.load_saved_plan(layout, plan.purge_id) == plan


def test_new_plan_is_saved_only_in_machine_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    machine_root = (tmp_path / "machine").resolve()
    project_root = (tmp_path / "project-state").resolve()
    layout = SimpleNamespace(state=project_root)
    plan = _plan("registry.lan/posttrain-job@sha256:" + "e" * 64)
    monkeypatch.setattr(purge_surface, "resolve_admission_state_root", lambda: machine_root)

    purge_surface.plan_store(layout).save_plan(plan)

    assert PurgeStore(machine_root).plan_path(plan.purge_id).is_file()
    assert not PurgeStore(project_root).plan_path(plan.purge_id).exists()


def test_render_plan_includes_only_safe_tombstone_outcomes() -> None:
    plan = _plan("registry.lan/posttrain-job@sha256:" + "f" * 64)
    assert plan.reason is not None
    tombstone = PurgeTombstone(
        purge_id=plan.purge_id,
        plan_digest=plan.digest,
        mode=plan.mode,
        project_id=plan.project_id,
        run_ids=plan.run_ids,
        reason=plan.reason,
        status="partial",
        plane_outcomes={
            "provider": "completed",
            "registry": "failed",
            "tracking": "pending",
            "local": "pending",
        },
        updated_at=datetime.now(UTC),
    )

    rendered = purge_surface.render_plan(plan, tombstone=tombstone)

    assert "Tombstone: partial" in rendered
    assert "Plane outcomes: provider=completed, registry=failed, tracking=pending, local=pending" in rendered


@pytest.mark.parametrize(
    ("provider_state", "tracking_status"),
    (("cancelled", "failed"), ("failed", "cancelled")),
)
def test_purge_accepts_identified_terminal_failure_disagreement(
    provider_state: str,
    tracking_status: str,
) -> None:
    assert purge_surface._reconciliation_allows_purge(
        {
            "state": "inconsistent",
            "provider_record": {"state": provider_state},
            "tracking_status": tracking_status,
            "tracking_provider_run_id": "trackio-1",
        }
    )


@pytest.mark.parametrize(
    "payload",
    (
        {
            "state": "pending",
            "provider_record": {"state": "cancelled"},
            "tracking_status": None,
            "tracking_provider_run_id": None,
        },
        {
            "state": "inconsistent",
            "provider_record": {"state": "succeeded"},
            "tracking_status": "failed",
            "tracking_provider_run_id": "trackio-1",
        },
        {
            "state": "inconsistent",
            "provider_record": {"state": "cancelled"},
            "tracking_status": "running",
            "tracking_provider_run_id": "trackio-1",
        },
        {
            "state": "inconsistent",
            "provider_record": {"state": "cancelled"},
            "tracking_status": "failed",
            "tracking_provider_run_id": None,
        },
    ),
)
def test_purge_rejects_unsettled_or_unidentified_disagreement(payload: dict[str, object]) -> None:
    assert not purge_surface._reconciliation_allows_purge(payload)


def test_project_preview_uses_the_real_cross_plane_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    layout = SimpleNamespace(project_id="fixture")
    candidate = PurgeRunCandidate(
        run_id="project-run",
        project_id="fixture",
        provider="local-docker",
        provider_id="container-1",
        state="succeeded",
        reconciled=True,
        evidence_provider="trackio",
        evidence_project="fixture",
        tracking_provider_run_id="trackio-1",
        image=RegistryManifestRef("registry.lan/posttrain-job", "sha256:" + "a" * 64),
        local_paths=((tmp_path / "project-run").resolve(),),
    )
    store = PurgeStore((tmp_path / "machine").resolve())
    monkeypatch.setattr(purge_surface, "candidate_catalog", lambda _layout: {candidate.run_id: candidate})
    monkeypatch.setattr(
        purge_surface,
        "_registry_image_inventory",
        lambda _layout, _candidates: ({candidate.image.value: (candidate.run_id,)}, ()),  # type: ignore[union-attr]
    )
    monkeypatch.setattr(purge_surface, "plan_store", lambda _layout: store)

    plan = purge_surface.save_project_preview(layout, reason=PurgeReason(category="decommission"))

    assert plan.blockers == ()
    assert [action.kind for action in plan.tracking_actions][-1] == "tracking.delete_project"


def test_single_run_preview_discovers_only_the_selected_root_lineage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    layout = SimpleNamespace(project_id="fixture")
    candidate = PurgeRunCandidate(
        run_id="selected-run",
        project_id="fixture",
        provider="local-docker",
        provider_id="container-1",
        state="succeeded",
        reconciled=True,
        evidence_provider="trackio",
        evidence_project="fixture",
        tracking_provider_run_id="trackio-1",
        local_paths=((tmp_path / "selected-run").resolve(),),
    )
    observed: dict[str, object] = {}
    store = PurgeStore((tmp_path / "machine").resolve())

    def catalog(_layout, *, discover_lineage_for=None, refresh_status_for=None):
        observed["discover_lineage_for"] = discover_lineage_for
        observed["refresh_status_for"] = refresh_status_for
        return {candidate.run_id: candidate}

    monkeypatch.setattr(purge_surface, "candidate_catalog", catalog)
    monkeypatch.setattr(purge_surface, "_registry_image_inventory", lambda _layout, _candidates: ({}, ()))
    monkeypatch.setattr(purge_surface, "plan_store", lambda _layout: store)

    purge_surface.save_run_preview(
        layout,
        candidate.run_id,
        cascade=False,
        reason=PurgeReason(category="disposable-fixture"),
    )

    assert observed["discover_lineage_for"] == (candidate.run_id,)
    assert observed["refresh_status_for"] == (candidate.run_id,)


# --- orphaned tracking-run purge -------------------------------------------------

_ORPHAN = "orphan-run"
_ORPHAN_IMAGE = "registry.lan/carbonteq/posttrain-job@sha256:" + "c" * 64


class _FakeProvider:
    def __init__(self, scope: str, active: tuple[str, ...] = ()) -> None:
        self.inventory_scope = scope
        self.active = active

    def active_executions_for_run(self, run_id: str) -> tuple[str, ...]:
        assert run_id == _ORPHAN
        return self.active


def _install_orphan_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    status: str = "failed",
    last_activity: datetime | None = None,
    recorded_provider: str | None = "dstack",
    consumers: tuple[str, ...] = (),
    active: tuple[str, ...] = (),
    admission: tuple[object, ...] = (),
    registry_owners: dict[str, tuple[str, ...]] | None = None,
    local_candidates: dict[str, PurgeRunCandidate] | None = None,
) -> tuple[SimpleNamespace, PurgeStore, dict[str, object]]:
    from datetime import timedelta

    state: dict[str, object] = {
        "status": status,
        "last_activity": last_activity or datetime.now(UTC) - timedelta(days=5),
        "active": active,
        "consumers": consumers,
        "admission": admission,
        "registry_owners": registry_owners or {},
        "local_candidates": local_candidates or {},
    }

    class Lookup:
        def __init__(self, project: str, *, server_url: str) -> None:
            assert project == "fixture" and server_url == "https://trackio.test"

        def find(self, run_id: str):
            if run_id != _ORPHAN:
                return None
            return SimpleNamespace(
                run_id=run_id,
                provider_run_id="trackio-orphan",
                status=state["status"],
                created_at=None,
                last_activity_at=state["last_activity"],
                recorded_provider=recorded_provider,
                recorded_job_image=_ORPHAN_IMAGE,
                evidence_retention="standard",
            )

    class Admin:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def plan_run_purge(self, *, project: str, provider_run_ids: tuple[str, ...]):
            assert project == "fixture" and provider_run_ids == ("trackio-orphan",)
            return SimpleNamespace(
                artifacts=(SimpleNamespace(consumer_run_ids=("trackio-orphan", *state["consumers"])),),  # type: ignore[misc]
                blockers=(),
            )

    tracking_module = SimpleNamespace(TrackioRunActivityLookup=Lookup, TrackioLifecycleAdmin=Admin)
    monkeypatch.setattr(
        purge_surface,
        "importlib",
        SimpleNamespace(import_module=lambda name: tracking_module),
    )
    monkeypatch.setattr(
        purge_surface,
        "project_tracking_environment",
        lambda _layout: {"POSTTRAIN_TRACKIO_SERVER_URL": "https://trackio.test", "TRACKIO_WRITE_TOKEN": "fixture"},
    )
    monkeypatch.setattr(purge_surface, "_machine_trust_bundle", lambda: None)
    monkeypatch.setattr(
        purge_surface,
        "execution_admission_service",
        lambda _layout: SimpleNamespace(list=lambda: state["admission"]),
    )
    monkeypatch.setattr(
        purge_surface,
        "load_local_execution_config",
        lambda *_args, **_kwargs: SimpleNamespace(dstack=SimpleNamespace(), local=SimpleNamespace()),
    )
    monkeypatch.setattr(
        purge_surface,
        "_inventory_provider",
        lambda _layout, _config, name: _FakeProvider(
            "dstack project 'main'" if name == "dstack" else "local Docker",
            tuple(state["active"]) if name == "dstack" else (),  # type: ignore[arg-type]
        ),
    )
    monkeypatch.setattr(
        purge_surface,
        "candidate_catalog",
        lambda _layout, **_kwargs: dict(state["local_candidates"]),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        purge_surface,
        "_registry_image_inventory",
        lambda _layout, _candidates: (dict(state["registry_owners"]), ()),  # type: ignore[arg-type]
    )
    store = PurgeStore((tmp_path / "machine").resolve())
    monkeypatch.setattr(purge_surface, "plan_store", lambda _layout: store)
    monkeypatch.setattr(purge_surface, "saved_plan_store", lambda _layout, _purge_id: store)
    layout = SimpleNamespace(project_id="fixture", tracking="trackio", state=(tmp_path / "state").resolve())
    return layout, store, state


def _orphan_preview(layout: SimpleNamespace) -> PurgePlan:
    return purge_surface.save_run_preview(
        layout,
        _ORPHAN,
        cascade=False,
        reason=PurgeReason(category="abandoned-run"),
        orphan=True,
    )


def test_orphan_terminal_run_plans_only_tracking_deletion(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    layout, _store, _state = _install_orphan_fakes(monkeypatch, tmp_path)

    plan = _orphan_preview(layout)

    assert plan.blockers == ()
    assert plan.provider_actions == plan.registry_actions == plan.local_actions == ()
    assert [action.action_id for action in plan.tracking_actions] == [f"tracking:{_ORPHAN}"]
    assert plan.basis is not None
    assert plan.basis["provider_inventories"] == ["machine admission ledger", "dstack project 'main'"]
    rendered = purge_surface.render_plan(plan)
    assert "Kind: orphan purge (no local submission receipt" in rendered
    assert "(a) provider: recorded provider dstack" in rendered
    assert "admission entry: none" in rendered
    assert "(b) registry: attributed images: none; delete: none; retain (shared): none; inventory complete" in rendered
    assert "(c) lineage: Trackio run trackio-orphan; tracked artifacts: 1; surviving consumers: none" in rendered
    assert "(d) state: tracking status failed" in rendered


def test_orphan_without_recorded_provider_queries_every_configured_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    layout, _store, _state = _install_orphan_fakes(monkeypatch, tmp_path, recorded_provider=None)

    plan = _orphan_preview(layout)

    assert plan.blockers == ()
    assert plan.basis is not None
    assert plan.basis["provider_inventories"] == [
        "machine admission ledger",
        "dstack project 'main'",
        "local Docker",
    ]


def test_orphan_stale_running_run_is_allowed(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from datetime import timedelta

    layout, _store, _state = _install_orphan_fakes(
        monkeypatch,
        tmp_path,
        status="running",
        last_activity=datetime.now(UTC) - timedelta(days=6),
    )

    plan = _orphan_preview(layout)

    assert plan.blockers == ()
    assert any("recorded as running but stale" in warning for warning in plan.warnings)


def test_orphan_recently_active_running_run_blocks(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from datetime import timedelta

    layout, _store, _state = _install_orphan_fakes(
        monkeypatch,
        tmp_path,
        status="running",
        last_activity=datetime.now(UTC) - timedelta(hours=1),
    )

    plan = _orphan_preview(layout)

    assert any("recorded as running and was active" in blocker for blocker in plan.blockers)


def test_orphan_active_provider_execution_blocks(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    layout, _store, _state = _install_orphan_fakes(monkeypatch, tmp_path, active=("dstack:pt-0123 (running)",))

    plan = _orphan_preview(layout)

    assert any("dstack:pt-0123 (running)" in blocker for blocker in plan.blockers)


def _ledger_entry(control_store: Path, *, state: str = "terminal_pending_evidence") -> SimpleNamespace:
    return SimpleNamespace(
        run_id=_ORPHAN,
        state=state,
        admission_key=f"run:{_ORPHAN}",
        control_locator=None,
        control_store_uri=control_store.as_uri(),
        provider_source=None,
        plan=SimpleNamespace(
            provider="dstack",
            native_plan_id="pt-48b3564ae80357d610c47259",
            request=SimpleNamespace(idempotency_key="posttrain-key", image=SimpleNamespace(value=_ORPHAN_IMAGE)),
        ),
    )


def _named_execution(monkeypatch: pytest.MonkeyPatch, state: str, native: str) -> list[Any]:
    handles: list[Any] = []

    class Provider:
        def status(self, handle):
            handles.append(handle)
            return SimpleNamespace(state=state, native_state=native)

    monkeypatch.setattr(purge_surface, "_entry_provider", lambda _layout, _config, _entry: Provider())
    return handles


def test_orphan_abandoned_admission_entry_is_settled_with_its_exclusive_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    gone = (tmp_path / "removed-worktree" / ".posttrain" / "state").resolve()
    layout, _store, _state = _install_orphan_fakes(
        monkeypatch,
        tmp_path,
        admission=(_ledger_entry(gone),),
        registry_owners={_ORPHAN_IMAGE: (_ORPHAN,)},
    )
    handles = _named_execution(monkeypatch, "cancelled", "terminated")

    plan = _orphan_preview(layout)

    assert plan.blockers == ()
    assert [handle.provider_id for handle in handles] == ["pt-48b3564ae80357d610c47259"]
    assert [action.kind for action in plan.registry_actions] == ["registry.delete_manifest"]
    assert [action.kind for action in plan.tracking_actions] == ["tracking.delete_run"]
    assert [action.kind for action in plan.local_actions] == ["local.settle_admission"]
    assert plan.provider_actions == ()
    rendered = purge_surface.render_plan(plan)
    assert f"control store {gone} (absent)" in rendered
    assert "dstack:pt-48b3564ae80357d610c47259 (cancelled, terminated); settle to completed" in rendered
    assert f"delete: {_ORPHAN_IMAGE}" in rendered


def test_orphan_admission_entry_whose_project_still_exists_blocks(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    owner = (tmp_path / "live-worktree" / ".posttrain" / "state").resolve()
    (owner / "executions" / _ORPHAN).mkdir(parents=True)
    layout, _store, _state = _install_orphan_fakes(
        monkeypatch,
        tmp_path,
        admission=(_ledger_entry(owner),),
        registry_owners={_ORPHAN_IMAGE: (_ORPHAN,)},
    )
    _named_execution(monkeypatch, "cancelled", "terminated")

    plan = _orphan_preview(layout)

    assert any("holds its submission receipt" in blocker for blocker in plan.blockers)
    assert plan.local_actions == ()


def test_orphan_admission_entry_whose_provider_execution_is_active_blocks(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    gone = (tmp_path / "removed-worktree" / ".posttrain" / "state").resolve()
    layout, _store, _state = _install_orphan_fakes(
        monkeypatch,
        tmp_path,
        admission=(_ledger_entry(gone),),
        registry_owners={_ORPHAN_IMAGE: (_ORPHAN,)},
    )
    _named_execution(monkeypatch, "running", "running")

    plan = _orphan_preview(layout)

    assert any("pt-48b3564ae80357d610c47259 which is running (running)" in blocker for blocker in plan.blockers)


def test_orphan_admission_image_shared_with_another_run_is_retained(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    gone = (tmp_path / "removed-worktree" / ".posttrain" / "state").resolve()
    layout, _store, _state = _install_orphan_fakes(
        monkeypatch,
        tmp_path,
        admission=(_ledger_entry(gone),),
        registry_owners={_ORPHAN_IMAGE: (_ORPHAN, "surviving-run")},
    )
    _named_execution(monkeypatch, "failed", "failed")

    plan = _orphan_preview(layout)

    assert plan.blockers == ()
    assert plan.registry_actions == ()
    assert [action.kind for action in plan.tracking_actions] == ["tracking.delete_run"]
    assert [action.kind for action in plan.local_actions] == ["local.settle_admission"]
    assert any("retained; referenced by unselected run(s): 'surviving-run'" in warning for warning in plan.warnings)
    assert f"retain (shared): {_ORPHAN_IMAGE}" in purge_surface.render_plan(plan)


def test_orphan_apply_settles_admission_after_tracking_deletion(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    gone = (tmp_path / "removed-worktree" / ".posttrain" / "state").resolve()
    layout, store, _state = _install_orphan_fakes(
        monkeypatch,
        tmp_path,
        admission=(_ledger_entry(gone),),
        registry_owners={_ORPHAN_IMAGE: (_ORPHAN,)},
    )
    _named_execution(monkeypatch, "failed", "failed")
    plan = _orphan_preview(layout)
    assert plan.blockers == ()
    applied: list[str] = []

    class Executor:
        def revalidate(self, action) -> None:
            del action

        def apply(self, action) -> None:
            applied.append(action.action_id)

    executor = Executor()
    monkeypatch.setattr(
        purge_surface,
        "_apply_executors",
        lambda _layout, _plan: {"registry": executor, "tracking": executor, "local": executor},
    )
    monkeypatch.setattr(purge_surface, "_revalidate_registry_ownership", lambda _layout, _plan: None)
    cli_state = cast(Any, SimpleNamespace(layout=lambda: layout, json_output=False))

    purge_surface.apply_saved_plan(cli_state, plan.purge_id, expected_digest=plan.digest, assume_yes=True)

    assert applied == [f"registry:{_ORPHAN}", f"tracking:{_ORPHAN}", f"local:{_ORPHAN}:admission"]
    tombstone = store.load_tombstone(plan.purge_id)
    assert tombstone.status == "purged"
    assert dict(tombstone.plane_outcomes) == {
        "provider": "not-applicable",
        "registry": "completed",
        "tracking": "completed",
        "local": "completed",
    }
    assert tombstone.basis is not None and tombstone.basis["admission_settle"] is True


def test_candidate_catalog_skips_runs_retired_by_a_completed_purge(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    machine = PurgeStore((tmp_path / "machine").resolve())
    retired = "cancelled-and-purged"
    plan = machine.save_plan(
        PurgePlan.build(
            mode="run",
            project_id="fixture",
            run_ids=(retired,),
            root_run_id=retired,
            reason=PurgeReason(category="disposable-fixture"),
        )
    )
    from posttrain.execution import apply_purge_plan

    apply_purge_plan(machine, plan.purge_id, {})
    image = RegistryManifestRef("registry.lan/posttrain-job", "sha256:" + "e" * 64)
    entry = SimpleNamespace(
        run_id=retired,
        state="cancelled",
        plan=SimpleNamespace(
            provider="local-docker",
            native_plan_id=None,
            request=SimpleNamespace(run_spec=SimpleNamespace(project_id="fixture"), image=image),
        ),
    )

    class Store:
        def list_submissions(self):
            return ()

    monkeypatch.setattr(purge_surface, "ExecutionSubmissionStore", lambda _state: Store())
    monkeypatch.setattr(purge_surface, "_plan_stores", lambda _layout: (machine,))
    monkeypatch.setattr(
        purge_surface, "execution_admission_service", lambda _layout: SimpleNamespace(list=lambda: (entry,))
    )
    monkeypatch.setattr(purge_surface, "_populate_trackio_lineage", lambda *_args, **_kwargs: None)

    candidates = purge_surface.candidate_catalog(SimpleNamespace(state=tmp_path.resolve(), project_id="fixture"))

    assert retired not in candidates


def test_orphan_attributed_registry_image_blocks(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    layout, _store, _state = _install_orphan_fakes(
        monkeypatch,
        tmp_path,
        registry_owners={_ORPHAN_IMAGE: (_ORPHAN,)},
    )

    plan = _orphan_preview(layout)

    assert any(f"attributed registry image {_ORPHAN_IMAGE!r}" in blocker for blocker in plan.blockers)


def test_orphan_surviving_consumer_blocks(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    layout, _store, _state = _install_orphan_fakes(monkeypatch, tmp_path, consumers=("trackio-consumer",))

    plan = _orphan_preview(layout)

    assert any("consumed by surviving run 'trackio-consumer'" in blocker for blocker in plan.blockers)


def test_orphan_flag_refuses_run_with_local_receipt(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    candidate = PurgeRunCandidate(
        run_id=_ORPHAN,
        project_id="fixture",
        provider="dstack",
        provider_id="pt-local",
        state="failed",
        reconciled=True,
        evidence_provider="trackio",
        evidence_project="fixture",
        tracking_provider_run_id="trackio-orphan",
    )
    layout, _store, _state = _install_orphan_fakes(monkeypatch, tmp_path, local_candidates={_ORPHAN: candidate})

    plan = _orphan_preview(layout)

    assert plan.blockers == (
        f"run {_ORPHAN!r} has local control state on this machine; omit --orphan to use the normal purge",
    )
    assert plan.actions == ()


def test_run_missing_from_tracking_blocks_orphan_preview(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    layout, _store, _state = _install_orphan_fakes(monkeypatch, tmp_path)

    plan = purge_surface.save_run_preview(
        layout,
        "not-in-tracking",
        cascade=False,
        reason=PurgeReason(category="abandoned-run"),
        orphan=True,
    )

    assert plan.blockers == ("run 'not-in-tracking' was not found in tracking project 'fixture'",)


def test_without_orphan_flag_missing_receipt_keeps_not_found_blocker(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    layout, _store, _state = _install_orphan_fakes(monkeypatch, tmp_path)

    plan = purge_surface.save_run_preview(
        layout,
        _ORPHAN,
        cascade=False,
        reason=PurgeReason(category="abandoned-run"),
    )

    assert plan.blockers == (f"run {_ORPHAN!r} was not found",)
    assert plan.basis is None


def test_orphan_apply_revalidates_before_deleting(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    layout, store, state = _install_orphan_fakes(monkeypatch, tmp_path)
    plan = _orphan_preview(layout)
    assert plan.blockers == ()
    applied: list[str] = []

    class Executor:
        def revalidate(self, action) -> None:
            del action

        def apply(self, action) -> None:
            applied.append(action.action_id)

    monkeypatch.setattr(purge_surface, "_apply_executors", lambda _layout, _plan: {"tracking": Executor()})
    cli_state = cast(Any, SimpleNamespace(layout=lambda: layout, json_output=False))

    state["active"] = ("dstack:pt-late (running)",)
    with pytest.raises(RuntimeError, match="orphan purge revalidation failed"):
        purge_surface.apply_saved_plan(cli_state, plan.purge_id, expected_digest=plan.digest, assume_yes=True)
    assert applied == []

    state["active"] = ()
    purge_surface.apply_saved_plan(cli_state, plan.purge_id, expected_digest=plan.digest, assume_yes=True)
    assert applied == [f"tracking:{_ORPHAN}"]
    tombstone = store.load_tombstone(plan.purge_id)
    assert tombstone.status == "purged"
    assert tombstone.basis is not None and tombstone.basis["kind"] == "orphan-tracking-run"


def test_orphan_rejects_cascade(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    layout, _store, _state = _install_orphan_fakes(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="cannot be combined with --cascade"):
        purge_surface.save_run_preview(
            layout,
            _ORPHAN,
            cascade=True,
            reason=PurgeReason(category="abandoned-run"),
            orphan=True,
        )
