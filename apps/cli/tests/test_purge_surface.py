from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

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
