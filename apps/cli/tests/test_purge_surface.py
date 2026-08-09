from __future__ import annotations

from types import SimpleNamespace

import pytest
from posttrain.execution import PurgeAction, PurgePlan, PurgeStore
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
