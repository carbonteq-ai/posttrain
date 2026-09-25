from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from posttrain.common import ContractError
from posttrain.execution import (
    ORPHAN_TRACKING_RUN_BASIS,
    OrphanAdmissionEntry,
    OrphanProviderInventory,
    OrphanTrackingRun,
    PurgeReason,
    PurgeStore,
    PurgeTombstone,
    apply_purge_plan,
    build_orphan_run_purge_plan,
)

_REASON = PurgeReason(category="abandoned-run")
_NOW = datetime(2026, 9, 23, 12, tzinfo=UTC)
_IMAGE = "registry.lan/carbonteq/posttrain-job@sha256:" + "a" * 64


def _orphan(**values: object) -> OrphanTrackingRun:
    orphan = OrphanTrackingRun(
        run_id="orphan-run",
        project_id="fixture",
        evidence_provider="trackio",
        evidence_project="fixture",
        tracking_provider_run_id="trackio-orphan",
        tracking_status="failed",
        last_activity_at=_NOW - timedelta(days=5),
        provider=OrphanProviderInventory(checked=("machine admission ledger", "dstack project 'main'")),
        recorded_provider="dstack",
        tracked_artifacts=2,
    )
    return replace(orphan, **values)  # type: ignore[arg-type]


def _plan(orphan: OrphanTrackingRun, *, owners: dict[str, tuple[str, ...]] | None = None, blockers=()):
    return build_orphan_run_purge_plan(
        orphan,
        reason=_REASON,
        registry_image_owners=owners or {},
        registry_inventory_blockers=tuple(blockers),
        now=_NOW,
    )


def test_terminal_orphan_plans_only_the_tracking_deletion() -> None:
    plan = _plan(_orphan())

    assert plan.blockers == ()
    assert plan.run_ids == ("orphan-run",)
    assert plan.provider_actions == plan.registry_actions == plan.local_actions == ()
    assert [(action.plane, action.kind) for action in plan.tracking_actions] == [("tracking", "tracking.delete_run")]
    assert dict(plan.tracking_actions[0].target) == {
        "provider": "trackio",
        "project": "fixture",
        "provider_run_id": "trackio-orphan",
    }
    assert plan.basis is not None
    assert plan.basis["kind"] == ORPHAN_TRACKING_RUN_BASIS
    assert plan.basis["local_receipt"] == "absent"
    assert plan.basis["tracking_status"] == "failed"
    assert any("orphan purge: run" in warning for warning in plan.warnings)


def test_stale_running_orphan_is_allowed() -> None:
    plan = _plan(_orphan(tracking_status="running", last_activity_at=_NOW - timedelta(hours=25)))

    assert plan.blockers == ()
    assert any("recorded as running but stale" in warning for warning in plan.warnings)


def test_recently_active_running_orphan_blocks() -> None:
    plan = _plan(_orphan(tracking_status="running", last_activity_at=_NOW - timedelta(hours=2)))

    assert any("recorded as running and was active" in blocker for blocker in plan.blockers)


def test_running_orphan_without_activity_timestamp_blocks() -> None:
    plan = _plan(_orphan(tracking_status="running", last_activity_at=None))

    assert any("no activity timestamp" in blocker for blocker in plan.blockers)


def test_custom_stale_threshold_is_bound_into_the_plan() -> None:
    orphan = _orphan(tracking_status="running", last_activity_at=_NOW - timedelta(hours=30))
    plan = build_orphan_run_purge_plan(
        orphan,
        reason=_REASON,
        registry_image_owners={},
        registry_inventory_blockers=(),
        now=_NOW,
        stale_after=timedelta(hours=48),
    )

    assert plan.blockers
    assert plan.basis is not None and plan.basis["stale_after_seconds"] == 48 * 3600


def test_active_provider_execution_blocks() -> None:
    inventory = OrphanProviderInventory(
        checked=("dstack project 'main'",),
        active=("dstack:pt-0123 (running)",),
    )
    plan = _plan(_orphan(provider=inventory))

    assert any("dstack:pt-0123 (running)" in blocker for blocker in plan.blockers)


def test_unavailable_or_missing_provider_inventory_blocks() -> None:
    failed = _plan(_orphan(provider=OrphanProviderInventory(blockers=("dstack inventory is unavailable",))))
    empty = _plan(_orphan(provider=OrphanProviderInventory()))

    assert any("provider check failed" in blocker for blocker in failed.blockers)
    assert any("no queryable provider inventory" in blocker for blocker in empty.blockers)


def test_attributed_registry_image_blocks() -> None:
    plan = _plan(_orphan(), owners={_IMAGE: ("orphan-run",), "other@sha256:" + "b" * 64: ("other-run",)})

    assert any(f"attributed registry image {_IMAGE!r}" in blocker for blocker in plan.blockers)
    assert plan.basis is not None and plan.basis["registry_attributed_images"] == [_IMAGE]


def test_incomplete_registry_inventory_blocks() -> None:
    plan = _plan(_orphan(), blockers=("machine admission image inventory is unavailable (OSError)",))

    assert any("registry check failed" in blocker for blocker in plan.blockers)
    assert plan.basis is not None and plan.basis["registry_inventory_complete"] is False


def test_surviving_consumer_blocks() -> None:
    plan = _plan(_orphan(consumers=("trackio-consumer",)))

    assert any("consumed by surviving run 'trackio-consumer'" in blocker for blocker in plan.blockers)


def test_incomplete_lineage_blocks() -> None:
    plan = _plan(_orphan(lineage_complete=False, lineage_blockers=("tracking lineage preview failed (OSError)",)))

    assert "tracking lineage preview failed (OSError)" in plan.blockers
    assert any("incomplete tracking lineage" in blocker for blocker in plan.blockers)


def test_non_terminal_unknown_status_blocks() -> None:
    plan = _plan(_orphan(tracking_status="queued"))

    assert any("is not terminal" in blocker for blocker in plan.blockers)


class _TrackingExecutor:
    def __init__(self) -> None:
        self.applied: list[str] = []

    def revalidate(self, action) -> None:
        del action

    def apply(self, action) -> None:
        self.applied.append(action.action_id)


def test_orphan_basis_is_digest_bound_and_recorded_in_the_tombstone(tmp_path) -> None:
    store = PurgeStore(tmp_path.resolve())
    plan = store.save_plan(_plan(_orphan()))
    loaded = store.load_plan(plan.purge_id)

    assert loaded.basis == plan.basis
    assert loaded.computed_digest() == plan.digest

    executor = _TrackingExecutor()
    apply_purge_plan(store, plan.purge_id, {"tracking": executor})

    tombstone = store.load_tombstone(plan.purge_id)
    assert executor.applied == ["tracking:orphan-run"]
    assert tombstone.status == "purged"
    assert dict(tombstone.plane_outcomes) == {
        "provider": "not-applicable",
        "registry": "not-applicable",
        "tracking": "completed",
        "local": "not-applicable",
    }
    assert tombstone.basis is not None
    assert tombstone.basis["kind"] == ORPHAN_TRACKING_RUN_BASIS
    assert tombstone.basis["tracking_provider_run_id"] == "trackio-orphan"


def test_basis_changes_the_plan_digest() -> None:
    first = _plan(_orphan())
    second = _plan(_orphan(last_activity_at=_NOW - timedelta(days=6)))

    assert first.digest != second.digest


def test_tombstone_rejects_basis_that_differs_from_its_plan(tmp_path) -> None:
    store = PurgeStore(tmp_path.resolve())
    plan = store.save_plan(_plan(_orphan()))
    tombstone = PurgeTombstone.from_plan(plan, ())

    with pytest.raises(ContractError, match="does not match its plan"):
        store.save_tombstone(replace(tombstone, basis={**dict(tombstone.basis or {}), "tracking_status": "running"}))


def test_basis_rejects_nested_values() -> None:
    with pytest.raises(ContractError, match="scalar or a list of strings"):
        PurgeTombstone(
            purge_id="purge-" + "0" * 16,
            plan_digest="sha256:" + "0" * 64,
            mode="run",
            project_id="fixture",
            run_ids=("orphan-run",),
            reason=_REASON,
            status="applying",
            plane_outcomes={
                "provider": "not-applicable",
                "registry": "not-applicable",
                "tracking": "pending",
                "local": "not-applicable",
            },
            updated_at=_NOW,
            basis={"kind": ORPHAN_TRACKING_RUN_BASIS, "nested": {"secret": "value"}},
        )


def _admission(**values: object) -> OrphanAdmissionEntry:
    entry = OrphanAdmissionEntry(
        state="terminal_pending_evidence",
        admission_key="run:orphan-run",
        provider="dstack",
        provider_id="pt-48b3564ae80357d610c47259",
        control_store="/tmp/removed-worktree/apps/lab/.posttrain/state",
        control_store_status="absent",
        provider_state="cancelled",
        provider_native_state="terminated",
        job_image=_IMAGE,
    )
    return replace(entry, **values)  # type: ignore[arg-type]


def test_abandoned_admission_entry_is_settled_and_its_exclusive_image_deleted() -> None:
    plan = _plan(_orphan(admission=_admission()), owners={_IMAGE: ("orphan-run",)})

    assert plan.blockers == ()
    assert plan.provider_actions == ()
    assert [(action.action_id, action.kind) for action in plan.registry_actions] == [
        ("registry:orphan-run", "registry.delete_manifest")
    ]
    assert dict(plan.registry_actions[0].target) == {"reference": _IMAGE, "run_id": "orphan-run"}
    assert plan.tracking_actions[0].depends_on == ("registry:orphan-run",)
    assert [(action.action_id, action.kind, action.depends_on) for action in plan.local_actions] == [
        ("local:orphan-run:admission", "local.settle_admission", ("tracking:orphan-run",))
    ]
    target = dict(plan.local_actions[0].target)
    assert target["admission_key"] == "run:orphan-run"
    assert target["provider_id"] == "pt-48b3564ae80357d610c47259"
    assert plan.basis is not None
    assert plan.basis["admission_settle"] is True
    assert plan.basis["registry_deletions"] == [_IMAGE]
    assert plan.basis["admission_control_store_status"] == "absent"


def test_admission_entry_whose_control_store_exists_without_receipt_is_settled() -> None:
    plan = _plan(_orphan(admission=_admission(control_store_status="no-receipt")), owners={_IMAGE: ("orphan-run",)})

    assert plan.blockers == ()
    assert len(plan.local_actions) == 1


def test_admission_entry_whose_project_still_holds_the_receipt_blocks() -> None:
    plan = _plan(_orphan(admission=_admission(control_store_status="has-receipt")), owners={_IMAGE: ("orphan-run",)})

    assert any("holds its submission receipt" in blocker for blocker in plan.blockers)
    assert plan.local_actions == ()


def test_admission_entry_without_inspectable_owner_blocks() -> None:
    plan = _plan(_orphan(admission=_admission(control_store=None, control_store_status="unknown")))

    assert any("no inspectable owning control store" in blocker for blocker in plan.blockers)


def test_admission_entry_whose_provider_execution_is_still_active_blocks() -> None:
    plan = _plan(
        _orphan(admission=_admission(provider_state="running", provider_native_state="running")),
        owners={_IMAGE: ("orphan-run",)},
    )

    assert any("pt-48b3564ae80357d610c47259 which is running" in blocker for blocker in plan.blockers)
    assert plan.local_actions == ()


def test_admission_entry_whose_provider_cannot_be_queried_blocks() -> None:
    plan = _plan(_orphan(admission=_admission(provider_state=None, provider_query_error="RuntimeError: offline")))

    assert any("could not be queried (RuntimeError: offline)" in blocker for blocker in plan.blockers)


def test_absent_provider_execution_counts_as_terminal() -> None:
    plan = _plan(
        _orphan(admission=_admission(provider_state="lost", provider_native_state="lost")),
        owners={_IMAGE: ("orphan-run",)},
    )

    assert plan.blockers == ()


def test_admission_entry_in_a_non_terminal_state_blocks() -> None:
    plan = _plan(_orphan(admission=_admission(state="submitted")), owners={_IMAGE: ("orphan-run",)})

    assert any("only terminal_pending_evidence can be settled" in blocker for blocker in plan.blockers)


def test_shared_image_is_retained_with_a_warning_and_the_entry_still_settles() -> None:
    plan = _plan(_orphan(admission=_admission()), owners={_IMAGE: ("orphan-run", "surviving-run")})

    assert plan.blockers == ()
    assert plan.registry_actions == ()
    assert plan.tracking_actions[0].depends_on == ()
    assert [action.kind for action in plan.local_actions] == ["local.settle_admission"]
    assert f"job image {_IMAGE!r} retained; referenced by unselected run(s): 'surviving-run'" in plan.warnings
    assert plan.basis is not None
    assert plan.basis["registry_deletions"] == []
    assert plan.basis["registry_retained_shared"] == [_IMAGE]


def test_shared_image_not_recorded_by_the_admission_entry_still_blocks() -> None:
    other = "registry.lan/carbonteq/posttrain-job@sha256:" + "f" * 64
    plan = _plan(_orphan(admission=_admission()), owners={other: ("orphan-run", "surviving-run")})

    assert any("owner record other than its admission entry" in blocker for blocker in plan.blockers)
    assert plan.registry_actions == ()


def test_image_not_recorded_by_the_admission_entry_blocks() -> None:
    other = "registry.lan/carbonteq/posttrain-job@sha256:" + "d" * 64
    plan = _plan(_orphan(admission=_admission()), owners={other: ("orphan-run",)})

    assert any("owner record other than its admission entry" in blocker for blocker in plan.blockers)


def test_settled_admission_entry_allows_its_exclusive_image_without_a_settle_action() -> None:
    plan = _plan(
        _orphan(admission=_admission(state="completed", provider_state=None)),
        owners={_IMAGE: ("orphan-run",)},
    )

    assert plan.blockers == ()
    assert [action.kind for action in plan.registry_actions] == ["registry.delete_manifest"]
    assert plan.local_actions == ()
