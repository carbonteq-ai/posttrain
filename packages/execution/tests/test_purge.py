from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from posttrain.common import ContractError
from posttrain.execution import (
    PurgeAction,
    PurgeApplyError,
    PurgePlan,
    PurgeReceipt,
    PurgeStore,
    apply_purge_plan,
)


def _plan(*, created_at: datetime | None = None, blockers: tuple[str, ...] = ()) -> PurgePlan:
    provider = PurgeAction(
        action_id="provider:run-1",
        plane="provider",
        kind="provider.cleanup",
        target={"provider": "dstack", "provider_id": "native-1"},
        precondition={"state": "terminal"},
    )
    tracking = PurgeAction(
        action_id="tracking:run-1",
        plane="tracking",
        kind="tracking.delete_run",
        target={"project": "purge-fixture", "provider_run_id": "trackio-1"},
        depends_on=(provider.action_id,),
        logical_bytes=128,
    )
    return PurgePlan.build(
        mode="run",
        project_id="purge-fixture",
        run_ids=("run-1",),
        root_run_id="run-1",
        provider_actions=(provider,),
        tracking_actions=(tracking,),
        blockers=blockers,
        created_at=created_at,
    )


def test_plan_is_content_addressed_and_excludes_creation_time() -> None:
    first = _plan(created_at=datetime(2026, 8, 2, 10, 0, tzinfo=UTC))
    second = _plan(created_at=datetime(2026, 8, 2, 11, 0, tzinfo=UTC))

    assert first.digest == second.digest
    assert first.purge_id == second.purge_id
    assert [action.action_id for action in first.actions] == [
        "provider:run-1",
        "tracking:run-1",
    ]
    with pytest.raises(TypeError):
        first.actions[0].target["provider"] = "other"  # type: ignore[index]


def test_plan_rejects_unknown_action_dependency() -> None:
    action = PurgeAction(
        action_id="tracking:run-1",
        plane="tracking",
        kind="tracking.delete_run",
        target={"project": "purge-fixture", "provider_run_id": "trackio-1"},
        depends_on=("provider:missing",),
    )

    with pytest.raises(ContractError, match="unknown action"):
        PurgePlan.build(
            mode="run",
            project_id="purge-fixture",
            run_ids=("run-1",),
            root_run_id="run-1",
            tracking_actions=(action,),
        )


def test_store_persists_plan_journal_and_receipt_idempotently(tmp_path: Path) -> None:
    store = PurgeStore(tmp_path.resolve())
    plan = _plan()

    assert store.save_plan(plan) == plan
    assert store.save_plan(_plan(created_at=datetime.now(UTC))) == plan
    assert store.plan_path(plan.purge_id).stat().st_mode & 0o777 == 0o600
    assert store.load_plan(plan.purge_id) == plan

    store.append_journal(plan.purge_id, action_id="provider:run-1", status="completed")
    store.append_journal(
        plan.purge_id,
        action_id="tracking:run-1",
        status="failed",
        detail="server unavailable",
    )
    events = store.journal(plan.purge_id)
    assert [event["status"] for event in events] == ["completed", "failed"]
    assert store.journal_path(plan.purge_id).stat().st_mode & 0o777 == 0o600

    receipt = PurgeReceipt(
        purge_id=plan.purge_id,
        plan_digest=plan.digest,
        completed_actions=("provider:run-1",),
        skipped_actions=(),
        failed_action="tracking:run-1",
        completed_at=datetime.now(UTC),
    )
    assert store.save_receipt(receipt) == receipt
    assert store.save_receipt(receipt) == receipt
    assert store.load_receipt(plan.purge_id) == receipt
    assert store.receipt_path(plan.purge_id).stat().st_mode & 0o777 == 0o600


def test_store_rejects_receipt_for_another_plan(tmp_path: Path) -> None:
    store = PurgeStore(tmp_path.resolve())
    plan = _plan()
    store.save_plan(plan)
    receipt = PurgeReceipt(
        purge_id=plan.purge_id,
        plan_digest="sha256:" + "b" * 64,
        completed_actions=(),
        skipped_actions=(),
        failed_action=None,
        completed_at=datetime.now(UTC),
    )

    with pytest.raises(ContractError, match="receipt"):
        store.save_receipt(receipt)


def test_apply_is_journaled_and_resumes_after_failure(tmp_path: Path) -> None:
    store = PurgeStore(tmp_path.resolve())
    plan = _plan()
    store.save_plan(plan)
    calls: list[str] = []

    class Executor:
        def revalidate(self, action: PurgeAction) -> None:
            calls.append(f"check:{action.action_id}")

        def apply(self, action: PurgeAction) -> None:
            calls.append(f"apply:{action.action_id}")
            if action.action_id == "tracking:run-1" and calls.count("apply:tracking:run-1") == 1:
                raise RuntimeError("temporary outage")

    executor = Executor()
    with pytest.raises(PurgeApplyError, match="tracking:run-1"):
        apply_purge_plan(
            store,
            plan.purge_id,
            {"provider": executor, "tracking": executor, "registry": executor, "local": executor},
        )
    assert [event["status"] for event in store.journal(plan.purge_id)] == [
        "started",
        "completed",
        "started",
        "failed",
    ]

    receipt = apply_purge_plan(
        store,
        plan.purge_id,
        {"provider": executor, "tracking": executor, "registry": executor, "local": executor},
    )
    assert receipt.completed_actions == ("provider:run-1", "tracking:run-1")
    assert calls == [
        "check:provider:run-1",
        "apply:provider:run-1",
        "check:tracking:run-1",
        "apply:tracking:run-1",
        "check:tracking:run-1",
        "apply:tracking:run-1",
    ]
