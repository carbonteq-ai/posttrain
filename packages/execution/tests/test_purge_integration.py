from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from posttrain.execution import (
    PurgeAction,
    PurgeApplyError,
    PurgePlane,
    PurgeRunCandidate,
    PurgeStore,
    RegistryManifestRef,
    apply_purge_plan,
    build_run_purge_plan,
)


class Catalog:
    def __init__(self, values: dict[str, PurgeRunCandidate]) -> None:
        self.values = values

    def get(self, run_id: str) -> PurgeRunCandidate | None:
        return self.values.get(run_id)

    def list(self) -> tuple[PurgeRunCandidate, ...]:
        return tuple(self.values.values())


class FixtureExecutor:
    def __init__(self, events: list[str], *, fail_once_on: str | None = None) -> None:
        self.events = events
        self.fail_once_on = fail_once_on
        self.failed = False

    def revalidate(self, action: PurgeAction) -> None:
        self.events.append(f"check:{action.action_id}")

    def apply(self, action: PurgeAction) -> None:
        self.events.append(f"apply:{action.action_id}")
        if action.action_id == self.fail_once_on and not self.failed:
            self.failed = True
            raise RuntimeError("fixture interruption")


def _candidate(run_id: str, consumers: tuple[str, ...] = ()) -> PurgeRunCandidate:
    return PurgeRunCandidate(
        run_id=run_id,
        project_id="disposable-fixture",
        provider="local-docker",
        provider_id=f"container-{run_id}",
        state="succeeded",
        reconciled=True,
        evidence_provider="trackio",
        evidence_project="disposable-fixture",
        tracking_provider_run_id=f"trackio-{run_id}",
        consumers=consumers,
        image=RegistryManifestRef(
            "registry.lan/carbonteq/posttrain-job",
            "sha256:" + hashlib.sha256(run_id.encode()).hexdigest(),
        ),
        workspace=Path("/tmp/posttrain-disposable") / run_id,
    )


def test_disposable_three_run_fixture_is_leaf_first_and_resumable(tmp_path: Path) -> None:
    catalog = Catalog(
        {
            "producer": _candidate("producer", ("consumer",)),
            "consumer": _candidate("consumer", ("leaf",)),
            "leaf": _candidate("leaf"),
        }
    )
    plan = build_run_purge_plan(catalog, root_run_id="producer", cascade=True)
    assert plan.blockers == ()
    store = PurgeStore(tmp_path.resolve())
    store.save_plan(plan)
    events: list[str] = []
    provider = FixtureExecutor(events)
    registry = FixtureExecutor(events, fail_once_on="registry:consumer")
    tracking = FixtureExecutor(events)
    local = FixtureExecutor(events)
    executors: dict[PurgePlane, FixtureExecutor] = {
        "provider": provider,
        "registry": registry,
        "tracking": tracking,
        "local": local,
    }

    with pytest.raises(PurgeApplyError, match="registry:consumer"):
        apply_purge_plan(store, plan.purge_id, executors)
    receipt = apply_purge_plan(store, plan.purge_id, executors)

    assert receipt.completed_actions == tuple(action.action_id for action in plan.actions)
    tracking_applies = [event for event in events if event.startswith("apply:tracking:")]
    assert tracking_applies == [
        "apply:tracking:leaf",
        "apply:tracking:consumer",
        "apply:tracking:producer",
    ]
    assert [event["status"] for event in store.journal(plan.purge_id)].count("failed") == 1
