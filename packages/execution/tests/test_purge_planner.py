from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from posttrain.execution import (
    PurgeRunCandidate,
    RegistryManifestRef,
    build_project_purge_plan,
    build_run_purge_plan,
)


def _candidate(run_id: str, *, consumers: tuple[str, ...] = ()) -> PurgeRunCandidate:
    return PurgeRunCandidate(
        run_id=run_id,
        project_id="fixture",
        provider="dstack",
        provider_id=f"job-{run_id}",
        state="succeeded",
        reconciled=True,
        evidence_provider="trackio",
        evidence_project="fixture",
        tracking_provider_run_id=f"trackio-{run_id}",
        consumers=consumers,
        image=RegistryManifestRef(
            "registry.lan/carbonteq/posttrain-job",
            "sha256:" + hashlib.sha256(run_id.encode()).hexdigest(),
        ),
        workspace=Path("/var/lib/posttrain/runs") / run_id,
    )


class Catalog:
    def __init__(self, records: dict[str, PurgeRunCandidate]) -> None:
        self.records = records

    def get(self, run_id: str) -> PurgeRunCandidate | None:
        return self.records.get(run_id)

    def list(self) -> tuple[PurgeRunCandidate, ...]:
        return tuple(self.records.values())


def test_run_plan_blocks_consumer_without_cascade() -> None:
    records = {"producer": _candidate("producer", consumers=("consumer",)), "consumer": _candidate("consumer")}
    plan = build_run_purge_plan(Catalog(records), root_run_id="producer")
    assert plan.run_ids == ("producer",)
    assert any("unselected run 'consumer'" in blocker for blocker in plan.blockers)


def test_cascade_orders_tracking_leaf_before_root() -> None:
    records = {
        "producer": _candidate("producer", consumers=("consumer",)),
        "consumer": _candidate("consumer", consumers=("leaf",)),
        "leaf": _candidate("leaf"),
    }
    plan = build_run_purge_plan(Catalog(records), root_run_id="producer", cascade=True)
    assert plan.blockers == ()
    assert plan.run_ids == ("producer", "consumer", "leaf")
    assert [action.action_id for action in plan.tracking_actions] == [
        "tracking:leaf",
        "tracking:consumer",
        "tracking:producer",
    ]
    assert plan.tracking_actions[-1].depends_on == ("tracking:consumer", "registry:producer")


def test_planner_blocks_unreconciled_or_cross_project_candidates() -> None:
    root = _candidate("producer")
    foreign = replace(root, run_id="foreign", project_id="other")
    records = {"producer": root, "foreign": foreign}
    root_with_consumer = _candidate("producer", consumers=("foreign",))
    records["producer"] = root_with_consumer
    plan = build_run_purge_plan(Catalog(records), root_run_id="producer", cascade=True)
    assert any("belongs to project 'other'" in blocker for blocker in plan.blockers)


def test_project_planner_makes_unmatched_inventory_explicit() -> None:
    records = {
        "producer": _candidate("producer", consumers=("missing",)),
        "foreign": replace(_candidate("foreign"), project_id="other"),
    }
    plan = build_project_purge_plan(Catalog(records), project_id="fixture")
    assert plan.mode == "project"
    assert plan.run_ids == ("producer",)
    assert any("unmatched consumer 'missing'" in blocker for blocker in plan.blockers)
    assert any("unmatched run 'foreign'" in blocker for blocker in plan.blockers)


def test_resume_plan_emits_only_unfinished_planes() -> None:
    candidate = replace(
        _candidate("resume"),
        completed_planes=("provider", "registry", "tracking"),
        local_paths=(Path("/tmp/posttrain-state/resume"),),
    )
    plan = build_run_purge_plan(Catalog({"resume": candidate}), root_run_id="resume")

    assert plan.blockers == ()
    assert plan.provider_actions == ()
    assert plan.registry_actions == ()
    assert plan.tracking_actions == ()
    assert [action.action_id for action in plan.local_actions] == ["local:resume:0"]
    assert plan.local_actions[0].depends_on == ()
