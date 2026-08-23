from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from posttrain.execution import (
    PurgeReason,
    PurgeRunCandidate,
    RegistryManifestRef,
    build_project_purge_plan,
    build_run_purge_plan,
)

_REASON = PurgeReason(category="disposable-fixture")


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

    def registry_image_owners(self) -> dict[str, tuple[str, ...]]:
        owners: dict[str, list[str]] = {}
        for candidate in self.records.values():
            if candidate.image is not None and "registry" not in candidate.completed_planes:
                owners.setdefault(candidate.image.value, []).append(candidate.run_id)
        return {reference: tuple(run_ids) for reference, run_ids in owners.items()}

    def registry_inventory_blockers(self) -> tuple[str, ...]:
        return ()


def test_run_plan_blocks_consumer_without_cascade() -> None:
    records = {"producer": _candidate("producer", consumers=("consumer",)), "consumer": _candidate("consumer")}
    plan = build_run_purge_plan(Catalog(records), root_run_id="producer", reason=_REASON)
    assert plan.run_ids == ("producer",)
    assert any("unselected run 'consumer'" in blocker for blocker in plan.blockers)


def test_cascade_orders_tracking_leaf_before_root() -> None:
    records = {
        "producer": _candidate("producer", consumers=("consumer",)),
        "consumer": _candidate("consumer", consumers=("leaf",)),
        "leaf": _candidate("leaf"),
    }
    plan = build_run_purge_plan(Catalog(records), root_run_id="producer", reason=_REASON, cascade=True)
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
    plan = build_run_purge_plan(Catalog(records), root_run_id="producer", reason=_REASON, cascade=True)
    assert any("belongs to project 'other'" in blocker for blocker in plan.blockers)


def test_project_planner_makes_unmatched_inventory_explicit() -> None:
    records = {
        "producer": _candidate("producer", consumers=("missing",)),
        "foreign": replace(_candidate("foreign"), project_id="other"),
    }
    plan = build_project_purge_plan(Catalog(records), project_id="fixture", reason=_REASON)
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
    plan = build_run_purge_plan(Catalog({"resume": candidate}), root_run_id="resume", reason=_REASON)

    assert plan.blockers == ()
    assert plan.provider_actions == ()
    assert plan.registry_actions == ()
    assert plan.tracking_actions == ()
    assert [action.action_id for action in plan.local_actions] == ["local:resume:0"]
    assert plan.local_actions[0].depends_on == ()


def test_run_plan_retains_image_referenced_by_unselected_run() -> None:
    retained = _candidate("retained")
    assert retained.image is not None
    smoke = replace(_candidate("smoke"), image=retained.image)

    plan = build_run_purge_plan(
        Catalog({"smoke": smoke, "retained": retained}),
        root_run_id="smoke",
        reason=_REASON,
    )

    assert plan.blockers == ()
    assert plan.registry_actions == ()
    assert plan.tracking_actions[0].depends_on == ()
    assert plan.warnings == (
        f"job image {retained.image.value!r} retained; referenced by unselected run(s): 'retained'",
    )


def test_run_plan_retains_image_referenced_by_cross_project_machine_owner() -> None:
    smoke = _candidate("smoke")
    image = smoke.image
    assert image is not None

    class MachineCatalog(Catalog):
        def registry_image_owners(self) -> dict[str, tuple[str, ...]]:
            return {image.value: ("smoke", "foreign-run")}

    plan = build_run_purge_plan(
        MachineCatalog({"smoke": smoke}),
        root_run_id="smoke",
        reason=_REASON,
    )

    assert plan.registry_actions == ()
    assert plan.warnings == (f"job image {image.value!r} retained; referenced by unselected run(s): 'foreign-run'",)


def test_run_plan_blocks_when_machine_registry_inventory_is_incomplete() -> None:
    class IncompleteCatalog(Catalog):
        def registry_inventory_blockers(self) -> tuple[str, ...]:
            return ("machine registry ownership inventory is incomplete",)

    plan = build_run_purge_plan(
        IncompleteCatalog({"smoke": _candidate("smoke")}),
        root_run_id="smoke",
        reason=_REASON,
    )

    assert plan.blockers == ("machine registry ownership inventory is incomplete",)


def test_cascade_deletes_shared_image_once_after_all_selected_providers() -> None:
    consumer = _candidate("consumer")
    producer = replace(
        _candidate("producer", consumers=("consumer",)),
        image=consumer.image,
    )

    plan = build_run_purge_plan(
        Catalog({"producer": producer, "consumer": consumer}),
        root_run_id="producer",
        reason=_REASON,
        cascade=True,
    )

    assert plan.blockers == ()
    assert len(plan.registry_actions) == 1
    assert plan.registry_actions[0].action_id == "registry:producer"
    assert plan.registry_actions[0].depends_on == ("provider:producer", "provider:consumer")
    assert [action.depends_on for action in plan.tracking_actions] == [
        ("registry:producer",),
        ("tracking:consumer", "registry:producer"),
    ]


def test_explicit_run_purge_warns_when_overriding_a_pin() -> None:
    pinned = replace(_candidate("pinned"), evidence_retention="pinned")

    plan = build_run_purge_plan(Catalog({"pinned": pinned}), root_run_id="pinned", reason=_REASON)

    assert plan.blockers == ()
    assert plan.warnings == ("run 'pinned' is pinned; explicit run purge overrides its retention pin",)


def test_cascade_and_project_purge_block_pinned_runs() -> None:
    pinned = replace(_candidate("pinned"), evidence_retention="pinned")
    root = _candidate("root", consumers=("pinned",))
    catalog = Catalog({"root": root, "pinned": pinned})

    cascade = build_run_purge_plan(catalog, root_run_id="root", reason=_REASON, cascade=True)
    project = build_project_purge_plan(catalog, project_id="fixture", reason=_REASON)

    assert "pinned run 'pinned' cannot be added by a cascade purge" in cascade.blockers
    assert "pinned run 'pinned' cannot be included in project purge without explicit run selection" in project.blockers
