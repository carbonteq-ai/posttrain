"""Tests for reusable work-package composition."""

from __future__ import annotations

from pathlib import Path

from posttrain.common import (
    Catalog,
    CatalogLayer,
    CatalogRef,
    ExecutionTarget,
    PublishedArtifact,
    StoredArtifactRef,
)
from posttrain.work import (
    FinalizedRunResult,
    JobDefinition,
    ProjectBrief,
    RunSpec,
    WorkPackageContext,
    load_work_package,
    prepare_work_package_job,
    run_work_package,
    run_work_package_job,
    validate_work_package,
)


def _fixture(path: Path) -> None:
    path.write_text(
        """
project_id: example
work_package_id: screen/cpu-check
stage: screen
description: Confirm that the selected CPU target resolves and executes.
recipe:
  type: inline
  id: recipes/cpu-check@1
  revision: "1"
  stage: screen
  seats:
    target: target
  jobs:
    - id: check
      kind: data.prepare
      definition: data/cpu-check@1
bindings:
  target:
    type: ref
    family: target
    id: targets/cpu
enabled_optional_jobs: []
metadata:
  question: Does reusable composition work outside the lab?
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_preflight_and_execution_are_available_without_lab(tmp_path: Path) -> None:
    path = tmp_path / "cpu-check.yaml"
    _fixture(path)
    target = ExecutionTarget("targets/cpu", "1", "cpu")
    catalog = Catalog(CatalogLayer("framework-v1", {CatalogRef("target", target.id): target}), (), "example")
    seen: list[str] = []
    definition = JobDefinition(
        "data/cpu-check@1",
        "data.prepare",
        {"target": ExecutionTarget},
        lambda context, seats: seen.append(seats["target"].id) or "checked",
        "Check one resolved CPU target.",
    )
    context = WorkPackageContext(catalog, {definition.id: definition})
    package = load_work_package(path)

    resolved = validate_work_package(context, package)
    assert seen == []
    assert resolved.seat("target", ExecutionTarget) is target

    result = run_work_package(context, package)
    assert seen == ["targets/cpu"]
    assert result.jobs[0].status == "succeeded"
    assert result.jobs[0].value == "checked"


def test_selected_job_preserves_preallocated_run_id(tmp_path: Path) -> None:
    path = tmp_path / "cpu-check.yaml"
    _fixture(path)
    target = ExecutionTarget("targets/cpu", "1", "cpu")
    catalog = Catalog(
        CatalogLayer(
            "framework-v1",
            {CatalogRef("target", target.id): target},
        ),
        (),
        "example",
    )
    definition = JobDefinition(
        "data/cpu-check@1",
        "data.prepare",
        {"target": ExecutionTarget},
        lambda context, seats: context.run_id,
    )
    specs: list[RunSpec] = []
    context = WorkPackageContext(
        catalog,
        {definition.id: definition},
        executor=lambda spec, operation: specs.append(spec) or operation,
    )

    result = run_work_package_job(
        context,
        load_work_package(path),
        "check",
        run_id="run-preallocated-1",
    )

    assert specs[0].run_id == "run-preallocated-1"
    assert result.jobs[0].run_id == "run-preallocated-1"


def test_selected_job_can_be_prepared_without_execution(tmp_path: Path) -> None:
    path = tmp_path / "cpu-check.yaml"
    _fixture(path)
    target = ExecutionTarget("targets/cpu", "1", "cpu")
    catalog = Catalog(
        CatalogLayer(
            "framework-v1",
            {CatalogRef("target", target.id): target},
        ),
        (),
        "example",
    )
    seen: list[str] = []
    definition = JobDefinition(
        "data/cpu-check@1",
        "data.prepare",
        {"target": ExecutionTarget},
        lambda context, seats: seen.append(context.run_id),
    )
    context = WorkPackageContext(catalog, {definition.id: definition})

    prepared = prepare_work_package_job(
        context,
        load_work_package(path),
        "check",
        run_id="run-planned-1",
    )

    assert seen == []
    assert prepared.recipe_job.id == "check"
    assert prepared.definition.id == "data/cpu-check@1"
    assert prepared.spec.run_id == "run-planned-1"
    assert prepared.seats["target"] is target


def test_run_snapshot_includes_project_brief_and_digest(tmp_path: Path) -> None:
    path = tmp_path / "cpu-check.yaml"
    _fixture(path)
    target = ExecutionTarget("targets/cpu", "1", "cpu")
    catalog = Catalog(CatalogLayer("framework-v1", {CatalogRef("target", target.id): target}), (), "example")
    definition = JobDefinition(
        "data/cpu-check@1",
        "data.prepare",
        {"target": ExecutionTarget},
        lambda context, seats: "checked",
    )
    specs: list[RunSpec] = []
    brief = ProjectBrief(objective="Confirm project-brief snapshot propagation.")
    context = WorkPackageContext(
        catalog,
        {definition.id: definition},
        project_brief=brief,
        executor=lambda spec, operation: specs.append(spec) or "captured",
    )

    run_work_package(context, load_work_package(path))

    assert len(specs) == 1
    snapshot = specs[0].resolved_inputs["project_brief"]
    assert snapshot["objective"] == brief.objective  # type: ignore[index]
    assert isinstance(snapshot["digest"], str)  # type: ignore[index]


def test_work_package_exposes_published_artifacts_without_wrapping_value(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cpu-check.yaml"
    _fixture(path)
    target = ExecutionTarget("targets/cpu", "1", "cpu")
    catalog = Catalog(
        CatalogLayer(
            "framework-v1",
            {CatalogRef("target", target.id): target},
        ),
        (),
        "example",
    )
    definition = JobDefinition(
        "data/cpu-check@1",
        "data.prepare",
        {"target": ExecutionTarget},
        lambda context, seats: "checked",
    )
    published = PublishedArtifact(
        "model/final",
        "model",
        StoredArtifactRef(
            "trackio",
            "example",
            "model-final",
            "v0",
            "a" * 64,
        ),
    )
    context = WorkPackageContext(
        catalog,
        {definition.id: definition},
        executor=lambda spec, operation: FinalizedRunResult(
            "checked",
            (published,),
        ),
    )

    result = run_work_package(context, load_work_package(path))

    assert result.jobs[0].value == "checked"
    assert result.jobs[0].published_artifacts == (published,)
