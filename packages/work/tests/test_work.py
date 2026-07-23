"""Tests for reusable work-package composition."""

from __future__ import annotations

from pathlib import Path

from posttrain.common import Catalog, CatalogLayer, CatalogRef, ExecutionTarget
from posttrain.work import (
    JobDefinition,
    WorkPackageContext,
    load_work_package,
    run_work_package,
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
