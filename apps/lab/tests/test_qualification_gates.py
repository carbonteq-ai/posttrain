"""Qualification-gate registry coverage independent of provider execution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from posttrain.catalog import ProjectLayout
from posttrain_lab.cli import main
from posttrain_lab.qualification import (
    QualificationGate,
    QualificationGateError,
    load_qualification_gates,
    validate_qualification_project,
)

WORKSPACE = Path(__file__).resolve().parents[3]


def _layout(tmp_path: Path) -> ProjectLayout:
    root = tmp_path / "project"
    control = root / ".posttrain"
    work_packages = control / "work_packages"
    work_packages.mkdir(parents=True)
    return ProjectLayout(
        project_id="tests",
        root=root.resolve(),
        control_dir=control.resolve(),
        manifest=(control / "project.toml").resolve(),
        catalog_overlays=(),
        work_packages=work_packages.resolve(),
        state=(control / "state").resolve(),
    )


def _write_work_package(layout: ProjectLayout, name: str = "sample.yaml") -> str:
    path = layout.work_packages / name
    path.write_text(
        """project_id: tests
work_package_id: screen/sample
stage: screen
recipe:
  type: inline
  id: recipes/sample@1
  revision: "1"
  stage: screen
  seats:
    target: target
  jobs:
    - id: prepare
      kind: data.prepare
      definition: data/sample@1
  expected_artifacts: []
bindings:
  target:
    type: ref
    family: target
    id: targets/test
""",
        encoding="utf-8",
    )
    return path.relative_to(layout.root).as_posix()


def _gate(work_package: str, **changes: str) -> QualificationGate:
    values = {
        "id": "sample",
        "work_package": work_package,
        "job_id": "prepare",
        "tier": "extended",
        "state": "active",
        "job_kind": "data.prepare",
        "acceptance": "dataset-canonicalization",
    }
    values.update(changes)
    return QualificationGate(**values)  # type: ignore[arg-type]


def test_package_registry_classifies_every_current_work_package() -> None:
    layout = ProjectLayout(
        project_id="foundation-models",
        root=WORKSPACE,
        control_dir=WORKSPACE / ".posttrain",
        manifest=WORKSPACE / ".posttrain" / "project.toml",
        catalog_overlays=(WORKSPACE / ".posttrain" / "catalog",),
        work_packages=WORKSPACE / ".posttrain" / "work_packages",
        state=WORKSPACE / ".posttrain" / "state",
    )

    inventory = validate_qualification_project(layout, load_qualification_gates())

    assert len(inventory.gates) == 25
    assert len(inventory.classified) == 25
    assert inventory.unclassified == ()
    assert {gate.id for gate in inventory.gates if gate.tier == "release"} == {
        "qwen2b-gsm8k-evaluation",
        "sft-data-prepare",
    }


def test_registry_rejects_a_missing_work_package(tmp_path: Path) -> None:
    layout = _layout(tmp_path)

    with pytest.raises(QualificationGateError, match="references missing work package"):
        validate_qualification_project(layout, (_gate(".posttrain/work_packages/missing.yaml"),))


def test_registry_rejects_duplicate_work_package_classification(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    path = _write_work_package(layout)

    with pytest.raises(QualificationGateError, match="duplicate work_package"):
        validate_qualification_project(
            layout,
            (
                _gate(path),
                _gate(path, id="same-package-twice"),
            ),
        )


def test_registry_rejects_unclassified_work_package(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    path = _write_work_package(layout)

    with pytest.raises(QualificationGateError, match=path):
        validate_qualification_project(layout, ())


def test_registry_rejects_unknown_job_and_retired_release_gate(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    path = _write_work_package(layout)

    with pytest.raises(QualificationGateError, match="unknown job 'absent'"):
        validate_qualification_project(layout, (_gate(path, job_id="absent"),))
    with pytest.raises(QualificationGateError, match="retired and release tier"):
        validate_qualification_project(layout, (_gate(path, tier="release", state="retired"),))


def test_qualification_list_emits_a_stable_json_inventory(capsys: pytest.CaptureFixture[str]) -> None:
    main(["qualification", "list", "--project-root", str(WORKSPACE), "--json"])

    payload = json.loads(capsys.readouterr().out)

    assert payload["schema_version"] == 1
    assert len(payload["gates"]) == 25
    assert payload["unclassified_work_packages"] == []
