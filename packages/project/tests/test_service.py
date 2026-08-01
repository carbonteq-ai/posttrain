from __future__ import annotations

from pathlib import Path

import pytest
from posttrain.common import ContractError
from posttrain.project import Project


def _write_project(root: Path, *, jobs: str) -> Path:
    control = root / ".posttrain"
    work_packages = control / "work_packages"
    work_packages.mkdir(parents=True)
    (control / "project.toml").write_text(
        "\n".join(
            (
                "schema_version = 1",
                'project_id = "project-service-test"',
                "catalog_overlays = []",
                'tracking = "none"',
                "",
            )
        ),
        encoding="utf-8",
    )
    path = work_packages / "sft.yaml"
    path.write_text(
        "\n".join(
            (
                "project_id: project-service-test",
                "work_package_id: train/sft",
                "stage: train",
                "recipe:",
                "  type: inline",
                "  id: recipes/sft@1",
                '  revision: "1"',
                "  stage: train",
                "  seats: {model: model, dataset: dataset, settings: training, training: training}",
                "  jobs:",
                jobs,
                "bindings:",
                "  model: {type: ref, family: model, id: models/qwen3.5-2b@bf16}",
                "  dataset: {type: ref, family: dataset, id: datasets/posttrain-sft-smoke@1}",
                "  settings: {type: ref, family: training, id: qwen3.5-2b/sft-smoke-v2}",
                "  training: {type: ref, family: training, id: training/qwen3.5-trl-lora@1}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return path


def test_open_and_plan_returns_a_provider_free_static_intent(tmp_path: Path) -> None:
    _write_project(tmp_path, jobs="    - {id: train, kind: train.sft, definition: train/trl-sft@1}")

    project = Project.open(tmp_path)
    intent = project.jobs.plan("sft.yaml")

    assert intent.layout is project.layout
    assert intent.catalog is project.catalog
    assert intent.work_package_path == tmp_path / ".posttrain" / "work_packages" / "sft.yaml"
    assert intent.job_id == "train"
    assert intent.prepared.definition.id == "train/trl-sft@1"
    assert intent.prepared.spec.project_id == "project-service-test"
    assert intent.prepared.spec.job_kind == "train.sft"


def test_plan_requires_an_explicit_job_when_several_are_enabled(tmp_path: Path) -> None:
    _write_project(
        tmp_path,
        jobs="\n".join(
            (
                "    - {id: first, kind: train.sft, definition: train/trl-sft@1}",
                "    - {id: second, kind: train.sft, definition: train/trl-sft@1}",
            )
        ),
    )

    with pytest.raises(ContractError, match="pass job="):
        Project.open(tmp_path).jobs.plan("sft.yaml")
