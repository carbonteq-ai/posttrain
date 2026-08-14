"""Nested-Lab source and control-tree packing proof.

Lab owns its control tree beside its Python source.  These tests deliberately
copy only the control closure each gate needs into a temporary
``apps/lab``-shaped project.  They exercise the public project planner plus
the primary CLI's actual-job planning seam without creating an OCI image or
consulting a provider.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from posttrain.common import ContractError
from posttrain.execution_pack import ImmutableSourceSnapshotter
from posttrain.project import Project
from posttrain.runtime_images.manifest import load_manifest as _load_manifest
from posttrain_cli.context import CliState
from posttrain_cli.execution_planning import _project_config_bundle, plan_job_package
from posttrain_cli.framework_distributions import FrameworkDistributions

WORKSPACE = Path(__file__).resolve().parents[3]
LAB = WORKSPACE / "apps" / "lab"


@pytest.fixture(autouse=True)
def _candidate_manifest_for_nested_packing_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep source/control planning tests independent of OCI publication.

    The actual installed-manifest loader remains strict.  These tests model a
    release candidate before its matching base/kind graph is materialized and
    exercise only the nested-project packaging boundary.
    """

    manifest = _load_manifest(verify_locks=False)
    monkeypatch.setattr("posttrain_cli.execution_config.load_manifest", lambda: manifest)


def _nested_lab_project(tmp_path: Path, work_package: str) -> Path:
    """Copy the minimum owned source and control closure into ``apps/lab``."""

    project = tmp_path / "workspace" / "apps" / "lab"
    project.mkdir(parents=True)
    for name in ("README.md", "pyproject.toml"):
        shutil.copy2(LAB / name, project / name)
    shutil.copytree(LAB / "src", project / "src")

    control = project / ".posttrain"
    control.mkdir()
    for name in ("project.toml", "project.yaml"):
        shutil.copy2(LAB / ".posttrain" / name, control / name)
    shutil.copytree(LAB / ".posttrain" / "catalog", control / "catalog")
    packages = control / "work_packages"
    packages.mkdir()
    shutil.copy2(LAB / ".posttrain" / "work_packages" / work_package, packages / work_package)
    return project


@pytest.mark.parametrize(
    ("work_package", "job"),
    (
        ("sft_data_prepare_qualification.yaml", "prepare"),
        ("qwen2b_eval_qualification.yaml", "evaluate"),
    ),
)
def test_release_gate_nested_lab_plan_keeps_source_and_control_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    work_package: str,
    job: str,
) -> None:
    """A staged wheelhouse makes Lab the only source snapshot root.

    The fake wheel materializer is intentional: obtaining the staged release
    wheel set is a release operation, not a unit-test dependency.  Everything
    on either side of that boundary is the real planner and project service.
    """

    project = _nested_lab_project(tmp_path, work_package)
    wheelhouse = tmp_path / "framework-wheels"
    wheelhouse.mkdir()
    framework_wheels = FrameworkDistributions((), "a" * 64)
    monkeypatch.setattr(
        "posttrain_cli.execution_planning.materialize_framework_distributions",
        lambda *_args, **_kwargs: framework_wheels,
    )

    intent = Project.open(project).jobs.plan(work_package, job=job)
    planned = plan_job_package(
        CliState(project_root=project),
        Path(work_package),
        job=job,
        intent=intent,
        local_publication=True,
        framework_wheelhouse=wheelhouse,
    )

    assert planned.layout.root == project.resolve()
    assert planned.work_package_path == (project / ".posttrain" / "work_packages" / work_package).resolve()
    assert planned.project_source_request.root == project.resolve()
    assert planned.project_source_request.install_roots == (".",)
    assert planned.project_source_request.includes == ("README.md", "pyproject.toml", "src")
    assert planned.framework_source_request is None
    assert planned.framework_distributions == framework_wheels

    snapshot = ImmutableSourceSnapshotter(cache_root=tmp_path / "snapshots").materialize(planned.project_source_request)
    source_files = {
        path.relative_to(snapshot.package.root).as_posix()
        for path in snapshot.package.root.rglob("*")
        if path.is_file()
    }
    assert ".posttrain/project.toml" not in source_files
    assert "src/posttrain_lab/qualification/gates.toml" in source_files

    project_config = _project_config_bundle(
        planned.layout,
        planned.work_package_path,
        planned.prepared,
        planned.catalog,
    )
    assert project_config.digest == planned.project_config_digest
    assert project_config.project_manifest == ".posttrain/project.toml"
    assert project_config.selected_work_package == f".posttrain/work_packages/{work_package}"
    assert set(project_config.files).isdisjoint(source_files)
    assert ".posttrain/project.toml" in project_config.files
    assert project_config.selected_work_package in project_config.files


@pytest.mark.parametrize(
    ("work_package", "job"),
    (
        ("sft_data_prepare_qualification.yaml", "prepare"),
        ("qwen2b_eval_qualification.yaml", "evaluate"),
    ),
)
def test_release_gate_nested_lab_plan_requires_staged_framework_wheels(
    tmp_path: Path,
    work_package: str,
    job: str,
) -> None:
    """Record the concrete prerequisite before local OCI materialization.

    The planner intentionally reaches the distribution boundary before any
    dataset/environment packaging, BuildKit invocation, or remote publication.
    """

    project = _nested_lab_project(tmp_path, work_package)
    wheelhouse = tmp_path / "empty-framework-wheels"
    wheelhouse.mkdir()
    intent = Project.open(project).jobs.plan(work_package, job=job)

    with pytest.raises(ContractError, match="framework wheelhouse must contain exactly one"):
        plan_job_package(
            CliState(project_root=project),
            Path(work_package),
            job=job,
            intent=intent,
            local_publication=True,
            framework_wheelhouse=wheelhouse,
        )
