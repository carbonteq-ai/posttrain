"""Tests for the lab qualification project entry."""

from __future__ import annotations

from pathlib import Path

from posttrain.catalog import ProjectLayout, open_catalog
from posttrain.jobs import standard_definitions
from posttrain.work import ProjectExecutionRequest
from posttrain_lab.entry import configure


def test_configure_uses_standard_definitions_and_git_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    work_packages = project / ".posttrain" / "work_packages"
    state = project / ".posttrain" / "state"
    work_packages.mkdir(parents=True)
    state.mkdir()
    layout = ProjectLayout(
        root=project,
        control_dir=project / ".posttrain",
        project_id="foundation-models",
        manifest=project / ".posttrain" / "project.toml",
        catalog_overlays=(),
        work_packages=work_packages,
        state=state,
    )
    catalog = open_catalog(scope=layout.project_id)
    path = work_packages / "foundation_screen.yaml"
    path.touch()
    monkeypatch.setenv("POSTTRAIN_PROJECT_REVISION", "test-revision")

    runtime = configure(
        ProjectExecutionRequest(
            project_id=layout.project_id,
            project_root=layout.root,
            state_dir=layout.state,
            work_package_path=path,
            catalog=catalog,
        )
    )

    assert runtime.catalog is catalog
    assert set(runtime.definitions) == set(standard_definitions())
    assert runtime.source_metadata["git_revision"] == "test-revision"
    assert runtime.source_metadata["tracking_backend"] == "trackio"
    assert (state / "scratch").is_dir()
