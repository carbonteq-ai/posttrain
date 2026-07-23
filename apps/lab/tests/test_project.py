"""Tests for portable post-training project discovery and layout."""

from __future__ import annotations

from pathlib import Path

import pytest
from posttrain.common import ContractError
from posttrain_lab.project import ProjectLayout, discover_project, load_project_layout


def _project(
    root: Path,
    *,
    project_id: str = "example",
    catalog_overlays: str = '["catalog"]',
    work_packages: str = "work_packages",
    state: str = "state",
) -> Path:
    control = root / ".posttrain"
    control.mkdir(parents=True)
    (control / "project.toml").write_text(
        "\n".join(
            (
                "schema_version = 1",
                f'project_id = "{project_id}"',
                f"catalog_overlays = {catalog_overlays}",
                f'work_packages = "{work_packages}"',
                f'state = "{state}"',
                "",
            )
        ),
        encoding="utf-8",
    )
    return root


def test_loads_default_paths_relative_to_control_directory(tmp_path: Path) -> None:
    layout = load_project_layout(_project(tmp_path))

    assert layout.project_id == "example"
    assert layout.root == tmp_path.resolve()
    assert layout.control_dir == (tmp_path / ".posttrain").resolve()
    assert layout.manifest == (tmp_path / ".posttrain" / "project.toml").resolve()
    assert layout.catalog_overlays == ((tmp_path / ".posttrain" / "catalog").resolve(),)
    assert layout.work_packages == (tmp_path / ".posttrain" / "work_packages").resolve()
    assert layout.state == (tmp_path / ".posttrain" / "state").resolve()


def test_discovers_project_from_nested_directory(tmp_path: Path) -> None:
    root = _project(tmp_path / "consumer")
    nested = root / "src" / "project" / "jobs"
    nested.mkdir(parents=True)

    assert discover_project(nested).root == root.resolve()


def test_explicit_root_precedes_environment_and_upward_discovery(tmp_path: Path) -> None:
    explicit = _project(tmp_path / "explicit", project_id="explicit")
    configured = _project(tmp_path / "configured", project_id="configured")
    upward = _project(tmp_path / "upward", project_id="upward")
    nested = upward / "nested"
    nested.mkdir()

    layout = discover_project(
        nested,
        explicit_root=explicit,
        environ={"POSTTRAIN_PROJECT_ROOT": str(configured)},
    )

    assert layout.project_id == "explicit"


def test_environment_precedes_upward_discovery(tmp_path: Path) -> None:
    configured = _project(tmp_path / "configured", project_id="configured")
    upward = _project(tmp_path / "upward", project_id="upward")
    nested = upward / "nested"
    nested.mkdir()

    layout = discover_project(
        nested,
        environ={"POSTTRAIN_PROJECT_ROOT": str(configured)},
    )

    assert layout.project_id == "configured"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("catalog_overlays", '["../../outside"]', "catalog overlay path escapes"),
        ("work_packages", "../../outside", "work_packages path escapes"),
    ],
)
def test_rejects_source_paths_that_escape_project_root(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    options: dict[str, str] = {}
    options[field] = value
    root = _project(tmp_path, **options)

    with pytest.raises(ContractError, match=message):
        load_project_layout(root)


def test_permits_absolute_state_directory(tmp_path: Path) -> None:
    external_state = tmp_path / "large-disk"
    root = _project(tmp_path / "consumer", state=str(external_state))

    assert load_project_layout(root).state == external_state.resolve()


def test_rejects_unknown_manifest_fields(tmp_path: Path) -> None:
    root = _project(tmp_path)
    manifest = root / ".posttrain" / "project.toml"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "unknown = true\n", encoding="utf-8")

    with pytest.raises(ContractError, match="invalid post-training project manifest"):
        load_project_layout(root)


def test_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    root = _project(tmp_path)
    manifest = root / ".posttrain" / "project.toml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace("schema_version = 1", "schema_version = 2"),
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="unsupported post-training project schema_version 2"):
        load_project_layout(root)


def test_missing_project_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="POSTTRAIN_PROJECT_ROOT"):
        discover_project(tmp_path, environ={})


def test_legacy_layout_is_explicit_and_does_not_require_a_manifest(tmp_path: Path) -> None:
    (tmp_path / "catalog" / "base").mkdir(parents=True)
    overlay = tmp_path / "catalog" / "projects" / "projects__example"
    overlay.mkdir(parents=True)

    layout = ProjectLayout.legacy(tmp_path, "projects/example")

    assert layout.project_id == "projects/example"
    assert layout.root == tmp_path.resolve()
    assert layout.catalog_overlays == (overlay.resolve(),)
    assert layout.base_catalog == (tmp_path / "catalog").resolve()
    assert layout.work_packages == (tmp_path / "work_packages").resolve()
    assert layout.state == (tmp_path / ".posttrain" / "state").resolve()
