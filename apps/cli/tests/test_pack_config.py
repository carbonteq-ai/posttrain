from __future__ import annotations

from pathlib import Path

import pytest
from posttrain.catalog import ProjectLayout
from posttrain.common import ContractError
from posttrain_cli.pack_config import load_project_pack_config


def _layout(root: Path) -> ProjectLayout:
    control = root / ".posttrain"
    control.mkdir()
    manifest = control / "project.toml"
    manifest.write_text("schema_version=1\nproject_id='example'\n")
    work_packages = control / "work_packages"
    work_packages.mkdir()
    return ProjectLayout(
        project_id="example",
        root=root.resolve(),
        control_dir=control.resolve(),
        manifest=manifest.resolve(),
        catalog_overlays=(),
        work_packages=work_packages.resolve(),
        state=(control / "state").resolve(),
    )


def test_loads_explicit_monorepo_source_selection(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    package = tmp_path / "apps" / "project"
    (package / "src" / "example").mkdir(parents=True)
    (package / "pyproject.toml").write_text("[project]\nname='example'\n")
    (tmp_path / "pyproject.toml").write_text(
        """
[tool.posttrain.pack]
project_packages = ["apps/project"]
source_includes = ["apps/project"]
""".strip()
        + "\n"
    )

    config = load_project_pack_config(layout)

    assert config.project_packages == ("apps/project",)
    assert config.source_includes == ("apps/project",)
    assert config.source_request(layout.root).root == layout.root


def test_defaults_to_installable_root_src_and_declared_readme(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    (tmp_path / "src" / "example").mkdir(parents=True)
    (tmp_path / "README.md").write_text("# Example\n")
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "example"
readme = "README.md"
""".strip()
        + "\n"
    )

    config = load_project_pack_config(layout)

    assert config.project_packages == (".",)
    assert config.source_includes == ("README.md", "pyproject.toml", "src")


def test_cli_source_override_wins_and_unsafe_paths_fail(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    package = tmp_path / "package"
    package.mkdir()
    (package / "pyproject.toml").write_text("[project]\nname='example'\n")
    (tmp_path / "pyproject.toml").write_text(
        "[tool.posttrain.pack]\nproject_packages=['missing']\n"
    )

    config = load_project_pack_config(
        layout,
        project_packages=("package",),
        source_includes=("package",),
    )
    assert config.project_packages == ("package",)

    with pytest.raises(ContractError, match="outside control state"):
        load_project_pack_config(
            layout,
            project_packages=("package",),
            source_includes=(".posttrain/state",),
        )
