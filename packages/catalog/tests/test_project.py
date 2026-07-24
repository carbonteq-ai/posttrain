from __future__ import annotations

from pathlib import Path

import pytest
from posttrain.catalog import load_project_layout
from posttrain.common import ContractError


def _write_manifest(root: Path, content: str) -> Path:
    control = root / ".posttrain"
    control.mkdir(parents=True)
    path = control / "project.toml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def test_schema_version_one_project_remains_compatible_without_brief(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        """
schema_version = 1
project_id = "legacy-project"
catalog_overlays = []
""",
    )

    layout = load_project_layout(tmp_path)

    assert layout.project_id == "legacy-project"
    assert layout.project_brief is None


def test_schema_version_two_resolves_project_brief(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        """
schema_version = 2
project_id = "capacity-project"
catalog_overlays = []
project_brief = "project.yaml"
""",
    )
    brief = tmp_path / ".posttrain" / "project.yaml"
    brief.write_text("schema_version: 1\nobjective: Screen serving candidates.\n", encoding="utf-8")

    layout = load_project_layout(tmp_path)

    assert layout.project_brief == brief.resolve()


def test_schema_version_two_requires_existing_project_brief(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        """
schema_version = 2
project_id = "capacity-project"
catalog_overlays = []
project_brief = "missing.yaml"
""",
    )

    with pytest.raises(ContractError, match="project brief not found"):
        load_project_layout(tmp_path)


def test_schema_version_one_rejects_project_brief_field(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        """
schema_version = 1
project_id = "legacy-project"
catalog_overlays = []
project_brief = "project.yaml"
""",
    )

    with pytest.raises(ContractError, match="requires project schema_version 2"):
        load_project_layout(tmp_path)
