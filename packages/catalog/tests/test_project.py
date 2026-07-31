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


def test_project_loads_non_secret_execution_defaults(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        """
schema_version = 1
project_id = "execution-project"
catalog_overlays = []

[execution]
provider = "dstack"
target = "targets/rtx-pro-6000@1"
runtime_profile = "training/verl@1"
timeout_seconds = 7200
max_attempts = 2
priority = 5
environment_names = ["TRACKIO_SERVER_URL", "TRACKIO_WRITE_TOKEN"]
""",
    )

    layout = load_project_layout(tmp_path)

    assert layout.execution.provider == "dstack"
    assert layout.execution.target == "targets/rtx-pro-6000@1"
    assert layout.execution.runtime_profile == "training/verl@1"
    assert layout.execution.timeout_seconds == 7200
    assert layout.execution.max_attempts == 2
    assert layout.execution.priority == 5
    assert layout.execution.environment_names == (
        "TRACKIO_SERVER_URL",
        "TRACKIO_WRITE_TOKEN",
    )


def test_project_loads_tracked_catalog_plugin_requirements(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        """
schema_version = 1
project_id = "plugin-project"
catalog_overlays = []

[catalog_plugins]
required = ["acme-posttrain-catalog>=1,<2"]
""",
    )

    layout = load_project_layout(tmp_path)

    assert layout.catalog_plugin_requirements == ("acme-posttrain-catalog>=1,<2",)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("provider", '"Dstack"', "lowercase identifier"),
        ("timeout_seconds", "0", "greater than or equal to 1"),
        (
            "environment_names",
            '["TRACKIO_WRITE_TOKEN=secret"]',
            "must be variable names",
        ),
    ],
)
def test_project_rejects_invalid_execution_defaults(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    _write_manifest(
        tmp_path,
        f"""
schema_version = 1
project_id = "execution-project"
catalog_overlays = []

[execution]
{field} = {value}
""",
    )

    with pytest.raises(ContractError, match=message):
        load_project_layout(tmp_path)
