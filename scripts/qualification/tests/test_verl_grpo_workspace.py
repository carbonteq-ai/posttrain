from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[3] / "tools" / "run_workspace.py"
SPEC = importlib.util.spec_from_file_location("run_workspace", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
prepare_run_workspace = MODULE.prepare_run_workspace


def test_prepare_workspace_accepts_empty_precreated_mount(tmp_path: Path) -> None:
    workspace = tmp_path / "mounted-run"
    workspace.mkdir()

    prepare_run_workspace(workspace)

    assert workspace.is_dir()
    assert list(workspace.iterdir()) == []


def test_prepare_workspace_creates_missing_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "new-run"

    prepare_run_workspace(workspace)

    assert workspace.is_dir()


def test_prepare_workspace_refuses_nonempty_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "existing-run"
    workspace.mkdir()
    (workspace / "evidence.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        prepare_run_workspace(workspace)
