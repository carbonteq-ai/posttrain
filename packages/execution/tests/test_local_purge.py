from __future__ import annotations

from pathlib import Path

import pytest
from posttrain.common import ContractError
from posttrain.execution import LocalStatePurgeExecutor, PurgeAction


def _action(path: Path, run_id: str = "run-1") -> PurgeAction:
    return PurgeAction(
        action_id="local:run-1",
        plane="local",
        kind="local.remove_path",
        target={"path": str(path), "run_id": run_id},
    )


def test_local_executor_removes_only_scoped_run_directory(tmp_path: Path) -> None:
    root = tmp_path / "state"
    target = root / "runs" / "run-1"
    target.mkdir(parents=True)
    (target / "receipt.json").write_text("{}", encoding="utf-8")
    executor = LocalStatePurgeExecutor((root,))

    executor.apply(_action(target))
    assert not target.exists()
    executor.apply(_action(target))


def test_local_executor_rejects_root_and_symlink_targets(tmp_path: Path) -> None:
    root = tmp_path / "state"
    root.mkdir()
    with pytest.raises(ContractError, match="exact run directory"):
        LocalStatePurgeExecutor((root,)).revalidate(_action(root, run_id="state"))
    target = root / "runs" / "run-1"
    target.parent.mkdir()
    target.symlink_to(tmp_path)
    with pytest.raises(ContractError, match="symlink"):
        LocalStatePurgeExecutor((root,)).revalidate(_action(target))
