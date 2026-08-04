"""Safe local-state action executor for the final purge plane."""

from __future__ import annotations

import shutil
from pathlib import Path

from posttrain.common import ContractError

from .purge import PurgeAction


class LocalStatePurgeExecutor:
    """Remove only exact run-scoped paths below configured state roots."""

    def __init__(self, allowed_roots: tuple[Path, ...]) -> None:
        if not allowed_roots:
            raise ValueError("local purge needs at least one allowed state root")
        self._roots = tuple(root.resolve() for root in allowed_roots)

    def revalidate(self, action: PurgeAction) -> None:
        if action.kind != "local.remove_path":
            raise ContractError(f"unsupported local purge action {action.kind!r}")
        raw_path = action.target.get("path")
        run_id = action.target.get("run_id")
        if not isinstance(raw_path, str) or not isinstance(run_id, str):
            raise ContractError("local purge action must name a path and run id")
        path = Path(raw_path)
        if not path.is_absolute() or path.name != run_id or path == Path("/"):
            raise ContractError("local purge target must be one exact run directory")
        if path.is_symlink():
            raise ContractError("local purge refuses symlinked state")
        resolved = path.resolve(strict=False)
        if any(resolved == root for root in self._roots):
            raise ContractError("local purge target must be one exact run directory")
        if not any(resolved.is_relative_to(root) for root in self._roots):
            raise ContractError("local purge target is outside the allowed state roots")

    def apply(self, action: PurgeAction) -> None:
        self.revalidate(action)
        path = Path(str(action.target["path"]))
        if not path.exists():
            return
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


__all__ = ["LocalStatePurgeExecutor"]
