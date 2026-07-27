from __future__ import annotations

from pathlib import Path


def prepare_run_workspace(workspace: Path) -> None:
    """Create a new workspace or accept an empty pre-mounted run directory."""
    if not workspace.exists():
        workspace.mkdir(parents=True)
        return
    if not workspace.is_dir():
        raise NotADirectoryError(f"training workspace is not a directory: {workspace}")
    if any(workspace.iterdir()):
        raise FileExistsError(f"training workspace is not empty: {workspace}")
