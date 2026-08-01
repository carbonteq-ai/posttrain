"""Classify and safely migrate project-local Posttrain state."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from posttrain.catalog import ProjectLayout
from posttrain.common import ContractError

_CACHE_CHILDREN = frozenset({"datasets", "pack", "runs", "runtime-builds", "scratch"})
_TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled", "lost"})


def cache_root(layout: ProjectLayout) -> Path:
    """Return the rebuildable state root for one project."""

    return (layout.state / "cache").resolve()


def cache_path(layout: ProjectLayout, *parts: str) -> Path:
    """Return an owned rebuildable state path without permitting escapes."""

    root = cache_root(layout)
    result = root.joinpath(*parts).resolve()
    if not result.is_relative_to(root):
        raise ContractError("project cache path escapes the state cache root")
    return result


@dataclass(frozen=True, slots=True)
class StateMigrationReport:
    source: Path
    destination: Path
    dry_run: bool
    copied_execution_files: int
    moved_cache_entries: tuple[str, ...]
    protected_entries: tuple[str, ...]
    unresolved_runs: tuple[str, ...]

    @property
    def changed(self) -> bool:
        return self.copied_execution_files > 0 or bool(self.moved_cache_entries)

    def as_json(self) -> dict[str, object]:
        return {
            "source": str(self.source),
            "destination": str(self.destination),
            "dry_run": self.dry_run,
            "copied_execution_files": self.copied_execution_files,
            "moved_cache_entries": list(self.moved_cache_entries),
            "protected_entries": list(self.protected_entries),
            "unresolved_runs": list(self.unresolved_runs),
            "changed": self.changed,
        }


def migrate_state(
    layout: ProjectLayout,
    *,
    source_project_root: Path | None = None,
    dry_run: bool = False,
) -> StateMigrationReport:
    """Split local state and copy verified execution control records.

    A source project is never deleted.  Cache entries are only reorganized
    when migrating the selected project in place; a relocation copies durable
    `executions/` records and deliberately leaves the old cache rebuildable.
    """

    destination = layout.state.resolve()
    source = (
        (source_project_root.resolve() / ".posttrain" / "state")
        if source_project_root is not None
        else destination
    )
    if source_project_root is not None and not source.is_relative_to(source_project_root.resolve()):
        raise ContractError("source project state path is invalid")
    if not source.exists():
        return StateMigrationReport(source, destination, dry_run, 0, (), (), ())
    if not source.is_dir() or source.is_symlink():
        raise ContractError(f"project state root is not a safe directory: {source}")

    unresolved = _unresolved_runs(source / "executions")
    if unresolved:
        raise ContractError(
            "state migration refuses unresolved executions: " + ", ".join(unresolved)
        )

    copied = _copy_executions(source / "executions", destination / "executions", dry_run=dry_run)
    moved: list[str] = []
    protected: list[str] = []
    if source == destination:
        for child in sorted(source.iterdir(), key=lambda value: value.name):
            if child.name in {"cache", "executions"}:
                continue
            if child.name in _CACHE_CHILDREN:
                target = cache_root(layout) / child.name
                if target.exists() and not _same_tree(child, target):
                    raise ContractError(f"state cache migration conflicts at {target}")
                moved.append(child.name)
                if not dry_run and not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(child), str(target))
            else:
                protected.append(child.name)
    return StateMigrationReport(
        source,
        destination,
        dry_run,
        copied,
        tuple(moved),
        tuple(protected),
        (),
    )


def _unresolved_runs(root: Path) -> tuple[str, ...]:
    if not root.is_dir():
        return ()
    unresolved: list[str] = []
    for run in sorted(path for path in root.iterdir() if path.is_dir() and not path.is_symlink()):
        submission = run / "submission.json"
        if not submission.is_file():
            continue
        reconciliation = run / "reconciliation.json"
        try:
            payload = json.loads(reconciliation.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            unresolved.append(run.name)
            continue
        record = payload.get("provider_record")
        state = record.get("state") if isinstance(record, dict) else None
        if payload.get("schema") != "posttrain.execution-reconciliation.v1" or state not in _TERMINAL_STATES:
            unresolved.append(run.name)
    return tuple(unresolved)


def _copy_executions(source: Path, destination: Path, *, dry_run: bool) -> int:
    if not source.is_dir():
        return 0
    copied = 0
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"execution state contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        target = destination / relative
        if target.exists():
            if not target.is_file() or target.read_bytes() != path.read_bytes():
                raise ContractError(f"execution state migration conflicts at {target}")
            continue
        copied += 1
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    return copied


def _same_tree(left: Path, right: Path) -> bool:
    if left.is_file() or right.is_file():
        return left.is_file() and right.is_file() and left.read_bytes() == right.read_bytes()
    left_files = {path.relative_to(left): path for path in left.rglob("*") if path.is_file()}
    right_files = {path.relative_to(right): path for path in right.rglob("*") if path.is_file()}
    return set(left_files) == set(right_files) and all(
        left_files[relative].read_bytes() == right_files[relative].read_bytes() for relative in left_files
    )
