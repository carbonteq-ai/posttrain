"""Create bounded, deterministic source snapshots for actual-job images."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from posttrain.common import ContractError

from .contracts import SourcePackage

_FORBIDDEN_NAMES = {
    ".env",
    ".git",
    ".netrc",
    "credentials",
    "id_dsa",
    "id_ed25519",
    "id_rsa",
    "secrets",
}
_FORBIDDEN_SUFFIXES = {".ckpt", ".gguf", ".safetensors"}
_IGNORED_NAMES = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}
_IGNORED_SUFFIXES = {".pyc", ".pyo"}


@dataclass(frozen=True, slots=True)
class SourceSnapshotRequest:
    """Explicit paths selected from one local source root."""

    root: Path
    includes: tuple[str, ...]
    install_roots: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.root.is_absolute() or not self.root.is_dir():
            raise ContractError("source snapshot root must be an existing absolute directory")
        includes = tuple(_relative_path(item, "source include") for item in self.includes)
        installs = tuple(_relative_path(item, "source install root") for item in self.install_roots)
        for label, values in (
            ("includes", includes),
            ("install roots", installs),
        ):
            if not values or len(set(values)) != len(values) or tuple(sorted(values)) != values:
                raise ContractError(f"source snapshot {label} must be non-empty, unique, and sorted")
        _reject_overlapping_includes(includes)


@dataclass(frozen=True, slots=True)
class MaterializedSourceSnapshot:
    package: SourcePackage
    digest: str
    created: bool


class ImmutableSourceSnapshotter:
    """Copy only declared source paths into a content-addressed local cache."""

    def __init__(
        self,
        *,
        cache_root: Path,
        max_files: int = 50_000,
        max_bytes: int = 2 * 1024 * 1024 * 1024,
    ) -> None:
        if not cache_root.is_absolute():
            raise ContractError("source snapshot cache root must be absolute")
        if max_files < 1 or max_bytes < 1:
            raise ContractError("source snapshot limits must be positive")
        self._cache_root = cache_root
        self._max_files = max_files
        self._max_bytes = max_bytes

    def materialize(
        self,
        request: SourceSnapshotRequest,
    ) -> MaterializedSourceSnapshot:
        self._cache_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".source-snapshot-", dir=self._cache_root))
        try:
            budget = _CopyBudget(self._max_files, self._max_bytes)
            for configured in request.includes:
                source = _selected_path(request.root, configured)
                destination = staging if configured == "." else staging.joinpath(*configured.split("/"))
                if configured == ".":
                    for child in sorted(
                        source.iterdir(),
                        key=lambda path: path.name,
                    ):
                        _copy_entry(child, staging / child.name, budget)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    _copy_entry(source, destination, budget)
            package = SourcePackage(staging, request.install_roots)
            digest = _digest_source_package(package)
            destination = self._cache_root / digest
            if destination.exists():
                existing = SourcePackage(destination, request.install_roots)
                if _digest_source_package(existing) != digest:
                    raise ContractError("retained source snapshot contains filesystem drift")
                return MaterializedSourceSnapshot(existing, digest, False)
            try:
                staging.replace(destination)
            except FileExistsError:
                existing = SourcePackage(destination, request.install_roots)
                if _digest_source_package(existing) != digest:
                    raise ContractError("retained source snapshot contains filesystem drift") from None
                return MaterializedSourceSnapshot(existing, digest, False)
            retained = SourcePackage(destination, request.install_roots)
            return MaterializedSourceSnapshot(retained, digest, True)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def inspect(self, request: SourceSnapshotRequest) -> str:
        """Return the staged-tree digest without creating cache or temporary files."""

        budget = _CopyBudget(self._max_files, self._max_bytes)
        entries: list[dict[str, object]] = []
        for configured in request.includes:
            source = _selected_path(request.root, configured)
            if configured == ".":
                for child in sorted(source.iterdir(), key=lambda path: path.name):
                    _inspect_entry(
                        child,
                        relative=child.relative_to(request.root).as_posix(),
                        budget=budget,
                        entries=entries,
                    )
            else:
                parent = PurePosixPath(configured).parent
                ancestors = tuple(reversed(parent.parents)) + (parent,)
                for ancestor in ancestors:
                    if ancestor.as_posix() != ".":
                        entries.append({"path": ancestor.as_posix(), "type": "directory"})
                _inspect_entry(
                    source,
                    relative=configured,
                    budget=budget,
                    entries=entries,
                )
        if not entries:
            raise ContractError("source snapshot cannot be empty")
        by_path = {str(entry["path"]): entry for entry in entries}
        return hashlib.sha256(
            json.dumps(
                {"entries": [by_path[path] for path in sorted(by_path, key=PurePosixPath)]},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()


def _digest_source_package(source: SourcePackage) -> str:
    """Delay the service import so planning can inspect a source without a cycle."""

    from .service import digest_source_package

    return digest_source_package(source)


@dataclass(slots=True)
class _CopyBudget:
    remaining_files: int
    remaining_bytes: int

    def consume(self, size: int) -> None:
        self.remaining_files -= 1
        self.remaining_bytes -= size
        if self.remaining_files < 0:
            raise ContractError("source snapshot exceeds its file-count limit")
        if self.remaining_bytes < 0:
            raise ContractError("source snapshot exceeds its byte limit")


def _selected_path(root: Path, configured: str) -> Path:
    selected = root if configured == "." else root.joinpath(*configured.split("/"))
    if not selected.exists():
        raise ContractError(f"source snapshot include does not exist: {configured}")
    resolved = selected.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ContractError("source snapshot include escapes its root")
    return selected


def _copy_entry(source: Path, destination: Path, budget: _CopyBudget) -> None:
    mode = source.lstat().st_mode
    name = source.name.lower()
    if name in _IGNORED_NAMES or source.suffix.lower() in _IGNORED_SUFFIXES:
        return
    if source.is_symlink():
        raise ContractError(f"source snapshots do not accept symlinks: {source}")
    if name in _FORBIDDEN_NAMES:
        raise ContractError(f"source snapshot contains a forbidden filename: {source.name}")
    if source.suffix.lower() in _FORBIDDEN_SUFFIXES or name in {
        "model.bin",
        "pytorch_model.bin",
    }:
        raise ContractError(f"source snapshot contains model weights: {source.name}")
    if stat.S_ISDIR(mode):
        destination.mkdir()
        for child in sorted(source.iterdir(), key=lambda path: path.name):
            _copy_entry(child, destination / child.name, budget)
        destination.chmod(0o755)
        os.utime(destination, (0, 0), follow_symlinks=False)
        return
    if not stat.S_ISREG(mode):
        raise ContractError(f"source snapshot contains a special file: {source.name}")
    size = source.stat().st_size
    budget.consume(size)
    shutil.copyfile(source, destination)
    destination.chmod(0o755 if mode & 0o111 else 0o644)
    os.utime(destination, (0, 0), follow_symlinks=False)


def _inspect_entry(
    source: Path,
    *,
    relative: str,
    budget: _CopyBudget,
    entries: list[dict[str, object]],
) -> None:
    mode = source.lstat().st_mode
    name = source.name.lower()
    if name in _IGNORED_NAMES or source.suffix.lower() in _IGNORED_SUFFIXES:
        return
    if source.is_symlink():
        raise ContractError(f"source snapshots do not accept symlinks: {source}")
    if name in _FORBIDDEN_NAMES:
        raise ContractError(f"source snapshot contains a forbidden filename: {source.name}")
    if source.suffix.lower() in _FORBIDDEN_SUFFIXES or name in {
        "model.bin",
        "pytorch_model.bin",
    }:
        raise ContractError(f"source snapshot contains model weights: {source.name}")
    if stat.S_ISDIR(mode):
        entries.append({"path": relative, "type": "directory"})
        for child in sorted(source.iterdir(), key=lambda path: path.name):
            _inspect_entry(
                child,
                relative=f"{relative}/{child.name}",
                budget=budget,
                entries=entries,
            )
        return
    if not stat.S_ISREG(mode):
        raise ContractError(f"source snapshot contains a special file: {source.name}")
    contents = source.read_bytes()
    budget.consume(len(contents))
    entries.append(
        {
            "path": relative,
            "type": "file",
            "sha256": hashlib.sha256(contents).hexdigest(),
            "executable": bool(mode & 0o111),
        }
    )


def _relative_path(value: str, label: str) -> str:
    if not value or value != value.strip() or "\\" in value:
        raise ContractError(f"{label} must be a normalized relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".."} for part in path.parts)
        or (path.as_posix() != "." and "." in path.parts)
        or path.as_posix() != value
    ):
        raise ContractError(f"{label} must remain inside its source root")
    if ".git" in path.parts:
        raise ContractError(f"{label} cannot select Git metadata")
    return value


def _reject_overlapping_includes(includes: tuple[str, ...]) -> None:
    paths = tuple(PurePosixPath(value) for value in includes)
    for index, path in enumerate(paths):
        for other in paths[index + 1 :]:
            if path == PurePosixPath(".") or other == PurePosixPath("."):
                raise ContractError("source snapshot root include cannot be combined with other paths")
            if path in other.parents or other in path.parents:
                raise ContractError("source snapshot includes cannot overlap")
