"""Materialize immutable Git sources for framework-owned job images."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from posttrain.common import ContractError
from posttrain.execution_pack import GitSourceRequest

_SCHEMA = "posttrain.git-source-lock.v1"


class GitGateway(Protocol):
    """Small non-shell boundary around Git, suitable for deterministic fakes."""

    def invoke(self, arguments: Sequence[str]) -> str: ...


class GitCli:
    """Invoke Git without a command shell or interpolated command text."""

    def __init__(self, executable: str = "git") -> None:
        self._executable = executable

    def invoke(self, arguments: Sequence[str]) -> str:
        result = subprocess.run(
            [self._executable, *arguments],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr.strip() or result.stdout.strip() or "no diagnostic")[-3000:]
            raise RuntimeError(f"git {arguments[0] if arguments else 'command'} failed: {detail}")
        return result.stdout


@dataclass(frozen=True, slots=True)
class LockedGitSubdirectory:
    path: str
    tree_digest: str

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "tree_digest": self.tree_digest}


@dataclass(frozen=True, slots=True)
class LockedGitSource:
    repository: str
    revision: str
    source_tree_digest: str
    subdirectories: tuple[LockedGitSubdirectory, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "repository": self.repository,
            "revision": self.revision,
            "source_tree_digest": self.source_tree_digest,
            "subdirectories": [item.as_dict() for item in self.subdirectories],
        }


@dataclass(frozen=True, slots=True)
class GitSourceLock:
    """Secret-free deterministic source identity retained with a job capsule."""

    sources: tuple[LockedGitSource, ...]
    schema: str = _SCHEMA

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "sources": [source.as_dict() for source in self.sources],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class MaterializedGitSource:
    """A local checkout paired with its portable lock entry."""

    root: Path
    lock: LockedGitSource


@dataclass(frozen=True, slots=True)
class MaterializedGitSources:
    sources: tuple[MaterializedGitSource, ...]
    lock: GitSourceLock


class ImmutableGitSourcePacker:
    """Fetch, verify, cache, and lock immutable sources used by a job image."""

    def __init__(
        self,
        *,
        cache_root: Path,
        gateway: GitGateway | None = None,
    ) -> None:
        if not cache_root.is_absolute():
            raise ContractError("Git source cache root must be absolute")
        self._cache_root = cache_root
        self._gateway = gateway or GitCli()

    def materialize(self, requests: Sequence[GitSourceRequest]) -> MaterializedGitSources:
        if not requests:
            raise ContractError("at least one Git source is required")

        revisions_by_repository: dict[str, set[str]] = {}
        for request in requests:
            revisions_by_repository.setdefault(request.repository, set()).add(request.revision)
        conflicting = sorted(
            repository for repository, revisions in revisions_by_repository.items() if len(revisions) > 1
        )
        if conflicting:
            raise ContractError(
                f"one job package cannot select multiple revisions of the same Git repository: {', '.join(conflicting)}"
            )

        selections: dict[tuple[str, str], set[str]] = {}
        for request in requests:
            selections.setdefault((request.repository, request.revision), set()).update(request.subdirectories)

        materialized: list[MaterializedGitSource] = []
        for (repository, revision), subdirectories in sorted(selections.items()):
            root = self._cache_root / _source_cache_key(repository, revision)
            if root.exists():
                self._verify_checkout(root, revision)
            else:
                self._fetch_checkout(root, repository, revision)
            lock = self._lock_source(
                root,
                repository=repository,
                revision=revision,
                subdirectories=sorted(subdirectories),
            )
            materialized.append(MaterializedGitSource(root=root, lock=lock))

        sources = tuple(materialized)
        return MaterializedGitSources(
            sources=sources,
            lock=GitSourceLock(sources=tuple(source.lock for source in sources)),
        )

    def _fetch_checkout(self, destination: Path, repository: str, revision: str) -> None:
        self._cache_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=self._cache_root))
        try:
            self._gateway.invoke(["init", "--quiet", str(temporary)])
            self._gateway.invoke(["-C", str(temporary), "remote", "add", "origin", repository])
            self._gateway.invoke(
                [
                    "-C",
                    str(temporary),
                    "fetch",
                    "--quiet",
                    "--depth=1",
                    "origin",
                    revision,
                ]
            )
            self._gateway.invoke(
                [
                    "-C",
                    str(temporary),
                    "checkout",
                    "--quiet",
                    "--detach",
                    "--force",
                    "FETCH_HEAD",
                ]
            )
            self._verify_checkout(temporary, revision)
            if destination.exists():
                self._verify_checkout(destination, revision)
            else:
                temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def _verify_checkout(self, root: Path, revision: str) -> None:
        if not root.is_absolute() or not root.is_dir():
            raise ContractError("Git source cache entry must be an absolute directory")
        observed = self._gateway.invoke(["-C", str(root), "rev-parse", "--verify", "HEAD"]).strip()
        if observed != revision:
            raise ContractError(f"Git source HEAD mismatch: expected {revision}, observed {observed or 'none'}")
        status = self._gateway.invoke(
            [
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignored=matching",
            ]
        ).strip()
        if status:
            raise ContractError("Git source cache entry contains dirty filesystem drift")
        _tree_digest(root)

    def _lock_source(
        self,
        root: Path,
        *,
        repository: str,
        revision: str,
        subdirectories: Sequence[str],
    ) -> LockedGitSource:
        if (root / ".gitmodules").exists():
            raise ContractError("Git sources with submodules require an explicit immutable submodule lock")
        locked: list[LockedGitSubdirectory] = []
        for configured in subdirectories:
            selected = root if configured == "." else root.joinpath(*configured.split("/"))
            if not selected.is_dir():
                raise ContractError(f"Git source subdirectory does not exist: {configured}")
            resolved = selected.resolve()
            if not resolved.is_relative_to(root.resolve()):
                raise ContractError("Git source subdirectory escapes its checkout")
            locked.append(
                LockedGitSubdirectory(
                    path=configured,
                    tree_digest=_tree_digest(selected),
                )
            )
        return LockedGitSource(
            repository=repository,
            revision=revision,
            source_tree_digest=_tree_digest(root),
            subdirectories=tuple(locked),
        )


def _source_cache_key(repository: str, revision: str) -> str:
    return hashlib.sha256(f"{repository}\0{revision}".encode()).hexdigest()


def _tree_digest(root: Path) -> str:
    entries: list[dict[str, object]] = []

    def visit(directory: Path, relative: PurePosixPath) -> None:
        with os.scandir(directory) as iterator:
            children = sorted(iterator, key=lambda entry: entry.name)
        for child in children:
            if relative == PurePosixPath(".") and child.name == ".git":
                continue
            child_relative = PurePosixPath(child.name) if relative == PurePosixPath(".") else relative / child.name
            if child.is_symlink():
                raise ContractError(f"Git sources do not accept symlinks: {child_relative.as_posix()}")
            if child.is_dir(follow_symlinks=False):
                entries.append({"path": child_relative.as_posix(), "type": "directory"})
                visit(Path(child.path), child_relative)
            elif child.is_file(follow_symlinks=False):
                path = Path(child.path)
                entries.append(
                    {
                        "path": child_relative.as_posix(),
                        "type": "file",
                        "sha256": _file_digest(path),
                        "executable": bool(path.stat().st_mode & 0o111),
                    }
                )
            else:
                raise ContractError(
                    f"Git sources only accept regular files and directories: {child_relative.as_posix()}"
                )

    visit(root, PurePosixPath("."))
    if not entries:
        raise ContractError("Git source tree cannot be empty")
    # Match execution-pack's canonical source-tree identity. Planning and
    # building live in separate packages so neither side may use a different
    # JSON envelope for the same file tree.
    return hashlib.sha256(
        json.dumps({"entries": entries}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
