"""Source identity resolution for the qualification project entry."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GitSource:
    revision: str
    dirty: bool
    dirty_digest: str | None = None

    def metadata(self) -> dict[str, str | bool]:
        values: dict[str, str | bool] = {"git_revision": self.revision, "git_dirty": self.dirty}
        if self.dirty_digest is not None:
            values["git_dirty_digest"] = self.dirty_digest
        return values


def resolve_git_source(path: Path) -> GitSource:
    """Resolve the checked-out revision without leaking repository paths to packages."""

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    dirty = bool(status.strip())
    return GitSource(revision=revision, dirty=dirty, dirty_digest=_dirty_digest(path) if dirty else None)


def _dirty_digest(path: Path) -> str:
    digest = hashlib.sha256()
    diff = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"],
        cwd=path,
        check=True,
        capture_output=True,
    ).stdout
    digest.update(diff)
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=path,
        check=True,
        capture_output=True,
    ).stdout
    for encoded in sorted(item for item in untracked.split(b"\0") if item):
        relative = encoded.decode("utf-8", errors="surrogateescape")
        digest.update(b"\0untracked\0")
        digest.update(encoded)
        file_path = path / relative
        if file_path.is_file():
            digest.update(file_path.read_bytes())
    return digest.hexdigest()
