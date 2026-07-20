"""Source identity resolution owned by the application host."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GitSource:
    revision: str
    dirty: bool

    def metadata(self) -> dict[str, str | bool]:
        return {"git_revision": self.revision, "git_dirty": self.dirty}


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
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return GitSource(revision=revision, dirty=bool(status.strip()))
