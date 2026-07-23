"""Tests for reproducible host source identity."""

from __future__ import annotations

import subprocess
from pathlib import Path

from posttrain_lab.source import resolve_git_source


def _git(repository: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repository, check=True, capture_output=True)


def test_dirty_source_digest_covers_tracked_and_untracked_content(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "tests@example.com")
    _git(tmp_path, "config", "user.name", "Tests")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-qm", "base")

    clean = resolve_git_source(tmp_path)
    assert clean.dirty is False
    assert clean.dirty_digest is None

    tracked.write_text("changed\n", encoding="utf-8")
    untracked = tmp_path / "untracked.txt"
    untracked.write_text("one\n", encoding="utf-8")
    dirty = resolve_git_source(tmp_path)
    assert dirty.dirty is True
    assert dirty.dirty_digest is not None and len(dirty.dirty_digest) == 64
    assert resolve_git_source(tmp_path).dirty_digest == dirty.dirty_digest

    untracked.write_text("two\n", encoding="utf-8")
    assert resolve_git_source(tmp_path).dirty_digest != dirty.dirty_digest
