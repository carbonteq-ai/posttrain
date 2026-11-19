"""The workstation cleanup keeps what is in use and removes only old, recoverable material."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "cleanup_workstation", Path(__file__).resolve().parents[1] / "cleanup_workstation.py"
)
assert _SPEC is not None and _SPEC.loader is not None
cleanup = importlib.util.module_from_spec(_SPEC)
sys.modules["cleanup_workstation"] = cleanup
_SPEC.loader.exec_module(cleanup)

WEEK = 7 * cleanup.DAY


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _age(path: Path, seconds: float) -> None:
    stamp = time.time() - seconds
    for root, dirs, files in os.walk(path):
        for name in (*dirs, *files):
            os.utime(os.path.join(root, name), (stamp, stamp), follow_symlinks=False)
    os.utime(path, (stamp, stamp))


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    remote = tmp_path / "remote.git"
    _git("init", "--bare", "--initial-branch=main", str(remote), cwd=tmp_path)
    main = tmp_path / "main"
    _git("init", "--initial-branch=main", str(main), cwd=tmp_path)
    for key, value in (("user.email", "t@example.com"), ("user.name", "t"), ("commit.gpgsign", "false")):
        _git("config", key, value, cwd=main)
    (main / "README").write_text("x\n")
    _git("add", ".", cwd=main)
    old = "2026-01-01T00:00:00+00:00"
    env = {**os.environ, "GIT_AUTHOR_DATE": old, "GIT_COMMITTER_DATE": old}
    subprocess.run(["git", "commit", "-qm", "base"], cwd=main, check=True, env=env)
    _git("remote", "add", "origin", str(remote), cwd=main)
    _git("push", "-q", "origin", "main", cwd=main)
    return main


def _context(repo: Path, tmp_path: Path, *, apply: bool = False) -> Any:
    return cleanup.Context(
        repo=cleanup.main_checkout(repo),
        max_age_seconds=WEEK,
        now=time.time(),
        buildkit_max_gb=80,
        apply=apply,
        scratch_root=tmp_path / "scratch",
    )


def test_clean_merged_worktrees_go_right_away_and_everything_else_stays(repository: Path, tmp_path: Path) -> None:
    done = tmp_path / "done"
    busy = tmp_path / "busy"
    dirty = tmp_path / "dirty"
    local = tmp_path / "local-only"
    for path in (done, busy, dirty):
        _git("worktree", "add", "-q", "--detach", str(path), "main", cwd=repository)
    _git("worktree", "add", "-q", "-b", "feature", str(local), "main", cwd=repository)
    (local / "new").write_text("y\n")
    _git("add", ".", cwd=local)
    _git("commit", "-qm", "local only", cwd=local)
    (dirty / "README").write_text("changed\n")

    context = _context(repository, tmp_path)
    context.processes_in = lambda path: [4242] if path.name == "busy" else []
    items = {Path(item.target).name: item for item in cleanup.plan_worktrees(context)}
    assert items["done"].action == "remove"
    assert items["busy"].action == "keep" and "process" in items["busy"].reason
    assert items["dirty"].action == "keep" and "changes" in items["dirty"].reason
    assert items["local-only"].action == "keep" and "remote" in items["local-only"].reason

    context.apply = True
    applied = cleanup.execute(context, ["worktrees"])
    assert [Path(item.target).name for item in applied if item.done] == ["done"]
    assert not done.exists() and busy.exists() and dirty.exists() and local.exists()


def test_a_worktree_whose_run_records_cannot_move_is_kept(repository: Path, tmp_path: Path) -> None:
    holder = tmp_path / "holder"
    _git("worktree", "add", "-q", "--detach", str(holder), "main", cwd=repository)
    records = holder / "apps" / "lab" / ".posttrain" / "state" / "executions" / "run-1"
    records.mkdir(parents=True)
    _age(holder, 10 * cleanup.DAY)
    context = _context(repository, tmp_path, apply=True)

    def refusing(argv, **kwargs):
        if "migrate" in argv:
            return subprocess.CompletedProcess(argv, 1, "", "state migration refuses unresolved executions: run-1")
        return cleanup.run(argv, **kwargs)

    context.runner = refusing
    item = cleanup.Item("worktrees", str(holder), "remove", "old", detail={"records": True})
    with pytest.raises(RuntimeError, match="unresolved"):
        cleanup.apply_worktree(context, item)
    assert holder.exists()


def test_scratch_sessions_untouched_for_a_week_are_removed(tmp_path: Path, repository: Path) -> None:
    project = tmp_path / "scratch" / "-home-user-project"
    old = project / "old-session" / "scratchpad"
    new = project / "new-session" / "scratchpad"
    for path in (old, new):
        path.mkdir(parents=True)
        (path / "notes.txt").write_text("n")
    _age(project / "old-session", 9 * cleanup.DAY)
    items = cleanup.execute(_context(repository, tmp_path, apply=True), ["scratch"])
    assert [Path(item.target).name for item in items] == ["old-session"] and items[0].done
    assert not (project / "old-session").exists() and new.exists()


def test_docker_and_buildx_output_is_parsed() -> None:
    assert cleanup._docker_time("2026-09-20T10:00:00.123456789Z") == pytest.approx(
        cleanup._docker_time("2026-09-20T10:00:00.123456Z")
    )
    assert (
        cleanup._buildx_total("ID\tRECLAIMABLE\tSIZE\nabc\ttrue\t1GB\nReclaimable:\t2.5GB\nTotal:\t\t3GB\n")
        == 3 * 1000**3
    )
    assert cleanup._parse_size("512MiB") == 512 * 1024**2
