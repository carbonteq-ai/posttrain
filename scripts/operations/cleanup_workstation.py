#!/usr/bin/env python3
"""Keep a Posttrain workstation lean: remove what is unused, keep a week only of what is costly to redo.

Preview by default; ``--apply`` removes. Removed right away once unused:

* ``worktrees``: extra checkouts of this repository with no changes, whose
  commit is merged into ``origin/main`` or on a remote branch, and in which no
  process is working. Run records a checkout holds are first copied into the
  main checkout with ``posttrain state migrate`` (which refuses while any run is
  unresolved, so a live run's launch checkout stays); a checkout is removed
  only after that succeeds.
* ``posttrain-images``: local Docker images of Posttrain jobs and the
  Observatory used by no container, except the two newest Observatory images.
* ``uv``: ``uv cache prune`` (entries no environment references).
* ``posttrain-cache``: ``posttrain cache prune`` of the main checkout.
* ``huggingface``: incomplete downloads.

Kept for the retention period (``--max-age-days``, default 7), because they are
costly to recreate or may be resumed:

* ``buildkit``: ``posttrain-builder`` cache entries unused for the period, and
  the cache capped at ``--buildkit-max-gb`` (the warm cache keeps builds fast).
* ``huggingface``: models and datasets not read for the period.
* ``scratch``: Claude session scratch directories untouched for the period.

Run evidence (runs, metrics, traces, checkpoints) is never touched here; it is
purged through ``posttrain run purge`` after review. Other projects' images,
containers and volumes on the same Docker daemon are never touched.
See ``docs/operations/cleanup-playbook.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

DAY = 24 * 60 * 60
CLASSES = ("worktrees", "posttrain-images", "buildkit", "uv", "huggingface", "posttrain-cache", "scratch")
POSTTRAIN_IMAGE_PREFIXES = ("posttrain-local", "posttrain-observatory", "registry.lan/carbonteq/posttrain-")
BUILDER = "posttrain-builder"

Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def run(argv: Sequence[str], *, cwd: Path | None = None, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), cwd=cwd, check=check, capture_output=True, text=True)


@dataclass
class Item:
    cls: str
    target: str
    action: str  # "remove", "keep" or "run"
    reason: str
    bytes: int = 0
    done: bool = False
    error: str | None = None
    detail: dict[str, object] = field(default_factory=dict)


@dataclass
class Context:
    repo: Path
    max_age_seconds: int
    now: float
    buildkit_max_gb: int
    apply: bool
    scratch_root: Path
    runner: Callable[..., subprocess.CompletedProcess[str]] = run
    processes_in: Callable[[Path], list[int]] = field(default_factory=lambda: processes_in)


# ---------------------------------------------------------------- helpers


def tree_bytes(path: Path) -> int:
    total = 0
    for root, dirs, files in os.walk(path, onerror=lambda _error: None):
        dirs[:] = [name for name in dirs if not os.path.islink(os.path.join(root, name))]
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def newest_mtime(path: Path, *, limit: int = 200_000) -> float:
    newest = path.lstat().st_mtime
    seen = 0
    for root, dirs, files in os.walk(path, onerror=lambda _error: None):
        dirs[:] = [name for name in dirs if name not in {".git", ".venv", "node_modules"}]
        for name in (*dirs, *files):
            seen += 1
            if seen > limit:
                return newest
            try:
                newest = max(newest, os.lstat(os.path.join(root, name)).st_mtime)
            except OSError:
                pass
    return newest


def human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


# ---------------------------------------------------------------- worktrees


def processes_in(path: Path) -> list[int]:
    """Process ids whose working directory is inside ``path`` (Linux ``/proc``)."""

    root = str(path.resolve())
    found = []
    for entry in Path("/proc").iterdir() if Path("/proc").is_dir() else ():
        if not entry.name.isdigit():
            continue
        try:
            cwd = os.readlink(entry / "cwd")
        except OSError:
            continue
        if cwd == root or cwd.startswith(root + os.sep):
            found.append(int(entry.name))
    return found


def main_checkout(path: Path, runner: Callable[..., subprocess.CompletedProcess[str]] = run) -> Path:
    """The repository's main checkout, from any of its worktrees."""

    common = runner(["git", "-C", str(path), "rev-parse", "--path-format=absolute", "--git-common-dir"])
    if common.returncode != 0:
        raise SystemExit(f"{path} is not inside a git checkout")
    return Path(common.stdout.strip()).parent.resolve()


def worktree_paths(ctx: Context) -> list[Path]:
    listed = ctx.runner(["git", "-C", str(ctx.repo), "worktree", "list", "--porcelain"])
    paths = [Path(line.split(" ", 1)[1]) for line in listed.stdout.splitlines() if line.startswith("worktree ")]
    return [path for path in paths if path.resolve() != ctx.repo]


def plan_worktrees(ctx: Context) -> list[Item]:
    items: list[Item] = []
    ctx.runner(["git", "-C", str(ctx.repo), "fetch", "--quiet", "origin"])
    for path in worktree_paths(ctx):
        target = str(path)
        if not path.exists():
            items.append(Item("worktrees", target, "remove", "directory is gone; prune the registration"))
            continue
        head = ctx.runner(["git", "-C", target, "rev-parse", "HEAD"]).stdout.strip()
        changes = ctx.runner(["git", "-C", target, "status", "--porcelain", "--untracked-files=normal"]).stdout.strip()
        if changes:
            items.append(Item("worktrees", target, "keep", "has uncommitted or untracked changes"))
            continue
        merged = (
            ctx.runner(["git", "-C", str(ctx.repo), "merge-base", "--is-ancestor", head, "origin/main"]).returncode == 0
        )
        on_remote = bool(ctx.runner(["git", "-C", str(ctx.repo), "branch", "-r", "--contains", head]).stdout.strip())
        if not (merged or on_remote):
            items.append(Item("worktrees", target, "keep", "commit is neither merged nor on a remote branch"))
            continue
        working = ctx.processes_in(path)
        if working:
            items.append(Item("worktrees", target, "keep", f"{len(working)} process(es) working in it"))
            continue
        records = path / "apps" / "lab" / ".posttrain" / "state" / "executions"
        has_records = records.is_dir() and any(records.iterdir())
        reason = "clean, " + ("merged" if merged else "pushed") + ", no process working in it"
        if has_records:
            reason += "; its run records are copied to the main checkout first"
        items.append(Item("worktrees", target, "remove", reason, tree_bytes(path), detail={"records": has_records}))
    return items


def apply_worktree(ctx: Context, item: Item) -> None:
    path = Path(item.target)
    if path.exists() and item.detail.get("records"):
        main_lab = ctx.repo / "apps" / "lab"
        migrate = ctx.runner(
            [
                "uv",
                "run",
                "--no-sync",
                "posttrain",
                "--project-root",
                str(main_lab),
                "state",
                "migrate",
                "--from-project-root",
                str(path / "apps" / "lab"),
            ],
            cwd=ctx.repo,
        )
        if migrate.returncode != 0:
            raise RuntimeError("run records not migrated: " + (migrate.stderr or migrate.stdout).strip()[-300:])
        source = path / "apps" / "lab" / ".posttrain" / "state" / "executions"
        target = main_lab / ".posttrain" / "state" / "executions"
        missing = sorted(child.name for child in source.iterdir() if not (target / child.name).exists())
        if missing:
            raise RuntimeError(f"run records missing from the main checkout after migration: {missing[:5]}")
    if path.exists():
        removed = ctx.runner(["git", "-C", str(ctx.repo), "worktree", "remove", str(path)])
        if removed.returncode != 0:
            raise RuntimeError(removed.stderr.strip()[-300:])
    ctx.runner(["git", "-C", str(ctx.repo), "worktree", "prune"])


# ---------------------------------------------------------------- docker


def plan_posttrain_images(ctx: Context) -> list[Item]:
    if shutil.which("docker") is None:
        return []
    listed = ctx.runner(["docker", "image", "ls", "--format", "{{json .}}"])
    used = set(ctx.runner(["docker", "ps", "-a", "--format", "{{.Image}}"]).stdout.split())
    used_ids = set(ctx.runner(["docker", "ps", "-a", "--format", "{{.ImageID}}"]).stdout.split())
    images = []
    for line in listed.stdout.splitlines():
        entry = json.loads(line)
        name = f"{entry['Repository']}:{entry['Tag']}"
        if not entry["Repository"].startswith(POSTTRAIN_IMAGE_PREFIXES) or entry["Tag"] == "<none>":
            continue
        inspected = ctx.runner(["docker", "image", "inspect", "--format", "{{.Created}}|{{.Id}}|{{.Size}}", name])
        created, image_id, size = inspected.stdout.strip().split("|")
        images.append((name, entry["Repository"], created, image_id, int(size)))
    newest_observatory = {
        name
        for name, *_ in sorted(
            (image for image in images if image[1].endswith("posttrain-observatory")), key=lambda image: image[2]
        )[-2:]
    }
    items = []
    for name, _repository, created, image_id, size in images:
        if name in used or image_id in used_ids or image_id.removeprefix("sha256:")[:12] in used_ids:
            items.append(Item("posttrain-images", name, "keep", "used by a container"))
        elif name in newest_observatory:
            items.append(Item("posttrain-images", name, "keep", "one of the two newest Observatory images"))
        else:
            age = int((ctx.now - _docker_time(created)) / DAY)
            items.append(Item("posttrain-images", name, "remove", f"used by no container (built {age} days ago)", size))
    return items


def _docker_time(value: str) -> float:
    from datetime import datetime

    text = value.strip()
    if "." in text:
        head, _, tail = text.partition(".")
        digits = "".join(ch for ch in tail if ch.isdigit())[:6]
        zone = tail[len("".join(ch for ch in tail if ch.isdigit())) :]
        text = f"{head}.{digits}{zone}"
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def apply_posttrain_image(ctx: Context, item: Item) -> None:
    removed = ctx.runner(["docker", "image", "rm", item.target])
    if removed.returncode != 0:
        raise RuntimeError(removed.stderr.strip()[-300:])


def plan_buildkit(ctx: Context) -> list[Item]:
    if shutil.which("docker") is None or ctx.runner(["docker", "buildx", "inspect", BUILDER]).returncode != 0:
        return []
    hours = ctx.max_age_seconds // 3600
    old = ctx.runner(["docker", "buildx", "du", "--builder", BUILDER, "--filter", f"until={hours}h"])
    reclaimable = _buildx_total(old.stdout)
    total = _buildx_total(ctx.runner(["docker", "buildx", "du", "--builder", BUILDER]).stdout)
    over_cap = max(0, total - reclaimable - ctx.buildkit_max_gb * 1024**3)
    return [
        Item(
            "buildkit",
            BUILDER,
            "run",
            f"entries unused for {hours // 24} days, then cap the cache at {ctx.buildkit_max_gb} GB "
            f"(cache {human(total)})",
            reclaimable + over_cap,
            detail={"hours": hours},
        )
    ]


def _buildx_total(text: str) -> int:
    for line in reversed(text.splitlines()):
        if line.strip().lower().startswith("total:"):
            return _parse_size(line.split(":", 1)[1].strip())
    return 0


def _parse_size(text: str) -> int:
    units = {
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
        "TIB": 1024**4,
    }
    number = "".join(ch for ch in text if ch.isdigit() or ch == ".")
    unit = text[len(number) :].strip().upper() or "B"
    return int(float(number or 0) * units.get(unit, 1))


def apply_buildkit(ctx: Context, item: Item) -> None:
    hours = item.detail["hours"]
    for argv in (
        ["docker", "buildx", "prune", "--builder", BUILDER, "--force", "--filter", f"until={hours}h"],
        ["docker", "buildx", "prune", "--builder", BUILDER, "--force", "--max-used-space", f"{ctx.buildkit_max_gb}GB"],
    ):
        result = ctx.runner(argv)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip()[-300:])


# ---------------------------------------------------------------- uv, huggingface, posttrain cache


def plan_uv(ctx: Context) -> list[Item]:
    if shutil.which("uv") is None:
        return []
    size = ctx.runner(["uv", "cache", "size", "--human"]).stdout.strip()
    return [Item("uv", "uv cache", "run", f"uv cache prune (removes entries no environment references; cache {size})")]


def apply_uv(ctx: Context, item: Item) -> None:
    result = ctx.runner(["uv", "cache", "prune"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[-300:])


def plan_huggingface(ctx: Context) -> list[Item]:
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return [Item("huggingface", "huggingface cache", "keep", "huggingface_hub is not installed here")]
    info = scan_cache_dir()
    items = []
    incomplete = int(getattr(info, "incomplete_size_on_disk", 0) or 0)
    if incomplete:
        items.append(Item("huggingface", "incomplete downloads", "remove", "incomplete downloads", incomplete))
    for repo in sorted(info.repos, key=lambda repo: -repo.size_on_disk):
        idle = ctx.now - max(repo.last_accessed, repo.last_modified)
        target = f"{repo.repo_type}:{repo.repo_id}"
        if idle < ctx.max_age_seconds:
            continue
        items.append(
            Item(
                "huggingface",
                target,
                "remove",
                f"not read for {int(idle / DAY)} days",
                repo.size_on_disk,
                detail={"revisions": sorted(revision.commit_hash for revision in repo.revisions)},
            )
        )
    return items


def apply_huggingface(ctx: Context, item: Item) -> None:
    from huggingface_hub import scan_cache_dir

    info = scan_cache_dir()
    if item.target == "incomplete downloads":
        for path in getattr(info, "incomplete_files", ()):
            Path(path).unlink(missing_ok=True)
        return
    revisions = item.detail["revisions"]
    assert isinstance(revisions, list)
    info.delete_revisions(*(str(revision) for revision in revisions)).execute()


def plan_posttrain_cache(ctx: Context) -> list[Item]:
    lab = ctx.repo / "apps" / "lab"
    if not (lab / ".posttrain").is_dir():
        return []
    report = ctx.runner(
        ["uv", "run", "--no-sync", "posttrain", "--project-root", str(lab), "--json", "cache", "prune"], cwd=ctx.repo
    )
    reclaimable = 0
    try:
        payload = json.loads(report.stdout)
        reclaimable = int(payload.get("reclaimable_bytes") or 0)
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        pass
    return [
        Item(
            "posttrain-cache",
            str(lab / ".posttrain" / "state" / "cache"),
            "run",
            "posttrain cache prune (keeps leased, unpublished and durable material)",
            reclaimable,
        )
    ]


def apply_posttrain_cache(ctx: Context, item: Item) -> None:
    lab = ctx.repo / "apps" / "lab"
    result = ctx.runner(
        ["uv", "run", "--no-sync", "posttrain", "--project-root", str(lab), "cache", "prune", "--apply"], cwd=ctx.repo
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip()[-300:])


# ---------------------------------------------------------------- scratch


def plan_scratch(ctx: Context) -> list[Item]:
    items = []
    if not ctx.scratch_root.is_dir():
        return items
    for project in sorted(ctx.scratch_root.iterdir()):
        if not project.is_dir() or project.is_symlink() or not project.name.startswith("-"):
            continue
        for session in sorted(project.iterdir()):
            if not session.is_dir() or session.is_symlink():
                continue
            idle = ctx.now - newest_mtime(session)
            if idle >= ctx.max_age_seconds:
                items.append(
                    Item(
                        "scratch", str(session), "remove", f"untouched for {int(idle / DAY)} days", tree_bytes(session)
                    )
                )
    return items


def apply_scratch(ctx: Context, item: Item) -> None:
    shutil.rmtree(item.target)


# ---------------------------------------------------------------- driver

PLANNERS: dict[str, tuple[Callable[[Context], list[Item]], Callable[[Context, Item], None]]] = {
    "worktrees": (plan_worktrees, apply_worktree),
    "posttrain-images": (plan_posttrain_images, apply_posttrain_image),
    "buildkit": (plan_buildkit, apply_buildkit),
    "uv": (plan_uv, apply_uv),
    "huggingface": (plan_huggingface, apply_huggingface),
    "posttrain-cache": (plan_posttrain_cache, apply_posttrain_cache),
    "scratch": (plan_scratch, apply_scratch),
}


def execute(ctx: Context, classes: Iterable[str]) -> list[Item]:
    items: list[Item] = []
    for name in classes:
        planner, applier = PLANNERS[name]
        try:
            planned = planner(ctx)
        except Exception as error:  # one broken class never stops the others
            planned = [Item(name, name, "keep", "could not be planned", error=f"{type(error).__name__}: {error}")]
        for item in planned:
            if ctx.apply and item.action in {"remove", "run"} and item.error is None:
                try:
                    applier(ctx, item)
                    item.done = True
                except Exception as error:
                    item.error = f"{type(error).__name__}: {error}"
            items.append(item)
    return items


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--apply", action="store_true", help="remove; the default is a preview")
    parser.add_argument(
        "--max-age-days", type=float, default=7.0, help="retention for the build cache, models and scratch (default 7)"
    )
    parser.add_argument("--only", action="append", choices=CLASSES, help="limit to these classes (repeatable)")
    parser.add_argument("--buildkit-max-gb", type=int, default=80, help="cap for the posttrain-builder cache")
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="any checkout of the repository; its main checkout is used",
    )
    parser.add_argument("--scratch-root", type=Path, default=Path(f"/tmp/claude-{os.getuid()}"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.max_age_days < 1:
        parser.error("--max-age-days must be at least 1")
    ctx = Context(
        repo=main_checkout(args.repo.resolve()),
        max_age_seconds=int(args.max_age_days * DAY),
        now=time.time(),
        buildkit_max_gb=args.buildkit_max_gb,
        apply=args.apply,
        scratch_root=args.scratch_root,
    )
    items = execute(ctx, args.only or CLASSES)
    if args.json:
        print(json.dumps([asdict(item) for item in items], indent=2))
    else:
        mode = "applied" if args.apply else "preview"
        print(f"cleanup {mode}: retention {args.max_age_days:g} days, repository {ctx.repo}")
        for item in items:
            state = "error" if item.error else ("done" if item.done else item.action)
            size = f" {human(item.bytes)}" if item.bytes else ""
            print(
                f"  [{item.cls}] {state:6} {item.target}{size}: {item.reason}"
                + (f" ({item.error})" if item.error else "")
            )
        planned = sum(item.bytes for item in items if item.action in {"remove", "run"} and not item.error)
        print(f"reclaimable (estimated): {human(planned)}")
    return 1 if any(item.error for item in items if item.action != "keep") else 0


if __name__ == "__main__":
    sys.exit(main())
