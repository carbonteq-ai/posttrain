"""Read-only repository ownership and documentation audit.

This module intentionally separates inspection from policy enforcement.  The
0.3.0 migration needs a stable inventory before its root-level compatibility
surfaces can be removed, so callers receive findings instead of an exception
when the current checkout still contains legacy paths.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

# The framework workspace may have only these reviewed root surfaces.  Entries
# absent from this set are findings, rather than being silently legitimised by
# their presence in the current checkout.  The migration deliberately leaves
# ``.posttrain``, ``scripts``, ``examples``, ``ops``, and ``.agents`` outside
# this list until each has an explicit owner or has been removed.
REVIEWED_ROOT_ENTRIES = frozenset(
    {
        ".github",
        ".gitignore",
        ".dockerignore",
        "AGENTS.md",
        "CHANGELOG.md",
        "COMPATIBILITY.md",
        "LICENSE",
        "NOTICE",
        "README.md",
        "SECURITY.md",
        "UPGRADING.md",
        "apps",
        "docs",
        "environments",
        "mise.toml",
        "packages",
        "pyproject.toml",
        "release",
        "tests",
        "tools",
        "uv.lock",
    }
)

_INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(\s*(<[^>]+>|[^\s)]+)")
_FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class BrokenMarkdownLink:
    """A relative Markdown-to-Markdown link whose target is absent."""

    source: Path
    target: str


@dataclass(frozen=True)
class RepositoryAudit:
    """A deterministic snapshot of ownership findings for one checkout."""

    unreviewed_root_entries: tuple[str, ...]
    tracked_ignored_paths: tuple[Path, ...]
    broken_markdown_links: tuple[BrokenMarkdownLink, ...]

    @property
    def is_clean(self) -> bool:
        return not (self.unreviewed_root_entries or self.tracked_ignored_paths or self.broken_markdown_links)

    def render(self) -> str:
        """Render a human-readable, intentionally non-failing report."""
        lines = [
            "repository ownership audit (report-only)",
            f"unreviewed root entries: {len(self.unreviewed_root_entries)}",
            f"tracked ignored paths: {len(self.tracked_ignored_paths)}",
            f"broken local Markdown links: {len(self.broken_markdown_links)}",
        ]
        if self.unreviewed_root_entries:
            lines.append("\nunreviewed root entries:")
            lines.extend(f"  - {entry}" for entry in self.unreviewed_root_entries)
        if self.tracked_ignored_paths:
            lines.append("\ntracked ignored paths:")
            lines.extend(f"  - {path.as_posix()}" for path in self.tracked_ignored_paths)
        if self.broken_markdown_links:
            lines.append("\nbroken local Markdown links:")
            lines.extend(f"  - {link.source.as_posix()} -> {link.target}" for link in self.broken_markdown_links)
        return "\n".join(lines)


def inspect_repository(repository_root: Path) -> RepositoryAudit:
    """Inspect a Git checkout without changing it.

    Git evaluates ignore patterns because reimplementing its pattern language
    would make the result subtly disagree with the repository users actually
    commit from.  All categorisation after that boundary is pure Python.
    """
    root = repository_root.resolve()
    tracked_paths = _tracked_paths(root)
    ignored_paths = _tracked_ignored_paths(root, tracked_paths)
    markdown_documents = {
        path: (root / path).read_text(encoding="utf-8")
        for path in tracked_paths
        if path.suffix.lower() == ".md" and (root / path).is_file()
    }
    return evaluate_repository(
        repository_root=root,
        root_entries=(entry.name for entry in root.iterdir() if entry.name != ".git"),
        tracked_ignored_paths=ignored_paths,
        markdown_documents=markdown_documents,
    )


def evaluate_repository(
    *,
    repository_root: Path,
    root_entries: Iterable[str],
    tracked_ignored_paths: Iterable[Path | str],
    markdown_documents: dict[Path, str],
) -> RepositoryAudit:
    """Classify supplied repository facts without invoking Git or writing files.

    The split makes the policy testable with tiny fixtures and lets a future CI
    adapter supply an immutable checkout snapshot instead of a live worktree.
    """
    root = repository_root.resolve()
    unreviewed = tuple(sorted(set(root_entries) - REVIEWED_ROOT_ENTRIES))
    ignored = tuple(sorted((Path(path) for path in tracked_ignored_paths), key=lambda path: path.as_posix()))
    broken_links = tuple(
        sorted(
            _broken_markdown_links(root, markdown_documents),
            key=lambda link: (link.source.as_posix(), link.target),
        )
    )
    return RepositoryAudit(unreviewed, ignored, broken_links)


def _tracked_paths(repository_root: Path) -> tuple[Path, ...]:
    result = _git(repository_root, "ls-files", "-z")
    if result.returncode:
        raise ValueError(f"not a Git checkout: {repository_root}")
    return tuple(Path(path.decode("utf-8")) for path in result.stdout.split(b"\0") if path)


def _tracked_ignored_paths(repository_root: Path, tracked_paths: tuple[Path, ...]) -> tuple[Path, ...]:
    if not tracked_paths:
        return ()
    stdin = b"\0".join(path.as_posix().encode("utf-8") for path in tracked_paths) + b"\0"
    result = _git(repository_root, "check-ignore", "--no-index", "-z", "--stdin", input=stdin)
    if result.returncode not in {0, 1}:
        raise ValueError(result.stderr.decode("utf-8").strip() or "git check-ignore failed")
    return tuple(Path(path.decode("utf-8")) for path in result.stdout.split(b"\0") if path)


def _git(repository_root: Path, *arguments: str, input: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        input=input,
        capture_output=True,
        check=False,
    )


def _broken_markdown_links(
    repository_root: Path,
    markdown_documents: dict[Path, str],
) -> list[BrokenMarkdownLink]:
    broken: list[BrokenMarkdownLink] = []
    for source, text in markdown_documents.items():
        for target in _markdown_targets(text):
            candidate = _local_markdown_target(repository_root, source, target)
            if candidate is None:
                continue
            try:
                candidate.relative_to(repository_root)
            except ValueError:
                broken.append(BrokenMarkdownLink(source, target))
                continue
            if not candidate.is_file():
                broken.append(BrokenMarkdownLink(source, target))
    return broken


def _markdown_targets(text: str) -> tuple[str, ...]:
    """Return inline link destinations while ignoring fenced examples."""
    outside_fence: list[str] = []
    fenced = False
    for line in text.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
            continue
        if not fenced:
            outside_fence.append(line)
    return tuple(match.group(1).strip("<>") for match in _INLINE_LINK.finditer("\n".join(outside_fence)))


def _local_markdown_target(repository_root: Path, source: Path, target: str) -> Path | None:
    """Return an in-tree target only for a relative link to a Markdown file."""
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or target.startswith("/") or not parsed.path:
        return None
    path = unquote(parsed.path)
    if Path(path).suffix.lower() != ".md":
        return None
    return (repository_root / source.parent / path).resolve()


__all__ = [
    "BrokenMarkdownLink",
    "REVIEWED_ROOT_ENTRIES",
    "RepositoryAudit",
    "evaluate_repository",
    "inspect_repository",
]
