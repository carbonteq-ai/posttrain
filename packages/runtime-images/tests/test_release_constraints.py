"""The release constraints must name the same sources the packages require."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_CONSTRAINTS = _ROOT / "release" / "github-constraints.txt"

# Workspace members are what the released wheelhouse installs, so every direct
# URL they require must be constrained. Sources outside the workspace, such as
# `environments/`, may also appear in the constraints but are not required to.
_WORKSPACE_GLOBS = ("apps/*/pyproject.toml", "packages/*/pyproject.toml")

_PIN = re.compile(r"^([A-Za-z0-9._-]+)\s*@\s*git\+\S+@([0-9a-f]{40})")


def _pins(text: str) -> dict[str, str]:
    found = {}
    for line in text.splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        match = _PIN.match(entry)
        if match:
            found[match.group(1)] = match.group(2)
    return found


def _workspace_requirements() -> dict[str, dict[Path, str]]:
    """Map each direct-URL requirement to the members requiring it, by commit."""
    required: dict[str, dict[Path, str]] = {}
    for glob in _WORKSPACE_GLOBS:
        for manifest in sorted(_ROOT.glob(glob)):
            project = tomllib.loads(manifest.read_text(encoding="utf-8")).get("project", {})
            declared = list(project.get("dependencies", []))
            for group in project.get("optional-dependencies", {}).values():
                declared.extend(group)
            for name, commit in _pins("\n".join(declared)).items():
                required.setdefault(name, {})[manifest.relative_to(_ROOT)] = commit
    return required


@pytest.mark.skipif(not _CONSTRAINTS.is_file(), reason="release constraints are repository-only")
def test_release_constraints_cover_every_workspace_url_requirement() -> None:
    """A drifted or missing pin here breaks the release, not the tests.

    The release workflow installs the built wheelhouse under these constraints.
    A direct URL that no constraint names cannot resolve at all, and a commit
    that disagrees with what a package requires fails during publication, long
    after the change that caused it. Deriving the expectation from the workspace
    means adding a new forked dependency cannot silently skip this check.
    """
    declared = _pins(_CONSTRAINTS.read_text(encoding="utf-8"))
    assert declared, "release constraints declare no immutable sources"

    for name, owners in sorted(_workspace_requirements().items()):
        commits = set(owners.values())
        assert len(commits) == 1, f"workspace members disagree on which commit of {name} to use: " + ", ".join(
            f"{owner} requires {commit}" for owner, commit in sorted(owners.items())
        )
        expected = commits.pop()
        origin = ", ".join(str(owner) for owner in sorted(owners))
        assert name in declared, f"release constraints do not pin {name}, required by {origin}"
        assert declared[name] == expected, (
            f"release constraints pin {name} at {declared[name]}, but {origin} requires {expected}"
        )
