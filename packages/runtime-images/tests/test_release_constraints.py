"""The release constraints must name the same sources the packages require."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_CONSTRAINTS = _ROOT / "release" / "github-constraints.txt"
_OWNERS = {
    "carbonteq-trackio": _ROOT / "packages" / "tracking-trackio" / "pyproject.toml",
    "carbonteq-automation-bench": _ROOT / "environments" / "automationbench_v1" / "pyproject.toml",
}


def _pins(text: str) -> dict[str, str]:
    found = {}
    for line in text.splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9._-]+)\s*@\s*git\+\S+@([0-9a-f]{40})", entry)
        if match:
            found[match.group(1)] = match.group(2)
    return found


@pytest.mark.skipif(not _CONSTRAINTS.is_file(), reason="release constraints are repository-only")
def test_release_constraints_match_the_packages_that_declare_them() -> None:
    """A drifted pin here breaks the release, not the tests.

    The release workflow installs the built wheelhouse under these constraints.
    If a commit here disagrees with what a package requires, resolution fails
    during publication, long after the change that caused it.
    """
    declared = _pins(_CONSTRAINTS.read_text(encoding="utf-8"))
    assert declared, "release constraints declare no immutable sources"

    for name, owner in _OWNERS.items():
        if not owner.is_file():
            continue
        requirements = tomllib.loads(owner.read_text(encoding="utf-8"))["project"]["dependencies"]
        expected = _pins("\n".join(requirements)).get(name)
        if expected is None:
            continue
        assert declared.get(name) == expected, (
            f"release constraints pin {name} at {declared.get(name)}, "
            f"but {owner.relative_to(_ROOT)} requires {expected}"
        )
