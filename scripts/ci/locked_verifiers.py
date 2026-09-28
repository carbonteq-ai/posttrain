"""The Verifiers build CI must test: the one ``uv.lock`` selects for the framework.

Environment packages declare their own Verifiers dependency. Installed after the
locked sync, one pinned to another revision silently replaced the framework's
Verifiers, so CI tested a Verifiers the release does not ship.

    .venv/bin/python scripts/ci/locked_verifiers.py constraint  # a pip constraint for the locked build
    .venv/bin/python scripts/ci/locked_verifiers.py check       # fail unless the locked build is installed

Only the standard library (Python 3.11+) is used.
"""

from __future__ import annotations

import json
import sys
import tomllib
from importlib import metadata
from pathlib import Path
from urllib.parse import urlsplit

_LOCK = Path(__file__).resolve().parents[2] / "uv.lock"


def locked_verifiers(lock: Path = _LOCK) -> tuple[str, str, str]:
    """Return the locked Verifiers version, Git repository and commit."""

    document = tomllib.loads(lock.read_text(encoding="utf-8"))
    packages = [package for package in document.get("package", []) if package.get("name") == "verifiers"]
    if len(packages) != 1:
        raise SystemExit(f"uv.lock must lock exactly one verifiers package, found {len(packages)}")
    package = packages[0]
    git = package.get("source", {}).get("git")
    if not isinstance(git, str) or "#" not in git:
        raise SystemExit("the locked verifiers package is not an immutable Git source")
    location, commit = git.split("#", 1)
    repository = urlsplit(location)._replace(query="", fragment="").geturl()
    if len(commit) != 40:
        raise SystemExit(f"the locked verifiers commit is not a full revision: {commit!r}")
    return str(package["version"]), repository, commit


def constraint(lock: Path = _LOCK) -> str:
    _version, repository, commit = locked_verifiers(lock)
    return f"verifiers @ git+{repository}@{commit}"


def check(lock: Path = _LOCK) -> str:
    """Fail unless the running interpreter has exactly the locked Verifiers build."""

    version, repository, commit = locked_verifiers(lock)
    try:
        distribution = metadata.distribution("verifiers")
    except metadata.PackageNotFoundError as error:
        raise SystemExit("verifiers is not installed") from error
    installed_version = distribution.version
    raw = distribution.read_text("direct_url.json")
    direct = json.loads(raw) if raw else {}
    installed_commit = direct.get("vcs_info", {}).get("commit_id")
    installed_repository = direct.get("url")
    if (installed_version, installed_repository, installed_commit) != (version, repository, commit):
        raise SystemExit(
            "installed verifiers is not the locked build: "
            f"{installed_version} {installed_repository}@{installed_commit}, "
            f"expected {version} {repository}@{commit}"
        )
    return f"verifiers {version} {repository}@{commit}"


def main(argv: list[str]) -> int:
    if argv == ["constraint"]:
        print(constraint())
        return 0
    if argv == ["check"]:
        print(f"installed the locked {check()}")
        return 0
    print("usage: locked_verifiers.py constraint|check", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
