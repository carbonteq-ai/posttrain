"""Framework packages must pin each other to exact versions.

The workspace releases as one unit, but its packages are published as separate
distributions. Declared by bare name, `posttrain` can be upgraded while every
sibling stays behind, producing an installation that is several versions of the
framework at once. Nothing detects that: each distribution is individually
satisfiable, and the mixture only surfaces as behaviour that matches no release.

It also reaches the job image. Framework code is staged into an actual-job
package from the installed distributions, so a mixed installation is packed and
published as if it were a coherent framework.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_GLOBS = ("pyproject.toml", "apps/*/pyproject.toml", "packages/*/pyproject.toml")

_REQUIREMENT = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)(?:\[[^\]]+\])?(?P<specifier>.*)$")


def _manifests() -> list[Path]:
    return [path for glob in _GLOBS for path in sorted(_ROOT.glob(glob))]


def _workspace_versions() -> dict[str, str]:
    versions = {}
    for manifest in _manifests():
        project = tomllib.loads(manifest.read_text(encoding="utf-8")).get("project", {})
        if "name" in project and "version" in project:
            versions[project["name"]] = project["version"]
    return versions


def test_intra_workspace_dependencies_pin_the_released_version() -> None:
    versions = _workspace_versions()
    assert versions, "no workspace packages were discovered"

    unpinned: list[str] = []
    for manifest in _manifests():
        project = tomllib.loads(manifest.read_text(encoding="utf-8")).get("project", {})
        declared = list(project.get("dependencies", []))
        for group in project.get("optional-dependencies", {}).values():
            declared.extend(group)
        for requirement in declared:
            match = _REQUIREMENT.match(requirement.strip())
            if match is None or match.group("name") not in versions:
                continue
            name = match.group("name")
            expected = f"=={versions[name]}"
            if match.group("specifier").strip() != expected:
                where = manifest.relative_to(_ROOT)
                unpinned.append(f"{where}: {requirement!r} should require {name}{expected}")

    assert not unpinned, "framework packages must pin their siblings exactly:\n" + "\n".join(unpinned)
