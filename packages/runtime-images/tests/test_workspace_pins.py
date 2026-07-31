"""Framework source metadata must remain release-neutral.

The workspace releases as one unit, but its packages are published as separate
distributions. Source projects use a stable template version and bare workspace
dependencies so a release version does not rewrite every manifest. The release
tool's staged-metadata tests own the complementary invariant: published wheels
carry the manifest version and exact sibling pins.
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


def test_workspace_metadata_is_a_release_neutral_source_template() -> None:
    versions = _workspace_versions()
    assert versions, "no workspace packages were discovered"
    assert set(versions.values()) == {"0.0.0"}

    constrained: list[str] = []
    for manifest in _manifests():
        project = tomllib.loads(manifest.read_text(encoding="utf-8")).get("project", {})
        declared = list(project.get("dependencies", []))
        for group in project.get("optional-dependencies", {}).values():
            declared.extend(group)
        for requirement in declared:
            match = _REQUIREMENT.match(requirement.strip())
            if match is None or match.group("name") not in versions:
                continue
            if match.group("specifier").strip():
                where = manifest.relative_to(_ROOT)
                constrained.append(f"{where}: {requirement!r} must be bare in source metadata")

    assert not constrained, "release pins belong only in staged metadata:\n" + "\n".join(constrained)
