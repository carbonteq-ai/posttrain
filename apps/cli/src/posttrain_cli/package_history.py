"""Find previously packed job packages for a work package and job."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from posttrain.catalog import ProjectLayout
from posttrain.common import ContractError

_MANIFEST_NAME = "package.json"
_CONTEXTS = "pack/contexts"


@dataclass(frozen=True, slots=True)
class RetainedPackage:
    """One packed job package retained on this machine."""

    package_key: str
    root: Path
    payload: dict[str, object]
    packed_at: float

    @property
    def work_package_id(self) -> str:
        return str(self.payload.get("work_package_id", ""))

    @property
    def job_id(self) -> str:
        return str(self.payload.get("job_id", ""))


def retained_packages(layout: ProjectLayout) -> tuple[RetainedPackage, ...]:
    """Return every retained package, newest first."""
    root = (layout.state / _CONTEXTS).resolve()
    if not root.is_dir():
        return ()
    found: list[RetainedPackage] = []
    for entry in root.iterdir():
        manifest = entry / _MANIFEST_NAME
        if not manifest.is_file():
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A partially written or unreadable context is not worth failing a
            # diagnostic over; it simply cannot participate in a comparison.
            continue
        if not isinstance(payload, dict):
            continue
        found.append(
            RetainedPackage(
                package_key=entry.name,
                root=entry,
                payload=payload,
                packed_at=manifest.stat().st_mtime,
            )
        )
    return tuple(sorted(found, key=lambda item: item.packed_at, reverse=True))


def packages_for(
    layout: ProjectLayout,
    *,
    work_package_id: str,
    job_id: str,
) -> tuple[RetainedPackage, ...]:
    """Return retained packages for one work package and job, newest first."""
    return tuple(
        package
        for package in retained_packages(layout)
        if package.work_package_id == work_package_id and package.job_id == job_id
    )


def resolve_package(
    layout: ProjectLayout,
    package_key: str,
) -> RetainedPackage:
    """Look up one retained package by key, allowing an unambiguous prefix."""
    candidates = [
        package
        for package in retained_packages(layout)
        if package.package_key == package_key
        or package.package_key.startswith(package_key)
    ]
    if not candidates:
        raise ContractError(f"no retained job package matches {package_key!r}")
    if len(candidates) > 1:
        keys = ", ".join(sorted(package.package_key[:16] for package in candidates))
        raise ContractError(f"{package_key!r} matches several retained packages: {keys}")
    return candidates[0]


__all__ = [
    "RetainedPackage",
    "packages_for",
    "resolve_package",
    "retained_packages",
]
