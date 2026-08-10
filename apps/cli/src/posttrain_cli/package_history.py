"""Find previously packed job packages for a work package and job."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from posttrain.catalog import ProjectLayout
from posttrain.common import ContractError
from posttrain.execution_pack import PackageMaterializationRecord

from .state_layout import cache_path

_MANIFEST_NAME = "package.json"
_CONTEXTS = "pack/contexts"
_RECORDS = Path("packages") / "materializations"
_LEGACY_RECORDS = "pack/records"


@dataclass(frozen=True, slots=True)
class RetainedPackage:
    """One packed job package retained on this machine."""

    package_key: str
    root: Path
    payload: Mapping[str, object]
    packed_at: float

    @property
    def work_package_id(self) -> str:
        return str(self.payload.get("work_package_id", ""))

    @property
    def job_id(self) -> str:
        return str(self.payload.get("job_id", ""))


def retained_packages(layout: ProjectLayout) -> tuple[RetainedPackage, ...]:
    """Return compact package records, with a legacy context fallback."""

    found: dict[str, RetainedPackage] = {}
    record_roots = [(layout.state / _RECORDS).resolve(), cache_path(layout, _LEGACY_RECORDS)]
    for records in record_roots:
        if not records.is_dir():
            continue
        for entry in records.iterdir():
            if entry.suffix != ".json" or entry.is_symlink() or not entry.is_file():
                continue
            try:
                record = PackageMaterializationRecord.from_bytes(entry.read_bytes())
            except (OSError, ContractError):
                continue
            context = cache_path(layout, _CONTEXTS, record.package_key)
            found[record.package_key] = RetainedPackage(
                package_key=record.package_key,
                root=context if context.is_dir() else entry,
                payload=record.manifest.to_payload(),
                packed_at=entry.stat().st_mtime,
            )

    contexts = cache_path(layout, _CONTEXTS)
    if contexts.is_dir():
        for entry in contexts.iterdir():
            manifest = entry / _MANIFEST_NAME
            if entry.name in found or not manifest.is_file():
                continue
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                # A partially written or unreadable legacy context is not
                # worth failing a diagnostic over.
                continue
            if not isinstance(payload, dict):
                continue
            found[entry.name] = RetainedPackage(
                package_key=entry.name,
                root=entry,
                payload=payload,
                packed_at=manifest.stat().st_mtime,
            )
    return tuple(sorted(found.values(), key=lambda item: item.packed_at, reverse=True))


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
        if package.package_key == package_key or package.package_key.startswith(package_key)
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
