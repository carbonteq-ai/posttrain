"""Project layout and catalog summary helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from posttrain.catalog import ProjectLayout
from posttrain.common import ContractError
from posttrain.common.selections import SelectionFamily


def layout_payload(layout: ProjectLayout) -> dict[str, object]:
    return {
        "project_id": layout.project_id,
        "root": str(layout.root),
        "manifest": str(layout.manifest),
        "catalog_overlays": [str(path) for path in layout.catalog_overlays],
        "work_packages": str(layout.work_packages),
        "state": str(layout.state),
        "tracking": layout.tracking,
        "entry": layout.entry,
    }


def work_package_path(layout: ProjectLayout, configured: Path) -> Path:
    candidate = configured if configured.is_absolute() else Path.cwd() / configured
    if not candidate.is_file() and not configured.is_absolute():
        candidate = layout.work_packages / configured
    resolved = candidate.resolve()
    if not resolved.is_relative_to(layout.work_packages):
        raise ContractError(f"work-package path must remain under {layout.work_packages}: {configured}")
    return resolved


def catalog_entries(catalog: Any, family: SelectionFamily | None = None) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for ref in catalog.list(family):
        resolved = catalog.resolve(ref)
        entries.append(
            {
                "family": ref.family,
                "id": ref.id,
                "source_layer": resolved.source_layer,
                "overlay_id": resolved.overlay_id,
            }
        )
    return entries


def catalog_summary(catalog: Any) -> dict[str, object]:
    entries = catalog_entries(catalog)
    overlay_entries = sum(entry["source_layer"] == "overlay" for entry in entries)
    return {
        "base_catalog_release": catalog.base_id,
        "overlay_ids": list(catalog.overlay_ids),
        "entries": len(entries),
        "base_entries": len(entries) - overlay_entries,
        "project_entries": overlay_entries,
    }
