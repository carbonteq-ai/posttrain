"""Write project catalog overlay YAML entries for CLI helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from posttrain.catalog import ProjectLayout
from posttrain.common import ContractError


def overlay_directory(layout: ProjectLayout) -> Path:
    """Return the first project catalog overlay directory, creating it if needed."""

    if layout.catalog_overlays:
        return layout.catalog_overlays[0]
    raise ContractError(
        "project has no catalog overlay; set catalog_overlays in .posttrain/project.toml "
        '(for example catalog_overlays = ["catalog"])'
    )


def ensure_overlay_file(overlay: Path, filename: str, *, layer_id: str) -> Path:
    """Ensure layer.yaml lists filename and return the document path."""

    overlay.mkdir(parents=True, exist_ok=True)
    manifest_path = overlay / "layer.yaml"
    if manifest_path.is_file():
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ContractError(f"invalid catalog layer manifest: {manifest_path}")
        files = payload.get("files", [])
        if files is None:
            files = []
        if not isinstance(files, list):
            raise ContractError(f"catalog layer files must be a list: {manifest_path}")
        names = [str(item) for item in files]
        if filename not in names:
            names.append(filename)
            payload["files"] = names
            payload.setdefault("schema_version", 1)
            payload.setdefault("layer_id", layer_id)
            manifest_path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )
    else:
        manifest_path.write_text(
            "\n".join(
                (
                    "schema_version: 1",
                    f"layer_id: {layer_id}",
                    "files:",
                    f"  - {filename}",
                    "",
                )
            ),
            encoding="utf-8",
        )
    return overlay / filename


def upsert_family_entry(
    path: Path,
    *,
    family: str,
    entry_id: str,
    entry: dict[str, Any],
) -> None:
    """Insert one family entry into an overlay YAML document."""

    document: dict[str, Any] = {}
    if path.is_file():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ContractError(f"invalid catalog document: {path}")
        document = loaded
    family_entries = document.setdefault(family, {})
    if not isinstance(family_entries, dict):
        raise ContractError(f"catalog family {family!r} must be a mapping in {path}")
    if entry_id in family_entries:
        raise ContractError(f"catalog already contains {family}/{entry_id}")
    family_entries[entry_id] = entry
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def selection_revision(selection_id: str) -> str:
    if "@" in selection_id:
        return selection_id.rsplit("@", maxsplit=1)[1]
    return "1"


__all__ = [
    "ensure_overlay_file",
    "overlay_directory",
    "selection_revision",
    "upsert_family_entry",
]
