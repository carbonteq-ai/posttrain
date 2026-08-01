"""Manifest-controlled YAML catalog loading."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from posttrain.common import ContractError
from pydantic import BaseModel, ConfigDict, RootModel, ValidationError, model_validator


class _Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CatalogLayerManifestSchema(_Schema):
    schema_version: Literal[1]
    layer_id: str
    files: tuple[str, ...]

    @model_validator(mode="after")
    def safe_unique_files(self) -> CatalogLayerManifestSchema:
        if len(set(self.files)) != len(self.files):
            raise ValueError("catalog manifests require unique files")
        for name in self.files:
            path = Path(name)
            if path.name != name or path.suffix not in {".yaml", ".yml"}:
                raise ValueError("catalog manifest files must be local YAML filenames")
        return self


class CatalogDocumentSchema(RootModel[dict[str, dict[str, object]]]):
    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def contains_entries(self) -> CatalogDocumentSchema:
        if not self.root:
            raise ValueError("catalog documents require at least one family entry")
        if any(not family or not entries for family, entries in self.root.items()):
            raise ValueError("catalog documents require non-empty family entries")
        return self


def load_catalog_layer(directory: Path) -> dict[str, object]:
    """Read one manifest-controlled catalog directory into a source mapping."""

    manifest_path = directory / "layer.yaml"
    manifest = _load_schema(manifest_path, CatalogLayerManifestSchema)
    families: dict[str, dict[str, object]] = {}
    for filename in manifest.files:
        path = directory / filename
        document = _load_schema(path, CatalogDocumentSchema)
        for family, entries in document.root.items():
            destination = families.setdefault(family, {})
            duplicate = set(destination).intersection(entries)
            if duplicate:
                ids = ", ".join(sorted(duplicate))
                raise ContractError(f"duplicate catalog ids in layer {manifest.layer_id!r}: {family}/{ids}")
            destination.update(entries)
    return {"layer_id": manifest.layer_id, **families}


def _load_schema[SchemaT: BaseModel](path: Path, schema: type[SchemaT]) -> SchemaT:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return schema.model_validate(payload)
    except FileNotFoundError as error:
        raise ContractError(f"catalog file not found: {path}") from error
    except (yaml.YAMLError, ValidationError) as error:
        raise ContractError(f"invalid catalog YAML {path}: {error}") from error


__all__ = [
    "CatalogDocumentSchema",
    "CatalogLayerManifestSchema",
    "load_catalog_layer",
]
