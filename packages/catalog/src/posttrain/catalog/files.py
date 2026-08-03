"""Manifest-controlled YAML catalog loading."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal, cast

import yaml
from posttrain.common import ContractError
from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError, field_validator, model_validator

from .providers import CatalogEntries, load_python_catalog_provider


class _Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


_LOCAL_YAML_SUFFIXES = {".yaml", ".yml"}


def _validate_local_yaml_filename(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or path.name != value or path.suffix not in _LOCAL_YAML_SUFFIXES:
        raise ValueError("catalog manifest YAML sources must be local YAML filenames")
    return value


class CatalogYamlSourceSchema(_Schema):
    kind: Literal["yaml"]
    path: str

    _safe_path = field_validator("path")(_validate_local_yaml_filename)


class CatalogPythonSourceSchema(_Schema):
    kind: Literal["python"]
    provider: str

    @field_validator("provider")
    @classmethod
    def valid_provider_reference(cls, value: str) -> str:
        module, separator, attribute = value.partition(":")
        if (
            not separator
            or not module
            or not attribute
            or ":" in attribute
            or not all(part.isidentifier() for part in module.split("."))
            or not all(part.isidentifier() for part in attribute.split("."))
        ):
            raise ValueError("catalog Python providers must use MODULE:CALLABLE syntax")
        return value


type CatalogLayerSourceSchema = Annotated[
    CatalogYamlSourceSchema | CatalogPythonSourceSchema,
    Field(discriminator="kind"),
]


class CatalogLayerManifestSchema(_Schema):
    schema_version: Literal[1, 2]
    layer_id: str
    files: tuple[str, ...] | None = None
    sources: tuple[CatalogLayerSourceSchema, ...] | None = None

    @model_validator(mode="after")
    def validate_versioned_sources(self) -> CatalogLayerManifestSchema:
        if self.schema_version == 1:
            if self.files is None:
                raise ValueError("catalog schema version 1 manifests require files")
            if "sources" in self.model_fields_set:
                raise ValueError("catalog schema version 1 manifests cannot define sources")
            if len(set(self.files)) != len(self.files):
                raise ValueError("catalog manifests require unique files")
            for name in self.files:
                _validate_local_yaml_filename(name)
            return self

        if "files" in self.model_fields_set:
            raise ValueError("catalog schema version 2 manifests cannot define files")
        if self.sources is None:
            raise ValueError("catalog schema version 2 manifests require sources")
        yaml_paths = [source.path for source in self.sources if isinstance(source, CatalogYamlSourceSchema)]
        if len(set(yaml_paths)) != len(yaml_paths):
            raise ValueError("catalog manifests require unique YAML source paths")
        providers = [source.provider for source in self.sources if isinstance(source, CatalogPythonSourceSchema)]
        if len(set(providers)) != len(providers):
            raise ValueError("catalog manifests require unique Python providers")
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
    sources: tuple[CatalogLayerSourceSchema, ...]
    if manifest.schema_version == 1:
        sources = tuple(CatalogYamlSourceSchema(kind="yaml", path=filename) for filename in manifest.files or ())
    else:
        sources = manifest.sources or ()
    for source in sources:
        if isinstance(source, CatalogYamlSourceSchema):
            loaded: Mapping[str, object] = cast(
                Mapping[str, object],
                _load_schema(directory / source.path, CatalogDocumentSchema).root,
            )
        else:
            try:
                loaded = _entries_mapping(load_python_catalog_provider(source.provider))
            except ContractError as error:
                raise ContractError(
                    f"catalog layer {manifest.layer_id!r} Python provider {source.provider!r} failed: {error}"
                ) from error
        _merge_families(families, loaded, manifest.layer_id)
    return {"layer_id": manifest.layer_id, **cast(dict[str, object], families)}


def _entries_mapping(entries: object) -> dict[str, object]:
    """Convert typed provider entries to the mapping consumed by ``Catalog``."""

    if not isinstance(entries, CatalogEntries):
        raise ContractError("Python catalog provider returned an invalid result")
    families: dict[str, dict[str, object]] = {}
    for ref, value in entries.entries.items():
        families.setdefault(ref.family, {})[ref.id] = value
    return cast(dict[str, object], families)


def _merge_families(
    destination: dict[str, dict[str, object]],
    source: Mapping[str, object],
    layer_id: str,
) -> None:
    for family, entries in source.items():
        if not isinstance(entries, Mapping):
            raise ContractError(f"catalog family {family!r} must contain an object")
        target = destination.setdefault(family, {})
        duplicate = set(target).intersection(entries)
        if duplicate:
            ids = ", ".join(sorted(duplicate))
            raise ContractError(f"duplicate catalog ids in layer {layer_id!r}: {family}/{ids}")
        target.update(entries)


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
    "CatalogLayerSourceSchema",
    "CatalogPythonSourceSchema",
    "CatalogYamlSourceSchema",
    "CatalogLayerManifestSchema",
    "load_catalog_layer",
]
