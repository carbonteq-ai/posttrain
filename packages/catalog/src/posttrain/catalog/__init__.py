"""Versioned framework base catalog and project-overlay composition."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from importlib.resources import as_file
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Any

from posttrain.common import Catalog, CatalogLayer, CatalogRef
from posttrain.common.catalog import SelectionDecoder
from posttrain.common.selections import Selection, SelectionFamily
from posttrain.eval import evaluation_catalog_decoders
from posttrain.eval.programs import GENERAL_ENVIRONMENT_FACTORIES
from posttrain.train import TRAIN_CATALOG_DECODERS

from .files import (
    CatalogDocumentSchema,
    CatalogLayerManifestSchema,
    load_catalog_layer,
)
from .project import ProjectLayout, discover_project, load_project_layout

BASE_CATALOG_RELEASE = "framework-v1"


def _automationbench_training_environment() -> Any:
    try:
        from verifiers.v1.env import EnvConfig, Environment  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("AutomationBench execution requires the pinned Verifiers environment package") from error
    config = EnvConfig.model_validate(
        {
            "taskset": {"id": "automationbench-v1"},
            "harness": {"id": "null", "runtime": {"type": "subprocess"}},
            "timeout": {"setup": 120, "rollout": 1800, "finalize": 60, "scoring": 120},
            "max_turns": 50,
            "max_total_tokens": 8192,
        }
    )
    return Environment(config)


def catalog_decoders() -> Mapping[SelectionFamily, SelectionDecoder]:
    """Return decoders for every selection family in the framework base."""

    environment_factories = {
        **GENERAL_ENVIRONMENT_FACTORIES,
        "automationbench-zapier-training": _automationbench_training_environment,
    }
    return {
        **evaluation_catalog_decoders(environment_factories),
        **TRAIN_CATALOG_DECODERS,
    }


def packaged_base_directory() -> AbstractContextManager[Path]:
    """Materialize the packaged base catalog as a filesystem directory."""

    return as_file(resource_files("posttrain.catalog").joinpath("base"))


def _framework_base_layer(catalog_root: Path | None) -> CatalogLayer:
    if catalog_root is not None:
        return _load_framework_base(catalog_root / "base")
    with packaged_base_directory() as directory:
        return _load_framework_base(directory)


def _load_framework_base(directory: Path) -> CatalogLayer:
    file_catalog = Catalog.open(
        load_catalog_layer(directory),
        scope="framework",
        decoders=catalog_decoders(),
    )
    entries: dict[CatalogRef, Selection] = {ref: file_catalog.resolve(ref).value for ref in file_catalog.list()}
    return CatalogLayer(BASE_CATALOG_RELEASE, entries)


def open_catalog(
    *,
    scope: str,
    overlays: tuple[Mapping[str, object] | Path, ...] = (),
    catalog_root: Path | None = None,
) -> Catalog:
    """Open the packaged framework base plus project overlay directories."""

    sources: list[Mapping[str, object]] = []
    if catalog_root is not None:
        project_directory = catalog_root / "projects" / scope.replace("/", "__")
        if project_directory.is_dir():
            sources.append(load_catalog_layer(project_directory))
    for source in overlays:
        sources.append(load_catalog_layer(source) if isinstance(source, Path) else source)
    return Catalog.open(
        _framework_base_layer(catalog_root),
        overlays=sources,
        scope=scope,
        decoders=catalog_decoders(),
    )


__all__ = [
    "BASE_CATALOG_RELEASE",
    "CatalogDocumentSchema",
    "CatalogLayerManifestSchema",
    "ProjectLayout",
    "catalog_decoders",
    "discover_project",
    "load_catalog_layer",
    "load_project_layout",
    "open_catalog",
    "packaged_base_directory",
]
