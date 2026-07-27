"""Versioned framework base catalog and project-overlay composition."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from importlib.metadata import entry_points
from importlib.resources import as_file
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Any

from posttrain.common import Catalog, CatalogLayer, CatalogRef
from posttrain.common.catalog import SelectionDecoder
from posttrain.common.selections import Selection, SelectionFamily
from posttrain.data import DATA_CATALOG_DECODERS
from posttrain.eval import (
    EnvironmentActivation,
    EnvironmentFactory,
    PythonFactoryActivation,
    VerifiersV1ConfigActivation,
    evaluation_catalog_decoders,
)
from posttrain.eval.programs import GENERAL_ENVIRONMENT_ACTIVATIONS
from posttrain.train import TRAIN_CATALOG_DECODERS

from .files import (
    CatalogDocumentSchema,
    CatalogLayerManifestSchema,
    load_catalog_layer,
)
from .project import (
    ProjectExecutionDefaults,
    ProjectLayout,
    discover_project,
    load_project_layout,
)

BASE_CATALOG_RELEASE = "framework-v1"


AUTOMATIONBENCH_TRAINING_ACTIVATION = VerifiersV1ConfigActivation(
    {
        "taskset": {"id": "automationbench-v1"},
        "harness": {"id": "null", "runtime": {"type": "subprocess"}},
        "timeout": {"setup": 120, "rollout": 1800, "finalize": 60, "scoring": 120},
        "max_turns": 50,
        "max_total_tokens": 8192,
    }
)


def environment_factory_registry(
    extras: Mapping[str, EnvironmentActivation | EnvironmentFactory] | None = None,
) -> Mapping[str, EnvironmentActivation]:
    """Return inert legacy aliases without importing environment packages."""

    activations: dict[str, EnvironmentActivation] = {
        **GENERAL_ENVIRONMENT_ACTIVATIONS,
        "automationbench-zapier-training": AUTOMATIONBENCH_TRAINING_ACTIVATION,
    }
    for entry_point in entry_points(group="posttrain.environment_factories"):
        if entry_point.name in activations:
            raise RuntimeError(f"duplicate environment factory entry point: {entry_point.name}")
        activations[entry_point.name] = PythonFactoryActivation(_entry_point_reference(entry_point))
    for name, activation in (extras or {}).items():
        activations[name] = (
            activation
            if isinstance(
                activation,
                (PythonFactoryActivation, VerifiersV1ConfigActivation),
            )
            else PythonFactoryActivation.from_callable(activation)
        )
    return activations


def _entry_point_reference(entry_point: Any) -> str:
    module = getattr(entry_point, "module", None)
    attribute = getattr(entry_point, "attr", None)
    if isinstance(module, str) and isinstance(attribute, str):
        return f"{module}:{attribute}"
    value = getattr(entry_point, "value", None)
    if not isinstance(value, str):
        raise RuntimeError(f"environment factory entry point {entry_point.name!r} has no import reference")
    return value.partition(" ")[0]


def catalog_decoders(
    *,
    environment_factories: Mapping[str, Any] | None = None,
) -> Mapping[SelectionFamily, SelectionDecoder]:
    """Return decoders for every selection family in the framework base."""

    return {
        **DATA_CATALOG_DECODERS,
        **evaluation_catalog_decoders(environment_factory_registry(environment_factories)),
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


def _load_framework_base(
    directory: Path,
    *,
    decoders: Mapping[SelectionFamily, SelectionDecoder] | None = None,
) -> CatalogLayer:
    file_catalog = Catalog.open(
        load_catalog_layer(directory),
        scope="framework",
        decoders=catalog_decoders() if decoders is None else decoders,
    )
    entries: dict[CatalogRef, Selection] = {ref: file_catalog.resolve(ref).value for ref in file_catalog.list()}
    return CatalogLayer(BASE_CATALOG_RELEASE, entries)


def open_catalog(
    *,
    scope: str,
    overlays: tuple[Mapping[str, object] | Path, ...] = (),
    catalog_root: Path | None = None,
    environment_factories: Mapping[str, Any] | None = None,
) -> Catalog:
    """Open the packaged framework base plus project overlay directories."""

    sources: list[Mapping[str, object]] = []
    if catalog_root is not None:
        project_directory = catalog_root / "projects" / scope.replace("/", "__")
        if project_directory.is_dir():
            sources.append(load_catalog_layer(project_directory))
    for source in overlays:
        sources.append(load_catalog_layer(source) if isinstance(source, Path) else source)
    decoders = catalog_decoders(environment_factories=environment_factories)
    base_directory: Path | None = catalog_root / "base" if catalog_root is not None else None
    if base_directory is None:
        with packaged_base_directory() as directory:
            base_layer = _load_framework_base(directory, decoders=decoders)
    else:
        base_layer = _load_framework_base(base_directory, decoders=decoders)
    return Catalog.open(
        base_layer,
        overlays=sources,
        scope=scope,
        decoders=decoders,
    )


__all__ = [
    "BASE_CATALOG_RELEASE",
    "CatalogDocumentSchema",
    "CatalogLayerManifestSchema",
    "ProjectLayout",
    "ProjectExecutionDefaults",
    "catalog_decoders",
    "discover_project",
    "environment_factory_registry",
    "load_catalog_layer",
    "load_project_layout",
    "open_catalog",
    "packaged_base_directory",
]
