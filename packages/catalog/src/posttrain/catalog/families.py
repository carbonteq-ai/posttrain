"""Deterministic catalog-family composition and its frozen resolution lock."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from importlib.metadata import EntryPoint, PackageNotFoundError, distribution, entry_points
from typing import Any

from posttrain.common import ContractError
from posttrain.common.catalog import SelectionDecoder


@dataclass(frozen=True, slots=True)
class CatalogFamilyDescriptor:
    """One family decoder with enough provenance to reproduce composition."""

    name: str
    schema_identity: str
    schema_revision: str
    decoder: SelectionDecoder | None
    origin: str = "core"
    distribution: str = "posttrain-catalog"
    distribution_version: str = "unknown"
    entry_point: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("-", "").isalnum() or not self.name[0].isalpha():
            raise ContractError(f"catalog family name is invalid: {self.name!r}")
        if not self.schema_identity or not self.schema_revision:
            raise ContractError("catalog family descriptors require schema identity and revision")
        if not self.origin or not self.distribution or not self.distribution_version:
            raise ContractError("catalog family descriptors require complete provenance")


@dataclass(frozen=True, slots=True, order=True)
class FamilyRegistryLockEntry:
    name: str
    schema_identity: str
    schema_revision: str
    origin: str
    distribution: str
    distribution_version: str
    entry_point: str | None


@dataclass(frozen=True, slots=True)
class FamilyRegistryLock:
    entries: tuple[FamilyRegistryLockEntry, ...]

    def __post_init__(self) -> None:
        if tuple(sorted(self.entries)) != self.entries:
            raise ContractError("family registry lock entries must be sorted")
        if len({entry.name for entry in self.entries}) != len(self.entries):
            raise ContractError("family registry lock families must be unique")

    @property
    def digest(self) -> str:
        payload = [asdict(entry) for entry in self.entries]
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def to_payload(self) -> dict[str, object]:
        return {"entries": [asdict(entry) for entry in self.entries], "digest": self.digest}


class FamilyRegistry:
    """An immutable, explicit registry composed from core and installed plugins."""

    def __init__(self, descriptors: Iterable[CatalogFamilyDescriptor]) -> None:
        ordered = tuple(sorted(descriptors, key=lambda item: item.name))
        duplicates: dict[str, list[CatalogFamilyDescriptor]] = {}
        for descriptor in ordered:
            duplicates.setdefault(descriptor.name, []).append(descriptor)
        duplicate = next((items for items in duplicates.values() if len(items) > 1), None)
        if duplicate is not None:
            origins = ", ".join(
                f"{item.origin} ({item.distribution}{' via ' + item.entry_point if item.entry_point else ''})"
                for item in duplicate
            )
            raise ContractError(f"duplicate catalog family {duplicate[0].name!r}: {origins}")
        self._descriptors = {descriptor.name: descriptor for descriptor in ordered}

    @classmethod
    def compose(
        cls,
        core: Iterable[CatalogFamilyDescriptor],
        entry_point_values: Iterable[EntryPoint] | None = None,
    ) -> FamilyRegistry:
        discovered = entry_point_values if entry_point_values is not None else _family_entry_points()
        extension: list[CatalogFamilyDescriptor] = []
        for point in sorted(discovered, key=_entry_point_sort_key):
            extension.extend(_load_entry_point(point))
        return cls((*core, *extension))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._descriptors)

    @property
    def lock(self) -> FamilyRegistryLock:
        return FamilyRegistryLock(
            tuple(
                FamilyRegistryLockEntry(
                    name=descriptor.name,
                    schema_identity=descriptor.schema_identity,
                    schema_revision=descriptor.schema_revision,
                    origin=descriptor.origin,
                    distribution=descriptor.distribution,
                    distribution_version=descriptor.distribution_version,
                    entry_point=descriptor.entry_point,
                )
                for descriptor in self._descriptors.values()
            )
        )

    def decoders(self) -> Mapping[str, SelectionDecoder]:
        return {
            name: descriptor.decoder for name, descriptor in self._descriptors.items() if descriptor.decoder is not None
        }

    def require_families(self, families: Iterable[str]) -> None:
        missing = tuple(sorted(set(families).difference(self._descriptors)))
        if not missing:
            return
        available = ", ".join(self.names) or "(none)"
        raise ContractError(
            f"catalog_family_unavailable: {', '.join(missing)}; installed catalog families: {available}"
        )

    def require_distributions(self, requirements: Iterable[str]) -> None:
        """Fail before decoding when a tracked project provider is absent."""

        required = tuple(requirements)
        installed = {_normalized_distribution(entry.distribution) for entry in self.lock.entries}
        missing = tuple(
            requirement
            for requirement in required
            if _normalized_distribution(_requirement_distribution(requirement)) not in installed
        )
        if not missing:
            return
        available = ", ".join(sorted(installed)) or "(none)"
        raise ContractError(
            "catalog_family_unavailable: required catalog plugin distribution is not installed: "
            f"{', '.join(missing)}; installed family providers: {available}"
        )


def core_catalog_family_descriptors(
    decoders: Mapping[str, SelectionDecoder],
) -> tuple[CatalogFamilyDescriptor, ...]:
    """Name the complete built-in family set rather than inferring it from imports."""

    version = _distribution_version("posttrain-catalog")
    names = (
        "model",
        "dataset",
        "environment",
        "inference",
        "training",
        "quantization",
        "evaluation",
        "workload",
        "target",
        "recipe",
    )
    return tuple(
        CatalogFamilyDescriptor(
            name=name,
            schema_identity=f"posttrain.catalog.{name}",
            schema_revision="1",
            decoder=decoders.get(name),
            distribution_version=version,
        )
        for name in names
    )


def _family_entry_points() -> tuple[EntryPoint, ...]:
    selected = entry_points(group="posttrain.catalog_families")
    return tuple(selected)


def _entry_point_sort_key(point: EntryPoint) -> tuple[str, str, str]:
    distribution_name = _entry_point_distribution(point)
    return (distribution_name.lower(), point.name, point.value)


def _entry_point_distribution(point: EntryPoint) -> str:
    metadata = getattr(getattr(point, "dist", None), "metadata", None)
    if metadata is not None:
        name = metadata.get("Name")
        if isinstance(name, str) and name:
            return name
    return "unknown-distribution"


def _load_entry_point(point: EntryPoint) -> tuple[CatalogFamilyDescriptor, ...]:
    loaded: Any = point.load()
    if callable(loaded) and not isinstance(loaded, CatalogFamilyDescriptor):
        loaded = loaded()
    descriptors = (loaded,) if isinstance(loaded, CatalogFamilyDescriptor) else tuple(loaded)
    if not descriptors or not all(isinstance(item, CatalogFamilyDescriptor) for item in descriptors):
        raise ContractError(f"catalog family entry point {point.name!r} must provide CatalogFamilyDescriptor values")
    distribution_name = _entry_point_distribution(point)
    version = getattr(getattr(point, "dist", None), "version", None)
    return tuple(
        replace(
            descriptor,
            origin=f"entry-point:{point.name}",
            distribution=distribution_name,
            distribution_version=version if isinstance(version, str) and version else "unknown",
            entry_point=point.name,
        )
        for descriptor in descriptors
    )


def _distribution_version(name: str) -> str:
    try:
        return distribution(name).version
    except PackageNotFoundError:
        return "unknown"


def _requirement_distribution(value: str) -> str:
    for marker in "<>=!~ ":
        if marker in value:
            return value.partition(marker)[0]
    return value


def _normalized_distribution(value: str) -> str:
    return value.replace("_", "-").replace(".", "-").lower()


__all__ = [
    "CatalogFamilyDescriptor",
    "FamilyRegistry",
    "FamilyRegistryLock",
    "FamilyRegistryLockEntry",
    "core_catalog_family_descriptors",
]
