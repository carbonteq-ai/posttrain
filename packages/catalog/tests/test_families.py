"""Deterministic installed catalog-family composition."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from posttrain.catalog import CatalogFamilyDescriptor, family_registry, open_catalog
from posttrain.common import CatalogRef, ContractError


@dataclass(frozen=True, slots=True)
class Widget:
    id: str
    value: str


class _Distribution:
    metadata = {"Name": "example-catalog-plugin"}
    version = "2.4.1"


class _EntryPoint:
    dist = _Distribution()
    value = "example_catalog:families"

    def __init__(self, name: str, descriptor: CatalogFamilyDescriptor) -> None:
        self.name = name
        self._descriptor = descriptor

    def load(self) -> CatalogFamilyDescriptor:
        return self._descriptor


def _widget_descriptor(name: str = "widget") -> CatalogFamilyDescriptor:
    def decode(ref, data, known):
        del known
        return Widget(id=ref.id, value=str(data["value"]))

    return CatalogFamilyDescriptor(
        name=name,
        schema_identity="example.widget",
        schema_revision="1",
        decoder=decode,
    )


def test_extension_family_is_sorted_and_frozen_into_the_catalog_lock() -> None:
    registry = family_registry(entry_point_values=(_EntryPoint("widgets", _widget_descriptor()),))
    catalog = open_catalog(
        scope="test",
        overlays=({"layer_id": "test", "widget": {"primary": {"value": "ok"}}},),
        registry=registry,
    )

    value = catalog.resolve(CatalogRef("widget", "primary")).value
    assert value == Widget("primary", "ok")
    assert catalog.family_registry_lock == registry.lock
    lock = registry.lock
    entry = next(item for item in lock.entries if item.name == "widget")
    assert entry.distribution == "example-catalog-plugin"
    assert entry.distribution_version == "2.4.1"
    assert entry.entry_point == "widgets"


def test_duplicate_extension_families_name_both_origins() -> None:
    with pytest.raises(ContractError, match="duplicate catalog family 'widget'.*first.*second"):
        family_registry(
            entry_point_values=(
                _EntryPoint("first", _widget_descriptor()),
                _EntryPoint("second", _widget_descriptor()),
            )
        )


def test_missing_family_fails_loudly_before_catalog_decoding() -> None:
    registry = family_registry(entry_point_values=())

    with pytest.raises(ContractError, match="catalog_family_unavailable: widget.*installed catalog families"):
        open_catalog(
            scope="test",
            overlays=({"layer_id": "test", "widget": {"primary": {"value": "ok"}}},),
            registry=registry,
        )


def test_complete_registry_lock_changes_for_an_unrelated_installed_family() -> None:
    baseline = family_registry(entry_point_values=()).lock
    extended = family_registry(entry_point_values=(_EntryPoint("widgets", _widget_descriptor()),)).lock

    assert baseline.digest != extended.digest
    assert "widget" not in {entry.name for entry in baseline.entries}
    assert "widget" in {entry.name for entry in extended.entries}
