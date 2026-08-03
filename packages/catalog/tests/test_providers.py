from __future__ import annotations

import sys
from pathlib import Path

import pytest
from posttrain.catalog import CatalogEntries, load_python_catalog_provider
from posttrain.common import CatalogRef, ContractError, Workload


def _write_provider(tmp_path: Path, source: str) -> None:
    (tmp_path / "provider_fixture.py").write_text(source, encoding="utf-8")


def test_catalog_entries_freezes_typed_values_and_supports_dataset_shortcut() -> None:
    value = Workload(id="datasets/example@1", revision="1", requests={"suite_id": "fixture"})
    entries = CatalogEntries(datasets=(value,))

    assert entries.entries[CatalogRef("dataset", "datasets/example@1")] is value
    with pytest.raises(TypeError):
        entries.entries[CatalogRef("dataset", "datasets/other@1")] = value  # type: ignore[index]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("def entries():\n    return object()\n", "must return CatalogEntries"),
        ("def entries():\n    raise RuntimeError('network access')\n", "failed: network access"),
        ("entries = 1\n", "target is not callable"),
    ],
)
def test_python_provider_wraps_invalid_results_and_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    message: str,
) -> None:
    _write_provider(tmp_path, source)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "provider_fixture", raising=False)

    with pytest.raises(ContractError, match=message):
        load_python_catalog_provider("provider_fixture:entries")


def test_python_provider_wraps_import_failures() -> None:
    with pytest.raises(ContractError, match="could not import Python catalog provider"):
        load_python_catalog_provider("module_that_does_not_exist:entries")


@pytest.mark.parametrize("reference", ["", "entries", ":entries", "module:", "module:entry:extra"])
def test_python_provider_rejects_malformed_references(reference: str) -> None:
    with pytest.raises(ContractError, match="invalid Python catalog provider reference"):
        load_python_catalog_provider(reference)
