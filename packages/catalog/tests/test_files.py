"""Tests for manifest-controlled catalog layers."""

from pathlib import Path

import pytest
from posttrain.catalog import environment_factory_registry, load_catalog_layer, open_catalog
from posttrain.common import CatalogRef, ContractError, ModelVariant
from posttrain.eval import EnvironmentBinding, PythonFactoryActivation


def test_empty_overlay_is_valid_and_composes_with_packaged_base(tmp_path: Path) -> None:
    (tmp_path / "layer.yaml").write_text(
        "schema_version: 1\nlayer_id: empty-project-v1\nfiles: []\n",
        encoding="utf-8",
    )

    layer = load_catalog_layer(tmp_path)
    catalog = open_catalog(scope="empty-project", overlays=(tmp_path,))

    assert layer == {"layer_id": "empty-project-v1"}
    assert catalog.overlay_ids == ("empty-project-v1",)
    assert CatalogRef("model", "models/qwen3.5-0.8b@bf16") in catalog.list("model")
    assert CatalogRef("model", "models/gemma4-e4b-it@bf16") in catalog.list("model")
    assert CatalogRef("dataset", "datasets/posttrain-sft-smoke@1") in catalog.list("dataset")
    assert CatalogRef("environment", "math-gsm8k") in catalog.list("environment")
    environment = catalog.resolve(CatalogRef("environment", "math-gsm8k")).value
    assert isinstance(environment, EnvironmentBinding)
    assert environment.activation.kind == "verifiers-config"

    e4b = catalog.resolve(CatalogRef("model", "models/gemma4-e4b-it@bf16")).value
    assert isinstance(e4b, ModelVariant)
    assert e4b.id == "models/gemma4-e4b-it@bf16"
    assert e4b.parameters == 7_996_156_490
    assert e4b.base.repo_id == "google/gemma-4-E4B-it"
    assert e4b.base.revision == "ee0ef6023621cff504d758262d4e04895a5af4a2"
    assert e4b.renderer_contract == "gemma4-e4b-tools@1"


def test_catalog_manifest_still_rejects_duplicate_files(tmp_path: Path) -> None:
    (tmp_path / "layer.yaml").write_text(
        "schema_version: 1\nlayer_id: duplicate-v1\nfiles: [models.yaml, models.yaml]\n",
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="unique files"):
        load_catalog_layer(tmp_path)


def test_environment_factory_registry_does_not_load_package_entry_points(monkeypatch) -> None:
    class FakeEntryPoint:
        name = "published-environment"
        module = "published_environment"
        attr = "create_environment"

        @staticmethod
        def load():
            raise AssertionError("detached catalog loading must not import environments")

    monkeypatch.setattr("posttrain.catalog.entry_points", lambda **kwargs: (FakeEntryPoint(),))

    registry = environment_factory_registry()

    assert registry["published-environment"] == PythonFactoryActivation("published_environment:create_environment")
