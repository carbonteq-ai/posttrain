"""Tests for manifest-controlled catalog layers."""

from pathlib import Path

import pytest
from posttrain.catalog import (
    CatalogLayerManifestSchema,
    environment_factory_registry,
    load_catalog_layer,
    open_catalog,
)
from posttrain.common import CatalogRef, ContractError
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
    assert CatalogRef("dataset", "datasets/posttrain-sft-smoke@1") in catalog.list("dataset")
    assert CatalogRef("environment", "math-gsm8k") in catalog.list("environment")
    environment = catalog.resolve(CatalogRef("environment", "math-gsm8k")).value
    assert isinstance(environment, EnvironmentBinding)
    assert environment.activation.kind == "verifiers-config"


def test_catalog_manifest_still_rejects_duplicate_files(tmp_path: Path) -> None:
    (tmp_path / "layer.yaml").write_text(
        "schema_version: 1\nlayer_id: duplicate-v1\nfiles: [models.yaml, models.yaml]\n",
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="unique files"):
        load_catalog_layer(tmp_path)


def test_schema_version_two_mixes_yaml_and_python_entries_in_one_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "layer.yaml").write_text(
        """
schema_version: 2
layer_id: mixed-project-v1
sources:
  - kind: yaml
    path: datasets.yaml
  - kind: python
    provider: catalog_fixture:entries
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "datasets.yaml").write_text(
        """
dataset:
  datasets/python-yaml@1:
    revision: "1"
    kind: supervised
    source:
      kind: fixture
      resource: posttrain.data.fixtures:sft_messages.jsonl
    format:
      kind: messages
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "builder_sentinel.py").write_text(
        "raise AssertionError('builder must not be imported during catalog loading')\n",
        encoding="utf-8",
    )
    (tmp_path / "catalog_fixture.py").write_text(
        """
from posttrain.catalog import CatalogEntries
from posttrain.common import CatalogRef, Workload

def entries():
    return CatalogEntries(entries={
        CatalogRef("workload", "workloads/python-entry@1"): Workload(
            id="workloads/python-entry@1",
            revision="1",
            requests={"builder": "builder_sentinel:build"},
        ),
    })
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    layer = load_catalog_layer(tmp_path)
    catalog = open_catalog(scope="mixed-project", overlays=(tmp_path,))

    assert layer["layer_id"] == "mixed-project-v1"
    assert CatalogRef("dataset", "datasets/python-yaml@1") in catalog.list("dataset")
    assert CatalogRef("workload", "workloads/python-entry@1") in catalog.list("workload")
    assert catalog.resolve(CatalogRef("workload", "workloads/python-entry@1")).value.id == "workloads/python-entry@1"
    assert "builder_sentinel" not in __import__("sys").modules


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (
            """
schema_version: 2
layer_id: duplicate-paths
sources:
  - kind: yaml
    path: data.yaml
  - kind: yaml
    path: data.yaml
""",
            "unique YAML source paths",
        ),
        (
            """
schema_version: 2
layer_id: duplicate-providers
sources:
  - kind: python
    provider: example:entries
  - kind: python
    provider: example:entries
""",
            "unique Python providers",
        ),
        (
            """
schema_version: 2
layer_id: traversal
sources:
  - kind: yaml
    path: ../data.yaml
""",
            "local YAML filenames",
        ),
        (
            """
schema_version: 2
layer_id: bad-provider
sources:
  - kind: python
    provider: not-a-reference
""",
            "MODULE:CALLABLE",
        ),
    ],
)
def test_schema_version_two_rejects_unsafe_or_duplicate_sources(
    tmp_path: Path,
    manifest: str,
    message: str,
) -> None:
    (tmp_path / "layer.yaml").write_text(manifest.lstrip(), encoding="utf-8")

    with pytest.raises(ContractError, match=message):
        load_catalog_layer(tmp_path)


def test_schema_version_two_rejects_duplicate_ids_across_yaml_and_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "layer.yaml").write_text(
        """
schema_version: 2
layer_id: duplicate-entry-v1
sources:
  - kind: yaml
    path: datasets.yaml
  - kind: python
    provider: duplicate_fixture:entries
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "datasets.yaml").write_text(
        """
workload:
  workloads/same@1:
    revision: "1"
    requests: {suite_id: yaml}
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "duplicate_fixture.py").write_text(
        """
from posttrain.catalog import CatalogEntries
from posttrain.common import CatalogRef, Workload

def entries():
    return CatalogEntries(entries={
        CatalogRef("workload", "workloads/same@1"): Workload(
            id="workloads/same@1", revision="1", requests={"suite_id": "python"}
        ),
    })
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ContractError, match="duplicate catalog ids.*workload/workloads/same@1"):
        load_catalog_layer(tmp_path)


def test_python_provider_failure_names_layer_and_reference(tmp_path: Path) -> None:
    (tmp_path / "layer.yaml").write_text(
        """
schema_version: 2
layer_id: provider-error-v1
sources:
  - kind: python
    provider: missing_provider:entries
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="provider-error-v1.*missing_provider:entries"):
        load_catalog_layer(tmp_path)


def test_schema_version_one_manifest_remains_unchanged() -> None:
    manifest = CatalogLayerManifestSchema(schema_version=1, layer_id="legacy", files=("a.yaml",))

    assert manifest.files == ("a.yaml",)
    assert manifest.sources is None


def test_schema_version_two_empty_source_list_is_valid(tmp_path: Path) -> None:
    (tmp_path / "layer.yaml").write_text(
        "schema_version: 2\nlayer_id: empty-v2\nsources: []\n",
        encoding="utf-8",
    )

    assert load_catalog_layer(tmp_path) == {"layer_id": "empty-v2"}


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
