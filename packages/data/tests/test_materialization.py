"""Tests for typed Python dataset builders and reproducible materialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from posttrain.common import CatalogRef, ContractError
from posttrain.data import (
    BuiltDatasetSource,
    DatasetAccessPolicy,
    DatasetProvenance,
    DatasetSelection,
    LocalDatasetInput,
    PythonDatasetBuilder,
    build_key,
    decode_dataset_selection,
    materialize_dataset,
)


def test_catalog_decodes_typed_python_source_shape() -> None:
    plan = decode_dataset_selection(
        CatalogRef("dataset", "datasets/typed-yaml@1"),
        {
            "id": "datasets/typed-yaml@1",
            "revision": "1",
            "kind": "supervised",
            "source": {
                "kind": "built",
                "builder": {"kind": "python", "target": "project.build:build"},
                "inputs": {"raw": {"kind": "local", "path": "data.jsonl"}},
            },
            "format": {"kind": "messages"},
        },
        {},
    )
    assert isinstance(plan.source, BuiltDatasetSource)
    assert plan.source.builder.target == "project.build:build"
    assert isinstance(plan.source.inputs["raw"], LocalDatasetInput)


def _project(tmp_path: Path, *, invalid: bool = False) -> DatasetSelection:
    (tmp_path / "raw.jsonl").write_text(
        '{"value": "first"}\n{"value": "second"}\n',
        encoding="utf-8",
    )
    (tmp_path / "builder.py").write_text(
        """from collections.abc import Mapping\n\ndef build(ctx):\n    for row in ctx.records('raw'):\n        if __INVALID__:\n            yield 'not-a-row'\n        else:\n            yield {'messages': [{'role': 'user', 'content': row['value']}, {'role': 'assistant', 'content': 'ok'}]}\n""".replace(
            "__INVALID__", "True" if invalid else "False"
        ),
        encoding="utf-8",
    )
    return DatasetSelection(
        id="datasets/typed@1",
        revision="1",
        kind="supervised",
        split="train",
        schema_version="messages-v1",
        provenance=DatasetProvenance(upstream=("fixture@1",), transformation="unit-test"),
        access=DatasetAccessPolicy(licenses=("Apache-2.0",), classification="test"),
        source=BuiltDatasetSource(
            builder=PythonDatasetBuilder(target="builder:build"),
            inputs={"raw": LocalDatasetInput("raw.jsonl")},
        ),
        format="messages",
    )


def test_typed_builder_materializes_in_child_process_and_reuses_cache(tmp_path: Path) -> None:
    plan = _project(tmp_path)
    first = materialize_dataset(plan, state_dir=tmp_path / "state", project_root=tmp_path)
    second = materialize_dataset(plan, state_dir=tmp_path / "state", project_root=tmp_path)

    assert first.created is True
    assert second.created is False
    assert first.build_key
    assert first.build_key == second.build_key
    assert first.path == second.path
    assert first.examples == 2
    assert first.content_sha256 == hashlib.sha256(first.path.read_bytes()).hexdigest()
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["build_key"] == first.build_key
    assert (
        manifest["inputs"]["raw"]["content_sha256"] == hashlib.sha256((tmp_path / "raw.jsonl").read_bytes()).hexdigest()
    )


def test_imported_project_code_changes_build_key(tmp_path: Path) -> None:
    plan = _project(tmp_path)
    first_key = build_key(plan, project_root=tmp_path)
    (tmp_path / "helper.py").write_text("VALUE = 'one'\n", encoding="utf-8")
    second_key = build_key(plan, project_root=tmp_path)
    (tmp_path / "helper.py").write_text("VALUE = 'two'\n", encoding="utf-8")
    third_key = build_key(plan, project_root=tmp_path)

    assert first_key != second_key
    assert second_key != third_key


def test_expected_content_digest_is_enforced_before_cache_promotion(tmp_path: Path) -> None:
    plan = _project(tmp_path)
    source = plan.source
    assert isinstance(source, BuiltDatasetSource)
    checked = DatasetSelection(
        id=plan.id,
        revision=plan.revision,
        kind=plan.kind,
        split=plan.split,
        schema_version=plan.schema_version,
        provenance=plan.provenance,
        access=plan.access,
        source=BuiltDatasetSource(
            builder=source.builder,
            inputs=source.inputs,
            expected_content_sha256="0" * 64,
        ),
        format=plan.format,
    )

    with pytest.raises(ContractError, match="content digest mismatch"):
        materialize_dataset(checked, state_dir=tmp_path / "state", project_root=tmp_path)
    assert not list((tmp_path / "state" / "datasets").glob("*/data.jsonl"))


def test_invalid_builder_rows_fail_without_leaving_cache(tmp_path: Path) -> None:
    plan = _project(tmp_path, invalid=True)
    with pytest.raises(ContractError, match="row 0 must be a mapping"):
        materialize_dataset(plan, state_dir=tmp_path / "state", project_root=tmp_path)
    assert not list((tmp_path / "state" / "datasets").glob("*/manifest.json"))


def test_builder_import_failure_is_actionable(tmp_path: Path) -> None:
    (tmp_path / "raw.jsonl").write_text('{"value": "x"}\n', encoding="utf-8")
    plan = DatasetSelection(
        id="datasets/missing@1",
        revision="1",
        kind="supervised",
        source=BuiltDatasetSource(
            builder=PythonDatasetBuilder(target="does_not_exist:build"),
            inputs={"raw": LocalDatasetInput("raw.jsonl")},
        ),
        format="messages",
    )
    with pytest.raises(ContractError, match="does_not_exist:build"):
        materialize_dataset(plan, state_dir=tmp_path / "state", project_root=tmp_path)


def test_builder_cannot_access_undeclared_inputs(tmp_path: Path) -> None:
    (tmp_path / "raw.jsonl").write_text('{"value": "x"}\n', encoding="utf-8")
    (tmp_path / "builder.py").write_text(
        "def build(ctx):\n    return ctx.records('not-declared')\n",
        encoding="utf-8",
    )
    plan = DatasetSelection(
        id="datasets/undeclared@1",
        revision="1",
        kind="supervised",
        source=BuiltDatasetSource(
            builder=PythonDatasetBuilder(target="builder:build"),
            inputs={"raw": LocalDatasetInput("raw.jsonl")},
        ),
        format="messages",
    )
    with pytest.raises(ContractError, match="undeclared input"):
        materialize_dataset(plan, state_dir=tmp_path / "state", project_root=tmp_path)
