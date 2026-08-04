from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from posttrain.data import (
    BuiltDatasetSource,
    DatasetLoadPlan,
    DatasetMaterialization,
    LocalDatasetInput,
    PythonDatasetBuilder,
)
from posttrain.execution_pack import DatasetPackRequest, ImmutableDatasetPackager


def _plan() -> DatasetLoadPlan:
    return DatasetLoadPlan(
        id="datasets/sft-smoke@1",
        revision="1",
        kind="supervised",
        source={"kind": "fixture", "resource": "example:data.jsonl"},
        format="messages",
    )


def _materializer(
    plan: DatasetLoadPlan,
    state_dir: Path,
    project_root: Path,
) -> DatasetMaterialization:
    assert state_dir.is_absolute()
    assert project_root.is_absolute()
    root = state_dir / "fake-materialization"
    root.mkdir(parents=True, exist_ok=True)
    data = b'{"messages":[]}\n'
    digest = hashlib.sha256(data).hexdigest()
    path = root / "data.jsonl"
    path.write_bytes(data)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selection_id": plan.id,
                "selection_revision": plan.revision,
                "dataset_revision": plan.dataset_revision,
                "source_kind": plan.source_kind,
                "content_sha256": digest,
                "examples": 1,
                "data": "data.jsonl",
            }
        ),
        encoding="utf-8",
    )
    return DatasetMaterialization(
        selection_id=plan.id,
        selection_revision=plan.revision,
        source_kind=plan.source_kind,
        path=path,
        manifest_path=manifest_path,
        content_sha256=digest,
        examples=1,
        created=True,
    )


def test_packages_verified_dataset_with_portable_seat_lock(
    tmp_path: Path,
) -> None:
    packager = ImmutableDatasetPackager(
        state_dir=(tmp_path / "state").resolve(),
        project_root=tmp_path.resolve(),
        materializer=_materializer,
    )
    output = (tmp_path / "context").resolve()

    result = packager.package(
        (DatasetPackRequest("dataset", _plan()),),
        output_root=output,
    )

    assert len(result.locks) == 1
    lock = result.locks[0]
    assert lock.seat_name == "dataset"
    assert lock.selection_revision == "1"
    assert lock.dataset_revision == "1"
    assert lock.kind == "supervised"
    assert lock.package_path.startswith("datasets/dataset-")
    assert (output / lock.package_path).is_file()
    assert (output / lock.manifest_path).is_file()


def test_rejects_duplicate_dataset_seats(tmp_path: Path) -> None:
    packager = ImmutableDatasetPackager(
        state_dir=(tmp_path / "state").resolve(),
        project_root=tmp_path.resolve(),
        materializer=_materializer,
    )
    request = DatasetPackRequest("dataset", _plan())

    with pytest.raises(ValueError, match="seat names must be unique"):
        packager.package(
            (request, request),
            output_root=(tmp_path / "context").resolve(),
        )


def test_typed_builder_request_serializes_without_importing_builder() -> None:
    selection = DatasetLoadPlan(
        id="datasets/python-reviewed@1",
        revision="1",
        kind="supervised",
        source=BuiltDatasetSource(
            builder=PythonDatasetBuilder("example.datasets:build"),
            inputs={"source": LocalDatasetInput("data/source.jsonl")},
        ),
        format="messages",
    )

    payload = DatasetPackRequest("dataset", selection).to_payload()

    assert payload["source"] == {
        "kind": "built",
        "builder": {"kind": "python", "target": "example.datasets:build"},
        "inputs": {"source": {"kind": "local", "path": "data/source.jsonl", "format": "jsonl"}},
    }


def test_typed_builder_is_materialized_and_locked_in_the_job_context(tmp_path: Path) -> None:
    code_snapshot_digest = "c" * 64
    dependency_lock_digest = "d" * 64
    (tmp_path / "raw.jsonl").write_text('{"value":"hello"}\n', encoding="utf-8")
    (tmp_path / "builder.py").write_text(
        "def build(ctx):\n"
        "    for row in ctx.records('raw'):\n"
        "        yield {'messages': [{'role': 'user', 'content': row['value']}, "
        "{'role': 'assistant', 'content': 'ok'}]}\n",
        encoding="utf-8",
    )
    selection = DatasetLoadPlan(
        id="datasets/python-reviewed@1",
        revision="1",
        kind="supervised",
        source=BuiltDatasetSource(
            builder=PythonDatasetBuilder("builder:build"),
            inputs={"raw": LocalDatasetInput("raw.jsonl")},
        ),
        format="messages",
    )
    context = tmp_path / "context"
    result = ImmutableDatasetPackager(
        state_dir=(tmp_path / "state").resolve(),
        project_root=tmp_path.resolve(),
        code_snapshot_digest=code_snapshot_digest,
        dependency_lock_digest=dependency_lock_digest,
    ).package(
        (DatasetPackRequest("dataset", selection),),
        output_root=context.resolve(),
    )

    lock = result.locks[0]
    assert lock.build_key
    assert lock.materializer_schema_version == 2
    assert lock.builder_target == "builder:build"
    assert lock.code_snapshot_digest == code_snapshot_digest
    assert lock.dependency_lock_digest == dependency_lock_digest
    manifest = json.loads((context / lock.manifest_path).read_text(encoding="utf-8"))
    assert manifest["build_key"] == lock.build_key
    assert manifest["content_sha256"] == lock.digest
    assert manifest["builder_target"] == lock.builder_target
    assert (context / lock.package_path).read_text(encoding="utf-8").count("hello") == 1
