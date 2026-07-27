from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from posttrain.data import DatasetLoadPlan, DatasetMaterialization
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
