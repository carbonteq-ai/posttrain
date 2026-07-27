"""Materialize and stage dataset selections for an actual-job image."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from posttrain.common import ContractError
from posttrain.data import DatasetLoadPlan, DatasetMaterialization, materialize_dataset
from posttrain.execution import DatasetPackageLock

from .contracts import DatasetPackRequest

type DatasetMaterializer = Callable[
    [DatasetLoadPlan, Path, Path],
    DatasetMaterialization,
]


@dataclass(frozen=True, slots=True)
class MaterializedDatasetPackages:
    """Portable dataset locks paired with their staged package root."""

    root: Path
    locks: tuple[DatasetPackageLock, ...]


class ImmutableDatasetPackager:
    """Normalize selected data at pack time and copy verified bytes into context."""

    def __init__(
        self,
        *,
        state_dir: Path,
        project_root: Path,
        materializer: DatasetMaterializer | None = None,
    ) -> None:
        if not state_dir.is_absolute() or not project_root.is_absolute():
            raise ContractError(
                "dataset packager state and project roots must be absolute"
            )
        self._state_dir = state_dir
        self._project_root = project_root
        self._materializer = materializer or _materialize

    def package(
        self,
        requests: Sequence[DatasetPackRequest],
        *,
        output_root: Path,
    ) -> MaterializedDatasetPackages:
        if not output_root.is_absolute():
            raise ContractError("dataset package output root must be absolute")
        ordered = tuple(sorted(requests, key=lambda item: item.seat_name))
        if len({request.seat_name for request in ordered}) != len(ordered):
            raise ContractError("dataset package seat names must be unique")

        locks: list[DatasetPackageLock] = []
        for request in ordered:
            materialized = self._materializer(
                request.selection,
                self._state_dir,
                self._project_root,
            )
            _verify_materialization(request.selection, materialized)
            relative_root = (
                Path("datasets")
                / _dataset_slug(request)
                / materialized.content_sha256
            )
            destination = output_root / relative_root
            destination.mkdir(parents=True, exist_ok=False)
            data_path = destination / "data.jsonl"
            manifest_path = destination / "manifest.json"
            shutil.copyfile(materialized.path, data_path)
            shutil.copyfile(materialized.manifest_path, manifest_path)
            _verify_copy(data_path, materialized.content_sha256)
            manifest = _read_manifest(manifest_path)
            locks.append(
                DatasetPackageLock(
                    seat_name=request.seat_name,
                    selection_id=request.selection.id,
                    selection_revision=request.selection.revision,
                    dataset_revision=request.selection.dataset_revision,
                    kind=request.selection.kind,
                    schema_version=_positive_int(
                        manifest.get("schema_version"), "dataset schema version"
                    ),
                    digest=materialized.content_sha256,
                    package_path=(relative_root / "data.jsonl").as_posix(),
                    manifest_path=(relative_root / "manifest.json").as_posix(),
                    size_bytes=data_path.stat().st_size,
                    num_records=materialized.examples,
                )
            )
        return MaterializedDatasetPackages(output_root, tuple(locks))


def _materialize(
    plan: DatasetLoadPlan,
    state_dir: Path,
    project_root: Path,
) -> DatasetMaterialization:
    return materialize_dataset(
        plan,
        state_dir=state_dir,
        project_root=project_root,
    )


def _verify_materialization(
    plan: DatasetLoadPlan,
    materialized: DatasetMaterialization,
) -> None:
    if (
        materialized.selection_id != plan.id
        or materialized.selection_revision != plan.revision
        or materialized.source_kind != plan.source_kind
    ):
        raise ContractError("dataset materialization conflicts with its selection")
    if (
        not materialized.path.is_file()
        or not materialized.manifest_path.is_file()
    ):
        raise ContractError("dataset materialization files are missing")
    _verify_copy(materialized.path, materialized.content_sha256)
    manifest = _read_manifest(materialized.manifest_path)
    expected = {
        "selection_id": plan.id,
        "selection_revision": plan.revision,
        "dataset_revision": plan.dataset_revision,
        "source_kind": plan.source_kind,
        "content_sha256": materialized.content_sha256,
        "examples": materialized.examples,
        "data": materialized.path.name,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ContractError(
                f"dataset materialization manifest has invalid {key}"
            )
    _positive_int(manifest.get("schema_version"), "dataset schema version")


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("dataset materialization manifest is invalid") from error
    if not isinstance(value, dict):
        raise ContractError("dataset materialization manifest must be an object")
    return value


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ContractError(f"{label} must be a positive integer")
    return value


def _verify_copy(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ContractError("dataset content digest does not match its lock")


def _dataset_slug(request: DatasetPackRequest) -> str:
    seat = re.sub(r"[^A-Za-z0-9._-]+", "-", request.seat_name).strip("-")
    identity = hashlib.sha256(
        (
            request.seat_name
            + "\0"
            + request.selection.id
            + "\0"
            + request.selection.revision
        ).encode()
    ).hexdigest()[:16]
    return f"{seat}-{identity}"
