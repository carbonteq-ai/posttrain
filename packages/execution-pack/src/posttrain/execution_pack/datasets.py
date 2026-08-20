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
from posttrain.execution import DatasetAssetLock, DatasetPackageLock

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
        input_root: Path | None = None,
        materializer: DatasetMaterializer | None = None,
        code_snapshot_digest: str | None = None,
        dependency_lock_digest: str | None = None,
    ) -> None:
        if not state_dir.is_absolute() or not project_root.is_absolute():
            raise ContractError("dataset packager state and project roots must be absolute")
        if input_root is not None and not input_root.is_absolute():
            raise ContractError("dataset packager input root must be absolute")
        self._state_dir = state_dir
        self._project_root = project_root
        self._input_root = input_root or project_root
        self._materializer = materializer
        self._code_snapshot_digest = _optional_digest(code_snapshot_digest, "dataset code snapshot digest")
        self._dependency_lock_digest = _optional_digest(
            dependency_lock_digest,
            "dataset dependency lock digest",
        )

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
            if self._materializer is None:
                materialized = materialize_dataset(
                    request.selection,
                    state_dir=self._state_dir,
                    project_root=self._project_root,
                    input_root=self._input_root,
                    code_snapshot_digest=self._code_snapshot_digest,
                    dependency_lock_digest=self._dependency_lock_digest,
                )
            else:
                materialized = self._materializer(
                    request.selection,
                    self._state_dir,
                    self._project_root,
                )
            _verify_materialization(request.selection, materialized)
            relative_root = Path("datasets") / _dataset_slug(request) / materialized.content_sha256
            destination = output_root / relative_root
            destination.mkdir(parents=True, exist_ok=False)
            data_path = destination / "data.jsonl"
            manifest_path = destination / "manifest.json"
            shutil.copyfile(materialized.path, data_path)
            shutil.copyfile(materialized.manifest_path, manifest_path)
            _verify_copy(data_path, materialized.content_sha256)
            manifest = _read_manifest(manifest_path)
            asset_locks: list[DatasetAssetLock] = []
            for asset in materialized.assets:
                source = materialized.manifest_path.parent.joinpath(*Path(asset.path).parts)
                target = destination.joinpath(*Path(asset.path).parts)
                if source.is_symlink() or not source.is_file():
                    raise ContractError(f"dataset materialization asset is not a regular file: {asset.path}")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                _verify_copy(target, asset.sha256)
                if target.stat().st_size != asset.size_bytes:
                    raise ContractError(f"dataset materialization asset size differs from its manifest: {asset.path}")
                asset_locks.append(
                    DatasetAssetLock(
                        package_path=(relative_root / asset.path).as_posix(),
                        digest=asset.sha256,
                        size_bytes=asset.size_bytes,
                    )
                )
            locks.append(
                DatasetPackageLock(
                    seat_name=request.seat_name,
                    selection_id=request.selection.id,
                    selection_revision=request.selection.revision,
                    dataset_revision=request.selection.dataset_revision,
                    kind=request.selection.kind,
                    schema_version=_positive_int(manifest.get("schema_version"), "dataset schema version"),
                    digest=materialized.content_sha256,
                    package_path=(relative_root / "data.jsonl").as_posix(),
                    manifest_path=(relative_root / "manifest.json").as_posix(),
                    size_bytes=data_path.stat().st_size,
                    num_records=materialized.examples,
                    build_key=_optional_digest(manifest.get("build_key"), "dataset build key"),
                    materializer_schema_version=_optional_positive_int(
                        manifest.get("schema_version"), "dataset materializer schema version"
                    ),
                    builder_target=_optional_target(manifest.get("builder_target")),
                    code_snapshot_digest=_optional_digest(
                        manifest.get("code_snapshot_digest"), "dataset code snapshot digest"
                    ),
                    dependency_lock_digest=_optional_digest(
                        manifest.get("dependency_lock_digest"), "dataset dependency lock digest"
                    ),
                    assets=tuple(asset_locks),
                    assets_digest=materialized.assets_digest,
                )
            )
        return MaterializedDatasetPackages(output_root, tuple(locks))


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
    if not materialized.path.is_file() or not materialized.manifest_path.is_file():
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
            raise ContractError(f"dataset materialization manifest has invalid {key}")
    _positive_int(manifest.get("schema_version"), "dataset schema version")
    if materialized.build_key:
        if manifest.get("build_key") != materialized.build_key:
            raise ContractError("dataset materialization build key differs from its manifest")
    expected_assets = [asset.to_payload() for asset in materialized.assets]
    if materialized.assets:
        if manifest.get("assets") != expected_assets or manifest.get("assets_digest") != materialized.assets_digest:
            raise ContractError("dataset materialization assets differ from its manifest")
        asset_root = materialized.manifest_path.parent / "assets"
        if not asset_root.is_dir() or asset_root.is_symlink():
            raise ContractError("dataset materialization asset directory is missing")
        observed: set[str] = set()
        for path in asset_root.rglob("*"):
            if path.is_symlink() or (not path.is_dir() and not path.is_file()):
                raise ContractError("dataset materialization assets contain a symlink or special file")
            if path.is_file():
                observed.add(path.relative_to(materialized.manifest_path.parent).as_posix())
        if observed != {asset.path for asset in materialized.assets}:
            raise ContractError("dataset materialization asset files differ from its manifest")
    elif manifest.get("assets") is not None or manifest.get("assets_digest") is not None:
        raise ContractError("text dataset materialization cannot declare asset metadata")


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


def _optional_positive_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, label)


def _optional_digest(value: object, label: str) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ContractError(f"{label} must be a SHA-256 digest")
    return value


def _optional_target(value: object) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*", value) is None:
        raise ContractError("dataset builder target must use module:callable syntax")
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
        (request.seat_name + "\0" + request.selection.id + "\0" + request.selection.revision).encode()
    ).hexdigest()[:16]
    return f"{seat}-{identity}"
