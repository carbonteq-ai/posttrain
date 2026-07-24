"""Declarative dataset selections and local materialization."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.resources import files as resource_files
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from posttrain.common import CatalogRef, ContractError, JsonValue
from posttrain.common.catalog import SelectionDecoder
from posttrain.common.selections import Selection, SelectionFamily, validate_revision, validate_selection_id

from .adapters import (
    PreferenceFormat,
    SFTFormat,
    preferences_from_huggingface,
    preferences_from_nemo,
    supervised_from_huggingface,
    supervised_from_nemo,
    to_huggingface_preference_rows,
    to_huggingface_sft_rows,
)
from .models import PreferenceDataset, SupervisedDataset

type DatasetKind = Literal["supervised", "preference"]
type DatasetSourceKind = Literal["fixture", "huggingface", "jsonl", "nemo", "parquet"]

_SFT_FORMATS = frozenset({"auto", "messages", "prompt-completion", "alpaca", "sharegpt"})
_PREFERENCE_FORMATS = frozenset({"auto", "trl", "tulu", "nemo-ranked"})
_NEMO_SFT_FORMATS = frozenset({"auto", "messages"})
_NEMO_PREFERENCE_FORMATS = frozenset({"auto", "nemo-ranked"})
_PATH_SOURCE_KINDS = frozenset({"jsonl", "nemo", "parquet"})


@dataclass(frozen=True, slots=True)
class DatasetLoadPlan:
    """A catalog selection describing how to resolve one dataset."""

    id: str
    revision: str
    kind: DatasetKind
    source: Mapping[str, JsonValue]
    format: str

    def __post_init__(self) -> None:
        validate_selection_id(self.id, "dataset selection id")
        validate_revision(self.revision, "dataset selection revision")
        source = dict(self.source)
        source_kind = source.get("kind")
        if source_kind not in {"fixture", "huggingface", "jsonl", "nemo", "parquet"}:
            raise ContractError("dataset source kind must be fixture, huggingface, jsonl, nemo, or parquet")
        _validate_source(cast(DatasetSourceKind, source_kind), source)
        if source_kind == "nemo":
            allowed_formats = _NEMO_SFT_FORMATS if self.kind == "supervised" else _NEMO_PREFERENCE_FORMATS
        else:
            allowed_formats = _SFT_FORMATS if self.kind == "supervised" else _PREFERENCE_FORMATS
        if self.format not in allowed_formats:
            allowed = ", ".join(sorted(allowed_formats))
            raise ContractError(f"{self.kind} dataset format must be one of: {allowed}")
        object.__setattr__(self, "source", MappingProxyType(source))

    @property
    def source_kind(self) -> DatasetSourceKind:
        return cast(DatasetSourceKind, self.source["kind"])

    @property
    def dataset_revision(self) -> str:
        source_revision = self.source.get("revision")
        return source_revision if isinstance(source_revision, str) else self.revision


@dataclass(frozen=True, slots=True)
class DatasetMaterialization:
    """Result of resolving and validating a dataset into project-local state."""

    selection_id: str
    selection_revision: str
    source_kind: DatasetSourceKind
    path: Path
    manifest_path: Path
    content_sha256: str
    examples: int
    created: bool


def decode_dataset_selection(
    ref: CatalogRef,
    data: Mapping[str, object],
    _known: Mapping[CatalogRef, Selection],
) -> DatasetLoadPlan:
    """Decode the public dataset catalog shape into a reusable load plan."""

    allowed = {"id", "revision", "kind", "source", "format"}
    unexpected = sorted(set(data).difference(allowed))
    if unexpected:
        raise ContractError(f"dataset catalog entry {ref.id!r} has unknown fields: {', '.join(unexpected)}")
    selection_id = data.get("id")
    revision = data.get("revision")
    kind = data.get("kind", "supervised")
    source = data.get("source")
    raw_format = data.get("format", {"kind": "auto"})
    if not isinstance(selection_id, str) or not isinstance(revision, str):
        raise ContractError(f"dataset catalog entry {ref.id!r} requires string id and revision")
    if kind not in {"supervised", "preference"}:
        raise ContractError(f"dataset catalog entry {ref.id!r} kind must be supervised or preference")
    if not isinstance(source, Mapping):
        raise ContractError(f"dataset catalog entry {ref.id!r} requires a source object")
    if not isinstance(raw_format, Mapping) or set(raw_format) != {"kind"}:
        raise ContractError(f"dataset catalog entry {ref.id!r} format must contain exactly kind")
    format_kind = raw_format.get("kind")
    if not isinstance(format_kind, str):
        raise ContractError(f"dataset catalog entry {ref.id!r} format kind must be a string")
    return DatasetLoadPlan(
        id=selection_id,
        revision=revision,
        kind=cast(DatasetKind, kind),
        source=cast(Mapping[str, JsonValue], source),
        format=format_kind,
    )


DATA_CATALOG_DECODERS: Mapping[SelectionFamily, SelectionDecoder] = {"dataset": decode_dataset_selection}


def materialize_dataset(
    plan: DatasetLoadPlan,
    *,
    state_dir: Path,
    project_root: Path,
) -> DatasetMaterialization:
    """Resolve, normalize, validate, and cache a dataset load plan."""

    fingerprint = hashlib.sha256(_plan_json(plan).encode("utf-8")).hexdigest()
    destination = state_dir / "datasets" / fingerprint
    data_path = destination / "data.jsonl"
    manifest_path = destination / "manifest.json"
    if data_path.is_file() and manifest_path.is_file():
        return _read_materialization(plan, data_path, manifest_path, created=False)

    rows = tuple(_source_rows(plan, project_root=project_root))
    dataset_id = plan.id.rsplit("@", maxsplit=1)[0]
    try:
        dataset = _adapt_rows(plan, rows, dataset_id=dataset_id)
        if plan.kind == "supervised":
            normalized = to_huggingface_sft_rows(cast(SupervisedDataset, dataset))
        else:
            normalized = to_huggingface_preference_rows(cast(PreferenceDataset, dataset))
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError(f"dataset {plan.id!r} failed adapter validation: {error}") from error

    serialized = "".join(json.dumps(row, sort_keys=True) + "\n" for row in normalized)
    content_sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    destination.mkdir(parents=True, exist_ok=True)
    data_path.write_text(serialized, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "selection_id": plan.id,
        "selection_revision": plan.revision,
        "dataset_revision": plan.dataset_revision,
        "source_kind": plan.source_kind,
        "content_sha256": content_sha256,
        "examples": len(normalized),
        "data": data_path.name,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _read_materialization(plan, data_path, manifest_path, created=True)


def load_materialized_dataset(
    plan: DatasetLoadPlan,
    materialization: DatasetMaterialization,
) -> SupervisedDataset | PreferenceDataset:
    """Load a validated cache through the same public adapters used at first resolve."""

    if materialization.selection_id != plan.id or materialization.selection_revision != plan.revision:
        raise ContractError("dataset materialization does not match the requested load plan")
    rows = _jsonl_rows(materialization.path.read_text(encoding="utf-8"), source=str(materialization.path))
    dataset_id = plan.id.rsplit("@", maxsplit=1)[0]
    if plan.kind == "supervised":
        return supervised_from_huggingface(
            rows,
            dataset_id=dataset_id,
            revision=plan.dataset_revision,
            format="messages",
            metadata={
                "source_kind": plan.source_kind,
                "content_sha256": materialization.content_sha256,
                "materialized_path": str(materialization.path),
            },
        )
    return preferences_from_huggingface(
        rows,
        dataset_id=dataset_id,
        revision=plan.dataset_revision,
        format="trl",
        metadata={
            "source_kind": plan.source_kind,
            "content_sha256": materialization.content_sha256,
            "materialized_path": str(materialization.path),
        },
    )


def resolve_dataset_source(
    plan: DatasetLoadPlan,
    *,
    state_dir: Path,
    project_root: Path,
) -> SupervisedDataset | PreferenceDataset:
    """Materialize on first use and return a canonical trainer-neutral source."""

    materialization = materialize_dataset(plan, state_dir=state_dir, project_root=project_root)
    return load_materialized_dataset(plan, materialization)


def _adapt_rows(
    plan: DatasetLoadPlan,
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset_id: str,
) -> SupervisedDataset | PreferenceDataset:
    metadata = {"source_kind": plan.source_kind}
    if plan.source_kind == "nemo":
        if plan.kind == "supervised":
            return supervised_from_nemo(
                rows,
                dataset_id=dataset_id,
                revision=plan.dataset_revision,
                metadata=metadata,
            )
        return preferences_from_nemo(
            rows,
            dataset_id=dataset_id,
            revision=plan.dataset_revision,
            metadata=metadata,
        )
    if plan.kind == "supervised":
        return supervised_from_huggingface(
            rows,
            dataset_id=dataset_id,
            revision=plan.dataset_revision,
            format=cast(SFTFormat, plan.format),
            metadata=metadata,
        )
    return preferences_from_huggingface(
        rows,
        dataset_id=dataset_id,
        revision=plan.dataset_revision,
        format=cast(PreferenceFormat, plan.format),
        metadata=metadata,
    )


def _validate_source(kind: DatasetSourceKind, source: Mapping[str, JsonValue]) -> None:
    required: dict[DatasetSourceKind, frozenset[str]] = {
        "fixture": frozenset({"kind", "resource"}),
        "huggingface": frozenset({"kind", "repo", "revision", "split"}),
        "jsonl": frozenset({"kind", "path"}),
        "nemo": frozenset({"kind", "path"}),
        "parquet": frozenset({"kind", "path"}),
    }
    optional: dict[DatasetSourceKind, frozenset[str]] = {
        "fixture": frozenset(),
        "huggingface": frozenset({"config"}),
        "jsonl": frozenset(),
        "nemo": frozenset(),
        "parquet": frozenset({"split"}),
    }
    keys = set(source)
    missing = required[kind].difference(keys)
    unexpected = keys.difference(required[kind] | optional[kind])
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {', '.join(sorted(missing))}")
        if unexpected:
            details.append(f"unknown {', '.join(sorted(unexpected))}")
        raise ContractError(f"{kind} dataset source is invalid: {'; '.join(details)}")
    for key in keys.difference({"kind"}):
        value = source[key]
        if not isinstance(value, str) or not value.strip():
            raise ContractError(f"{kind} dataset source {key} must be a non-empty string")


def _project_path(plan: DatasetLoadPlan, *, project_root: Path) -> Path:
    configured = Path(cast(str, plan.source["path"]))
    if configured.is_absolute():
        raise ContractError(f"{plan.source_kind} dataset path must be relative to the project root")
    path = (project_root / configured).resolve()
    if not path.is_relative_to(project_root.resolve()):
        raise ContractError(f"{plan.source_kind} dataset path escapes the project root: {configured}")
    return path


def _source_rows(plan: DatasetLoadPlan, *, project_root: Path) -> Iterable[Mapping[str, Any]]:
    if plan.source_kind == "fixture":
        resource = cast(str, plan.source["resource"])
        package, separator, name = resource.partition(":")
        if not separator or not package or not name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ContractError("fixture dataset resource must use PACKAGE:RELATIVE_PATH syntax")
        text = resource_files(package).joinpath(name).read_text(encoding="utf-8")
        return _jsonl_rows(text, source=resource)
    if plan.source_kind in _PATH_SOURCE_KINDS:
        configured = cast(str, plan.source["path"])
        path = _project_path(plan, project_root=project_root)
        if plan.source_kind in {"jsonl", "nemo"}:
            try:
                return _jsonl_rows(path.read_text(encoding="utf-8"), source=configured)
            except FileNotFoundError as error:
                raise ContractError(f"{plan.source_kind} dataset source not found: {path}") from error
        if not path.is_file():
            raise ContractError(f"parquet dataset source not found: {path}")
        return _load_huggingface_rows(
            "parquet",
            data_files=str(path),
            split=cast(str, plan.source.get("split", "train")),
        )

    return _load_huggingface_rows(
        cast(str, plan.source["repo"]),
        cast(str | None, plan.source.get("config")),
        revision=cast(str, plan.source["revision"]),
        split=cast(str, plan.source["split"]),
    )


def _load_huggingface_rows(path: str, name: str | None = None, **kwargs: Any) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ContractError(
            "Hugging Face and Parquet dataset materialization requires posttrain-data[huggingface]"
        ) from error
    loaded = load_dataset(path, name, **kwargs)
    return cast(Iterable[Mapping[str, Any]], loaded)


def _jsonl_rows(text: str, *, source: str) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    for index, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ContractError(f"invalid JSONL in {source} at line {index}: {error}") from error
        if not isinstance(row, Mapping):
            raise ContractError(f"JSONL row in {source} at line {index} must be an object")
        rows.append(cast(Mapping[str, Any], row))
    if not rows:
        raise ContractError(f"dataset source {source} contains no rows")
    return tuple(rows)


def _plan_json(plan: DatasetLoadPlan) -> str:
    return json.dumps(
        {
            "id": plan.id,
            "revision": plan.revision,
            "kind": plan.kind,
            "source": dict(plan.source),
            "format": plan.format,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _read_materialization(
    plan: DatasetLoadPlan,
    data_path: Path,
    manifest_path: Path,
    *,
    created: bool,
) -> DatasetMaterialization:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        content = data_path.read_bytes()
    except (json.JSONDecodeError, OSError) as error:
        raise ContractError(f"invalid materialized dataset cache at {manifest_path.parent}: {error}") from error
    actual_digest = hashlib.sha256(content).hexdigest()
    expected = manifest.get("content_sha256")
    if expected != actual_digest:
        raise ContractError(f"materialized dataset cache digest mismatch at {data_path}")
    if manifest.get("selection_id") != plan.id or manifest.get("selection_revision") != plan.revision:
        raise ContractError(f"materialized dataset cache identity mismatch at {manifest_path}")
    if manifest.get("dataset_revision") != plan.dataset_revision:
        raise ContractError(f"materialized dataset cache source revision mismatch at {manifest_path}")
    examples = manifest.get("examples")
    if not isinstance(examples, int) or examples < 1:
        raise ContractError(f"materialized dataset cache has invalid example count at {manifest_path}")
    return DatasetMaterialization(
        selection_id=plan.id,
        selection_revision=plan.revision,
        source_kind=plan.source_kind,
        path=data_path,
        manifest_path=manifest_path,
        content_sha256=actual_digest,
        examples=examples,
        created=created,
    )


__all__ = [
    "DATA_CATALOG_DECODERS",
    "DatasetLoadPlan",
    "DatasetMaterialization",
    "decode_dataset_selection",
    "load_materialized_dataset",
    "materialize_dataset",
    "resolve_dataset_source",
]
