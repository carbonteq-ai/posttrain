"""Provider-neutral dataset preparation operation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from posttrain.common import (
    JsonValue,
    LocalArtifactRef,
    ProducedArtifact,
    RunContext,
)

from .adapters import to_huggingface_preference_rows, to_huggingface_sft_rows
from .models import (
    DatasetDescriptor,
    PreferenceDataset,
    PreferenceDataSource,
    SupervisedDataset,
    SupervisedDataSource,
)

type DatasetPrepareSource = SupervisedDataSource | PreferenceDataSource


@dataclass(frozen=True, slots=True)
class DatasetPrepareRequest:
    """One already-resolved dataset snapshot to validate and canonicalize."""

    data: DatasetPrepareSource


@dataclass(frozen=True, slots=True)
class DatasetPrepareResult:
    """The canonical snapshot and retained dataset artifact."""

    descriptor: DatasetDescriptor
    content_sha256: str
    num_examples: int
    size_bytes: int
    native_artifact: ProducedArtifact


def prepare(
    context: RunContext,
    request: DatasetPrepareRequest,
) -> DatasetPrepareResult:
    """Validate and retain one deterministic supervised or preference snapshot."""

    source_descriptor = request.data.descriptor
    attributes = _attributes(source_descriptor)
    context.event("data_prepare_started", attributes)
    context.cancellation.raise_if_cancelled()

    loaded = request.data.load()
    context.cancellation.raise_if_cancelled()
    descriptor = loaded.descriptor
    _validate_loaded_descriptor(source_descriptor, descriptor)
    attributes = _attributes(descriptor)
    rows = _canonical_rows(loaded)
    serialized = "".join(_canonical_json(row) + "\n" for row in rows).encode()
    content_sha256 = hashlib.sha256(serialized).hexdigest()

    output_dir = (context.workspace / "data" / "prepared").resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    data_path = output_dir / "data.jsonl"
    data_path.write_bytes(serialized)
    manifest = {
        "schema_version": 1,
        "dataset_id": descriptor.id,
        "dataset_revision": descriptor.revision,
        "dataset_kind": descriptor.kind,
        "dataset_schema_version": descriptor.schema_version,
        "content_sha256": content_sha256,
        "examples": len(rows),
        "size_bytes": len(serialized),
        "data": data_path.name,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact = ProducedArtifact(
        name=f"data/{descriptor.id}/snapshot",
        kind="dataset",
        reference=LocalArtifactRef(output_dir, _directory_digest(output_dir)),
        metadata={
            **attributes,
            "content_sha256": content_sha256,
            "examples": len(rows),
            "size_bytes": len(serialized),
            **_source_metadata(descriptor.metadata),
        },
        role="dataset",
    )
    context.artifact(artifact)
    context.metrics(
        {
            "data/examples": len(rows),
            "data/bytes": len(serialized),
        },
        attributes=attributes,
    )
    context.event(
        "data_prepare_completed",
        {
            **attributes,
            "content_sha256": content_sha256,
            "examples": len(rows),
            "size_bytes": len(serialized),
        },
    )
    return DatasetPrepareResult(
        descriptor=descriptor,
        content_sha256=content_sha256,
        num_examples=len(rows),
        size_bytes=len(serialized),
        native_artifact=artifact,
    )


def _canonical_rows(
    loaded: SupervisedDataset | PreferenceDataset,
) -> list[dict[str, Any]]:
    if isinstance(loaded, SupervisedDataset):
        return to_huggingface_sft_rows(loaded)
    if isinstance(loaded, PreferenceDataset):
        return to_huggingface_preference_rows(loaded)
    raise TypeError(f"unsupported prepared dataset type: {type(loaded).__name__}")


def _validate_loaded_descriptor(
    source: DatasetDescriptor,
    loaded: DatasetDescriptor,
) -> None:
    if (
        source.id != loaded.id
        or source.revision != loaded.revision
        or source.kind != loaded.kind
        or source.schema_version != loaded.schema_version
    ):
        raise ValueError("loaded dataset conflicts with its source descriptor")
    if source.num_examples is not None and source.num_examples != loaded.num_examples:
        raise ValueError("loaded dataset example count conflicts with its source descriptor")


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _directory_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(child.read_bytes())
    return digest.hexdigest()


def _attributes(descriptor: DatasetDescriptor) -> dict[str, JsonValue]:
    return {
        "dataset_id": descriptor.id,
        "dataset_revision": descriptor.revision,
        "dataset_kind": descriptor.kind,
        "dataset_schema_version": descriptor.schema_version,
    }


def _source_metadata(
    metadata: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    retained: dict[str, JsonValue] = {}
    source_kind = metadata.get("source_kind")
    if isinstance(source_kind, str):
        retained["source_kind"] = source_kind
    source_digest = metadata.get("content_sha256")
    if isinstance(source_digest, str):
        retained["source_content_sha256"] = source_digest
    return retained


__all__ = [
    "DatasetPrepareRequest",
    "DatasetPrepareResult",
    "DatasetPrepareSource",
    "prepare",
]
