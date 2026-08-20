"""Declarative dataset selections and local materialization."""

from __future__ import annotations

import hashlib
import json
import re
import runpy
import shutil
import tempfile
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.resources import files as resource_files
from pathlib import Path, PurePosixPath
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
from .definitions import (
    BuiltDatasetSource,
    DatasetAccessPolicy,
    DatasetKind,
    DatasetProvenance,
    DatasetSource,
    HuggingFaceDatasetInput,
    LocalDatasetInput,
    PackageResourceInput,
    PythonDatasetBuilder,
)
from .models import PreferenceDataset, SupervisedDataset

type DatasetSourceKind = Literal["fixture", "huggingface", "jsonl", "nemo", "parquet", "built"]

_SFT_FORMATS = frozenset({"auto", "messages", "prompt-completion", "alpaca", "sharegpt"})
_PREFERENCE_FORMATS = frozenset({"auto", "trl", "tulu", "nemo-ranked"})
_NEMO_SFT_FORMATS = frozenset({"auto", "messages"})
_NEMO_PREFERENCE_FORMATS = frozenset({"auto", "nemo-ranked"})
_PATH_SOURCE_KINDS = frozenset({"jsonl", "nemo", "parquet"})


@dataclass(frozen=True, slots=True)
class DatasetSelection:
    """A catalog selection describing how to resolve one dataset."""

    id: str
    revision: str
    kind: DatasetKind
    source: DatasetSource
    format: str
    split: str = "train"
    schema_version: str = "1"
    provenance: DatasetProvenance = DatasetProvenance()
    access: DatasetAccessPolicy = DatasetAccessPolicy()

    def __post_init__(self) -> None:
        validate_selection_id(self.id, "dataset selection id")
        validate_revision(self.revision, "dataset selection revision")
        if isinstance(self.source, BuiltDatasetSource):
            source: DatasetSource = self.source
            source_kind = "built"
        else:
            source = dict(self.source)
            source_kind = source.get("kind")
        if source_kind not in {"fixture", "huggingface", "jsonl", "nemo", "parquet", "built"}:
            raise ContractError("dataset source kind must be fixture, huggingface, jsonl, nemo, parquet, or built")
        _validate_source(cast(DatasetSourceKind, source_kind), source)
        if source_kind == "nemo":
            allowed_formats = _NEMO_SFT_FORMATS if self.kind == "supervised" else _NEMO_PREFERENCE_FORMATS
        else:
            allowed_formats = _SFT_FORMATS if self.kind == "supervised" else _PREFERENCE_FORMATS
        if self.format not in allowed_formats:
            allowed = ", ".join(sorted(allowed_formats))
            raise ContractError(f"{self.kind} dataset format must be one of: {allowed}")
        if isinstance(source, Mapping):
            object.__setattr__(self, "source", MappingProxyType(dict(source)))
        else:
            object.__setattr__(self, "source", source)
        if not self.split.strip() or not self.schema_version.strip():
            raise ContractError("dataset split and schema_version cannot be empty")

    @property
    def source_kind(self) -> DatasetSourceKind:
        if isinstance(self.source, BuiltDatasetSource):
            return "built"
        return cast(DatasetSourceKind, self.source["kind"])

    @property
    def dataset_revision(self) -> str:
        if isinstance(self.source, BuiltDatasetSource):
            return self.revision
        source_revision = self.source.get("revision")
        return source_revision if isinstance(source_revision, str) else self.revision


# Historical callers use DatasetLoadPlan. Keep a single model implementation.
DatasetLoadPlan = DatasetSelection


@dataclass(frozen=True, slots=True)
class MaterializedDatasetAsset:
    """One verified regular file owned by a dataset materialization."""

    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        path = PurePosixPath(self.path)
        if (
            not self.path
            or "\\" in self.path
            or path.is_absolute()
            or path.as_posix() != self.path
            or not path.is_relative_to(PurePosixPath("assets"))
            or len(path.parts) < 2
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ContractError("dataset asset path must be a normalized relative path below assets/")
        if re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ContractError("dataset asset digest must be SHA-256")
        if self.size_bytes < 0:
            raise ContractError("dataset asset size cannot be negative")

    def to_payload(self) -> dict[str, JsonValue]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}


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
    build_key: str = ""
    assets: tuple[MaterializedDatasetAsset, ...] = ()
    assets_digest: str | None = None


def decode_dataset_selection(
    ref: CatalogRef,
    data: Mapping[str, object],
    _known: Mapping[CatalogRef, Selection],
) -> DatasetLoadPlan:
    """Decode the public dataset catalog shape into a reusable load plan."""

    allowed = {
        "id",
        "revision",
        "kind",
        "split",
        "schema_version",
        "provenance",
        "access",
        "source",
        "format",
    }
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
    split = data.get("split", "train")
    schema_version = data.get("schema_version", "1")
    if not isinstance(split, str) or not isinstance(schema_version, str):
        raise ContractError(f"dataset catalog entry {ref.id!r} split and schema_version must be strings")
    provenance = _decode_provenance(data.get("provenance"), ref)
    access = _decode_access(data.get("access"), ref)
    decoded_source: DatasetSource = cast(Mapping[str, JsonValue], source)
    if source.get("kind") == "built" and isinstance(source.get("builder"), Mapping):
        builder = source["builder"]
        if builder.get("kind") == "python":
            decoded_source = _decode_typed_built_source(source, ref)
    return DatasetLoadPlan(
        id=selection_id,
        revision=revision,
        kind=cast(DatasetKind, kind),
        source=decoded_source,
        format=format_kind,
        split=split,
        schema_version=schema_version,
        provenance=provenance,
        access=access,
    )


DATA_CATALOG_DECODERS: Mapping[SelectionFamily, SelectionDecoder] = {"dataset": decode_dataset_selection}


def _decode_provenance(value: object, ref: CatalogRef) -> DatasetProvenance:
    if value is None:
        return DatasetProvenance()
    if not isinstance(value, Mapping):
        raise ContractError(f"dataset catalog entry {ref.id!r} provenance must be an object")
    allowed = {"upstream", "transformation", "references"}
    unexpected = sorted(set(value).difference(allowed))
    if unexpected:
        raise ContractError(f"dataset catalog entry {ref.id!r} provenance has unknown fields: {', '.join(unexpected)}")
    upstream = value.get("upstream", ())
    references = value.get("references", ())
    transformation = value.get("transformation")
    if not _string_sequence(upstream) or not _string_sequence(references):
        raise ContractError(f"dataset catalog entry {ref.id!r} provenance lists must contain strings")
    if transformation is not None and not isinstance(transformation, str):
        raise ContractError(f"dataset catalog entry {ref.id!r} provenance transformation must be a string")
    return DatasetProvenance(tuple(upstream), transformation, tuple(references))


def _decode_access(value: object, ref: CatalogRef) -> DatasetAccessPolicy:
    if value is None:
        return DatasetAccessPolicy()
    if not isinstance(value, Mapping):
        raise ContractError(f"dataset catalog entry {ref.id!r} access must be an object")
    allowed = {"licenses", "classification"}
    unexpected = sorted(set(value).difference(allowed))
    if unexpected:
        raise ContractError(f"dataset catalog entry {ref.id!r} access has unknown fields: {', '.join(unexpected)}")
    licenses = value.get("licenses", ())
    classification = value.get("classification", "public")
    if not _string_sequence(licenses) or not isinstance(classification, str):
        raise ContractError(f"dataset catalog entry {ref.id!r} access values are invalid")
    return DatasetAccessPolicy(tuple(licenses), classification)


def _string_sequence(value: object) -> bool:
    return isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value)


def _decode_typed_built_source(source: Mapping[str, object], ref: CatalogRef) -> BuiltDatasetSource:
    builder = source.get("builder")
    raw_inputs = source.get("inputs")
    if (
        not isinstance(builder, Mapping)
        or set(builder).difference({"kind", "target"})
        or builder.get("kind") != "python"
        or not isinstance(builder.get("target"), str)
        or not isinstance(raw_inputs, Mapping)
    ):
        raise ContractError(f"dataset catalog entry {ref.id!r} built source has an invalid Python builder")
    inputs: dict[str, object] = {}
    for name, raw in raw_inputs.items():
        if not isinstance(name, str) or not isinstance(raw, Mapping):
            raise ContractError(f"dataset catalog entry {ref.id!r} built source inputs must be named objects")
        kind = raw.get("kind")
        if kind == "local" or kind == "jsonl":
            path = raw.get("path")
            if not isinstance(path, str):
                raise ContractError(f"dataset catalog entry {ref.id!r} local input requires path")
            inputs[name] = LocalDatasetInput(path, str(raw.get("format", "jsonl")))
        elif kind == "package-resource":
            resource = raw.get("resource")
            if not isinstance(resource, str):
                raise ContractError(f"dataset catalog entry {ref.id!r} package resource input requires resource")
            inputs[name] = PackageResourceInput(resource)
        elif kind == "huggingface":
            repo = raw.get("repo")
            revision = raw.get("revision")
            split = raw.get("split")
            config = raw.get("config")
            if not all(isinstance(value, str) for value in (repo, revision, split)) or (
                config is not None and not isinstance(config, str)
            ):
                raise ContractError(f"dataset catalog entry {ref.id!r} Hugging Face input is invalid")
            inputs[name] = HuggingFaceDatasetInput(
                cast(str, repo),
                cast(str, revision),
                cast(str, split),
                cast(str | None, config),
            )
        else:
            raise ContractError(f"dataset catalog entry {ref.id!r} has unsupported built input kind {kind!r}")
    expected = source.get("expected_content_sha256")
    if expected is not None and not isinstance(expected, str):
        raise ContractError(f"dataset catalog entry {ref.id!r} expected content digest must be a string")
    return BuiltDatasetSource(
        builder=PythonDatasetBuilder(builder["target"]),
        inputs=cast(Mapping[str, Any], inputs),
        expected_content_sha256=expected,
    )


def materialize_dataset(
    plan: DatasetLoadPlan,
    *,
    state_dir: Path,
    project_root: Path,
    input_root: Path | None = None,
    code_snapshot_digest: str | None = None,
    dependency_lock_digest: str | None = None,
) -> DatasetMaterialization:
    """Resolve, normalize, validate, and cache a dataset load plan."""

    if isinstance(plan.source, BuiltDatasetSource):
        return _materialize_typed_dataset(
            plan,
            state_dir=state_dir,
            project_root=project_root,
            input_root=input_root,
            code_snapshot_digest=code_snapshot_digest,
            dependency_lock_digest=dependency_lock_digest,
        )

    resolved_input_root = input_root or project_root
    fingerprint = hashlib.sha256(
        _plan_json(
            plan,
            project_root=project_root,
            input_root=resolved_input_root,
        ).encode("utf-8")
    ).hexdigest()
    destination = state_dir / "datasets" / fingerprint
    data_path = destination / "data.jsonl"
    manifest_path = destination / "manifest.json"
    if data_path.is_file() and manifest_path.is_file():
        return _read_materialization(plan, data_path, manifest_path, created=False)

    rows = tuple(
        _source_rows(
            plan,
            project_root=project_root,
            input_root=resolved_input_root,
        )
    )
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
    assets = _materialize_assets(
        plan,
        _declared_assets(normalized),
        destination=destination,
        project_root=project_root,
        input_root=resolved_input_root,
    )
    manifest = {
        "schema_version": 1,
        "selection_id": plan.id,
        "selection_revision": plan.revision,
        "dataset_revision": plan.dataset_revision,
        "source_kind": plan.source_kind,
        "build_key": fingerprint,
        "content_sha256": content_sha256,
        "examples": len(normalized),
        "data": data_path.name,
    }
    if assets:
        manifest["assets"] = [asset.to_payload() for asset in assets]
        manifest["assets_digest"] = _assets_digest(assets)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _read_materialization(plan, data_path, manifest_path, created=True)


def project_dataset_input_paths(plan: DatasetLoadPlan) -> tuple[str, ...]:
    """Return project-relative data inputs declared by one dataset selection.

    Builder source code is deliberately excluded. This closure is used by job
    packing to keep declared dataset bytes out of the generic project source
    snapshot while still admitting the code that produces a built dataset.
    """

    source = plan.source
    if isinstance(source, BuiltDatasetSource):
        paths = [item.path for item in source.inputs.values() if isinstance(item, LocalDatasetInput)]
    elif plan.source_kind in _PATH_SOURCE_KINDS:
        paths = [cast(str, cast(Mapping[str, JsonValue], source)["path"])]
    elif plan.source_kind == "built":
        paths = list(cast(list[str], cast(Mapping[str, JsonValue], source)["inputs"]))
    else:
        paths = []
    return tuple(sorted({_normalized_project_input_path(path) for path in paths}))


def _materialize_typed_dataset(
    plan: DatasetLoadPlan,
    *,
    state_dir: Path,
    project_root: Path,
    input_root: Path | None,
    code_snapshot_digest: str | None,
    dependency_lock_digest: str | None,
) -> DatasetMaterialization:
    from .materialization import (
        MATERIALIZER_SCHEMA_VERSION,
        build_key,
        input_identities,
        lock_digest,
        run_typed_builder,
        source_code_digest,
    )

    source = cast(BuiltDatasetSource, plan.source)
    resolved_input_root = input_root or project_root
    code_digest = code_snapshot_digest or source.builder.code_digest or source_code_digest(project_root)
    dependency_digest = dependency_lock_digest or source.builder.dependency_lock_digest or lock_digest(project_root)
    fingerprint = build_key(
        plan,
        project_root=project_root,
        input_root=resolved_input_root,
        code_snapshot_digest=code_digest,
        dependency_lock_digest=dependency_digest,
    )
    destination = state_dir / "datasets" / fingerprint
    data_path = destination / "data.jsonl"
    manifest_path = destination / "manifest.json"
    if data_path.is_file() and manifest_path.is_file():
        return _read_materialization(plan, data_path, manifest_path, created=False)

    cache_parent = state_dir / "datasets"
    cache_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{fingerprint[:12]}-", dir=cache_parent))
    try:
        rows = run_typed_builder(
            source,
            project_root=project_root,
            input_root=resolved_input_root,
            workspace=staging / "builder",
        )
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
        if source.expected_content_sha256 is not None and content_sha256 != source.expected_content_sha256:
            raise ContractError(
                f"dataset {plan.id!r} content digest mismatch: expected "
                f"{source.expected_content_sha256}, got {content_sha256}"
            )
        staging_data = staging / data_path.name
        staging_manifest = staging / manifest_path.name
        staging_data.write_text(serialized, encoding="utf-8")
        assets = _materialize_assets(
            plan,
            _declared_assets(normalized),
            destination=staging,
            project_root=project_root,
            input_root=resolved_input_root,
        )
        manifest = {
            "schema_version": MATERIALIZER_SCHEMA_VERSION,
            "selection_id": plan.id,
            "selection_revision": plan.revision,
            "dataset_revision": plan.dataset_revision,
            "dataset_kind": plan.kind,
            "split": plan.split,
            "dataset_schema_version": plan.schema_version,
            "source_kind": "built",
            "source": source.identity(),
            "inputs": input_identities(source, project_root=resolved_input_root),
            "provenance": {
                "upstream": list(plan.provenance.upstream),
                "transformation": plan.provenance.transformation,
                "references": list(plan.provenance.references),
            },
            "access": {
                "licenses": list(plan.access.licenses),
                "classification": plan.access.classification,
            },
            "build_key": fingerprint,
            "builder_target": source.builder.target,
            "code_snapshot_digest": code_digest,
            "dependency_lock_digest": dependency_digest,
            "content_sha256": content_sha256,
            "examples": len(normalized),
            "size_bytes": len(serialized.encode("utf-8")),
            "data": data_path.name,
        }
        if assets:
            manifest["assets"] = [asset.to_payload() for asset in assets]
            manifest["assets_digest"] = _assets_digest(assets)
        staging_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # Validate before promotion; an incomplete staging directory is never a
        # cache hit.  ``_read_materialization`` also checks output digest.
        _read_materialization(plan, staging_data, staging_manifest, created=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            return _read_materialization(
                plan, destination / data_path.name, destination / manifest_path.name, created=False
            )
        staging.replace(destination)
        return _read_materialization(plan, data_path, manifest_path, created=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


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


def validate_materialized_dataset(path: Path) -> int:
    """Parse a staged normalized JSONL dataset and return its record count."""

    return len(_jsonl_rows(path.read_text(encoding="utf-8"), source=str(path)))


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


def _validate_source(kind: DatasetSourceKind, source: object) -> None:
    if isinstance(source, BuiltDatasetSource):
        if kind != "built":
            raise ContractError("typed dataset source must have kind built")
        return
    if not isinstance(source, Mapping):
        raise ContractError("dataset source must be an object")
    required: dict[DatasetSourceKind, frozenset[str]] = {
        "fixture": frozenset({"kind", "resource"}),
        "huggingface": frozenset({"kind", "repo", "revision", "split"}),
        "jsonl": frozenset({"kind", "path"}),
        "nemo": frozenset({"kind", "path"}),
        "parquet": frozenset({"kind", "path"}),
        "built": frozenset({"kind", "builder", "inputs"}),
    }
    optional: dict[DatasetSourceKind, frozenset[str]] = {
        "fixture": frozenset(),
        "huggingface": frozenset({"config"}),
        "jsonl": frozenset(),
        "nemo": frozenset(),
        "parquet": frozenset({"split"}),
        "built": frozenset(),
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
    if kind == "built":
        builder = source["builder"]
        inputs = source["inputs"]
        if not isinstance(builder, Mapping) or set(builder) != {"kind", "path", "callable"}:
            raise ContractError("built dataset source builder must contain kind, path, and callable")
        builder_path = builder.get("path")
        builder_callable = builder.get("callable")
        if (
            builder.get("kind") != "python-file"
            or not isinstance(builder_path, str)
            or not builder_path.strip()
            or not isinstance(builder_callable, str)
            or not builder_callable.strip()
        ):
            raise ContractError("built dataset source builder must be a python-file with path and callable")
        if (
            not isinstance(inputs, list)
            or not inputs
            or not all(isinstance(path, str) and path.strip() for path in inputs)
        ):
            raise ContractError("built dataset source inputs must be a non-empty string list")
        return
    for key in keys.difference({"kind"}):
        value = source[key]
        if not isinstance(value, str) or not value.strip():
            raise ContractError(f"{kind} dataset source {key} must be a non-empty string")


def _project_path(plan: DatasetLoadPlan, *, project_root: Path) -> Path:
    source = cast(Mapping[str, JsonValue], plan.source)
    configured = Path(cast(str, source["path"]))
    if configured.is_absolute():
        raise ContractError(f"{plan.source_kind} dataset path must be relative to the project root")
    _reject_symlinked_project_path(project_root, configured, f"{plan.source_kind} dataset source")
    path = (project_root / configured).resolve()
    if not path.is_relative_to(project_root.resolve()):
        raise ContractError(f"{plan.source_kind} dataset path escapes the project root: {configured}")
    return path


def _source_rows(
    plan: DatasetLoadPlan,
    *,
    project_root: Path,
    input_root: Path | None = None,
) -> Iterable[Mapping[str, Any]]:
    if isinstance(plan.source, BuiltDatasetSource):
        raise ContractError("typed built dataset sources must use the child-process materializer")
    source = cast(Mapping[str, JsonValue], plan.source)
    resolved_input_root = input_root or project_root
    if plan.source_kind == "built":
        warnings.warn(
            "dataset source builder kind 'python-file' is deprecated; use PythonDatasetBuilder(module:callable)",
            DeprecationWarning,
            stacklevel=2,
        )
        builder = cast(Mapping[str, str], source["builder"])
        path = _safe_project_file(project_root, builder["path"], "built dataset builder")
        namespace = runpy.run_path(str(path))
        factory = namespace.get(builder["callable"])
        if not callable(factory):
            raise ContractError(f"built dataset builder callable is unavailable: {builder['callable']}")
        rows = factory()
        if not isinstance(rows, Iterable):
            raise ContractError("built dataset builder must return an iterable of row objects")
        return cast(Iterable[Mapping[str, Any]], rows)
    if plan.source_kind == "fixture":
        resource = cast(str, source["resource"])
        package, separator, name = resource.partition(":")
        if not separator or not package or not name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ContractError("fixture dataset resource must use PACKAGE:RELATIVE_PATH syntax")
        text = resource_files(package).joinpath(name).read_text(encoding="utf-8")
        return _jsonl_rows(text, source=resource)
    if plan.source_kind in _PATH_SOURCE_KINDS:
        configured = cast(str, source["path"])
        path = _project_path(plan, project_root=resolved_input_root)
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
            split=cast(str, source.get("split", "train")),
        )

    return _load_huggingface_rows(
        cast(str, source["repo"]),
        cast(str | None, source.get("config")),
        revision=cast(str, source["revision"]),
        split=cast(str, source["split"]),
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


def _plan_json(
    plan: DatasetLoadPlan,
    *,
    project_root: Path | None = None,
    input_root: Path | None = None,
) -> str:
    if isinstance(plan.source, BuiltDatasetSource):
        raise ContractError("typed built dataset sources use the typed materialization plan")
    source: dict[str, JsonValue] = dict(cast(Mapping[str, JsonValue], plan.source))
    if plan.source_kind == "built" and project_root is not None:
        builder = cast(Mapping[str, str], source["builder"])
        inputs = cast(list[str], source["inputs"])
        resolved_input_root = input_root or project_root
        source["input_digests"] = {
            builder["path"]: hashlib.sha256(
                _safe_project_file(project_root, builder["path"], "built dataset builder").read_bytes()
            ).hexdigest(),
            **{
                path: hashlib.sha256(
                    _safe_project_file(resolved_input_root, path, "built dataset input").read_bytes()
                ).hexdigest()
                for path in sorted(set(inputs))
            },
        }
    return json.dumps(
        {
            "id": plan.id,
            "revision": plan.revision,
            "kind": plan.kind,
            "split": plan.split,
            "schema_version": plan.schema_version,
            "provenance": {
                "upstream": list(plan.provenance.upstream),
                "transformation": plan.provenance.transformation,
                "references": list(plan.provenance.references),
            },
            "access": {
                "licenses": list(plan.access.licenses),
                "classification": plan.access.classification,
            },
            "source": source,
            "format": plan.format,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _safe_project_file(project_root: Path, configured: str, label: str) -> Path:
    path = Path(configured)
    if path.is_absolute() or not configured or ".." in path.parts:
        raise ContractError(f"{label} path must be relative to the project root")
    _reject_symlinked_project_path(project_root, path, label)
    resolved = (project_root / path).resolve()
    if not resolved.is_relative_to(project_root.resolve()) or not resolved.is_file():
        raise ContractError(f"{label} file not found: {configured}")
    return resolved


def _reject_symlinked_project_path(root: Path, path: Path, label: str) -> None:
    current = root
    for part in path.parts:
        current /= part
        if current.is_symlink():
            raise ContractError(f"{label} cannot traverse symlinks: {path.as_posix()}")


def _normalized_project_input_path(configured: str) -> str:
    path = Path(configured)
    if (
        not configured
        or configured != configured.strip()
        or "\\" in configured
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != configured
    ):
        raise ContractError("dataset project input must be a normalized relative POSIX path")
    return configured


def _declared_assets(rows: Iterable[Mapping[str, Any]]) -> tuple[tuple[str, str], ...]:
    declared: dict[str, str] = {}
    for row in rows:
        media = row.get("media", ())
        if not isinstance(media, list | tuple):
            raise ContractError("normalized supervised media must be a sequence")
        for value in media:
            if not isinstance(value, Mapping):
                raise ContractError("normalized supervised media entries must be objects")
            path = value.get("path")
            digest = value.get("sha256")
            if not isinstance(path, str) or not isinstance(digest, str):
                raise ContractError("normalized supervised media entries require path and sha256")
            MaterializedDatasetAsset(path, digest, 0)
            previous = declared.setdefault(path, digest)
            if previous != digest:
                raise ContractError(f"dataset asset path {path!r} has conflicting digests")
    return tuple(sorted(declared.items()))


def _materialize_assets(
    plan: DatasetLoadPlan,
    declared: tuple[tuple[str, str], ...],
    *,
    destination: Path,
    project_root: Path,
    input_root: Path,
) -> tuple[MaterializedDatasetAsset, ...]:
    del project_root
    assets: list[MaterializedDatasetAsset] = []
    for relative, expected_digest in declared:
        contents = _asset_source_bytes(plan, relative, input_root=input_root)
        observed_digest = hashlib.sha256(contents).hexdigest()
        if observed_digest != expected_digest:
            raise ContractError(
                f"dataset asset {relative!r} digest mismatch: expected {expected_digest}, got {observed_digest}"
            )
        if not contents:
            raise ContractError(f"dataset asset {relative!r} cannot be empty")
        target = destination.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise ContractError(f"dataset asset destination already exists: {relative}")
        target.write_bytes(contents)
        assets.append(MaterializedDatasetAsset(relative, observed_digest, len(contents)))
    return tuple(assets)


def _asset_source_bytes(plan: DatasetLoadPlan, relative: str, *, input_root: Path) -> bytes:
    if plan.source_kind == "fixture":
        source = cast(Mapping[str, JsonValue], plan.source)
        package, _, _ = cast(str, source["resource"]).partition(":")
        resource = resource_files(package).joinpath(relative)
        if not resource.is_file():
            raise ContractError(f"fixture dataset asset not found: {relative}")
        return resource.read_bytes()
    source = _safe_project_file(input_root, relative, "dataset asset")
    return source.read_bytes()


def _assets_digest(assets: tuple[MaterializedDatasetAsset, ...]) -> str:
    payload = [asset.to_payload() for asset in assets]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _manifest_assets(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    content: bytes,
) -> tuple[tuple[MaterializedDatasetAsset, ...], str | None]:
    raw_assets = manifest.get("assets")
    raw_digest = manifest.get("assets_digest")
    if raw_assets is None and raw_digest is None:
        if (root / "assets").exists():
            raise ContractError(f"materialized text dataset contains an undeclared assets directory at {root}")
        if _declared_assets(_jsonl_rows(content.decode("utf-8"), source=str(root / "data.jsonl"))):
            raise ContractError(f"materialized visual dataset omits its asset manifest at {root}")
        return (), None
    if not isinstance(raw_assets, list) or not raw_assets or not isinstance(raw_digest, str):
        raise ContractError(f"materialized dataset has invalid asset metadata at {root}")
    assets: list[MaterializedDatasetAsset] = []
    for value in raw_assets:
        if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "size_bytes"}:
            raise ContractError(f"materialized dataset has invalid asset record at {root}")
        path = value.get("path")
        digest = value.get("sha256")
        size = value.get("size_bytes")
        if (
            not isinstance(path, str)
            or not isinstance(digest, str)
            or not isinstance(size, int)
            or isinstance(size, bool)
        ):
            raise ContractError(f"materialized dataset has invalid asset record at {root}")
        assets.append(MaterializedDatasetAsset(path, digest, size))
    ordered = tuple(assets)
    if tuple(asset.path for asset in ordered) != tuple(sorted(asset.path for asset in ordered)):
        raise ContractError(f"materialized dataset asset records are not deterministically ordered at {root}")
    if len({asset.path for asset in ordered}) != len(ordered):
        raise ContractError(f"materialized dataset contains duplicate asset records at {root}")
    if _assets_digest(ordered) != raw_digest:
        raise ContractError(f"materialized dataset asset bundle digest mismatch at {root}")
    declared = _declared_assets(_jsonl_rows(content.decode("utf-8"), source=str(root / "data.jsonl")))
    if declared != tuple((asset.path, asset.sha256) for asset in ordered):
        raise ContractError(f"materialized dataset asset records differ from data.jsonl at {root}")
    asset_root = root / "assets"
    if not asset_root.is_dir() or asset_root.is_symlink():
        raise ContractError(f"materialized dataset asset directory is missing at {root}")
    expected = {asset.path for asset in ordered}
    observed: set[str] = set()
    for path in asset_root.rglob("*"):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise ContractError(f"materialized dataset assets contain a symlink or special file at {root}")
        if path.is_file():
            observed.add(path.relative_to(root).as_posix())
    if observed != expected:
        raise ContractError(f"materialized dataset asset files differ from the manifest at {root}")
    for asset in ordered:
        path = root.joinpath(*PurePosixPath(asset.path).parts)
        contents = path.read_bytes()
        if len(contents) != asset.size_bytes or hashlib.sha256(contents).hexdigest() != asset.sha256:
            raise ContractError(f"materialized dataset asset differs from its lock: {asset.path}")
    return ordered, raw_digest


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
    build_key_value = manifest.get("build_key", "")
    if not isinstance(build_key_value, str):
        raise ContractError(f"materialized dataset cache has invalid build_key at {manifest_path}")
    if isinstance(plan.source, BuiltDatasetSource):
        expected_content = plan.source.expected_content_sha256
        if expected_content is not None and expected_content != actual_digest:
            raise ContractError(
                f"dataset {plan.id!r} content digest mismatch: expected {expected_content}, got {actual_digest}"
            )
        if not build_key_value:
            raise ContractError(f"typed dataset cache is missing build_key at {manifest_path}")
    examples = manifest.get("examples")
    if not isinstance(examples, int) or examples < 1:
        raise ContractError(f"materialized dataset cache has invalid example count at {manifest_path}")
    assets, assets_digest = _manifest_assets(manifest, root=manifest_path.parent, content=content)
    return DatasetMaterialization(
        selection_id=plan.id,
        selection_revision=plan.revision,
        source_kind=plan.source_kind,
        path=data_path,
        manifest_path=manifest_path,
        content_sha256=actual_digest,
        examples=examples,
        created=created,
        build_key=build_key_value,
        assets=assets,
        assets_digest=assets_digest,
    )


__all__ = [
    "DATA_CATALOG_DECODERS",
    "DatasetLoadPlan",
    "DatasetMaterialization",
    "MaterializedDatasetAsset",
    "decode_dataset_selection",
    "load_materialized_dataset",
    "materialize_dataset",
    "resolve_dataset_source",
]
