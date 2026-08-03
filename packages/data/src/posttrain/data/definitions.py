"""Typed dataset definitions and build inputs.

These values are intentionally inert.  Constructing a definition validates its
identity and source metadata, but never imports a builder or resolves data.
Materialization is performed explicitly by :mod:`posttrain.data.materialization`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

from posttrain.common import ContractError, JsonValue
from posttrain.common.selections import validate_revision

if TYPE_CHECKING:
    from .catalog import DatasetSelection as DatasetSelection  # noqa: F401

type DatasetKind = Literal["supervised", "preference"]


@dataclass(frozen=True, slots=True)
class DatasetProvenance:
    """Human-readable upstream and transformation provenance."""

    upstream: tuple[str, ...] = ()
    transformation: str | None = None
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name, values in (("upstream", self.upstream), ("references", self.references)):
            if any(not value or not value.strip() for value in values):
                raise ContractError(f"dataset provenance {field_name} values cannot be empty")
            if len(set(values)) != len(values):
                raise ContractError(f"dataset provenance {field_name} values must be unique")
        if self.transformation is not None and not self.transformation.strip():
            raise ContractError("dataset provenance transformation cannot be empty")


@dataclass(frozen=True, slots=True)
class DatasetAccessPolicy:
    """License and handling information carried into materialization manifests."""

    licenses: tuple[str, ...] = ()
    classification: str = "public"

    def __post_init__(self) -> None:
        if any(not value or not value.strip() for value in self.licenses):
            raise ContractError("dataset access licenses cannot be empty")
        if len(set(self.licenses)) != len(self.licenses):
            raise ContractError("dataset access licenses must be unique")
        if not self.classification.strip():
            raise ContractError("dataset access classification cannot be empty")


@runtime_checkable
class DatasetBuildInput(Protocol):
    """A declared input to a Python dataset builder."""

    @property
    def kind(self) -> str: ...

    def identity(self) -> Mapping[str, JsonValue]: ...


@dataclass(frozen=True, slots=True)
class HuggingFaceDatasetInput:
    """A pinned Hugging Face dataset split."""

    repo: str
    revision: str
    split: str
    config: str | None = None

    @property
    def kind(self) -> Literal["huggingface"]:
        return "huggingface"

    def __post_init__(self) -> None:
        for name, value in (("repo", self.repo), ("revision", self.revision), ("split", self.split)):
            if not value.strip():
                raise ContractError(f"Hugging Face input {name} cannot be empty")
        if self.config is not None and not self.config.strip():
            raise ContractError("Hugging Face input config cannot be empty")
        validate_revision(self.revision, "Hugging Face input revision")

    def identity(self) -> Mapping[str, JsonValue]:
        value: dict[str, JsonValue] = {
            "kind": self.kind,
            "repo": self.repo,
            "revision": self.revision,
            "split": self.split,
        }
        if self.config is not None:
            value["config"] = self.config
        return value


@dataclass(frozen=True, slots=True)
class PackageResourceInput:
    """A resource shipped by an installed Python package."""

    resource: str

    @property
    def kind(self) -> Literal["package-resource"]:
        return "package-resource"

    def __post_init__(self) -> None:
        package, separator, path = self.resource.partition(":")
        if not separator or not package.strip() or not path.strip() or Path(path).is_absolute():
            raise ContractError("package resource must use PACKAGE:RELATIVE_PATH syntax")
        if ".." in Path(path).parts:
            raise ContractError("package resource path cannot escape its package")

    def identity(self) -> Mapping[str, JsonValue]:
        return {"kind": self.kind, "resource": self.resource}


@dataclass(frozen=True, slots=True)
class LocalDatasetInput:
    """A project-relative local file, usually JSONL, supplied to a builder."""

    path: str
    format: str = "jsonl"

    @property
    def kind(self) -> Literal["local"]:
        return "local"

    def __post_init__(self) -> None:
        if not self.path.strip() or Path(self.path).is_absolute() or ".." in Path(self.path).parts:
            raise ContractError("local dataset input path must be a project-relative path")
        if not self.format.strip():
            raise ContractError("local dataset input format cannot be empty")

    def identity(self) -> Mapping[str, JsonValue]:
        return {"kind": self.kind, "path": self.path, "format": self.format}


@dataclass(frozen=True, slots=True)
class PythonDatasetBuilder:
    """An importable module-level ``module:callable`` builder reference."""

    target: str
    code_digest: str | None = None
    dependency_lock_digest: str | None = None

    def __post_init__(self) -> None:
        if not self.target.strip() or self.target.count(":") != 1:
            raise ContractError("Python dataset builder target must use module:callable syntax")
        module, callable_name = self.target.split(":", maxsplit=1)
        if (
            not module.strip()
            or not callable_name.strip()
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", module)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", callable_name)
            or "<locals>" in callable_name
        ):
            raise ContractError("Python dataset builder target must name a module-level callable")
        for name, value in (("code digest", self.code_digest), ("dependency lock digest", self.dependency_lock_digest)):
            if value is not None and (len(value) != 64 or not re.fullmatch(r"[0-9a-f]{64}", value)):
                raise ContractError(f"Python dataset builder {name} must be a SHA-256 digest")

    @property
    def module(self) -> str:
        return self.target.split(":", maxsplit=1)[0]

    @property
    def callable_name(self) -> str:
        return self.target.split(":", maxsplit=1)[1]

    def identity(self) -> Mapping[str, JsonValue]:
        value: dict[str, JsonValue] = {"kind": "python", "target": self.target}
        if self.code_digest is not None:
            value["code_digest"] = self.code_digest
        if self.dependency_lock_digest is not None:
            value["dependency_lock_digest"] = self.dependency_lock_digest
        return value


@dataclass(frozen=True, slots=True)
class BuiltDatasetSource:
    """A typed custom build recipe with named, declared inputs."""

    builder: PythonDatasetBuilder
    inputs: Mapping[str, DatasetBuildInput]
    expected_content_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ContractError("built dataset source inputs must be non-empty")
        if any(not name or not name.strip() for name in self.inputs):
            raise ContractError("built dataset source input names cannot be empty")
        if any(not isinstance(value, DatasetBuildInput) for value in self.inputs.values()):
            raise ContractError("built dataset source inputs must be typed dataset build inputs")
        if self.expected_content_sha256 is not None and (
            len(self.expected_content_sha256) != 64
            or not re.fullmatch(r"[0-9a-f]{64}", self.expected_content_sha256)
        ):
            raise ContractError("expected dataset content digest must be a SHA-256 digest")
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))

    @property
    def kind(self) -> Literal["built"]:
        return "built"

    def identity(self) -> Mapping[str, JsonValue]:
        return {
            "kind": self.kind,
            "builder": dict(self.builder.identity()),
            "inputs": {
                name: dict(value.identity()) for name, value in sorted(self.inputs.items())
            },
            **(
                {"expected_content_sha256": self.expected_content_sha256}
                if self.expected_content_sha256 is not None
                else {}
            ),
        }


@dataclass(frozen=True, slots=True)
class ResolvedDatasetBuildInput:
    """A digest-checked, local input made available to a builder child process."""

    name: str
    kind: str
    path: Path
    digest: str
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.kind.strip() or not self.path.is_absolute():
            raise ContractError("resolved dataset build input requires a named absolute path")
        if len(self.digest) != 64 or not re.fullmatch(r"[0-9a-f]{64}", self.digest):
            raise ContractError("resolved dataset build input digest must be a SHA-256 digest")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class DatasetBuildContext:
    """The narrow context supplied to a Python builder."""

    inputs: Mapping[str, ResolvedDatasetBuildInput]
    workspace: Path

    def __post_init__(self) -> None:
        if not self.workspace.is_absolute():
            raise ContractError("dataset builder workspace must be an absolute path")
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))

    def input(self, name: str) -> ResolvedDatasetBuildInput:
        try:
            return self.inputs[name]
        except KeyError as error:
            raise ContractError(f"dataset builder requested undeclared input {name!r}") from error

    def records(self, name: str) -> tuple[Mapping[str, Any], ...]:
        """Read a declared JSONL input as object rows."""

        item = self.input(name)
        if item.kind not in {"local", "jsonl", "huggingface", "package-resource"}:
            raise ContractError(f"dataset builder input {name!r} does not contain JSONL records")
        rows: list[Mapping[str, Any]] = []
        for index, line in enumerate(item.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            import json

            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ContractError(f"invalid JSONL in builder input {name!r} at line {index}: {error}") from error
            if not isinstance(value, Mapping):
                raise ContractError(f"builder input {name!r} row {index} must be an object")
            rows.append(cast(Mapping[str, Any], value))
        return tuple(rows)

    def resource(self, name: str) -> Path:
        """Return the local path for a declared package or file resource."""

        return self.input(name).path


# ``DatasetSource`` is deliberately broad for the compatibility decoder.  New
# Python-authored definitions use ``BuiltDatasetSource``; existing YAML plans
# continue to carry their immutable mapping representation.
type DatasetSource = BuiltDatasetSource | Mapping[str, JsonValue]


__all__ = [
    "BuiltDatasetSource",
    "DatasetAccessPolicy",
    "DatasetBuildContext",
    "DatasetBuildInput",
    "DatasetKind",
    "DatasetProvenance",
    "DatasetSource",
    "HuggingFaceDatasetInput",
    "LocalDatasetInput",
    "PackageResourceInput",
    "PythonDatasetBuilder",
    "ResolvedDatasetBuildInput",
]


def __getattr__(name: str) -> object:
    """Lazily expose the compatibility selection class without a cycle."""

    if name == "DatasetSelection":
        from .catalog import DatasetSelection

        return DatasetSelection
    raise AttributeError(name)
