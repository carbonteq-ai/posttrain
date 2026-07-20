"""Immutable artifact identities shared across package boundaries."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from .errors import ContractError

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type ArtifactMetadata = Mapping[str, JsonValue]

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_NAME = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")


def _require_name(value: str, field_name: str) -> str:
    if not _NAME.fullmatch(value):
        raise ContractError(f"{field_name} must be a lowercase stable identifier, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class HubModelRef:
    """An immutable Hugging Face model revision."""

    repo_id: str
    revision: str

    def __post_init__(self) -> None:
        if self.repo_id.count("/") != 1 or any(not part for part in self.repo_id.split("/")):
            raise ContractError(f"repo_id must be an owner/repository pair, got {self.repo_id!r}")
        if not _COMMIT_SHA.fullmatch(self.revision):
            raise ContractError("Hub model revisions must be full 40-character commit SHAs")

    @property
    def uri(self) -> str:
        return f"hf://{self.repo_id}@{self.revision}"


@dataclass(frozen=True, slots=True)
class TrackioArtifactRef:
    """A Trackio artifact reference, optionally resolved from an alias."""

    project: str
    name: str
    version: str
    alias: str | None = None

    def __post_init__(self) -> None:
        _require_name(self.project, "project")
        _require_name(self.name, "name")
        if not self.version.strip():
            raise ContractError("Trackio artifact version must be resolved before execution")


@dataclass(frozen=True, slots=True)
class LocalArtifactRef:
    """Content-addressed local output that has not yet been promoted."""

    path: Path
    digest: str

    def __post_init__(self) -> None:
        if not self.path.is_absolute():
            raise ContractError("Local artifact paths must be absolute")
        if not _SHA256.fullmatch(self.digest):
            raise ContractError("Local artifact digest must be a SHA-256 value")


type ArtifactRef = HubModelRef | TrackioArtifactRef | LocalArtifactRef


@dataclass(frozen=True, slots=True)
class ProducedArtifact:
    """An operation output together with its promotion requirements."""

    name: str
    kind: str
    reference: ArtifactRef
    required: bool = True
    metadata: ArtifactMetadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_name(self.name, "artifact name")
        _require_name(self.kind, "artifact kind")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
