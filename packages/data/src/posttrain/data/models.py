"""Canonical, framework-neutral post-training dataset records."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

from posttrain.common import JsonValue

_ID = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})

type MessageRecord = Mapping[str, JsonValue]
type ToolRecord = Mapping[str, JsonValue]
type DatasetKind = Literal["supervised", "preference", "rollout"]


def _validate_id(value: str) -> None:
    if not _ID.fullmatch(value):
        raise ValueError(f"dataset and example ids must be stable and lowercase, got {value!r}")


def _freeze_mapping(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return MappingProxyType(dict(value))


def _freeze_records(records: tuple[Mapping[str, JsonValue], ...]) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(_freeze_mapping(record) for record in records)


def _freeze_messages(messages: tuple[MessageRecord, ...], *, field_name: str) -> tuple[MessageRecord, ...]:
    if not messages:
        raise ValueError(f"{field_name} messages cannot be empty")
    frozen: list[MessageRecord] = []
    for message in messages:
        role = message.get("role")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"{field_name} messages require a non-empty role")
        if message.get("content") is None and message.get("tool_calls") is None:
            raise ValueError(f"{field_name} messages require content or tool calls")
        frozen.append(_freeze_mapping(message))
    return tuple(frozen)


@dataclass(frozen=True, slots=True)
class DatasetDescriptor:
    """Identity available before a potentially expensive source materialization."""

    id: str
    revision: str
    kind: DatasetKind
    schema_version: int = 1
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    num_examples: int | None = None

    def __post_init__(self) -> None:
        _validate_id(self.id)
        if not self.revision.strip() or self.schema_version < 1:
            raise ValueError("dataset revision and positive schema version are required")
        if self.num_examples is not None and self.num_examples < 1:
            raise ValueError("declared dataset example count must be positive")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class SupervisedMedia:
    """One immutable media asset associated with a supervised example."""

    path: str
    sha256: str
    mime_type: str
    kind: Literal["image"] = "image"
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        path = PurePosixPath(self.path)
        if (
            not self.path
            or "\\" in self.path
            or path.is_absolute()
            or path.as_posix() != self.path
            or len(path.parts) < 2
            or path.parts[0] != "assets"
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("supervised media paths must be normalized relative POSIX paths below assets/")
        if self.kind != "image":
            raise ValueError("supervised media kind must be image")
        if self.mime_type not in _IMAGE_MIME_TYPES:
            raise ValueError("supervised image media must use image/jpeg, image/png, or image/webp")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("supervised media sha256 must be 64 lowercase hexadecimal characters")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def as_record(self) -> dict[str, JsonValue]:
        return {
            "kind": self.kind,
            "path": self.path,
            "sha256": self.sha256,
            "mime_type": self.mime_type,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SupervisedExample:
    """One conversation with explicit message-level training targets."""

    id: str
    messages: tuple[MessageRecord, ...]
    trainable_message_indices: tuple[int, ...]
    tools: tuple[ToolRecord, ...] = ()
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    media: tuple[SupervisedMedia, ...] = ()

    def __post_init__(self) -> None:
        _validate_id(self.id)
        messages = _freeze_messages(self.messages, field_name="supervised")
        indices = self.trainable_message_indices
        if not indices or tuple(sorted(set(indices))) != indices:
            raise ValueError("supervised targets must be non-empty, unique, ordered message indices")
        if indices[-1] >= len(messages) or indices[0] < 0:
            raise ValueError("supervised target index is outside the message sequence")
        media = tuple(self.media)
        media_paths = tuple(item.path for item in media)
        if len(media_paths) != len(set(media_paths)):
            raise ValueError("supervised media paths must be unique within an example")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "tools", _freeze_records(self.tools))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        object.__setattr__(self, "media", media)

    def message_records(self) -> list[dict[str, JsonValue]]:
        return [dict(message) for message in self.messages]

    def tool_records(self) -> list[dict[str, JsonValue]]:
        return [dict(tool) for tool in self.tools]

    def media_records(self) -> list[dict[str, JsonValue]]:
        return [item.as_record() for item in self.media]


@dataclass(frozen=True, slots=True)
class SupervisedDataset:
    """Immutable materialized SFT snapshot; also a zero-cost source."""

    id: str
    revision: str
    examples: tuple[SupervisedExample, ...]
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        ids = tuple(example.id for example in self.examples)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("supervised datasets require non-empty, unique example ids")
        descriptor = self.descriptor
        object.__setattr__(self, "metadata", descriptor.metadata)

    @property
    def descriptor(self) -> DatasetDescriptor:
        return DatasetDescriptor(
            self.id,
            self.revision,
            "supervised",
            self.schema_version,
            self.metadata,
            len(self.examples),
        )

    def load(self) -> SupervisedDataset:
        return self


@dataclass(frozen=True, slots=True)
class PreferenceExample:
    """One prompt with preferred and rejected canonical continuations."""

    id: str
    prompt: tuple[MessageRecord, ...]
    chosen: tuple[MessageRecord, ...]
    rejected: tuple[MessageRecord, ...]
    chosen_score: float | None = None
    rejected_score: float | None = None
    chosen_trace_id: str | None = None
    rejected_trace_id: str | None = None
    tools: tuple[ToolRecord, ...] = ()
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_id(self.id)
        prompt = _freeze_messages(self.prompt, field_name="preference prompt")
        chosen = _freeze_messages(self.chosen, field_name="chosen continuation")
        rejected = _freeze_messages(self.rejected, field_name="rejected continuation")
        if chosen == rejected:
            raise ValueError("chosen and rejected continuations must differ")
        if (self.chosen_score is None) != (self.rejected_score is None):
            raise ValueError("preference scores must either both be present or both be absent")
        if self.chosen_score is not None and self.rejected_score is not None:
            if self.chosen_score <= self.rejected_score:
                raise ValueError("chosen score must be strictly greater than rejected score")
        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "chosen", chosen)
        object.__setattr__(self, "rejected", rejected)
        object.__setattr__(self, "tools", _freeze_records(self.tools))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def prompt_records(self) -> list[dict[str, JsonValue]]:
        return [dict(message) for message in self.prompt]

    def chosen_records(self) -> list[dict[str, JsonValue]]:
        return [dict(message) for message in self.chosen]

    def rejected_records(self) -> list[dict[str, JsonValue]]:
        return [dict(message) for message in self.rejected]

    def tool_records(self) -> list[dict[str, JsonValue]]:
        return [dict(tool) for tool in self.tools]


@dataclass(frozen=True, slots=True)
class PreferenceDataset:
    """Immutable materialized preference snapshot; also a zero-cost source."""

    id: str
    revision: str
    examples: tuple[PreferenceExample, ...]
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        ids = tuple(example.id for example in self.examples)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("preference datasets require non-empty, unique example ids")
        descriptor = self.descriptor
        object.__setattr__(self, "metadata", descriptor.metadata)

    @property
    def descriptor(self) -> DatasetDescriptor:
        return DatasetDescriptor(
            self.id,
            self.revision,
            "preference",
            self.schema_version,
            self.metadata,
            len(self.examples),
        )

    def load(self) -> PreferenceDataset:
        return self


@runtime_checkable
class SupervisedDataSource(Protocol):
    @property
    def descriptor(self) -> DatasetDescriptor: ...

    def load(self) -> SupervisedDataset: ...


@runtime_checkable
class PreferenceDataSource(Protocol):
    @property
    def descriptor(self) -> DatasetDescriptor: ...

    def load(self) -> PreferenceDataset: ...


@dataclass(frozen=True, slots=True)
class RolloutExample:
    id: str
    prompt: str
    metadata: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        _validate_id(self.id)
        if not self.prompt.strip():
            raise ValueError("rollout examples require a non-empty prompt")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class RolloutDataset:
    id: str
    revision: str
    examples: tuple[RolloutExample, ...]

    def __post_init__(self) -> None:
        DatasetDescriptor(self.id, self.revision, "rollout")
        ids = tuple(example.id for example in self.examples)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("rollout datasets require non-empty, unique example ids")


__all__ = [
    "DatasetDescriptor",
    "MessageRecord",
    "PreferenceDataSource",
    "PreferenceDataset",
    "PreferenceExample",
    "RolloutDataset",
    "RolloutExample",
    "SupervisedDataSource",
    "SupervisedDataset",
    "SupervisedExample",
    "ToolRecord",
]
