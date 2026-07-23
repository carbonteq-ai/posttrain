"""Primitive selections shared across post-training capability packages."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

from .artifacts import JsonValue
from .errors import ContractError
from .models import ModelVariant

_SELECTION_ID = re.compile(r"^[a-z0-9][a-z0-9._/@:-]*$")
_REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+:-]*$")

type SelectionFamily = Literal[
    "model",
    "dataset",
    "environment",
    "inference",
    "training",
    "quantization",
    "evaluation",
    "workload",
    "target",
    "recipe",
]
type Purpose = Literal["screen", "eval", "rollout", "teacher-score", "smoke", "handoff"]
type JsonMapping = Mapping[str, JsonValue]


def validate_selection_id(value: str, field_name: str = "selection id") -> str:
    if not _SELECTION_ID.fullmatch(value):
        raise ContractError(f"{field_name} must be a lowercase stable identifier, got {value!r}")
    return value


def validate_revision(value: str, field_name: str = "revision") -> str:
    if not _REVISION.fullmatch(value):
        raise ContractError(f"{field_name} is invalid: {value!r}")
    return value


def immutable_json_mapping(value: JsonMapping) -> JsonMapping:
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class ExecutionTarget:
    """Hardware and placement constraints for an execution."""

    id: str
    revision: str
    device_class: str
    memory_gb: float | None = None
    placement: JsonMapping = field(default_factory=dict)
    host_constraints: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_selection_id(self.id, "execution target id")
        validate_revision(self.revision, "execution target revision")
        if not self.device_class.strip():
            raise ContractError("execution target device_class cannot be empty")
        if self.memory_gb is not None and self.memory_gb <= 0:
            raise ContractError("execution target memory_gb must be positive")
        object.__setattr__(self, "placement", immutable_json_mapping(self.placement))
        object.__setattr__(self, "host_constraints", immutable_json_mapping(self.host_constraints))


@dataclass(frozen=True, slots=True)
class Workload:
    """A serving workload used for repeatable operating measurements."""

    id: str
    revision: str
    requests: JsonMapping
    concurrency: tuple[int, ...] = (1,)
    warmup_repetitions: int = 0
    measured_repetitions: int = 1
    required_measures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_selection_id(self.id, "workload id")
        validate_revision(self.revision, "workload revision")
        if not self.concurrency or any(value < 1 for value in self.concurrency):
            raise ContractError("workload concurrency values must be positive")
        if self.warmup_repetitions < 0 or self.measured_repetitions < 1:
            raise ContractError("workload repetitions are invalid")
        if any(not value.strip() for value in self.required_measures):
            raise ContractError("workload required measure names cannot be empty")
        object.__setattr__(self, "requests", immutable_json_mapping(self.requests))


@dataclass(frozen=True, slots=True)
class InferenceBinding:
    """A versioned model-to-runtime binding for a declared purpose."""

    id: str
    revision: str
    model: ModelVariant
    backend: str
    renderer: str
    engine: JsonMapping
    sampling: JsonMapping
    target: ExecutionTarget
    purpose: tuple[Purpose, ...]

    def __post_init__(self) -> None:
        validate_selection_id(self.id, "inference binding id")
        validate_revision(self.revision, "inference binding revision")
        if not self.backend.strip() or "@" not in self.backend:
            raise ContractError("inference backend must include a product and version")
        if not self.renderer.strip():
            raise ContractError("inference renderer cannot be empty")
        if not self.purpose or len(self.purpose) != len(set(self.purpose)):
            raise ContractError("inference purpose must be non-empty and unique")
        object.__setattr__(self, "engine", immutable_json_mapping(self.engine))
        object.__setattr__(self, "sampling", immutable_json_mapping(self.sampling))


@runtime_checkable
class Selection(Protocol):
    """Structural catalog value owned by common or a capability package."""

    @property
    def id(self) -> str: ...


__all__ = [
    "ExecutionTarget",
    "InferenceBinding",
    "JsonMapping",
    "Purpose",
    "Selection",
    "SelectionFamily",
    "Workload",
]
