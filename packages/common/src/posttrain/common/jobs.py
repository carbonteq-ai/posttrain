"""Stable job, action, invocation, and attempt identities."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from .errors import ContractError

_ID = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _stable_id(value: str, field_name: str) -> str:
    if not _ID.fullmatch(value):
        raise ContractError(f"{field_name} must be a lowercase stable identifier, got {value!r}")
    return value


def _uuid(value: str, field_name: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except ValueError as error:
        raise ContractError(f"{field_name} must be a UUID") from error
    return str(parsed)


@dataclass(frozen=True, slots=True)
class Job:
    id: str
    version: str
    name: str

    def __post_init__(self) -> None:
        _stable_id(self.id, "job id")
        if not _GIT_SHA.fullmatch(self.version):
            raise ContractError("job version must be the full Git commit SHA of its definition")
        if not self.name.strip():
            raise ContractError("job name cannot be empty")


@dataclass(frozen=True, slots=True)
class JobAction:
    job_id: str
    id: str
    kind: str

    def __post_init__(self) -> None:
        _stable_id(self.job_id, "job id")
        _stable_id(self.id, "action id")
        _stable_id(self.kind, "action kind")


@dataclass(frozen=True, slots=True)
class Invocation:
    id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _uuid(self.id, "invocation id"))

    @classmethod
    def new(cls) -> Invocation:
        return cls(str(uuid.uuid4()))


@dataclass(frozen=True, slots=True)
class RunAttempt:
    id: str
    number: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _uuid(self.id, "run attempt id"))
        if self.number < 1:
            raise ContractError("run attempt number must be positive")

    @classmethod
    def new(cls, number: int = 1) -> RunAttempt:
        return cls(str(uuid.uuid4()), number)
