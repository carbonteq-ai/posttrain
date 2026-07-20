"""Typed, task-neutral supervised and preference inputs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from posttrain.common import JsonValue, TraceObservation

_ID = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")


def _validate_id(value: str) -> None:
    if not _ID.fullmatch(value):
        raise ValueError(f"training example id must be stable and lowercase, got {value!r}")


@dataclass(frozen=True, slots=True)
class SupervisedExample:
    id: str
    prompt: str
    response: str
    system_prompt: str | None = None

    def __post_init__(self) -> None:
        _validate_id(self.id)
        if not self.prompt.strip() or not self.response.strip():
            raise ValueError("supervised examples require non-empty prompt and response")

    def messages(self) -> list[dict[str, str]]:
        values: list[dict[str, str]] = []
        if self.system_prompt is not None:
            values.append({"role": "system", "content": self.system_prompt})
        values.extend(
            (
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": self.response},
            )
        )
        return values


@dataclass(frozen=True, slots=True)
class SupervisedDataset:
    id: str
    revision: str
    examples: tuple[SupervisedExample, ...]

    def __post_init__(self) -> None:
        _validate_id(self.id)
        ids = tuple(example.id for example in self.examples)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("supervised datasets require non-empty, unique example ids")
        if not self.revision.strip():
            raise ValueError("dataset revision cannot be empty")


@dataclass(frozen=True, slots=True)
class PreferenceExample:
    id: str
    prompt: str
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float
    system_prompt: str | None = None
    rejected_trace_id: str | None = None

    def __post_init__(self) -> None:
        _validate_id(self.id)
        if not self.prompt.strip() or not self.chosen.strip() or not self.rejected.strip():
            raise ValueError("preference examples require non-empty prompt, chosen, and rejected text")
        if self.chosen == self.rejected:
            raise ValueError("chosen and rejected responses must differ")
        if self.chosen_score <= self.rejected_score:
            raise ValueError("chosen score must be strictly greater than rejected score")

    def prompt_messages(self) -> list[dict[str, str]]:
        values: list[dict[str, str]] = []
        if self.system_prompt is not None:
            values.append({"role": "system", "content": self.system_prompt})
        values.append({"role": "user", "content": self.prompt})
        return values


@dataclass(frozen=True, slots=True)
class PreferenceDataset:
    id: str
    revision: str
    examples: tuple[PreferenceExample, ...]

    def __post_init__(self) -> None:
        _validate_id(self.id)
        ids = tuple(example.id for example in self.examples)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("preference datasets require non-empty, unique example ids")
        if not self.revision.strip():
            raise ValueError("dataset revision cannot be empty")


@dataclass(frozen=True, slots=True)
class RolloutExample:
    id: str
    prompt: str
    metadata: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        _validate_id(self.id)
        if not self.prompt.strip():
            raise ValueError("rollout examples require a non-empty prompt")


@dataclass(frozen=True, slots=True)
class RolloutDataset:
    id: str
    revision: str
    examples: tuple[RolloutExample, ...]

    def __post_init__(self) -> None:
        _validate_id(self.id)
        ids = tuple(example.id for example in self.examples)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("rollout datasets require non-empty, unique example ids")
        if not self.revision.strip():
            raise ValueError("dataset revision cannot be empty")


@dataclass(frozen=True, slots=True)
class CompletedRollout:
    """One policy completion passed from a training backend to an RL environment."""

    example_id: str
    completion: str
    token_ids: tuple[int, ...]
    step: int
    terminated: bool

    def __post_init__(self) -> None:
        _validate_id(self.example_id)
        if self.step < 0:
            raise ValueError("rollout step cannot be negative")

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    @property
    def is_truncated(self) -> bool:
        return not self.terminated


@dataclass(frozen=True, slots=True)
class RolloutScore:
    """Environment-owned reward and its full-fidelity rollout observation."""

    reward: float
    trace: TraceObservation


__all__ = [
    "CompletedRollout",
    "PreferenceDataset",
    "PreferenceExample",
    "RolloutDataset",
    "RolloutExample",
    "RolloutScore",
    "SupervisedDataset",
    "SupervisedExample",
]
