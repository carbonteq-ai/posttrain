"""Typed, backend-neutral evaluation requests and reusable programs."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from posttrain.common import ModelProfile

_ID = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")

type EnvironmentFactory = Callable[[], object]


def _stable_id(value: str, field: str) -> None:
    if not _ID.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase stable identifier, got {value!r}")


@dataclass(frozen=True, slots=True)
class EnvironmentSource:
    """Immutable source of an independently installable environment package."""

    package: str
    repository: str
    revision: str
    subdirectory: str | None = None

    def __post_init__(self) -> None:
        _stable_id(self.package, "environment package")
        parsed = urlparse(self.repository)
        if parsed.scheme not in {"https", "ssh"} or not parsed.netloc:
            raise ValueError("environment repository must be an absolute HTTPS or SSH URL")
        if not _COMMIT_SHA.fullmatch(self.revision):
            raise ValueError("environment source revision must be a full commit SHA")
        if self.subdirectory is not None and (not self.subdirectory or self.subdirectory.startswith("/")):
            raise ValueError("environment subdirectory must be a non-empty relative path")


@dataclass(frozen=True, slots=True)
class SamplingPolicy:
    """Provider-independent sampling controls used by an evaluation cell."""

    max_tokens: int
    temperature: float = 0.0
    top_p: float | None = None
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("sampling max_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("sampling temperature cannot be negative")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("sampling top_p must be in (0, 1]")
        if self.reasoning_effort is not None and not self.reasoning_effort.strip():
            raise ValueError("reasoning_effort cannot be empty")


@dataclass(frozen=True, slots=True)
class EnvironmentProgram:
    """One independently runnable Verifiers environment inside a program."""

    id: str
    category: str
    source: EnvironmentSource
    factory: EnvironmentFactory
    sampling: SamplingPolicy
    num_tasks: int
    num_rollouts: int = 1
    max_concurrent: int = 4

    def __post_init__(self) -> None:
        _stable_id(self.id, "environment id")
        _stable_id(self.category, "environment category")
        if not callable(self.factory):
            raise ValueError("environment factory must be callable")
        if self.num_tasks < 1 or self.num_rollouts < 1 or self.max_concurrent < 1:
            raise ValueError("evaluation task, rollout, and concurrency counts must be positive")


@dataclass(frozen=True, slots=True)
class EvaluationProgram:
    """Reusable selection of independently runnable environment cells."""

    id: str
    kind: Literal["general", "domain"]
    environments: tuple[EnvironmentProgram, ...]

    def __post_init__(self) -> None:
        _stable_id(self.id, "evaluation program id")
        ids = tuple(environment.id for environment in self.environments)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("evaluation programs require non-empty, unique environment ids")

    def environment(self, environment_id: str) -> EnvironmentProgram:
        for environment in self.environments:
            if environment.id == environment_id:
                return environment
        available = ", ".join(item.id for item in self.environments)
        raise ValueError(f"unknown environment {environment_id!r}; available: {available}")

    def select(self, *environment_ids: str) -> tuple[EnvironmentProgram, ...]:
        if not environment_ids:
            return self.environments
        requested = set(environment_ids)
        selected = tuple(item for item in self.environments if item.id in requested)
        missing = requested - {item.id for item in selected}
        if missing:
            raise ValueError(f"unknown environment ids: {', '.join(sorted(missing))}")
        return selected


@dataclass(frozen=True, slots=True)
class EvaluationTarget:
    """An OpenAI-compatible generation target, independent of its serving engine."""

    base_url: str
    served_model: str
    api_key_var: str = "LOCAL_INFERENCE_API_KEY"

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("evaluation target base_url must be an absolute HTTP URL")
        if not self.served_model.strip():
            raise ValueError("evaluation target requires a served model name")
        if not self.api_key_var.isidentifier() or not self.api_key_var.isupper():
            raise ValueError("api_key_var must be an uppercase environment-variable name")


@dataclass(frozen=True, slots=True)
class EvaluationBudget:
    """Optional invocation-sized subset without mutating a reusable program."""

    num_tasks: int | None = None
    num_rollouts: int | None = None
    max_concurrent: int | None = None

    def __post_init__(self) -> None:
        values = (self.num_tasks, self.num_rollouts, self.max_concurrent)
        if any(value is not None and value < 1 for value in values):
            raise ValueError("evaluation budget overrides must be positive")

    def resolve(self, environment: EnvironmentProgram) -> tuple[int, int, int]:
        return (
            self.num_tasks or environment.num_tasks,
            self.num_rollouts or environment.num_rollouts,
            self.max_concurrent or environment.max_concurrent,
        )


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    """One model evaluated on one environment cell from a reusable program."""

    model: ModelProfile
    target: EvaluationTarget
    program: EvaluationProgram
    environment_id: str
    context_window: int
    reasoning_mode: str | None = None
    shuffle: bool = False
    budget: EvaluationBudget = EvaluationBudget()

    def __post_init__(self) -> None:
        environment = self.program.environment(self.environment_id)
        if self.context_window < 1:
            raise ValueError("evaluation context window must be positive")
        if self.context_window > self.model.capabilities.native_context_window:
            raise ValueError("evaluation context exceeds the model's native context window")
        if environment.sampling.max_tokens >= self.context_window:
            raise ValueError("evaluation response budget must be smaller than the served context window")
        self.model.conversation.reasoning_mode(self.resolved_reasoning_mode)

    @property
    def environment(self) -> EnvironmentProgram:
        return self.program.environment(self.environment_id)

    @property
    def resolved_reasoning_mode(self) -> str:
        return self.reasoning_mode or self.model.default_reasoning_mode

    @property
    def resolved_budget(self) -> tuple[int, int, int]:
        return self.budget.resolve(self.environment)


__all__ = [
    "EnvironmentFactory",
    "EnvironmentProgram",
    "EnvironmentSource",
    "EvaluationBudget",
    "EvaluationProgram",
    "EvaluationRequest",
    "EvaluationTarget",
    "SamplingPolicy",
]
