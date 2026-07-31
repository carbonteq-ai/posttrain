"""Typed, backend-neutral evaluation requests and reusable programs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal
from urllib.parse import urlsplit

from posttrain.common import (
    ExecutionTarget,
    InferenceBinding,
    JsonValue,
    ModelVariant,
)
from posttrain.environment import (
    EnvironmentActivation,
    EnvironmentBinding,
    EnvironmentFactory,
    EnvironmentSource,
    PythonFactoryActivation,
    SamplingPolicy,
    VerifiersV1ConfigActivation,
)

_ID = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")


def _stable_id(value: str, field: str) -> None:
    if not _ID.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase stable identifier, got {value!r}")


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    """Reusable selection and interpretation policy for environment cells."""

    id: str
    kind: Literal["general", "domain"]
    environments: tuple[EnvironmentBinding, ...]
    revision: str = "1"
    inference_requirements: Mapping[str, JsonValue] = field(default_factory=dict)
    metrics_and_slices: tuple[str, ...] = ()
    aggregation: Mapping[str, JsonValue] = field(default_factory=dict)
    comparison: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _stable_id(self.id, "evaluation program id")
        ids = tuple(environment.id for environment in self.environments)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("evaluation plans require non-empty, unique environment ids")
        if not self.revision.strip():
            raise ValueError("evaluation plan revision cannot be empty")
        object.__setattr__(self, "inference_requirements", MappingProxyType(dict(self.inference_requirements)))
        object.__setattr__(self, "aggregation", MappingProxyType(dict(self.aggregation)))
        object.__setattr__(self, "comparison", MappingProxyType(dict(self.comparison)))

    def environment(self, environment_id: str) -> EnvironmentBinding:
        for environment in self.environments:
            if environment.id == environment_id:
                return environment
        available = ", ".join(item.id for item in self.environments)
        raise ValueError(f"unknown environment {environment_id!r}; available: {available}")

    def select(self, *environment_ids: str) -> tuple[EnvironmentBinding, ...]:
        if not environment_ids:
            return self.environments
        requested = set(environment_ids)
        selected = tuple(item for item in self.environments if item.id in requested)
        missing = requested - {item.id for item in selected}
        if missing:
            raise ValueError(f"unknown environment ids: {', '.join(sorted(missing))}")
        return selected


@dataclass(frozen=True, slots=True)
class EvaluationEndpoint:
    """An OpenAI-compatible generation target, independent of its serving engine."""

    base_url: str
    served_model: str
    api_key_var: str = "LOCAL_INFERENCE_API_KEY"

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
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

    def resolve(self, environment: EnvironmentBinding) -> tuple[int, int, int]:
        return (
            self.num_tasks or environment.num_tasks,
            self.num_rollouts or environment.num_rollouts,
            self.max_concurrent or environment.max_concurrent,
        )


@dataclass(frozen=True, slots=True)
class EvaluateRequest:
    """Canonical evaluation seats plus the host-provided live endpoint."""

    model: ModelVariant
    plan: EvaluationPlan
    inference: InferenceBinding
    target: ExecutionTarget
    endpoint: EvaluationEndpoint
    environment_id: str
    context_window: int
    reasoning_mode: str | None = None
    shuffle: bool = False
    budget: EvaluationBudget = EvaluationBudget()

    def __post_init__(self) -> None:
        environment = self.plan.environment(self.environment_id)
        if self.inference.model != self.model:
            raise ValueError("evaluation model conflicts with its inference binding")
        if self.inference.target != self.target:
            raise ValueError("evaluation target conflicts with its inference binding")
        if "eval" not in self.inference.purpose:
            raise ValueError("evaluation requires an inference binding with eval purpose")
        if self.context_window < 1:
            raise ValueError("evaluation context window must be positive")
        if self.context_window > self.model.capabilities.native_context_window:
            raise ValueError("evaluation context exceeds the model's native context window")
        if environment.sampling.max_tokens >= self.context_window:
            raise ValueError("evaluation response budget must be smaller than the served context window")
        self.model.conversation.reasoning_mode(self.resolved_reasoning_mode)

    @property
    def environment(self) -> EnvironmentBinding:
        return self.plan.environment(self.environment_id)

    @property
    def resolved_reasoning_mode(self) -> str:
        return self.reasoning_mode or self.model.default_reasoning_mode

    @property
    def resolved_budget(self) -> tuple[int, int, int]:
        return self.budget.resolve(self.environment)


__all__ = [
    "EnvironmentActivation",
    "EnvironmentFactory",
    "EnvironmentBinding",
    "EnvironmentSource",
    "PythonFactoryActivation",
    "VerifiersV1ConfigActivation",
    "EvaluateRequest",
    "EvaluationBudget",
    "EvaluationEndpoint",
    "EvaluationPlan",
    "SamplingPolicy",
]
