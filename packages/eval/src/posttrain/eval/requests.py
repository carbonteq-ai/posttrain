"""Typed, backend-neutral evaluation requests and reusable programs."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Literal
from urllib.parse import unquote, urlsplit

from posttrain.common import (
    ExecutionTarget,
    InferenceBinding,
    JsonValue,
    ModelVariant,
)

_ID = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_PYTHON_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")

type EnvironmentFactory = Callable[[], object]


def _stable_id(value: str, field: str) -> None:
    if not _ID.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase stable identifier, got {value!r}")


@dataclass(frozen=True, slots=True)
class PythonFactoryActivation:
    """Importable environment factory resolved only inside the execution runtime."""

    reference: str

    def __post_init__(self) -> None:
        module, separator, attribute = self.reference.partition(":")
        if separator != ":" or not _PYTHON_PATH.fullmatch(module) or not _PYTHON_PATH.fullmatch(attribute):
            raise ValueError("environment factory reference must use module:callable syntax")

    @property
    def module(self) -> str:
        return self.reference.partition(":")[0]

    @property
    def attribute(self) -> str:
        return self.reference.partition(":")[2]

    @property
    def kind(self) -> Literal["python-factory"]:
        return "python-factory"

    @property
    def digest(self) -> str:
        return _activation_digest(self.to_payload())

    def to_payload(self) -> dict[str, JsonValue]:
        return {"kind": self.kind, "reference": self.reference}

    def activate(self) -> object:
        return _resolve_environment_factory(self.reference)()

    @classmethod
    def from_callable(cls, factory: EnvironmentFactory) -> PythonFactoryActivation:
        """Convert an importable module-level callable for compatibility callers."""

        module = getattr(factory, "__module__", "")
        attribute = getattr(factory, "__qualname__", "")
        if not module or not attribute or "<locals>" in attribute:
            raise ValueError("environment factories must be importable module-level callables")
        activation = cls(f"{module}:{attribute}")
        if _resolve_environment_factory(activation.reference) is not factory:
            raise ValueError("environment factory callable does not resolve to its declared import path")
        return activation


@dataclass(frozen=True, slots=True)
class VerifiersV1ConfigActivation:
    """Declarative native Verifiers configuration activated at job startup."""

    config: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        config = dict(self.config)
        try:
            json.dumps(config, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("Verifiers activation config must contain only JSON values") from error
        object.__setattr__(self, "config", MappingProxyType(config))

    @property
    def kind(self) -> Literal["verifiers-config"]:
        return "verifiers-config"

    @property
    def digest(self) -> str:
        return _activation_digest(self.to_payload())

    def to_payload(self) -> dict[str, JsonValue]:
        return {"kind": self.kind, "config": dict(self.config)}

    def activate(self) -> object:
        try:
            from verifiers.v1.env import EnvConfig  # pyright: ignore[reportMissingImports]
        except ImportError as error:
            raise RuntimeError("install the Verifiers integration dependencies") from error
        return EnvConfig.model_validate(dict(self.config))


type EnvironmentActivation = PythonFactoryActivation | VerifiersV1ConfigActivation


def _activation_digest(payload: Mapping[str, JsonValue]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@cache
def _resolve_environment_factory(value: str) -> EnvironmentFactory:
    module_name, _, attribute = value.partition(":")
    try:
        resolved: object = importlib.import_module(module_name)
    except ImportError as error:
        raise RuntimeError(f"environment factory module is not installed: {module_name}") from error
    for part in attribute.split("."):
        try:
            resolved = getattr(resolved, part)
        except AttributeError as error:
            raise RuntimeError(f"environment factory is not available: {value}") from error
    if not callable(resolved):
        raise TypeError(f"environment factory reference is not callable: {value}")
    return resolved


@dataclass(frozen=True, slots=True)
class EnvironmentSource:
    """Immutable source of an independently installable environment package."""

    package: str
    repository: str
    revision: str
    subdirectory: str | None = None

    def __post_init__(self) -> None:
        _stable_id(self.package, "environment package")
        parsed = urlsplit(self.repository)
        expected_netloc = parsed.hostname
        if parsed.port is not None and expected_netloc is not None:
            expected_netloc = f"{expected_netloc}:{parsed.port}"
        segments = parsed.path.split("/")[1:]
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or unquote(parsed.path) != parsed.path
            or parsed.netloc != expected_netloc
            or not segments
            or any(not part or part in {".", ".."} for part in segments)
            or parsed.path.endswith("/")
        ):
            raise ValueError("environment repository must be a secret-free canonical HTTPS URL")
        if not _COMMIT_SHA.fullmatch(self.revision):
            raise ValueError("environment source revision must be a full commit SHA")
        if self.subdirectory is not None:
            path = PurePosixPath(self.subdirectory)
            if (
                not self.subdirectory
                or "\\" in self.subdirectory
                or path.is_absolute()
                or path.as_posix() != self.subdirectory
                or any(part in {"", ".", "..", ".git"} for part in path.parts)
            ):
                raise ValueError("environment subdirectory must be a normalized relative path")


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
class EnvironmentBinding:
    """One versioned, independently runnable environment inside a plan."""

    id: str
    category: str
    source: EnvironmentSource
    activation: EnvironmentActivation
    sampling: SamplingPolicy
    num_tasks: int
    num_rollouts: int = 1
    max_concurrent: int = 4
    parameters: Mapping[str, JsonValue] = field(default_factory=dict)
    reward_components: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _stable_id(self.id, "environment id")
        _stable_id(self.category, "environment category")
        if not isinstance(
            self.activation,
            (PythonFactoryActivation, VerifiersV1ConfigActivation),
        ):
            raise TypeError("environment activation must be a supported serializable activation")
        if self.num_tasks < 1 or self.num_rollouts < 1 or self.max_concurrent < 1:
            raise ValueError("evaluation task, rollout, and concurrency counts must be positive")
        if any(not value.strip() for value in self.reward_components):
            raise ValueError("environment reward component names cannot be empty")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    @property
    def revision(self) -> str:
        return self.source.revision

    def activate(self) -> object:
        """Construct the native environment config only when execution begins."""

        return self.activation.activate()


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
