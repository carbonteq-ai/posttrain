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
from posttrain.common.selections import validate_selection_id

_ID = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_PYTHON_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_REMOTE_PROTOCOL = "openai-chat@1"
_SECRET_HEADERS = frozenset({"authorization", "cookie", "proxy-authorization", "x-api-key"})
_OWNED_REQUEST_FIELDS = frozenset(
    {
        "model",
        "messages",
        "tools",
        "stream",
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "n",
        "seed",
    }
)

type EnvironmentFactory = Callable[[], object]


def _stable_id(value: str, field: str) -> None:
    if not _ID.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase stable identifier, got {value!r}")


def _selection_id(value: str, field: str) -> None:
    try:
        validate_selection_id(value, field)
    except Exception as error:
        raise ValueError(str(error)) from error


def _secret_free_http_url(value: str, field: str) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{field} must be a secret-free absolute HTTP URL")


def _json_mapping(value: Mapping[str, JsonValue], field: str) -> Mapping[str, JsonValue]:
    copied = dict(value)
    try:
        json.dumps(copied, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must contain only JSON values") from error
    return MappingProxyType(copied)


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
class RemotePolicy:
    """An evaluation-only remote model selector with no local weight artifact."""

    id: str
    revision: str
    model: str
    context_window: int
    capabilities: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _selection_id(self.id, "remote policy id")
        if not self.revision.strip():
            raise ValueError("remote policy revision cannot be empty")
        if not self.model.strip():
            raise ValueError("remote policy model cannot be empty")
        if self.context_window < 1:
            raise ValueError("remote policy context_window must be positive")
        object.__setattr__(self, "capabilities", _json_mapping(self.capabilities, "remote policy capabilities"))


@dataclass(frozen=True, slots=True)
class ExternalInferenceService:
    """One secret-free OpenAI-chat service configuration for remote evaluation."""

    id: str
    revision: str
    base_url: str
    api_key_var: str
    headers: Mapping[str, str] = field(default_factory=dict)
    request_defaults: Mapping[str, JsonValue] = field(default_factory=dict)
    protocol: Literal["openai-chat@1"] = _REMOTE_PROTOCOL

    def __post_init__(self) -> None:
        _selection_id(self.id, "external inference service id")
        if not self.revision.strip():
            raise ValueError("external inference service revision cannot be empty")
        _secret_free_http_url(self.base_url, "external inference service base_url")
        if self.protocol != _REMOTE_PROTOCOL:
            raise ValueError(f"unsupported external inference protocol: {self.protocol!r}")
        if not self.api_key_var.isidentifier() or not self.api_key_var.isupper():
            raise ValueError("api_key_var must be an uppercase environment-variable name")
        headers = dict(self.headers)
        if any(not isinstance(name, str) or not name.strip() or not isinstance(value, str) for name, value in headers.items()):
            raise ValueError("external inference service headers must be non-empty string pairs")
        blocked = sorted(name for name in headers if name.casefold() in _SECRET_HEADERS)
        if blocked:
            raise ValueError(f"external inference service headers must not carry credentials: {', '.join(blocked)}")
        defaults = _json_mapping(self.request_defaults, "external inference service request_defaults")
        collisions = sorted(set(defaults).intersection(_OWNED_REQUEST_FIELDS))
        if collisions:
            raise ValueError(
                "external inference service request_defaults cannot override evaluation-owned fields: "
                + ", ".join(collisions)
            )
        object.__setattr__(self, "headers", MappingProxyType(headers))
        object.__setattr__(self, "request_defaults", defaults)

    @property
    def origin(self) -> str:
        parsed = urlsplit(self.base_url)
        return f"{parsed.scheme}://{parsed.netloc}"


@dataclass(frozen=True, slots=True)
class RemoteEvaluationBinding:
    """A versioned remote policy-service pair qualified only for evaluation."""

    id: str
    revision: str
    policy: RemotePolicy
    service: ExternalInferenceService
    purpose: tuple[Literal["screen", "eval"], ...]

    def __post_init__(self) -> None:
        _selection_id(self.id, "remote evaluation binding id")
        if not self.revision.strip():
            raise ValueError("remote evaluation binding revision cannot be empty")
        if not self.purpose or len(self.purpose) != len(set(self.purpose)):
            raise ValueError("remote evaluation binding purpose must be non-empty and unique")


@dataclass(frozen=True, slots=True)
class EvaluationEndpoint:
    """A live local evaluation target supplied by a host-owned serving lifecycle."""

    base_url: str
    served_model: str
    api_key_var: str = "LOCAL_INFERENCE_API_KEY"

    def __post_init__(self) -> None:
        _secret_free_http_url(self.base_url, "evaluation target base_url")
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
    """One local or remote evaluation subject and its compatible generation binding."""

    model: ModelVariant | RemotePolicy
    plan: EvaluationPlan
    inference: InferenceBinding | RemoteEvaluationBinding
    target: ExecutionTarget
    endpoint: EvaluationEndpoint | None
    environment_id: str
    context_window: int
    reasoning_mode: str | None = None
    shuffle: bool = False
    budget: EvaluationBudget = EvaluationBudget()

    def __post_init__(self) -> None:
        environment = self.plan.environment(self.environment_id)
        if self.context_window < 1:
            raise ValueError("evaluation context window must be positive")
        if environment.sampling.max_tokens >= self.context_window:
            raise ValueError("evaluation response budget must be smaller than the served context window")
        if isinstance(self.model, ModelVariant):
            if not isinstance(self.inference, InferenceBinding):
                raise ValueError("local evaluation requires an InferenceBinding")
            if self.endpoint is None:
                raise ValueError("local evaluation requires a host-provided EvaluationEndpoint")
            if self.inference.model != self.model:
                raise ValueError("evaluation model conflicts with its inference binding")
            if self.inference.target != self.target:
                raise ValueError("evaluation target conflicts with its inference binding")
            if "eval" not in self.inference.purpose:
                raise ValueError("evaluation requires an inference binding with eval purpose")
            if self.context_window > self.model.capabilities.native_context_window:
                raise ValueError("evaluation context exceeds the model's native context window")
            self.model.conversation.reasoning_mode(self.resolved_reasoning_mode)
            return
        if not isinstance(self.inference, RemoteEvaluationBinding):
            raise ValueError("remote evaluation requires a RemoteEvaluationBinding")
        if self.endpoint is not None:
            raise ValueError("remote evaluation resolves its endpoint from the remote binding")
        if self.inference.policy != self.model:
            raise ValueError("remote evaluation policy conflicts with its remote binding")
        if "eval" not in self.inference.purpose:
            raise ValueError("remote evaluation binding requires eval purpose")
        if self.context_window > self.model.context_window:
            raise ValueError("evaluation context exceeds the remote policy context window")
        if self.reasoning_mode is not None:
            raise ValueError("remote evaluation reasoning belongs in external service request_defaults")

    @property
    def environment(self) -> EnvironmentBinding:
        return self.plan.environment(self.environment_id)

    @property
    def resolved_reasoning_mode(self) -> str:
        if isinstance(self.model, RemotePolicy):
            return "provider-default"
        return self.reasoning_mode or self.model.default_reasoning_mode

    @property
    def resolved_endpoint(self) -> EvaluationEndpoint:
        if self.endpoint is not None:
            return self.endpoint
        assert isinstance(self.inference, RemoteEvaluationBinding)
        return EvaluationEndpoint(
            base_url=self.inference.service.base_url,
            served_model=self.inference.policy.model,
            api_key_var=self.inference.service.api_key_var,
        )

    @property
    def remote_service(self) -> ExternalInferenceService | None:
        return self.inference.service if isinstance(self.inference, RemoteEvaluationBinding) else None

    @property
    def resolved_budget(self) -> tuple[int, int, int]:
        return self.budget.resolve(self.environment)


__all__ = [
    "EnvironmentActivation",
    "EnvironmentFactory",
    "EnvironmentBinding",
    "EnvironmentSource",
    "ExternalInferenceService",
    "PythonFactoryActivation",
    "RemoteEvaluationBinding",
    "RemotePolicy",
    "VerifiersV1ConfigActivation",
    "EvaluateRequest",
    "EvaluationBudget",
    "EvaluationEndpoint",
    "EvaluationPlan",
    "SamplingPolicy",
]
