"""Typed, backend-neutral evaluation requests and reusable programs."""

from __future__ import annotations

import json
import math
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
from posttrain.common.selections import validate_selection_id
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
class EvaluationSignalRef:
    """One namespaced numeric signal emitted by a Verifiers environment."""

    namespace: Literal["reward", "metric"]
    name: str

    def __post_init__(self) -> None:
        _stable_id(self.name, "evaluation signal name")


@dataclass(frozen=True, slots=True)
class EvaluationNumericPredicate:
    """A bounded numeric predicate used to derive semantic success."""

    operator: Literal["eq", "gt", "gte", "lt", "lte", "between"]
    value: float
    upper: float | None = None
    tolerance: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError("evaluation success predicate value must be finite")
        if self.upper is not None and not math.isfinite(self.upper):
            raise ValueError("evaluation success predicate upper bound must be finite")
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("evaluation success predicate tolerance must be finite and non-negative")
        if self.operator == "between":
            if self.upper is None or self.upper < self.value:
                raise ValueError("between predicate requires upper >= value")
        elif self.upper is not None:
            raise ValueError("evaluation success predicate upper is valid only for between")


@dataclass(frozen=True, slots=True)
class EvaluationSuccessDefinition:
    """Run-owned pass/fail meaning for one environment in an evaluation plan."""

    id: str
    label: str
    source: EvaluationSignalRef
    predicate: EvaluationNumericPredicate
    missing: Literal["error", "exclude"] = "error"

    def __post_init__(self) -> None:
        _stable_id(self.id, "evaluation success id")
        if not self.label.strip():
            raise ValueError("evaluation success label cannot be empty")


@dataclass(frozen=True, slots=True)
class EvaluationBreakdownDefinition:
    """Run-owned compound grouping over environment-declared task facets."""

    id: str
    label: str
    dimensions: tuple[str, ...]
    presentation: Literal["matrix"] = "matrix"
    multi_value: Literal["reject", "cross"] = "reject"
    missing: Literal["exclude", "bucket"] = "exclude"

    def __post_init__(self) -> None:
        _stable_id(self.id, "evaluation breakdown id")
        if not self.label.strip():
            raise ValueError("evaluation breakdown label cannot be empty")
        if len(self.dimensions) != 2 or len(set(self.dimensions)) != 2:
            raise ValueError("evaluation matrix breakdowns require exactly two unique dimensions")
        for dimension in self.dimensions:
            _stable_id(dimension, "evaluation breakdown dimension")


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    """Reusable selection and interpretation policy for environment cells."""

    id: str
    kind: Literal["general", "domain"]
    environments: tuple[EnvironmentBinding, ...]
    revision: str = "1"
    inference_requirements: Mapping[str, JsonValue] = field(default_factory=dict)
    metrics_and_slices: tuple[str, ...] = ()
    success: Mapping[str, EvaluationSuccessDefinition] = field(default_factory=dict)
    breakdowns: Mapping[str, tuple[EvaluationBreakdownDefinition, ...]] = field(default_factory=dict)
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
        unknown_success = set(self.success) - set(ids)
        if unknown_success:
            raise ValueError(
                "evaluation success definitions reference unknown environments: " + ", ".join(sorted(unknown_success))
            )
        object.__setattr__(self, "success", MappingProxyType(dict(self.success)))
        unknown_breakdowns = set(self.breakdowns) - set(ids)
        if unknown_breakdowns:
            raise ValueError(
                "evaluation breakdowns reference unknown environments: " + ", ".join(sorted(unknown_breakdowns))
            )
        environment_by_id = {environment.id: environment for environment in self.environments}
        normalized_breakdowns: dict[str, tuple[EvaluationBreakdownDefinition, ...]] = {}
        for environment_id, definitions in self.breakdowns.items():
            if len({definition.id for definition in definitions}) != len(definitions):
                raise ValueError(f"evaluation breakdown ids must be unique for {environment_id!r}")
            available_dimensions = {facet.dimension for facet in environment_by_id[environment_id].observation.facets}
            for definition in definitions:
                missing_dimensions = set(definition.dimensions) - available_dimensions
                if missing_dimensions:
                    raise ValueError(
                        f"evaluation breakdown {definition.id!r} references undeclared dimensions: "
                        + ", ".join(sorted(missing_dimensions))
                    )
            normalized_breakdowns[environment_id] = tuple(definitions)
        object.__setattr__(self, "breakdowns", MappingProxyType(normalized_breakdowns))
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

    def success_for(self, environment_id: str) -> EvaluationSuccessDefinition:
        self.environment(environment_id)
        try:
            return self.success[environment_id]
        except KeyError as error:
            raise ValueError(
                f"evaluation plan {self.id!r} requires an explicit success definition "
                f"for environment {environment_id!r}"
            ) from error

    def breakdowns_for(self, environment_id: str) -> tuple[EvaluationBreakdownDefinition, ...]:
        self.environment(environment_id)
        return self.breakdowns.get(environment_id, ())


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
        if any(
            not isinstance(name, str) or not name.strip() or not isinstance(value, str)
            for name, value in headers.items()
        ):
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
    # ``None`` means inherit the request's legacy ``shuffle`` field.  Keeping
    # this override on the invocation budget lets standard job definitions
    # express the complete subset policy without constructing a custom request.
    shuffle: bool | None = None

    def __post_init__(self) -> None:
        values = (self.num_tasks, self.num_rollouts, self.max_concurrent)
        if any(value is not None and value < 1 for value in values):
            raise ValueError("evaluation budget overrides must be positive")
        if self.shuffle is not None and not isinstance(self.shuffle, bool):
            raise TypeError("evaluation budget shuffle override must be a boolean")

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
        self.plan.success_for(self.environment_id)
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
            missing_capabilities = sorted(
                set(environment.required_inference_capabilities).difference(self.inference.capabilities)
            )
            if missing_capabilities:
                raise ValueError(
                    "evaluation inference binding is missing environment capabilities: "
                    + ", ".join(missing_capabilities)
                )
            speculative = self.inference.engine.get("speculative_config", self.inference.engine.get("speculative"))
            if (
                isinstance(speculative, Mapping)
                and speculative.get("method") == "mtp"
                and not self.model.capabilities.mtp
            ):
                raise ValueError(f"evaluation MTP requires a model with a native MTP head: {self.model.id!r}")
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
        missing_capabilities = sorted(
            capability
            for capability in environment.required_inference_capabilities
            if self.model.capabilities.get(capability) is not True
        )
        if missing_capabilities:
            raise ValueError(
                "remote evaluation policy is missing environment capabilities: " + ", ".join(missing_capabilities)
            )
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

    @property
    def resolved_shuffle(self) -> bool:
        """Return the effective deterministic Verifiers subset policy."""

        return self.budget.shuffle if self.budget.shuffle is not None else self.shuffle


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
    "EvaluationNumericPredicate",
    "EvaluationSignalRef",
    "EvaluationSuccessDefinition",
    "SamplingPolicy",
]
