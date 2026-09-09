"""Framework-neutral identity for API-only models and inference services."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable
from urllib.parse import urlsplit

from .artifacts import JsonValue
from .selections import validate_selection_id

_EXTERNAL_PROTOCOL = "openai-chat@1"
_SECRET_HEADERS = frozenset({"authorization", "cookie", "proxy-authorization", "x-api-key"})
_CALL_OWNED_FIELDS = frozenset(
    {
        "model",
        "messages",
        "input",
        "instructions",
        "tools",
        "tool_choice",
        "response_format",
        "text",
        "stream",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "reasoning",
        "reasoning_effort",
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
        "n",
        "seed",
    }
)


def _json_mapping(value: Mapping[str, JsonValue], field_name: str) -> Mapping[str, JsonValue]:
    copied = dict(value)
    try:
        json.dumps(copied, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must contain only JSON values") from error
    return MappingProxyType(copied)


def validate_secret_free_http_url(value: str, field_name: str) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{field_name} must be a secret-free absolute HTTP URL")


@dataclass(frozen=True, slots=True)
class HostedModel:
    """A versioned API model selector with no framework-owned weight artifact."""

    id: str
    revision: str
    model: str
    context_window: int
    capabilities: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_selection_id(self.id, "hosted model id")
        if not self.revision.strip():
            raise ValueError("hosted model revision cannot be empty")
        if not self.model.strip():
            raise ValueError("hosted model API identifier cannot be empty")
        if self.context_window < 1:
            raise ValueError("hosted model context_window must be positive")
        object.__setattr__(self, "capabilities", _json_mapping(self.capabilities, "hosted model capabilities"))

    def trace_identity(self) -> dict[str, JsonValue]:
        return {
            "selection_type": "hosted-model",
            "id": self.id,
            "revision": self.revision,
            "api_model": self.model,
            "context_window": self.context_window,
            "capabilities": dict(self.capabilities),
        }


@dataclass(frozen=True, slots=True)
class ExternalInferenceService:
    """A secret-free OpenAI-compatible external service declaration."""

    id: str
    revision: str
    base_url: str
    api_key_var: str
    headers: Mapping[str, str] = field(default_factory=dict)
    request_defaults: Mapping[str, JsonValue] = field(default_factory=dict)
    provider_policy: Mapping[str, JsonValue] = field(default_factory=dict)
    protocol: Literal["openai-chat@1"] = _EXTERNAL_PROTOCOL

    def __post_init__(self) -> None:
        validate_selection_id(self.id, "external inference service id")
        if not self.revision.strip():
            raise ValueError("external inference service revision cannot be empty")
        validate_secret_free_http_url(self.base_url, "external inference service base_url")
        if self.protocol != _EXTERNAL_PROTOCOL:
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
        collisions = sorted(set(defaults).intersection(_CALL_OWNED_FIELDS))
        if collisions:
            raise ValueError(
                "external inference service request_defaults cannot override evaluation-owned fields "
                "or other call-owned fields: " + ", ".join(collisions)
            )
        object.__setattr__(self, "headers", MappingProxyType(headers))
        object.__setattr__(self, "request_defaults", defaults)
        object.__setattr__(
            self,
            "provider_policy",
            _json_mapping(self.provider_policy, "external inference service provider_policy"),
        )

    @property
    def origin(self) -> str:
        parsed = urlsplit(self.base_url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def trace_identity(self) -> dict[str, JsonValue]:
        return {
            "selection_type": "external-inference-service",
            "id": self.id,
            "revision": self.revision,
            "origin": self.origin,
            "protocol": self.protocol,
            "headers": dict(self.headers),
            "request_defaults": dict(self.request_defaults),
            "provider_policy": dict(self.provider_policy),
            "api_key_var": self.api_key_var,
        }


@dataclass(frozen=True, slots=True)
class HostedInferenceBinding:
    """An API-only hosted model bound to an external service for judging."""

    id: str
    revision: str
    model: HostedModel
    service: ExternalInferenceService
    provider: str
    sampling: Mapping[str, JsonValue]
    purpose: tuple[Literal["judge"], ...] = ("judge",)
    max_cost_usd_micros: int = 4_990_000

    def __post_init__(self) -> None:
        validate_selection_id(self.id, "hosted inference binding id")
        if not self.revision.strip():
            raise ValueError("hosted inference binding revision cannot be empty")
        if not self.provider.strip():
            raise ValueError("hosted inference binding provider cannot be empty")
        if self.provider != self.provider.strip() or any(character.isspace() for character in self.provider):
            raise ValueError("hosted inference binding provider must be an exact provider slug")
        if self.purpose != ("judge",):
            raise ValueError("hosted inference binding is supported only for judge inference")
        if (
            isinstance(self.max_cost_usd_micros, bool)
            or not isinstance(self.max_cost_usd_micros, int)
            or self.max_cost_usd_micros < 1
        ):
            raise ValueError("hosted inference max_cost_usd_micros must be a positive integer")
        sampling = _json_mapping(self.sampling, "hosted inference sampling")
        max_tokens = sampling.get("max_tokens")
        if max_tokens is not None and (
            isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1
        ):
            raise ValueError("hosted inference max_tokens must be a positive integer when declared")
        if isinstance(max_tokens, int) and max_tokens > self.model.context_window:
            raise ValueError("hosted inference max_tokens cannot exceed the hosted model context window")
        object.__setattr__(self, "sampling", sampling)


@runtime_checkable
class JudgeInferenceBinding(Protocol):
    """The structural selection a composition host may bind to a judge plugin."""

    @property
    def id(self) -> str: ...

    @property
    def revision(self) -> str: ...

    @property
    def model(self) -> object: ...

    @property
    def sampling(self) -> Mapping[str, JsonValue]: ...

    @property
    def purpose(self) -> tuple[str, ...]: ...


__all__ = [
    "ExternalInferenceService",
    "HostedInferenceBinding",
    "HostedModel",
    "JudgeInferenceBinding",
    "validate_secret_free_http_url",
]
