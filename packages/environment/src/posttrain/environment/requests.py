"""Portable, serializable contracts for independently packaged environments."""

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

from posttrain.common import JsonValue

_ID = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_PYTHON_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_RESOURCE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,127}$")

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
        module = getattr(factory, "__module__", "")
        attribute = getattr(factory, "__qualname__", "")
        if not module or not attribute or "<locals>" in attribute:
            raise ValueError("environment factories must be importable module-level callables")
        activation = cls(f"{module}:{attribute}")
        if _resolve_environment_factory(activation.reference) is not factory:
            raise ValueError("environment factory callable does not resolve to its declared import path")
        return activation


@dataclass(frozen=True, slots=True)
class ProjectPathActivationResource:
    """A regular file below the project root staged into an environment activation."""

    path: str

    def __post_init__(self) -> None:
        value = self.path
        normalized = PurePosixPath(value)
        if (
            not value
            or "\\" in value
            or normalized.is_absolute()
            or normalized.as_posix() != value
            or any(part in {"", ".", ".."} for part in normalized.parts)
        ):
            raise ValueError("project-path activation resource must be a normalized relative file path")

    @property
    def kind(self) -> Literal["project-path"]:
        return "project-path"

    def to_payload(self) -> dict[str, JsonValue]:
        return {"source": {"kind": self.kind, "path": self.path}}


type ActivationResource = ProjectPathActivationResource


@dataclass(frozen=True, slots=True)
class VerifiersV1ConfigActivation:
    """Declarative native Verifiers configuration activated at job startup."""

    config: Mapping[str, JsonValue]
    resources: Mapping[str, ActivationResource] = field(default_factory=dict)

    def __post_init__(self) -> None:
        config = dict(self.config)
        try:
            json.dumps(config, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("Verifiers activation config must contain only JSON values") from error
        resources = dict(self.resources)
        if any(not _RESOURCE_NAME.fullmatch(name) for name in resources):
            raise ValueError("Verifiers activation resource names must be stable identifiers")
        if any(not isinstance(resource, ProjectPathActivationResource) for resource in resources.values()):
            raise TypeError("Verifiers activation resources must use a supported source")
        object.__setattr__(self, "config", MappingProxyType(config))
        object.__setattr__(self, "resources", MappingProxyType(dict(sorted(resources.items()))))

    @property
    def kind(self) -> Literal["verifiers-config"]:
        return "verifiers-config"

    @property
    def digest(self) -> str:
        return _activation_digest(self.to_payload())

    def to_payload(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {"kind": self.kind, "config": dict(self.config)}
        if self.resources:
            payload["resources"] = {name: resource.to_payload() for name, resource in self.resources.items()}
        return payload

    def activate(self) -> object:
        try:
            from verifiers.v1.env import EnvConfig  # pyright: ignore[reportMissingImports]
        except ImportError as error:
            raise RuntimeError("install the Verifiers integration dependencies") from error
        return EnvConfig.model_validate(dict(self.config))


type EnvironmentActivation = PythonFactoryActivation | VerifiersV1ConfigActivation


def _activation_digest(payload: Mapping[str, JsonValue]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode()
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

    @property
    def kind(self) -> Literal["git"]:
        return "git"


@dataclass(frozen=True, slots=True)
class ProjectPathEnvironmentSource:
    """An installable environment package rooted below the selected project."""

    package: str
    path: str

    def __post_init__(self) -> None:
        _stable_id(self.package, "environment package")
        normalized = PurePosixPath(self.path)
        if (
            not self.path
            or "\\" in self.path
            or normalized.is_absolute()
            or normalized.as_posix() != self.path
            or any(part in {"", ".", "..", ".git"} for part in normalized.parts)
        ):
            raise ValueError("project-path environment source must be a normalized relative path")

    @property
    def kind(self) -> Literal["project-path"]:
        return "project-path"

    @property
    def revision(self) -> str:
        """Stable declared label for APIs that require a source revision string.

        Immutable package identity is the separately derived tree digest; this
        label deliberately does not claim to be a Git revision.
        """

        return f"project-path:{self.path}"


type EnvironmentPackageSource = EnvironmentSource | ProjectPathEnvironmentSource


def environment_source_payload(source: EnvironmentPackageSource) -> dict[str, str]:
    """Return the declared source without inventing Git metadata for a project path."""

    if isinstance(source, ProjectPathEnvironmentSource):
        return {"kind": source.kind, "path": source.path}
    return {
        "kind": source.kind,
        "repository": source.repository,
        "revision": source.revision,
        "subdirectory": source.subdirectory or ".",
    }


@dataclass(frozen=True, slots=True)
class SamplingPolicy:
    """Provider-independent sampling controls used by an environment cell."""

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
    source: EnvironmentPackageSource
    activation: EnvironmentActivation
    sampling: SamplingPolicy
    num_tasks: int
    num_rollouts: int = 1
    max_concurrent: int = 4
    qualification: Literal["required", "deferred"] = "required"
    parameters: Mapping[str, JsonValue] = field(default_factory=dict)
    reward_components: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _stable_id(self.id, "environment id")
        _stable_id(self.category, "environment category")
        if not isinstance(self.activation, (PythonFactoryActivation, VerifiersV1ConfigActivation)):
            raise TypeError("environment activation must be a supported serializable activation")
        if self.num_tasks < 1 or self.num_rollouts < 1 or self.max_concurrent < 1:
            raise ValueError("evaluation task, rollout, and concurrency counts must be positive")
        if self.qualification not in {"required", "deferred"}:
            raise ValueError("environment qualification must be required or deferred")
        if any(not value.strip() for value in self.reward_components):
            raise ValueError("environment reward component names cannot be empty")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    @property
    def revision(self) -> str:
        if isinstance(self.source, EnvironmentSource):
            return self.source.revision
        return self.source.revision

    def activate(self) -> object:
        return self.activation.activate()


__all__ = [
    "ActivationResource",
    "EnvironmentActivation",
    "EnvironmentBinding",
    "EnvironmentFactory",
    "EnvironmentSource",
    "environment_source_payload",
    "EnvironmentPackageSource",
    "PythonFactoryActivation",
    "ProjectPathActivationResource",
    "ProjectPathEnvironmentSource",
    "SamplingPolicy",
    "VerifiersV1ConfigActivation",
]
