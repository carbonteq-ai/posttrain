"""Provider-neutral execution-setting precedence for opened projects."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal

from posttrain.catalog import ProjectExecutionDefaults
from posttrain.common import ContractError

type SettingSource = Literal["cli", "local", "project", "job"]


@dataclass(frozen=True, slots=True)
class ExecutionOverrides:
    """An optional setting layer applied while planning one job."""

    provider: str | None = None
    target: str | None = None
    runtime_profile: str | None = None
    timeout_seconds: int | None = None
    max_attempts: int | None = None
    priority: int | None = None
    environment_names: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class PackageOverrides:
    """Options allowed to change immutable package contents."""

    target: str | None = None
    runtime_profile: str | None = None
    registry_prefix: str | None = None

    def as_execution_overrides(self) -> ExecutionOverrides:
        return ExecutionOverrides(target=self.target, runtime_profile=self.runtime_profile)


@dataclass(frozen=True, slots=True)
class LaunchOverrides:
    """Options that affect one provider launch but not package bytes."""

    provider: str | None = None
    timeout_seconds: int | None = None
    max_attempts: int | None = None
    priority: int | None = None
    environment_names: tuple[str, ...] | None = None

    def as_execution_overrides(self) -> ExecutionOverrides:
        return ExecutionOverrides(
            provider=self.provider,
            timeout_seconds=self.timeout_seconds,
            max_attempts=self.max_attempts,
            priority=self.priority,
            environment_names=self.environment_names,
        )


@dataclass(frozen=True, slots=True)
class ResolvedExecutionSettings:
    """The concrete settings and the layer that supplied every setting."""

    provider: str
    target: str | None
    runtime_profile: str | None
    timeout_seconds: int
    max_attempts: int
    priority: int
    environment_names: tuple[str, ...]
    sources: dict[str, SettingSource]


_DEFAULT_JOB_SETTINGS = ExecutionOverrides(
    provider="local",
    runtime_profile="framework/job@1",
    timeout_seconds=3600,
    max_attempts=1,
    priority=0,
    environment_names=(),
)


def resolve_execution_settings(
    project: ProjectExecutionDefaults,
    *,
    local: ExecutionOverrides | None = None,
    cli: ExecutionOverrides | None = None,
    job: ExecutionOverrides | None = None,
) -> ResolvedExecutionSettings:
    """Resolve CLI, local, project, then job defaults with provenance."""

    project_layer = ExecutionOverrides(
        provider=project.provider,
        target=project.target,
        runtime_profile=project.runtime_profile,
        timeout_seconds=project.timeout_seconds,
        max_attempts=project.max_attempts,
        priority=project.priority,
        environment_names=project.environment_names or None,
    )
    layers: tuple[tuple[SettingSource, ExecutionOverrides], ...] = (
        ("cli", cli or ExecutionOverrides()),
        ("local", local or ExecutionOverrides()),
        ("project", project_layer),
        ("job", job or _DEFAULT_JOB_SETTINGS),
    )
    values: dict[str, object] = {}
    sources: dict[str, SettingSource] = {}
    for field in fields(ExecutionOverrides):
        if field.name == "environment_names":
            continue
        for source, layer in layers:
            value = getattr(layer, field.name)
            if value is not None:
                values[field.name] = value
                sources[field.name] = source
                break
    environment_names: list[str] = []
    environment_source: SettingSource = "job"
    for source, layer in reversed(layers):
        if layer.environment_names is None:
            continue
        environment_source = source
        for name in layer.environment_names:
            if name not in environment_names:
                environment_names.append(name)
    values["environment_names"] = tuple(environment_names)
    sources["environment_names"] = environment_source

    provider = values.get("provider")
    timeout_seconds = values.get("timeout_seconds")
    max_attempts = values.get("max_attempts")
    priority = values.get("priority")
    resolved_environment_names = values.get("environment_names")
    if not isinstance(provider, str):
        raise ContractError("execution provider could not be resolved")
    if not isinstance(timeout_seconds, int):
        raise ContractError("execution timeout could not be resolved")
    if not isinstance(max_attempts, int):
        raise ContractError("execution attempts could not be resolved")
    if not isinstance(priority, int):
        raise ContractError("execution priority could not be resolved")
    if not isinstance(resolved_environment_names, tuple):
        raise ContractError("execution environment names could not be resolved")
    resolved = ResolvedExecutionSettings(
        provider=provider,
        target=_optional_string(values.get("target")),
        runtime_profile=_optional_string(values.get("runtime_profile")),
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        priority=priority,
        environment_names=resolved_environment_names,
        sources=sources,
    )
    _validate_resolved(resolved)
    return resolved


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _validate_resolved(settings: ResolvedExecutionSettings) -> None:
    if not settings.provider.strip():
        raise ContractError("resolved execution provider cannot be empty")
    if settings.timeout_seconds < 1 or settings.max_attempts < 1:
        raise ContractError("resolved execution timeout and attempts must be positive")
    if len(set(settings.environment_names)) != len(settings.environment_names):
        raise ContractError("resolved execution environment names must be unique")
    if any(not name.strip() or "=" in name for name in settings.environment_names):
        raise ContractError("resolved execution environment entries must be variable names")


__all__ = [
    "ExecutionOverrides",
    "LaunchOverrides",
    "PackageOverrides",
    "ResolvedExecutionSettings",
    "SettingSource",
    "resolve_execution_settings",
]
