"""Framework-neutral values for explainable configuration validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .artifacts import JsonValue

_CODE = re.compile(r"^[A-Z][A-Z0-9_]*$")

type SettingOriginKind = Literal["explicit", "default", "derived", "runtime"]
type ConfigurationSeverity = Literal["error", "warning", "recommendation", "info"]
type ValidationStage = Literal["static", "readiness", "runtime", "request"]
type CheckOutcome = Literal["passed", "failed", "deferred", "skipped", "not_applicable"]


@dataclass(frozen=True, slots=True)
class SettingOrigin:
    """Where one effective configuration value came from."""

    path: str
    value: JsonValue
    kind: SettingOriginKind
    source: str
    revision: str | None = None

    def __post_init__(self) -> None:
        if not self.path.strip() or not self.source.strip():
            raise ValueError("setting origins require a path and source")
        if self.revision is not None and not self.revision.strip():
            raise ValueError("setting origin revision cannot be empty")

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "path": self.path,
            "value": self.value,
            "kind": self.kind,
            "source": self.source,
            "revision": self.revision,
        }


@dataclass(frozen=True, slots=True)
class ConfigurationIssue:
    """One safe, stable configuration finding for humans and automation."""

    code: str
    severity: ConfigurationSeverity
    stage: ValidationStage
    role: str
    path: str
    message: str
    hint: str | None = None
    related_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if _CODE.fullmatch(self.code) is None:
            raise ValueError("configuration issue code must be uppercase snake case")
        for name, value in (("role", self.role), ("path", self.path), ("message", self.message)):
            if not value.strip():
                raise ValueError(f"configuration issue {name} cannot be empty")
        if self.hint is not None and not self.hint.strip():
            raise ValueError("configuration issue hint cannot be empty")
        if any(not path.strip() for path in self.related_paths):
            raise ValueError("configuration issue related paths cannot be empty")
        if len(set(self.related_paths)) != len(self.related_paths):
            raise ValueError("configuration issue related paths must be unique")

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "code": self.code,
            "severity": self.severity,
            "stage": self.stage,
            "role": self.role,
            "path": self.path,
            "message": self.message,
            "hint": self.hint,
            "related_paths": list(self.related_paths),
        }


__all__ = [
    "CheckOutcome",
    "ConfigurationIssue",
    "ConfigurationSeverity",
    "SettingOrigin",
    "SettingOriginKind",
    "ValidationStage",
]
