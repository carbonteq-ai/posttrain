"""Stable configuration report values produced during detached job planning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from posttrain.common import CheckOutcome, ConfigurationIssue, JsonValue, SettingOrigin


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    id: str
    outcome: CheckOutcome
    detail: str

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.detail.strip():
            raise ValueError("validation checks require an id and detail")

    def as_dict(self) -> dict[str, JsonValue]:
        return {"id": self.id, "outcome": self.outcome, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class JobValidationReport:
    """Provider-free configuration explanation bound to resolved job inputs."""

    resolved_input_digest: str
    origins: tuple[SettingOrigin, ...] = ()
    issues: tuple[ConfigurationIssue, ...] = ()
    checks: tuple[ValidationCheck, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if len(self.resolved_input_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.resolved_input_digest
        ):
            raise ValueError("validation report requires a sha256 resolved-input digest")
        if self.schema_version < 1:
            raise ValueError("validation report schema version must be positive")

    @classmethod
    def for_resolved_inputs(
        cls,
        resolved_inputs: dict[str, JsonValue],
        *,
        origins: tuple[SettingOrigin, ...] = (),
        issues: tuple[ConfigurationIssue, ...] = (),
        checks: tuple[ValidationCheck, ...] = (),
    ) -> JobValidationReport:
        payload = json.dumps(resolved_inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return cls(hashlib.sha256(payload.encode()).hexdigest(), origins, issues, checks)

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "resolved_input_digest": self.resolved_input_digest,
            "origins": [origin.as_dict() for origin in self.origins],
            "issues": [issue.as_dict() for issue in self.issues],
            "checks": [check.as_dict() for check in self.checks],
        }


__all__ = ["JobValidationReport", "ValidationCheck"]
