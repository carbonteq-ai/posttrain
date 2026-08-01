"""Secret-free provider identity retained with detached execution intent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from posttrain.common import ContractError


@dataclass(frozen=True, slots=True)
class ExecutionProviderSource:
    """Stable adapter configuration whose referenced credentials may rotate."""

    provider: str
    profile_id: str
    binding_fingerprint: str
    endpoint_scope: str | None = None
    adapter_python: Path | None = None
    credential_file: Path | None = None
    trust_bundle: Path | None = None
    canonical_hostname: str | None = None
    capacity_wait_seconds: int = 0

    def __post_init__(self) -> None:
        for label, value in (
            ("provider", self.provider),
            ("profile id", self.profile_id),
            ("binding fingerprint", self.binding_fingerprint),
        ):
            if not value.strip() or "\x00" in value:
                raise ContractError(f"execution provider {label} is invalid")
        for label, path in (
            ("adapter Python", self.adapter_python),
            ("credential file", self.credential_file),
            ("trust bundle", self.trust_bundle),
        ):
            if path is not None and not path.is_absolute():
                raise ContractError(f"execution provider {label} must be absolute")
        if self.endpoint_scope is not None and (not self.endpoint_scope.strip() or "\x00" in self.endpoint_scope):
            raise ContractError("execution provider endpoint scope is invalid")
        if self.canonical_hostname is not None and (
            not self.canonical_hostname.strip() or "\x00" in self.canonical_hostname
        ):
            raise ContractError("execution provider canonical hostname is invalid")
        if self.capacity_wait_seconds < 0:
            raise ContractError("execution provider capacity wait must not be negative")

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "profile_id": self.profile_id,
            "binding_fingerprint": self.binding_fingerprint,
            "endpoint_scope": self.endpoint_scope,
            "adapter_python": str(self.adapter_python) if self.adapter_python is not None else None,
            "credential_file": str(self.credential_file) if self.credential_file is not None else None,
            "trust_bundle": str(self.trust_bundle) if self.trust_bundle is not None else None,
            "canonical_hostname": self.canonical_hostname,
            "capacity_wait_seconds": self.capacity_wait_seconds,
        }

    @classmethod
    def from_dict(cls, value: object) -> ExecutionProviderSource:
        if not isinstance(value, dict):
            raise ContractError("execution provider source is invalid")
        payload: dict[str, Any] = value
        try:
            return cls(
                provider=str(payload["provider"]),
                profile_id=str(payload["profile_id"]),
                binding_fingerprint=str(payload["binding_fingerprint"]),
                endpoint_scope=(str(payload["endpoint_scope"]) if payload.get("endpoint_scope") is not None else None),
                adapter_python=(
                    Path(str(payload["adapter_python"])) if payload.get("adapter_python") is not None else None
                ),
                credential_file=(
                    Path(str(payload["credential_file"])) if payload.get("credential_file") is not None else None
                ),
                trust_bundle=(Path(str(payload["trust_bundle"])) if payload.get("trust_bundle") is not None else None),
                canonical_hostname=(
                    str(payload["canonical_hostname"]) if payload.get("canonical_hostname") is not None else None
                ),
                capacity_wait_seconds=int(payload.get("capacity_wait_seconds", 0)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("execution provider source is invalid") from error


__all__ = ["ExecutionProviderSource"]
