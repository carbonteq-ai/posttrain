"""Provider-neutral contracts for deleting one OCI manifest by digest."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from posttrain.common import ContractError

_REFERENCE = re.compile(r"^(?P<repository>[A-Za-z0-9][A-Za-z0-9._:/-]*)@(?P<digest>sha256:[0-9a-f]{64})$")


@dataclass(frozen=True, slots=True)
class RegistryManifestRef:
    """An OCI repository and immutable manifest digest, never a tag."""

    repository: str
    digest: str

    def __post_init__(self) -> None:
        if not self.repository or "@" in self.repository or ":" in self.repository.rsplit("/", 1)[-1]:
            raise ContractError("OCI repository must not contain a tag")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.digest):
            raise ContractError("OCI manifest digest must be SHA-256")

    @property
    def value(self) -> str:
        return f"{self.repository}@{self.digest}"

    @classmethod
    def parse(cls, value: str) -> RegistryManifestRef:
        match = _REFERENCE.fullmatch(value)
        if match is None:
            raise ContractError("OCI deletion target must be repository@sha256:<64 hex>")
        return cls(repository=match.group("repository"), digest=match.group("digest"))


@dataclass(frozen=True, slots=True)
class RegistryManifestDeletePlan:
    """Authenticated HEAD result for one exact manifest deletion."""

    reference: RegistryManifestRef
    exists: bool
    eligible: bool
    blockers: tuple[str, ...]
    digest: str
    created_at: datetime

    def __post_init__(self) -> None:
        if len(set(self.blockers)) != len(self.blockers):
            raise ContractError("registry blockers must be unique")
        expected = _plan_digest(self.reference, self.exists, self.eligible, self.blockers)
        if self.digest != expected:
            raise ContractError("registry plan digest does not match its contents")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ContractError("registry plan time must be timezone-aware")

    @classmethod
    def build(
        cls,
        reference: RegistryManifestRef,
        *,
        exists: bool,
        eligible: bool = True,
        blockers: tuple[str, ...] = (),
        created_at: datetime | None = None,
    ) -> RegistryManifestDeletePlan:
        return cls(
            reference=reference,
            exists=exists,
            eligible=eligible,
            blockers=blockers,
            digest=_plan_digest(reference, exists, eligible, blockers),
            created_at=created_at or datetime.now(UTC),
        )


@dataclass(frozen=True, slots=True)
class RegistryManifestDeleteReceipt:
    reference: RegistryManifestRef
    plan_digest: str
    deleted: bool
    completed_at: datetime

    def __post_init__(self) -> None:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.plan_digest):
            raise ContractError("registry receipt digest must be SHA-256")
        if self.completed_at.tzinfo is None or self.completed_at.utcoffset() is None:
            raise ContractError("registry receipt time must be timezone-aware")


class RegistryLifecycleAdmin(Protocol):
    """Optional registry deletion capability used by the purge planner."""

    def plan_manifest_delete(self, reference: RegistryManifestRef) -> RegistryManifestDeletePlan: ...

    def delete_manifest(self, plan: RegistryManifestDeletePlan) -> RegistryManifestDeleteReceipt: ...


def _plan_digest(
    reference: RegistryManifestRef,
    exists: bool,
    eligible: bool,
    blockers: tuple[str, ...],
) -> str:
    payload = {
        "reference": reference.value,
        "exists": exists,
        "eligible": eligible,
        "blockers": list(blockers),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = [
    "RegistryLifecycleAdmin",
    "RegistryManifestDeletePlan",
    "RegistryManifestDeleteReceipt",
    "RegistryManifestRef",
]
