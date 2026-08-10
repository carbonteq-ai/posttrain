"""Durable compact records for materialized job packages."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from posttrain.common import ContractError, JsonValue
from posttrain.execution import JobPackageManifest

_SCHEMA = "posttrain.package-materialization.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class PackageMaterializationRecord:
    """Portable evidence that one package context was materialized.

    The record contains no local filesystem path.  The manifest is the
    immutable job meaning; the context digest identifies the assembled bytes,
    and the publication key links those bytes to a registry publication plan.
    """

    package_key: str
    context_digest: str
    publication_key: str
    manifest: JobPackageManifest
    plan_key: str | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("package", self.package_key),
            ("context", self.context_digest),
            ("publication", self.publication_key),
        ):
            if _SHA256.fullmatch(value) is None:
                raise ContractError(f"materialization record {label} key must be SHA-256")
        if self.manifest.package_key != self.package_key:
            raise ContractError("materialization record package key differs from its manifest")
        if self.plan_key is not None and _SHA256.fullmatch(self.plan_key) is None:
            raise ContractError("materialization record plan key must be SHA-256")

    @property
    def manifest_digest(self) -> str:
        """Return the digest of the canonical manifest bytes."""

        return hashlib.sha256(self.manifest.to_bytes()).hexdigest()

    def to_payload(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "schema": _SCHEMA,
            "package_key": self.package_key,
            "context_digest": self.context_digest,
            "publication_key": self.publication_key,
            "manifest_digest": self.manifest_digest,
            "manifest": self.manifest.to_payload(),
        }
        if self.plan_key is not None:
            payload["plan_key"] = self.plan_key
        return payload

    def to_bytes(self) -> bytes:
        return (json.dumps(self.to_payload(), indent=2, sort_keys=True) + "\n").encode()

    @classmethod
    def from_payload(cls, payload: object) -> PackageMaterializationRecord:
        if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
            raise ContractError("materialization record schema is unsupported")
        allowed = {
            "schema",
            "package_key",
            "context_digest",
            "publication_key",
            "manifest_digest",
            "manifest",
            "plan_key",
        }
        if unknown := sorted(set(payload) - allowed):
            raise ContractError(f"materialization record has unknown fields: {', '.join(unknown)}")
        try:
            manifest = JobPackageManifest.from_payload(payload["manifest"])
            record = cls(
                package_key=str(payload["package_key"]),
                context_digest=str(payload["context_digest"]),
                publication_key=str(payload["publication_key"]),
                manifest=manifest,
                plan_key=(str(payload["plan_key"]) if payload.get("plan_key") is not None else None),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("materialization record fields are invalid") from error
        if payload.get("manifest_digest") != record.manifest_digest:
            raise ContractError("materialization record manifest digest is invalid")
        return record

    @classmethod
    def from_bytes(cls, value: bytes) -> PackageMaterializationRecord:
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContractError("materialization record is invalid JSON") from error
        return cls.from_payload(payload)


class PackageMaterializationStore:
    """Small application-supplied store for compact package records."""

    def __init__(self, root: Path) -> None:
        if not root.is_absolute():
            raise ContractError("materialization record root must be absolute")
        self._root = root

    def resolve(self, plan_key: str) -> PackageMaterializationRecord | None:
        if _SHA256.fullmatch(plan_key) is None:
            raise ContractError("materialization record plan key must be SHA-256")
        if not self._root.is_dir():
            return None
        for path in self._root.glob("*.json"):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                record = PackageMaterializationRecord.from_bytes(path.read_bytes())
            except (OSError, ContractError):
                continue
            if record.plan_key == plan_key:
                return record
        return None

    def commit(self, record: PackageMaterializationRecord) -> Path:
        self._root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._root.chmod(0o700)
        path = self._root / f"{record.package_key}.json"
        encoded = record.to_bytes()
        temporary = self._root / f".{record.package_key}.tmp"
        try:
            descriptor = temporary.open("xb")
        except FileExistsError:
            temporary.unlink(missing_ok=True)
            descriptor = temporary.open("xb")
        try:
            descriptor.write(encoded)
            descriptor.flush()
            os.fsync(descriptor.fileno())
        finally:
            descriptor.close()
        try:
            path.hardlink_to(temporary)
        except FileExistsError:
            existing = PackageMaterializationRecord.from_bytes(path.read_bytes())
            if existing != record:
                temporary.unlink(missing_ok=True)
                raise ContractError("materialization record conflicts with an existing package") from None
        finally:
            temporary.unlink(missing_ok=True)
        path.chmod(0o600)
        return path


__all__ = ["PackageMaterializationRecord", "PackageMaterializationStore"]
