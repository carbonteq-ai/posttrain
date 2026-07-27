"""Versioned worker manifest for reconstructing one registered job execution."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from posttrain.common import ContractError, JsonValue

from .contracts import RuntimeImageRef

_SCHEMA = "posttrain.execution-job.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@-]{0,255}$")
JOB_MANIFEST_PATH = PurePosixPath(".posttrain/job.json")
BUNDLE_MANIFEST_PATH = PurePosixPath(".posttrain/bundle.json")
WORKER_MANIFEST_PATH = PurePosixPath("/opt/posttrain/bundle") / JOB_MANIFEST_PATH
WORKER_COMMAND = (
    "posttrain-runtime",
    "execute",
    "--manifest",
    str(WORKER_MANIFEST_PATH),
)


def resolved_inputs_digest(inputs: Mapping[str, JsonValue]) -> str:
    """Return the canonical digest verified before a worker executes a job."""

    encoded = json.dumps(
        dict(inputs),
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ManifestMount:
    purpose: str
    container_path: str

    def __post_init__(self) -> None:
        if not self.purpose.strip():
            raise ContractError("execution manifest mount purpose cannot be empty")
        path = PurePosixPath(self.container_path)
        if not path.is_absolute() or path == PurePosixPath("/"):
            raise ContractError("execution manifest mount path must be an absolute non-root path")


@dataclass(frozen=True, slots=True)
class ExecutionJobManifest:
    run_id: str
    project_id: str
    work_package_id: str
    job_id: str
    job_definition_id: str
    provider: str
    project_manifest: str
    work_package: str
    runtime_image: str
    resolved_inputs_digest: str
    expected_artifact_roles: tuple[str, ...] = ()
    environment_names: tuple[str, ...] = ()
    mounts: tuple[ManifestMount, ...] = ()
    retention: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("run id", self.run_id),
            ("project id", self.project_id),
            ("work package id", self.work_package_id),
            ("job id", self.job_id),
            ("job definition id", self.job_definition_id),
            ("provider", self.provider),
        ):
            if not _IDENTITY.fullmatch(value):
                raise ContractError(f"execution manifest {name} is invalid")
        _bundle_relative(self.project_manifest, "project manifest")
        _bundle_relative(self.work_package, "work package")
        RuntimeImageRef(self.runtime_image)
        if not _SHA256.fullmatch(self.resolved_inputs_digest):
            raise ContractError("execution manifest resolved-input digest must be SHA-256")
        if len(set(self.expected_artifact_roles)) != len(self.expected_artifact_roles):
            raise ContractError("execution manifest artifact roles must be unique")
        if any(not role.strip() for role in self.expected_artifact_roles):
            raise ContractError("execution manifest artifact roles cannot be empty")
        if len(set(self.environment_names)) != len(self.environment_names):
            raise ContractError("execution manifest environment names must be unique")
        if any(not name.strip() or "=" in name for name in self.environment_names):
            raise ContractError("execution manifest accepts environment names, not values")
        paths = [mount.container_path for mount in self.mounts]
        if len(set(paths)) != len(paths):
            raise ContractError("execution manifest mount paths must be unique")

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "schema": _SCHEMA,
            "bundle_manifest": str(BUNDLE_MANIFEST_PATH),
            "run_id": self.run_id,
            "project_id": self.project_id,
            "work_package_id": self.work_package_id,
            "job_id": self.job_id,
            "job_definition_id": self.job_definition_id,
            "provider": self.provider,
            "project_manifest": self.project_manifest,
            "work_package": self.work_package,
            "runtime_image": self.runtime_image,
            "resolved_inputs_digest": self.resolved_inputs_digest,
            "expected_artifact_roles": list(self.expected_artifact_roles),
            "environment_names": list(self.environment_names),
            "mounts": [
                {
                    "purpose": mount.purpose,
                    "container_path": mount.container_path,
                }
                for mount in self.mounts
            ],
            "retention": dict(self.retention),
        }

    def to_bytes(self) -> bytes:
        return (json.dumps(self.to_payload(), indent=2, sort_keys=True) + "\n").encode()

    @classmethod
    def from_payload(cls, payload: object) -> ExecutionJobManifest:
        if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
            raise ContractError("execution job manifest schema is unsupported")
        if payload.get("bundle_manifest") != str(BUNDLE_MANIFEST_PATH):
            raise ContractError("execution job manifest bundle reference is unsupported")
        allowed = {
            "schema",
            "bundle_manifest",
            "run_id",
            "project_id",
            "work_package_id",
            "job_id",
            "job_definition_id",
            "provider",
            "project_manifest",
            "work_package",
            "runtime_image",
            "resolved_inputs_digest",
            "expected_artifact_roles",
            "environment_names",
            "mounts",
            "retention",
        }
        if unknown := sorted(set(payload) - allowed):
            raise ContractError(f"execution job manifest has unknown fields: {', '.join(unknown)}")
        try:
            mounts_payload = payload.get("mounts", [])
            if not isinstance(mounts_payload, list):
                raise TypeError("mounts")
            mounts = tuple(_mount_from_payload(value) for value in mounts_payload)
            roles = _string_tuple(payload.get("expected_artifact_roles", []), "artifact roles")
            environments = _string_tuple(payload.get("environment_names", []), "environment names")
            retention = payload.get("retention", {})
            if not isinstance(retention, dict):
                raise TypeError("retention")
            return cls(
                run_id=str(payload["run_id"]),
                project_id=str(payload["project_id"]),
                work_package_id=str(payload["work_package_id"]),
                job_id=str(payload["job_id"]),
                job_definition_id=str(payload["job_definition_id"]),
                provider=str(payload["provider"]),
                project_manifest=str(payload["project_manifest"]),
                work_package=str(payload["work_package"]),
                runtime_image=str(payload["runtime_image"]),
                resolved_inputs_digest=str(payload["resolved_inputs_digest"]),
                expected_artifact_roles=roles,
                environment_names=environments,
                mounts=mounts,
                retention=retention,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("execution job manifest fields are invalid") from error

    @classmethod
    def load(cls, path: Path) -> ExecutionJobManifest:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ContractError(f"execution job manifest is missing: {path}") from None
        except json.JSONDecodeError as error:
            raise ContractError(f"execution job manifest is invalid: {path}") from error
        return cls.from_payload(payload)


def _bundle_relative(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ContractError(f"execution manifest {label} must be a normalized bundle-relative path")
    return path


def _mount_from_payload(value: object) -> ManifestMount:
    if not isinstance(value, dict) or set(value) != {"purpose", "container_path"}:
        raise TypeError("mount")
    return ManifestMount(str(value["purpose"]), str(value["container_path"]))


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(label)
    return tuple(value)


__all__ = [
    "BUNDLE_MANIFEST_PATH",
    "JOB_MANIFEST_PATH",
    "ExecutionJobManifest",
    "ManifestMount",
    "resolved_inputs_digest",
]
