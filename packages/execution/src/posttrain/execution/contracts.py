"""Stable execution requests and lifecycle values independent of a scheduler."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from posttrain.common import ContractError, ExecutionTarget, JsonValue, PublishedArtifact
from posttrain.tracking import RunSpec

type ExecutionState = Literal[
    "planned",
    "queued",
    "starting",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "lost",
]
type ExecutionMountPurpose = Literal["model-cache", "compile-cache", "run-workspace"]
type ProviderCleanupDisposition = Literal[
    "removed",
    "already-absent",
    "not-created",
    "provider-managed",
]


class ProviderCleanupDeferred(RuntimeError):
    """Exact provider cleanup is safe to retry but cannot complete yet."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ACTUAL_JOB_COMMAND = (
    "posttrain-runtime",
    "execute",
    "--manifest",
    "/opt/posttrain/job/package.json",
)
EXECUTION_LAUNCH_ENVIRONMENT = "POSTTRAIN_EXECUTION"


@dataclass(frozen=True, slots=True)
class BundleRef:
    """Content-addressed bundle location, planned or already materialized."""

    path: Path
    digest: str

    def __post_init__(self) -> None:
        if not self.path.is_absolute():
            raise ContractError("execution bundle path must be absolute")
        if self.path.exists() and not self.path.is_dir():
            raise ContractError("execution bundle path must be a directory when materialized")
        if not _SHA256.fullmatch(self.digest):
            raise ContractError("execution bundle digest must be SHA-256")


@dataclass(frozen=True, slots=True)
class RuntimeImageRef:
    value: str

    def __post_init__(self) -> None:
        if "@sha256:" not in self.value or not _SHA256.fullmatch(self.value.rsplit("@sha256:", 1)[1]):
            raise ContractError("runtime image must use an immutable sha256 digest")

    @property
    def digest(self) -> str:
        """Return the image manifest digest without the ``sha256:`` prefix."""

        return self.value.rsplit("@sha256:", 1)[1]


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    timeout_seconds: int
    max_attempts: int = 1
    priority: int = 0

    def __post_init__(self) -> None:
        if self.timeout_seconds < 1 or self.max_attempts < 1:
            raise ContractError("execution timeout and attempts must be positive")


@dataclass(frozen=True, slots=True)
class ExecutionMount:
    """One provider-neutral host path made available to an execution."""

    instance_path: Path
    container_path: Path
    purpose: ExecutionMountPurpose
    optional: bool = False

    def __post_init__(self) -> None:
        if not self.instance_path.is_absolute() or not self.container_path.is_absolute():
            raise ContractError("execution mount paths must be absolute")
        if self.instance_path == Path("/") or self.container_path == Path("/"):
            raise ContractError("execution mounts cannot expose a filesystem root")


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    """One run-specific launch envelope for an immutable actual-job image."""

    run_spec: RunSpec
    job_definition_id: str
    image: RuntimeImageRef
    target: ExecutionTarget
    command: tuple[str, ...]
    idempotency_key: str
    policy: ExecutionPolicy
    attempt: int = 1
    environment_names: tuple[str, ...] = ()
    mounts: tuple[ExecutionMount, ...] = ()
    # A machine-local transport tag for a direct daemon-loaded image. The
    # immutable ``image`` digest remains authoritative for identity and
    # remote providers ignore this optional field.
    local_image: str | None = None
    # Read/migration compatibility for pre-OCI plans. Normal providers never
    # upload or mount this directory; new callers must leave it unset.
    bundle: BundleRef | None = None

    def __post_init__(self) -> None:
        if self.job_definition_id != self.run_spec.job_definition_version:
            raise ContractError("execution job definition conflicts with RunSpec")
        if not self.command or any(not part for part in self.command):
            raise ContractError("execution command cannot be empty")
        if self.bundle is None and self.command[: len(_ACTUAL_JOB_COMMAND)] != _ACTUAL_JOB_COMMAND:
            raise ContractError("actual-job execution must use the stable packaged worker entrypoint")
        if not self.idempotency_key.strip():
            raise ContractError("execution idempotency key cannot be empty")
        if self.attempt < 1:
            raise ContractError("execution attempt must be positive")
        if len(set(self.environment_names)) != len(self.environment_names):
            raise ContractError("execution environment names must be unique")
        if any("=" in name or not name.strip() for name in self.environment_names):
            raise ContractError("execution request accepts environment names, not secret values")
        container_paths = [mount.container_path for mount in self.mounts]
        if len(set(container_paths)) != len(container_paths):
            raise ContractError("execution mount container paths must be unique")
        for mount in self.mounts:
            if mount.purpose == "run-workspace" and self.run_spec.run_id not in mount.instance_path.parts:
                raise ContractError("run workspace mount must contain the run id as one path component")
        if self.local_image is not None and (
            not self.local_image.startswith("posttrain-local:")
            or not self.local_image.strip()
            or "@" in self.local_image
            or any(character.isspace() for character in self.local_image)
        ):
            raise ContractError("local execution image tag is invalid")

    def launch_environment(self, *, provider: str) -> dict[str, str]:
        """Encode non-secret run context separately from the packaged job."""

        if not provider.strip():
            raise ContractError("execution launch provider cannot be empty")
        payload = {
            "schema": "posttrain.execution-launch.v1",
            "run": {
                "run_id": self.run_spec.run_id,
                "project_id": self.run_spec.project_id,
                "work_package_id": self.run_spec.work_package_id,
                "stage": self.run_spec.stage,
                "job_kind": self.run_spec.job_kind,
                "job_definition_id": self.job_definition_id,
            },
            "attempt": self.attempt,
            "provider": provider,
            "job_image": self.image.value,
            "target": {
                "id": self.target.id,
                "revision": self.target.revision,
                "device_class": self.target.device_class,
                "memory_gb": self.target.memory_gb,
                "placement": dict(self.target.placement),
                "host_constraints": dict(self.target.host_constraints),
            },
            # Artifact inputs are run-scoped selections.  Carrying them in
            # the launch envelope lets an explicit checkpoint/model binding
            # survive the worker's reconstruction of the packaged job.
            "overrides": {
                "artifacts": {
                    name: {
                        "kind": item.kind,
                        "reference": {
                            "provider": item.reference.provider,
                            "namespace": item.reference.namespace,
                            "name": item.reference.name,
                            "version": item.reference.version,
                            "digest": item.reference.digest,
                            "provider_metadata": dict(item.reference.provider_metadata),
                        },
                    }
                    for name, item in self.run_spec.artifacts.items()
                },
                "resolved_inputs": {
                    name: value
                    for name, value in self.run_spec.resolved_inputs.items()
                    if name in {"model_source", "recovery_checkpoint"}
                },
            },
        }
        return {
            EXECUTION_LAUNCH_ENVIRONMENT: json.dumps(
                payload,
                separators=(",", ":"),
                sort_keys=True,
            )
        }


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    provider: str
    request: ExecutionRequest
    native_plan_id: str | None = None
    details: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExecutionHandle:
    provider: str
    provider_id: str
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    handle: ExecutionHandle
    state: ExecutionState
    attempt: int
    target_id: str
    observed_at: datetime
    native_state: str
    message: str | None = None


@dataclass(frozen=True, slots=True)
class LogCursor:
    offset: int = 0

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ContractError("log cursor offset cannot be negative")


@dataclass(frozen=True, slots=True)
class LogPage:
    lines: tuple[str, ...]
    next_cursor: LogCursor
    truncated: bool


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    record: ExecutionRecord
    exit_code: int | None
    tracking_run_id: str | None = None
    published_artifacts: tuple[PublishedArtifact, ...] = ()


@dataclass(frozen=True, slots=True)
class ProviderCleanupResult:
    """Provider-owned resources released after a terminal execution."""

    handle: ExecutionHandle
    disposition: ProviderCleanupDisposition
    message: str
    workspace_disposition: ProviderCleanupDisposition = "provider-managed"
    workspace_reclaimed_bytes: int = 0

    def __post_init__(self) -> None:
        if self.workspace_reclaimed_bytes < 0:
            raise ContractError("cleanup reclaimed bytes cannot be negative")


class ExecutionProvider(Protocol):
    def plan(self, request: ExecutionRequest) -> ExecutionPlan: ...

    def submit(self, plan: ExecutionPlan) -> ExecutionHandle: ...

    def status(self, handle: ExecutionHandle) -> ExecutionRecord: ...

    def logs(
        self,
        handle: ExecutionHandle,
        cursor: LogCursor | None = None,
        *,
        limit: int = 200,
    ) -> LogPage: ...

    def cancel(self, handle: ExecutionHandle) -> None: ...

    def collect(self, handle: ExecutionHandle) -> ExecutionResult: ...

    def cleanup(
        self,
        handle: ExecutionHandle,
        *,
        run_id: str,
        run_workspace: Path | None,
        runtime_image: RuntimeImageRef,
    ) -> ProviderCleanupResult: ...
