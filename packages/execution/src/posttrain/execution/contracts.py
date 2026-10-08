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
type ExecutionLogStream = Literal["workload", "diagnostic"]
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

# Container runtimes default /dev/shm to 64 MiB. vLLM's shared-memory message
# queues, NCCL's intra-node transport and PyTorch dataloader workers allocate
# far more (veRL's rollout server needs 160 MiB at start), so every job
# container gets an explicit size. A target may declare ``shm_size_gb`` in its
# placement; ``host_memory_gb`` bounds it. /dev/shm is a tmpfs limit, not a
# reservation: pages count against the container's memory only once written.
DEFAULT_SHARED_MEMORY_GB = 16


def execution_shared_memory_gb(target: ExecutionTarget) -> int:
    """The /dev/shm size, in GiB, every job container on ``target`` receives.

    An explicit placement ``shm_size_gb`` wins. Otherwise the framework default
    applies, capped at half of a declared ``host_memory_gb`` so the tmpfs can
    never claim most of a small host. An explicit size larger than the declared
    host memory is a contradictory target and is rejected.
    """

    declared_host_memory = target.placement.get("host_memory_gb")
    host_memory: float | None = None
    if declared_host_memory is not None:
        if (
            isinstance(declared_host_memory, bool)
            or not isinstance(declared_host_memory, int | float)
            or declared_host_memory <= 0
        ):
            raise ContractError(f"execution target {target.id} host_memory_gb must be a positive number")
        host_memory = float(declared_host_memory)
    requested = target.placement.get("shm_size_gb")
    if requested is not None:
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
            raise ContractError(f"execution target {target.id} shm_size_gb must be a positive integer")
        if host_memory is not None and requested > host_memory:
            raise ContractError(
                f"execution target {target.id} shm_size_gb {requested} exceeds its host_memory_gb {host_memory:g}"
            )
        return requested
    if host_memory is None:
        return DEFAULT_SHARED_MEMORY_GB
    return max(1, min(DEFAULT_SHARED_MEMORY_GB, int(host_memory // 2)))


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
        execution_shared_memory_gb(self.target)

    @property
    def shared_memory_gb(self) -> int:
        """The /dev/shm size, in GiB, the provider must give this job's container."""

        return execution_shared_memory_gb(self.target)

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
                    if name in {"model_source", "recovery_checkpoint", "curriculum_state"}
                },
            },
        }
        return {
            **self.compile_cache_environment(),
            EXECUTION_LAUNCH_ENVIRONMENT: json.dumps(
                payload,
                separators=(",", ":"),
                sort_keys=True,
            ),
        }

    def compile_cache_environment(self) -> dict[str, str]:
        """Point every kernel and graph compiler at the persistent compile-cache mount.

        Without these, vLLM's torch.compile output, Triton and FlashInfer kernels, and the
        CUDA driver's compute cache land in the container's home directory and are rebuilt
        on every start (about 80 s of CUDA graph capture on the 2.6B eval server).
        """

        mounts = [mount for mount in self.mounts if mount.purpose == "compile-cache"]
        if not mounts:
            return {}
        root = mounts[0].container_path
        return {
            "VLLM_CACHE_ROOT": str(root / "vllm"),
            "TRITON_CACHE_DIR": str(root / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(root / "torchinductor"),
            "FLASHINFER_WORKSPACE_BASE": str(root / "flashinfer"),
            "CUDA_CACHE_PATH": str(root / "nv"),
            "CUDA_CACHE_MAXSIZE": str(4 * 1024**3),
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
        stream: ExecutionLogStream = "workload",
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
