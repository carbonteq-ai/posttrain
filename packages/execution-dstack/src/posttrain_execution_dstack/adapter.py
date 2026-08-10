"""Thin dstack lifecycle adapter over an isolated Python-SDK bridge."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from posttrain.execution import (
    ExecutionHandle,
    ExecutionPlan,
    ExecutionRecord,
    ExecutionRequest,
    ExecutionResult,
    ExecutionState,
    LogCursor,
    LogPage,
    ProviderCleanupDeferred,
    ProviderCleanupResult,
    RuntimeImageRef,
)

TRUST_BUNDLE_CONTAINER_PATH = Path("/opt/posttrain/trust/ca-certificates.crt")
# Online-RL environments and vLLM open many short-lived sockets/files while a
# large rollout population is in flight. Keep this as a provider-side launch
# guard rather than relying on the worker image's inherited shell limit.
POSTTRAIN_NOFILE_LIMIT = 65536
# The job image merges this with the authorities it already trusts. Setting
# SSL_CERT_FILE here instead would replace that set rather than extend it,
# leaving an internally-trusting job unable to verify anything public.
_EXTRA_TRUST_VARIABLE = "POSTTRAIN_EXTRA_CA_BUNDLE"
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HOSTNAME = re.compile(r"^(?=.{1,253}$)[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?$")

_STATE: dict[str, ExecutionState] = {
    "pending": "queued",
    "submitted": "queued",
    "provisioning": "starting",
    "running": "running",
    # dstack enters TERMINATING after both successful commands and explicit
    # stops. It is not terminal; wait for DONE, FAILED, or TERMINATED.
    "terminating": "running",
    "terminated": "cancelled",
    "failed": "failed",
    "done": "succeeded",
}


class DstackGateway(Protocol):
    def invoke(self, action: str, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


class DstackSdkBridge:
    """Call dstack's Python SDK in its Pydantic-1-compatible environment."""

    def __init__(
        self,
        python: Path,
        bridge: Path | None = None,
        *,
        environment_file: Path | None = None,
        runtime_environment: Mapping[str, str] | None = None,
    ) -> None:
        # Preserve a virtualenv's ``bin/python`` symlink: resolving it escapes
        # the environment and loses the dstack installation.
        self._python = python.absolute()
        self._bridge = (bridge or Path(__file__).with_name("sdk_bridge.py")).resolve()
        self._environment_file = environment_file.resolve() if environment_file else None
        # The SDK process may need ambient or provider-credential values to
        # authenticate to dstack. Job values are deliberately a separate map:
        # they cross the bridge only for names declared by the execution
        # request, never by reading the submitting shell.
        self._runtime_environment = dict(runtime_environment or {})
        if not self._python.is_file():
            raise FileNotFoundError(self._python)
        if not self._bridge.is_file():
            raise FileNotFoundError(self._bridge)
        if self._environment_file is not None and not self._environment_file.is_file():
            raise FileNotFoundError(self._environment_file)

    def _environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        if self._environment_file is not None:
            for raw_line in self._environment_file.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                name, separator, raw_value = line.partition("=")
                if not separator or not name.strip():
                    raise RuntimeError("invalid dstack bridge environment file")
                parsed = shlex.split(raw_value, comments=False, posix=True)
                if len(parsed) != 1:
                    raise RuntimeError("invalid dstack bridge environment value")
                environment[name.strip()] = parsed[0]
        return environment

    def invoke(self, action: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        bridge_payload = dict(payload)
        configuration = payload.get("configuration")
        if configuration is not None:
            if not isinstance(configuration, Mapping):
                raise RuntimeError("dstack SDK bridge configuration is invalid")
            bridge_configuration = dict(configuration)
            # This private field is added immediately before the isolated bridge
            # process. It is intentionally absent from plans, submission receipts,
            # and gateway-visible configuration so no public representation can
            # accidentally serialize a secret value.
            bridge_configuration["_posttrain_runtime_env"] = dict(self._runtime_environment)
            bridge_payload["configuration"] = bridge_configuration
        result = subprocess.run(
            [str(self._python), str(self._bridge), action],
            input=json.dumps(bridge_payload, separators=(",", ":"), sort_keys=True),
            text=True,
            capture_output=True,
            check=False,
            env=self._environment(),
        )
        if result.returncode != 0:
            error_lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
            detail = error_lines[-1][:500] if error_lines else "no diagnostic was returned"
            raise RuntimeError(f"dstack SDK bridge {action} failed with exit code {result.returncode}: {detail}")
        value = json.loads(result.stdout)
        if not isinstance(value, dict):
            raise RuntimeError("dstack SDK bridge returned a non-object response")
        return cast(dict[str, Any], value)


def _run_name(key: str) -> str:
    return f"pt-{hashlib.sha256(key.encode()).hexdigest()[:24]}"


def _cleanup_name(provider_id: str, run_id: str) -> str:
    identity = f"{provider_id}\0{run_id}\0workspace-cleanup-v2"
    return f"pt-clean-{hashlib.sha256(identity.encode()).hexdigest()[:18]}"


def _sequence(value: Any) -> list[Any] | None:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return list(value)
    return None


def _integer(value: Any, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    return int(value)


class DstackExecutionProvider:
    def __init__(
        self,
        gateway: DstackGateway,
        *,
        project: str,
        trust_bundle: Path | None = None,
        capacity_wait_seconds: int = 0,
    ) -> None:
        if capacity_wait_seconds < 0:
            raise ValueError("dstack capacity wait must not be negative")
        self._gateway = gateway
        self._project = project
        self._capacity_wait_seconds = capacity_wait_seconds
        # This is an instance-side path. It is deliberately not resolved or
        # inspected on the submitting machine.
        self._trust_bundle = trust_bundle

    @classmethod
    def from_sdk_environment(
        cls,
        *,
        project: str,
        python: Path,
        environment_file: Path | None = None,
        runtime_environment: Mapping[str, str] | None = None,
        trust_bundle: Path | None = None,
        capacity_wait_seconds: int = 0,
    ) -> DstackExecutionProvider:
        return cls(
            DstackSdkBridge(
                python,
                environment_file=environment_file,
                runtime_environment=runtime_environment,
            ),
            project=project,
            trust_bundle=trust_bundle,
            capacity_wait_seconds=capacity_wait_seconds,
        )

    def _configuration(
        self,
        request: ExecutionRequest,
        *,
        allow_legacy_bundle: bool = False,
    ) -> dict[str, Any]:
        if request.bundle is not None and not allow_legacy_bundle:
            raise RuntimeError(
                "dstack no longer accepts execution bundles; pack and submit an immutable actual-job image"
            )
        placement = request.target.placement
        gpu: dict[str, Any] = {"count": _integer(placement.get("gpu_count"), 1)}
        if names := _sequence(placement.get("gpu_names")):
            gpu["name"] = [str(name) for name in names]
        if request.target.memory_gb:
            minimum_memory = int(request.target.memory_gb)
            maximum_memory_value = placement.get("gpu_memory_max_gb")
            if maximum_memory_value is None:
                gpu["memory"] = f"{minimum_memory}GB.."
            else:
                maximum_memory = _integer(maximum_memory_value, 0)
                if maximum_memory < minimum_memory:
                    raise ValueError(
                        "dstack maximum GPU memory must be greater than or equal to the execution target minimum"
                    )
                gpu["memory"] = f"{minimum_memory}GB..{maximum_memory}GB"
        launch_environment = request.launch_environment(provider="dstack")
        if self._trust_bundle is not None:
            launch_environment.update({_EXTRA_TRUST_VARIABLE: str(TRUST_BUNDLE_CONTAINER_PATH)})
        command = shlex.join(request.command)
        command = f"ulimit -n {POSTTRAIN_NOFILE_LIMIT} 2>/dev/null || true; exec {command}"
        configuration: dict[str, Any] = {
            "name": _run_name(request.idempotency_key),
            "image": request.image.value,
            "commands": [command],
            "working_dir": "/opt/posttrain/job",
            "env": list(request.environment_names),
            "_posttrain_launch_env": launch_environment,
            "resources": {
                "gpu": gpu,
                "disk": {"size": f"{_integer(placement.get('disk_gb'), 100)}GB.."},
            },
            "priority": request.policy.priority,
            # Capacity waiting happens before arbitrary user code starts and
            # does not create another framework execution attempt. Other
            # provider retry events stay disabled because they may repeat an
            # admitted training attempt after user code has run.
            "retry": (
                {
                    "on_events": ["no-capacity"],
                    "duration": self._capacity_wait_seconds,
                }
                if self._capacity_wait_seconds
                else False
            ),
            "max_duration": request.policy.timeout_seconds,
            "tags": {
                "posttrain_run_id": request.run_spec.run_id,
                "posttrain_attempt": str(request.attempt),
                "posttrain_job_image_digest": request.image.digest,
            },
        }
        if fleets := _sequence(placement.get("fleets")):
            configuration["fleets"] = fleets
        if instances := _sequence(placement.get("instances")):
            configuration["instances"] = instances
        if request.mounts:
            configuration["volumes"] = [
                {
                    "instance_path": str(mount.instance_path),
                    "path": str(mount.container_path),
                    "optional": mount.optional,
                }
                for mount in request.mounts
            ]
        if self._trust_bundle is not None:
            volumes = configuration.setdefault("volumes", [])
            volumes.append(
                {
                    "instance_path": str(self._trust_bundle),
                    "path": str(TRUST_BUNDLE_CONTAINER_PATH),
                    "optional": False,
                }
            )
            configuration["setup"] = [f"test -f {shlex.quote(str(TRUST_BUNDLE_CONTAINER_PATH))}"]
        return configuration

    def plan(self, request: ExecutionRequest) -> ExecutionPlan:
        configuration = self._configuration(request, allow_legacy_bundle=True)
        response = self._gateway.invoke(
            "plan",
            {"project": self._project, "configuration": configuration},
        )
        return ExecutionPlan(
            provider="dstack",
            request=request,
            native_plan_id=str(response.get("run_name") or configuration["name"]),
            details={
                "project": self._project,
                "run_name": str(response.get("run_name") or configuration["name"]),
                "offers": int(response.get("offers") or 0),
                "job_image": request.image.value,
                "submission_ready": request.bundle is None,
                "capacity_wait_seconds": self._capacity_wait_seconds,
            },
        )

    def submit(self, plan: ExecutionPlan) -> ExecutionHandle:
        response = self._gateway.invoke(
            "submit",
            {
                "project": self._project,
                "configuration": self._configuration(plan.request),
            },
        )
        return ExecutionHandle(
            "dstack",
            str(response["run_name"]),
            plan.request.idempotency_key,
        )

    def status(self, handle: ExecutionHandle) -> ExecutionRecord:
        response = self._gateway.invoke(
            "status",
            {"project": self._project, "run_name": handle.provider_id},
        )
        native = str(response["status"]).lower()
        return ExecutionRecord(
            handle=handle,
            state=_STATE.get(native, "lost"),
            attempt=int(response.get("attempt") or 1),
            target_id=str(response.get("hostname") or "unassigned"),
            observed_at=datetime.now(UTC),
            native_state=native,
            message=str(response["message"]) if response.get("message") else None,
        )

    def logs(
        self,
        handle: ExecutionHandle,
        cursor: LogCursor | None = None,
        *,
        limit: int = 200,
    ) -> LogPage:
        if limit < 1:
            raise ValueError("log limit must be positive")
        offset = (cursor or LogCursor()).offset
        response = self._gateway.invoke(
            "logs",
            {"project": self._project, "run_name": handle.provider_id},
        )
        all_lines = tuple(str(value) for value in response.get("lines", ()))
        page = all_lines[offset : offset + limit]
        next_offset = offset + len(page)
        return LogPage(page, LogCursor(next_offset), next_offset < len(all_lines))

    def cancel(self, handle: ExecutionHandle) -> None:
        self._gateway.invoke(
            "cancel",
            {"project": self._project, "run_name": handle.provider_id},
        )

    def collect(self, handle: ExecutionHandle) -> ExecutionResult:
        record = self.status(handle)
        if record.state not in {"succeeded", "failed", "cancelled", "lost"}:
            raise RuntimeError(f"dstack run is not terminal: {record.native_state}")
        return ExecutionResult(record, 0 if record.state == "succeeded" else None)

    def cleanup(
        self,
        handle: ExecutionHandle,
        *,
        run_id: str,
        run_workspace: Path | None,
        runtime_image: RuntimeImageRef,
        local_image: str | None = None,
    ) -> ProviderCleanupResult:
        if not _RUN_ID.fullmatch(run_id):
            raise RuntimeError("dstack cleanup run id is not path-safe")
        if (
            run_workspace is None
            or not run_workspace.is_absolute()
            or run_workspace.name != run_id
            or run_workspace.parent == Path("/")
            or ".." in run_workspace.parts
        ):
            raise RuntimeError("dstack cleanup workspace must be one exact absolute run directory")
        record = self.status(handle)
        if record.state not in {"succeeded", "failed", "cancelled"}:
            raise RuntimeError("dstack run must be terminal and provider-visible before cleanup")
        hostname = record.target_id
        if hostname != "unassigned" and not _HOSTNAME.fullmatch(hostname):
            raise RuntimeError("dstack cleanup requires the terminal run's observed worker hostname")
        response = self._gateway.invoke(
            "cleanup_workspace",
            {
                "project": self._project,
                "source_run_name": handle.provider_id,
                "cleanup_run_name": _cleanup_name(handle.provider_id, run_id),
                "hostname": None if hostname == "unassigned" else hostname,
                "run_id": run_id,
                "workspace": str(run_workspace),
                "image": runtime_image.value,
            },
        )
        if (
            response.get("hostname") == hostname
            and response.get("workspace") == str(run_workspace)
            and response.get("workspace_state") == "deferred"
            and response.get("emptied") is False
            and response.get("reclaimed_bytes") == 0
        ):
            cleanup_run_name = response.get("cleanup_run_name")
            if not isinstance(cleanup_run_name, str) or not cleanup_run_name:
                raise RuntimeError("dstack deferred cleanup did not identify its provider task")
            cleanup_status = str(response.get("cleanup_status") or "pending")
            raise ProviderCleanupDeferred(
                f"exact-worker cleanup task {cleanup_run_name!r} is {cleanup_status}; "
                "retry the same immutable purge after dstack capacity is available"
            )
        if (
            hostname == "unassigned"
            and response.get("hostname") is None
            and response.get("workspace") == str(run_workspace)
            and response.get("workspace_state") == "not-created"
            and response.get("emptied") is False
            and response.get("reclaimed_bytes") == 0
        ):
            return ProviderCleanupResult(
                handle,
                "provider-managed",
                (
                    "dstack retained terminal run history; provider-native "
                    "assignment history proves no worker workspace was created"
                ),
                workspace_disposition="not-created",
                workspace_reclaimed_bytes=0,
            )
        if (
            response.get("hostname") != hostname
            or response.get("workspace") != str(run_workspace)
            or response.get("emptied") is not True
        ):
            raise RuntimeError("dstack cleanup task did not verify the exact worker workspace")
        reclaimed = response.get("reclaimed_bytes")
        if isinstance(reclaimed, bool) or not isinstance(reclaimed, int) or reclaimed < 0:
            raise RuntimeError("dstack cleanup task returned invalid reclaimed bytes")
        return ProviderCleanupResult(
            handle,
            "provider-managed",
            (f"dstack retained run history and emptied the exact run workspace on {hostname}"),
            workspace_disposition="removed",
            workspace_reclaimed_bytes=reclaimed,
        )
