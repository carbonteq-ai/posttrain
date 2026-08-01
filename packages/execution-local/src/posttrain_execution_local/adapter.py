"""Local Docker lifecycle adapter for immutable actual-job images."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from posttrain.execution import (
    ExecutionHandle,
    ExecutionPlan,
    ExecutionRecord,
    ExecutionRequest,
    ExecutionResult,
    LogCursor,
    LogPage,
    ProviderCleanupResult,
    RuntimeImageRef,
)

TRUST_BUNDLE_CONTAINER_PATH = Path("/opt/posttrain/trust/ca-certificates.crt")
# The job image merges this with the authorities it already trusts. Setting
# SSL_CERT_FILE here instead would replace that set rather than extend it,
# leaving an internally-trusting job unable to verify anything public.
_EXTRA_TRUST_VARIABLE = "POSTTRAIN_EXTRA_CA_BUNDLE"


class DockerGateway(Protocol):
    def invoke(self, action: str, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


class DockerCli:
    """Small JSON-oriented gateway around the non-shell Docker CLI."""

    def __init__(
        self,
        executable: str = "docker",
        *,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self._executable = executable
        self._environment = dict(os.environ if environment is None else environment)

    def _run(self, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [self._executable, *arguments],
            text=True,
            capture_output=True,
            check=False,
            env=self._environment,
        )
        if check and result.returncode != 0:
            detail = result.stderr.strip().splitlines()[-1:] or ["no diagnostic was returned"]
            raise RuntimeError(f"docker {' '.join(arguments[:2])} failed: {detail[0][:500]}")
        return result

    def invoke(self, action: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        name = str(payload.get("name", ""))
        if action == "exists":
            result = self._run("container", "inspect", name, check=False)
            return {"exists": result.returncode == 0}
        if action == "identity":
            result = self._run(
                "container",
                "inspect",
                name,
                "--format",
                "{{json .Config.Labels}}",
                check=False,
            )
            if result.returncode != 0:
                return {"exists": False, "labels": {}}
            return {"exists": True, "labels": json.loads(result.stdout)}
        if action == "pull":
            image = str(payload["image"])
            self._run("pull", image)
            inspected = self._run(
                "image",
                "inspect",
                image,
                "--format",
                "{{json .RepoDigests}}",
            )
            return {"repo_digests": json.loads(inspected.stdout)}
        if action == "submit":
            arguments = ["run", "--detach", "--name", name]
            for key, value in cast_mapping(payload.get("labels")).items():
                arguments.extend(("--label", f"{key}={value}"))
            for environment_name in cast_sequence(payload.get("environment_names")):
                arguments.extend(("--env", str(environment_name)))
            for key, value in cast_mapping(payload.get("launch_environment")).items():
                arguments.extend(("--env", f"{key}={value}"))
            for volume in cast_sequence(payload.get("volumes")):
                arguments.extend(("--volume", str(volume)))
            for dns_server in cast_sequence(payload.get("dns_servers")):
                arguments.extend(("--dns", str(dns_server)))
            if bool(payload.get("gpu")):
                arguments.extend(("--gpus", "all"))
            command = tuple(str(value) for value in cast_sequence(payload.get("command")))
            if not command:
                raise ValueError("local Docker submission command cannot be empty")
            arguments.extend(
                (
                    "--workdir",
                    "/opt/posttrain/job",
                    "--entrypoint",
                    command[0],
                    str(payload["image"]),
                    *command[1:],
                )
            )
            result = self._run(*arguments)
            return {"container_id": result.stdout.strip()}
        if action == "inspect":
            result = self._run(
                "container",
                "inspect",
                name,
                "--format",
                "{{json .State}}",
                check=False,
            )
            if result.returncode != 0:
                return {"status": "missing", "exit_code": None}
            state = json.loads(result.stdout)
            return {
                "status": str(state["Status"]),
                "exit_code": int(state["ExitCode"]),
                "error": str(state.get("Error") or ""),
            }
        if action == "logs":
            result = self._run("logs", name, check=False)
            return {"lines": (result.stdout + result.stderr).splitlines()}
        if action == "cancel":
            self._run("stop", "--time", "10", name)
            return {"cancelled": True}
        if action == "cleanup":
            self._run("container", "rm", name)
            return {"removed": True}
        if action == "cleanup_workspace":
            self._run(
                "run",
                "--rm",
                "--entrypoint",
                "/bin/sh",
                "--volume",
                f"{payload['workspace']}:/opt/posttrain/cleanup",
                str(payload["image"]),
                "-c",
                ("find /opt/posttrain/cleanup -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +"),
            )
            return {"emptied": True}
        raise ValueError(f"unsupported Docker gateway action: {action}")


def cast_sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def cast_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _container_name(idempotency_key: str) -> str:
    return f"pt-{hashlib.sha256(idempotency_key.encode()).hexdigest()[:24]}"


class LocalDockerExecutionProvider:
    def __init__(
        self,
        gateway: DockerGateway | None = None,
        *,
        state_root: Path,
        environment: Mapping[str, str] | None = None,
        dns_servers: Sequence[str] = (),
        trust_bundle: Path | None = None,
    ) -> None:
        self._environment = dict(os.environ)
        if environment is not None:
            self._environment.update(environment)
        self._gateway = gateway or DockerCli(environment=self._environment)
        self._state_root = state_root.resolve()
        self._dns_servers = tuple(dns_servers)
        self._trust_bundle = trust_bundle.expanduser().resolve() if trust_bundle is not None else None

    def _cancel_marker(self, provider_id: str) -> Path:
        return self._state_root / "cancelled" / provider_id

    def _volumes(self, request: ExecutionRequest) -> list[str]:
        volumes = [f"{mount.instance_path}:{mount.container_path}" for mount in request.mounts]
        if self._trust_bundle is not None:
            volumes.append(f"{self._trust_bundle}:{TRUST_BUNDLE_CONTAINER_PATH}:ro")
        return volumes

    def _launch_environment(self, request: ExecutionRequest) -> dict[str, str]:
        environment = request.launch_environment(provider="local-docker")
        if self._trust_bundle is not None:
            environment.update({_EXTRA_TRUST_VARIABLE: str(TRUST_BUNDLE_CONTAINER_PATH)})
        return environment

    def _payload(self, request: ExecutionRequest) -> dict[str, Any]:
        if request.bundle is not None:
            raise RuntimeError(
                "local Docker no longer accepts execution bundles; pack and submit an immutable actual-job image"
            )
        return {
            "name": _container_name(request.idempotency_key),
            "image": request.image.value,
            "gpu": request.target.device_class in {"cuda", "nvidia-cuda"},
            "environment_names": list(request.environment_names),
            "launch_environment": self._launch_environment(request),
            "volumes": self._volumes(request),
            "dns_servers": list(self._dns_servers),
            "labels": {
                "posttrain.run_id": request.run_spec.run_id,
                "posttrain.attempt": str(request.attempt),
                "posttrain.job_image_digest": request.image.digest,
            },
            "command": list(request.command),
        }

    def plan(self, request: ExecutionRequest) -> ExecutionPlan:
        name = _container_name(request.idempotency_key)
        return ExecutionPlan(
            provider="local-docker",
            request=request,
            native_plan_id=name,
            details={
                "container_name": name,
                "submission_ready": request.bundle is None,
            },
        )

    def submit(self, plan: ExecutionPlan) -> ExecutionHandle:
        request = plan.request
        if request.bundle is not None:
            raise RuntimeError(
                "local Docker no longer accepts execution bundles; pack and submit an immutable actual-job image"
            )
        missing = [name for name in request.environment_names if name not in self._environment]
        if missing:
            raise RuntimeError(f"local Docker execution is missing environment names: {', '.join(missing)}")
        if self._trust_bundle is not None and not self._trust_bundle.is_file():
            raise RuntimeError(f"local execution trust bundle is missing: {self._trust_bundle}")
        for mount in request.mounts:
            if mount.instance_path.exists() and not mount.instance_path.is_dir():
                raise RuntimeError(f"local execution mount is not a directory: {mount.instance_path}")
            mount.instance_path.mkdir(parents=True, exist_ok=True)
        payload = self._payload(request)
        name = str(payload["name"])
        if not bool(self._gateway.invoke("exists", {"name": name}).get("exists")):
            pulled = self._gateway.invoke("pull", {"image": request.image.value})
            repo_digests = [str(value) for value in cast_sequence(pulled.get("repo_digests"))]
            if request.image.value not in repo_digests:
                raise RuntimeError("Docker did not resolve the requested immutable image digest")
            self._gateway.invoke("submit", payload)
        else:
            identity = self._gateway.invoke("identity", {"name": name})
            expected_labels = cast_mapping(payload["labels"])
            actual_labels = cast_mapping(identity.get("labels"))
            if any(actual_labels.get(key) != value for key, value in expected_labels.items()):
                raise RuntimeError(
                    f"existing Docker execution conflicts with the idempotent submission identity: {name}"
                )
        return ExecutionHandle("local-docker", name, request.idempotency_key)

    def status(self, handle: ExecutionHandle) -> ExecutionRecord:
        response = self._gateway.invoke("inspect", {"name": handle.provider_id})
        native = str(response["status"])
        exit_code = response.get("exit_code")
        if native in {"created", "restarting"}:
            state = "starting"
        elif native in {"running", "paused"}:
            state = "running"
        elif native == "exited":
            state = (
                "cancelled"
                if self._cancel_marker(handle.provider_id).exists()
                else ("succeeded" if exit_code == 0 else "failed")
            )
        elif native in {"dead", "removing"}:
            state = "failed"
        else:
            state = "lost"
        return ExecutionRecord(
            handle=handle,
            state=state,
            attempt=1,
            target_id="localhost",
            observed_at=datetime.now(UTC),
            native_state=native,
            message=str(response.get("error") or "") or None,
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
        response = self._gateway.invoke("logs", {"name": handle.provider_id})
        lines = tuple(str(value) for value in cast_sequence(response.get("lines")))
        page = lines[offset : offset + limit]
        next_offset = offset + len(page)
        return LogPage(page, LogCursor(next_offset), next_offset < len(lines))

    def cancel(self, handle: ExecutionHandle) -> None:
        marker = self._cancel_marker(handle.provider_id)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch(mode=0o600, exist_ok=True)
        self._gateway.invoke("cancel", {"name": handle.provider_id})

    def collect(self, handle: ExecutionHandle) -> ExecutionResult:
        record = self.status(handle)
        if record.state not in {"succeeded", "failed", "cancelled", "lost"}:
            raise RuntimeError(f"local Docker run is not terminal: {record.native_state}")
        response = self._gateway.invoke("inspect", {"name": handle.provider_id})
        exit_code = response.get("exit_code")
        return ExecutionResult(
            record,
            int(exit_code) if isinstance(exit_code, int) else None,
        )

    def cleanup(
        self,
        handle: ExecutionHandle,
        *,
        run_id: str,
        run_workspace: Path | None,
        runtime_image: RuntimeImageRef,
    ) -> ProviderCleanupResult:
        record = self.status(handle)
        container_disposition = "already-absent"
        if record.native_state == "missing":
            self._cancel_marker(handle.provider_id).unlink(missing_ok=True)
        elif record.state not in {"succeeded", "failed", "cancelled", "lost"}:
            raise RuntimeError("local Docker run must be terminal before cleanup")
        else:
            self._gateway.invoke("cleanup", {"name": handle.provider_id})
            container_disposition = "removed"
        self._cancel_marker(handle.provider_id).unlink(missing_ok=True)
        if run_workspace is not None and run_workspace.exists():
            if (
                not run_workspace.is_absolute()
                or run_workspace.name != run_id
                or run_workspace.is_symlink()
                or not run_workspace.is_dir()
            ):
                raise RuntimeError("local Docker cleanup workspace is not an exact run directory")
            self._gateway.invoke(
                "cleanup_workspace",
                {
                    "workspace": str(run_workspace),
                    "image": runtime_image.value,
                },
            )
        return ProviderCleanupResult(
            handle,
            container_disposition,
            ("released the exact local Docker container and emptied its run-scoped workspace"),
        )
