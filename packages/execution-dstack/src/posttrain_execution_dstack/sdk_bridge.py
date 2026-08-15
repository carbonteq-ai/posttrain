"""Standalone JSON bridge to dstack's Python SDK.

This file intentionally imports no framework package so it can run in dstack's
Pydantic-1 environment while the caller remains in the framework's Pydantic-2
environment.
"""

from __future__ import annotations

import json
import re
import sys
import time

from dstack.api import Client, Task, VirtualRepo
from native_state import assignment_state

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TERMINAL = frozenset({"terminated", "failed", "done"})
_RECLAIMED_PREFIX = "POSTTRAIN_CLEANUP_RECLAIMED_BYTES="
_CLEANUP_CAPACITY_WAIT_SECONDS = 86_400


def _client(payload):
    return Client.from_config(project_name=payload["project"])


def _configuration(payload):
    configuration = dict(payload["configuration"])
    launch_environment = configuration.pop("_posttrain_launch_env", {})
    runtime_environment = configuration.pop("_posttrain_runtime_env", {})
    if not isinstance(launch_environment, dict):
        raise RuntimeError("invalid posttrain launch environment")
    if not isinstance(runtime_environment, dict) or not all(
        isinstance(name, str) and isinstance(value, str) for name, value in runtime_environment.items()
    ):
        raise RuntimeError("invalid posttrain runtime environment")
    environment = configuration.get("env")
    if isinstance(environment, list):
        if not all(isinstance(name, str) for name in environment):
            raise RuntimeError("invalid dstack execution environment")
        missing = [name for name in environment if name not in runtime_environment]
        if missing:
            raise RuntimeError("required execution environment names are unavailable from posttrain.env")
        configuration["env"] = {name: runtime_environment[name] for name in environment}
    if launch_environment:
        configured_environment = configuration.setdefault("env", {})
        if not isinstance(configured_environment, dict):
            raise RuntimeError("invalid dstack execution environment")
        configured_environment.update({str(name): str(value) for name, value in launch_environment.items()})
    return Task(**configuration)


def plan(payload):
    configuration = _configuration(payload)
    native = _client(payload).runs.get_run_plan(
        configuration=configuration,
        repo=VirtualRepo(),
    )
    offers = sum(len(job.offers) for job in native.job_plans)
    return {"run_name": configuration.name, "offers": offers}


def submit(payload):
    client = _client(payload)
    configuration = _configuration(payload)
    run = client.runs.get(configuration.name)
    if run is None:
        native = client.runs.get_run_plan(
            configuration=configuration,
            repo=VirtualRepo(),
        )
        run = client.runs.apply_plan(
            run_plan=native,
            repo=VirtualRepo(),
            reserve_ports=False,
        )
    return {"run_name": run.name}


def status(payload):
    run = _client(payload).runs.get(payload["run_name"])
    if run is None:
        return {"status": "lost", "hostname": None, "attempt": 1}
    run.refresh()
    native = str(getattr(run.status, "value", run.status)).lower()
    try:
        hostname = run.hostname
    except (AttributeError, RuntimeError, ValueError):
        hostname = None
    message = (
        getattr(run._run, "error", None)
        or getattr(run._run, "status_message", None)
        or getattr(run._run, "termination_reason", None)
    )
    return {
        "status": native,
        "hostname": hostname,
        "attempt": 1,
        "message": str(message)[:500] if message else None,
    }


def logs(payload):
    run = _client(payload).runs.get(payload["run_name"])
    if run is None:
        return {"lines": []}
    lines = []
    for value in run.logs(replica_num=0, job_num=0, diagnose=bool(payload.get("diagnose", False))):
        lines.extend(value.decode("utf-8", errors="replace").splitlines())
    return {"lines": lines}


def cancel(payload):
    run = _client(payload).runs.get(payload["run_name"])
    if run is None:
        return {"cancelled": False, "missing": True}
    native = str(getattr(run.status, "value", run.status)).lower()
    if native not in {"terminated", "failed", "done"}:
        run.stop(abort=False)
        return {"cancelled": True, "missing": False}
    return {"cancelled": False, "missing": False}


def _native_status(run):
    run.refresh()
    return str(getattr(run.status, "value", run.status)).lower()


def _cleanup_command():
    cleanup_path = "/opt/posttrain/cleanup"
    return "\n".join(
        (
            "set -eu",
            (
                "before=$(find " + cleanup_path + " -mindepth 1 -printf '%s\\n' "
                "| awk '{total += $1} END {print total + 0}')"
            ),
            ("find " + cleanup_path + " -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +"),
            ('test -z "$(find ' + cleanup_path + ' -mindepth 1 -print -quit)"'),
            'printf "' + _RECLAIMED_PREFIX + '%s\\n" "$before"',
        )
    )


def _apply_cleanup_when_worker_is_available(client, base_configuration):
    """Submit or resume one exact-worker cleanup task.

    A terminal training run can be followed immediately by another job on the
    same single-slot worker. dstack then reports no offer even though the
    worker is healthy and will become idle later. Cleanup is resumable
    control-plane work, so persist a provider-native no-capacity retry instead
    of holding the caller open or turning this normal race into a failed purge.
    """

    cleanup_name = str(base_configuration["name"])
    cleanup_run = client.runs.get(cleanup_name)
    if cleanup_run is not None and _native_status(cleanup_run) in {"failed", "terminated"}:
        # Keep failed tasks as diagnostic history. Resume the first existing
        # non-failed retry before allocating another deterministic name; a
        # polling purge must never multiply queued exact-worker tasks.
        for retry in range(1, 100):
            retry_name = f"{cleanup_name}-retry-{retry}"
            retry_run = client.runs.get(retry_name)
            if retry_run is None:
                base_configuration = {**base_configuration, "name": retry_name}
                cleanup_run = None
                break
            if _native_status(retry_run) not in {"failed", "terminated"}:
                return retry_run
        else:
            raise RuntimeError("dstack cleanup task retry names are exhausted")
    if cleanup_run is not None:
        return cleanup_run

    deferred_plan = None
    for gpu_count in (0, 1):
        configuration = Task(
            **base_configuration,
            resources={"gpu": {"count": gpu_count}, "disk": {"size": "100GB.."}},
        )
        plan = client.runs.get_run_plan(
            configuration=configuration,
            repo=VirtualRepo(),
        )
        if deferred_plan is None:
            deferred_plan = plan
        if any(job.offers for job in plan.job_plans):
            return client.runs.apply_plan(
                run_plan=plan,
                repo=VirtualRepo(),
                reserve_ports=False,
            )
    # With retry.on_events=no-capacity, applying the first zero-offer plan is
    # intentional: dstack durably holds the exact-worker task until capacity is
    # available. A later purge apply observes this same deterministic task.
    if deferred_plan is None:
        raise RuntimeError("exact-worker cleanup planning returned no resource shapes")
    return client.runs.apply_plan(
        run_plan=deferred_plan,
        repo=VirtualRepo(),
        reserve_ports=False,
    )


def cleanup_workspace(payload):
    client = _client(payload)
    source = client.runs.get(payload["source_run_name"])
    if source is None:
        raise RuntimeError("cleanup source run is unavailable")
    native = _native_status(source)
    if native not in _TERMINAL:
        raise RuntimeError("cleanup source run is not terminal")

    run_id = str(payload["run_id"])
    workspace = str(payload["workspace"])
    if not _RUN_ID.fullmatch(run_id):
        raise RuntimeError("cleanup run id is not path-safe")
    if not workspace.startswith("/") or workspace.rstrip("/").split("/")[-1] != run_id:
        raise RuntimeError("cleanup workspace is not the exact run directory")
    if "/../" in workspace or workspace.endswith("/.."):
        raise RuntimeError("cleanup workspace contains parent traversal")

    expected_hostname = payload.get("hostname")
    try:
        observed_hostname = source.hostname
    except (AttributeError, RuntimeError, ValueError):
        observed_hostname = None
    if not observed_hostname:
        if native not in {"failed", "terminated"} or assignment_state(source._run) != "never-assigned":
            raise RuntimeError(
                "cleanup source run has no worker hostname but assignment history is not conclusively empty"
            )
        return {
            "cleanup_run_name": None,
            "hostname": None,
            "workspace": workspace,
            "workspace_state": "not-created",
            "emptied": False,
            "reclaimed_bytes": 0,
        }
    if not isinstance(expected_hostname, str) or observed_hostname != expected_hostname:
        raise RuntimeError("cleanup source run worker does not match")

    base_configuration = {
        "name": str(payload["cleanup_run_name"]),
        "image": str(payload["image"]),
        "commands": [_cleanup_command()],
        "instances": [{"hostname": expected_hostname}],
        "volumes": [
            {
                "instance_path": workspace,
                "path": "/opt/posttrain/cleanup",
                "optional": False,
            }
        ],
        "retry": {
            "on_events": ["no-capacity"],
            "duration": _CLEANUP_CAPACITY_WAIT_SECONDS,
        },
        "max_duration": 300,
        "tags": {
            "posttrain_cleanup_run_id": run_id,
            "posttrain_cleanup_source_run": str(payload["source_run_name"]),
        },
    }
    cleanup_run = _apply_cleanup_when_worker_is_available(client, base_configuration)

    deadline = time.monotonic() + 300
    cleanup_status = _native_status(cleanup_run)
    if cleanup_status in {"pending", "submitted", "provisioning"}:
        return {
            "cleanup_run_name": cleanup_run.name,
            "cleanup_status": cleanup_status,
            "hostname": expected_hostname,
            "workspace": workspace,
            "workspace_state": "deferred",
            "emptied": False,
            "reclaimed_bytes": 0,
        }
    while cleanup_status not in _TERMINAL and time.monotonic() < deadline:
        time.sleep(2)
        cleanup_status = _native_status(cleanup_run)
    if cleanup_status != "done":
        raise RuntimeError(f"exact-worker cleanup task did not succeed (status={cleanup_status})")

    reclaimed = None
    for value in cleanup_run.logs(replica_num=0, job_num=0):
        for line in value.decode("utf-8", errors="replace").splitlines():
            if line.startswith(_RECLAIMED_PREFIX):
                raw = line.removeprefix(_RECLAIMED_PREFIX)
                if raw.isdigit():
                    reclaimed = int(raw)
    if reclaimed is None:
        raise RuntimeError("cleanup task did not return verification evidence")
    return {
        "cleanup_run_name": cleanup_run.name,
        "hostname": expected_hostname,
        "workspace": workspace,
        "emptied": True,
        "reclaimed_bytes": reclaimed,
    }


def main():
    action = sys.argv[1]
    payload = json.load(sys.stdin)
    handlers = {
        "plan": plan,
        "submit": submit,
        "status": status,
        "logs": logs,
        "cancel": cancel,
        "cleanup_workspace": cleanup_workspace,
    }
    result = handlers[action](payload)
    json.dump(result, sys.stdout, sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    main()
