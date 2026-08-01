from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from posttrain.common import ExecutionTarget
from posttrain.execution import (
    JOB_PACKAGE_WORKER_COMMAND,
    BundleRef,
    ExecutionMount,
    ExecutionPolicy,
    ExecutionRequest,
    RuntimeImageRef,
)
from posttrain.tracking import RunSpec
from posttrain_execution_local import DockerCli, LocalDockerExecutionProvider


class FakeDocker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.status = "running"
        self.exit_code = 0
        self.exists = False
        self.labels: dict[str, str] = {}

    def invoke(self, action: str, payload):
        self.calls.append((action, dict(payload)))
        if action == "exists":
            return {"exists": self.exists}
        if action == "identity":
            return {"exists": self.exists, "labels": self.labels}
        if action == "pull":
            return {"repo_digests": [payload["image"]]}
        if action == "submit":
            self.exists = True
            self.labels = dict(payload["labels"])
            return {"container_id": "container-id"}
        if action == "inspect":
            return {"status": self.status, "exit_code": self.exit_code, "error": ""}
        if action == "logs":
            return {"lines": ["one", "two"]}
        if action == "cancel":
            self.status = "exited"
            self.exit_code = 137
            return {"cancelled": True}
        if action == "cleanup":
            return {"removed": True}
        if action == "cleanup_workspace":
            return {"emptied": True}
        raise AssertionError(action)


def test_docker_cli_uses_packaged_workdir_and_explicit_worker_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def run(arguments, **kwargs):
        del kwargs
        calls.append(list(arguments))
        return subprocess.CompletedProcess(arguments, 0, "container-id\n", "")

    monkeypatch.setattr(
        "posttrain_execution_local.adapter.subprocess.run",
        run,
    )
    DockerCli(environment={"TRACKIO_SERVER_URL": "https://trackio.example"}).invoke(
        "submit",
        {
            "name": "pt-test",
            "image": f"registry.lan/posttrain@sha256:{'b' * 64}",
            "gpu": False,
            "environment_names": ["TRACKIO_SERVER_URL"],
            "launch_environment": {"POSTTRAIN_EXECUTION": '{"schema":"test"}'},
            "volumes": [],
            "dns_servers": ["192.0.2.53", "2001:db8::53"],
            "labels": {},
            "command": list(JOB_PACKAGE_WORKER_COMMAND),
        },
    )

    arguments = calls[0]
    assert arguments[:4] == ["docker", "run", "--detach", "--name"]
    assert "--workdir" in arguments
    assert arguments[arguments.index("--workdir") + 1] == "/opt/posttrain/job"
    assert arguments[arguments.index("--entrypoint") + 1] == "posttrain-runtime"
    assert "TRACKIO_SERVER_URL" in arguments
    assert arguments.count("--dns") == 2
    assert arguments[arguments.index("--dns") + 1] == "192.0.2.53"
    assert 'POSTTRAIN_EXECUTION={"schema":"test"}' in arguments
    assert "/opt/posttrain/bundle" not in arguments


def _request(tmp_path: Path) -> ExecutionRequest:
    return ExecutionRequest(
        run_spec=RunSpec(
            project_id="tests",
            work_package_id="qualify/local",
            stage="qualify",
            job_kind="eval.smoke",
            job_definition_version="eval/smoke@1",
        ),
        job_definition_id="eval/smoke@1",
        image=RuntimeImageRef(f"registry.lan/posttrain@sha256:{'b' * 64}"),
        target=ExecutionTarget(
            "targets/local",
            "1",
            "nvidia-cuda",
            20,
            placement={},
        ),
        command=JOB_PACKAGE_WORKER_COMMAND,
        idempotency_key="local-attempt-1",
        policy=ExecutionPolicy(300),
        environment_names=("TRACKIO_SERVER_URL",),
        mounts=(
            ExecutionMount(
                (tmp_path / "cache").resolve(),
                Path("/root/.cache/huggingface"),
                "model-cache",
            ),
        ),
    )


def test_local_docker_lifecycle_and_cancel_are_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRACKIO_SERVER_URL", "https://trackio.example")
    gateway = FakeDocker()
    provider = LocalDockerExecutionProvider(
        gateway,
        state_root=(tmp_path / "state").resolve(),
        dns_servers=("192.0.2.53",),
    )
    plan = provider.plan(_request(tmp_path))
    handle = provider.submit(plan)
    submit = next(payload for action, payload in gateway.calls if action == "submit")
    assert all(mount.instance_path.is_dir() for mount in plan.request.mounts)
    assert submit["environment_names"] == ["TRACKIO_SERVER_URL"]
    launch = json.loads(submit["launch_environment"]["POSTTRAIN_EXECUTION"])
    assert launch["schema"] == "posttrain.execution-launch.v1"
    assert launch["run"]["run_id"] == plan.request.run_spec.run_id
    assert launch["provider"] == "local-docker"
    assert launch["attempt"] == 1
    assert launch["job_image"] == plan.request.image.value
    assert launch["target"]["id"] == plan.request.target.id
    assert submit["gpu"] is True
    assert submit["dns_servers"] == ["192.0.2.53"]
    assert all("trackio.example" not in str(payload) for _, payload in gateway.calls)
    assert submit["command"] == [
        "posttrain-runtime",
        "execute",
        "--manifest",
        "/opt/posttrain/job/package.json",
    ]
    assert all("/opt/posttrain/bundle" not in str(payload) for _, payload in gateway.calls)
    assert submit["labels"]["posttrain.attempt"] == "1"
    assert submit["labels"]["posttrain.job_image_digest"] == "b" * 64
    assert provider.status(handle).state == "running"
    provider.cancel(handle)
    assert provider.status(handle).state == "cancelled"
    assert provider.collect(handle).exit_code == 137
    workspace = tmp_path / "test-run"
    workspace.mkdir()
    cleanup = provider.cleanup(
        handle,
        run_id="test-run",
        run_workspace=workspace,
        runtime_image=plan.request.image,
    )
    assert cleanup.disposition == "removed"
    assert ("cleanup", {"name": handle.provider_id}) in gateway.calls
    assert (
        "cleanup_workspace",
        {
            "workspace": str(workspace),
            "image": plan.request.image.value,
        },
    ) in gateway.calls


def test_local_docker_mounts_trust_bundle_as_additional_authorities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRACKIO_SERVER_URL", "https://trackio.example")
    trust_bundle = (tmp_path / "internal-ca.pem").resolve()
    trust_bundle.write_text("test certificate bundle\n", encoding="utf-8")
    gateway = FakeDocker()
    provider = LocalDockerExecutionProvider(
        gateway,
        state_root=(tmp_path / "state").resolve(),
        trust_bundle=trust_bundle,
    )
    provider.submit(provider.plan(_request(tmp_path)))

    submit = next(payload for action, payload in gateway.calls if action == "submit")
    stable_path = "/opt/posttrain/trust/ca-certificates.crt"
    assert f"{trust_bundle}:{stable_path}:ro" in submit["volumes"]
    launch = submit["launch_environment"]
    assert launch["POSTTRAIN_EXTRA_CA_BUNDLE"] == stable_path
    # The image merges this with the authorities it already trusts. Setting
    # SSL_CERT_FILE here would replace that set instead, so a job that gained
    # an internal registry would lose every public authority with it, and fail
    # much later verifying something unrelated.
    assert "SSL_CERT_FILE" not in launch
    assert "REQUESTS_CA_BUNDLE" not in launch
    assert "test certificate bundle" not in str(submit)


def test_local_docker_submit_is_idempotent_only_for_matching_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRACKIO_SERVER_URL", "https://trackio.example")
    gateway = FakeDocker()
    provider = LocalDockerExecutionProvider(
        gateway,
        state_root=(tmp_path / "state").resolve(),
    )
    plan = provider.plan(_request(tmp_path))

    first = provider.submit(plan)
    second = provider.submit(plan)

    assert second == first
    assert len([call for call in gateway.calls if call[0] == "submit"]) == 1

    gateway.labels["posttrain.run_id"] = "conflicting-run"
    with pytest.raises(RuntimeError, match="conflicts with the idempotent"):
        provider.submit(plan)


def test_local_docker_fails_closed_when_trust_bundle_disappears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRACKIO_SERVER_URL", "https://trackio.example")
    trust_bundle = (tmp_path / "internal-ca.pem").resolve()
    trust_bundle.write_text("test certificate bundle\n", encoding="utf-8")
    provider = LocalDockerExecutionProvider(
        FakeDocker(),
        state_root=(tmp_path / "state").resolve(),
        trust_bundle=trust_bundle,
    )
    trust_bundle.unlink()

    with pytest.raises(RuntimeError, match="trust bundle is missing"):
        provider.submit(provider.plan(_request(tmp_path)))


def test_local_docker_requires_named_environment_to_exist(tmp_path: Path) -> None:
    provider = LocalDockerExecutionProvider(
        FakeDocker(),
        state_root=(tmp_path / "state").resolve(),
    )
    with pytest.raises(RuntimeError, match="missing environment"):
        provider.submit(provider.plan(_request(tmp_path)))


def test_local_docker_legacy_bundle_is_plan_only(tmp_path: Path) -> None:
    provider = LocalDockerExecutionProvider(
        FakeDocker(),
        state_root=(tmp_path / "state").resolve(),
    )
    request = replace(
        _request(tmp_path),
        bundle=BundleRef((tmp_path / "legacy-bundle").resolve(), "c" * 64),
    )

    plan = provider.plan(request)

    assert plan.details["submission_ready"] is False
    with pytest.raises(RuntimeError, match="no longer accepts execution bundles"):
        provider.submit(plan)
