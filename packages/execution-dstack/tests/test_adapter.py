from __future__ import annotations

import importlib.util
import json
import sys
import types
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
    LogCursor,
    RuntimeImageRef,
)
from posttrain.tracking import RunSpec
from posttrain_execution_dstack import DstackExecutionProvider, DstackSdkBridge
from posttrain_execution_dstack.native_state import assignment_state


class FakeGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.status = "pending"
        self.cleanup_response: dict = {
            "hostname": "gpu-worker-a",
            "workspace": "/var/lib/posttrain/runs/test-run",
            "emptied": True,
            "reclaimed_bytes": 41,
        }

    def invoke(self, action: str, payload):
        self.calls.append((action, dict(payload)))
        configuration = payload.get("configuration", {})
        if action == "plan":
            return {"run_name": configuration["name"], "offers": 2}
        if action == "submit":
            return {"run_name": configuration["name"]}
        if action == "status":
            return {
                "status": self.status,
                "hostname": "gpu-worker-a" if self.status == "done" else None,
                "attempt": 1,
            }
        if action == "logs":
            return {"lines": ["one", "two", "three"]}
        if action == "cancel":
            self.status = "terminated"
            return {"cancelled": True}
        if action == "cleanup_workspace":
            return self.cleanup_response
        raise AssertionError(action)


def _request(tmp_path: Path) -> ExecutionRequest:
    return ExecutionRequest(
        run_spec=RunSpec(
            project_id="tests",
            work_package_id="train/test",
            stage="train",
            run_id="test-run",
            job_kind="train.grpo",
            job_definition_version="train/grpo@1",
        ),
        job_definition_id="train/grpo@1",
        image=RuntimeImageRef(f"registry.lan/posttrain@sha256:{'b' * 64}"),
        target=ExecutionTarget(
            "targets/remote",
            "1",
            "cuda",
            24,
            placement={
                "fleets": ["local-gpu-workers"],
                "instances": [{"hostname": "remote.lan"}],
                "gpu_count": 1,
                "gpu_memory_max_gb": 30,
                "disk_gb": 120,
            },
        ),
        command=JOB_PACKAGE_WORKER_COMMAND,
        idempotency_key="run-1-attempt-1",
        policy=ExecutionPolicy(900, max_attempts=2),
        environment_names=("TRACKIO_URL", "TRACKIO_WRITE_TOKEN"),
        mounts=(
            ExecutionMount(
                Path("/var/lib/posttrain/cache/huggingface"),
                Path("/root/.cache/huggingface"),
                "model-cache",
            ),
            ExecutionMount(
                Path("/var/lib/posttrain/runs/test-run"),
                Path("/opt/posttrain/run"),
                "run-workspace",
            ),
        ),
    )


def test_translation_and_submit_have_no_secret_values(tmp_path: Path) -> None:
    gateway = FakeGateway()
    provider = DstackExecutionProvider(gateway, project="posttrain")
    plan = provider.plan(_request(tmp_path))
    handle = provider.submit(plan)
    assert plan.details["offers"] == 2
    assert handle.provider_id == plan.native_plan_id
    configurations = [
        (action, payload["configuration"]) for action, payload in gateway.calls if action in {"plan", "submit"}
    ]
    assert all(config["env"] == ["TRACKIO_URL", "TRACKIO_WRITE_TOKEN"] for _, config in configurations)
    launch = json.loads(configurations[0][1]["_posttrain_launch_env"]["POSTTRAIN_EXECUTION"])
    assert launch["schema"] == "posttrain.execution-launch.v1"
    assert launch["run"]["run_id"] == plan.request.run_spec.run_id
    assert launch["provider"] == "dstack"
    assert launch["attempt"] == 1
    assert launch["job_image"] == plan.request.image.value
    assert launch["target"]["id"] == plan.request.target.id
    plan_config = next(config for action, config in configurations if action == "plan")
    submit_config = next(config for action, config in configurations if action == "submit")
    assert "files" not in plan_config
    assert "files" not in submit_config
    assert plan.details["job_image"] == plan.request.image.value
    assert all(config["retry"] is False for _, config in configurations)
    assert all(
        config["volumes"]
        == [
            {
                "instance_path": "/var/lib/posttrain/cache/huggingface",
                "path": "/root/.cache/huggingface",
                "optional": False,
            },
            {
                "instance_path": "/var/lib/posttrain/runs/test-run",
                "path": "/opt/posttrain/run",
                "optional": False,
            },
        ]
        for _, config in configurations
    )
    assert all(config["resources"]["gpu"]["memory"] == "24GB..30GB" for _, config in configurations)
    assert all(config["instances"] == [{"hostname": "remote.lan"}] for _, config in configurations)
    assert all(
        config["commands"] == ["posttrain-runtime execute --manifest /opt/posttrain/job/package.json"]
        for _, config in configurations
    )
    assert all(config["working_dir"] == "/opt/posttrain/job" for _, config in configurations)
    assert all(config["tags"]["posttrain_job_image_digest"] == "b" * 64 for _, config in configurations)
    assert all(config["tags"]["posttrain_attempt"] == "1" for _, config in configurations)
    assert "secret" not in str(configurations)


def _sdk_bridge_module(monkeypatch: pytest.MonkeyPatch):
    """Load the standalone bridge with a tiny dstack API double."""

    class Task:
        def __init__(self, **values) -> None:
            self.values = values

    dstack = types.ModuleType("dstack")
    api = types.ModuleType("dstack.api")
    api.__dict__["Client"] = object
    api.__dict__["Task"] = Task
    api.__dict__["VirtualRepo"] = object
    native_state = types.ModuleType("native_state")
    native_state.__dict__["assignment_state"] = lambda _run: "never-assigned"
    monkeypatch.setitem(sys.modules, "dstack", dstack)
    monkeypatch.setitem(sys.modules, "dstack.api", api)
    monkeypatch.setitem(sys.modules, "native_state", native_state)
    path = Path(__file__).parents[1] / "src/posttrain_execution_dstack/sdk_bridge.py"
    specification = importlib.util.spec_from_file_location("test_dstack_sdk_bridge", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_sdk_bridge_uses_only_the_private_runtime_map_for_declared_job_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _sdk_bridge_module(monkeypatch)
    monkeypatch.setenv("TRACKIO_WRITE_TOKEN", "from-submitting-shell")

    task = module._configuration(
        {
            "configuration": {
                "env": ["TRACKIO_WRITE_TOKEN"],
                "_posttrain_runtime_env": {"TRACKIO_WRITE_TOKEN": "from-posttrain-env"},
            }
        }
    )

    assert task.values["env"] == {"TRACKIO_WRITE_TOKEN": "from-posttrain-env"}
    with pytest.raises(RuntimeError, match="posttrain.env"):
        module._configuration({"configuration": {"env": ["TRACKIO_WRITE_TOKEN"]}})


def test_sdk_bridge_private_runtime_values_are_not_part_of_public_provider_configuration(
    tmp_path: Path,
) -> None:
    python = tmp_path / "python"
    python.symlink_to(Path(sys.executable))
    bridge = tmp_path / "bridge.py"
    bridge.write_text(
        "import json, sys\n"
        "payload = json.load(sys.stdin)\n"
        "print(json.dumps({'runtime': payload['configuration'].pop('_posttrain_runtime_env')}))\n",
        encoding="utf-8",
    )
    sdk = DstackSdkBridge(
        python,
        bridge=bridge,
        runtime_environment={"TRACKIO_WRITE_TOKEN": "from-posttrain-env"},
    )
    payload = {"project": "posttrain", "configuration": {"env": ["TRACKIO_WRITE_TOKEN"]}}

    response = sdk.invoke("plan", payload)

    assert response == {"runtime": {"TRACKIO_WRITE_TOKEN": "from-posttrain-env"}}
    assert payload == {"project": "posttrain", "configuration": {"env": ["TRACKIO_WRITE_TOKEN"]}}


def test_sdk_bridge_lifecycle_actions_do_not_require_submission_configuration(tmp_path: Path) -> None:
    python = tmp_path / "python"
    python.symlink_to(Path(sys.executable))
    bridge = tmp_path / "bridge.py"
    bridge.write_text(
        "import json, sys\npayload = json.load(sys.stdin)\nprint(json.dumps(payload))\n",
        encoding="utf-8",
    )
    sdk = DstackSdkBridge(python, bridge=bridge, runtime_environment={"SECRET": "not-forwarded"})

    response = sdk.invoke("status", {"project": "posttrain", "run_name": "pt-example"})

    assert response == {"project": "posttrain", "run_name": "pt-example"}


def test_sdk_cleanup_waits_through_transient_worker_capacity_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _sdk_bridge_module(monkeypatch)
    attempts: list[int] = []
    applied = object()

    class Runs:
        def get(self, _name):
            return None

        def get_run_plan(self, *, configuration, repo):
            del repo
            gpu_count = configuration.values["resources"]["gpu"]["count"]
            attempts.append(gpu_count)
            offers = () if len(attempts) == 1 else (object(),)
            return types.SimpleNamespace(job_plans=(types.SimpleNamespace(offers=offers),))

        def apply_plan(self, *, run_plan, repo, reserve_ports):
            del run_plan, repo, reserve_ports
            return applied

    client = types.SimpleNamespace(runs=Runs())
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    result = module._apply_cleanup_when_worker_is_available(
        client,
        {"name": "pt-clean-test", "image": "registry.lan/posttrain@sha256:" + "a" * 64},
    )

    assert result is applied
    assert attempts == [0, 1]


def test_dstack_maps_mandatory_instance_trust_bundle_as_additional_authorities(
    tmp_path: Path,
) -> None:
    gateway = FakeGateway()
    instance_bundle = Path("/etc/posttrain/trust/internal-ca.pem")
    provider = DstackExecutionProvider(
        gateway,
        project="posttrain",
        trust_bundle=instance_bundle,
    )

    provider.plan(_request(tmp_path))

    configuration = gateway.calls[0][1]["configuration"]
    stable_path = "/opt/posttrain/trust/ca-certificates.crt"
    assert configuration["volumes"][-1] == {
        "instance_path": str(instance_bundle),
        "path": stable_path,
        "optional": False,
    }
    assert configuration["setup"] == [f"test -f {stable_path}"]
    launch = configuration["_posttrain_launch_env"]
    assert launch["POSTTRAIN_EXTRA_CA_BUNDLE"] == stable_path
    # The image merges this with the authorities it already trusts. Setting
    # SSL_CERT_FILE here would replace that set instead, so a job that gained
    # an internal registry would lose every public authority with it, and fail
    # much later verifying something unrelated.
    assert "SSL_CERT_FILE" not in launch
    assert "REQUESTS_CA_BUNDLE" not in launch
    assert "test certificate bundle" not in str(configuration).lower()


@pytest.mark.parametrize("max_attempts", [1, 2, 5])
def test_admitted_task_is_fail_fast_for_every_framework_attempt_policy(
    tmp_path: Path,
    max_attempts: int,
) -> None:
    gateway = FakeGateway()
    provider = DstackExecutionProvider(gateway, project="posttrain")
    request = replace(
        _request(tmp_path),
        policy=ExecutionPolicy(900, max_attempts=max_attempts),
    )

    plan = provider.plan(request)
    provider.submit(plan)

    plan_configuration = gateway.calls[0][1]["configuration"]
    submit_configuration = gateway.calls[1][1]["configuration"]
    assert request.policy.max_attempts == max_attempts
    assert plan_configuration["retry"] is False
    assert submit_configuration["retry"] is False
    assert "files" not in plan_configuration
    assert "files" not in submit_configuration


def test_capacity_wait_retries_only_pre_start_no_capacity(
    tmp_path: Path,
) -> None:
    gateway = FakeGateway()
    provider = DstackExecutionProvider(
        gateway,
        project="posttrain",
        capacity_wait_seconds=86_400,
    )

    plan = provider.plan(_request(tmp_path))
    provider.submit(plan)

    configurations = [payload["configuration"] for action, payload in gateway.calls if action in {"plan", "submit"}]
    assert plan.details["capacity_wait_seconds"] == 86_400
    assert all(
        configuration["retry"]
        == {
            "on_events": ["no-capacity"],
            "duration": 86_400,
        }
        for configuration in configurations
    )


def test_gpu_memory_maximum_must_cover_the_target_minimum(tmp_path: Path) -> None:
    gateway = FakeGateway()
    provider = DstackExecutionProvider(gateway, project="posttrain")
    request = _request(tmp_path)
    request = replace(
        request,
        target=replace(
            request.target,
            placement={**request.target.placement, "gpu_memory_max_gb": 20},
        ),
    )
    with pytest.raises(ValueError, match="maximum GPU memory"):
        provider.plan(request)


def test_status_bounded_logs_cancel_and_collect(tmp_path: Path) -> None:
    gateway = FakeGateway()
    provider = DstackExecutionProvider(gateway, project="posttrain")
    request = _request(tmp_path)
    handle = provider.submit(provider.plan(request))
    assert provider.status(handle).state == "queued"
    page = provider.logs(handle, LogCursor(1), limit=1)
    assert page.lines == ("two",)
    assert page.next_cursor.offset == 2
    assert page.truncated is True
    provider.cancel(handle)
    assert provider.status(handle).state == "cancelled"
    assert provider.collect(handle).record.state == "cancelled"
    gateway.status = "done"
    cleanup = provider.cleanup(
        handle,
        run_id="test-run",
        run_workspace=Path("/var/lib/posttrain/runs/test-run"),
        runtime_image=request.image,
    )
    assert cleanup.disposition == "provider-managed"
    assert cleanup.workspace_disposition == "removed"
    assert cleanup.workspace_reclaimed_bytes == 41
    action, payload = gateway.calls[-1]
    assert action == "cleanup_workspace"
    assert payload["project"] == "posttrain"
    assert payload["source_run_name"] == handle.provider_id
    assert payload["hostname"] == "gpu-worker-a"
    assert payload["run_id"] == "test-run"
    assert payload["workspace"] == "/var/lib/posttrain/runs/test-run"
    assert payload["image"] == request.image.value
    assert payload["cleanup_run_name"].startswith("pt-clean-")
    assert "retained run history" in cleanup.message


@pytest.mark.parametrize(
    ("run_id", "workspace"),
    [
        ("../test-run", Path("/var/lib/posttrain/runs/test-run")),
        ("test-run", Path("/var/lib/posttrain/runs")),
        ("test-run", Path("var/lib/posttrain/runs/test-run")),
        ("test-run", Path("/test-run")),
    ],
)
def test_cleanup_rejects_any_non_exact_run_workspace(
    tmp_path: Path,
    run_id: str,
    workspace: Path,
) -> None:
    gateway = FakeGateway()
    gateway.status = "done"
    provider = DstackExecutionProvider(gateway, project="posttrain")
    handle = provider.submit(provider.plan(_request(tmp_path)))

    with pytest.raises(RuntimeError, match="cleanup"):
        provider.cleanup(
            handle,
            run_id=run_id,
            run_workspace=workspace,
            runtime_image=_request(tmp_path).image,
        )

    assert not any(action == "cleanup_workspace" for action, _ in gateway.calls)


def test_cleanup_fails_closed_when_task_does_not_verify_scope(
    tmp_path: Path,
) -> None:
    gateway = FakeGateway()
    gateway.status = "done"
    gateway.cleanup_response = {
        "hostname": "different-worker",
        "workspace": "/var/lib/posttrain/runs/test-run",
        "emptied": True,
        "reclaimed_bytes": 0,
    }
    provider = DstackExecutionProvider(gateway, project="posttrain")
    request = _request(tmp_path)
    handle = provider.submit(provider.plan(request))

    with pytest.raises(RuntimeError, match="did not verify"):
        provider.cleanup(
            handle,
            run_id="test-run",
            run_workspace=Path("/var/lib/posttrain/runs/test-run"),
            runtime_image=request.image,
        )


def test_pre_assignment_failure_records_workspace_as_not_created(
    tmp_path: Path,
) -> None:
    gateway = FakeGateway()
    gateway.status = "failed"
    gateway.cleanup_response = {
        "cleanup_run_name": None,
        "hostname": None,
        "workspace": "/var/lib/posttrain/runs/test-run",
        "workspace_state": "not-created",
        "emptied": False,
        "reclaimed_bytes": 0,
    }
    provider = DstackExecutionProvider(gateway, project="posttrain")
    request = _request(tmp_path)
    handle = provider.submit(provider.plan(request))

    cleanup = provider.cleanup(
        handle,
        run_id="test-run",
        run_workspace=Path("/var/lib/posttrain/runs/test-run"),
        runtime_image=request.image,
    )

    assert cleanup.workspace_disposition == "not-created"
    assert cleanup.workspace_reclaimed_bytes == 0
    assert "no worker workspace was created" in cleanup.message
    action, payload = gateway.calls[-1]
    assert action == "cleanup_workspace"
    assert payload["hostname"] is None


def test_unassigned_failure_fails_closed_without_provider_native_proof(
    tmp_path: Path,
) -> None:
    gateway = FakeGateway()
    gateway.status = "failed"
    gateway.cleanup_response = {
        "hostname": None,
        "workspace": "/var/lib/posttrain/runs/test-run",
        "workspace_state": "ambiguous",
        "emptied": False,
        "reclaimed_bytes": 0,
    }
    provider = DstackExecutionProvider(gateway, project="posttrain")
    request = _request(tmp_path)
    handle = provider.submit(provider.plan(request))

    with pytest.raises(RuntimeError, match="did not verify"):
        provider.cleanup(
            handle,
            run_id="test-run",
            run_workspace=Path("/var/lib/posttrain/runs/test-run"),
            runtime_image=request.image,
        )


class _Native:
    def __init__(self, **values) -> None:
        self.__dict__.update(values)


def test_native_assignment_classifier_requires_complete_empty_history() -> None:
    never_assigned = _Native(
        jobs=[
            _Native(
                job_connection_info=None,
                job_submissions=[_Native(job_provisioning_data=None, job_runtime_data=None)],
            )
        ]
    )
    assigned_without_hostname = _Native(
        jobs=[
            _Native(
                job_connection_info=None,
                job_submissions=[
                    _Native(
                        job_provisioning_data=_Native(hostname=None),
                        job_runtime_data=None,
                    )
                ],
            )
        ]
    )

    assert assignment_state(never_assigned) == "never-assigned"
    assert assignment_state(assigned_without_hostname) == "assigned"
    assert assignment_state(_Native(jobs=[])) == "ambiguous"
    assert assignment_state(_Native(jobs=[_Native(job_connection_info=None, job_submissions=[])])) == "ambiguous"


def test_dstack_legacy_bundle_is_plan_only_and_never_uploaded(
    tmp_path: Path,
) -> None:
    gateway = FakeGateway()
    provider = DstackExecutionProvider(gateway, project="posttrain")
    request = replace(
        _request(tmp_path),
        bundle=BundleRef((tmp_path / "legacy-bundle").resolve(), "c" * 64),
    )

    plan = provider.plan(request)

    configuration = gateway.calls[0][1]["configuration"]
    assert "files" not in configuration
    assert plan.details["submission_ready"] is False
    with pytest.raises(RuntimeError, match="no longer accepts execution bundles"):
        provider.submit(plan)
    assert [action for action, _ in gateway.calls] == ["plan"]
