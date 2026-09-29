from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
from posttrain.common import ContractError, ExecutionTarget
from posttrain.execution import (
    JOB_PACKAGE_WORKER_COMMAND,
    BundleRef,
    ExecutionMount,
    ExecutionPolicy,
    ExecutionRequest,
    LogCursor,
    ProviderCleanupDeferred,
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
    assert all(
        config["retry"]
        == {
            "on_events": ["interruption"],
            "duration": 7_200,
            "duration_by_event": {"interruption": 7_200},
            "max_attempts_by_event": {"interruption": 5},
        }
        for _, config in configurations
    )
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
    # dstack otherwise leaves Docker's 64 MiB /dev/shm; the offer must hold it.
    assert all(config["resources"]["shm_size"] == "16GB" for _, config in configurations)
    assert all(config["resources"]["memory"] == "16GB.." for _, config in configurations)
    assert plan.details["shared_memory_gb"] == 16
    assert all(config["instances"] == [{"hostname": "remote.lan"}] for _, config in configurations)
    assert all(
        config["commands"]
        == [
            "ulimit -n 65536 2>/dev/null || true; exec "
            "posttrain-runtime execute --manifest /opt/posttrain/job/package.json"
        ]
        for _, config in configurations
    )
    assert all(config["working_dir"] == "/opt/posttrain/job" for _, config in configurations)
    assert all(config["tags"]["posttrain_job_image_digest"] == "b" * 64 for _, config in configurations)
    assert all(config["tags"]["posttrain_attempt"] == "1" for _, config in configurations)
    assert "secret" not in str(configurations)


def test_managed_run_storage_omits_instance_mounts_and_private_ca(tmp_path: Path) -> None:
    gateway = FakeGateway()
    provider = DstackExecutionProvider(
        gateway,
        project="posttrain",
        trust_bundle=Path("/var/lib/posttrain/trust/ca-certificates.crt"),
    )
    request = _request(tmp_path)
    target = replace(
        request.target,
        placement={
            "backends": ["runpod"],
            "regions": ["US-MO-1"],
            "spot_policy": "spot",
            "managed_run_storage": True,
        },
    )

    provider.plan(replace(request, target=target))

    configuration = gateway.calls[-1][1]["configuration"]
    assert "volumes" not in configuration
    assert configuration["tags"]["posttrain_managed_run_storage"] == "true"
    assert "setup" not in configuration
    launch_environment = configuration["_posttrain_launch_env"]
    assert "POSTTRAIN_EXTRA_CA_BUNDLE" not in launch_environment
    assert launch_environment["HF_HOME"] == "/opt/posttrain/run/.cache/huggingface"
    assert launch_environment["TORCHINDUCTOR_CACHE_DIR"] == "/opt/posttrain/run/.cache/compile/torchinductor"
    assert launch_environment["TRITON_CACHE_DIR"] == "/opt/posttrain/run/.cache/compile/triton"


def test_managed_run_storage_requires_boolean(tmp_path: Path) -> None:
    request = _request(tmp_path)
    target = replace(request.target, placement={"managed_run_storage": "yes"})

    with pytest.raises(ValueError, match="managed_run_storage must be a boolean"):
        DstackExecutionProvider(FakeGateway(), project="posttrain").plan(replace(request, target=target))


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


def test_sdk_bridge_uses_native_dstack_secret_reference_without_receiving_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _sdk_bridge_module(monkeypatch)

    task = module._configuration(
        {
            "configuration": {
                "env": ["OPENROUTER_API_KEY"],
                "_posttrain_runtime_env": {},
                "_posttrain_runtime_secret_refs": {
                    "OPENROUTER_API_KEY": "openrouter-job-key",
                },
            }
        }
    )

    assert task.values["env"] == {
        "OPENROUTER_API_KEY": "${{ secrets.openrouter-job-key }}",
    }


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


def test_sdk_bridge_failure_preserves_the_bounded_validation_context(tmp_path: Path) -> None:
    python = tmp_path / "python"
    python.symlink_to(Path(sys.executable))
    bridge = tmp_path / "bridge.py"
    bridge.write_text(
        "import sys\n"
        "for line in ('trace header', 'retry -> duration_by_event', "
        "'extra fields not permitted', 'value could not be parsed to a boolean'):\n"
        "    print(line, file=sys.stderr)\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )
    sdk = DstackSdkBridge(python, bridge=bridge)

    with pytest.raises(RuntimeError) as error:
        sdk.invoke("plan", {"project": "posttrain"})

    message = str(error.value)
    assert "retry -> duration_by_event" in message
    assert "extra fields not permitted" in message
    assert "value could not be parsed to a boolean" in message


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


def test_sdk_cleanup_submits_durable_zero_offer_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _sdk_bridge_module(monkeypatch)
    plans: list[object] = []
    configurations: list[dict] = []
    applied = object()

    class Runs:
        def get(self, _name):
            return None

        def get_run_plan(self, *, configuration, repo):
            del repo
            configurations.append(configuration.values)
            plan = types.SimpleNamespace(job_plans=(types.SimpleNamespace(offers=()),))
            plans.append(plan)
            return plan

        def apply_plan(self, *, run_plan, repo, reserve_ports):
            del repo, reserve_ports
            assert run_plan is plans[0]
            return applied

    client = types.SimpleNamespace(runs=Runs())
    result = module._apply_cleanup_when_worker_is_available(
        client,
        {
            "name": "pt-clean-test",
            "image": "registry.lan/posttrain@sha256:" + "a" * 64,
            "retry": {"on_events": ["no-capacity"], "duration": 86_400},
        },
    )

    assert result is applied
    assert [value["resources"]["gpu"]["count"] for value in configurations] == [0, 1]
    assert all(value["retry"]["on_events"] == ["no-capacity"] for value in configurations)


def test_sdk_cleanup_resumes_existing_nonterminal_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _sdk_bridge_module(monkeypatch)

    class Run:
        def __init__(self, status: str) -> None:
            self.status = types.SimpleNamespace(value=status)

        def refresh(self):
            return None

    failed = Run("failed")
    queued = Run("submitted")

    class Runs:
        def get(self, name):
            return {
                "pt-clean-test": failed,
                "pt-clean-test-retry-1": queued,
            }.get(name)

        def get_run_plan(self, **_kwargs):
            raise AssertionError("an existing queued retry must be resumed")

        def apply_plan(self, **_kwargs):
            raise AssertionError("an existing queued retry must not be resubmitted")

    result = module._apply_cleanup_when_worker_is_available(
        types.SimpleNamespace(runs=Runs()),
        {"name": "pt-clean-test", "image": "registry.lan/posttrain@sha256:" + "a" * 64},
    )

    assert result is queued


def test_sdk_cleanup_returns_deferred_evidence_for_queued_exact_worker_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _sdk_bridge_module(monkeypatch)

    class Run:
        def __init__(self, name: str, status: str, *, hostname: str | None = None) -> None:
            self.name = name
            self.status = types.SimpleNamespace(value=status)
            self.hostname = hostname
            self._run = object()

        def refresh(self):
            return None

    source = Run("source-run", "done", hostname="gpu-worker-a")
    cleanup = Run("pt-clean-test", "pending")
    configurations: list[dict] = []

    class Runs:
        def get(self, name):
            return source if name == "source-run" else None

        def get_run_plan(self, *, configuration, repo):
            del repo
            configurations.append(configuration.values)
            return types.SimpleNamespace(job_plans=(types.SimpleNamespace(offers=()),))

        def apply_plan(self, *, run_plan, repo, reserve_ports):
            del run_plan, repo, reserve_ports
            return cleanup

    monkeypatch.setattr(module, "_client", lambda _payload: types.SimpleNamespace(runs=Runs()))
    response = module.cleanup_workspace(
        {
            "project": "posttrain",
            "source_run_name": "source-run",
            "cleanup_run_name": "pt-clean-test",
            "hostname": "gpu-worker-a",
            "run_id": "test-run",
            "workspace": "/var/lib/posttrain/runs/test-run",
            "image": "registry.lan/posttrain@sha256:" + "a" * 64,
        }
    )

    assert response == {
        "cleanup_run_name": "pt-clean-test",
        "cleanup_status": "pending",
        "hostname": "gpu-worker-a",
        "workspace": "/var/lib/posttrain/runs/test-run",
        "workspace_state": "deferred",
        "emptied": False,
        "reclaimed_bytes": 0,
    }
    assert configurations[0]["retry"] == {"on_events": ["no-capacity"], "duration": 86_400}
    assert configurations[0]["priority"] == 100


@pytest.mark.parametrize(
    ("active", "workspace_state", "emptied"),
    [
        (True, "provider-managed-deferred", False),
        (False, "provider-managed-removed", True),
    ],
)
def test_sdk_cleanup_uses_managed_volume_absence_as_terminal_evidence(
    monkeypatch: pytest.MonkeyPatch,
    active: bool,
    workspace_state: str,
    emptied: bool,
) -> None:
    module = _sdk_bridge_module(monkeypatch)
    volume_name = "run-12345678123456781234567812345678"
    native = _Native(
        id="12345678-1234-5678-1234-567812345678",
        run_spec=_Native(
            configuration=_Native(
                tags={},
                volumes=[_Native(name=volume_name)],
            )
        ),
    )
    source = _Native(_run=native)
    volumes = [_Native(name=volume_name)] if active else []
    client = _Native(
        project="posttrain",
        client=_Native(volumes=_Native(list=lambda **_kwargs: volumes)),
    )

    response = module._managed_run_storage_cleanup(
        client,
        source,
        "/var/lib/posttrain/runs/test-run",
        "gpu-worker-a",
    )

    assert response is not None
    assert response["workspace_state"] == workspace_state
    assert response["emptied"] is emptied
    assert response["reclaimed_bytes"] == 0


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
def test_training_retries_only_provider_interruption_for_every_framework_attempt_policy(
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
    expected_retry = {
        "on_events": ["interruption"],
        "duration": 7_200,
        "duration_by_event": {"interruption": 7_200},
        "max_attempts_by_event": {"interruption": 5},
    }
    assert plan_configuration["retry"] == expected_retry
    assert submit_configuration["retry"] == expected_retry
    assert plan_configuration["_posttrain_launch_env"]["POSTTRAIN_INTERRUPTION_RECOVERY"] == "1"
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
            "on_events": ["no-capacity", "interruption"],
            "duration": 86_400,
            "duration_by_event": {
                "no-capacity": 86_400,
                "interruption": 7_200,
            },
            "max_attempts_by_event": {"interruption": 5},
        }
        for configuration in configurations
    )


def test_target_can_constrain_backend_region_and_spot_policy(tmp_path: Path) -> None:
    gateway = FakeGateway()
    provider = DstackExecutionProvider(gateway, project="posttrain")
    request = _request(tmp_path)
    request = replace(
        request,
        target=replace(
            request.target,
            placement={
                **request.target.placement,
                "backends": ["runpod"],
                "regions": ["US-MO-1"],
                "spot_policy": "spot",
                "max_price": 1.0,
            },
        ),
    )

    provider.plan(request)

    configuration = gateway.calls[0][1]["configuration"]
    assert configuration["backends"] == ["runpod"]
    assert configuration["regions"] == ["US-MO-1"]
    assert configuration["spot_policy"] == "spot"
    assert configuration["max_price"] == 1.0


def test_target_declares_shared_memory_and_the_host_memory_that_bounds_it(tmp_path: Path) -> None:
    gateway = FakeGateway()
    provider = DstackExecutionProvider(gateway, project="posttrain")
    request = _request(tmp_path)

    def resources(**placement: object) -> dict[str, object]:
        gateway.calls.clear()
        target = replace(request.target, placement={**request.target.placement, **placement})
        plan = provider.plan(replace(request, target=target))
        configuration = gateway.calls[0][1]["configuration"]
        assert plan.details["shared_memory_gb"] == int(configuration["resources"]["shm_size"].removesuffix("GB"))
        return configuration["resources"]

    assert resources(shm_size_gb=32) | {"gpu": None} == {
        "gpu": None,
        "disk": {"size": "120GB.."},
        "shm_size": "32GB",
        "memory": "32GB..",
    }
    declared_host = resources(host_memory_gb=62.5)
    assert (declared_host["shm_size"], declared_host["memory"]) == ("16GB", "63GB..")
    small_host = resources(host_memory_gb=12)
    assert (small_host["shm_size"], small_host["memory"]) == ("6GB", "12GB..")
    with pytest.raises(ContractError, match="exceeds its host_memory_gb"):
        resources(host_memory_gb=12, shm_size_gb=16)


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
    diagnostic_page = provider.logs(handle, limit=1, stream="diagnostic")
    assert diagnostic_page.lines == ("one",)
    diagnostic_action, diagnostic_payload = gateway.calls[-1]
    assert diagnostic_action == "logs"
    assert diagnostic_payload["diagnose"] is True
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


def test_dstack_max_duration_is_the_execution_policy_timeout(tmp_path: Path) -> None:
    gateway = FakeGateway()
    provider = DstackExecutionProvider(gateway, project="posttrain")
    request = replace(_request(tmp_path), policy=ExecutionPolicy(timeout_seconds=90_000))

    provider.plan(request)

    action, payload = gateway.calls[-1]
    assert action == "plan"
    assert payload["configuration"]["max_duration"] == 90_000


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


def test_cleanup_reports_exact_worker_capacity_wait_as_deferred(tmp_path: Path) -> None:
    gateway = FakeGateway()
    gateway.status = "done"
    gateway.cleanup_response = {
        "cleanup_run_name": "pt-clean-test",
        "cleanup_status": "pending",
        "hostname": "gpu-worker-a",
        "workspace": "/var/lib/posttrain/runs/test-run",
        "workspace_state": "deferred",
        "emptied": False,
        "reclaimed_bytes": 0,
    }
    provider = DstackExecutionProvider(gateway, project="posttrain")
    request = _request(tmp_path)
    handle = provider.submit(provider.plan(request))

    with pytest.raises(ProviderCleanupDeferred, match="retry the same immutable purge"):
        provider.cleanup(
            handle,
            run_id="test-run",
            run_workspace=Path("/var/lib/posttrain/runs/test-run"),
            runtime_image=request.image,
        )


def test_cleanup_accepts_provider_managed_volume_absence(tmp_path: Path) -> None:
    gateway = FakeGateway()
    gateway.status = "done"
    gateway.cleanup_response = {
        "cleanup_run_name": None,
        "hostname": "gpu-worker-a",
        "workspace": "/var/lib/posttrain/runs/test-run",
        "workspace_state": "provider-managed-removed",
        "emptied": True,
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

    assert cleanup.disposition == "provider-managed"
    assert cleanup.workspace_disposition == "removed"
    assert cleanup.workspace_reclaimed_bytes == 0
    assert "run-owned storage volume is absent" in cleanup.message


def test_cleanup_defers_while_provider_managed_volume_exists(tmp_path: Path) -> None:
    gateway = FakeGateway()
    gateway.status = "done"
    gateway.cleanup_response = {
        "cleanup_run_name": None,
        "cleanup_status": "provider-storage-deleting",
        "hostname": "gpu-worker-a",
        "workspace": "/var/lib/posttrain/runs/test-run",
        "workspace_state": "provider-managed-deferred",
        "emptied": False,
        "reclaimed_bytes": 0,
    }
    provider = DstackExecutionProvider(gateway, project="posttrain")
    request = _request(tmp_path)
    handle = provider.submit(provider.plan(request))

    with pytest.raises(ProviderCleanupDeferred, match="still deleting"):
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


def test_active_executions_for_run_matches_only_the_posttrain_run_tag() -> None:
    class InventoryGateway:
        def __init__(self, response: dict) -> None:
            self.response = response
            self.calls: list[tuple[str, dict]] = []

        def invoke(self, action: str, payload):
            self.calls.append((action, dict(payload)))
            return self.response

    gateway = InventoryGateway(
        {
            "runs": [
                {"run_name": "pt-a", "status": "running", "posttrain_run_id": "orphan-run"},
                {"run_name": "pt-b", "status": "running", "posttrain_run_id": "other-run"},
                {"run_name": "manual", "status": "pending", "posttrain_run_id": None},
            ],
            "complete": True,
        }
    )
    provider = DstackExecutionProvider(gateway, project="main")

    assert provider.active_executions_for_run("orphan-run") == ("dstack:pt-a (running)",)
    assert provider.inventory_scope == "dstack project 'main'"
    assert gateway.calls == [("active_runs", {"project": "main"})]

    gateway.response = {"runs": [], "complete": False}
    with pytest.raises(RuntimeError, match="incomplete"):
        provider.active_executions_for_run("orphan-run")


def test_sdk_bridge_active_runs_skips_terminal_runs_and_reports_page_completeness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _sdk_bridge_module(monkeypatch)

    def run(name: str, status: str, tags: dict | None) -> types.SimpleNamespace:
        configuration = types.SimpleNamespace(tags=tags)
        return types.SimpleNamespace(
            name=name,
            status=types.SimpleNamespace(value=status),
            _run=types.SimpleNamespace(run_spec=types.SimpleNamespace(configuration=configuration)),
        )

    runs = [
        run("pt-live", "running", {"posttrain_run_id": "orphan-run"}),
        run("pt-done", "done", {"posttrain_run_id": "orphan-run"}),
        run("manual", "pending", None),
    ]
    client = types.SimpleNamespace(runs=types.SimpleNamespace(list=lambda: runs))
    monkeypatch.setattr(module, "_client", lambda _payload: client)

    assert module.active_runs({"project": "main"}) == {
        "runs": [
            {"run_name": "pt-live", "status": "running", "posttrain_run_id": "orphan-run"},
            {"run_name": "manual", "status": "pending", "posttrain_run_id": None},
        ],
        "complete": True,
    }


def _done_cleanup_payload() -> dict[str, str]:
    return {
        "project": "posttrain",
        "source_run_name": "source-run",
        "cleanup_run_name": "pt-clean-test",
        "hostname": "gpu-worker-a",
        "run_id": "test-run",
        "workspace": "/var/lib/posttrain/runs/test-run",
        "image": "registry.lan/posttrain@sha256:" + "a" * 64,
    }


def _done_cleanup_client(log_batches: list[list[bytes]]):
    """A finished source run and an existing done cleanup task whose logs arrive per call."""

    class Run:
        def __init__(self, name: str, status: str, hostname: str | None = None) -> None:
            self.name = name
            self.status = types.SimpleNamespace(value=status)
            self.hostname = hostname
            self._run = object()
            self.log_calls = 0

        def refresh(self):
            return None

        def logs(self, *, replica_num, job_num):
            assert (replica_num, job_num) == (0, 0)
            self.log_calls += 1
            return iter(log_batches[min(self.log_calls, len(log_batches)) - 1])

    source = Run("source-run", "done", hostname="gpu-worker-a")
    cleanup = Run("pt-clean-test", "done")

    class Runs:
        def get(self, name):
            return {"source-run": source, "pt-clean-test": cleanup}.get(name)

    return types.SimpleNamespace(runs=Runs()), cleanup


@pytest.mark.parametrize(
    ("log_batches", "reclaimed", "log_calls"),
    [
        # lfm26-sampo-cont40-fixed-tools-20260928-r1: awk printed the 4.6 GB total in scientific notation.
        ([[b"POSTTRAIN_CLEANUP_RECLAIMED_BYTES=4.63293e+09\n"]], 4_632_930_000, 1),
        # Logs of a just-finished job arrive late.
        ([[], [], [b"POSTTRAIN_CLEANUP_RECLAIMED_BYTES=1234\n"]], 1234, 3),
    ],
)
def test_sdk_cleanup_reads_the_reclaimed_bytes_of_a_done_task(
    monkeypatch: pytest.MonkeyPatch, log_batches: list[list[bytes]], reclaimed: int, log_calls: int
) -> None:
    module = _sdk_bridge_module(monkeypatch)
    client, cleanup = _done_cleanup_client(log_batches)
    monkeypatch.setattr(module, "_client", lambda _payload: client)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    response = module.cleanup_workspace(_done_cleanup_payload())

    assert response == {
        "cleanup_run_name": "pt-clean-test",
        "hostname": "gpu-worker-a",
        "workspace": "/var/lib/posttrain/runs/test-run",
        "emptied": True,
        "reclaimed_bytes": reclaimed,
    }
    assert cleanup.log_calls == log_calls


def test_sdk_cleanup_rerun_completes_when_the_done_task_has_no_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The task exits 0 only after verifying the workspace is empty; missing logs do not undo that."""

    module = _sdk_bridge_module(monkeypatch)
    client, cleanup = _done_cleanup_client([[]])
    monkeypatch.setattr(module, "_client", lambda _payload: client)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    response = module.cleanup_workspace(_done_cleanup_payload())

    assert response["emptied"] is True and response["reclaimed_bytes"] == 0
    assert response["reclaimed_bytes_evidence"] == "unavailable"
    assert cleanup.log_calls == module._CLEANUP_LOG_ATTEMPTS


def test_cleanup_command_prints_whole_bytes_for_large_workspaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sdk_bridge_module(monkeypatch)
    workspace = tmp_path / "cleanup"
    workspace.mkdir()
    big = workspace / "checkpoint.bin"
    with big.open("wb") as stream:
        stream.truncate(3_000_000_123)  # sparse: above awk's integer print range
    command = module._cleanup_command().replace("/opt/posttrain/cleanup", str(workspace))

    result = subprocess.run(["/bin/sh", "-c", command], capture_output=True, text=True, check=True)

    assert module.parse_reclaimed_bytes(result.stdout.splitlines()) == 3_000_000_123
    assert "e+" not in result.stdout
    assert list(workspace.iterdir()) == []
