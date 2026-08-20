from __future__ import annotations

import hashlib
import json
import signal
import sys
import time
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest
from posttrain.catalog import open_catalog
from posttrain.common import (
    CatalogRef,
    ContractError,
    ExecutionTarget,
    JsonValue,
    LocalArtifactRef,
)
from posttrain.data import (
    DatasetLoadPlan,
    SupervisedDataSource,
)
from posttrain.execution import (
    DatasetAssetLock,
    DatasetPackageLock,
    EnvironmentActivationLock,
    EnvironmentPackageLock,
    JobPackageManifest,
    RuntimeImageRef,
)
from posttrain.tracking import ArtifactInput, RunOutcome, RunSpec
from posttrain.work import (
    JobDefinition,
    WorkPackageContext,
    execute_run_tracked_finalized,
    load_work_package,
    override_job_execution_target,
    prepare_work_package_job,
)
from posttrain_runtime import execute_manifest, qualify_manifest
from posttrain_runtime.execute import _project_config_digest, _qualification_timeout, _qualify_activation, _tree_digest


def test_qualification_loads_each_verifiers_taskset_offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    loaded: list[dict[str, object]] = []

    class EnvConfig:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        @classmethod
        def model_validate(cls, value: dict[str, object]) -> EnvConfig:
            return cls(value)

    class Environment:
        def __init__(self, config: EnvConfig) -> None:
            self.config = config
            self.taskset = type("Taskset", (), {"load": lambda _self: loaded.append(config.payload)})()

    verifiers = ModuleType("verifiers")
    v1 = ModuleType("verifiers.v1")
    env = ModuleType("verifiers.v1.env")
    env.__dict__["EnvConfig"] = EnvConfig
    env.__dict__["Environment"] = Environment
    monkeypatch.setitem(sys.modules, "verifiers", verifiers)
    monkeypatch.setitem(sys.modules, "verifiers.v1", v1)
    monkeypatch.setitem(sys.modules, "verifiers.v1.env", env)
    config = cast(Mapping[str, JsonValue], {"taskset": {"id": "offline"}})
    lock = EnvironmentActivationLock(
        environment_id="offline",
        package="offline-env",
        kind="verifiers-config",
        digest=hashlib.sha256(
            json.dumps({"kind": "verifiers-config", "config": config}, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        config=config,
    )

    _qualify_activation(lock, tmp_path)

    assert loaded == [config]


def test_qualification_rejects_a_factory_result_that_is_not_a_verifiers_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class EnvConfig:
        pass

    class Environment:
        def __init__(self, config: EnvConfig) -> None:
            self.config = config

    env = ModuleType("verifiers.v1.env")
    env.__dict__["EnvConfig"] = EnvConfig
    env.__dict__["Environment"] = Environment
    monkeypatch.setitem(sys.modules, "verifiers.v1.env", env)
    monkeypatch.setattr(
        "posttrain_runtime.execute.PythonFactoryActivation",
        lambda _reference: SimpleNamespace(activate=lambda: object()),
    )
    lock = SimpleNamespace(kind="python-factory", reference="example:environment")

    with pytest.raises(ContractError, match="did not produce a Verifiers EnvConfig"):
        _qualify_activation(lock, tmp_path)


def test_qualification_timeout_names_the_blocked_environment() -> None:
    with pytest.raises(TimeoutError, match="stuck"):
        with _qualification_timeout(0.01, "stuck"):
            time.sleep(0.1)


def test_deferred_qualification_requires_a_waiver_and_skips_taskset_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    required = SimpleNamespace(environment_id="offline", qualification="required")
    deferred = SimpleNamespace(environment_id="network-backed", qualification="deferred")
    package = SimpleNamespace(
        root=tmp_path,
        manifest=SimpleNamespace(environment_activations=(required, deferred)),
        datasets={"dataset": object()},
    )
    loaded: list[str] = []
    monkeypatch.setattr("posttrain_runtime.execute._verify_package", lambda _path: package)
    monkeypatch.setattr(
        "posttrain_runtime.execute._qualify_activation",
        lambda lock, _root: loaded.append(lock.environment_id),
    )

    with pytest.raises(ContractError, match="explicit waiver"):
        qualify_manifest(tmp_path / "package.json")

    loaded.clear()
    result = qualify_manifest(tmp_path / "package.json", allow_deferred=True)

    assert loaded == ["offline"]
    assert result.environment_ids == ("offline",)
    assert result.deferred_environment_ids == ("network-backed",)
    assert result.dataset_seats == ("dataset",)


def _project(
    root: Path,
    *,
    with_dataset: bool = False,
    with_remote_target: bool = False,
    tracking: str = "none",
) -> tuple[Path, Path]:
    control = root / ".posttrain"
    catalog = control / "catalog"
    work_packages = control / "work_packages"
    catalog.mkdir(parents=True)
    work_packages.mkdir()
    overlay_files = [
        name
        for enabled, name in (
            (with_dataset, "datasets.yaml"),
            (with_remote_target, "targets.yaml"),
        )
        if enabled
    ]
    (catalog / "layer.yaml").write_text(
        (f"schema_version: 1\nlayer_id: runtime-test-overlay\nfiles: [{', '.join(overlay_files)}]\n"),
        encoding="utf-8",
    )
    if with_dataset:
        (catalog / "datasets.yaml").write_text(
            """
dataset:
  datasets/runtime@1:
    revision: "1"
    kind: supervised
    source:
      kind: fixture
      resource: posttrain.data.fixtures:sft_messages.jsonl
    format:
      kind: messages
""".lstrip(),
            encoding="utf-8",
        )
    if with_remote_target:
        (catalog / "targets.yaml").write_text(
            """
target:
  targets/remote-24gb:
    revision: "1"
    device_class: nvidia-cuda
    memory_gb: 24
    placement:
      world_size: 1
      instances: [remote.lan]
""".lstrip(),
            encoding="utf-8",
        )
    project = control / "project.toml"
    project.write_text(
        "\n".join(
            (
                "schema_version = 1",
                'project_id = "runtime-tests"',
                'catalog_overlays = ["catalog"]',
                'work_packages = "work_packages"',
                'state = "state"',
                f'tracking = "{tracking}"',
                "",
            )
        ),
        encoding="utf-8",
    )
    work_package = work_packages / "check.yaml"
    dataset_seat = "\n    dataset: dataset" if with_dataset else ""
    dataset_binding = (
        """
  dataset:
    type: ref
    family: dataset
    id: datasets/runtime@1"""
        if with_dataset
        else ""
    )
    work_package.write_text(
        f"""
project_id: runtime-tests
work_package_id: screen/runtime
stage: screen
recipe:
  type: inline
  id: recipes/runtime-check@1
  revision: "1"
  stage: screen
  seats:
    target: target{dataset_seat}
  jobs:
    - id: check
      kind: data.prepare
      definition: data/runtime-check@1
bindings:
  target:
    type: ref
    family: target
    id: targets/local-cuda-8gb{dataset_binding}
enabled_optional_jobs: []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return project, work_package


def _runtime(
    catalog,
    seen: list[str],
    source_metadata: list[dict[str, object]] | None = None,
    *,
    with_dataset: bool = False,
    terminate: bool = False,
    terminate_signal: int = signal.SIGTERM,
    fail: bool = False,
) -> WorkPackageContext:
    def execute(context, seats):
        if with_dataset:
            dataset = seats["dataset"]
            assert isinstance(dataset, SupervisedDataSource)
            assert dataset.load().examples
        seen.append(context.run_id)
        if source_metadata is not None:
            source_metadata.append(dict(context.source_metadata))
        if terminate:
            signal.raise_signal(terminate_signal)
        if fail:
            raise RuntimeError("expected worker failure")
        return {"checked": True}

    seat_types: dict[str, type[object]] = {"target": ExecutionTarget}
    selection_seats: dict[str, type[object]] = {}
    if with_dataset:
        seat_types["dataset"] = SupervisedDataSource
        selection_seats["dataset"] = DatasetLoadPlan
    definition = JobDefinition(
        "data/runtime-check@1",
        "data.prepare",
        seat_types,
        execute,
        selection_seats=selection_seats,
    )
    return WorkPackageContext(catalog, {definition.id: definition})


class _TrackedRun:
    repeat_signal: int = signal.SIGTERM

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.outcomes: list[RunOutcome] = []

    def materialize_inputs(
        self,
        inputs: Mapping[str, ArtifactInput],
        root: Path,
    ) -> Mapping[str, LocalArtifactRef]:
        del inputs, root
        return {}

    def event(self, observation) -> None:
        del observation

    def metric(self, observation) -> None:
        del observation

    def metrics(self, observation) -> None:
        del observation

    def trace(self, observation) -> None:
        del observation

    def trace_fact_update(self, observation) -> None:
        del observation

    def artifact(self, artifact) -> None:
        del artifact

    def published_artifacts(self):
        return ()

    def flush_artifacts(self, timeout: float | None = None):
        del timeout
        return ()

    def finish(self, outcome: RunOutcome) -> None:
        signal.raise_signal(self.repeat_signal)
        self.outcomes.append(outcome)


class _TrackingBackend:
    def __init__(self, repeat_signal: int = signal.SIGTERM) -> None:
        self.tracked: _TrackedRun | None = None
        self._repeat_signal = repeat_signal

    def start_run(self, spec: RunSpec) -> _TrackedRun:
        self.tracked = _TrackedRun(spec.run_id)
        self.tracked.repeat_signal = self._repeat_signal
        return self.tracked


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _semantic_digest(value: object) -> str:
    return _digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _actual_job(
    root: Path,
    *,
    with_dataset: bool = False,
    with_dataset_assets: bool = False,
    target_override: bool = False,
    tracking: str = "none",
) -> tuple[Path, JobPackageManifest]:
    if with_dataset_assets and not with_dataset:
        raise ValueError("dataset assets require with_dataset=True")
    job = root.resolve()
    (job / "locks").mkdir(parents=True)
    (job / "sources/framework/runtime").mkdir(parents=True)
    (job / "sources/project/project_pkg").mkdir(parents=True)
    (job / "wheels/environments").mkdir(parents=True)
    (job / "datasets").mkdir()
    (job / "config/project").mkdir(parents=True)
    (job / "sources/framework/runtime/__init__.py").write_text('"""runtime."""\n', encoding="utf-8")
    (job / "sources/project/project_pkg/__init__.py").write_text('"""project."""\n', encoding="utf-8")
    project_manifest, work_package = _project(
        job / "config/project",
        with_dataset=with_dataset,
        with_remote_target=target_override,
        tracking=tracking,
    )

    runtime_lock = b""
    code_lock = b"./sources/framework/runtime\n"
    (job / "locks/runtime.requirements.txt").write_bytes(runtime_lock)
    (job / "locks/code.requirements.txt").write_bytes(code_lock)

    catalog = open_catalog(
        scope="runtime-tests",
        overlays=(job / "config/project/.posttrain/catalog",),
    )
    package = load_work_package(work_package)
    runtime = _runtime(catalog, [], with_dataset=with_dataset)
    if target_override:
        remote = catalog.resolve(CatalogRef("target", "targets/remote-24gb")).value
        assert isinstance(remote, ExecutionTarget)
        package = override_job_execution_target(
            runtime,
            package,
            "check",
            remote,
        )
    prepared = prepare_work_package_job(
        runtime,
        package,
        "check",
        run_id="run-runtime-1",
    )
    resolved_inputs = dict(prepared.spec.resolved_inputs)
    resolved = {
        "schema": "posttrain.resolved-job.v1",
        "project_id": "runtime-tests",
        "work_package_id": "screen/runtime",
        "job_id": "check",
        "job_definition_id": "data/runtime-check@1",
        "runtime_variant": "supervised",
        "project_root": "project",
        "project_manifest": "project/.posttrain/project.toml",
        "selected_work_package": ("project/.posttrain/work_packages/check.yaml"),
        "resolved_inputs": resolved_inputs,
    }
    resolved_bytes = (json.dumps(resolved, indent=2, sort_keys=True) + "\n").encode()
    (job / "config/resolved.json").write_bytes(resolved_bytes)
    dataset_locks: tuple[DatasetPackageLock, ...] = ()
    if with_dataset:
        dataset_root = job / "datasets/sft/locked"
        dataset_root.mkdir(parents=True)
        asset_locks: tuple[DatasetAssetLock, ...] = ()
        asset_record: dict[str, object] | None = None
        assets_digest: str | None = None
        if with_dataset_assets:
            asset_bytes = b"page image"
            asset_digest = _digest(asset_bytes)
            asset_relative = "assets/document/page.png"
            asset_path = dataset_root / asset_relative
            asset_path.parent.mkdir(parents=True)
            asset_path.write_bytes(asset_bytes)
            asset_record = {
                "path": asset_relative,
                "sha256": asset_digest,
                "size_bytes": len(asset_bytes),
            }
            assets_digest = _semantic_digest([asset_record])
            asset_locks = (
                DatasetAssetLock(
                    "datasets/sft/locked/assets/document/page.png",
                    asset_digest,
                    len(asset_bytes),
                ),
            )
            row = {
                "messages": [
                    {"content": "hello", "role": "user"},
                    {"content": "world", "role": "assistant"},
                ],
                "media": [
                    {
                        "kind": "image",
                        "path": asset_relative,
                        "sha256": asset_digest,
                        "mime_type": "image/png",
                        "metadata": {},
                    }
                ],
            }
            data = (json.dumps(row, sort_keys=True) + "\n").encode()
        else:
            data = b'{"messages":[{"content":"hello","role":"user"},{"content":"world","role":"assistant"}]}\n'
        data_digest = _digest(data)
        (dataset_root / "data.jsonl").write_bytes(data)
        dataset_manifest = {
            "schema_version": 1,
            "selection_id": "datasets/runtime@1",
            "selection_revision": "1",
            "dataset_revision": "1",
            "source_kind": "fixture",
            "content_sha256": data_digest,
            "examples": 1,
            "data": "data.jsonl",
        }
        if with_dataset_assets:
            assert asset_record is not None
            dataset_manifest["assets"] = [asset_record]
            dataset_manifest["assets_digest"] = assets_digest
        (dataset_root / "manifest.json").write_text(
            json.dumps(dataset_manifest),
            encoding="utf-8",
        )
        dataset_locks = (
            DatasetPackageLock(
                seat_name="dataset",
                selection_id="datasets/runtime@1",
                selection_revision="1",
                dataset_revision="1",
                kind="supervised",
                schema_version=1,
                digest=data_digest,
                package_path="datasets/sft/locked/data.jsonl",
                manifest_path="datasets/sft/locked/manifest.json",
                size_bytes=len(data),
                num_records=1,
                assets=asset_locks,
                assets_digest=assets_digest,
            ),
        )
    manifest = JobPackageManifest(
        project_id="runtime-tests",
        work_package_id="screen/runtime",
        job_id="check",
        job_definition_id="data/runtime-check@1",
        job_kind="data.prepare",
        resolved_inputs_digest=_semantic_digest(resolved_inputs),
        framework_source_digest=_tree_digest(job / "sources/framework"),
        project_source_digest=_tree_digest(job / "sources/project"),
        runtime_dependencies_digest=_digest(runtime_lock),
        code_requirements_digest=_digest(code_lock),
        resolved_config_digest=_digest(resolved_bytes),
        project_config_digest=_project_config_digest(
            job / "config/project",
            project_manifest=project_manifest.resolve(),
            selected_work_package=work_package.resolve(),
        ),
        universal_image=RuntimeImageRef(f"registry.lan/posttrain/base@sha256:{'a' * 64}"),
        kind_image=RuntimeImageRef(f"registry.lan/posttrain/data@sha256:{'b' * 64}"),
        runtime_variant="supervised",
        datasets=dataset_locks,
    )
    manifest_path = job / "package.json"
    manifest_path.write_bytes(manifest.to_bytes())
    return manifest_path, manifest


def _launch(
    manifest: JobPackageManifest,
    *,
    run_id: str = "run-runtime-1",
    target_id: str = "targets/local-cuda-8gb",
    memory_gb: int = 8,
) -> str:
    return json.dumps(
        {
            "schema": "posttrain.execution-launch.v1",
            "run": {
                "run_id": run_id,
                "project_id": manifest.project_id,
                "work_package_id": manifest.work_package_id,
                "stage": "screen",
                "job_kind": manifest.job_kind,
                "job_definition_id": manifest.job_definition_id,
            },
            "attempt": 2,
            "provider": "local-docker",
            "job_image": (f"registry.lan/posttrain/jobs@sha256:{'c' * 64}"),
            "target": {
                "id": target_id,
                "revision": "1",
                "device_class": "nvidia-cuda",
                "memory_gb": memory_gb,
                "placement": {
                    "world_size": 1,
                    **({"instances": ["remote.lan"]} if target_id == "targets/remote-24gb" else {}),
                },
                "host_constraints": {},
            },
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def test_worker_executes_verified_actual_job_with_launch_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _actual_job(tmp_path / "job")
    seen: list[str] = []
    source_metadata: list[dict[str, object]] = []
    run_root = (tmp_path / "runs").resolve()
    monkeypatch.setenv("POSTTRAIN_EXECUTION", _launch(manifest))
    monkeypatch.setattr("posttrain_runtime.execute._RUN_ROOT", run_root)
    monkeypatch.setattr(
        "posttrain_runtime.execute.build_job_runtime",
        lambda request, tracking: _runtime(request.catalog, seen, source_metadata),
    )

    result = execute_manifest(manifest_path)

    assert result.run_id == "run-runtime-1"
    assert result.status == "succeeded"
    assert seen == ["run-runtime-1"]
    workspace = run_root / "run-runtime-1"
    assert workspace.is_dir()
    marker = json.loads((workspace / ".posttrain-terminal.json").read_text(encoding="utf-8"))
    assert marker == {
        "schema": "posttrain.worker-terminal.v1",
        "run_id": "run-runtime-1",
        "status": "succeeded",
        "finished_at": marker["finished_at"],
        "attempt": 2,
        "provider": "local-docker",
        "job_image": f"registry.lan/posttrain/jobs@sha256:{'c' * 64}",
    }
    assert datetime.fromisoformat(marker["finished_at"]).tzinfo is not None
    assert not tuple(workspace.glob(".posttrain-terminal.json.*.tmp"))
    execution = source_metadata[0]["execution"]
    assert isinstance(execution, dict)
    assert execution["provider"] == "local-docker"
    assert execution["attempt"] == 2
    assert execution["job_image"].endswith("c" * 64)
    package = source_metadata[0]["job_package"]
    assert isinstance(package, dict)
    assert package["package_key"] == manifest.package_key
    assert package["framework_source_digest"] == (manifest.framework_source_digest)


@pytest.mark.parametrize(
    "cancel_signal",
    [signal.SIGTERM, signal.SIGINT],
    ids=["sigterm", "sigint"],
)
def test_worker_cancel_signal_durably_cancels_tracking_before_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel_signal: int,
) -> None:
    """Container runtimes cancel with SIGTERM; the dstack runner interrupts the
    job with SIGINT. Both must finalize tracking as cancelled before exit."""

    manifest_path, manifest = _actual_job(tmp_path / "job")
    backend = _TrackingBackend(repeat_signal=cancel_signal)
    previous = signal.getsignal(cancel_signal)
    monkeypatch.setenv("POSTTRAIN_EXECUTION", _launch(manifest))
    monkeypatch.setattr(
        "posttrain_runtime.execute._RUN_ROOT",
        (tmp_path / "runs").resolve(),
    )

    def build(request, tracking):
        del tracking
        runtime = _runtime(
            request.catalog,
            [],
            terminate=True,
            terminate_signal=cancel_signal,
        )
        return replace(
            runtime,
            executor=partial(
                execute_run_tracked_finalized,
                backend=backend,
                scratch_root=request.state_dir / "scratch",
            ),
        )

    monkeypatch.setattr(
        "posttrain_runtime.execute.build_job_runtime",
        build,
    )

    with pytest.raises(SystemExit) as captured:
        execute_manifest(manifest_path)

    assert captured.value.code == 128 + cancel_signal
    assert signal.getsignal(cancel_signal) == previous
    assert backend.tracked is not None
    assert [outcome.status for outcome in backend.tracked.outcomes] == ["cancelled"]
    marker = json.loads((tmp_path / "runs" / "run-runtime-1" / ".posttrain-terminal.json").read_text(encoding="utf-8"))
    assert marker["status"] == "cancelled"


def test_worker_failure_writes_terminal_marker_after_unwind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _actual_job(tmp_path / "job")
    run_root = (tmp_path / "runs").resolve()
    monkeypatch.setenv("POSTTRAIN_EXECUTION", _launch(manifest))
    monkeypatch.setattr("posttrain_runtime.execute._RUN_ROOT", run_root)
    monkeypatch.setattr(
        "posttrain_runtime.execute.build_job_runtime",
        lambda request, tracking: _runtime(request.catalog, [], fail=True),
    )

    with pytest.raises(RuntimeError, match="expected worker failure"):
        execute_manifest(manifest_path)

    marker = json.loads((run_root / "run-runtime-1" / ".posttrain-terminal.json").read_text(encoding="utf-8"))
    assert marker["schema"] == "posttrain.worker-terminal.v1"
    assert marker["run_id"] == "run-runtime-1"
    assert marker["status"] == "failed"


def test_worker_requires_remote_trackio_before_runtime_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _actual_job(
        tmp_path / "job",
        tracking="trackio",
    )
    events: list[tuple[str, object]] = []
    monkeypatch.setenv("POSTTRAIN_EXECUTION", _launch(manifest))
    monkeypatch.setenv(
        "POSTTRAIN_TRACKIO_SERVER_URL",
        "https://trackio.example",
    )
    monkeypatch.setattr(
        "posttrain_runtime.execute._RUN_ROOT",
        (tmp_path / "runs").resolve(),
    )

    def preflight(*, project: str, server_url: str | None) -> None:
        events.append(("preflight", (project, server_url)))
        raise ContractError("remote tracking unavailable")

    def build(request, tracking):
        events.append(("build", tracking))
        return _runtime(request.catalog, [])

    monkeypatch.setattr(
        "posttrain_tracking_trackio.require_remote_trackio_ready",
        preflight,
    )
    monkeypatch.setattr(
        "posttrain_runtime.execute.build_job_runtime",
        build,
    )

    with pytest.raises(ContractError, match="remote tracking unavailable"):
        execute_manifest(manifest_path)

    assert events == [
        (
            "preflight",
            ("runtime-tests", "https://trackio.example"),
        )
    ]


def test_worker_skips_remote_trackio_gate_for_explicit_offline_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _actual_job(tmp_path / "job")
    seen: list[str] = []
    monkeypatch.setenv("POSTTRAIN_EXECUTION", _launch(manifest))
    monkeypatch.setattr(
        "posttrain_runtime.execute._RUN_ROOT",
        (tmp_path / "runs").resolve(),
    )
    monkeypatch.setattr(
        "posttrain_tracking_trackio.require_remote_trackio_ready",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
    )
    monkeypatch.setattr(
        "posttrain_runtime.execute.build_job_runtime",
        lambda request, tracking: _runtime(request.catalog, seen),
    )

    result = execute_manifest(manifest_path)

    assert result.status == "succeeded"
    assert seen == ["run-runtime-1"]


def test_worker_reconstructs_the_packaged_target_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _actual_job(
        tmp_path / "job",
        target_override=True,
    )
    seen: list[str] = []
    monkeypatch.setenv(
        "POSTTRAIN_EXECUTION",
        _launch(
            manifest,
            target_id="targets/remote-24gb",
            memory_gb=24,
        ),
    )
    monkeypatch.setattr(
        "posttrain_runtime.execute._RUN_ROOT",
        (tmp_path / "runs").resolve(),
    )
    monkeypatch.setattr(
        "posttrain_runtime.execute.build_job_runtime",
        lambda request, tracking: _runtime(request.catalog, seen),
    )

    result = execute_manifest(manifest_path)

    assert result.status == "succeeded"
    assert seen == ["run-runtime-1"]


def test_worker_loads_the_verified_packaged_dataset_without_rematerializing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _actual_job(
        tmp_path / "job",
        with_dataset=True,
    )
    seen: list[str] = []
    monkeypatch.setenv("POSTTRAIN_EXECUTION", _launch(manifest))
    monkeypatch.setattr(
        "posttrain_runtime.execute._RUN_ROOT",
        (tmp_path / "runs").resolve(),
    )
    monkeypatch.setattr(
        "posttrain_runtime.execute.build_job_runtime",
        lambda request, tracking: _runtime(
            request.catalog,
            seen,
            with_dataset=True,
        ),
    )

    result = execute_manifest(manifest_path)

    assert result.status == "succeeded"
    assert seen == ["run-runtime-1"]


def test_worker_loads_verified_visual_dataset_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path, manifest = _actual_job(
        tmp_path / "job",
        with_dataset=True,
        with_dataset_assets=True,
    )
    seen: list[str] = []
    monkeypatch.setenv("POSTTRAIN_EXECUTION", _launch(manifest))
    monkeypatch.setattr("posttrain_runtime.execute._RUN_ROOT", (tmp_path / "runs").resolve())
    monkeypatch.setattr(
        "posttrain_runtime.execute.build_job_runtime",
        lambda request, tracking: _runtime(request.catalog, seen, with_dataset=True),
    )

    result = execute_manifest(manifest_path)

    assert result.status == "succeeded"
    assert seen == ["run-runtime-1"]


@pytest.mark.parametrize("operation", ["modify", "remove", "add"])
def test_worker_rejects_visual_dataset_asset_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    manifest_path, manifest = _actual_job(
        tmp_path / "job",
        with_dataset=True,
        with_dataset_assets=True,
    )
    monkeypatch.setenv("POSTTRAIN_EXECUTION", _launch(manifest))
    asset = manifest_path.parent / "datasets/sft/locked/assets/document/page.png"
    if operation == "modify":
        asset.write_bytes(b"tampered")
    elif operation == "remove":
        asset.unlink()
    else:
        (asset.parent / "additional.png").write_bytes(b"additional")

    with pytest.raises(ContractError, match="dataset (files differ|asset differs)"):
        execute_manifest(manifest_path)


@pytest.mark.parametrize(
    ("relative", "message"),
    [
        (
            "sources/framework/runtime/__init__.py",
            "packaged framework code differs",
        ),
        (
            "config/project/.posttrain/catalog/layer.yaml",
            "packaged project configuration differs",
        ),
        (
            "config/resolved.json",
            "resolved job config differs",
        ),
    ],
)
def test_worker_rejects_package_drift_before_runtime_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
    message: str,
) -> None:
    manifest_path, manifest = _actual_job(tmp_path / "job")
    monkeypatch.setenv("POSTTRAIN_EXECUTION", _launch(manifest))
    manifest_path.parent.joinpath(relative).write_text("tampered\n", encoding="utf-8")
    called = False

    def build(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(
        "posttrain_runtime.execute.build_job_runtime",
        build,
    )

    with pytest.raises(ContractError, match=message):
        execute_manifest(manifest_path)

    assert called is False


def test_worker_rejects_launch_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _actual_job(tmp_path / "job")
    payload = json.loads(_launch(manifest))
    payload["run"]["job_kind"] = "train.sft"
    monkeypatch.setenv("POSTTRAIN_EXECUTION", json.dumps(payload))

    with pytest.raises(ContractError, match="job kind conflicts"):
        execute_manifest(manifest_path)


def test_worker_rejects_dataset_bytes_that_differ_from_the_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _actual_job(tmp_path / "job")
    dataset_root = manifest_path.parent / "datasets/sft/locked"
    dataset_root.mkdir(parents=True)
    data = b'{"messages":[{"role":"user","content":"hello"}]}\n'
    digest = _digest(data)
    data_path = dataset_root / "data.jsonl"
    data_path.write_bytes(data)
    dataset_manifest = {
        "schema_version": 1,
        "selection_id": "datasets/runtime@1",
        "selection_revision": "1",
        "dataset_revision": "locked",
        "source_kind": "jsonl",
        "content_sha256": digest,
        "examples": 1,
        "data": "data.jsonl",
    }
    (dataset_root / "manifest.json").write_text(
        json.dumps(dataset_manifest),
        encoding="utf-8",
    )
    lock = DatasetPackageLock(
        seat_name="dataset",
        selection_id="datasets/runtime@1",
        selection_revision="1",
        dataset_revision="locked",
        kind="supervised",
        schema_version=1,
        digest=digest,
        package_path="datasets/sft/locked/data.jsonl",
        manifest_path="datasets/sft/locked/manifest.json",
        size_bytes=len(data),
        num_records=1,
    )
    manifest = replace(manifest, datasets=(lock,))
    manifest_path.write_bytes(manifest.to_bytes())
    monkeypatch.setenv("POSTTRAIN_EXECUTION", _launch(manifest))
    data_path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ContractError, match="packaged dataset differs"):
        execute_manifest(manifest_path)


def test_worker_rejects_environment_wheel_bytes_that_differ_from_the_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _actual_job(tmp_path / "job")
    contents = b"wheel bytes"
    wheel = manifest_path.parent / "wheels/environments/runtime_env.whl"
    wheel.write_bytes(contents)
    lock = EnvironmentPackageLock(
        package="runtime-env",
        repository="https://github.com/CarbonTeq/runtime-env",
        revision="1" * 40,
        subdirectory=".",
        tree_digest="2" * 64,
        wheel_filename=wheel.name,
        wheel_digest=_digest(contents),
        wheel_size_bytes=len(contents),
    )
    manifest = replace(manifest, environment_packages=(lock,))
    manifest_path.write_bytes(manifest.to_bytes())
    monkeypatch.setenv("POSTTRAIN_EXECUTION", _launch(manifest))
    wheel.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ContractError, match="environment wheel differs"):
        execute_manifest(manifest_path)


def test_worker_requires_launch_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _ = _actual_job(tmp_path / "job")
    monkeypatch.delenv("POSTTRAIN_EXECUTION", raising=False)

    with pytest.raises(ContractError, match="missing POSTTRAIN_EXECUTION"):
        execute_manifest(manifest_path)


def test_worker_rejects_an_unknown_worker_contract_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _actual_job(tmp_path / "job")
    unsupported = replace(manifest, worker_contract_version="2")
    manifest_path.write_bytes(unsupported.to_bytes())
    monkeypatch.setenv("POSTTRAIN_EXECUTION", _launch(unsupported))

    with pytest.raises(ContractError, match="worker contract version"):
        execute_manifest(manifest_path)
