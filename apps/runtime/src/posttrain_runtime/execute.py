"""Verify and execute one immutable actual-job OCI package."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import FrameType
from typing import cast

from posttrain.catalog import FamilyRegistryLock, load_project_layout, open_catalog
from posttrain.common import (
    Catalog,
    CatalogRef,
    ContractError,
    ExecutionTarget,
    JsonValue,
    OperationCancelled,
)
from posttrain.data import (
    DatasetLoadPlan,
    DatasetMaterialization,
    load_materialized_dataset,
    validate_materialized_dataset,
)
from posttrain.environment import (
    EnvironmentBinding,
    PythonFactoryActivation,
    VerifiersV1ConfigActivation,
)
from posttrain.eval import EvaluationPlan
from posttrain.execution import (
    EXECUTION_LAUNCH_ENVIRONMENT,
    DatasetPackageLock,
    JobPackageManifest,
    RuntimeImageRef,
    resolved_inputs_digest,
)
from posttrain.jobs import build_job_runtime
from posttrain.work import (
    JobRuntime,
    ProjectEntry,
    ProjectExecutionRequest,
    ResolvedSeat,
    WorkPackageContext,
    load_project_brief,
    load_work_package,
    override_job_execution_target,
    prepare_work_package_job,
    run_work_package_job,
)

_RESOLVED_SCHEMA = "posttrain.resolved-job.v1"
_LAUNCH_SCHEMA = "posttrain.execution-launch.v1"
_RUN_ROOT = Path("/opt/posttrain/run")
_PROJECT_ROOT = PurePosixPath("project")
_RESOLVED_FIELDS = {
    "schema",
    "project_id",
    "work_package_id",
    "job_id",
    "job_definition_id",
    "runtime_variant",
    "project_root",
    "project_manifest",
    "selected_work_package",
    "resolved_inputs",
}
_OPTIONAL_RESOLVED_FIELDS = {"family_registry_lock"}
_LAUNCH_FIELDS = {
    "schema",
    "run",
    "attempt",
    "provider",
    "job_image",
    "target",
}
_LAUNCH_RUN_FIELDS = {
    "run_id",
    "project_id",
    "work_package_id",
    "stage",
    "job_kind",
    "job_definition_id",
}
_TERMINAL_MARKER = ".posttrain-terminal.json"
_TERMINAL_SCHEMA = "posttrain.worker-terminal.v1"


@dataclass(frozen=True, slots=True)
class WorkerExecutionResult:
    run_id: str
    project_id: str
    work_package_id: str
    job_id: str
    job_definition_id: str
    status: str
    published_artifact_roles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ExecutionLaunch:
    run_id: str
    project_id: str
    work_package_id: str
    stage: str
    job_kind: str
    job_definition_id: str
    attempt: int
    provider: str
    job_image: RuntimeImageRef
    target: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class _VerifiedPackage:
    root: Path
    manifest: JobPackageManifest
    resolved_inputs: Mapping[str, JsonValue]
    family_registry_lock: Mapping[str, JsonValue]
    project_root: Path
    project_manifest: Path
    selected_work_package: Path
    datasets: Mapping[str, tuple[DatasetPackageLock, Mapping[str, object]]]


@dataclass(frozen=True, slots=True)
class QualificationResult:
    """Offline package qualification facts emitted before image publication."""

    environment_ids: tuple[str, ...]
    dataset_seats: tuple[str, ...]


def execute_manifest(path: Path) -> WorkerExecutionResult:
    """Run exactly the verified registered job embedded in an actual-job image."""

    with _graceful_cancellation():
        return _execute_manifest(path)


def qualify_manifest(
    path: Path,
    *,
    timeout_seconds: float = 60.0,
    allow_deferred: bool = False,
) -> QualificationResult:
    """Verify every staged activation and taskset without creating a run."""

    if timeout_seconds <= 0:
        raise ValueError("qualification timeout must be positive")
    package = _verify_package(path)
    qualified: list[str] = []
    with tempfile.TemporaryDirectory(prefix="posttrain-qualify-") as temporary:
        previous_tmpdir = os.environ.get("TMPDIR")
        os.environ["TMPDIR"] = temporary
        try:
            for lock in package.manifest.environment_activations:
                if lock.qualification == "deferred" and not allow_deferred:
                    raise ContractError(
                        f"environment {lock.environment_id!r} defers qualification; "
                        "production packaging requires a taskset load"
                    )
                with _qualification_timeout(timeout_seconds, lock.environment_id):
                    _qualify_activation(lock, package.root)
                qualified.append(lock.environment_id)
        finally:
            if previous_tmpdir is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = previous_tmpdir
    return QualificationResult(tuple(qualified), tuple(sorted(package.datasets)))


def _execute_manifest(path: Path) -> WorkerExecutionResult:
    package = _verify_package(path)
    launch = _load_launch()
    _verify_launch_identity(package.manifest, launch)

    layout = load_project_layout(package.project_root)
    if layout.manifest != package.project_manifest:
        raise ContractError("resolved job project path conflicts with the packaged project")
    if layout.project_id != package.manifest.project_id:
        raise ContractError("job package project id conflicts with the packaged project")

    run_root = _run_workspace(launch.run_id)
    with _terminal_workspace(run_root, launch):
        layout = replace(layout, state=run_root)
        catalog = open_catalog(
            scope=layout.project_id,
            overlays=layout.catalog_overlays,
            catalog_root=layout.base_catalog,
            required_plugin_distributions=layout.catalog_plugin_requirements,
        )
        if package.family_registry_lock and (
            not isinstance(catalog.family_registry_lock, FamilyRegistryLock)
            or catalog.family_registry_lock.to_payload() != package.family_registry_lock
        ):
            raise ContractError("installed catalog family registry differs from the packaged registry lock")
        if not package.selected_work_package.is_relative_to(layout.work_packages):
            raise ContractError("resolved job work package is outside the project work-package directory")
        work_package = load_work_package(package.selected_work_package)
        if work_package.work_package_id != package.manifest.work_package_id:
            raise ContractError("job package work-package id conflicts with the packaged project")

        request = ProjectExecutionRequest(
            project_id=layout.project_id,
            project_root=layout.root,
            state_dir=layout.state,
            work_package_path=package.selected_work_package,
            catalog=catalog,
            project_brief=(load_project_brief(layout.project_brief) if layout.project_brief is not None else None),
        )
        runtime = _build_runtime(request, layout.entry, layout.tracking)
        runtime = _configure_runtime(runtime, package, launch)
        selected_target = _resolve_launch_target(catalog, launch.target)
        work_package = override_job_execution_target(
            runtime,
            work_package,
            package.manifest.job_id,
            selected_target,
            allow_unchanged=True,
        )
        prepared = prepare_work_package_job(
            runtime,
            work_package,
            package.manifest.job_id,
            run_id=launch.run_id,
        )
        _verify_prepared_job(package, prepared, launch)

        result = run_work_package_job(
            runtime,
            work_package,
            package.manifest.job_id,
            run_id=launch.run_id,
        )
        job = result.jobs[0]
        return WorkerExecutionResult(
            run_id=launch.run_id,
            project_id=package.manifest.project_id,
            work_package_id=package.manifest.work_package_id,
            job_id=package.manifest.job_id,
            job_definition_id=package.manifest.job_definition_id,
            status=job.status,
            published_artifact_roles=tuple(
                artifact.role for artifact in job.published_artifacts if artifact.role is not None
            ),
        )


@contextmanager
def _terminal_workspace(
    run_root: Path,
    launch: _ExecutionLaunch,
) -> Iterator[None]:
    """Write one durable terminal marker only after run finalization unwinds."""

    run_root.mkdir(parents=True, exist_ok=True)
    try:
        yield
    except (KeyboardInterrupt, SystemExit, OperationCancelled) as error:
        _write_terminal_marker_preserving_error(
            run_root,
            launch,
            "cancelled",
            error,
        )
        raise
    except BaseException as error:
        _write_terminal_marker_preserving_error(
            run_root,
            launch,
            "failed",
            error,
        )
        raise
    _write_terminal_marker(run_root, launch, "succeeded")


def _write_terminal_marker_preserving_error(
    run_root: Path,
    launch: _ExecutionLaunch,
    status: str,
    error: BaseException,
) -> None:
    try:
        _write_terminal_marker(run_root, launch, status)
    except BaseException as marker_error:
        error.add_note(f"post-training terminal workspace marker failed: {type(marker_error).__name__}")


def _write_terminal_marker(
    run_root: Path,
    launch: _ExecutionLaunch,
    status: str,
) -> None:
    if status not in {"succeeded", "failed", "cancelled"}:
        raise ContractError("worker terminal marker status is invalid")
    marker = run_root / _TERMINAL_MARKER
    temporary = run_root / f"{_TERMINAL_MARKER}.{uuid.uuid4().hex}.tmp"
    payload = {
        "schema": _TERMINAL_SCHEMA,
        "run_id": launch.run_id,
        "status": status,
        "finished_at": datetime.now(UTC).isoformat(),
        "attempt": launch.attempt,
        "provider": launch.provider,
        "job_image": launch.job_image.value,
    }
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, marker)
        directory = os.open(run_root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


_CANCELLATION_SIGNALS = (signal.SIGTERM, signal.SIGINT)


@contextmanager
def _graceful_cancellation() -> Iterator[None]:
    """Let tracked execution durably finalize before a worker exits on cancel.

    Container runtimes cancel with SIGTERM, but the dstack runner interrupts the
    job with SIGINT (``os.Interrupt``), so both must reach the same finalizing
    path. SIGINT is handled explicitly rather than left to Python's default
    handler because backends initialized inside this scope, such as vLLM and
    Torch, may install their own SIGINT handler and silently swallow the
    cancellation.

    Raising ``SystemExit`` transfers control through the tracked-run cancellation
    path while preserving the conventional ``128 + signal`` process exit code.
    A repeated signal is ignored during unwinding so it cannot interrupt the
    bounded tracking finalizer; the provider may still enforce its hard-kill
    timeout with SIGKILL.
    """

    previous = {number: signal.getsignal(number) for number in _CANCELLATION_SIGNALS}
    terminating = False

    def request_termination(
        signum: int,
        frame: FrameType | None,
    ) -> None:
        del frame
        nonlocal terminating
        if terminating:
            return
        terminating = True
        raise SystemExit(128 + signum)

    for number in _CANCELLATION_SIGNALS:
        signal.signal(number, request_termination)
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


@contextmanager
def _qualification_timeout(seconds: float, environment_id: str) -> Iterator[None]:
    """Bound one offline taskset load without creating a subprocess tree."""

    if not hasattr(signal, "setitimer") or not hasattr(signal, "SIGALRM"):
        raise RuntimeError("offline qualification requires POSIX interval timers")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)

    def expire(_signum: int, _frame: FrameType | None) -> None:
        raise TimeoutError(f"environment qualification timed out after {seconds:g}s: {environment_id}")

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _verify_package(path: Path) -> _VerifiedPackage:
    manifest_path = path.resolve()
    if path.is_symlink() or manifest_path.name != "package.json" or not manifest_path.is_file():
        raise ContractError(f"job package manifest is not a regular package.json: {path}")
    root = manifest_path.parent
    manifest_bytes = manifest_path.read_bytes()
    manifest = JobPackageManifest.from_bytes(manifest_bytes)
    if manifest.worker_contract_version != "1":
        raise ContractError("job package worker contract version is unsupported")

    _verify_digest_file(
        root / "locks" / "runtime.requirements.txt",
        manifest.runtime_dependencies_digest,
        "runtime dependency lock",
    )
    for lock in manifest.runtime_dependency_locks:
        _verify_digest_file(
            root.joinpath(*PurePosixPath(lock.requirements_path).parts),
            lock.requirements_digest,
            f"{lock.role} runtime dependency lock",
        )
    _verify_digest_file(
        root / "locks" / "code.requirements.txt",
        manifest.code_requirements_digest,
        "code requirements lock",
    )
    resolved_path = root / "config" / "resolved.json"
    resolved_bytes = _verify_digest_file(
        resolved_path,
        manifest.resolved_config_digest,
        "resolved job config",
    )
    if _staged_framework_digest(root) != manifest.framework_source_digest:
        raise ContractError("packaged framework code differs from its manifest digest")
    if _tree_digest(root / "sources" / "project") != (manifest.project_source_digest):
        raise ContractError("packaged project source differs from its manifest digest")

    resolved = _load_json_object(resolved_bytes, "resolved job config")
    if resolved.get("schema") != _RESOLVED_SCHEMA:
        raise ContractError("resolved job config schema is unsupported")
    if unknown := sorted(set(resolved) - _RESOLVED_FIELDS - _OPTIONAL_RESOLVED_FIELDS):
        raise ContractError("resolved job config has unknown fields: " + ", ".join(unknown))
    if missing := sorted(_RESOLVED_FIELDS - set(resolved)):
        raise ContractError("resolved job config is missing fields: " + ", ".join(missing))
    for field, expected in (
        ("project_id", manifest.project_id),
        ("work_package_id", manifest.work_package_id),
        ("job_id", manifest.job_id),
        ("job_definition_id", manifest.job_definition_id),
        ("runtime_variant", manifest.runtime_variant),
    ):
        if resolved.get(field) != expected:
            raise ContractError(f"resolved job {field} conflicts with the package manifest")
    if resolved.get("project_root") != _PROJECT_ROOT.as_posix():
        raise ContractError("resolved job project root must be config/project")
    resolved_inputs = resolved.get("resolved_inputs")
    if not isinstance(resolved_inputs, dict):
        raise ContractError("resolved job inputs must be a JSON object")
    typed_resolved_inputs = cast(Mapping[str, JsonValue], resolved_inputs)
    if resolved_inputs_digest(typed_resolved_inputs) != (manifest.resolved_inputs_digest):
        raise ContractError("resolved job inputs differ from the package manifest")
    family_registry_lock = resolved.get("family_registry_lock", {})
    if not isinstance(family_registry_lock, dict):
        raise ContractError("resolved job family registry lock must be an object")

    config_root = root / "config"
    project_root = config_root / _PROJECT_ROOT.as_posix()
    project_manifest = _resolved_path(
        config_root,
        resolved.get("project_manifest"),
        "project manifest",
        prefix=_PROJECT_ROOT,
    )
    selected_work_package = _resolved_path(
        config_root,
        resolved.get("selected_work_package"),
        "selected work package",
        prefix=_PROJECT_ROOT,
    )
    observed_project_digest = _project_config_digest(
        project_root,
        project_manifest=project_manifest,
        selected_work_package=selected_work_package,
    )
    if observed_project_digest != manifest.project_config_digest:
        raise ContractError("packaged project configuration differs from its manifest digest")

    _verify_environment_wheels(root, manifest)
    _verify_activation_resources(root, manifest)
    _verify_backend_runtime(root, manifest)
    datasets = _verify_datasets(root, manifest)
    return _VerifiedPackage(
        root=root,
        manifest=manifest,
        resolved_inputs=typed_resolved_inputs,
        family_registry_lock=cast(Mapping[str, JsonValue], family_registry_lock),
        project_root=project_root.resolve(),
        project_manifest=project_manifest,
        selected_work_package=selected_work_package,
        datasets=datasets,
    )


def _verify_backend_runtime(root: Path, manifest: JobPackageManifest) -> None:
    backend = manifest.backend_runtime
    if backend is None:
        return
    if root != Path("/opt/posttrain/job"):
        raise ContractError("veRL package must execute from its capsule job root")
    locks = {lock.role: lock for lock in manifest.runtime_dependency_locks}
    control = locks["control"]
    worker = locks["backend"]
    control_python = Path(control.python_executable)
    # uv/venv interpreters are often symlinks to a managed CPython. Compare the
    # resolved targets so a capsule shebang like .../bin/python3 still matches
    # the locked .../bin/python path.
    if Path(sys.executable).resolve() != control_python.resolve():
        raise ContractError("veRL control process uses a non-capsule interpreter")
    backend_python = Path(worker.python_executable)
    if not backend_python.exists() or not os.access(backend_python, os.X_OK) or not backend_python.resolve().is_file():
        raise ContractError("veRL backend interpreter is not executable")
    dependency_lock = Path(backend.dependency_lock_path)
    _verify_digest_file(
        dependency_lock,
        backend.dependency_lock_digest,
        "veRL kind dependency lock",
    )
    projection = Path(backend.projection_path)
    if _tree_digest(projection) != backend.projection_digest:
        raise ContractError("veRL worker projection differs from its manifest")
    worktree = Path(backend.working_directory)
    try:
        head = subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(worktree), "status", "--porcelain=v1"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ContractError("veRL capsule worktree cannot be verified") from error
    if head != backend.source_revision or status:
        raise ContractError("veRL capsule worktree differs from its manifest")


def _load_launch() -> _ExecutionLaunch:
    try:
        raw = os.environ[EXECUTION_LAUNCH_ENVIRONMENT]
    except KeyError as error:
        raise ContractError(f"execution environment is missing {EXECUTION_LAUNCH_ENVIRONMENT}") from error
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ContractError("execution launch envelope is invalid JSON") from error
    if not isinstance(payload, dict) or payload.get("schema") != _LAUNCH_SCHEMA:
        raise ContractError("execution launch envelope schema is unsupported")
    if set(payload) != _LAUNCH_FIELDS:
        raise ContractError("execution launch envelope fields are invalid")
    run = payload.get("run")
    target = payload.get("target")
    if not isinstance(run, dict) or set(run) != _LAUNCH_RUN_FIELDS:
        raise ContractError("execution launch run identity is invalid")
    if not isinstance(target, dict):
        raise ContractError("execution launch target is invalid")
    attempt = payload.get("attempt")
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise ContractError("execution launch attempt must be positive")
    provider = _required_string(payload.get("provider"), "launch provider")
    if not provider.strip():
        raise ContractError("execution launch provider cannot be empty")
    try:
        image = RuntimeImageRef(_required_string(payload.get("job_image"), "launch job image"))
    except ContractError as error:
        raise ContractError("execution launch job image is invalid") from error
    return _ExecutionLaunch(
        run_id=_required_string(run.get("run_id"), "launch run id"),
        project_id=_required_string(run.get("project_id"), "launch project id"),
        work_package_id=_required_string(run.get("work_package_id"), "launch work-package id"),
        stage=_required_string(run.get("stage"), "launch stage"),
        job_kind=_required_string(run.get("job_kind"), "launch job kind"),
        job_definition_id=_required_string(run.get("job_definition_id"), "launch job definition"),
        attempt=attempt,
        provider=provider,
        job_image=image,
        target=cast(Mapping[str, JsonValue], target),
    )


def _verify_launch_identity(
    manifest: JobPackageManifest,
    launch: _ExecutionLaunch,
) -> None:
    for label, observed, expected in (
        ("project", launch.project_id, manifest.project_id),
        ("work package", launch.work_package_id, manifest.work_package_id),
        ("job kind", launch.job_kind, manifest.job_kind),
        (
            "job definition",
            launch.job_definition_id,
            manifest.job_definition_id,
        ),
    ):
        if observed != expected:
            raise ContractError(f"execution launch {label} conflicts with the job package")


def _build_runtime(
    request: ProjectExecutionRequest,
    entry: str | None,
    tracking: str,
) -> JobRuntime:
    if tracking == "trackio":
        from posttrain_tracking_trackio import require_remote_trackio_ready

        require_remote_trackio_ready(
            project=os.getenv("POSTTRAIN_TRACKIO_PROJECT", request.project_id),
            server_url=os.getenv("POSTTRAIN_TRACKIO_SERVER_URL"),
        )
    if entry is None:
        return build_job_runtime(request, tracking=tracking)
    factory = _load_project_entry(entry)
    runtime = factory(request)
    if not isinstance(runtime, WorkPackageContext):
        raise ContractError(f"project entry {entry!r} must return a JobRuntime")
    if runtime.catalog is not request.catalog:
        raise ContractError("project entry must use the catalog supplied in ProjectExecutionRequest")
    if runtime.project_brief is None and request.project_brief is not None:
        return replace(runtime, project_brief=request.project_brief)
    if runtime.project_brief != request.project_brief:
        raise ContractError("project entry project brief conflicts with the packaged project")
    return runtime


def _load_project_entry(spec: str) -> ProjectEntry:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute or ":" in attribute:
        raise ContractError("project entry must use MODULE:CALLABLE syntax")
    try:
        resolved: object = importlib.import_module(module_name)
    except (ImportError, ValueError) as error:
        raise ContractError(f"cannot import project entry module {module_name!r}: {error}") from error
    for part in attribute.split("."):
        try:
            resolved = getattr(resolved, part)
        except AttributeError as error:
            raise ContractError(f"project entry module has no callable {attribute!r}") from error
    if not callable(resolved):
        raise ContractError(f"project entry {spec!r} is not callable")
    return cast(ProjectEntry, resolved)


def _configure_runtime(
    runtime: JobRuntime,
    package: _VerifiedPackage,
    launch: _ExecutionLaunch,
) -> JobRuntime:
    upstream_resolver = runtime.seat_resolver

    def resolve_seat(seat: ResolvedSeat):
        value = seat.value
        if isinstance(value, DatasetLoadPlan):
            try:
                lock, dataset_manifest = package.datasets[seat.name]
            except KeyError as error:
                raise ContractError(f"job package has no dataset for seat {seat.name!r}") from error
            _verify_dataset_selection(value, lock, dataset_manifest)
            examples = dataset_manifest.get("examples")
            if not isinstance(examples, int) or isinstance(examples, bool) or examples < 1:
                raise ContractError(f"packaged dataset {seat.name!r} has invalid example count")
            return load_materialized_dataset(
                value,
                DatasetMaterialization(
                    selection_id=lock.selection_id,
                    selection_revision=lock.selection_revision,
                    source_kind=value.source_kind,
                    path=package.root.joinpath(*PurePosixPath(lock.package_path).parts),
                    manifest_path=package.root.joinpath(*PurePosixPath(lock.manifest_path).parts),
                    content_sha256=lock.digest,
                    examples=examples,
                    created=False,
                ),
            )
        if isinstance(value, EnvironmentBinding):
            return _resolve_environment_resources(value, package)
        if isinstance(value, EvaluationPlan):
            return replace(
                value,
                environments=tuple(_resolve_environment_resources(item, package) for item in value.environments),
            )
        if upstream_resolver is not None:
            return upstream_resolver(seat)
        return value

    manifest = package.manifest
    package_evidence = {
        "schema": "posttrain.job-package-evidence.v1",
        "package_key": manifest.package_key,
        "universal_image": manifest.universal_image.value,
        "kind_image": manifest.kind_image.value,
        "resolved_inputs_digest": manifest.resolved_inputs_digest,
        "framework_source_digest": manifest.framework_source_digest,
        "project_source_digest": manifest.project_source_digest,
        "project_config_digest": manifest.project_config_digest,
        "runtime_dependencies_digest": manifest.runtime_dependencies_digest,
        "code_requirements_digest": manifest.code_requirements_digest,
        "resolved_config_digest": manifest.resolved_config_digest,
        "environment_packages": [item.to_payload() for item in manifest.environment_packages],
        "environment_activations": [item.to_payload() for item in manifest.environment_activations],
        "datasets": [item.to_payload() for item in manifest.datasets],
    }
    return replace(
        runtime,
        source_metadata={
            **dict(runtime.source_metadata),
            "execution": {
                "schema": _LAUNCH_SCHEMA,
                "provider": launch.provider,
                "attempt": launch.attempt,
                "job_image": launch.job_image.value,
                "target": dict(launch.target),
                "worker": _worker_context(),
            },
            "job_package": package_evidence,
        },
        seat_resolver=resolve_seat,
    )


def _verify_prepared_job(
    package: _VerifiedPackage,
    prepared: object,
    launch: _ExecutionLaunch,
) -> None:
    # Keep this helper structurally narrow while preserving the public prepared
    # job type at the call site.
    from posttrain.work import PreparedWorkPackageJob

    if not isinstance(prepared, PreparedWorkPackageJob):
        raise TypeError("prepared job")
    manifest = package.manifest
    if prepared.spec.run_id != launch.run_id:
        raise ContractError("registered job run id conflicts with the execution launch")
    if prepared.spec.stage != launch.stage:
        raise ContractError("registered job stage conflicts with the execution launch")
    if prepared.recipe_job.kind != manifest.job_kind:
        raise ContractError("job package kind conflicts with the registered work-package job")
    if prepared.definition.id != manifest.job_definition_id:
        raise ContractError("job package definition conflicts with the registered work-package job")
    if resolved_inputs_digest(prepared.spec.resolved_inputs) != (manifest.resolved_inputs_digest):
        raise ContractError("registered job resolved selections differ from the job package")
    if tuple(sorted(prepared.definition.required_artifact_roles)) != (manifest.expected_artifact_roles):
        raise ContractError("job package expected artifacts conflict with the registered job")
    _verify_environment_selections(prepared, manifest)
    _verify_execution_target(prepared, launch.target)
    selected_datasets = {
        name: seat.value for name, seat in prepared.resolved.seats.items() if isinstance(seat.value, DatasetLoadPlan)
    }
    if set(selected_datasets) != set(package.datasets):
        raise ContractError("job package datasets differ from the registered job selections")
    for name, plan in selected_datasets.items():
        lock, dataset_manifest = package.datasets[name]
        _verify_dataset_selection(plan, lock, dataset_manifest)


def _verify_execution_target(
    prepared: object,
    launch_target: Mapping[str, JsonValue],
) -> None:
    from posttrain.work import PreparedWorkPackageJob

    if not isinstance(prepared, PreparedWorkPackageJob):
        raise TypeError("prepared job")
    expected_fields = {
        "id",
        "revision",
        "device_class",
        "memory_gb",
        "placement",
        "host_constraints",
    }
    if set(launch_target) != expected_fields:
        raise ContractError("execution launch target fields are invalid")
    direct = [value for value in prepared.seats.values() if isinstance(value, ExecutionTarget)]
    training = [
        target
        for name, value in prepared.seats.items()
        if name == "training"
        and isinstance(
            (target := getattr(value, "target", None)),
            ExecutionTarget,
        )
    ]
    candidates = training or direct
    if not candidates:
        candidates = [
            target
            for value in prepared.seats.values()
            if isinstance(
                (target := getattr(value, "target", None)),
                ExecutionTarget,
            )
        ]
    unique: list[ExecutionTarget] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    if len(unique) != 1:
        raise ContractError("registered job has no unambiguous primary execution target")
    selected = unique[0]
    expected = _execution_target_payload(selected)
    if dict(launch_target) != expected:
        raise ContractError("execution launch target conflicts with the registered job")


def _resolve_launch_target(
    catalog: Catalog,
    launch_target: Mapping[str, JsonValue],
) -> ExecutionTarget:
    target_id = _required_string(
        launch_target.get("id"),
        "launch execution target id",
    )
    resolved = catalog.resolve(CatalogRef("target", target_id)).value
    if not isinstance(resolved, ExecutionTarget):
        raise ContractError("execution launch target did not resolve to an ExecutionTarget")
    if dict(launch_target) != _execution_target_payload(resolved):
        raise ContractError("execution launch target differs from its packaged catalog selection")
    return resolved


def _execution_target_payload(target: ExecutionTarget) -> dict[str, JsonValue]:
    return {
        "id": target.id,
        "revision": target.revision,
        "device_class": target.device_class,
        "memory_gb": target.memory_gb,
        "placement": dict(target.placement),
        "host_constraints": dict(target.host_constraints),
    }


def _verify_environment_selections(
    prepared: object,
    manifest: JobPackageManifest,
) -> None:
    from posttrain.work import PreparedWorkPackageJob

    if not isinstance(prepared, PreparedWorkPackageJob):
        raise TypeError("prepared job")
    selected: dict[str, EnvironmentBinding] = {}
    for seat in prepared.resolved.seats.values():
        value = seat.value
        bindings: tuple[EnvironmentBinding, ...]
        if isinstance(value, EnvironmentBinding):
            bindings = (value,)
        elif isinstance(value, EvaluationPlan):
            bindings = value.environments
        else:
            continue
        for binding in bindings:
            previous = selected.get(binding.id)
            if previous is not None and previous != binding:
                raise ContractError(f"registered job selects conflicting environment {binding.id!r}")
            selected[binding.id] = binding

    locks = {item.environment_id: item for item in manifest.environment_activations}
    if set(selected) != set(locks):
        raise ContractError("job package environment activations differ from registered selections")
    package_locks = {item.package: item for item in manifest.environment_packages}
    selected_packages: set[str] = set()
    for environment_id, binding in selected.items():
        activation = locks[environment_id]
        source = binding.source
        selected_packages.add(source.package)
        package_lock = package_locks.get(source.package)
        if package_lock is None:
            raise ContractError(f"job package omits environment package {source.package!r}")
        if (
            package_lock.repository != source.repository
            or package_lock.revision != source.revision
            or package_lock.subdirectory != (source.subdirectory or ".")
        ):
            raise ContractError(f"job package environment source differs for {environment_id!r}")
        if (
            activation.package != source.package
            or activation.kind != binding.activation.kind
            or activation.digest != binding.activation.digest
        ):
            raise ContractError(f"job package environment activation differs for {environment_id!r}")
    if selected_packages != set(package_locks):
        raise ContractError("job package environment wheels differ from registered selections")


def _verify_dataset_selection(
    plan: DatasetLoadPlan,
    lock: DatasetPackageLock,
    dataset_manifest: Mapping[str, object],
) -> None:
    if (
        plan.id != lock.selection_id
        or plan.revision != lock.selection_revision
        or plan.dataset_revision != lock.dataset_revision
        or plan.kind != lock.kind
    ):
        raise ContractError(f"packaged dataset differs from selection for seat {lock.seat_name!r}")
    if dataset_manifest.get("source_kind") != plan.source_kind:
        raise ContractError(f"packaged dataset source differs for seat {lock.seat_name!r}")


def _verify_environment_wheels(
    root: Path,
    manifest: JobPackageManifest,
) -> None:
    directory = root / "wheels" / "environments"
    if not directory.is_dir() or directory.is_symlink():
        raise ContractError("job package environment wheel directory is missing")
    expected = {lock.wheel_filename for lock in manifest.environment_packages}
    observed: set[str] = set()
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ContractError("job package environment wheels must be regular files")
        observed.add(path.name)
    if observed != expected:
        raise ContractError("job package environment wheel files differ from the manifest")
    for lock in manifest.environment_packages:
        path = directory / lock.wheel_filename
        contents = path.read_bytes()
        if len(contents) != lock.wheel_size_bytes or _bytes_digest(contents) != lock.wheel_digest:
            raise ContractError(f"environment wheel differs from its lock: {lock.package}")


def _verify_activation_resources(root: Path, manifest: JobPackageManifest) -> None:
    expected = {
        resource.staged_path: resource
        for activation in manifest.environment_activations
        for resource in activation.resources.values()
    }
    resource_root = root / "environment-resources"
    if not expected and not resource_root.exists():
        return
    if not resource_root.is_dir() or resource_root.is_symlink():
        raise ContractError("job package activation resource directory is missing")
    observed: set[str] = set()
    for path in resource_root.rglob("*"):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise ContractError("job package activation resources contain a symlink or special file")
        if path.is_file():
            observed.add(path.relative_to(root).as_posix())
    if observed != set(expected):
        raise ContractError("job package activation resources differ from the manifest")
    for staged_path, resource in expected.items():
        path = _package_path(
            root,
            staged_path,
            f"activation resource {resource.name}",
            prefix="environment-resources",
        )
        contents = path.read_bytes()
        if len(contents) != resource.size_bytes or _bytes_digest(contents) != resource.digest:
            raise ContractError(f"activation resource differs from its lock: {resource.name}")


def _resolve_environment_resources(
    binding: EnvironmentBinding,
    package: _VerifiedPackage,
) -> EnvironmentBinding:
    activation = binding.activation
    if not isinstance(activation, VerifiersV1ConfigActivation):
        return binding
    locks = {item.environment_id: item for item in package.manifest.environment_activations}
    try:
        lock = locks[binding.id]
    except KeyError as error:
        raise ContractError(f"job package has no activation lock for environment {binding.id!r}") from error
    resolved = _resolve_activation_value(activation.config, root=package.root, resources=lock.resources)
    if not isinstance(resolved, Mapping):
        raise AssertionError("environment activation config must remain an object")
    return replace(
        binding,
        activation=VerifiersV1ConfigActivation(cast(Mapping[str, JsonValue], resolved)),
    )


def _qualify_activation(lock: object, root: Path) -> None:
    """Construct one installed Verifiers environment and load its taskset offline."""

    kind = getattr(lock, "kind", None)
    if kind == "verifiers-config":
        config = getattr(lock, "config", None)
        resources = getattr(lock, "resources", None)
        if not isinstance(config, Mapping) or not isinstance(resources, Mapping):
            raise ContractError("environment activation lock is invalid")
        resolved = _resolve_activation_value(config, root=root, resources=resources)
        if not isinstance(resolved, Mapping):
            raise ContractError("environment activation config must remain an object")
        native = VerifiersV1ConfigActivation(cast(Mapping[str, JsonValue], resolved)).activate()
    elif kind == "python-factory":
        reference = getattr(lock, "reference", None)
        if not isinstance(reference, str):
            raise ContractError("environment factory activation lock is invalid")
        native = PythonFactoryActivation(reference).activate()
    else:
        raise ContractError("environment activation kind is unsupported")
    try:
        from verifiers.v1.env import EnvConfig, Environment  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install the Verifiers integration dependencies") from error
    environment = native if isinstance(native, Environment) else Environment(native)
    if not isinstance(getattr(environment, "config", native), EnvConfig):
        raise ContractError("environment activation did not produce a Verifiers EnvConfig")
    taskset = getattr(environment, "taskset", None)
    load = getattr(taskset, "load", None)
    if not callable(load):
        raise ContractError("Verifiers environment has no loadable taskset")
    load()


def _resolve_activation_value(
    value: object,
    *,
    root: Path,
    resources: Mapping[str, object],
) -> object:
    if isinstance(value, Mapping):
        if "$resource" in value:
            if set(value) != {"$resource"} or not isinstance(value["$resource"], str):
                raise ContractError("activation resource references must be exactly { $resource: NAME }")
            name = value["$resource"]
            try:
                resource = resources[name]
            except KeyError as error:
                raise ContractError(f"activation references undeclared resource {name!r}") from error
            staged_path = getattr(resource, "staged_path", None)
            if not isinstance(staged_path, str):
                raise ContractError(f"activation resource lock is invalid: {name}")
            return str(_package_path(root, staged_path, f"activation resource {name}", prefix="environment-resources"))
        return {
            str(key): _resolve_activation_value(child, root=root, resources=resources)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_resolve_activation_value(child, root=root, resources=resources) for child in value]
    if isinstance(value, tuple):
        return tuple(_resolve_activation_value(child, root=root, resources=resources) for child in value)
    return value


def _verify_datasets(
    root: Path,
    manifest: JobPackageManifest,
) -> Mapping[str, tuple[DatasetPackageLock, Mapping[str, object]]]:
    dataset_root = root / "datasets"
    if not dataset_root.is_dir() or dataset_root.is_symlink():
        raise ContractError("job package dataset directory is missing")
    expected_files = {relative for lock in manifest.datasets for relative in (lock.package_path, lock.manifest_path)}
    observed_files: set[str] = set()
    for path in dataset_root.rglob("*"):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise ContractError("job package datasets contain a symlink or special file")
        if path.is_file():
            observed_files.add(path.relative_to(root).as_posix())
    if observed_files != expected_files:
        raise ContractError("job package dataset files differ from the manifest")
    verified: dict[
        str,
        tuple[DatasetPackageLock, Mapping[str, object]],
    ] = {}
    for lock in manifest.datasets:
        data_path = _package_path(root, lock.package_path, "dataset data", prefix="datasets")
        dataset_manifest_path = _package_path(
            root,
            lock.manifest_path,
            "dataset manifest",
            prefix="datasets",
        )
        contents = data_path.read_bytes()
        if len(contents) != lock.size_bytes or _bytes_digest(contents) != lock.digest:
            raise ContractError(f"packaged dataset differs from its lock: {lock.seat_name}")
        dataset_manifest = _load_json_object(
            dataset_manifest_path.read_bytes(),
            f"dataset manifest {lock.seat_name}",
        )
        expected = {
            "selection_id": lock.selection_id,
            "selection_revision": lock.selection_revision,
            "dataset_revision": lock.dataset_revision,
            "content_sha256": lock.digest,
            "data": data_path.name,
            "schema_version": lock.schema_version,
        }
        for field, value in expected.items():
            if dataset_manifest.get(field) != value:
                raise ContractError(f"packaged dataset manifest has invalid {field}: {lock.seat_name}")
        examples = dataset_manifest.get("examples")
        if (
            not isinstance(examples, int)
            or isinstance(examples, bool)
            or examples < 1
            or (lock.num_records is not None and examples != lock.num_records)
        ):
            raise ContractError(f"packaged dataset manifest has invalid examples: {lock.seat_name}")
        if validate_materialized_dataset(data_path) != examples:
            raise ContractError(f"packaged dataset record count differs from its lock: {lock.seat_name}")
        verified[lock.seat_name] = (lock, dataset_manifest)
    return verified


def _verify_digest_file(
    path: Path,
    expected: str,
    label: str,
) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{label} is not a regular file")
    contents = path.read_bytes()
    if _bytes_digest(contents) != expected:
        raise ContractError(f"{label} differs from its manifest digest")
    return contents


def _project_config_digest(
    project_root: Path,
    *,
    project_manifest: Path,
    selected_work_package: Path,
) -> str:
    if not project_root.is_dir() or project_root.is_symlink():
        raise ContractError("packaged project configuration is missing")
    entries: list[dict[str, JsonValue]] = []
    for path in sorted(project_root.rglob("*")):
        if path.is_symlink():
            raise ContractError("packaged project configuration cannot contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ContractError("packaged project configuration contains a special file")
        entries.append(
            {
                "path": path.relative_to(project_root).as_posix(),
                "sha256": _bytes_digest(path.read_bytes()),
                "size_bytes": path.stat().st_size,
            }
        )
    payload = {
        "files": entries,
        "project_manifest": project_manifest.relative_to(project_root).as_posix(),
        "selected_work_package": selected_work_package.relative_to(project_root).as_posix(),
    }
    return _semantic_digest(payload)


def _tree_digest(root: Path) -> str:
    if not root.is_dir() or root.is_symlink():
        raise ContractError("packaged source root must be a regular directory")
    entries: list[dict[str, JsonValue]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        if path.is_symlink():
            raise ContractError(f"packaged source contains a symlink: {relative}")
        if stat.S_ISDIR(mode):
            entries.append({"path": relative, "type": "directory"})
        elif stat.S_ISREG(mode):
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "sha256": _bytes_digest(path.read_bytes()),
                    "executable": bool(mode & 0o111),
                }
            )
        else:
            raise ContractError(f"packaged source contains a special file: {relative}")
    if not entries:
        raise ContractError("packaged source tree cannot be empty")
    return _semantic_digest({"entries": entries})


def _resolved_path(
    root: Path,
    value: object,
    label: str,
    *,
    prefix: PurePosixPath,
) -> Path:
    if not isinstance(value, str):
        raise ContractError(f"resolved job {label} path must be a string")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or any(part in {"", ".", ".."} for part in relative.parts)
        or not relative.is_relative_to(prefix)
    ):
        raise ContractError(f"resolved job {label} must stay below config/{prefix}")
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"resolved job {label} is missing")
    return path.resolve()


def _package_path(
    root: Path,
    value: str,
    label: str,
    *,
    prefix: str,
) -> Path:
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or not relative.is_relative_to(PurePosixPath(prefix))
    ):
        raise ContractError(f"{label} must stay below {prefix}/")
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{label} is not a regular file")
    return path


def _load_json_object(contents: bytes, label: str) -> dict[str, object]:
    try:
        payload = json.loads(contents)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{label} is invalid JSON") from error
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must be a JSON object")
    return payload


def _semantic_digest(value: object) -> str:
    return _bytes_digest(
        json.dumps(
            value,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )


def _staged_framework_digest(root: Path) -> str:
    """Digest whichever way framework code was packaged.

    A checkout is copied in as a source tree; an installed framework is
    packaged as the built distributions it was installed from. The two are
    alternatives, so a package carrying both has no single identity and is
    rejected rather than resolved by preference.
    """
    wheels = tuple(sorted(path for path in (root / "wheels" / "framework").glob("*.whl") if path.is_file()))
    source = root / "sources" / "framework"
    has_source = source.is_dir() and any(path.is_file() for path in source.rglob("*"))
    if wheels and has_source:
        raise ContractError("packaged framework code is both a source tree and built distributions")
    if wheels:
        entries = [{"name": wheel.name, "sha256": _bytes_digest(wheel.read_bytes())} for wheel in wheels]
        return _semantic_digest(entries)
    if has_source:
        return _tree_digest(source)
    raise ContractError("package contains no framework code")


def _bytes_digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a non-empty string")
    return value


def _worker_context() -> dict[str, JsonValue]:
    context: dict[str, JsonValue] = {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return context
    if result.returncode != 0:
        return context
    devices: list[JsonValue] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            memory_mib = int(parts[1])
        except ValueError:
            continue
        devices.append(
            {
                "name": parts[0],
                "memory_bytes": memory_mib * 1024 * 1024,
                "driver_version": parts[2],
            }
        )
    if devices:
        context["gpu_devices"] = devices
    return context


def _run_workspace(run_id: str) -> Path:
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ContractError("execution launch run id is invalid")
    root = _RUN_ROOT.resolve()
    workspace = (root / run_id).resolve()
    if workspace.parent != root:
        raise ContractError("execution run workspace escapes its mount root")
    return workspace


__all__ = ["WorkerExecutionResult", "execute_manifest"]
