"""Execute one canonical job run locally or through a tracking backend."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

from posttrain.common import (
    ContractError,
    LocalArtifactRef,
    NullObserver,
    Observer,
    OperationCancelled,
    PublishedArtifact,
    RunContext,
)
from posttrain.tracking import (
    ArtifactInput,
    RunError,
    RunOutcome,
    RunOutcomeStatus,
    RunSpec,
    TrackedRun,
    TrackingBackend,
)

type RunOperation[ResultT] = Callable[[RunContext], ResultT]
type RunFinalizer[ResultT] = Callable[[RunContext, ResultT], None]
type RunFailureFinalizer = Callable[[RunContext, BaseException], None]
type ArtifactMaterializer = Callable[[Mapping[str, ArtifactInput], Path], Mapping[str, LocalArtifactRef]]

_WORKSPACE_MARKER = ".posttrain-recovery.json"


@dataclass(frozen=True, slots=True)
class FinalizedRunResult[ResultT]:
    """An operation value plus provider-committed output identities."""

    value: ResultT
    published_artifacts: tuple[PublishedArtifact, ...] = ()


class _PublishedArtifactSource(Protocol):
    def published_artifacts(self) -> tuple[PublishedArtifact, ...]: ...


def execute_run[ResultT](
    spec: RunSpec,
    operation: RunOperation[ResultT],
    *,
    observer: Observer | None = None,
    scratch_root: Path | None = None,
    materialize: ArtifactMaterializer | None = None,
    finalize: RunFinalizer[ResultT] | None = None,
    finalize_failure: RunFailureFinalizer | None = None,
    recoverable: bool = False,
) -> ResultT:
    """Execute one canonical job-kind run in an ephemeral or retained workspace."""

    if scratch_root is not None:
        scratch_root.mkdir(parents=True, exist_ok=True)
    if recoverable:
        workspace_path, _ = _prepare_recoverable_workspace(spec, scratch_root)
        workspace_scope = nullcontext(str(workspace_path))
    else:
        directory = str(scratch_root) if scratch_root is not None else None
        workspace_scope = tempfile.TemporaryDirectory(prefix="posttrain-", dir=directory)
    with workspace_scope as workspace:
        workspace_path = Path(workspace).resolve()
        if spec.artifacts and materialize is None:
            raise RuntimeError("this run requires an artifact materializer")
        inputs = materialize(spec.artifacts, workspace_path) if materialize is not None else {}
        context = RunContext(
            project_id=spec.project_id,
            work_package_id=spec.work_package_id,
            run_id=spec.run_id,
            job_kind=spec.job_kind,
            job_definition_version=spec.job_definition_version,
            workspace=workspace_path,
            observer=observer or NullObserver(),
            source_metadata={
                **dict(spec.source_metadata),
                "resolved_selections": dict(spec.resolved_inputs),
            },
            input_artifacts=inputs,
        )
        context.event("operation_started", {"resolved_selections": dict(spec.resolved_inputs)})
        try:
            with context.phase("operation"):
                result = operation(context)
        except BaseException as error:
            context.event("operation_failed", {"error_type": type(error).__name__})
            if finalize_failure is not None:
                try:
                    finalize_failure(context, error)
                except BaseException as finalization_error:
                    error.add_note(
                        "failure artifact finalization also failed: "
                        f"{type(finalization_error).__name__}: {finalization_error}"
                    )
                    context.event(
                        "failure_artifact_finalization_failed",
                        {"error_type": type(finalization_error).__name__},
                    )
                else:
                    context.event("failure_artifact_finalization_completed")
            raise
        context.event("operation_completed")
        if finalize is not None:
            try:
                finalize(context, result)
            except BaseException as error:
                context.event(
                    "artifact_finalization_failed",
                    {"error_type": type(error).__name__},
                )
                raise
            context.event("artifact_finalization_completed")
        return result


def execute_run_tracked[ResultT](
    spec: RunSpec,
    operation: RunOperation[ResultT],
    *,
    backend: TrackingBackend,
    scratch_root: Path | None = None,
    success_status: RunOutcomeStatus = "succeeded",
    recoverable: bool = False,
) -> ResultT:
    """Execute and durably finalize one run through the selected backend.

    ``success_status`` permits operations that completed normally but produced a
    canonical partial or unsupported result. Exceptions are always re-raised
    after their terminal outcome has been made durable.
    """

    if success_status == "failed" or success_status == "cancelled":
        raise ValueError("failed and cancelled statuses are derived from exceptional exits")

    started_at, tracked = _start_tracked_run(backend, spec, scratch_root, recoverable)

    def materialize(inputs: Mapping[str, ArtifactInput], workspace: Path) -> Mapping[str, LocalArtifactRef]:
        return tracked.materialize_inputs(inputs, workspace / "inputs")

    def flush_artifacts(context: RunContext, value: object) -> None:
        del context, value
        flusher = getattr(tracked, "flush_artifacts", None)
        if callable(flusher):
            flusher(timeout=120)

    try:
        result = execute_run(
            spec,
            operation,
            observer=tracked,
            scratch_root=scratch_root,
            materialize=materialize,
            finalize=flush_artifacts,
            finalize_failure=flush_artifacts,
            recoverable=recoverable,
        )
    except (KeyboardInterrupt, SystemExit, OperationCancelled) as error:
        try:
            tracked.finish(RunOutcome("cancelled", started_at, datetime.now(UTC)))
        except BaseException as finalization_error:
            error.add_note(
                f"tracking cancellation finalization also failed: {type(finalization_error).__name__}: "
                f"{finalization_error}"
            )
        raise
    except BaseException as error:
        try:
            tracked.finish(
                RunOutcome(
                    "failed",
                    started_at,
                    datetime.now(UTC),
                    RunError(type(error).__name__, "operation failed; inspect run logs"),
                )
            )
        except BaseException as finalization_error:
            error.add_note(
                f"tracking failure finalization also failed: {type(finalization_error).__name__}: {finalization_error}"
            )
        raise
    tracked.finish(RunOutcome(success_status, started_at, datetime.now(UTC)))
    return result


def execute_run_tracked_finalized[ResultT](
    spec: RunSpec,
    operation: RunOperation[ResultT],
    *,
    backend: TrackingBackend,
    scratch_root: Path | None = None,
    success_status: RunOutcomeStatus = "succeeded",
    recoverable: bool = False,
) -> FinalizedRunResult[ResultT]:
    """Execute a tracked run and resolve every committed output before cleanup."""

    if success_status == "failed" or success_status == "cancelled":
        raise ValueError("failed and cancelled statuses are derived from exceptional exits")

    started_at, tracked = _start_tracked_run(backend, spec, scratch_root, recoverable)
    published: tuple[PublishedArtifact, ...] = ()

    def materialize(inputs: Mapping[str, ArtifactInput], workspace: Path) -> Mapping[str, LocalArtifactRef]:
        return tracked.materialize_inputs(inputs, workspace / "inputs")

    def finalize_failure(context: RunContext, error: BaseException) -> None:
        del context, error
        flusher = getattr(tracked, "flush_artifacts", None)
        if callable(flusher):
            flusher(timeout=120)

    def finalize(context: RunContext, result: ResultT) -> None:
        del context, result
        nonlocal published
        flusher = getattr(tracked, "flush_artifacts", None)
        if callable(flusher):
            flusher(timeout=120)
        resolver = getattr(tracked, "published_artifacts", None)
        if not callable(resolver):
            raise ContractError("tracking backend cannot resolve committed output artifacts")
        resolved = tuple(cast(_PublishedArtifactSource, tracked).published_artifacts())
        names: set[str] = set()
        for artifact in resolved:
            if artifact.logical_name in names:
                raise ContractError(f"run published duplicate logical artifact name: {artifact.logical_name}")
            names.add(artifact.logical_name)
            if artifact.required and artifact.reference.digest is None:
                raise ContractError(f"required artifact has no committed digest: {artifact.logical_name}")
        for role in spec.required_artifact_roles:
            matches = tuple(artifact for artifact in resolved if artifact.role == role)
            if len(matches) != 1:
                raise ContractError(f"required artifact role {role!r} resolved {len(matches)} outputs; expected 1")
            if matches[0].reference.digest is None:
                raise ContractError(f"required artifact role has no committed digest: {role}")
        published = resolved

    try:
        result = execute_run(
            spec,
            operation,
            observer=tracked,
            scratch_root=scratch_root,
            materialize=materialize,
            finalize=finalize,
            finalize_failure=finalize_failure,
            recoverable=recoverable,
        )
    except (KeyboardInterrupt, SystemExit, OperationCancelled) as error:
        try:
            tracked.finish(RunOutcome("cancelled", started_at, datetime.now(UTC)))
        except BaseException as finalization_error:
            error.add_note(
                f"tracking cancellation finalization also failed: {type(finalization_error).__name__}: "
                f"{finalization_error}"
            )
        raise
    except BaseException as error:
        try:
            tracked.finish(
                RunOutcome(
                    "failed",
                    started_at,
                    datetime.now(UTC),
                    RunError(type(error).__name__, "operation failed; inspect run logs"),
                )
            )
        except BaseException as finalization_error:
            error.add_note(
                f"tracking failure finalization also failed: {type(finalization_error).__name__}: {finalization_error}"
            )
        raise
    tracked.finish(RunOutcome(success_status, started_at, datetime.now(UTC)))
    return FinalizedRunResult(result, published)


def _prepare_recoverable_workspace(spec: RunSpec, scratch_root: Path | None) -> tuple[Path, datetime]:
    if scratch_root is None:
        raise ContractError("recoverable execution requires a persistent scratch root")
    root = scratch_root.resolve()
    workspace = (root / spec.run_id).resolve()
    if workspace.parent != root:
        raise ContractError("recoverable workspace escapes its scratch root")
    marker = workspace / _WORKSPACE_MARKER
    if marker.exists():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("run_id") != spec.run_id or payload.get("job_kind") != spec.job_kind:
            raise ContractError("recoverable workspace identity differs from the requested run")
        started_at = datetime.fromisoformat(str(payload["started_at"]))
        if started_at.tzinfo is None:
            raise ContractError("recoverable workspace start time must be timezone-aware")
        return workspace, started_at
    workspace.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(UTC)
    payload = {
        "schema": "posttrain.recoverable-workspace.v1",
        "run_id": spec.run_id,
        "job_kind": spec.job_kind,
        "started_at": started_at.isoformat(),
    }
    temporary = marker.with_suffix(f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, marker)
    return workspace, started_at


def _start_tracked_run(
    backend: TrackingBackend,
    spec: RunSpec,
    scratch_root: Path | None,
    recoverable: bool,
) -> tuple[datetime, TrackedRun]:
    if not recoverable:
        return datetime.now(UTC), backend.start_run(spec)
    _, started_at = _prepare_recoverable_workspace(spec, scratch_root)
    opener = getattr(backend, "start_or_resume_run", None)
    if not callable(opener):
        raise ContractError("recoverable execution requires an idempotent tracking backend")
    return started_at, cast(TrackedRun, opener(spec, started_at=started_at))


__all__ = [
    "ArtifactInput",
    "FinalizedRunResult",
    "RunSpec",
    "execute_run",
    "execute_run_tracked",
    "execute_run_tracked_finalized",
]
