"""Execute one canonical job run locally or through a tracking backend."""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping
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
    TrackingBackend,
)

type RunOperation[ResultT] = Callable[[RunContext], ResultT]
type RunFinalizer[ResultT] = Callable[[RunContext, ResultT], None]
type ArtifactMaterializer = Callable[[Mapping[str, ArtifactInput], Path], Mapping[str, LocalArtifactRef]]


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
) -> ResultT:
    """Execute one canonical job-kind run in an ephemeral workspace."""

    if scratch_root is not None:
        scratch_root.mkdir(parents=True, exist_ok=True)
    directory = str(scratch_root) if scratch_root is not None else None
    with tempfile.TemporaryDirectory(prefix="posttrain-", dir=directory) as workspace:
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
) -> ResultT:
    """Execute and durably finalize one run through the selected backend.

    ``success_status`` permits operations that completed normally but produced a
    canonical partial or unsupported result. Exceptions are always re-raised
    after their terminal outcome has been made durable.
    """

    if success_status == "failed" or success_status == "cancelled":
        raise ValueError("failed and cancelled statuses are derived from exceptional exits")

    started_at = datetime.now(UTC)
    tracked = backend.start_run(spec)

    def materialize(inputs: Mapping[str, ArtifactInput], workspace: Path) -> Mapping[str, LocalArtifactRef]:
        return tracked.materialize_inputs(inputs, workspace / "inputs")

    try:
        result = execute_run(
            spec,
            operation,
            observer=tracked,
            scratch_root=scratch_root,
            materialize=materialize,
        )
    except (KeyboardInterrupt, SystemExit, OperationCancelled):
        tracked.finish(RunOutcome("cancelled", started_at, datetime.now(UTC)))
        raise
    except BaseException as error:
        tracked.finish(
            RunOutcome(
                "failed",
                started_at,
                datetime.now(UTC),
                RunError(type(error).__name__, "operation failed; inspect run logs"),
            )
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
) -> FinalizedRunResult[ResultT]:
    """Execute a tracked run and resolve every committed output before cleanup."""

    if success_status == "failed" or success_status == "cancelled":
        raise ValueError("failed and cancelled statuses are derived from exceptional exits")

    started_at = datetime.now(UTC)
    tracked = backend.start_run(spec)
    published: tuple[PublishedArtifact, ...] = ()

    def materialize(inputs: Mapping[str, ArtifactInput], workspace: Path) -> Mapping[str, LocalArtifactRef]:
        return tracked.materialize_inputs(inputs, workspace / "inputs")

    def finalize(context: RunContext, result: ResultT) -> None:
        del context, result
        nonlocal published
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
        )
    except (KeyboardInterrupt, SystemExit, OperationCancelled):
        tracked.finish(RunOutcome("cancelled", started_at, datetime.now(UTC)))
        raise
    except BaseException as error:
        tracked.finish(
            RunOutcome(
                "failed",
                started_at,
                datetime.now(UTC),
                RunError(type(error).__name__, "operation failed; inspect run logs"),
            )
        )
        raise
    tracked.finish(RunOutcome(success_status, started_at, datetime.now(UTC)))
    return FinalizedRunResult(result, published)


__all__ = [
    "ArtifactInput",
    "FinalizedRunResult",
    "RunSpec",
    "execute_run",
    "execute_run_tracked",
    "execute_run_tracked_finalized",
]
