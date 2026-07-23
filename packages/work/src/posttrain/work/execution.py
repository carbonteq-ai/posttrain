"""Execute one canonical job run locally or through a tracking backend."""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from posttrain.common import LocalArtifactRef, NullObserver, Observer, RunContext
from posttrain.tracking import (
    ArtifactInput,
    RunError,
    RunOutcome,
    RunOutcomeStatus,
    RunSpec,
    TrackingBackend,
)

type RunOperation[ResultT] = Callable[[RunContext], ResultT]
type ArtifactMaterializer = Callable[[Mapping[str, ArtifactInput], Path], Mapping[str, LocalArtifactRef]]


def execute_run[ResultT](
    spec: RunSpec,
    operation: RunOperation[ResultT],
    *,
    observer: Observer | None = None,
    scratch_root: Path | None = None,
    materialize: ArtifactMaterializer | None = None,
) -> ResultT:
    """Execute one canonical job-kind run in an ephemeral workspace."""

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
    except (KeyboardInterrupt, SystemExit):
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


__all__ = ["ArtifactInput", "RunSpec", "execute_run", "execute_run_tracked"]
