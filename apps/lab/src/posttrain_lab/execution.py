"""Execute one job action locally or as a Trackio-observed run attempt."""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import trackio
from posttrain.common import (
    ExecutionContext,
    Invocation,
    Job,
    JobAction,
    JsonValue,
    NullObserver,
    Observer,
    RunAttempt,
)

from .tracking import TrackioObserver

type Operation[ResultT] = Callable[[ExecutionContext], ResultT]


@dataclass(frozen=True, slots=True)
class AttemptSpec:
    job: Job
    action: JobAction
    invocation: Invocation = field(default_factory=Invocation.new)
    attempt: RunAttempt = field(default_factory=RunAttempt.new)
    inputs: Mapping[str, JsonValue] = field(default_factory=dict)
    source_metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action.job_id != self.job.id:
            raise ValueError("action must belong to the attempt job")


def execute[ResultT](
    spec: AttemptSpec,
    operation: Operation[ResultT],
    *,
    observer: Observer | None = None,
    scratch_root: Path | None = None,
) -> ResultT:
    """Execute with ephemeral scratch space and no durable local run directory."""

    directory = str(scratch_root) if scratch_root is not None else None
    with tempfile.TemporaryDirectory(prefix="posttrain-", dir=directory) as workspace:
        context = ExecutionContext(
            job=spec.job,
            action=spec.action,
            invocation=spec.invocation,
            attempt=spec.attempt,
            workspace=Path(workspace).resolve(),
            observer=observer or NullObserver(),
            source_metadata=spec.source_metadata,
        )
        context.event("operation_started", {"action_kind": spec.action.kind})
        try:
            result = operation(context)
        except BaseException as error:
            context.event(
                "operation_failed",
                {"error_type": type(error).__name__, "error": str(error)},
            )
            raise
        context.event("operation_completed", {"action_kind": spec.action.kind})
        return result


def _run_config(spec: AttemptSpec) -> dict[str, JsonValue]:
    return {
        "schema_version": 1,
        "job_id": spec.job.id,
        "job_version": spec.job.version,
        "action_id": spec.action.id,
        "action_kind": spec.action.kind,
        "invocation_id": spec.invocation.id,
        "attempt_id": spec.attempt.id,
        "attempt_number": spec.attempt.number,
        "resolved_inputs": dict(spec.inputs),
        "source": dict(spec.source_metadata),
    }


def execute_tracked[ResultT](
    spec: AttemptSpec,
    operation: Operation[ResultT],
    *,
    project: str,
    scratch_root: Path | None = None,
    auto_log_gpu: bool = True,
    auto_log_cpu: bool = True,
) -> ResultT:
    """Create exactly one Trackio run for one attempt and finalize it once."""

    name = f"{spec.action.id}-{spec.invocation.id[:8]}-a{spec.attempt.number}"
    run = trackio.init(
        project=project,
        name=name,
        group=spec.job.id,
        config=_run_config(spec),
        embed=False,
        auto_log_gpu=auto_log_gpu,
        auto_log_cpu=auto_log_cpu,
    )
    observer = TrackioObserver(run)
    try:
        result = execute(spec, operation, observer=observer, scratch_root=scratch_root)
    except BaseException as error:
        run.log({"run/status": "failed", "run/error_type": type(error).__name__})
        raise
    else:
        run.log({"run/status": "complete", "run/success": 1})
        return result
    finally:
        run.finish()
