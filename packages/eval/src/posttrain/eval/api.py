"""Public endpoint-neutral evaluation operation."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

from posttrain.common import LocalArtifactRef, ProducedArtifact, RunContext

from .backends.verifiers import VerifiersRunResult, run_verifiers
from .requests import EvaluateRequest
from .results import EvaluationResult, TraceSynchronization

type EvaluationContext = RunContext
type EvaluationRunner = Callable[[EvaluationContext, EvaluateRequest, Path], VerifiersRunResult]


def _directory_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with child.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def evaluate(
    context: EvaluationContext,
    request: EvaluateRequest,
    *,
    runner: EvaluationRunner = run_verifiers,
) -> EvaluationResult:
    """Evaluate one model/environment cell and retain its native result bundle."""

    environment = request.environment
    num_tasks, num_rollouts, max_concurrent = request.resolved_budget
    attributes = _attributes(request)
    attributes.update(
        {
            "environment_id": environment.id,
            "environment_category": environment.category,
            "reasoning_mode": request.resolved_reasoning_mode,
            "num_tasks": num_tasks,
            "num_rollouts": num_rollouts,
            "max_concurrent": max_concurrent,
        }
    )
    context.event("evaluation_started", attributes)
    output_dir = context.workspace / "evaluation" / environment.id
    output_dir.mkdir(parents=True, exist_ok=False)
    backend = runner(context, request, output_dir)
    artifact = ProducedArtifact(
        name=f"evaluation/{_model_id(request)}/{_plan_id(request)}/{environment.id}",
        kind="verifiers-evaluation",
        reference=LocalArtifactRef(output_dir, _directory_digest(output_dir)),
        metadata={
            **attributes,
            "environment_package": environment.source.package,
            "environment_revision": environment.source.revision,
        },
    )
    context.artifact(artifact)
    sync = TraceSynchronization(
        observed=backend.synchronization.observed_records,
        emitted=backend.synchronization.emitted_records,
        invalid=backend.synchronization.invalid_records,
        failed_batches=backend.synchronization.failed_batches,
        unsynchronized=backend.synchronization.unsynchronized_records,
        errors=tuple(backend.synchronization.errors),
    )
    context.metrics(
        {
            "eval/traces_observed": sync.observed,
            "eval/traces_emitted": sync.emitted,
            "eval/trace_records_invalid": sync.invalid,
            "eval/trace_sync_failed_batches": sync.failed_batches,
            "eval/traces_unsynchronized": sync.unsynchronized,
            "eval/trace_sync_complete": int(sync.complete),
        },
        attributes=attributes,
    )
    context.event(
        "evaluation_completed",
        {
            **attributes,
            "rollouts": len(backend.trace_ids),
            "trace_sync_complete": sync.complete,
            "evaluation_status": sync.status,
        },
    )
    return EvaluationResult(
        plan_id=_plan_id(request),
        environment_id=environment.id,
        model_id=_model_id(request),
        trace_ids=backend.trace_ids,
        native_artifact=artifact,
        synchronization=sync,
    )


def general(
    context: EvaluationContext,
    request: EvaluateRequest,
    *,
    runner: EvaluationRunner = run_verifiers,
) -> EvaluationResult:
    """Run one cell from a general evaluation plan."""

    if request.plan.kind != "general":
        raise ValueError("eval.general requires a general evaluation plan")
    return evaluate(context, request, runner=runner)


def domain(
    context: EvaluationContext,
    request: EvaluateRequest,
    *,
    runner: EvaluationRunner = run_verifiers,
) -> EvaluationResult:
    """Run one cell from a domain evaluation plan."""

    if request.plan.kind != "domain":
        raise ValueError("eval.domain requires a domain evaluation plan")
    return evaluate(context, request, runner=runner)


def _attributes(request: EvaluateRequest) -> dict[str, str | int]:
    return {
        "model_variant_id": request.model.id,
        "evaluation_plan_id": request.plan.id,
        "evaluation_plan_kind": request.plan.kind,
        "inference_binding_id": request.inference.id,
        "execution_target_id": request.target.id,
    }


def _model_id(request: EvaluateRequest) -> str:
    return request.model.id


def _plan_id(request: EvaluateRequest) -> str:
    return request.plan.id


__all__ = ["EvaluationContext", "EvaluationRunner", "domain", "evaluate", "general"]
