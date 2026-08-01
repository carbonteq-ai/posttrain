"""Public endpoint-neutral evaluation operation."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

from posttrain.common import LocalArtifactRef, ProducedArtifact, RunContext

from .backends.verifiers import VerifiersRunResult, run_verifiers
from .requests import EvaluateRequest, RemoteEvaluationBinding, RemotePolicy
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
            "environment_revision": environment.revision,
        },
        role="evaluation",
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
            "eval/run/rollouts_attempted": backend.population.attempted,
            "eval/run/rollouts_complete": backend.population.complete,
            "eval/run/rollouts_failed": backend.population.failed,
            "eval/run/rollouts_truncated": backend.population.truncated,
            "eval/run/coverage_missing": backend.population.coverage_missing,
            "eval/traces_observed": sync.observed,
            "eval/traces_emitted": sync.emitted,
            "eval/trace_records_invalid": sync.invalid,
            "eval/trace_sync_failed_batches": sync.failed_batches,
            "eval/traces_unsynchronized": sync.unsynchronized,
            "eval/trace_sync_complete": int(sync.complete),
        },
        attributes=attributes,
    )
    evaluation_status = "complete" if sync.complete and backend.population.coverage_missing == 0 else "partial"
    context.event(
        "evaluation_completed",
        {
            **attributes,
            "rollouts": backend.population.attempted,
            "rollouts_complete": backend.population.complete,
            "rollouts_failed": backend.population.failed,
            "rollouts_truncated": backend.population.truncated,
            "coverage_missing": backend.population.coverage_missing,
            "trace_sync_complete": sync.complete,
            "evaluation_status": evaluation_status,
        },
    )
    return EvaluationResult(
        plan_id=_plan_id(request),
        environment_id=environment.id,
        model_id=_model_id(request),
        trace_ids=backend.trace_ids,
        native_artifact=artifact,
        synchronization=sync,
        population=backend.population,
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
    attributes: dict[str, str | int] = {
        "evaluation_subject_id": request.model.id,
        "evaluation_plan_id": request.plan.id,
        "evaluation_plan_kind": request.plan.kind,
        "inference_binding_id": request.inference.id,
        "execution_target_id": request.target.id,
    }
    if isinstance(request.model, RemotePolicy):
        assert isinstance(request.inference, RemoteEvaluationBinding)
        attributes.update(
            {
                "evaluation_subject_kind": "remote-policy",
                "remote_policy_revision": request.model.revision,
                "remote_service_id": request.inference.service.id,
                "remote_service_revision": request.inference.service.revision,
                "remote_service_protocol": request.inference.service.protocol,
                "remote_service_origin": request.inference.service.origin,
            }
        )
    else:
        attributes["evaluation_subject_kind"] = "model-variant"
        attributes["model_variant_id"] = request.model.id
    return attributes


def _model_id(request: EvaluateRequest) -> str:
    return request.model.id


def _plan_id(request: EvaluateRequest) -> str:
    return request.plan.id


__all__ = ["EvaluationContext", "EvaluationRunner", "domain", "evaluate", "general"]
