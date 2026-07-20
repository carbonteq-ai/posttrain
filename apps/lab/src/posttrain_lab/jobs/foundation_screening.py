"""Reusable foundation-model screening job composition."""

from __future__ import annotations

from dataclasses import dataclass

from posttrain.common import ExecutionContext, Job, JobAction, ModelProfile
from posttrain.eval import (
    EvaluationBudget,
    EvaluationProgram,
    EvaluationRequest,
    EvaluationResult,
    EvaluationTarget,
    evaluate,
)
from posttrain.serve import (
    BenchmarkRequest,
    BenchmarkResult,
    GenerationRequest,
    GenerationResult,
    LaunchRequest,
    benchmark,
    generate,
    launch,
    probe,
)

JOB_ID = "platform/foundation-screening"


@dataclass(frozen=True, slots=True)
class ManagedEvaluationRequest:
    launch: LaunchRequest
    program: EvaluationProgram
    environment_id: str
    context_window: int
    reasoning_mode: str | None = None
    budget: EvaluationBudget = EvaluationBudget()

    def __post_init__(self) -> None:
        self.program.environment(self.environment_id)
        if self.context_window > self.launch.profile.engine.max_model_len:
            raise ValueError("evaluation context exceeds the launched serving context")


def foundation_screening_job(version: str) -> Job:
    return Job(id=JOB_ID, version=version, name="Foundation model screening")


def serving_benchmark_action(request: BenchmarkRequest) -> JobAction:
    return JobAction(
        job_id=JOB_ID,
        id=f"serve/{request.model.id}/{request.cell.id}",
        kind="serving-benchmark",
    )


def run_serving_cell(context: ExecutionContext, request: BenchmarkRequest) -> BenchmarkResult:
    return benchmark(context, request)


def online_smoke_action(model: ModelProfile) -> JobAction:
    return JobAction(
        job_id=JOB_ID,
        id=f"serve-online/{model.id}",
        kind="serving-online-smoke",
    )


def run_online_smoke(context: ExecutionContext, request: LaunchRequest) -> GenerationResult:
    with launch(context, request) as endpoint:
        health = probe(context, endpoint)
        if not health.model_available:
            raise RuntimeError(f"launched endpoint does not expose {endpoint.model!r}")
        result = generate(
            context,
            GenerationRequest(
                endpoint=endpoint,
                messages=({"role": "user", "content": "What is 2 + 2? Answer concisely."},),
                max_tokens=request.profile.sampling.max_tokens,
            ),
            request.model,
        )
        if not result.content.strip():
            raise RuntimeError(f"online smoke produced no final answer (finish_reason={result.finish_reason!r})")
        return result


def evaluation_action(request: ManagedEvaluationRequest) -> JobAction:
    return JobAction(
        job_id=JOB_ID,
        id=f"eval/{request.program.kind}/{request.launch.model.id}/{request.environment_id}",
        kind=f"{request.program.kind}-evaluation",
    )


def run_managed_evaluation(
    context: ExecutionContext,
    request: ManagedEvaluationRequest,
) -> EvaluationResult:
    """Compose serving and evaluation without either reusable package importing the other."""

    with launch(context, request.launch) as endpoint:
        health = probe(context, endpoint)
        if not health.model_available:
            raise RuntimeError(f"launched endpoint does not expose {endpoint.model!r}")
        return evaluate(
            context,
            EvaluationRequest(
                model=request.launch.model,
                target=EvaluationTarget(endpoint.base_url, endpoint.model),
                program=request.program,
                environment_id=request.environment_id,
                context_window=request.context_window,
                reasoning_mode=request.reasoning_mode,
                budget=request.budget,
            ),
        )
