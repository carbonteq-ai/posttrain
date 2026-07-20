"""Reusable foundation-model screening job composition."""

from __future__ import annotations

from posttrain.common import ExecutionContext, Job, JobAction, ModelProfile
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
        return generate(
            context,
            GenerationRequest(
                endpoint=endpoint,
                messages=({"role": "user", "content": "Answer with exactly the word ready."},),
                max_tokens=16,
            ),
            request.model,
        )
