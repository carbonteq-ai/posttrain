"""Reusable foundation-model screening job composition."""

from __future__ import annotations

from posttrain.common import ExecutionContext, Job, JobAction
from posttrain.serve import BenchmarkRequest, BenchmarkResult, benchmark

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
