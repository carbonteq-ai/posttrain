"""Small reference job used to prove the composition boundary."""

from __future__ import annotations

from posttrain.common import ExecutionContext, Job, JobAction


def noop_job(version: str) -> Job:
    return Job(id="platform/noop", version=version, name="Platform no-op")


def noop_action() -> JobAction:
    return JobAction(job_id="platform/noop", id="smoke", kind="smoke")


def run_noop(context: ExecutionContext) -> str:
    context.metric("noop/completed", 1)
    return "ok"
