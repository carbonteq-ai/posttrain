"""Qualification project entry for the primary Posttrain CLI."""

from __future__ import annotations

import os
import subprocess
from dataclasses import replace

from posttrain.jobs import build_job_runtime
from posttrain.work import JobRuntime, ProjectExecutionRequest

from .source import resolve_git_source


def _source_metadata(request: ProjectExecutionRequest) -> dict[str, str | bool]:
    try:
        return resolve_git_source(request.project_root).metadata()
    except subprocess.CalledProcessError:
        revision = os.getenv("POSTTRAIN_PROJECT_REVISION")
        return {
            "git_revision": revision or "unavailable",
            "git_dirty": False,
        }


def configure(request: ProjectExecutionRequest) -> JobRuntime:
    """Build the standard job runtime and attach lab qualification source metadata."""

    runtime = build_job_runtime(request)
    metadata = dict(runtime.source_metadata)
    metadata.update(_source_metadata(request))
    return replace(runtime, source_metadata=metadata)


__all__ = ["configure"]
