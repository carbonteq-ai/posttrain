"""Small cross-engine contracts for the post-training workspace."""

from __future__ import annotations

import os
from pathlib import Path

# Workspace root: packages/common/src/common/__init__.py -> parents[4]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
RUNS_DIR = WORKSPACE_ROOT / "runs"
PROFILES_DIR = WORKSPACE_ROOT / "profiles"
JOBS_DIR = WORKSPACE_ROOT / "jobs"

TRACKIO_PROJECT = os.environ.get("LAB_TRACKIO_PROJECT", "lab")


from common.profiles import ProfileError, ProfileResolver, ResolvedProfile  # noqa: E402
from common.runs import RUN_KINDS, RunContext, RunKind  # noqa: E402
from common.tracking import TrackedRun  # noqa: E402

__all__ = [
    "JOBS_DIR",
    "PROFILES_DIR",
    "ProfileError",
    "ProfileResolver",
    "ResolvedProfile",
    "RUNS_DIR",
    "RUN_KINDS",
    "RunContext",
    "RunKind",
    "TRACKIO_PROJECT",
    "TrackedRun",
    "WORKSPACE_ROOT",
]
