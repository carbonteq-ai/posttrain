"""Compatibility exports for the reusable `posttrain.work` runner."""

from posttrain.work.runner import (
    ResolvedSeat,
    ResolvedWorkPackage,
    RunExecutor,
    WorkPackageContext,
    resolve_work_package,
    run_work_package,
    validate_work_package,
)

__all__ = [
    "ResolvedSeat",
    "ResolvedWorkPackage",
    "RunExecutor",
    "WorkPackageContext",
    "resolve_work_package",
    "run_work_package",
    "validate_work_package",
]
