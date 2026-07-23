"""Compatibility exports for work-package contracts now owned by `posttrain.work`."""

from posttrain.work.contracts import (
    JobDefinition,
    JobKind,
    JobOperation,
    JobStatus,
    Recipe,
    RecipeJob,
    ResolvedSeats,
    SeatBinding,
    Stage,
    WorkPackage,
    WorkPackageJobResult,
    WorkPackageResult,
    WorkPackageSchema,
    load_work_package,
)

__all__ = [
    "JobDefinition",
    "JobKind",
    "JobOperation",
    "JobStatus",
    "Recipe",
    "RecipeJob",
    "ResolvedSeats",
    "SeatBinding",
    "Stage",
    "WorkPackage",
    "WorkPackageJobResult",
    "WorkPackageResult",
    "WorkPackageSchema",
    "load_work_package",
]
