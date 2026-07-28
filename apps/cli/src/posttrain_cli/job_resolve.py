"""Resolve which job to run when ``--job`` is omitted."""

from __future__ import annotations

from posttrain.common import Catalog, ContractError
from posttrain.work import WorkPackage, resolve_work_package


def resolve_job_id(catalog: Catalog, package: WorkPackage, job: str | None) -> str:
    """Return ``job``, or the sole enabled job when the package has exactly one."""
    resolved = resolve_work_package(catalog, package)
    enabled = tuple(
        item.id
        for item in resolved.recipe.jobs
        if not item.optional or item.id in package.enabled_optional_jobs
    )
    if job is not None:
        if job not in enabled:
            available = ", ".join(enabled) if enabled else "(none)"
            raise ContractError(f"job {job!r} is not enabled; available: {available}")
        return job
    if len(enabled) == 1:
        return enabled[0]
    if not enabled:
        raise ContractError("work package has no enabled jobs")
    raise ContractError(
        f"pass --job; work package has {len(enabled)} enabled jobs: {', '.join(enabled)}"
    )


__all__ = ["resolve_job_id"]
