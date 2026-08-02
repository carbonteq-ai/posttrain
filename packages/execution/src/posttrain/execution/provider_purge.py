"""Execution-provider action executor used by the cross-plane purge."""

from __future__ import annotations

from collections.abc import Mapping

from posttrain.common import ContractError

from .purge import PurgeAction
from .service import JobExecutionService


class ExecutionProviderPurgeExecutor:
    """Revalidate and release exact provider records through existing services."""

    def __init__(self, services: Mapping[str, JobExecutionService]) -> None:
        self._services = services

    def _service(self, action: PurgeAction) -> tuple[JobExecutionService, str]:
        provider = action.target.get("provider")
        run_id = action.target.get("run_id")
        if not isinstance(provider, str) or not isinstance(run_id, str):
            raise ContractError("provider purge action must name provider and run id")
        service = self._services.get(run_id)
        if service is None:
            raise ContractError(f"no execution service is configured for run {run_id!r}")
        if service.submission(run_id).provider != provider:
            raise ContractError(f"provider identity changed for run {run_id!r}")
        return service, run_id

    def revalidate(self, action: PurgeAction) -> None:
        service, run_id = self._service(action)
        record = service.status(run_id)
        if record.state not in {"succeeded", "failed", "cancelled", "lost"}:
            raise ContractError(f"provider run {run_id!r} is not terminal")

    def apply(self, action: PurgeAction) -> None:
        service, run_id = self._service(action)
        service.cleanup(run_id)


__all__ = ["ExecutionProviderPurgeExecutor"]
