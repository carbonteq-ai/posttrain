"""Purge executor that settles one abandoned machine admission entry."""

from __future__ import annotations

from typing import Protocol

from posttrain.common import ContractError

from .admission import AdmissionEntry
from .purge import PurgeAction

SETTLE_ADMISSION_KIND = "local.settle_admission"


class AdmissionSettlement(Protocol):
    def get(self, run_id: str) -> AdmissionEntry: ...

    def settle_orphaned(
        self,
        run_id: str,
        *,
        admission_key: str,
        provider_id: str | None,
        note: str,
    ) -> bool: ...


class AdmissionSettlePurgeExecutor:
    """Move one exact orphaned admission entry to ``completed``.

    The composition root re-proves the orphan conditions (owning control store
    gone, provider execution terminal or absent) before apply. This executor
    re-checks that the ledger entry is still the one the plan names.
    """

    def __init__(self, admission: AdmissionSettlement) -> None:
        self._admission = admission

    @staticmethod
    def _target(action: PurgeAction) -> tuple[str, str, str | None, str]:
        if action.kind != SETTLE_ADMISSION_KIND:
            raise ContractError(f"unsupported admission purge action {action.kind!r}")
        run_id = action.target.get("run_id")
        admission_key = action.target.get("admission_key")
        provider_id = action.target.get("provider_id")
        purge_id = action.target.get("note")
        if (
            not isinstance(run_id, str)
            or not isinstance(admission_key, str)
            or not (provider_id is None or isinstance(provider_id, str))
            or not isinstance(purge_id, str)
        ):
            raise ContractError("admission settle action has an invalid target")
        return run_id, admission_key, provider_id, purge_id

    def revalidate(self, action: PurgeAction) -> None:
        run_id, admission_key, provider_id, _note = self._target(action)
        entry = self._admission.get(run_id)
        if entry.state == "completed":
            return
        if entry.state != "terminal_pending_evidence":
            raise ContractError(f"admission run {run_id!r} is {entry.state!r}, not terminal_pending_evidence")
        if entry.admission_key != admission_key or entry.plan.native_plan_id != provider_id:
            raise ContractError(f"admission run {run_id!r} changed after the purge preview")

    def apply(self, action: PurgeAction) -> None:
        run_id, admission_key, provider_id, note = self._target(action)
        entry = self._admission.get(run_id)
        if entry.state == "completed":
            return
        self._admission.settle_orphaned(run_id, admission_key=admission_key, provider_id=provider_id, note=note)


__all__ = ["SETTLE_ADMISSION_KIND", "AdmissionSettlePurgeExecutor"]
