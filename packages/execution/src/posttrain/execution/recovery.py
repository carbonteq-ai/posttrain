"""Explicit, audited repair of interrupted tracking finalization."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from posttrain.common import ContractError
from posttrain.tracking import RunDataSource, RunOutcomeStatus, RunSummary

from .service import ExecutionSubmissionStore, JobExecutionService

type TrackingRecoveryDisposition = Literal["recovered", "already-cancelled"]
type TerminalTrackingRecoveryDisposition = Literal["recovered", "already-terminal"]

_CANCELLATION_SCHEMA = "posttrain.tracking-cancellation-recovery.v1"
_TERMINAL_SCHEMA = "posttrain.tracking-terminal-recovery.v1"


class CancelledTrackingWriter(Protocol):
    """Writer that rechecks and finalizes one exact retained tracking run."""

    def recover_cancelled(
        self,
        expected: RunSummary,
        *,
        finished_at: datetime,
    ) -> TrackingRecoveryDisposition: ...


class TerminalTrackingWriter(Protocol):
    """Writer that finalizes one exact stranded run to the provider outcome."""

    def recover_terminal(
        self,
        expected: RunSummary,
        *,
        outcome: RunOutcomeStatus,
        finished_at: datetime,
    ) -> TerminalTrackingRecoveryDisposition: ...


@dataclass(frozen=True, slots=True)
class TrackingCancellationRecovery:
    """Auditable result of one tightly guarded cancellation repair."""

    run_id: str
    disposition: TrackingRecoveryDisposition
    execution_provider: str
    execution_provider_id: str
    tracking_provider: str
    tracking_provider_run_id: str
    tracking_started_at: datetime
    recovered_at: datetime


@dataclass(frozen=True, slots=True)
class TerminalTrackingRecovery:
    """Auditable result of provider-terminal tracking repair."""

    run_id: str
    disposition: TerminalTrackingRecoveryDisposition
    outcome: RunOutcomeStatus
    execution_provider: str
    execution_provider_id: str
    tracking_provider: str
    tracking_provider_run_id: str
    tracking_started_at: datetime
    recovered_at: datetime


async def recover_terminal_tracking(
    service: JobExecutionService,
    source: RunDataSource,
    writer: TerminalTrackingWriter,
    run_id: str,
    *,
    project_id: str,
) -> TerminalTrackingRecovery:
    """Finalize an exact running tracking record from terminal provider truth."""

    submission = service.submission(run_id)
    provider = service.collect(run_id)
    outcome_by_state: dict[str, RunOutcomeStatus] = {
        "succeeded": "succeeded",
        "failed": "failed",
        "cancelled": "cancelled",
    }
    outcome = outcome_by_state.get(provider.record.state)
    if outcome is None:
        raise ContractError(
            "terminal tracking recovery requires a terminal provider run, "
            f"observed {provider.record.state!r}"
        )

    detail = await source.get_run(run_id)
    summary = detail.summary
    if summary.provider != "trackio":
        raise ContractError("terminal tracking recovery currently supports Trackio only")
    if summary.run_id != run_id:
        raise ContractError("terminal tracking recovery resolved a different canonical run")
    if summary.project_id != project_id:
        raise ContractError("terminal tracking recovery project identity does not match")
    if summary.provider_run_id is None:
        raise ContractError("terminal tracking recovery requires an exact provider run id")
    if summary.status not in {"running", outcome}:
        raise ContractError(
            "terminal tracking recovery requires tracking to be running or already match "
            f"provider outcome {outcome!r}, observed {summary.status!r}"
        )

    recovered_at = datetime.now(UTC)
    disposition = writer.recover_terminal(
        summary,
        outcome=outcome,
        finished_at=recovered_at,
    )
    return TerminalTrackingRecovery(
        run_id=run_id,
        disposition=disposition,
        outcome=outcome,
        execution_provider=submission.provider,
        execution_provider_id=submission.provider_id,
        tracking_provider=summary.provider,
        tracking_provider_run_id=summary.provider_run_id,
        tracking_started_at=summary.started_at,
        recovered_at=recovered_at,
    )


async def recover_cancelled_tracking(
    service: JobExecutionService,
    source: RunDataSource,
    writer: CancelledTrackingWriter,
    run_id: str,
    *,
    project_id: str,
) -> TrackingCancellationRecovery:
    """Finalize one exact interrupted tracking run after provider cancellation."""

    submission = service.submission(run_id)
    provider = service.collect(run_id)
    if provider.record.state != "cancelled":
        raise ContractError(
            "tracking cancellation recovery requires a terminal cancelled "
            f"provider run, observed {provider.record.state!r}"
        )

    detail = await source.get_run(run_id)
    summary = detail.summary
    if summary.provider != "trackio":
        raise ContractError("tracking cancellation recovery currently supports Trackio only")
    if summary.run_id != run_id:
        raise ContractError("tracking cancellation recovery resolved a different canonical run")
    if summary.project_id != project_id:
        raise ContractError("tracking cancellation recovery project identity does not match")
    if summary.provider_run_id is None:
        raise ContractError("tracking cancellation recovery requires an exact provider run id")
    if summary.status not in {"running", "cancelled"}:
        raise ContractError(
            "tracking cancellation recovery requires tracking to be running "
            f"or already cancelled, observed {summary.status!r}"
        )

    recovered_at = datetime.now(UTC)
    disposition = writer.recover_cancelled(
        summary,
        finished_at=recovered_at,
    )
    return TrackingCancellationRecovery(
        run_id=run_id,
        disposition=disposition,
        execution_provider=submission.provider,
        execution_provider_id=submission.provider_id,
        tracking_provider=summary.provider,
        tracking_provider_run_id=summary.provider_run_id,
        tracking_started_at=summary.started_at,
        recovered_at=recovered_at,
    )


def save_tracking_recovery(
    store: ExecutionSubmissionStore,
    recovery: TrackingCancellationRecovery | TerminalTrackingRecovery,
) -> Path:
    """Append an audit record and atomically replace its protected snapshot."""

    root = store.run_root(recovery.run_id)
    root.mkdir(parents=True, exist_ok=True)
    schema = (
        _TERMINAL_SCHEMA
        if isinstance(recovery, TerminalTrackingRecovery)
        else _CANCELLATION_SCHEMA
    )
    payload = {
        "schema": schema,
        **asdict(recovery),
        "tracking_started_at": recovery.tracking_started_at.isoformat(),
        "recovered_at": recovery.recovered_at.isoformat(),
    }
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    journal = root / "tracking-recovery.jsonl"
    descriptor = os.open(
        journal,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o600,
    )
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    snapshot = root / "tracking-recovery.json"
    temporary = root / f".tracking-recovery-{uuid.uuid4().hex}.tmp"
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, snapshot)
    return snapshot


__all__ = [
    "CancelledTrackingWriter",
    "TerminalTrackingRecovery",
    "TerminalTrackingRecoveryDisposition",
    "TerminalTrackingWriter",
    "TrackingCancellationRecovery",
    "TrackingRecoveryDisposition",
    "recover_cancelled_tracking",
    "recover_terminal_tracking",
    "save_tracking_recovery",
]
