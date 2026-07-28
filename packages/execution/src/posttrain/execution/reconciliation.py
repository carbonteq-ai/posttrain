"""Join provider termination with provider-neutral retained run evidence."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from posttrain.tracking import ArtifactSet, RunDataSource, RunStatus

from .contracts import ExecutionRecord, ExecutionResult
from .service import ExecutionSubmission, ExecutionSubmissionStore, JobExecutionService

type ReconciliationState = Literal["pending", "consistent", "inconsistent"]
type ReconciledOutcome = Literal[
    "succeeded",
    "partial",
    "failed",
    "cancelled",
    "unsupported",
    "lost",
]

_SCHEMA = "posttrain.execution-reconciliation.v1"
_TERMINAL_TRACKING_STATES = frozenset({"succeeded", "partial", "failed", "cancelled", "unsupported"})
_SUCCESS_TRACKING_STATES = frozenset({"succeeded", "partial", "unsupported"})


@dataclass(frozen=True, slots=True)
class ReconciledArtifact:
    """One retained output identity used by the terminal evidence barrier."""

    logical_name: str
    role: str | None
    provider: str
    namespace: str
    name: str
    version: str
    digest: str | None


@dataclass(frozen=True, slots=True)
class ExecutionReconciliation:
    """One retryable reconciliation observation for a submitted run."""

    run_id: str
    state: ReconciliationState
    outcome: ReconciledOutcome
    provider_record: ExecutionRecord
    provider_exit_code: int | None
    tracking_status: RunStatus | None
    tracking_provider_run_id: str | None
    required_artifact_roles: tuple[str, ...]
    retained_artifacts: tuple[ReconciledArtifact, ...]
    missing_artifact_roles: tuple[str, ...]
    observed_at: datetime
    message: str

    @property
    def complete(self) -> bool:
        return self.state == "consistent"

    @property
    def settled(self) -> bool:
        """Whether any further observation could change this run's evidence.

        A run the provider reports as failed or cancelled owes no artifacts,
        which is already how a run with tracking disabled is judged. Such a run
        can die before it ever opens a tracking run, so its evidence never
        arrives and reconciliation stays pending forever. That is only a label
        until it reaches admission, which holds the machine's placement until
        the run settles: one crashed run would otherwise queue every later run
        on that host behind evidence that cannot exist.
        """
        return self.complete or self.provider_record.state in {"failed", "cancelled"}


async def reconcile_execution(
    service: JobExecutionService,
    source: RunDataSource | None,
    run_id: str,
) -> ExecutionReconciliation:
    """Collect a terminal provider result and compare it with retained evidence."""

    submission = service.submission(run_id)
    provider = service.collect(run_id)
    observed_at = datetime.now(UTC)
    if source is None:
        return ExecutionReconciliation(
            run_id=run_id,
            state="consistent",
            outcome=_outcome(provider.record.state, None),
            provider_record=provider.record,
            provider_exit_code=provider.exit_code,
            tracking_status=None,
            tracking_provider_run_id=None,
            required_artifact_roles=submission.required_artifact_roles,
            retained_artifacts=(),
            missing_artifact_roles=(
                () if provider.record.state in {"failed", "cancelled"} else submission.required_artifact_roles
            ),
            observed_at=observed_at,
            message=(
                "provider is terminal; durable tracking was explicitly disabled, so no retained evidence was asserted"
            ),
        )
    try:
        detail = await source.get_run(run_id)
    except Exception as error:
        return _pending(
            submission,
            provider,
            observed_at,
            f"tracking evidence is unavailable ({type(error).__name__})",
        )

    tracking = detail.summary
    if tracking.status not in _TERMINAL_TRACKING_STATES:
        return _pending(
            submission,
            provider,
            observed_at,
            "tracking run is not terminal",
            tracking_status=tracking.status,
            tracking_provider_run_id=tracking.provider_run_id,
        )

    try:
        artifacts = await source.artifacts(run_id)
    except Exception as error:
        return _pending(
            submission,
            provider,
            observed_at,
            f"retained artifacts are unavailable ({type(error).__name__})",
            tracking_status=tracking.status,
            tracking_provider_run_id=tracking.provider_run_id,
        )
    retained = _retained_outputs(artifacts)
    missing = (
        ()
        if provider.record.state in {"failed", "cancelled"}
        else _missing_roles(submission.required_artifact_roles, retained)
    )
    consistency_message = _consistency_message(
        provider.record.state,
        tracking.status,
        missing,
    )
    state: ReconciliationState = "consistent" if consistency_message is None else "inconsistent"
    return ExecutionReconciliation(
        run_id=run_id,
        state=state,
        outcome=_outcome(provider.record.state, tracking.status),
        provider_record=provider.record,
        provider_exit_code=provider.exit_code,
        tracking_status=tracking.status,
        tracking_provider_run_id=tracking.provider_run_id,
        required_artifact_roles=submission.required_artifact_roles,
        retained_artifacts=retained,
        missing_artifact_roles=missing,
        observed_at=observed_at,
        message=consistency_message or "provider and retained evidence are consistent",
    )


def save_reconciliation(
    store: ExecutionSubmissionStore,
    result: ExecutionReconciliation,
) -> Path:
    """Append the pass and atomically replace its compact latest snapshot."""

    root = store.run_root(result.run_id)
    root.mkdir(parents=True, exist_ok=True)
    payload = _payload(result)
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    journal = root / "reconciliation.jsonl"
    descriptor = os.open(journal, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    snapshot = root / "reconciliation.json"
    temporary = root / f".reconciliation-{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, snapshot)
    return snapshot


def _pending(
    submission: ExecutionSubmission,
    provider: ExecutionResult,
    observed_at: datetime,
    message: str,
    *,
    tracking_status: RunStatus | None = None,
    tracking_provider_run_id: str | None = None,
) -> ExecutionReconciliation:
    return ExecutionReconciliation(
        run_id=submission.run_id,
        state="pending",
        outcome=_outcome(provider.record.state, tracking_status),
        provider_record=provider.record,
        provider_exit_code=provider.exit_code,
        tracking_status=tracking_status,
        tracking_provider_run_id=tracking_provider_run_id,
        required_artifact_roles=submission.required_artifact_roles,
        retained_artifacts=(),
        missing_artifact_roles=submission.required_artifact_roles,
        observed_at=observed_at,
        message=message,
    )


def _retained_outputs(artifacts: ArtifactSet) -> tuple[ReconciledArtifact, ...]:
    retained = []
    for link in artifacts.outputs:
        role = link.artifact.provider_metadata.get("posttrain_role")
        retained.append(
            ReconciledArtifact(
                logical_name=link.logical_name,
                role=role if isinstance(role, str) and role.strip() else None,
                provider=link.artifact.provider,
                namespace=link.artifact.namespace,
                name=link.artifact.name,
                version=link.artifact.version,
                digest=link.artifact.digest,
            )
        )
    return tuple(sorted(retained, key=lambda item: item.logical_name))


def _missing_roles(
    required: tuple[str, ...],
    artifacts: tuple[ReconciledArtifact, ...],
) -> tuple[str, ...]:
    missing = []
    for role in required:
        matches = tuple(artifact for artifact in artifacts if artifact.role == role and artifact.digest is not None)
        if len(matches) != 1:
            missing.append(role)
    return tuple(missing)


def _consistency_message(
    provider_state: str,
    tracking_status: RunStatus,
    missing_roles: tuple[str, ...],
) -> str | None:
    if provider_state == "succeeded" and tracking_status not in _SUCCESS_TRACKING_STATES:
        return f"provider succeeded but retained tracking outcome is {tracking_status}"
    if provider_state == "failed" and tracking_status != "failed":
        return f"provider failed but retained tracking outcome is {tracking_status}"
    if provider_state == "cancelled" and tracking_status != "cancelled":
        return f"provider cancelled but retained tracking outcome is {tracking_status}"
    if provider_state == "lost":
        return "provider execution was lost"
    if missing_roles:
        return "required retained artifact roles are missing or ambiguous: " + ", ".join(missing_roles)
    return None


def _outcome(
    provider_state: str,
    tracking_status: RunStatus | None,
) -> ReconciledOutcome:
    if provider_state == "lost":
        return "lost"
    if tracking_status == "succeeded":
        return "succeeded"
    if tracking_status == "partial":
        return "partial"
    if tracking_status == "failed":
        return "failed"
    if tracking_status == "cancelled":
        return "cancelled"
    if tracking_status == "unsupported":
        return "unsupported"
    if provider_state == "succeeded":
        return "succeeded"
    if provider_state == "failed":
        return "failed"
    if provider_state == "cancelled":
        return "cancelled"
    raise RuntimeError(f"provider result is not terminal: {provider_state}")


def _payload(result: ExecutionReconciliation) -> dict[str, object]:
    payload = asdict(result)
    payload["schema"] = _SCHEMA
    provider_record = payload["provider_record"]
    assert isinstance(provider_record, dict)
    provider_record["observed_at"] = result.provider_record.observed_at.isoformat()
    payload["observed_at"] = result.observed_at.isoformat()
    return payload


__all__ = [
    "ExecutionReconciliation",
    "ReconciledArtifact",
    "ReconciledOutcome",
    "ReconciliationState",
    "reconcile_execution",
    "save_reconciliation",
]
