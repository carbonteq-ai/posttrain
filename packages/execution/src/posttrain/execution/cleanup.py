"""Evidence-gated cleanup for terminal provider executions."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from posttrain.common import ContractError
from posttrain.tracking import RunDataSource

from .contracts import ProviderCleanupDisposition
from .reconciliation import reconcile_execution, save_reconciliation
from .service import ExecutionSubmissionStore, JobExecutionService

type CleanupEvidenceState = Literal["reconciled", "provider-terminal"]
type WorkspaceCleanupDisposition = Literal[
    "removed",
    "already-absent",
    "not-created",
    "provider-managed",
]

_PLAN_SCHEMA = "posttrain.execution-cleanup-plan.v1"
_RECEIPT_SCHEMA = "posttrain.execution-cleanup.v1"


@dataclass(frozen=True, slots=True)
class ExecutionCleanupReceipt:
    """Durable proof that one exact execution was safely cleaned."""

    run_id: str
    outcome: str
    evidence_state: CleanupEvidenceState
    provider: str
    provider_id: str
    provider_disposition: ProviderCleanupDisposition
    workspace_disposition: WorkspaceCleanupDisposition
    workspace_reclaimed_bytes: int
    reconciliation_file: str
    retained_artifact_count: int
    diagnostic_file: str | None
    diagnostic_digest: str | None
    diagnostic_line_count: int
    diagnostic_truncated: bool
    completed_at: datetime


async def cleanup_execution(
    service: JobExecutionService,
    store: ExecutionSubmissionStore,
    source: RunDataSource | None,
    run_id: str,
    *,
    diagnostic_limit: int = 500,
) -> ExecutionCleanupReceipt:
    """Reconcile evidence, release provider state, and remove disposable data.

    Successful runs require a consistent retained-evidence barrier. A failed or
    cancelled startup that never created a tracking run may instead retain a
    bounded provider diagnostic before cleanup. An inconsistent tracking record
    is never bypassed.
    """

    if diagnostic_limit < 1:
        raise ValueError("cleanup diagnostic limit must be positive")
    root = store.run_root(run_id)
    submission = service.submission(run_id)
    workspace = submission.run_workspace or store.default_run_workspace(run_id)
    receipt_path = root / "cleanup.json"
    if receipt_path.is_file():
        existing_receipt = _load_receipt(receipt_path, run_id)
        # dstack receipts written before exact-worker workspace cleanup existed
        # are incomplete, not a terminal idempotency result. Reuse their
        # evidence-gated plan and upgrade the receipt after native cleanup.
        if not (
            existing_receipt.provider == "dstack"
            and existing_receipt.workspace_disposition == "provider-managed"
        ):
            return existing_receipt

    plan_path = root / "cleanup-plan.json"
    if plan_path.is_file():
        plan = _load_payload(plan_path, _PLAN_SCHEMA, run_id)
    else:
        reconciliation = await reconcile_execution(service, source, run_id)
        reconciliation_path = save_reconciliation(store, reconciliation)
        if (
            reconciliation.state == "consistent"
            and reconciliation.tracking_status is None
            and reconciliation.outcome in {"succeeded", "partial", "unsupported"}
            and reconciliation.required_artifact_roles
        ):
            raise RuntimeError(
                "cleanup refuses a successful untracked run with required outputs; "
                "retain or deliberately discard the workspace first"
            )
        if reconciliation.state == "consistent":
            evidence_state: CleanupEvidenceState = "reconciled"
        elif (
            reconciliation.state == "pending"
            and reconciliation.tracking_status is None
            and reconciliation.outcome in {"failed", "cancelled"}
        ):
            evidence_state = "provider-terminal"
        else:
            raise RuntimeError(
                "execution cleanup requires consistent retained evidence, or a "
                "terminal failed/cancelled provider run with no tracking record"
            )

        diagnostic = _retain_diagnostic(
            service,
            root,
            run_id,
            limit=diagnostic_limit,
            required=evidence_state == "provider-terminal",
        )
        plan = {
            "schema": _PLAN_SCHEMA,
            "run_id": run_id,
            "outcome": reconciliation.outcome,
            "evidence_state": evidence_state,
            "provider": reconciliation.provider_record.handle.provider,
            "provider_id": reconciliation.provider_record.handle.provider_id,
            "reconciliation_file": reconciliation_path.name,
            "retained_artifact_count": len(reconciliation.retained_artifacts),
            "workspace_logical_bytes_before": _workspace_bytes_before(
                submission.provider,
                workspace,
                run_id,
            ),
            **diagnostic,
            "planned_at": datetime.now(UTC).isoformat(),
        }
        _write_atomic(plan_path, plan)

    if (
        plan["provider"] != submission.provider
        or plan["provider_id"] != submission.provider_id
    ):
        raise ContractError("cleanup plan conflicts with the persisted provider handle")

    provider_cleanup = service.cleanup(run_id)
    if submission.provider == "local-docker":
        workspace_disposition, reclaimed = _cleanup_workspace(
            submission.provider,
            workspace,
            run_id,
        )
    else:
        workspace_disposition = provider_cleanup.workspace_disposition
        reclaimed = provider_cleanup.workspace_reclaimed_bytes
    reclaimed = max(
        reclaimed,
        int(plan.get("workspace_logical_bytes_before", 0)),
    )
    completed_at = datetime.now(UTC)
    receipt = ExecutionCleanupReceipt(
        run_id=run_id,
        outcome=str(plan["outcome"]),
        evidence_state=_evidence_state(plan["evidence_state"]),
        provider=submission.provider,
        provider_id=submission.provider_id,
        provider_disposition=provider_cleanup.disposition,
        workspace_disposition=workspace_disposition,
        workspace_reclaimed_bytes=reclaimed,
        reconciliation_file=str(plan["reconciliation_file"]),
        retained_artifact_count=int(plan["retained_artifact_count"]),
        diagnostic_file=_optional_string(plan.get("diagnostic_file")),
        diagnostic_digest=_optional_string(plan.get("diagnostic_digest")),
        diagnostic_line_count=int(plan["diagnostic_line_count"]),
        diagnostic_truncated=bool(plan["diagnostic_truncated"]),
        completed_at=completed_at,
    )
    payload = asdict(receipt)
    payload["schema"] = _RECEIPT_SCHEMA
    payload["completed_at"] = completed_at.isoformat()
    _write_atomic(receipt_path, payload)
    return receipt


def _retain_diagnostic(
    service: JobExecutionService,
    root: Path,
    run_id: str,
    *,
    limit: int,
    required: bool,
) -> dict[str, object]:
    if not required:
        return {
            "diagnostic_file": None,
            "diagnostic_digest": None,
            "diagnostic_line_count": 0,
            "diagnostic_truncated": False,
        }
    page = service.logs(run_id, limit=limit)
    encoded = (
        ("\n".join(page.lines) + ("\n" if page.lines else "")).encode()
    )
    path = root / "diagnostic.log"
    _write_bytes_atomic(path, encoded)
    return {
        "diagnostic_file": path.name,
        "diagnostic_digest": hashlib.sha256(encoded).hexdigest(),
        "diagnostic_line_count": len(page.lines),
        "diagnostic_truncated": page.truncated,
    }


def _cleanup_workspace(
    provider: str,
    workspace: Path,
    run_id: str,
) -> tuple[WorkspaceCleanupDisposition, int]:
    if provider != "local-docker":
        return "provider-managed", 0
    if not workspace.is_absolute() or workspace.name != run_id:
        raise ContractError("cleanup workspace is not scoped to the canonical run id")
    if workspace.is_symlink():
        raise ContractError("cleanup refuses a symlinked run workspace")
    if not workspace.exists():
        return "already-absent", 0
    if not workspace.is_dir():
        raise ContractError("cleanup run workspace is not a directory")
    reclaimed = _logical_bytes(workspace)
    shutil.rmtree(workspace)
    return "removed", reclaimed


def _workspace_bytes_before(provider: str, workspace: Path, run_id: str) -> int:
    if provider != "local-docker" or not workspace.exists():
        return 0
    if (
        not workspace.is_absolute()
        or workspace.name != run_id
        or workspace.is_symlink()
        or not workspace.is_dir()
    ):
        raise ContractError("cleanup workspace is not scoped to the canonical run id")
    return _logical_bytes(workspace)


def _logical_bytes(path: Path) -> int:
    return sum(
        child.lstat().st_size
        for child in path.rglob("*")
        if child.is_file() or child.is_symlink()
    )


def _load_receipt(path: Path, run_id: str) -> ExecutionCleanupReceipt:
    payload = _load_payload(path, _RECEIPT_SCHEMA, run_id)
    try:
        return ExecutionCleanupReceipt(
            run_id=run_id,
            outcome=str(payload["outcome"]),
            evidence_state=_evidence_state(payload["evidence_state"]),
            provider=str(payload["provider"]),
            provider_id=str(payload["provider_id"]),
            provider_disposition=_provider_disposition(
                payload["provider_disposition"]
            ),
            workspace_disposition=_workspace_disposition(
                payload["workspace_disposition"]
            ),
            workspace_reclaimed_bytes=int(payload["workspace_reclaimed_bytes"]),
            reconciliation_file=str(payload["reconciliation_file"]),
            retained_artifact_count=int(payload["retained_artifact_count"]),
            diagnostic_file=_optional_string(payload.get("diagnostic_file")),
            diagnostic_digest=_optional_string(payload.get("diagnostic_digest")),
            diagnostic_line_count=int(payload["diagnostic_line_count"]),
            diagnostic_truncated=bool(payload["diagnostic_truncated"]),
            completed_at=datetime.fromisoformat(str(payload["completed_at"])),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError(f"execution cleanup receipt is invalid for run {run_id}") from error


def _load_payload(path: Path, schema: str, run_id: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ContractError(f"execution cleanup state is invalid for run {run_id}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != schema
        or payload.get("run_id") != run_id
    ):
        raise ContractError(f"execution cleanup state is invalid for run {run_id}")
    return payload


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    _write_bytes_atomic(
        path,
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )


def _write_bytes_atomic(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}-{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _evidence_state(value: object) -> CleanupEvidenceState:
    if value not in {"reconciled", "provider-terminal"}:
        raise ContractError("execution cleanup evidence state is invalid")
    return cast(CleanupEvidenceState, value)


def _provider_disposition(value: object) -> ProviderCleanupDisposition:
    if value not in {"removed", "already-absent", "not-created", "provider-managed"}:
        raise ContractError("execution provider cleanup disposition is invalid")
    return cast(ProviderCleanupDisposition, value)


def _workspace_disposition(value: object) -> WorkspaceCleanupDisposition:
    if value not in {"removed", "already-absent", "not-created", "provider-managed"}:
        raise ContractError("execution workspace cleanup disposition is invalid")
    return cast(WorkspaceCleanupDisposition, value)


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


__all__ = [
    "CleanupEvidenceState",
    "ExecutionCleanupReceipt",
    "WorkspaceCleanupDisposition",
    "cleanup_execution",
]
