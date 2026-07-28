"""Application service for durable provider-neutral execution lifecycles."""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from posttrain.common import ContractError

from .contracts import (
    ExecutionHandle,
    ExecutionPlan,
    ExecutionProvider,
    ExecutionRecord,
    ExecutionRequest,
    ExecutionResult,
    LogCursor,
    LogPage,
    ProviderCleanupResult,
    RuntimeImageRef,
)
from .lifecycle import wait_for_terminal
from .receipts import ExecutionJournal

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCHEMA = "posttrain.execution-submission.v5"
_SUPPORTED_SCHEMAS = frozenset(
    {
        "posttrain.execution-submission.v1",
        "posttrain.execution-submission.v2",
        "posttrain.execution-submission.v3",
        "posttrain.execution-submission.v4",
        _SCHEMA,
    }
)
_CANCEL_SCHEMA = "posttrain.execution-cancel-intent.v1"
_SUBMIT_INTENT_SCHEMA = "posttrain.execution-submit-intent.v1"
_CLEANUP_SCHEMA = "posttrain.execution-cleanup.v1"
_RECONCILIATION_SCHEMA = "posttrain.execution-reconciliation.v1"
_TERMINAL_EXECUTION_STATES = frozenset({"succeeded", "failed", "cancelled", "lost"})


@dataclass(frozen=True, slots=True)
class ExecutionEvidenceSource:
    """Secret-free immutable locator for one run's retained evidence source."""

    provider: str
    source_id: str
    project: str
    endpoint: str | None = None
    scope: str | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("provider", self.provider),
            ("source id", self.source_id),
            ("project", self.project),
        ):
            if not value.strip() or "\x00" in value:
                raise ContractError(f"execution evidence {label} cannot be empty")
        if self.scope is not None and (not self.scope.strip() or "\x00" in self.scope):
            raise ContractError("execution evidence scope cannot be empty")
        if self.endpoint is not None:
            parsed = urlsplit(self.endpoint)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                raise ContractError("execution evidence endpoint must be a credential-free HTTP(S) base URL")

    def _identity(self) -> tuple[str, ...]:
        return (
            self.provider,
            self.source_id,
            self.project,
            self.endpoint or "",
            self.scope or "",
        )


@dataclass(frozen=True, slots=True)
class ExecutionSubmission:
    """Compact durable mapping from a canonical run to a provider execution."""

    run_id: str
    provider: str
    provider_id: str
    idempotency_key: str
    job_image: str
    submitted_at: datetime
    required_artifact_roles: tuple[str, ...] = ()
    run_workspace: Path | None = None
    evidence_source: ExecutionEvidenceSource | None = None
    evidence_source_recorded: bool = True
    legacy_bundle_digest: str | None = None

    def __post_init__(self) -> None:
        if not _RUN_ID.fullmatch(self.run_id):
            raise ContractError("execution submission run id is not path-safe")
        if not self.provider.strip() or not self.provider_id.strip():
            raise ContractError("execution submission provider identity cannot be empty")
        if not self.idempotency_key.strip():
            raise ContractError("execution submission idempotency key cannot be empty")
        RuntimeImageRef(self.job_image)
        if self.legacy_bundle_digest is not None and not _SHA256.fullmatch(self.legacy_bundle_digest):
            raise ContractError("legacy execution submission bundle digest must be SHA-256")
        if self.submitted_at.tzinfo is None or self.submitted_at.utcoffset() is None:
            raise ContractError("execution submission timestamp must be timezone-aware")
        if any(not role.strip() for role in self.required_artifact_roles):
            raise ContractError("execution submission artifact roles cannot be empty")
        if len(set(self.required_artifact_roles)) != len(self.required_artifact_roles):
            raise ContractError("execution submission artifact roles must be unique")
        if self.run_workspace is not None:
            if not self.run_workspace.is_absolute():
                raise ContractError("execution submission run workspace must be absolute")
            if self.run_workspace.name != self.run_id:
                raise ContractError("execution submission run workspace must end with its run id")
        if self.evidence_source is not None and not self.evidence_source_recorded:
            raise ContractError("execution evidence source cannot be present when its locator was not recorded")

    @property
    def handle(self) -> ExecutionHandle:
        return ExecutionHandle(
            provider=self.provider,
            provider_id=self.provider_id,
            idempotency_key=self.idempotency_key,
        )

    def _identity(self) -> tuple[str, ...]:
        return (
            self.run_id,
            self.provider,
            self.provider_id,
            self.idempotency_key,
            self.job_image,
            *self.required_artifact_roles,
            str(self.run_workspace) if self.run_workspace is not None else "",
            str(self.evidence_source_recorded),
            *(self.evidence_source._identity() if self.evidence_source is not None else ("",)),
            self.legacy_bundle_digest or "",
        )


class ExecutionSubmissionStore:
    """Filesystem store under a project's ignored machine-local state root."""

    def __init__(self, state_root: Path) -> None:
        if not state_root.is_absolute():
            raise ValueError("execution state root must be absolute")
        self._state_root = state_root.resolve()
        self._root = self._state_root / "executions"

    @property
    def state_root(self) -> Path:
        return self._state_root

    def default_run_workspace(self, run_id: str) -> Path:
        if not _RUN_ID.fullmatch(run_id):
            raise ContractError("execution run id is not path-safe")
        return self._state_root / "runs" / run_id

    def run_root(self, run_id: str) -> Path:
        if not _RUN_ID.fullmatch(run_id):
            raise ContractError("execution run id is not path-safe")
        return self._root / run_id

    def submission_path(self, run_id: str) -> Path:
        return self.run_root(run_id) / "submission.json"

    def journal(self, run_id: str) -> ExecutionJournal:
        return ExecutionJournal((self.run_root(run_id) / "journal.jsonl").resolve())

    def cancel_intent_path(self, run_id: str) -> Path:
        return self.run_root(run_id) / "cancel-intent.json"

    def submit_intent_path(self, run_id: str) -> Path:
        return self.run_root(run_id) / "submit-intent.json"

    def record_submit_intent(
        self,
        plan: ExecutionPlan,
        evidence_source: ExecutionEvidenceSource | None,
    ) -> datetime:
        """Persist immutable launch intent before contacting a provider."""

        run_id = plan.request.run_spec.run_id
        path = self.submit_intent_path(run_id)
        identity = {
            "provider": plan.provider,
            "native_plan_id": plan.native_plan_id,
            "idempotency_key": plan.request.idempotency_key,
            "job_image": plan.request.image.value,
            "evidence_source": (
                {
                    "provider": evidence_source.provider,
                    "source_id": evidence_source.source_id,
                    "project": evidence_source.project,
                    "endpoint": evidence_source.endpoint,
                    "scope": evidence_source.scope,
                }
                if evidence_source is not None
                else None
            ),
        }
        if path.is_file():
            payload = _load_json_object(
                path,
                label=f"execution submit intent for run {run_id}",
            )
            if (
                payload.get("schema") != _SUBMIT_INTENT_SCHEMA
                or payload.get("run_id") != run_id
                or payload.get("identity") != identity
            ):
                raise ContractError(f"execution submit intent conflicts with run {run_id}")
            try:
                return datetime.fromisoformat(str(payload["requested_at"]))
            except (KeyError, ValueError) as error:
                raise ContractError(f"execution submit intent is invalid for run {run_id}") from error

        requested_at = datetime.now(UTC)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (
            json.dumps(
                {
                    "schema": _SUBMIT_INTENT_SCHEMA,
                    "run_id": run_id,
                    "requested_at": requested_at.isoformat(),
                    "identity": identity,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return self.record_submit_intent(plan, evidence_source)
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        finally:
            os.close(descriptor)
        return requested_at

    def cleaned_result(self, run_id: str) -> ExecutionResult | None:
        """Load the terminal provider result retained before intentional cleanup.

        Provider resources are disposable after the evidence barrier. Once a
        cleanup receipt exists, a missing provider object must not erase the
        append-only terminal observation that authorized its removal.
        """

        root = self.run_root(run_id)
        receipt_path = root / "cleanup.json"
        if not receipt_path.is_file():
            return None
        receipt = _load_json_object(
            receipt_path,
            label=f"execution cleanup receipt for run {run_id}",
        )
        if receipt.get("schema") != _CLEANUP_SCHEMA or receipt.get("run_id") != run_id:
            raise ContractError(f"execution cleanup receipt is invalid for run {run_id}")
        submission = self.load(run_id)
        if receipt.get("provider") != submission.provider or receipt.get("provider_id") != submission.provider_id:
            raise ContractError(f"execution cleanup receipt conflicts with run {run_id}")
        try:
            completed_at = datetime.fromisoformat(str(receipt["completed_at"]))
        except (KeyError, ValueError) as error:
            raise ContractError(f"execution cleanup receipt is invalid for run {run_id}") from error
        if completed_at.tzinfo is None or completed_at.utcoffset() is None:
            raise ContractError(f"execution cleanup receipt is invalid for run {run_id}")

        journal_path = root / "reconciliation.jsonl"
        try:
            lines = journal_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            raise ContractError(f"execution cleanup has no retained reconciliation for run {run_id}") from None
        candidates: list[tuple[datetime, ExecutionResult]] = []
        for line_number, line in enumerate(lines, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ContractError(
                    f"execution reconciliation journal is invalid for run {run_id} at line {line_number}"
                ) from error
            if not isinstance(payload, dict):
                raise ContractError(
                    f"execution reconciliation journal is invalid for run {run_id} at line {line_number}"
                )
            if (
                payload.get("schema") != _RECONCILIATION_SCHEMA
                or payload.get("run_id") != run_id
                or payload.get("outcome") != receipt.get("outcome")
            ):
                continue
            state = payload.get("state")
            evidence_state = receipt.get("evidence_state")
            if not (
                (evidence_state == "reconciled" and state == "consistent")
                or (evidence_state == "provider-terminal" and state == "pending")
            ):
                continue
            result = _execution_result_from_reconciliation(
                payload,
                submission=submission,
                run_id=run_id,
                line_number=line_number,
            )
            if result.record.observed_at <= completed_at:
                candidates.append((result.record.observed_at, result))
        if not candidates:
            raise ContractError(f"execution cleanup has no matching terminal reconciliation for run {run_id}")
        return max(candidates, key=lambda item: item[0])[1]

    def record_cancel_intent(self, run_id: str) -> datetime:
        """Persist cancellation intent before a provider is contacted."""

        path = self.cancel_intent_path(run_id)
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if (
                    isinstance(payload, dict)
                    and payload.get("schema") == _CANCEL_SCHEMA
                    and payload.get("run_id") == run_id
                ):
                    return datetime.fromisoformat(str(payload["requested_at"]))
            except (json.JSONDecodeError, KeyError, ValueError) as error:
                raise ContractError(f"execution cancellation intent is invalid for run {run_id}") from error
            raise ContractError(f"execution cancellation intent is invalid for run {run_id}")

        requested_at = datetime.now(UTC)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (
            json.dumps(
                {
                    "schema": _CANCEL_SCHEMA,
                    "run_id": run_id,
                    "requested_at": requested_at.isoformat(),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return self.record_cancel_intent(run_id)
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        finally:
            os.close(descriptor)
        return requested_at

    def load(self, run_id: str) -> ExecutionSubmission:
        path = self.submission_path(run_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"execution submission is missing for run {run_id}") from None
        except json.JSONDecodeError as error:
            raise ContractError(f"execution submission is invalid for run {run_id}") from error
        if not isinstance(payload, dict) or payload.get("schema") not in _SUPPORTED_SCHEMAS:
            raise ContractError(f"execution submission schema is unsupported for run {run_id}")
        submission = _submission_from_payload(payload)
        if submission.run_id != run_id:
            raise ContractError("execution submission run id conflicts with its state path")
        return submission

    def load_optional(self, run_id: str) -> ExecutionSubmission | None:
        try:
            return self.load(run_id)
        except FileNotFoundError:
            return None

    def list_submissions(self) -> tuple[ExecutionSubmission, ...]:
        """List durable submissions newest first and fail closed on corrupt state.

        Run roots may exist for sidecar receipts (tracking recovery, reconciliation)
        without a submission yet; those directories are skipped. A present but unreadable
        or schema-invalid ``submission.json`` still fails closed.
        """

        if not self._root.exists():
            return ()
        submissions: list[ExecutionSubmission] = []
        for run_root in sorted(self._root.iterdir(), key=lambda path: path.name):
            if not run_root.is_dir():
                raise ContractError(f"unexpected execution state entry is not a directory: {run_root}")
            if not self.submission_path(run_root.name).is_file():
                continue
            submissions.append(self.load(run_root.name))
        return tuple(
            sorted(
                submissions,
                key=lambda submission: (
                    submission.submitted_at,
                    submission.run_id,
                ),
                reverse=True,
            )
        )

    def save(self, submission: ExecutionSubmission) -> ExecutionSubmission:
        existing = self.load_optional(submission.run_id)
        if existing is not None:
            return _same_or_conflict(existing, submission)

        path = self.submission_path(submission.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f".submission-{uuid.uuid4().hex}.tmp"
        encoded = (
            json.dumps(
                {
                    "schema": _SCHEMA,
                    "run_id": submission.run_id,
                    "provider": submission.provider,
                    "provider_id": submission.provider_id,
                    "idempotency_key": submission.idempotency_key,
                    "job_image": submission.job_image,
                    "submitted_at": submission.submitted_at.isoformat(),
                    "required_artifact_roles": submission.required_artifact_roles,
                    "run_workspace": (str(submission.run_workspace) if submission.run_workspace is not None else None),
                    "evidence_source_recorded": submission.evidence_source_recorded,
                    "evidence_source": (
                        {
                            "provider": submission.evidence_source.provider,
                            "source_id": submission.evidence_source.source_id,
                            "project": submission.evidence_source.project,
                            "endpoint": submission.evidence_source.endpoint,
                            "scope": submission.evidence_source.scope,
                        }
                        if submission.evidence_source is not None
                        else None
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
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
        try:
            try:
                os.link(temporary, path)
            except FileExistsError:
                existing = self.load(submission.run_id)
                return _same_or_conflict(existing, submission)
        finally:
            temporary.unlink(missing_ok=True)
        return submission


class JobExecutionService:
    """One lifecycle used by the CLI, SDK, and qualification callers."""

    def __init__(
        self,
        provider: ExecutionProvider,
        store: ExecutionSubmissionStore,
        *,
        provider_name: str,
        evidence_source: ExecutionEvidenceSource | None = None,
    ) -> None:
        if not provider_name.strip():
            raise ValueError("execution provider name cannot be empty")
        self._provider = provider
        self._store = store
        self._provider_name = provider_name
        self._evidence_source = evidence_source

    def plan(self, request: ExecutionRequest) -> ExecutionPlan:
        plan = self._provider.plan(request)
        if plan.provider != self._provider_name:
            raise ContractError(f"execution provider planned as {plan.provider!r}, expected {self._provider_name!r}")
        if plan.request != request:
            raise ContractError("execution provider changed the canonical request while planning")
        return plan

    def submit(self, plan: ExecutionPlan) -> ExecutionSubmission:
        if plan.provider != self._provider_name:
            raise ContractError(f"execution plan provider {plan.provider!r} does not match {self._provider_name!r}")
        if plan.request.bundle is not None:
            raise ContractError(
                "legacy execution bundles are planning-only; pack and submit an immutable actual-job image"
            )
        run_id = plan.request.run_spec.run_id
        existing = self._store.load_optional(run_id)
        if existing is not None:
            _validate_plan_identity(existing, plan)
            if existing.evidence_source_recorded and existing.evidence_source != self._evidence_source:
                raise ContractError(f"execution run {run_id} already has a conflicting evidence source")
            return existing

        self._store.record_submit_intent(plan, self._evidence_source)
        handle = self._provider.submit(plan)
        if handle.provider != self._provider_name:
            raise ContractError(
                f"execution provider returned handle {handle.provider!r}, expected {self._provider_name!r}"
            )
        if handle.idempotency_key != plan.request.idempotency_key:
            raise ContractError("execution provider changed the request idempotency key")
        submission = ExecutionSubmission(
            run_id=run_id,
            provider=handle.provider,
            provider_id=handle.provider_id,
            idempotency_key=handle.idempotency_key,
            job_image=plan.request.image.value,
            submitted_at=datetime.now(UTC),
            required_artifact_roles=plan.request.run_spec.required_artifact_roles,
            run_workspace=_run_workspace(plan.request),
            evidence_source=self._evidence_source,
            evidence_source_recorded=True,
        )
        return self._store.save(submission)

    def status(self, run_id: str) -> ExecutionRecord:
        submission = self._submission(run_id)
        record = self._provider.status(submission.handle)
        _validate_record_handle(submission, record)
        if record.state == "lost":
            cleaned = self._store.cleaned_result(run_id)
            if cleaned is not None:
                record = cleaned.record
        self._store.journal(run_id).append(record)
        return record

    def logs(
        self,
        run_id: str,
        cursor: LogCursor | None = None,
        *,
        limit: int = 200,
    ) -> LogPage:
        submission = self._submission(run_id)
        return self._provider.logs(submission.handle, cursor, limit=limit)

    def cancel(self, run_id: str) -> None:
        submission = self._submission(run_id)
        self._store.record_cancel_intent(run_id)
        self._provider.cancel(submission.handle)

    def collect(self, run_id: str) -> ExecutionResult:
        submission = self._submission(run_id)
        result = self._provider.collect(submission.handle)
        _validate_record_handle(submission, result.record)
        if result.record.state == "lost":
            cleaned = self._store.cleaned_result(run_id)
            if cleaned is not None:
                result = cleaned
        self._store.journal(run_id).append(result.record)
        return result

    def wait(
        self,
        run_id: str,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float = 5.0,
        cancel_on_timeout: bool = False,
        on_transition: Callable[[ExecutionRecord], None] | None = None,
    ) -> ExecutionRecord:
        """Wait for terminal provider state without cancelling unless requested."""

        submission = self._submission(run_id)
        return wait_for_terminal(
            self._provider,
            submission.handle,
            timeout_seconds=timeout_seconds,
            journal=self._store.journal(run_id),
            poll_interval_seconds=poll_interval_seconds,
            cancel_on_timeout=cancel_on_timeout,
            on_transition=on_transition,
        )

    def cleanup(self, run_id: str) -> ProviderCleanupResult:
        submission = self._submission(run_id)
        result = self._provider.cleanup(
            submission.handle,
            run_id=run_id,
            run_workspace=(submission.run_workspace or self._store.default_run_workspace(run_id)),
            runtime_image=RuntimeImageRef(submission.job_image),
        )
        if result.handle != submission.handle:
            raise ContractError("execution provider changed the handle during cleanup")
        return result

    def submission(self, run_id: str) -> ExecutionSubmission:
        """Return the immutable submission used by reconciliation and SDK callers."""

        return self._submission(run_id)

    def _submission(self, run_id: str) -> ExecutionSubmission:
        submission = self._store.load(run_id)
        if submission.provider != self._provider_name:
            raise ContractError(f"execution run uses provider {submission.provider!r}, not {self._provider_name!r}")
        return submission


def _submission_from_payload(payload: dict[str, Any]) -> ExecutionSubmission:
    try:
        submitted_at = datetime.fromisoformat(str(payload["submitted_at"]))
        schema = str(payload["schema"])
        job_image_field = (
            "job_image"
            if schema
            in {
                "posttrain.execution-submission.v4",
                "posttrain.execution-submission.v5",
            }
            else "runtime_image"
        )
        recorded_payload = payload.get("evidence_source_recorded")
        if schema == _SCHEMA and recorded_payload is not True:
            raise ContractError("execution submission v5 must record whether tracking was configured")
        evidence_source_recorded = schema == _SCHEMA
        evidence_payload = payload.get("evidence_source")
        if schema == _SCHEMA and evidence_payload is not None and not isinstance(evidence_payload, dict):
            raise ContractError("execution submission evidence source must be an object or null")
        evidence_source = (
            ExecutionEvidenceSource(
                provider=str(evidence_payload["provider"]),
                source_id=str(evidence_payload["source_id"]),
                project=str(evidence_payload["project"]),
                endpoint=(str(evidence_payload["endpoint"]) if evidence_payload.get("endpoint") is not None else None),
                scope=(str(evidence_payload["scope"]) if evidence_payload.get("scope") is not None else None),
            )
            if schema == _SCHEMA and isinstance(evidence_payload, dict)
            else None
        )
        return ExecutionSubmission(
            run_id=str(payload["run_id"]),
            provider=str(payload["provider"]),
            provider_id=str(payload["provider_id"]),
            idempotency_key=str(payload["idempotency_key"]),
            job_image=str(payload[job_image_field]),
            submitted_at=submitted_at,
            required_artifact_roles=tuple(str(role) for role in payload.get("required_artifact_roles", ())),
            run_workspace=(Path(str(payload["run_workspace"])) if payload.get("run_workspace") is not None else None),
            evidence_source=evidence_source,
            evidence_source_recorded=evidence_source_recorded,
            legacy_bundle_digest=(str(payload["bundle_digest"]) if payload.get("bundle_digest") is not None else None),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError("execution submission fields are invalid") from error


def _run_workspace(request: ExecutionRequest) -> Path | None:
    workspaces = tuple(mount.instance_path for mount in request.mounts if mount.purpose == "run-workspace")
    if len(workspaces) > 1:
        raise ContractError("execution request has multiple run workspaces")
    return workspaces[0] if workspaces else None


def _same_or_conflict(
    existing: ExecutionSubmission,
    candidate: ExecutionSubmission,
) -> ExecutionSubmission:
    if existing._identity() != candidate._identity():
        raise ContractError(f"execution run {candidate.run_id} already names a different provider submission")
    return existing


def _validate_plan_identity(
    submission: ExecutionSubmission,
    plan: ExecutionPlan,
) -> None:
    request = plan.request
    if (
        submission.provider != plan.provider
        or submission.idempotency_key != request.idempotency_key
        or submission.job_image != request.image.value
    ):
        raise ContractError(f"execution run {submission.run_id} already has a conflicting immutable submission")


def _validate_record_handle(
    submission: ExecutionSubmission,
    record: ExecutionRecord,
) -> None:
    if record.handle != submission.handle:
        raise ContractError("execution provider returned a record for a different handle")


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ContractError(f"{label} is invalid") from error
    if not isinstance(payload, dict):
        raise ContractError(f"{label} is invalid")
    return payload


def _execution_result_from_reconciliation(
    payload: dict[str, Any],
    *,
    submission: ExecutionSubmission,
    run_id: str,
    line_number: int,
) -> ExecutionResult:
    label = f"execution reconciliation journal for run {run_id} at line {line_number}"
    try:
        record_payload = payload["provider_record"]
        if not isinstance(record_payload, dict):
            raise TypeError
        handle_payload = record_payload["handle"]
        if not isinstance(handle_payload, dict):
            raise TypeError
        handle = ExecutionHandle(
            provider=str(handle_payload["provider"]),
            provider_id=str(handle_payload["provider_id"]),
            idempotency_key=str(handle_payload["idempotency_key"]),
        )
        observed_at = datetime.fromisoformat(str(record_payload["observed_at"]))
        state = str(record_payload["state"])
        if state not in _TERMINAL_EXECUTION_STATES:
            raise ValueError
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError
        exit_code_payload = payload.get("provider_exit_code")
        if exit_code_payload is not None and not isinstance(exit_code_payload, int):
            raise TypeError
        record = ExecutionRecord(
            handle=handle,
            state=state,  # type: ignore[arg-type]
            attempt=int(record_payload["attempt"]),
            target_id=str(record_payload["target_id"]),
            observed_at=observed_at,
            native_state=str(record_payload["native_state"]),
            message=(str(record_payload["message"]) if record_payload.get("message") is not None else None),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError(f"{label} is invalid") from error
    _validate_record_handle(submission, record)
    return ExecutionResult(record=record, exit_code=exit_code_payload)
