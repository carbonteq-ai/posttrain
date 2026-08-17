from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest
from posttrain.common import ContractError, ExecutionTarget
from posttrain.execution import (
    JOB_PACKAGE_WORKER_COMMAND,
    ExecutionHandle,
    ExecutionPlan,
    ExecutionPolicy,
    ExecutionRecord,
    ExecutionRequest,
    ExecutionResult,
    ExecutionSubmissionStore,
    JobExecutionService,
    ProviderCleanupResult,
    RuntimeImageRef,
    recover_cancelled_tracking,
    recover_terminal_tracking,
    save_tracking_recovery,
)
from posttrain.tracking import RunDetail, RunOutcomeStatus, RunSpec, RunSummary

STARTED = datetime(2026, 7, 26, 23, 4, tzinfo=UTC)


class _Provider:
    def __init__(self, state: str = "cancelled") -> None:
        self.state = state

    def plan(self, request: ExecutionRequest) -> ExecutionPlan:
        return ExecutionPlan("fake", request)

    def submit(self, plan: ExecutionPlan) -> ExecutionHandle:
        return ExecutionHandle(
            "fake",
            "provider-1",
            plan.request.idempotency_key,
        )

    def status(self, handle: ExecutionHandle) -> ExecutionRecord:
        return ExecutionRecord(
            handle,
            self.state,  # type: ignore[arg-type]
            1,
            "targets/test",
            datetime.now(UTC),
            self.state,
        )

    def collect(self, handle: ExecutionHandle) -> ExecutionResult:
        return ExecutionResult(self.status(handle), 143)

    def cleanup(
        self,
        handle: ExecutionHandle,
        *,
        run_id: str,
        run_workspace: Path | None,
        runtime_image: RuntimeImageRef,
    ) -> ProviderCleanupResult:
        del run_id, run_workspace, runtime_image
        return ProviderCleanupResult(handle, "removed", "unused")


class _Source:
    def __init__(
        self,
        *,
        status: str = "running",
        project_id: str = "tests",
        provider_run_id: str | None = "trackio-1",
    ) -> None:
        self.status = status
        self.project_id = project_id
        self.provider_run_id = provider_run_id
        self.reads = 0

    async def get_run(self, run_id: str) -> RunDetail:
        self.reads += 1
        return RunDetail(
            summary=RunSummary(
                provider="trackio",
                provider_run_id=self.provider_run_id,
                run_id=run_id,
                display_name=f"train.sft-{run_id[:8]}",
                project_id=self.project_id,
                work_package_id="train/recovery",
                stage="train",
                job_kind="train.sft",
                job_definition_version="train/sft@1",
                status=self.status,  # type: ignore[arg-type]
                started_at=STARTED,
                finished_at=(STARTED if self.status == "cancelled" else None),
            )
        )


class _Writer:
    def __init__(
        self,
        disposition: Literal["recovered", "already-cancelled"] = "recovered",
        error: Exception | None = None,
    ) -> None:
        self.disposition: Literal["recovered", "already-cancelled"] = disposition
        self.error = error
        self.expected: RunSummary | None = None

    def recover_cancelled(
        self,
        expected: RunSummary,
        *,
        finished_at: datetime,
    ) -> Literal["recovered", "already-cancelled"]:
        assert finished_at >= expected.started_at
        self.expected = expected
        if self.error is not None:
            raise self.error
        return self.disposition

    def recover_terminal(
        self,
        expected: RunSummary,
        *,
        outcome: RunOutcomeStatus,
        finished_at: datetime,
    ) -> Literal["recovered", "already-terminal"]:
        assert finished_at >= expected.started_at
        assert outcome in {"succeeded", "failed", "cancelled"}
        self.expected = expected
        if self.error is not None:
            raise self.error
        return "already-terminal" if self.disposition == "already-cancelled" else "recovered"


def _service(
    tmp_path: Path,
    *,
    provider_state: str = "cancelled",
) -> tuple[JobExecutionService, ExecutionSubmissionStore]:
    store = ExecutionSubmissionStore((tmp_path / "state").resolve())
    provider = _Provider(provider_state)
    service = JobExecutionService(
        provider,  # type: ignore[arg-type]
        store,
        provider_name="fake",
    )
    request = ExecutionRequest(
        run_spec=RunSpec(
            project_id="tests",
            work_package_id="train/recovery",
            stage="train",
            run_id="run-recovery-1",
            job_kind="train.sft",
            job_definition_version="train/sft@1",
        ),
        job_definition_id="train/sft@1",
        image=RuntimeImageRef(f"registry.lan/posttrain@sha256:{'a' * 64}"),
        target=ExecutionTarget("targets/test", "1", "cuda", 24),
        command=JOB_PACKAGE_WORKER_COMMAND,
        idempotency_key="run-recovery-1-attempt-1",
        policy=ExecutionPolicy(300),
    )
    service.submit(service.plan(request))
    return service, store


@pytest.mark.asyncio
async def test_recovery_requires_exact_cancelled_provider_and_writes_audit(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)
    writer = _Writer()

    recovery = await recover_cancelled_tracking(
        service,
        _Source(),  # type: ignore[arg-type]
        writer,  # type: ignore[arg-type]
        "run-recovery-1",
        project_id="tests",
    )
    receipt = save_tracking_recovery(store, recovery)

    assert recovery.disposition == "recovered"
    assert recovery.execution_provider_id == "provider-1"
    assert recovery.tracking_provider_run_id == "trackio-1"
    assert writer.expected is not None
    assert writer.expected.started_at == STARTED
    assert receipt.stat().st_mode & 0o777 == 0o600
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["schema"] == ("posttrain.tracking-cancellation-recovery.v1")
    assert payload["disposition"] == "recovered"


@pytest.mark.asyncio
async def test_recovery_is_idempotent_when_exact_tracking_run_is_cancelled(
    tmp_path: Path,
) -> None:
    service, _ = _service(tmp_path)
    writer = _Writer("already-cancelled")

    recovery = await recover_cancelled_tracking(
        service,
        _Source(status="cancelled"),  # type: ignore[arg-type]
        writer,  # type: ignore[arg-type]
        "run-recovery-1",
        project_id="tests",
    )

    assert recovery.disposition == "already-cancelled"
    assert writer.expected is not None


@pytest.mark.asyncio
async def test_recovery_fails_before_tracking_write_for_wrong_provider_state(
    tmp_path: Path,
) -> None:
    service, _ = _service(tmp_path, provider_state="failed")
    source = _Source()
    writer = _Writer()

    with pytest.raises(ContractError, match="terminal cancelled"):
        await recover_cancelled_tracking(
            service,
            source,  # type: ignore[arg-type]
            writer,  # type: ignore[arg-type]
            "run-recovery-1",
            project_id="tests",
        )

    assert source.reads == 0
    assert writer.expected is None


@pytest.mark.asyncio
async def test_terminal_recovery_finalizes_failed_provider_without_republishing(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path, provider_state="failed")
    writer = _Writer()

    recovery = await recover_terminal_tracking(
        service,
        _Source(),  # type: ignore[arg-type]
        writer,  # type: ignore[arg-type]
        "run-recovery-1",
        project_id="tests",
    )
    receipt = save_tracking_recovery(store, recovery)

    assert recovery.outcome == "failed"
    assert recovery.disposition == "recovered"
    assert writer.expected is not None
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["schema"] == "posttrain.tracking-terminal-recovery.v1"
    assert payload["outcome"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "message"),
    [
        (_Source(project_id="other"), "project identity"),
        (_Source(provider_run_id=None), "exact provider run id"),
        (_Source(status="succeeded"), "running or already cancelled"),
    ],
)
async def test_recovery_fails_closed_on_tracking_mismatch(
    tmp_path: Path,
    source: _Source,
    message: str,
) -> None:
    service, _ = _service(tmp_path)
    writer = _Writer()

    with pytest.raises(ContractError, match=message):
        await recover_cancelled_tracking(
            service,
            source,  # type: ignore[arg-type]
            writer,  # type: ignore[arg-type]
            "run-recovery-1",
            project_id="tests",
        )

    assert writer.expected is None


@pytest.mark.asyncio
async def test_recovery_propagates_writer_ambiguity_without_receipt(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)
    writer = _Writer(error=ContractError("ambiguous canonical tracking run"))

    with pytest.raises(ContractError, match="ambiguous"):
        await recover_cancelled_tracking(
            service,
            _Source(),  # type: ignore[arg-type]
            writer,  # type: ignore[arg-type]
            "run-recovery-1",
            project_id="tests",
        )

    assert not (store.run_root("run-recovery-1") / "tracking-recovery.json").exists()
