from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from posttrain.common import ExecutionTarget
from posttrain.execution import (
    JOB_PACKAGE_WORKER_COMMAND,
    ExecutionHandle,
    ExecutionMount,
    ExecutionPlan,
    ExecutionPolicy,
    ExecutionRecord,
    ExecutionRequest,
    ExecutionResult,
    ExecutionSubmissionStore,
    JobExecutionService,
    LogCursor,
    LogPage,
    ProviderCleanupResult,
    RuntimeImageRef,
    cleanup_execution,
    reconcile_execution,
)
from posttrain.tracking import (
    ArtifactLink,
    ArtifactSet,
    RunDetail,
    RunSpec,
    RunSummary,
    StoredArtifact,
)


class _Provider:
    def __init__(
        self,
        state: str = "succeeded",
        *,
        name: str = "fake",
    ) -> None:
        self.state = state
        self.name = name
        self.cleanup_calls = 0
        self.cleanup_result: ProviderCleanupResult | None = None

    def plan(self, request: ExecutionRequest) -> ExecutionPlan:
        return ExecutionPlan(self.name, request)

    def submit(self, plan: ExecutionPlan) -> ExecutionHandle:
        return ExecutionHandle(
            self.name,
            "provider-cleanup-1",
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

    def logs(
        self,
        handle: ExecutionHandle,
        cursor: LogCursor | None = None,
        *,
        limit: int = 200,
    ) -> LogPage:
        del handle
        offset = (cursor or LogCursor()).offset
        lines = ("startup", "failure detail")[offset : offset + limit]
        return LogPage(lines, LogCursor(offset + len(lines)), False)

    def cancel(self, handle: ExecutionHandle) -> None:
        del handle

    def collect(self, handle: ExecutionHandle) -> ExecutionResult:
        return ExecutionResult(
            self.status(handle),
            0 if self.state == "succeeded" else 1,
        )

    def cleanup(
        self,
        handle: ExecutionHandle,
        *,
        run_id: str,
        run_workspace: Path | None,
        runtime_image: RuntimeImageRef,
    ) -> ProviderCleanupResult:
        del run_id, run_workspace, runtime_image
        self.cleanup_calls += 1
        if self.cleanup_result is not None:
            return self.cleanup_result
        return ProviderCleanupResult(handle, "removed", "removed fake execution")


class _Source:
    def __init__(self, *, unavailable: bool = False) -> None:
        self.unavailable = unavailable

    async def get_run(self, run_id: str) -> RunDetail:
        if self.unavailable:
            raise LookupError(run_id)
        now = datetime.now(UTC)
        return RunDetail(
            summary=RunSummary(
                provider="trackio",
                provider_run_id="trackio-cleanup-1",
                run_id=run_id,
                display_name=run_id,
                project_id="tests",
                work_package_id="train/cleanup",
                stage="train",
                job_kind="train.sft",
                job_definition_version="train/sft@1",
                status="succeeded",
                started_at=now,
                finished_at=now,
            )
        )

    async def artifacts(self, run_id: str) -> ArtifactSet:
        del run_id
        return ArtifactSet(
            items=(
                ArtifactLink(
                    direction="output",
                    logical_name="model-output",
                    kind="model",
                    artifact=StoredArtifact(
                        provider="trackio",
                        namespace="tests",
                        name="model-output",
                        version="v1",
                        digest="sha256:model",
                        provider_metadata={"posttrain_role": "model"},
                    ),
                ),
            )
        )


def _service(
    tmp_path: Path,
    *,
    provider_state: str = "succeeded",
    provider_name: str = "fake",
) -> tuple[_Provider, JobExecutionService, ExecutionSubmissionStore, Path]:
    state = (tmp_path / "state").resolve()
    workspace = state / "runs" / "cleanup-run-1"
    workspace.mkdir(parents=True)
    (workspace / "disposable.bin").write_bytes(b"x" * 32)
    request = ExecutionRequest(
        run_spec=RunSpec(
            project_id="tests",
            work_package_id="train/cleanup",
            stage="train",
            run_id="cleanup-run-1",
            job_kind="train.sft",
            job_definition_version="train/sft@1",
            required_artifact_roles=("model",),
        ),
        job_definition_id="train/sft@1",
        image=RuntimeImageRef(f"registry.lan/posttrain@sha256:{'a' * 64}"),
        target=ExecutionTarget("targets/test", "1", "cuda", 24),
        command=JOB_PACKAGE_WORKER_COMMAND,
        idempotency_key="cleanup-run-1-attempt-1",
        policy=ExecutionPolicy(300),
        mounts=(
            ExecutionMount(
                workspace,
                Path("/opt/posttrain/run/cleanup-run-1"),
                "run-workspace",
            ),
        ),
    )
    provider = _Provider(provider_state, name=provider_name)
    store = ExecutionSubmissionStore(state)
    service = JobExecutionService(provider, store, provider_name=provider_name)
    service.submit(service.plan(request))
    return provider, service, store, workspace


@pytest.mark.asyncio
async def test_cleanup_requires_reconciliation_and_is_idempotent(
    tmp_path: Path,
) -> None:
    provider, service, store, workspace = _service(tmp_path)

    receipt = await cleanup_execution(
        service,
        store,
        _Source(),  # type: ignore[arg-type]
        "cleanup-run-1",
    )
    repeated = await cleanup_execution(
        service,
        store,
        _Source(unavailable=True),  # type: ignore[arg-type]
        "cleanup-run-1",
    )

    assert receipt == repeated
    assert receipt.evidence_state == "reconciled"
    assert receipt.retained_artifact_count == 1
    assert receipt.workspace_disposition == "provider-managed"
    assert workspace.is_dir()
    assert provider.cleanup_calls == 1
    root = store.run_root("cleanup-run-1")
    assert (root / "submission.json").is_file()
    assert (root / "reconciliation.json").is_file()
    assert (root / "cleanup.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_intentional_cleanup_preserves_terminal_result_for_reconciliation(
    tmp_path: Path,
) -> None:
    provider, service, store, _ = _service(tmp_path)

    await cleanup_execution(
        service,
        store,
        _Source(),  # type: ignore[arg-type]
        "cleanup-run-1",
    )
    provider.state = "lost"

    status = service.status("cleanup-run-1")
    collected = service.collect("cleanup-run-1")
    reconciled = await reconcile_execution(
        service,
        _Source(),  # type: ignore[arg-type]
        "cleanup-run-1",
    )

    assert status.state == "succeeded"
    assert collected.record.state == "succeeded"
    assert collected.exit_code == 0
    assert reconciled.state == "consistent"
    assert reconciled.outcome == "succeeded"


@pytest.mark.asyncio
async def test_provider_workspace_cleanup_evidence_is_retained(
    tmp_path: Path,
) -> None:
    provider, service, store, _ = _service(tmp_path)
    provider.cleanup_result = ProviderCleanupResult(
        ExecutionHandle(
            "fake",
            "provider-cleanup-1",
            "cleanup-run-1-attempt-1",
        ),
        "provider-managed",
        "provider retained history",
        workspace_disposition="removed",
        workspace_reclaimed_bytes=73,
    )

    receipt = await cleanup_execution(
        service,
        store,
        _Source(),  # type: ignore[arg-type]
        "cleanup-run-1",
    )

    assert receipt.provider_disposition == "provider-managed"
    assert receipt.workspace_disposition == "removed"
    assert receipt.workspace_reclaimed_bytes == 73


@pytest.mark.asyncio
async def test_pre_assignment_workspace_not_created_evidence_is_retained(
    tmp_path: Path,
) -> None:
    provider, service, store, _ = _service(
        tmp_path,
        provider_state="failed",
        provider_name="dstack",
    )
    provider.cleanup_result = ProviderCleanupResult(
        ExecutionHandle(
            "dstack",
            "provider-cleanup-1",
            "cleanup-run-1-attempt-1",
        ),
        "provider-managed",
        "provider proved no worker assignment",
        workspace_disposition="not-created",
        workspace_reclaimed_bytes=0,
    )

    receipt = await cleanup_execution(
        service,
        store,
        _Source(unavailable=True),  # type: ignore[arg-type]
        "cleanup-run-1",
    )

    assert receipt.evidence_state == "provider-terminal"
    assert receipt.provider_disposition == "provider-managed"
    assert receipt.workspace_disposition == "not-created"
    assert receipt.workspace_reclaimed_bytes == 0
    assert receipt.diagnostic_file == "diagnostic.log"


@pytest.mark.asyncio
async def test_legacy_dstack_provider_managed_receipt_is_upgraded(
    tmp_path: Path,
) -> None:
    provider, service, store, _ = _service(tmp_path, provider_name="dstack")
    legacy = await cleanup_execution(
        service,
        store,
        _Source(),  # type: ignore[arg-type]
        "cleanup-run-1",
    )
    assert legacy.workspace_disposition == "provider-managed"

    provider.cleanup_result = ProviderCleanupResult(
        ExecutionHandle(
            "dstack",
            "provider-cleanup-1",
            "cleanup-run-1-attempt-1",
        ),
        "provider-managed",
        "provider retained history",
        workspace_disposition="removed",
        workspace_reclaimed_bytes=19,
    )
    upgraded = await cleanup_execution(
        service,
        store,
        _Source(unavailable=True),  # type: ignore[arg-type]
        "cleanup-run-1",
    )

    assert provider.cleanup_calls == 2
    assert upgraded.workspace_disposition == "removed"
    assert upgraded.workspace_reclaimed_bytes == 19


@pytest.mark.asyncio
async def test_failed_startup_retains_bounded_diagnostic_before_cleanup(
    tmp_path: Path,
) -> None:
    provider, service, store, _ = _service(tmp_path, provider_state="failed")

    receipt = await cleanup_execution(
        service,
        store,
        _Source(unavailable=True),  # type: ignore[arg-type]
        "cleanup-run-1",
        diagnostic_limit=1,
    )

    assert receipt.evidence_state == "provider-terminal"
    assert receipt.diagnostic_line_count == 1
    assert receipt.diagnostic_digest is not None
    diagnostic = store.run_root("cleanup-run-1") / "diagnostic.log"
    assert diagnostic.read_text(encoding="utf-8") == "startup\n"
    assert diagnostic.stat().st_mode & 0o777 == 0o600
    assert provider.cleanup_calls == 1


@pytest.mark.asyncio
async def test_success_without_tracking_evidence_is_not_cleaned(
    tmp_path: Path,
) -> None:
    provider, service, store, workspace = _service(tmp_path)

    with pytest.raises(RuntimeError, match="consistent retained evidence"):
        await cleanup_execution(
            service,
            store,
            _Source(unavailable=True),  # type: ignore[arg-type]
            "cleanup-run-1",
        )

    assert provider.cleanup_calls == 0
    assert workspace.is_dir()
    assert not (store.run_root("cleanup-run-1") / "cleanup-plan.json").exists()


@pytest.mark.asyncio
async def test_explicitly_untracked_success_keeps_required_output_workspace(
    tmp_path: Path,
) -> None:
    provider, service, store, workspace = _service(tmp_path)

    with pytest.raises(RuntimeError, match="successful untracked run"):
        await cleanup_execution(service, store, None, "cleanup-run-1")

    assert provider.cleanup_calls == 0
    assert workspace.is_dir()
    assert not (store.run_root("cleanup-run-1") / "cleanup-plan.json").exists()


@pytest.mark.asyncio
async def test_local_cleanup_removes_only_the_exact_run_workspace(
    tmp_path: Path,
) -> None:
    _, service, store, workspace = _service(
        tmp_path,
        provider_name="local-docker",
    )
    sibling = workspace.parent / "another-run"
    sibling.mkdir()
    (sibling / "keep.bin").write_bytes(b"keep")

    receipt = await cleanup_execution(
        service,
        store,
        _Source(),  # type: ignore[arg-type]
        "cleanup-run-1",
    )

    assert receipt.workspace_disposition == "removed"
    assert receipt.workspace_reclaimed_bytes == 32
    assert not workspace.exists()
    assert (sibling / "keep.bin").read_bytes() == b"keep"
