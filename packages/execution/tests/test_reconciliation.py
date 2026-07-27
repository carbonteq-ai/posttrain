from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from posttrain.common import ExecutionTarget
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
    reconcile_execution,
    save_reconciliation,
)
from posttrain.tracking import (
    ArtifactLink,
    ArtifactSet,
    RunDetail,
    RunSpec,
    RunSummary,
    SafeRunError,
    StoredArtifact,
)


class _Provider:
    def __init__(self, state: str = "succeeded") -> None:
        self.state = state

    def plan(self, request: ExecutionRequest) -> ExecutionPlan:
        return ExecutionPlan("fake", request)

    def submit(self, plan: ExecutionPlan) -> ExecutionHandle:
        return ExecutionHandle("fake", "provider-1", plan.request.idempotency_key)

    def status(self, handle: ExecutionHandle) -> ExecutionRecord:
        return ExecutionRecord(
            handle,
            self.state,  # type: ignore[arg-type]
            1,
            "targets/test",
            datetime.now(UTC),
            "exited",
        )

    def collect(self, handle: ExecutionHandle) -> ExecutionResult:
        return ExecutionResult(self.status(handle), 0)

    def cleanup(
        self,
        handle: ExecutionHandle,
        *,
        run_id: str,
        run_workspace: Path | None,
        runtime_image: RuntimeImageRef,
    ) -> ProviderCleanupResult:
        del run_id, run_workspace, runtime_image
        return ProviderCleanupResult(handle, "removed", "removed fake execution")


class _Source:
    def __init__(
        self,
        *,
        status: str = "succeeded",
        artifacts: ArtifactSet | None = None,
        unavailable: bool = False,
    ) -> None:
        self.status = status
        self.artifact_set = artifacts or ArtifactSet()
        self.unavailable = unavailable

    async def get_run(self, run_id: str) -> RunDetail:
        if self.unavailable:
            raise LookupError(run_id)
        started = datetime.now(UTC)
        return RunDetail(
            summary=RunSummary(
                provider="trackio",
                provider_run_id="trackio-1",
                run_id=run_id,
                display_name=run_id,
                project_id="tests",
                work_package_id="train/reconcile",
                stage="train",
                job_kind="train.sft",
                job_definition_version="train/sft@1",
                status=self.status,  # type: ignore[arg-type]
                started_at=started,
                finished_at=None if self.status == "running" else started,
                error=(
                    SafeRunError(type="TrainingError", message="training failed")
                    if self.status == "failed"
                    else None
                ),
            )
        )

    async def artifacts(self, run_id: str) -> ArtifactSet:
        del run_id
        return self.artifact_set


def _request(tmp_path: Path) -> ExecutionRequest:
    del tmp_path
    return ExecutionRequest(
        run_spec=RunSpec(
            project_id="tests",
            work_package_id="train/reconcile",
            stage="train",
            run_id="run-reconcile-1",
            job_kind="train.sft",
            job_definition_version="train/sft@1",
            required_artifact_roles=("model", "summary"),
        ),
        job_definition_id="train/sft@1",
        image=RuntimeImageRef(f"registry.lan/posttrain@sha256:{'a' * 64}"),
        target=ExecutionTarget("targets/test", "1", "cuda", 24),
        command=JOB_PACKAGE_WORKER_COMMAND,
        idempotency_key="run-reconcile-1-attempt-1",
        policy=ExecutionPolicy(300),
    )


def _artifacts(*roles: str) -> ArtifactSet:
    return ArtifactSet(
        items=tuple(
            ArtifactLink(
                direction="output",
                logical_name=f"{role}-output",
                kind="model" if role == "model" else "report",
                artifact=StoredArtifact(
                    provider="trackio",
                    namespace="tests",
                    name=f"{role}-output",
                    version="v1",
                    digest=f"sha256:{role}",
                    provider_metadata={"posttrain_role": role},
                ),
            )
            for role in roles
        )
    )


def _service(
    tmp_path: Path,
    *,
    provider_state: str = "succeeded",
) -> tuple[JobExecutionService, ExecutionSubmissionStore]:
    store = ExecutionSubmissionStore((tmp_path / "state").resolve())
    service = JobExecutionService(
        _Provider(provider_state),  # type: ignore[arg-type]
        store,
        provider_name="fake",
    )
    service.submit(service.plan(_request(tmp_path)))
    return service, store


@pytest.mark.asyncio
async def test_reconcile_requires_provider_and_exact_retained_roles(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)

    result = await reconcile_execution(
        service,
        _Source(artifacts=_artifacts("model", "summary")),  # type: ignore[arg-type]
        "run-reconcile-1",
    )

    assert result.state == "consistent"
    assert result.complete is True
    assert result.outcome == "succeeded"
    assert result.missing_artifact_roles == ()
    assert {artifact.role for artifact in result.retained_artifacts} == {
        "model",
        "summary",
    }
    snapshot = save_reconciliation(store, result)
    assert snapshot.stat().st_mode & 0o777 == 0o600
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    assert payload["schema"] == "posttrain.execution-reconciliation.v1"
    assert payload["state"] == "consistent"


@pytest.mark.asyncio
async def test_reconcile_explicitly_untracked_run_uses_provider_terminal_barrier(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)

    result = await reconcile_execution(service, None, "run-reconcile-1")
    save_reconciliation(store, result)

    assert result.state == "consistent"
    assert result.outcome == "succeeded"
    assert result.tracking_status is None
    assert result.retained_artifacts == ()
    assert result.missing_artifact_roles == ("model", "summary")
    assert "explicitly disabled" in result.message


@pytest.mark.asyncio
async def test_reconcile_is_pending_while_tracking_is_unavailable(
    tmp_path: Path,
) -> None:
    service, _ = _service(tmp_path)

    result = await reconcile_execution(
        service,
        _Source(unavailable=True),  # type: ignore[arg-type]
        "run-reconcile-1",
    )

    assert result.state == "pending"
    assert result.missing_artifact_roles == ("model", "summary")
    assert "LookupError" in result.message


@pytest.mark.asyncio
async def test_reconcile_rejects_missing_or_ambiguous_required_role(
    tmp_path: Path,
) -> None:
    service, _ = _service(tmp_path)

    missing = await reconcile_execution(
        service,
        _Source(artifacts=_artifacts("model")),  # type: ignore[arg-type]
        "run-reconcile-1",
    )
    duplicate = await reconcile_execution(
        service,
        _Source(artifacts=_artifacts("model", "summary", "summary")),  # type: ignore[arg-type]
        "run-reconcile-1",
    )

    assert missing.state == "inconsistent"
    assert missing.missing_artifact_roles == ("summary",)
    assert duplicate.state == "inconsistent"
    assert duplicate.missing_artifact_roles == ("summary",)


@pytest.mark.asyncio
async def test_reconcile_keeps_tracking_and_provider_outcome_separate(
    tmp_path: Path,
) -> None:
    service, _ = _service(tmp_path)

    result = await reconcile_execution(
        service,
        _Source(status="failed", artifacts=_artifacts("model", "summary")),  # type: ignore[arg-type]
        "run-reconcile-1",
    )

    assert result.state == "inconsistent"
    assert result.provider_record.state == "succeeded"
    assert result.tracking_status == "failed"
    assert result.outcome == "failed"


@pytest.mark.asyncio
async def test_reconcile_does_not_require_success_outputs_after_cancellation(
    tmp_path: Path,
) -> None:
    service, _ = _service(tmp_path, provider_state="cancelled")

    result = await reconcile_execution(
        service,
        _Source(status="cancelled"),  # type: ignore[arg-type]
        "run-reconcile-1",
    )

    assert result.state == "consistent"
    assert result.outcome == "cancelled"
    assert result.required_artifact_roles == ("model", "summary")
    assert result.missing_artifact_roles == ()
