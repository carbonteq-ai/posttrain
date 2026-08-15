from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from posttrain.common import ContractError, ExecutionTarget
from posttrain.execution import (
    JOB_PACKAGE_WORKER_COMMAND,
    BundleRef,
    ExecutionEvidenceSource,
    ExecutionHandle,
    ExecutionPlan,
    ExecutionPolicy,
    ExecutionProviderSource,
    ExecutionRecord,
    ExecutionRequest,
    ExecutionResult,
    ExecutionSubmission,
    ExecutionSubmissionStore,
    JobExecutionService,
    LogCursor,
    LogPage,
    ProviderCleanupResult,
    RuntimeImageRef,
)
from posttrain.tracking import RunSpec


class FakeProvider:
    def __init__(self) -> None:
        self.submit_calls = 0
        self.cancelled: list[ExecutionHandle] = []
        self.cleaned: list[ExecutionHandle] = []
        self.state = "running"

    def plan(self, request: ExecutionRequest) -> ExecutionPlan:
        return ExecutionPlan("fake", request, native_plan_id="plan-1")

    def submit(self, plan: ExecutionPlan) -> ExecutionHandle:
        self.submit_calls += 1
        return ExecutionHandle("fake", "provider-run-1", plan.request.idempotency_key)

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
        stream: str = "workload",
    ) -> LogPage:
        del handle, stream
        offset = (cursor or LogCursor()).offset
        lines = ("zero", "one", "two")[offset : offset + limit]
        return LogPage(lines, LogCursor(offset + len(lines)), offset + len(lines) < 3)

    def cancel(self, handle: ExecutionHandle) -> None:
        self.cancelled.append(handle)
        self.state = "cancelled"

    def collect(self, handle: ExecutionHandle) -> ExecutionResult:
        if self.state not in {"succeeded", "failed", "cancelled", "lost"}:
            raise RuntimeError("execution is not terminal")
        return ExecutionResult(self.status(handle), 0 if self.state == "succeeded" else None)

    def cleanup(
        self,
        handle: ExecutionHandle,
        *,
        run_id: str,
        run_workspace: Path | None,
        runtime_image: RuntimeImageRef,
    ) -> ProviderCleanupResult:
        del run_id, run_workspace, runtime_image
        self.cleaned.append(handle)
        return ProviderCleanupResult(handle, "removed", "removed fake execution")


class AcceptedThenInterruptedProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.accepted = False

    def submit(self, plan: ExecutionPlan) -> ExecutionHandle:
        self.submit_calls += 1
        if not self.accepted:
            self.accepted = True
            raise RuntimeError("connection dropped after provider acceptance")
        return ExecutionHandle("fake", "provider-run-1", plan.request.idempotency_key)


def _request(tmp_path: Path) -> ExecutionRequest:
    del tmp_path
    return ExecutionRequest(
        run_spec=RunSpec(
            project_id="tests",
            work_package_id="train/service",
            stage="train",
            run_id="run-service-1",
            job_kind="train.sft",
            job_definition_version="train/sft@1",
        ),
        job_definition_id="train/sft@1",
        image=RuntimeImageRef(f"registry.lan/posttrain@sha256:{'a' * 64}"),
        target=ExecutionTarget("targets/test", "1", "cuda", 24),
        command=JOB_PACKAGE_WORKER_COMMAND,
        idempotency_key="run-service-1-attempt-1",
        policy=ExecutionPolicy(300),
    )


def test_submission_store_is_idempotent_and_rejects_conflicts(tmp_path: Path) -> None:
    store = ExecutionSubmissionStore(tmp_path.resolve())
    submission = ExecutionSubmission(
        run_id="run-service-1",
        provider="fake",
        provider_id="provider-run-1",
        idempotency_key="key-1",
        job_image=f"registry.lan/posttrain@sha256:{'c' * 64}",
        submitted_at=datetime.now(UTC),
        local_image="posttrain-local:" + ("d" * 64),
    )

    assert store.save(submission) == submission
    assert store.load(submission.run_id) == submission
    assert store.submission_path(submission.run_id).stat().st_mode & 0o777 == 0o600
    payload = json.loads(store.submission_path(submission.run_id).read_text(encoding="utf-8"))
    assert payload["schema"] == "posttrain.execution-submission.v6"
    assert payload["evidence_source_recorded"] is True
    assert payload["evidence_source"] is None
    assert payload["job_image"] == submission.job_image
    assert payload["local_image"] == submission.local_image
    assert "bundle_digest" not in payload
    assert "runtime_image" not in payload
    assert store.save(replace(submission, submitted_at=datetime.now(UTC))) == submission

    with pytest.raises(ContractError, match="different provider submission"):
        store.save(replace(submission, provider_id="provider-run-2"))


def test_submission_records_the_exact_execution_policy(tmp_path: Path) -> None:
    store = ExecutionSubmissionStore(tmp_path.resolve())
    submission = ExecutionSubmission(
        run_id="run-with-policy",
        provider="fake",
        provider_id="provider-run-1",
        idempotency_key="key-with-policy",
        job_image=f"registry.lan/posttrain@sha256:{'c' * 64}",
        submitted_at=datetime.now(UTC),
        execution_policy=ExecutionPolicy(timeout_seconds=90_000, max_attempts=2, priority=4),
    )

    loaded = store.load(store.save(submission).run_id)
    assert loaded.execution_policy == ExecutionPolicy(timeout_seconds=90_000, max_attempts=2, priority=4)
    payload = json.loads(store.submission_path(submission.run_id).read_text(encoding="utf-8"))
    assert payload["schema"] == "posttrain.execution-submission.v7"
    assert payload["execution_policy"] == {
        "timeout_seconds": 90_000,
        "max_attempts": 2,
        "priority": 4,
    }


def test_submit_intent_survives_provider_acceptance_crash_and_retry(
    tmp_path: Path,
) -> None:
    store = ExecutionSubmissionStore(tmp_path.resolve())
    provider = AcceptedThenInterruptedProvider()
    service = JobExecutionService(provider, store, provider_name="fake")
    plan = service.plan(_request(tmp_path))

    with pytest.raises(RuntimeError, match="connection dropped"):
        service.submit(plan)

    intent = store.submit_intent_path("run-service-1")
    assert intent.is_file()
    assert intent.stat().st_mode & 0o777 == 0o600
    assert store.load_optional("run-service-1") is None

    submission = service.submit(plan)

    assert submission.provider_id == "provider-run-1"
    assert provider.submit_calls == 2


def test_submission_store_loads_legacy_bundle_receipt_for_lifecycle_only(
    tmp_path: Path,
) -> None:
    store = ExecutionSubmissionStore(tmp_path.resolve())
    path = store.submission_path("legacy-run")
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema": "posttrain.execution-submission.v3",
                "run_id": "legacy-run",
                "provider": "dstack",
                "provider_id": "legacy-provider-run",
                "idempotency_key": "legacy-run-attempt-1",
                "bundle_digest": "b" * 64,
                "runtime_image": f"registry.lan/runtime@sha256:{'c' * 64}",
                "submitted_at": datetime.now(UTC).isoformat(),
                "required_artifact_roles": [],
                "run_workspace": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    submission = store.load("legacy-run")

    assert submission.job_image == f"registry.lan/runtime@sha256:{'c' * 64}"
    assert submission.legacy_bundle_digest == "b" * 64
    assert submission.evidence_source_recorded is False


def test_submission_store_loads_v4_job_image_without_rewriting(
    tmp_path: Path,
) -> None:
    store = ExecutionSubmissionStore(tmp_path.resolve())
    path = store.submission_path("v4-run")
    path.parent.mkdir(parents=True)
    original = (
        json.dumps(
            {
                "schema": "posttrain.execution-submission.v4",
                "run_id": "v4-run",
                "provider": "dstack",
                "provider_id": "v4-provider-run",
                "idempotency_key": "v4-run-attempt-1",
                "job_image": f"registry.lan/job@sha256:{'c' * 64}",
                "submitted_at": datetime.now(UTC).isoformat(),
                "required_artifact_roles": [],
                "run_workspace": None,
            },
            sort_keys=True,
        )
        + "\n"
    )
    path.write_text(original, encoding="utf-8")

    submission = store.load("v4-run")

    assert submission.evidence_source_recorded is False
    assert submission.evidence_source is None
    assert path.read_text(encoding="utf-8") == original


def test_submission_store_round_trips_secret_free_evidence_locator(
    tmp_path: Path,
) -> None:
    store = ExecutionSubmissionStore(tmp_path.resolve())
    source = ExecutionEvidenceSource(
        provider="trackio",
        source_id="trackio-research",
        project="foundation-models",
        endpoint="https://trackio.internal",
    )
    provider_source = ExecutionProviderSource(
        provider="fake",
        profile_id="machine-default",
        binding_fingerprint="a" * 64,
        credential_file=(tmp_path / "provider.env").resolve(),
        dns_servers=("192.0.2.53",),
    )
    submission = ExecutionSubmission(
        run_id="run-evidence-1",
        provider="fake",
        provider_id="provider-run-1",
        idempotency_key="key-1",
        job_image=f"registry.lan/posttrain@sha256:{'c' * 64}",
        submitted_at=datetime.now(UTC),
        evidence_source=source,
        provider_source=provider_source,
    )

    loaded = store.load(store.save(submission).run_id)
    assert loaded.evidence_source == source
    assert loaded.provider_source == provider_source
    payload = store.submission_path(submission.run_id).read_text(encoding="utf-8")
    assert "token" not in payload.lower()
    assert "api_key" not in payload.lower()


@pytest.mark.parametrize(
    "endpoint",
    (
        "https://user:secret@trackio.internal",
        "https://trackio.internal?token=secret",
        "file:///tmp/trackio",
    ),
)
def test_evidence_locator_rejects_credential_bearing_or_non_http_endpoints(
    endpoint: str,
) -> None:
    with pytest.raises(ContractError, match="credential-free HTTP"):
        ExecutionEvidenceSource(
            provider="trackio",
            source_id="trackio-research",
            project="foundation-models",
            endpoint=endpoint,
        )


def test_service_recovers_handle_without_resubmitting(tmp_path: Path) -> None:
    provider = FakeProvider()
    store = ExecutionSubmissionStore((tmp_path / "state").resolve())
    first = JobExecutionService(provider, store, provider_name="fake")
    plan = first.plan(_request(tmp_path))
    submission = first.submit(plan)

    assert submission.execution_policy == plan.request.policy
    payload = json.loads(store.submission_path(submission.run_id).read_text(encoding="utf-8"))
    assert payload["execution_policy"]["timeout_seconds"] == plan.request.policy.timeout_seconds

    second = JobExecutionService(provider, store, provider_name="fake")
    assert second.submit(plan) == submission
    assert provider.submit_calls == 1
    assert second.status(submission.run_id).state == "running"
    journal = store.run_root(submission.run_id) / "journal.jsonl"
    assert len(journal.read_text(encoding="utf-8").splitlines()) == 1


def test_submission_store_lists_newest_first_and_fails_on_corrupt_state(
    tmp_path: Path,
) -> None:
    store = ExecutionSubmissionStore(tmp_path.resolve())
    older = ExecutionSubmission(
        run_id="older-run",
        provider="fake",
        provider_id="provider-older",
        idempotency_key="older-key",
        job_image=f"registry.lan/posttrain@sha256:{'a' * 64}",
        submitted_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    newer = replace(
        older,
        run_id="newer-run",
        provider_id="provider-newer",
        idempotency_key="newer-key",
        submitted_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    store.save(older)
    store.save(newer)

    assert [submission.run_id for submission in store.list_submissions()] == [
        "newer-run",
        "older-run",
    ]

    sidecar_only = store.run_root("recovery-only")
    sidecar_only.mkdir()
    (sidecar_only / "tracking-recovery.json").write_text("{}", encoding="utf-8")
    assert [submission.run_id for submission in store.list_submissions()] == [
        "newer-run",
        "older-run",
    ]

    corrupt = store.run_root("corrupt-run")
    corrupt.mkdir()
    (corrupt / "submission.json").write_text("{", encoding="utf-8")
    with pytest.raises(ContractError, match="corrupt-run"):
        store.list_submissions()


def test_service_wait_does_not_cancel_on_timeout_by_default(tmp_path: Path) -> None:
    provider = FakeProvider()
    store = ExecutionSubmissionStore((tmp_path / "state").resolve())
    service = JobExecutionService(provider, store, provider_name="fake")
    submission = service.submit(service.plan(_request(tmp_path)))

    with pytest.raises(TimeoutError, match="provider-run-1.*running"):
        service.wait(
            submission.run_id,
            timeout_seconds=0.001,
            poll_interval_seconds=0,
        )

    assert provider.cancelled == []


def test_legacy_bundle_can_be_planned_but_not_submitted(tmp_path: Path) -> None:
    provider = FakeProvider()
    service = JobExecutionService(
        provider,
        ExecutionSubmissionStore((tmp_path / "state").resolve()),
        provider_name="fake",
    )
    bundle_path = (tmp_path / "legacy-bundle").resolve()
    request = replace(
        _request(tmp_path),
        bundle=BundleRef(bundle_path, "b" * 64),
    )

    plan = service.plan(request)

    with pytest.raises(ContractError, match="planning-only"):
        service.submit(plan)
    assert provider.submit_calls == 0


def test_service_pages_logs_cancels_and_collects_by_run_id(tmp_path: Path) -> None:
    provider = FakeProvider()
    store = ExecutionSubmissionStore((tmp_path / "state").resolve())
    service = JobExecutionService(provider, store, provider_name="fake")
    submission = service.submit(service.plan(_request(tmp_path)))

    page = service.logs(submission.run_id, LogCursor(1), limit=1)
    assert page.lines == ("one",)
    assert page.next_cursor == LogCursor(2)
    assert page.truncated is True

    with pytest.raises(RuntimeError, match="not terminal"):
        service.collect(submission.run_id)
    service.cancel(submission.run_id)
    assert provider.cancelled == [submission.handle]
    cancel_intent = store.cancel_intent_path(submission.run_id)
    assert cancel_intent.is_file()
    assert cancel_intent.stat().st_mode & 0o777 == 0o600
    first_intent = cancel_intent.read_text(encoding="utf-8")
    service.cancel(submission.run_id)
    assert cancel_intent.read_text(encoding="utf-8") == first_intent
    assert service.collect(submission.run_id).record.state == "cancelled"
    assert service.cleanup(submission.run_id).disposition == "removed"
    assert provider.cleaned == [submission.handle]


def test_service_rejects_conflicting_immutable_resubmission(tmp_path: Path) -> None:
    provider = FakeProvider()
    store = ExecutionSubmissionStore((tmp_path / "state").resolve())
    service = JobExecutionService(provider, store, provider_name="fake")
    request = _request(tmp_path)
    service.submit(service.plan(request))

    with pytest.raises(ContractError, match="conflicting immutable submission"):
        service.submit(service.plan(replace(request, image=RuntimeImageRef(f"x@y@sha256:{'d' * 64}"))))
