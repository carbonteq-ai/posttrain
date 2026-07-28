from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Thread

import posttrain.execution.admission as admission_module
import pytest
from posttrain.common import ContractError, ExecutionTarget
from posttrain.execution import (
    ExecutionAdmissionService,
    ExecutionHandle,
    ExecutionPlan,
    ExecutionRecord,
    ExecutionSubmissionStore,
    JobExecutionService,
    ProviderCleanupResult,
)


class FakeProvider:
    def __init__(
        self,
        name: str,
        *,
        fail_submissions: bool = False,
        fail_after_accept_once: bool = False,
    ) -> None:
        self.name = name
        self.fail_submissions = fail_submissions
        self.fail_after_accept_once = fail_after_accept_once
        self.submitted: list[ExecutionPlan] = []
        self.cancelled: list[ExecutionHandle] = []
        self.records: dict[str, ExecutionRecord] = {}

    def plan(self, request):
        return ExecutionPlan(self.name, request)

    def submit(self, plan):
        provider_id = f"{self.name}-{plan.request.run_spec.run_id}"
        if self.fail_submissions:
            raise RuntimeError("provider unavailable")
        handle = ExecutionHandle(
            self.name,
            provider_id,
            plan.request.idempotency_key,
        )
        if provider_id not in self.records:
            self.submitted.append(plan)
            self.records[provider_id] = ExecutionRecord(
                handle,
                "queued",
                1,
                plan.request.target.id,
                datetime.now(UTC),
                "pending",
            )
            if self.fail_after_accept_once:
                self.fail_after_accept_once = False
                raise RuntimeError("provider response was lost")
        return handle

    def status(self, handle):
        return self.records[handle.provider_id]

    def cancel(self, handle):
        self.cancelled.append(handle)

    def logs(self, handle, cursor=None, *, limit=200):
        raise NotImplementedError

    def collect(self, handle):
        raise NotImplementedError

    def cleanup(self, handle, *, run_id, run_workspace, runtime_image):
        return ProviderCleanupResult(handle, "removed", "removed")


def _admission(
    tmp_path: Path,
    providers: dict[str, FakeProvider],
    *,
    physical_host_factory=None,
) -> ExecutionAdmissionService:
    store = ExecutionSubmissionStore(tmp_path.resolve())

    def factory(provider_name, evidence_source):
        return JobExecutionService(
            providers[provider_name],
            store,
            provider_name=provider_name,
            evidence_source=evidence_source,
        )

    return ExecutionAdmissionService(
        tmp_path.resolve(),
        factory,
        physical_host_factory=physical_host_factory,
    )


def _on_worker(plan: ExecutionPlan, hostname: str) -> ExecutionPlan:
    return replace(
        plan,
        request=replace(
            plan.request,
            target=ExecutionTarget(
                f"targets/{hostname}",
                "1",
                "cuda",
                24,
                placement={"instances": [{"hostname": hostname}]},
            ),
        ),
    )


def test_waiting_cancel_never_contacts_any_provider(request_factory, tmp_path) -> None:
    providers = {"local-docker": FakeProvider("local-docker")}
    admission = _admission(tmp_path, providers)
    first = _on_worker(
        ExecutionPlan("local-docker", request_factory("first")),
        "worker-a.lan",
    )
    second = _on_worker(
        ExecutionPlan("local-docker", request_factory("second")),
        "worker-a.lan",
    )

    assert admission.enqueue(first, evidence_source=None).entry.state == "submitted"
    queued = admission.enqueue(second, evidence_source=None).entry
    assert queued.state == "waiting"
    assert queued.position == 1

    cancelled = admission.cancel(second.request.run_spec.run_id)

    assert cancelled.state == "cancelled"
    assert len(providers["local-docker"].submitted) == 1
    assert providers["local-docker"].submitted[0].request.run_spec.run_id == (first.request.run_spec.run_id)
    assert not providers["local-docker"].cancelled


def test_reconciled_terminal_run_releases_exactly_one_waiter(
    request_factory,
    tmp_path,
) -> None:
    providers = {"local-docker": FakeProvider("local-docker")}
    admission = _admission(tmp_path, providers)
    first = _on_worker(
        ExecutionPlan("local-docker", request_factory("first")),
        "worker-a.lan",
    )
    second = _on_worker(
        ExecutionPlan("local-docker", request_factory("second")),
        "worker-a.lan",
    )
    first_run_id = first.request.run_spec.run_id
    second_run_id = second.request.run_spec.run_id
    admission.enqueue(first, evidence_source=None)
    admission.enqueue(second, evidence_source=None)
    first_handle = ExecutionHandle(
        "local-docker",
        f"local-docker-{first_run_id}",
        first.request.idempotency_key,
    )
    providers["local-docker"].records[first_handle.provider_id] = replace(
        providers["local-docker"].records[first_handle.provider_id],
        state="succeeded",
        native_state="done",
    )

    entry, record = admission.status(first_run_id)

    assert record is not None and record.state == "succeeded"
    assert entry.state == "terminal_pending_evidence"
    assert len(providers["local-docker"].submitted) == 1

    admitted = admission.acknowledge_reconciled(first_run_id)

    assert admitted is not None
    assert admitted.entry.run_id == second_run_id
    assert admitted.entry.state == "submitted"
    assert len(providers["local-docker"].submitted) == 2
    assert admission.get(first_run_id).state == "completed"

    assert admission.acknowledge_reconciled(first_run_id) is None
    completed, completed_record = admission.status(first_run_id)
    assert completed.state == "completed"
    assert completed_record is not None
    assert completed_record.state == "succeeded"


def test_snapshot_restores_plan_without_replanning(request_factory, tmp_path) -> None:
    providers = {"local": FakeProvider("local")}
    admission = _admission(tmp_path, providers)
    plan = ExecutionPlan(
        "local",
        request_factory("restart"),
        native_plan_id="native-restart",
        details={"resource": "gpu"},
    )
    admission.enqueue(plan, evidence_source=None)

    restored = _admission(tmp_path, providers).get(plan.request.run_spec.run_id)

    assert restored.plan == plan
    assert restored.state == "submitted"
    assert (tmp_path / "admission" / "queue.json").stat().st_mode & 0o777 == 0o600


def test_independent_worker_placements_are_admitted_concurrently(
    request_factory,
    tmp_path,
) -> None:
    providers = {"local-docker": FakeProvider("local-docker")}
    admission = _admission(tmp_path, providers)
    first_request = replace(
        request_factory("worker-a"),
        target=ExecutionTarget(
            "targets/worker-a",
            "1",
            "cuda",
            24,
            placement={"instances": [{"hostname": "worker-a.lan"}]},
        ),
    )
    second_request = replace(
        request_factory("worker-b"),
        target=ExecutionTarget(
            "targets/worker-b",
            "1",
            "cuda",
            96,
            placement={"instances": [{"hostname": "worker-b.lan"}]},
        ),
    )

    first = admission.enqueue(ExecutionPlan("local-docker", first_request), evidence_source=None)
    second = admission.enqueue(ExecutionPlan("local-docker", second_request), evidence_source=None)

    assert first.entry.state == "submitted"
    assert second.entry.state == "submitted"
    assert len(providers["local-docker"].submitted) == 2


def test_dstack_runs_do_not_queue_behind_each_other(
    request_factory,
    tmp_path,
) -> None:
    """dstack schedules across clients; posttrain must not invent a host lock."""
    providers = {"dstack": FakeProvider("dstack")}
    admission = _admission(tmp_path, providers)
    first = _on_worker(
        ExecutionPlan("dstack", request_factory("first")),
        "worker-a.lan",
    )
    second = _on_worker(
        ExecutionPlan("dstack", request_factory("second")),
        "worker-a.lan",
    )

    assert admission.enqueue(first, evidence_source=None).entry.state == "submitted"
    assert admission.enqueue(second, evidence_source=None).entry.state == "submitted"
    assert len(providers["dstack"].submitted) == 2


def test_submission_failure_quarantines_worker_until_retry_resolves_it(
    request_factory,
    tmp_path,
) -> None:
    failing = FakeProvider("local-docker", fail_submissions=True)
    providers = {"local-docker": failing}
    admission = _admission(tmp_path, providers)
    first = _on_worker(
        ExecutionPlan("local-docker", request_factory("first-fails")),
        "worker-a.lan",
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        admission.enqueue(first, evidence_source=None)

    failed = admission.get(first.request.run_spec.run_id)
    assert failed.state == "submission_failed"
    assert failed.message is not None
    # The cause must survive. A deterministic failure, such as an unset
    # environment variable, otherwise reads as an ambiguous provider response
    # and the advice to retry loops with nothing to act on.
    assert "provider unavailable" in failed.message

    failing.fail_submissions = False
    second = _on_worker(
        ExecutionPlan("local-docker", request_factory("second-runs")),
        "worker-a.lan",
    )
    waiting = admission.enqueue(second, evidence_source=None)

    assert waiting.entry.state == "waiting"
    assert len(failing.submitted) == 0

    failing.fail_submissions = False
    retried = admission.retry_submission(first.request.run_spec.run_id)

    assert retried.entry.state == "submitted"
    assert len(failing.submitted) == 1


def test_ambiguous_accepted_submission_can_be_retried_idempotently(
    request_factory,
    tmp_path,
) -> None:
    provider = FakeProvider("dstack", fail_after_accept_once=True)
    admission = _admission(tmp_path, {"dstack": provider})
    plan = _on_worker(
        ExecutionPlan("dstack", request_factory("accepted-then-lost")),
        "worker-a.lan",
    )
    run_id = plan.request.run_spec.run_id

    with pytest.raises(RuntimeError, match="provider submission is unresolved for run"):
        admission.enqueue(plan, evidence_source=None)

    retried = admission.retry_submission(run_id)

    assert retried.entry.state == "submitted"
    assert retried.submission is not None
    assert len(provider.submitted) == 1


def test_process_death_during_submit_is_recoverable_without_releasing_worker(
    request_factory,
    tmp_path,
) -> None:
    class CrashingProvider(FakeProvider):
        crash_after_accept = True

        def submit(self, plan):
            handle = super().submit(plan)
            if self.crash_after_accept:
                self.crash_after_accept = False
                raise SystemExit(137)
            return handle

    provider = CrashingProvider("local-docker")
    admission = _admission(tmp_path, {"local-docker": provider})
    first = _on_worker(
        ExecutionPlan("local-docker", request_factory("submitter-dies")),
        "worker-a.lan",
    )
    second = _on_worker(
        ExecutionPlan("local-docker", request_factory("must-remain-waiting")),
        "worker-a.lan",
    )

    with pytest.raises(SystemExit, match="137"):
        admission.enqueue(first, evidence_source=None)

    assert admission.get(first.request.run_spec.run_id).state == "submitting"
    assert admission.enqueue(second, evidence_source=None).entry.state == "waiting"

    recovered = admission.retry_submission(first.request.run_spec.run_id)

    assert recovered.entry.state == "submitted"
    assert recovered.submission is not None
    assert len(provider.submitted) == 1


def test_concurrent_enqueues_allow_only_one_provider_submitter(
    request_factory,
    tmp_path,
) -> None:
    class BlockingProvider(FakeProvider):
        def __init__(self, name: str) -> None:
            super().__init__(name)
            self.started = Event()
            self.release = Event()

        def submit(self, plan):
            self.started.set()
            assert self.release.wait(timeout=5)
            return super().submit(plan)

    provider = BlockingProvider("dstack")
    first_service = _admission(tmp_path, {"dstack": provider})
    second_service = _admission(tmp_path, {"dstack": provider})
    plan = _on_worker(
        ExecutionPlan("dstack", request_factory("concurrent-submit")),
        "worker-a.lan",
    )
    results = []
    failures = []

    def submit_first() -> None:
        try:
            results.append(first_service.enqueue(plan, evidence_source=None))
        except BaseException as error:
            failures.append(error)

    thread = Thread(target=submit_first)
    thread.start()
    assert provider.started.wait(timeout=5)

    duplicate = second_service.enqueue(plan, evidence_source=None)

    assert duplicate.entry.state == "submitting"
    assert duplicate.submission is None
    provider.release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert failures == []
    assert len(results) == 1
    assert results[0].entry.state == "submitted"
    assert len(provider.submitted) == 1


def test_next_submission_failure_does_not_undo_completed_reconciliation(
    request_factory,
    tmp_path,
) -> None:
    provider = FakeProvider("local-docker")
    admission = _admission(tmp_path, {"local-docker": provider})
    first = _on_worker(
        ExecutionPlan("local-docker", request_factory("first-completes")),
        "worker-a.lan",
    )
    second = _on_worker(
        ExecutionPlan("local-docker", request_factory("next-fails")),
        "worker-a.lan",
    )
    first_run_id = first.request.run_spec.run_id
    admission.enqueue(first, evidence_source=None)
    admission.enqueue(second, evidence_source=None)
    handle = ExecutionHandle(
        "local-docker",
        f"local-docker-{first_run_id}",
        first.request.idempotency_key,
    )
    provider.records[handle.provider_id] = replace(
        provider.records[handle.provider_id],
        state="succeeded",
        native_state="done",
    )
    admission.status(first_run_id)
    provider.fail_submissions = True

    next_result = admission.acknowledge_reconciled(first_run_id)

    assert admission.get(first_run_id).state == "completed"
    assert next_result is not None
    assert next_result.entry.state == "submission_failed"


def test_restored_waiter_rejects_provider_binding_drift(
    request_factory,
    tmp_path,
) -> None:
    provider = FakeProvider("local-docker")
    store = ExecutionSubmissionStore(tmp_path.resolve())
    binding = ["binding-a"]

    def factory(provider_name, evidence_source):
        return JobExecutionService(
            provider,
            store,
            provider_name=provider_name,
            evidence_source=evidence_source,
        )

    admission = ExecutionAdmissionService(
        tmp_path.resolve(),
        factory,
        provider_binding_factory=lambda provider_name: f"{provider_name}:{binding[0]}",
    )
    first = _on_worker(
        ExecutionPlan("local-docker", request_factory("binding-active")),
        "worker-a.lan",
    )
    second = _on_worker(
        ExecutionPlan("local-docker", request_factory("binding-waiter")),
        "worker-a.lan",
    )
    first_run_id = first.request.run_spec.run_id
    admission.enqueue(first, evidence_source=None)
    admission.enqueue(second, evidence_source=None)
    handle = ExecutionHandle(
        "local-docker",
        f"local-docker-{first_run_id}",
        first.request.idempotency_key,
    )
    provider.records[handle.provider_id] = replace(
        provider.records[handle.provider_id],
        state="succeeded",
        native_state="done",
    )
    admission.status(first_run_id)
    binding[0] = "binding-b"

    next_result = admission.acknowledge_reconciled(first_run_id)

    assert next_result is not None
    assert next_result.entry.state == "submission_failed"
    assert "binding changed" in (next_result.entry.message or "")
    assert len(provider.submitted) == 1


def test_queue_positions_are_scoped_to_one_worker(
    request_factory,
    tmp_path,
) -> None:
    provider = FakeProvider("local-docker")
    admission = _admission(tmp_path, {"local-docker": provider})
    active_a = _on_worker(
        ExecutionPlan("local-docker", request_factory("active-a")),
        "worker-a.lan",
    )
    active_b = _on_worker(
        ExecutionPlan("local-docker", request_factory("active-b")),
        "worker-b.lan",
    )
    waiting_a = _on_worker(
        ExecutionPlan("local-docker", request_factory("waiting-a")),
        "worker-a.lan",
    )
    waiting_b = _on_worker(
        ExecutionPlan("local-docker", request_factory("waiting-b")),
        "worker-b.lan",
    )
    admission.enqueue(active_a, evidence_source=None)
    admission.enqueue(active_b, evidence_source=None)

    assert admission.enqueue(waiting_a, evidence_source=None).entry.position == 1
    assert admission.enqueue(waiting_b, evidence_source=None).entry.position == 1


def test_dstack_admission_does_not_require_a_canonical_hostname(
    request_factory,
    tmp_path,
) -> None:
    providers = {"dstack": FakeProvider("dstack")}
    admission = _admission(tmp_path, providers)

    result = admission.enqueue(
        ExecutionPlan("dstack", request_factory("fleet-selected")),
        evidence_source=None,
    )

    assert result.entry.state == "submitted"
    assert len(providers["dstack"].submitted) == 1


def test_local_target_aliases_share_one_physical_worker_admission(
    request_factory,
    tmp_path,
) -> None:
    provider = FakeProvider("local-docker")
    admission = _admission(tmp_path, {"local-docker": provider})
    first = ExecutionPlan(
        "local-docker",
        replace(
            request_factory("local-alias-a"),
            target=ExecutionTarget("targets/local-a", "1", "cuda", 24),
        ),
    )
    second = ExecutionPlan(
        "local-docker",
        replace(
            request_factory("local-alias-b"),
            target=ExecutionTarget("targets/local-b", "1", "cuda", 24),
        ),
    )

    assert admission.enqueue(first, evidence_source=None).entry.state == "submitted"
    assert admission.enqueue(second, evidence_source=None).entry.state == "waiting"
    assert len(provider.submitted) == 1


def test_local_and_dstack_providers_do_not_share_host_placements(
    request_factory,
    tmp_path,
) -> None:
    providers = {
        "local-docker": FakeProvider("local-docker"),
        "dstack": FakeProvider("dstack"),
    }
    admission = _admission(
        tmp_path,
        providers,
        physical_host_factory=lambda plan: "POP-OS.LAN." if plan.provider == "local-docker" else None,
    )
    local = ExecutionPlan(
        "local-docker",
        replace(
            request_factory("local-controller"),
            target=ExecutionTarget("targets/local", "1", "cuda", 24),
        ),
    )
    remote = _on_worker(
        ExecutionPlan("dstack", request_factory("dstack-controller")),
        "pop-os.lan",
    )

    assert admission.enqueue(local, evidence_source=None).entry.state == "submitted"
    assert admission.enqueue(remote, evidence_source=None).entry.state == "submitted"
    assert len(providers["dstack"].submitted) == 1


def test_shared_admission_root_serializes_two_project_factories(
    request_factory,
    tmp_path,
) -> None:
    """Host placements are machine-scoped: two services, one ledger root."""
    ledger = tmp_path / "machine"
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    provider_a = FakeProvider("local-docker")
    provider_b = FakeProvider("local-docker")
    store_a = ExecutionSubmissionStore(project_a.resolve())
    store_b = ExecutionSubmissionStore(project_b.resolve())

    def factory_a(provider_name, evidence_source):
        return JobExecutionService(
            provider_a,
            store_a,
            provider_name=provider_name,
            evidence_source=evidence_source,
        )

    def factory_b(provider_name, evidence_source):
        return JobExecutionService(
            provider_b,
            store_b,
            provider_name=provider_name,
            evidence_source=evidence_source,
        )

    admission_a = ExecutionAdmissionService(
        ledger.resolve(),
        factory_a,
        physical_host_factory=lambda plan: "pop-os.lan",
    )
    admission_b = ExecutionAdmissionService(
        ledger.resolve(),
        factory_b,
        physical_host_factory=lambda plan: "pop-os.lan",
    )
    first = ExecutionPlan(
        "local-docker",
        replace(
            request_factory("project-a-run"),
            target=ExecutionTarget("targets/local", "1", "cuda", 24),
        ),
    )
    second = ExecutionPlan(
        "local-docker",
        replace(
            request_factory("project-b-run"),
            target=ExecutionTarget("targets/local", "1", "cuda", 24),
        ),
    )

    assert admission_a.enqueue(first, evidence_source=None).entry.state == "submitted"
    waiting = admission_b.enqueue(second, evidence_source=None).entry
    assert waiting.state == "waiting"
    assert waiting.position == 1
    assert len(provider_b.submitted) == 0

    placements = admission_a.placements()
    assert len(placements) == 1
    assert placements[0].key == "host:pop-os.lan"
    assert placements[0].holder == first.request.run_spec.run_id
    assert placements[0].waiting == (second.request.run_spec.run_id,)


def test_snapshot_rejects_dangling_active_placement(
    request_factory,
    tmp_path,
) -> None:
    providers = {"local": FakeProvider("local")}
    admission = _admission(tmp_path, providers)
    plan = ExecutionPlan("local", request_factory("corrupt-active"))
    admission.enqueue(plan, evidence_source=None)
    snapshot = tmp_path / "admission" / "queue.json"
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["active_by_key"] = {"target:missing": "missing-run"}
    snapshot.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ContractError, match="active placement"):
        admission.list()


def test_terminal_snapshot_pruning_retains_compact_receipts(
    request_factory,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(admission_module, "_TERMINAL_RETENTION", 1)
    provider = FakeProvider("dstack")
    admission = _admission(tmp_path, {"dstack": provider})

    for index in range(3):
        plan = _on_worker(
            ExecutionPlan("dstack", request_factory(f"terminal-{index}")),
            "worker-a.lan",
        )
        run_id = plan.request.run_spec.run_id
        admission.enqueue(plan, evidence_source=None)
        provider_id = f"dstack-{run_id}"
        provider.records[provider_id] = replace(
            provider.records[provider_id],
            state="succeeded",
            native_state="done",
        )
        admission.status(run_id)
        admission.acknowledge_reconciled(run_id)

    archive = tmp_path / "admission" / "terminal"
    receipts = list(archive.glob("*.json"))
    assert receipts
    archived = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert archived["schema"] == "posttrain.execution-admission-terminal.v1"
    assert archived["state"] == "completed"
    assert datetime.fromisoformat(archived["terminal_at"]).tzinfo is not None
    assert "plan" not in archived
    assert receipts[0].stat().st_mode & 0o777 == 0o600
