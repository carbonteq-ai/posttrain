from __future__ import annotations

import hashlib
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import trackio.context_vars as context_vars
from posttrain.common import (
    ContractError,
    EventObservation,
    LocalArtifactRef,
    MetricBatchObservation,
    MetricObservation,
    ProducedArtifact,
    TraceObservation,
)
from posttrain.tracking import (
    ArtifactInput,
    RunError,
    RunOutcome,
    RunOutcomeStatus,
    RunQuery,
    RunSpec,
    RunSummary,
    StoredArtifactRef,
    TraceQuery,
)
from posttrain_tracking_trackio import (
    TrackioBackend,
    TrackioCancelledRunRecovery,
    TrackioDataSource,
    TrackioLifecycleAdmin,
    TrackioProjectCatalog,
    TrackioPurgeActionExecutor,
    TrackioSettings,
    require_remote_trackio_ready,
)
from trackio.sqlite_storage import SQLiteStorage

from packages.tracking.tests.conformance import (
    artifact_input,
    assert_conformance_snapshot,
    conformance_spec,
    emit_conformance_run,
    logical_snapshot,
    terminal_outcome,
)

STARTED = datetime(2026, 7, 22, 2, 0, tzinfo=UTC)


def test_trackio_project_catalog_returns_stable_unique_names(monkeypatch: pytest.MonkeyPatch) -> None:
    class Client:
        def predict(self, *, api_name: str) -> list[str]:
            assert api_name == "/get_all_projects"
            return ["beta", "alpha", "beta"]

    monkeypatch.setattr("posttrain_tracking_trackio.adapter.RemoteClient", lambda _: Client())

    assert TrackioProjectCatalog("http://trackio:7860").list_projects() == ("alpha", "beta")


def test_trackio_lifecycle_admin_maps_digest_bound_run_purge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "sha256:" + "a" * 64

    class Client:
        def run_purge_plan(self, project: str, run_ids: tuple[str, ...]) -> dict[str, Any]:
            assert (project, run_ids) == ("alpha", ("run-a",))
            return {
                "provider": "trackio",
                "project": project,
                "run_ids": ["run-a"],
                "artifacts": [
                    {
                        "version_id": 7,
                        "name": "model",
                        "version": 2,
                        "size_bytes": 128,
                        "consumer_run_ids": [],
                    }
                ],
                "blockers": [],
                "digest": digest,
                "created_at": "2026-08-02T00:00:00+00:00",
            }

        def purge_runs(self, project: str, run_ids: tuple[str, ...], plan_digest: str) -> dict[str, Any]:
            assert (project, run_ids, plan_digest) == ("alpha", ("run-a",), digest)
            return {
                "provider": "trackio",
                "project": project,
                "plan_digest": digest,
                "deleted_provider_run_ids": ["run-a"],
                "deleted_artifact_version_ids": [7],
                "already_absent_provider_run_ids": [],
                "completed_at": "2026-08-02T00:01:00+00:00",
            }

    monkeypatch.setenv("TRACKIO_WRITE_TOKEN", "test-token")
    monkeypatch.setattr(
        "posttrain_tracking_trackio.adapter.RemoteClient",
        lambda *args, **kwargs: Client(),
    )

    admin = TrackioLifecycleAdmin("http://trackio:7860")
    plan = admin.plan_run_purge(project="alpha", provider_run_ids=("run-a",))
    assert plan.artifacts[0].version_id == "7"
    receipt = admin.apply_run_purge(plan)
    assert receipt.deleted_provider_run_ids == ("run-a",)


def test_trackio_purge_action_executor_revalidates_before_apply() -> None:
    class Admin:
        def __init__(self) -> None:
            self.applied = []

        def plan_run_purge(self, *, project: str, provider_run_ids: tuple[str, ...]):
            from posttrain.tracking import TrackingPurgePlan

            return TrackingPurgePlan(
                provider="trackio",
                project=project,
                provider_run_ids=provider_run_ids,
                run_ids=provider_run_ids,
                artifacts=(),
                blockers=(),
                digest="sha256:" + "b" * 64,
                created_at=datetime.now(UTC),
            )

        def apply_run_purge(self, plan):
            self.applied.append(plan.project)

    admin = Admin()
    executor = TrackioPurgeActionExecutor(admin)  # type: ignore[arg-type]
    action = SimpleNamespace(
        action_id="tracking:run-a",
        kind="tracking.delete_run",
        target={"project": "alpha", "provider_run_id": "provider-a"},
    )
    executor.revalidate(action)
    executor.apply(action)
    assert admin.applied == ["alpha"]


def test_trackio_lifecycle_admin_maps_digest_bound_project_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "sha256:" + "d" * 64

    class Client:
        def project_delete_plan(self, project: str) -> dict[str, Any]:
            return {
                "provider": "trackio",
                "project": project,
                "exists": True,
                "runs": 2,
                "artifacts": 1,
                "artifact_versions": 3,
                "artifact_logical_bytes": 128,
                "artifact_storage_bytes": 256,
                "media_storage_bytes": 64,
                "digest": digest,
                "created_at": "2026-08-02T00:00:00+00:00",
            }

        def delete_project(self, project: str, plan_digest: str) -> dict[str, Any]:
            assert (project, plan_digest) == ("alpha", digest)
            return {
                "provider": "trackio",
                "project": project,
                "plan_digest": plan_digest,
                "deleted": True,
                "completed_at": "2026-08-02T00:01:00+00:00",
            }

    monkeypatch.setenv("TRACKIO_WRITE_TOKEN", "test-token")
    monkeypatch.setattr(
        "posttrain_tracking_trackio.adapter.RemoteClient",
        lambda *args, **kwargs: Client(),
    )
    admin = TrackioLifecycleAdmin("http://trackio:7860")
    plan = admin.project_delete_plan(project="alpha")
    assert plan.storage_bytes == 320
    receipt = admin.delete_project(plan)
    assert receipt.deleted is True


@pytest.mark.parametrize("payload", [{"project": "alpha"}, ["alpha", 1], [" "]])
def test_trackio_project_catalog_rejects_malformed_payload(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
) -> None:
    class Client:
        def predict(self, *, api_name: str) -> object:
            del api_name
            return payload

    monkeypatch.setattr("posttrain_tracking_trackio.adapter.RemoteClient", lambda _: Client())

    with pytest.raises(ContractError, match="project"):
        TrackioProjectCatalog("http://trackio:7860").list_projects()


@pytest.fixture
def trackio_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    for module in ("trackio", "trackio.sqlite_storage", "trackio.utils"):
        monkeypatch.setattr(f"{module}.TRACKIO_DIR", tmp_path)
    monkeypatch.setattr("trackio.bucket_storage.TRACKIO_DIR", tmp_path)
    monkeypatch.setattr("trackio.utils.ARTIFACTS_DIR", tmp_path / "artifacts")
    context_vars.current_run.set(None)
    context_vars.current_project.set(None)
    context_vars.current_server.set(None)
    yield tmp_path
    context_vars.current_run.set(None)
    context_vars.current_project.set(None)
    context_vars.current_server.set(None)


def _spec(run_id: str, artifacts: dict[str, ArtifactInput] | None = None) -> RunSpec:
    return RunSpec(
        project_id="conformance",
        work_package_id="train/qwen",
        stage="train",
        run_id=run_id,
        job_kind="train.sft",
        job_definition_version="train/sft@1",
        resolved_inputs={"model": {"selection_id": "models/qwen@bf16"}},
        source_metadata={"revision": "a" * 40},
        artifacts=artifacts or {},
    )


def test_trackio_settings_require_positive_monitor_intervals() -> None:
    with pytest.raises(ValueError, match="GPU log interval"):
        TrackioSettings(gpu_log_interval=0)
    with pytest.raises(ValueError, match="CPU log interval"):
        TrackioSettings(cpu_log_interval=-1)


def test_required_remote_trackio_probe_checks_authenticated_storage_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class StubClient:
        def __init__(self, source: str, **kwargs: object) -> None:
            captured["source"] = source
            captured["client"] = kwargs

        def predict(self, **kwargs: object) -> object:
            captured["probe"] = kwargs
            return {"present": []}

    monkeypatch.setattr(
        "posttrain_tracking_trackio.adapter.RemoteClient",
        StubClient,
    )
    monkeypatch.setenv("TRACKIO_WRITE_TOKEN", "not-serialized")

    require_remote_trackio_ready(
        project="posttrain",
        server_url="https://trackio.example",
    )

    assert captured["source"] == "https://trackio.example"
    client = captured["client"]
    assert isinstance(client, dict)
    assert client["write_token"] == "not-serialized"
    assert client["httpx_kwargs"] == {"timeout": 10.0}
    assert captured["probe"] == {
        "api_name": "/check_artifact_blobs",
        "project": "posttrain",
        "digests": [],
        "hf_token": None,
    }


def test_required_remote_trackio_probe_rejects_missing_destination_or_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRACKIO_WRITE_TOKEN", raising=False)

    with pytest.raises(ContractError, match="SERVER_URL"):
        require_remote_trackio_ready(project="posttrain", server_url=None)
    with pytest.raises(ContractError, match="WRITE_TOKEN"):
        require_remote_trackio_ready(
            project="posttrain",
            server_url="https://trackio.example",
        )


def test_required_remote_trackio_probe_sanitizes_connectivity_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("response included secret-token")

    monkeypatch.setattr(
        "posttrain_tracking_trackio.adapter.RemoteClient",
        FailingClient,
    )
    monkeypatch.setenv("TRACKIO_WRITE_TOKEN", "secret-token")

    with pytest.raises(ContractError, match="TLS trust") as raised:
        require_remote_trackio_ready(
            project="posttrain",
            server_url="https://trackio.example",
        )

    assert "secret-token" not in str(raised.value)


def test_trackio_backend_forwards_monitor_intervals(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class StubRun:
        id = "provider-run"

    def fake_init(**kwargs: object) -> Any:
        captured.update(kwargs)
        return StubRun()

    monkeypatch.setattr("posttrain_tracking_trackio.adapter.trackio.init", fake_init)
    backend = TrackioBackend(
        TrackioSettings(
            project="monitoring",
            auto_log_gpu=True,
            auto_log_cpu=True,
            gpu_log_interval=0.5,
            cpu_log_interval=3.0,
        )
    )

    backend.start_run(_spec("00000000-0000-4000-8000-000000000099"))

    assert captured["gpu_log_interval"] == 0.5
    assert captured["cpu_log_interval"] == 3.0


def test_trackio_backend_resumes_without_replacing_config_or_starting_monitors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class StubRun:
        id = "provider-run"

    def fake_init(**kwargs: object) -> Any:
        captured.update(kwargs)
        return StubRun()

    monkeypatch.setattr("posttrain_tracking_trackio.adapter.trackio.init", fake_init)
    backend = TrackioBackend(TrackioSettings(project="monitoring", auto_log_gpu=True, auto_log_cpu=True))

    tracked = backend.resume_run(
        _spec("00000000-0000-4000-8000-000000000099"),
        started_at=STARTED,
    )

    assert tracked.provider_run_id == "provider-run"
    assert captured["resume"] == "must"
    config = captured["config"]
    assert isinstance(config, dict)
    assert config["started_at"] == STARTED.isoformat()
    assert captured["auto_log_gpu"] is False
    assert captured["auto_log_cpu"] is False


def test_exact_cancelled_recovery_rechecks_identity_and_provider_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "00000000-0000-4000-8000-000000000099"
    history: list[dict[str, object]] = []
    config = {
        "schema_version": 4,
        "provider": "trackio",
        "project_id": "conformance",
        "work_package_id": "train/qwen",
        "stage": "train",
        "run_id": run_id,
        "job_kind": "train.sft",
        "job_definition_version": "train/sft@1",
        "started_at": STARTED.isoformat(),
    }

    class ProviderRun:
        id = "provider-exact"
        name = "train.sft-00000000"

        @property
        def config(self) -> dict[str, object]:
            return config

        def summary(self) -> dict[str, object]:
            return {"config": config, "last_step": 7}

        def history(self, keys=None):
            del keys
            return history

    provider_run = ProviderRun()

    class Api:
        def __init__(self, server_url=None) -> None:
            del server_url

        def runs(self, project: str):
            assert project == "conformance"
            return [provider_run]

    created: dict[str, object] = {}

    class ExactRun:
        def __init__(self, **kwargs: object) -> None:
            created.update(kwargs)
            self.id = str(kwargs["run_id"])

        def log(self, values: dict[str, object]) -> None:
            history.append(values)

        def finish(self) -> None:
            created["finished"] = True

    monkeypatch.setattr(
        "posttrain_tracking_trackio.adapter.trackio.Api",
        Api,
    )
    monkeypatch.setattr(
        "posttrain_tracking_trackio.adapter.TrackioSDKRun",
        ExactRun,
    )
    monkeypatch.setattr(
        "posttrain_tracking_trackio.adapter.require_remote_trackio_ready",
        lambda **kwargs: None,
    )
    monkeypatch.setenv("TRACKIO_WRITE_TOKEN", "secret")
    expected = RunSummary(
        provider="trackio",
        provider_run_id="provider-exact",
        run_id=run_id,
        display_name="train.sft-00000000",
        project_id="conformance",
        work_package_id="train/qwen",
        stage="train",
        job_kind="train.sft",
        job_definition_version="train/sft@1",
        status="running",
        started_at=STARTED,
    )

    disposition = TrackioCancelledRunRecovery(
        TrackioSettings(
            project="conformance",
            server_url="https://trackio.example",
        )
    ).recover_cancelled(
        expected,
        finished_at=STARTED + timedelta(seconds=10),
    )

    assert disposition == "recovered"
    assert created["run_id"] == "provider-exact"
    assert created["config"] == {}
    assert created["initial_last_step"] == 7
    assert created["finished"] is True
    assert history[-1]["run/status"] == "cancelled"


def test_exact_cancelled_recovery_fails_closed_on_ambiguous_canonical_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "00000000-0000-4000-8000-000000000099"

    class ProviderRun:
        def __init__(self, provider_id: str) -> None:
            self.id = provider_id
            self.name = "train.sft-00000000"
            self.config = {"run_id": run_id}

    class Api:
        def __init__(self, server_url=None) -> None:
            del server_url

        def runs(self, project: str):
            del project
            return [ProviderRun("provider-exact"), ProviderRun("duplicate")]

    monkeypatch.setattr(
        "posttrain_tracking_trackio.adapter.trackio.Api",
        Api,
    )
    monkeypatch.setattr(
        "posttrain_tracking_trackio.adapter.require_remote_trackio_ready",
        lambda **kwargs: None,
    )
    expected = RunSummary(
        provider="trackio",
        provider_run_id="provider-exact",
        run_id=run_id,
        display_name="train.sft-00000000",
        project_id="conformance",
        work_package_id="train/qwen",
        stage="train",
        job_kind="train.sft",
        job_definition_version="train/sft@1",
        status="running",
        started_at=STARTED,
    )

    with pytest.raises(ContractError, match="exactly one"):
        TrackioCancelledRunRecovery(
            TrackioSettings(
                project="conformance",
                server_url="https://trackio.example",
            )
        ).recover_cancelled(
            expected,
            finished_at=STARTED + timedelta(seconds=10),
        )


def _verifiers_trace() -> dict:
    return {
        "id": "rollout-1",
        "version": 2,
        "agent": {"model": "org/model"},
        "task": {"type": "ExampleTask", "data": {"idx": 1}},
        "nodes": [
            {"message": {"role": "user", "content": "2+2?"}},
            {"parent": 0, "message": {"role": "assistant", "content": "4"}},
        ],
        "calls": [],
        "rewards": {"correct": 1.0},
        "metrics": {},
        "errors": [],
        "stop_condition": "agent_completed",
        "is_completed": True,
    }


@pytest.mark.asyncio
async def test_trackio_round_trips_timing_only_inference_trace(trackio_dir: Path) -> None:
    backend = TrackioBackend(TrackioSettings(project="trackio-inference-timing"))
    tracked = backend.start_run(_spec("00000000-0000-4000-8000-000000000102"))
    tracked.trace(
        TraceObservation(
            "inference",
            "request-1",
            {
                "queue_seconds": 0.001,
                "prefill_seconds": 0.010,
                "decode_seconds": 0.200,
            },
        )
    )
    tracked.finish(RunOutcome("succeeded", STARTED, STARTED + timedelta(seconds=1)))

    source = TrackioDataSource("trackio-inference-timing")
    traces = await source.traces(tracked.run_id, TraceQuery(trace_type="inference"))

    assert len(traces.items) == 1
    assert traces.items[0].external_id == "request-1"
    assert traces.items[0].payload["messages"] == []
    assert traces.items[0].payload["prefill_seconds"] == 0.010


@pytest.mark.asyncio
async def test_trackio_trace_reader_pages_beyond_provider_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {
            "id": f"trace-{index}",
            "messages": [],
            "metadata": {
                "external_id": f"request-{index}",
                "observation_type": "inference",
                "posttrain_payload_extra": {"queue_seconds": index / 1000},
            },
        }
        for index in range(1001)
    ]

    class ProviderRun:
        def traces(self, *, limit: int, offset: int, sort: str) -> list[dict[str, Any]]:
            assert sort == "step_asc"
            return records[offset : offset + limit]

    source = TrackioDataSource("trackio-trace-pages")
    monkeypatch.setattr(source, "_provider_run", lambda _run_id: ProviderRun())

    first = await source.traces("run-1", TraceQuery(trace_type="inference", limit=1000))
    second = await source.traces(
        "run-1",
        TraceQuery(trace_type="inference", cursor=first.next_cursor, limit=1000),
    )

    assert len(first.items) == 1000
    assert first.next_cursor == "1000"
    assert len(second.items) == 1
    assert second.items[0].external_id == "request-1000"


@pytest.mark.asyncio
async def test_trackio_trace_reader_passes_verifiers_filter_before_paging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class ProviderRun:
        def traces(
            self,
            *,
            limit: int,
            offset: int,
            sort: str,
            trace_type: str | None = None,
        ) -> list[dict[str, Any]]:
            calls.append(
                {
                    "limit": limit,
                    "offset": offset,
                    "sort": sort,
                    "trace_type": trace_type,
                }
            )
            return [
                {
                    "id": "verifier-1",
                    "trace_type": "verifiers",
                    "external_id": "verifier-1",
                    "metadata": {},
                    "payload": {"rewards": {"correct": 1.0}},
                }
            ][offset : offset + limit]

    source = TrackioDataSource("trackio-verifier-pages")
    monkeypatch.setattr(source, "_provider_run", lambda _run_id: ProviderRun())

    page = await source.traces("run-1", TraceQuery(trace_type="verifiers", limit=10))

    assert calls == [{"limit": 10, "offset": 0, "sort": "step_asc", "trace_type": "verifiers"}]
    assert [item.external_id for item in page.items] == ["verifier-1"]
    assert page.next_cursor is None


@pytest.mark.asyncio
async def test_trackio_trace_detail_reads_one_full_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class ProviderRun:
        def traces(self, **kwargs: Any) -> list[dict[str, Any]]:
            calls.append(kwargs)
            return [
                {
                    "id": "trace-1",
                    "external_id": "trace-1",
                    "trace_type": "verifiers",
                    "metadata": {},
                    "payload": {"messages": [{"role": "assistant", "content": "full"}]},
                }
            ]

    source = TrackioDataSource("trackio-trace-detail")
    monkeypatch.setattr(source, "_provider_run", lambda _run_id: ProviderRun())

    record = await source.get_trace("run-1", "trace-1")

    assert record is not None
    assert record.payload["messages"] == [{"role": "assistant", "content": "full"}]
    assert calls == [
        {
            "search": "trace-1",
            "limit": 1,
            "offset": 0,
            "sort": "step_asc",
            "include_payload": True,
        }
    ]


@pytest.mark.asyncio
async def test_trackio_shared_logical_conformance(trackio_dir: Path) -> None:
    project = "trackio-shared-conformance"
    backend = TrackioBackend(TrackioSettings(project=project))
    source = TrackioDataSource(project)

    producer_id = "00000000-0000-4000-8000-000000000091"
    producer = backend.start_run(conformance_spec(producer_id))
    emit_conformance_run(producer, trackio_dir / "producer" / "adapter.bin")
    producer_output = (await source.artifacts(producer_id)).outputs[0]

    consumer_id = "00000000-0000-4000-8000-000000000092"
    input_value = artifact_input(producer_output.artifact)
    consumer = backend.start_run(conformance_spec(consumer_id, artifacts={"starting_model": input_value}))
    materialized = consumer.materialize_inputs({"starting_model": input_value}, trackio_dir / "consumer" / "inputs")
    assert next(materialized["starting_model"].path.rglob("adapter.bin")).read_bytes() == b"adapter"
    emit_conformance_run(consumer, trackio_dir / "consumer" / "adapter.bin")

    snapshot = await logical_snapshot(source, consumer_id)
    assert_conformance_snapshot(
        snapshot,
        run_id=consumer_id,
        status="succeeded",
        expect_input=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "suffix"),
    [
        ("succeeded", "1"),
        ("partial", "2"),
        ("failed", "3"),
        ("cancelled", "4"),
        ("unsupported", "5"),
    ],
)
async def test_trackio_shared_terminal_outcomes(
    trackio_dir: Path,
    status: RunOutcomeStatus,
    suffix: str,
) -> None:
    del trackio_dir
    project = f"trackio-terminal-{suffix}"
    run_id = f"00000000-0000-4000-8000-00000000008{suffix}"
    tracked = TrackioBackend(TrackioSettings(project=project)).start_run(conformance_spec(run_id))
    outcome = terminal_outcome(status)
    tracked.finish(outcome)
    tracked.finish(outcome)

    detail = await TrackioDataSource(project).get_run(run_id)
    assert detail.summary.status == status
    assert detail.summary.error is not None if status == "failed" else detail.summary.error is None


@pytest.mark.asyncio
async def test_trackio_write_read_conformance(trackio_dir: Path) -> None:
    backend = TrackioBackend(TrackioSettings(project="trackio-conformance"))
    tracked = backend.start_run(_spec("00000000-0000-4000-8000-000000000101"))
    tracked.event(EventObservation("operation_started", STARTED, {"phase": "train"}))
    tracked.metric(
        MetricObservation(
            "train/loss",
            2.0,
            0,
            attributes={"observation_source": "verifiers", "source_step": 0},
        )
    )
    tracked.metrics(MetricBatchObservation({"train/loss": 1.0, "train/tokens_per_s": 42.0}, 1))
    SQLiteStorage.bulk_log_system(
        "trackio-conformance",
        "train.sft-00000000",
        [
            {"gpu/mean_utilization": 99, "gpu/total_allocated_memory": 9.0},
            {"gpu/mean_utilization": 40, "gpu/total_allocated_memory": 2.5},
            {"gpu/mean_utilization": 75, "gpu/total_allocated_memory": 3.25},
            {"gpu/mean_utilization": 98, "gpu/total_allocated_memory": 8.0},
        ],
        timestamps=[
            (STARTED - timedelta(seconds=1)).isoformat(),
            STARTED.isoformat(),
            (STARTED + timedelta(seconds=5)).isoformat(),
            (STARTED + timedelta(seconds=6)).isoformat(),
        ],
        run_id=tracked.provider_run_id,
    )
    tracked.trace(
        TraceObservation(
            "conversation",
            "trace-1",
            {"messages": [{"role": "assistant", "content": "done"}]},
            {"split": "test"},
        )
    )
    tracked.trace(TraceObservation("verifiers", "rollout-1", _verifiers_trace()))

    output_path = trackio_dir / "adapter.bin"
    output_path.write_bytes(b"adapter")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    tracked.artifact(
        ProducedArtifact(
            "training/qwen-adapter",
            "model-adapter",
            LocalArtifactRef(output_path.resolve(), digest),
            metadata={"format": "peft"},
        )
    )
    published = tracked.published_artifacts()
    assert len(published) == 1
    assert published[0].logical_name == "training/qwen-adapter"
    assert published[0].reference.provider == "trackio"
    assert published[0].reference.version == "v0"
    assert published[0].reference.digest is not None
    assert published[0].reference.provider_metadata["posttrain_content_digest"] == digest
    assert published[0].reference.provider_metadata["posttrain_content_digest_kind"] == "file"
    outcome = RunOutcome("succeeded", STARTED, STARTED + timedelta(seconds=5))
    tracked.finish(outcome)
    tracked.finish(outcome)

    source = TrackioDataSource("trackio-conformance")
    summaries = await source.list_runs(RunQuery(work_package_id="train/qwen"))
    assert len(summaries) == 1
    assert summaries[0].run_id == tracked.run_id
    assert summaries[0].provider_run_id == tracked.provider_run_id
    assert summaries[0].status == "succeeded"
    assert summaries[0].duration_seconds == 5

    detail = await source.get_run(tracked.run_id)
    assert detail.resolved_inputs == {"model": {"selection_id": "models/qwen@bf16"}}
    assert detail.events[0].name == "operation_started"
    assert detail.metric_names == (
        "system/gpu_utilization",
        "system/gpu_vram_used_bytes",
        "system/wall_time_s",
        "train/loss",
        "train/tokens_per_s",
    )
    assert detail.trace_count == 2

    series = await source.metric_series(tracked.run_id, ("train/loss", "missing"))
    assert [point.value for point in series[0].points] == [2.0, 1.0]
    assert [point.step for point in series[0].points] == [0, 1]
    assert series[0].points[0].attributes == {
        "observation_source": "verifiers",
        "source_step": 0,
    }
    assert series[1].points == ()

    system = await source.metric_series(
        tracked.run_id,
        ("system/gpu_utilization", "system/gpu_vram_used_bytes", "system/wall_time_s"),
    )
    assert [point.value for point in system[0].points] == [40, 75]
    assert [point.value for point in system[1].points] == [2.5 * 1024**3, 3.25 * 1024**3]
    assert system[2].points[-1].value >= system[2].points[0].value

    traces = await source.traces(tracked.run_id, TraceQuery(limit=1))
    assert len(traces.items) == 1
    assert traces.next_cursor == "1"
    remaining = await source.traces(tracked.run_id, TraceQuery(limit=10, cursor=traces.next_cursor))
    assert {item.external_id for item in (*traces.items, *remaining.items)} == {
        "trace-1",
        "rollout-1",
    }

    artifacts = await source.artifacts(tracked.run_id)
    assert artifacts.outputs[0].logical_name == "training/qwen-adapter"
    assert artifacts.outputs[0].artifact.digest is not None
    assert artifacts.outputs[0].artifact.version == "v0"

    with pytest.raises(ContractError, match="different outcome"):
        tracked.finish(RunOutcome("cancelled", STARTED, STARTED + timedelta(seconds=6)))


def test_trackio_artifact_queue_backpressure_drains_before_retry(
    monkeypatch: pytest.MonkeyPatch,
    trackio_dir: Path,
) -> None:
    tracked = TrackioBackend(TrackioSettings(project="trackio-artifact-backpressure")).start_run(
        _spec("00000000-0000-4000-8000-000000000108")
    )
    output = trackio_dir / "diagnostic.log"
    output.write_text("diagnostic\n", encoding="utf-8")
    attempts: list[bool] = []
    drains: list[float | None] = []

    def log_artifact(artifact: Any, *, background: bool = False) -> Any:
        attempts.append(background)
        if len(attempts) == 1:
            raise RuntimeError("Trackio artifact publication queue is full")
        return artifact

    monkeypatch.setattr(tracked._run, "log_artifact", log_artifact)
    monkeypatch.setattr(tracked, "flush_artifacts", lambda *, timeout=None: drains.append(timeout) or ())

    tracked.artifact(
        ProducedArtifact(
            "training/diagnostics/log",
            "training-runtime-log",
            LocalArtifactRef(output.resolve(), hashlib.sha256(output.read_bytes()).hexdigest()),
        )
    )

    assert attempts == [True, True]
    assert drains == [30]


@pytest.mark.asyncio
async def test_direct_canonical_system_metric_precedes_sampler_fallback(
    trackio_dir: Path,
) -> None:
    del trackio_dir
    project = "trackio-direct-system-metric"
    run_id = "00000000-0000-4000-8000-000000000109"
    tracked = TrackioBackend(TrackioSettings(project=project)).start_run(_spec(run_id))
    tracked.metric(MetricObservation("system/gpu_vram_used_bytes", 12_345.0, 0))
    tracked.finish(RunOutcome("succeeded", STARTED, STARTED + timedelta(seconds=1)))

    source = TrackioDataSource(project)
    detail = await source.get_run(run_id)
    assert "system/gpu_vram_used_bytes" in detail.metric_names
    (series,) = await source.metric_series(run_id, ("system/gpu_vram_used_bytes",))

    assert [point.value for point in series.points] == [12_345.0]
    assert [point.step for point in series.points] == [0]


@pytest.mark.asyncio
async def test_trackio_failure_steps_and_input_materialization(trackio_dir: Path) -> None:
    backend = TrackioBackend(TrackioSettings(project="trackio-failure"))
    producer = backend.start_run(_spec("00000000-0000-4000-8000-000000000201"))
    output_path = trackio_dir / "model.bin"
    output_path.write_bytes(b"model")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    producer.artifact(
        ProducedArtifact(
            "model/final",
            "model",
            LocalArtifactRef(output_path.resolve(), digest),
        )
    )
    producer.finish(RunOutcome("succeeded", STARTED, STARTED + timedelta(seconds=1)))

    source = TrackioDataSource("trackio-failure")
    output = (await source.artifacts(producer.run_id)).outputs[0]
    artifact_input = ArtifactInput(
        StoredArtifactRef(
            output.artifact.provider,
            output.artifact.namespace,
            output.artifact.name,
            output.artifact.version,
            output.artifact.digest,
        ),
        "model",
    )
    consumer = backend.start_run(
        _spec(
            "00000000-0000-4000-8000-000000000202",
            {"base_model": artifact_input},
        )
    )
    materialized = consumer.materialize_inputs({"base_model": artifact_input}, trackio_dir / "inputs")
    assert len(materialized["base_model"].digest) == 64
    assert next(materialized["base_model"].path.rglob("model.bin")).read_bytes() == b"model"
    consumer.metric(MetricObservation("train/loss", 1.0, 1))
    with pytest.raises(ContractError, match="nondecreasing"):
        consumer.metric(MetricObservation("train/loss", 2.0, 0))
    consumer.finish(
        RunOutcome(
            "failed",
            STARTED,
            STARTED + timedelta(seconds=2),
            RunError("RuntimeError", "safe failure"),
        )
    )

    detail = await source.get_run(consumer.run_id)
    assert detail.summary.status == "failed"
    assert detail.summary.error is not None
    assert detail.summary.error.message == "safe failure"
    assert (await source.artifacts(consumer.run_id)).inputs[0].logical_name == "model/final"


@pytest.mark.asyncio
async def test_trackio_rejects_materialized_input_with_wrong_expected_digest(
    trackio_dir: Path,
) -> None:
    backend = TrackioBackend(TrackioSettings(project="trackio-input-integrity"))
    producer = backend.start_run(_spec("00000000-0000-4000-8000-000000000211"))
    output_path = trackio_dir / "integrity-model.bin"
    output_path.write_bytes(b"model")
    producer.artifact(
        ProducedArtifact(
            "model/final",
            "model",
            LocalArtifactRef(
                output_path.resolve(),
                hashlib.sha256(output_path.read_bytes()).hexdigest(),
            ),
        )
    )
    published = producer.published_artifacts()[0]
    producer.finish(RunOutcome("succeeded", STARTED, STARTED + timedelta(seconds=1)))
    wrong_digest = "0" * 64
    artifact_input = ArtifactInput(
        StoredArtifactRef(
            published.reference.provider,
            published.reference.namespace,
            published.reference.name,
            published.reference.version,
            published.reference.digest,
            {"posttrain_content_digest": wrong_digest},
        ),
        "model",
    )
    consumer = backend.start_run(
        _spec(
            "00000000-0000-4000-8000-000000000212",
            {"base_model": artifact_input},
        )
    )
    with pytest.raises(ContractError, match="content digest does not match"):
        consumer.materialize_inputs(
            {"base_model": artifact_input},
            trackio_dir / "integrity-inputs",
        )
