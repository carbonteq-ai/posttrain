from __future__ import annotations

import hashlib
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

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
    StoredArtifactRef,
    TraceQuery,
)
from posttrain_tracking_trackio import TrackioBackend, TrackioDataSource, TrackioSettings
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
    tracked.metric(MetricObservation("train/loss", 2.0, 0))
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
