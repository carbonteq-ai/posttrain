from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from posttrain.common import (
    ContractError,
    EventObservation,
    LocalArtifactRef,
    MetricBatchObservation,
    MetricObservation,
    ProducedArtifact,
    TraceObservation,
)
from posttrain.tracking import RunOutcome, RunQuery, RunSpec, TraceQuery
from posttrain_tracking_wandb import WandbBackend, WandbDataSource, WandbSettings

STARTED = datetime(2026, 7, 22, 2, 30, tzinfo=UTC)


class FakeArtifact:
    def __init__(self, name: str, type: str, metadata: dict | None = None) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata or {}
        self.files: list[tuple[str, str | None]] = []
        self.contents: str | None = None
        self.version = "v0"
        self.digest = "a" * 64
        self.size = 0

    def add_file(self, path: str, name: str | None = None) -> None:
        self.files.append((path, name))
        self.contents = Path(path).read_text()
        self.size += Path(path).stat().st_size

    def add_dir(self, path: str) -> None:
        self.files.append((path, None))

    def wait(self) -> FakeArtifact:
        return self


class FakeWriterRun:
    def __init__(self) -> None:
        self.id = "provider-id"
        self.logs: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {}
        self.artifacts: list[FakeArtifact] = []
        self.finished: list[int] = []
        self.defined: list[tuple[str, str]] = []

    def define_metric(self, name: str, *, step_metric: str) -> None:
        self.defined.append((name, step_metric))

    def log(self, values: dict[str, Any]) -> None:
        self.logs.append(values)

    def log_artifact(self, artifact: FakeArtifact) -> FakeArtifact:
        self.artifacts.append(artifact)
        return artifact

    def finish(self, *, exit_code: int) -> None:
        self.finished.append(exit_code)


def _spec() -> RunSpec:
    return RunSpec(
        project_id="tests",
        work_package_id="train/qwen",
        stage="train",
        run_id="00000000-0000-4000-8000-000000000301",
        job_kind="train.sft",
        job_definition_version="train/sft@1",
        resolved_inputs={"model": {"selection_id": "qwen"}},
    )


def test_wandb_writer_maps_logical_steps_traces_and_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    run = FakeWriterRun()
    init_arguments: dict[str, Any] = {}

    def fake_init(**kwargs: Any) -> FakeWriterRun:
        init_arguments.update(kwargs)
        return run

    monkeypatch.setattr("posttrain_tracking_wandb.adapter.wandb.init", fake_init)
    monkeypatch.setattr("posttrain_tracking_wandb.adapter.wandb.Artifact", FakeArtifact)

    tracked = WandbBackend(WandbSettings(entity="team", project="tests")).start_run(_spec())
    tracked.event(EventObservation("started", STARTED, {"phase": "train"}))
    tracked.metric(MetricObservation("train/loss", 2.0, 0))
    tracked.metrics(MetricBatchObservation({"train/loss": 1.0}, 1))
    tracked.trace(
        TraceObservation(
            "conversation",
            "trace-1",
            {"messages": [{"role": "assistant", "content": "done"}]},
        )
    )
    outcome = RunOutcome("succeeded", STARTED, STARTED + timedelta(seconds=3))
    tracked.finish(outcome)
    tracked.finish(outcome)

    assert init_arguments["id"] == _spec().run_id
    assert init_arguments["group"] == "train/qwen"
    assert init_arguments["job_type"] == "train.sft"
    assert run.defined == [("*", "posttrain/step")]
    assert [row.get("posttrain/step") for row in run.logs] == [None, 0, 1]
    assert [row["posttrain/sequence"] for row in run.logs] == [0, 1, 2]
    assert run.summary["posttrain/status"] == "succeeded"
    assert run.finished == [0]
    trace_artifact = next(item for item in run.artifacts if item.type == "posttrain-traces")
    assert trace_artifact.contents is not None
    assert json.loads(trace_artifact.contents)["external_id"] == "trace-1"

    with pytest.raises(ContractError, match="different outcome"):
        tracked.finish(RunOutcome("cancelled", STARTED, STARTED + timedelta(seconds=4)))


def test_wandb_writer_resolves_published_artifact_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = FakeWriterRun()
    monkeypatch.setattr(
        "posttrain_tracking_wandb.adapter.wandb.init",
        lambda **kwargs: run,
    )
    monkeypatch.setattr(
        "posttrain_tracking_wandb.adapter.wandb.Artifact",
        FakeArtifact,
    )
    output = tmp_path / "model.bin"
    output.write_text("weights")
    tracked = WandbBackend(
        WandbSettings(entity="team", project="tests")
    ).start_run(_spec())

    tracked.artifact(
        ProducedArtifact(
            "model/final",
            "model",
            LocalArtifactRef(output.resolve(), "b" * 64),
        )
    )
    published = tracked.published_artifacts()

    assert published[0].logical_name == "model/final"
    assert published[0].reference.provider == "wandb"
    assert published[0].reference.namespace == "team/tests"
    assert published[0].reference.version == "v0"
    assert published[0].reference.digest == "a" * 64


class FakePublicArtifact:
    def __init__(
        self,
        tmp_path: Path,
        *,
        name: str,
        type: str,
        metadata: dict[str, Any],
    ) -> None:
        self.id = f"artifact-{name}"
        self.name = name
        self.type = type
        self.metadata = metadata
        self.version = "v0"
        self.digest = "sha256:" + "a" * 64
        self._tmp_path = tmp_path

    def download(self) -> str:
        return str(self._tmp_path)


class FakePublicRun:
    def __init__(self, tmp_path: Path) -> None:
        self.id = _spec().run_id
        self.name = "train.sft-test"
        self.config = {
            "schema_version": 4,
            "provider": "wandb",
            "project_id": "tests",
            "work_package_id": "train/qwen",
            "stage": "train",
            "run_id": self.id,
            "job_kind": "train.sft",
            "job_definition_version": "train/sft@1",
            "started_at": STARTED.isoformat(),
            "resolved_selections": {"model": {"selection_id": "qwen"}},
            "source_metadata": {},
        }
        self.summary = {
            "posttrain/status": "succeeded",
            "posttrain/started_at": STARTED.isoformat(),
            "posttrain/finished_at": (STARTED + timedelta(seconds=3)).isoformat(),
            "train/loss": 1.0,
        }
        self._history = [
            {
                "event/name": "started",
                "event/occurred_at": STARTED.isoformat(),
                "event/attributes": {"phase": "train"},
                "_timestamp": STARTED.timestamp(),
            },
            {"train/loss": 2.0, "posttrain/step": 0, "_timestamp": STARTED.timestamp()},
            {"train/loss": 1.0, "posttrain/step": 1, "_timestamp": STARTED.timestamp() + 1},
        ]
        self._system_history = [
            {
                "_runtime": 1.5,
                "_timestamp": STARTED.timestamp() + 1,
                "system.cpu": 12.5,
                "system.gpu.0.gpu": 80.0,
                "system.gpu.1.gpu": 60.0,
                "system.gpu.0.memoryAllocatedBytes": 2_000_000_000,
                "system.gpu.1.memoryAllocatedBytes": 1_000_000_000,
                "system.proc.memory.rssMB": 512.0,
            },
            {
                "_runtime": 2.5,
                "_timestamp": STARTED.timestamp() + 2,
                "system.cpu": 20.0,
                "system.gpu.0.gpu": 90.0,
                "system.gpu.1.gpu": 70.0,
                "system.gpu.0.memoryAllocatedBytes": 2_500_000_000,
                "system.gpu.1.memoryAllocatedBytes": 1_500_000_000,
                "system.proc.memory.rssMB": 768.0,
            },
        ]
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()
        (trace_dir / "traces.jsonl").write_text(
            json.dumps(
                {
                    "trace_type": "conversation",
                    "external_id": "trace-1",
                    "payload": {"messages": []},
                    "attributes": {},
                }
            )
            + "\n"
        )
        self._logged = [
            FakePublicArtifact(
                trace_dir,
                name="posttrain-traces",
                type="posttrain-traces",
                metadata={"posttrain_role": "traces"},
            ),
            FakePublicArtifact(
                tmp_path,
                name="adapter:v0",
                type="model",
                metadata={"logical_name": "training/adapter"},
            ),
        ]

    def scan_history(self, keys: list[str] | None = None) -> list[dict[str, Any]]:
        if keys is None:
            return self._history
        return [{key: row[key] for key in keys if key in row} for row in self._history]

    def history(
        self,
        *,
        samples: int,
        pandas: bool,
        stream: str,
    ) -> list[dict[str, Any]]:
        assert samples == 10_000
        assert pandas is False
        assert stream == "system"
        return self._system_history

    def logged_artifacts(self) -> list[FakePublicArtifact]:
        return self._logged

    def used_artifacts(self) -> list[FakePublicArtifact]:
        return []


class FakeApi:
    def __init__(self, run: FakePublicRun) -> None:
        self._run = run
        self.flush_calls = 0
        self.flush_thread_id: int | None = None

    def flush(self) -> None:
        self.flush_calls += 1
        self.flush_thread_id = threading.get_ident()

    def runs(self, path: str) -> list[FakePublicRun]:
        assert path == "team/tests"
        return [self._run]

    def run(self, path: str) -> FakePublicRun:
        assert path == f"team/tests/{self._run.id}"
        return self._run


@pytest.mark.asyncio
async def test_wandb_reader_normalizes_public_api(tmp_path: Path) -> None:
    public_run = FakePublicRun(tmp_path)
    api = FakeApi(public_run)
    source = WandbDataSource(WandbSettings(entity="team", project="tests"), api=api)

    summaries = await source.list_runs(RunQuery(job_kinds=("train.sft",)))
    assert api.flush_calls == 1
    assert api.flush_thread_id != threading.get_ident()
    assert summaries[0].duration_seconds == 3
    detail = await source.get_run(_spec().run_id)
    assert detail.events[0].name == "started"
    assert detail.trace_count == 1
    series = await source.metric_series(_spec().run_id, ("train/loss",))
    assert [point.step for point in series[0].points] == [0, 1]
    assert [point.value for point in series[0].points] == [2.0, 1.0]
    system = await source.metric_series(
        _spec().run_id,
        (
            "system/gpu_utilization",
            "system/gpu_vram_used_bytes",
            "system/cpu_percent",
            "system/process_rss_bytes",
            "system/wall_time_s",
        ),
    )
    assert [point.value for point in system[0].points] == [70.0, 80.0]
    assert [point.value for point in system[1].points] == [3_000_000_000, 4_000_000_000]
    assert [point.value for point in system[2].points] == [12.5, 20.0]
    assert [point.value for point in system[3].points] == [512 * 1024**2, 768 * 1024**2]
    assert [point.value for point in system[4].points] == [1.5, 2.5]
    assert system[0].points[0].attributes["provider_metrics"] == [
        "system.gpu.0.gpu",
        "system.gpu.1.gpu",
    ]
    traces = await source.traces(_spec().run_id, TraceQuery())
    assert traces.items[0].external_id == "trace-1"
    artifacts = await source.artifacts(_spec().run_id)
    assert artifacts.outputs[0].logical_name == "training/adapter"
