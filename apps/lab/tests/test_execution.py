"""Tests for lab execution and tracking."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import trackio
from posttrain.common import (
    CatalogRef,
    LocalArtifactRef,
    ProducedArtifact,
    RunContext,
    TraceObservation,
)
from posttrain.eval import EvaluationEndpoint
from posttrain.tracking import RunOutcome, RunSpec, StoredArtifactRef
from posttrain.train import QWEN35_SFT_SMOKE, QuantizationPlan, SFTSettings, TrainingBinding
from posttrain.work import execute_run_tracked_finalized
from posttrain_lab.catalog import QWEN35_TRL_QLORA, QWEN_35_2B_AWQ_4BIT, open_catalog, resolved_snapshot
from posttrain_lab.execution import ArtifactInput, execute_run, execute_run_tracked
from posttrain_lab.tracking import TrackioObserver, trackio_artifact_name
from posttrain_lab.tracking import verifiers_rollout as read_verifiers_rollout
from posttrain_lab.work_packages import (
    GSM8K_QUALIFICATION,
    QWEN_FOUNDATION_SCREEN,
    resolve_qualification_package,
    resolve_screen_package,
)
from posttrain_tracking_trackio import (
    TrackioBackend,
    TrackioDataSource,
    TrackioSettings,
)
from posttrain_tracking_wandb import WandbBackend, WandbSettings

REVISION = "a" * 40


class FakeRun:
    def __init__(self) -> None:
        self.logs: list[tuple[dict[str, Any], int | None]] = []
        self.artifacts: list[trackio.Artifact] = []
        self.used_artifacts: list[tuple[str, str | None]] = []
        self.finished = 0

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.logs.append((metrics, step))

    def log_artifact(
        self,
        artifact_or_path: trackio.Artifact | str | Path,
        name: str | None = None,
        type: str | None = None,
        aliases: list[str] | None = None,
    ) -> trackio.Artifact:
        del name, type, aliases
        assert isinstance(artifact_or_path, trackio.Artifact)
        self.artifacts.append(artifact_or_path)
        return artifact_or_path

    def use_artifact(self, artifact_or_name: trackio.Artifact | str, type: str | None = None) -> Any:
        assert isinstance(artifact_or_name, str)
        self.used_artifacts.append((artifact_or_name, type))

        class DownloadableArtifact:
            def download(self, root: str | Path | None = None) -> str:
                assert root is not None
                destination = Path(root)
                (destination / "adapter_config.json").write_text("{}\n", encoding="utf-8")
                return str(destination)

        return DownloadableArtifact()

    def finish(self) -> None:
        self.finished += 1


class FakeTrackedRun:
    def __init__(self, run: FakeRun, run_id: str) -> None:
        self.observer = TrackioObserver(run)
        self._run_id = run_id
        self.outcomes: list[RunOutcome] = []

    @property
    def run_id(self) -> str:
        return self._run_id

    def materialize_inputs(self, inputs: Mapping[str, ArtifactInput], root: Path) -> Mapping[str, LocalArtifactRef]:
        return self.observer.materialize_inputs(inputs, root, project="tests")

    def event(self, observation: Any) -> None:
        self.observer.event(observation)

    def metric(self, observation: Any) -> None:
        self.observer.metric(observation)

    def metrics(self, observation: Any) -> None:
        self.observer.metrics(observation)

    def trace(self, observation: Any) -> None:
        self.observer.trace(observation)

    def artifact(self, artifact: ProducedArtifact) -> None:
        self.observer.artifact(artifact)

    def finish(self, outcome: RunOutcome) -> None:
        if self.outcomes and self.outcomes[-1] != outcome:
            raise AssertionError("conflicting outcome")
        if not self.outcomes:
            self.outcomes.append(outcome)


class FakeBackend:
    def __init__(self, run: FakeRun) -> None:
        self.run = run
        self.specs: list[RunSpec] = []
        self.tracked: FakeTrackedRun | None = None

    def start_run(self, spec: RunSpec) -> FakeTrackedRun:
        self.specs.append(spec)
        self.tracked = FakeTrackedRun(self.run, spec.run_id)
        return self.tracked


class FakeWandbRun:
    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}
        self.logs: list[dict[str, Any]] = []
        self.exit_codes: list[int] = []

    def define_metric(self, name: str, *, step_metric: str) -> None:
        del name, step_metric

    def log(self, values: dict[str, Any]) -> None:
        self.logs.append(values)

    def finish(self, *, exit_code: int) -> None:
        self.exit_codes.append(exit_code)


def _spec(**changes: Any) -> RunSpec:
    values = {
        "project_id": "tests",
        "work_package_id": "train/example",
        "stage": "train",
        "run_id": "00000000-0000-4000-8000-000000000001",
        "job_kind": "train.sft",
        "job_definition_version": "train/test@1",
        "resolved_inputs": {"model": {"source_layer": "base"}},
        "source_metadata": {"git_revision": REVISION, "git_dirty": False},
    }
    values.update(changes)
    return RunSpec(**values)


def test_execute_run_uses_ephemeral_workspace_and_canonical_identity() -> None:
    observed: Path | None = None

    def operation(context: RunContext) -> str:
        nonlocal observed
        observed = context.workspace
        assert context.source_metadata["resolved_selections"] == {"model": {"source_layer": "base"}}
        return context.work_package_id

    assert execute_run(_spec(), operation) == "train/example"
    assert observed is not None and not observed.exists()


def test_trackio_observer_projects_trace_and_stages_artifact(tmp_path: Path) -> None:
    run = FakeRun()
    observer = TrackioObserver(run)
    observer.trace(TraceObservation("verifiers", "trace-1", {"id": "trace-1", "version": 1, "nodes": []}))
    output = tmp_path / "result.json"
    output.write_text("{}\n", encoding="utf-8")
    observer.artifact(
        ProducedArtifact(
            "evaluation/native-result",
            "evaluation",
            LocalArtifactRef(output, hashlib.sha256(output.read_bytes()).hexdigest()),
        )
    )

    assert run.logs[0][0]["traces/verifiers"].trace_id == "trace-1"
    assert run.artifacts[0].metadata["logical_name"] == "evaluation/native-result"
    assert trackio_artifact_name("training/models/qwen3.5-2b@bf16/sft/adapter") == (
        "training-models-qwen3.5-2b-bf16-sft-adapter"
    )


def test_execute_run_tracked_uses_backend_and_materializes_inputs() -> None:
    run = FakeRun()
    backend = FakeBackend(run)
    reference = StoredArtifactRef("trackio", "tests", "training-qwen-sft-adapter", "v0")
    spec = _spec(artifacts={"model_adapter": ArtifactInput(reference, "model-adapter")})

    digest = execute_run_tracked(
        spec,
        lambda context: context.input_artifact("model_adapter").digest,
        backend=backend,
        success_status="partial",
    )

    assert len(digest) == 64
    assert backend.specs == [spec]
    assert run.used_artifacts == [("training-qwen-sft-adapter:v0", "model-adapter")]
    assert backend.tracked is not None
    assert backend.tracked.outcomes[0].status == "partial"


def test_execute_run_tracked_finishes_failed_run() -> None:
    run = FakeRun()
    backend = FakeBackend(run)
    with pytest.raises(RuntimeError, match="broken"):
        execute_run_tracked(
            _spec(),
            lambda _: (_ for _ in ()).throw(RuntimeError("broken")),
            backend=backend,
        )
    assert backend.tracked is not None
    assert backend.tracked.outcomes[0].status == "failed"
    assert backend.tracked.outcomes[0].error is not None
    assert backend.tracked.outcomes[0].error.type == "RuntimeError"


def test_execute_run_tracked_finishes_interrupted_run_as_cancelled() -> None:
    backend = FakeBackend(FakeRun())
    with pytest.raises(KeyboardInterrupt):
        execute_run_tracked(
            _spec(),
            lambda _: (_ for _ in ()).throw(KeyboardInterrupt()),
            backend=backend,
        )
    assert backend.tracked is not None
    assert backend.tracked.outcomes[0].status == "cancelled"


def test_lab_executes_synthetic_job_through_trackio_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = FakeRun()
    monkeypatch.setattr(trackio, "init", lambda **_: run)
    backend = TrackioBackend(TrackioSettings(project="tests", auto_log_gpu=False, auto_log_cpu=False))

    result = execute_run_tracked(
        _spec(),
        lambda context: context.metric("train/loss", 1.0, step=0) or "done",
        backend=backend,
    )

    assert result == "done"
    assert any(values.get("run/status") == "succeeded" for values, _ in run.logs)
    assert run.finished == 1


@pytest.mark.asyncio
async def test_finalized_trackio_artifact_is_consumed_by_exact_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for module in ("trackio", "trackio.sqlite_storage", "trackio.utils"):
        monkeypatch.setattr(f"{module}.TRACKIO_DIR", tmp_path)
    monkeypatch.setattr("trackio.utils.ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr("trackio.bucket_storage.TRACKIO_DIR", tmp_path)
    project = "finalized-e2e"
    backend = TrackioBackend(
        TrackioSettings(
            project=project,
            auto_log_gpu=False,
            auto_log_cpu=False,
        )
    )

    def produce(context: RunContext) -> str:
        output = context.workspace / "model.bin"
        output.write_bytes(b"durable-weights")
        context.artifact(
            ProducedArtifact(
                "model/final",
                "model",
                LocalArtifactRef(
                    output.resolve(),
                    hashlib.sha256(output.read_bytes()).hexdigest(),
                ),
            )
        )
        return "produced"

    producer = execute_run_tracked_finalized(
        _spec(run_id="00000000-0000-4000-8000-000000000401"),
        produce,
        backend=backend,
        scratch_root=tmp_path,
    )
    reference = producer.published_artifacts[0].reference
    assert reference.version == "v0"

    consumer = execute_run_tracked_finalized(
        _spec(
            run_id="00000000-0000-4000-8000-000000000402",
            artifacts={"model": ArtifactInput(reference, "model")},
        ),
        lambda context: next(context.input_artifact("model").path.rglob("model.bin")).read_bytes(),
        backend=backend,
        scratch_root=tmp_path,
    )

    assert consumer.value == b"durable-weights"
    source = TrackioDataSource(project)
    producer_artifacts = await source.artifacts("00000000-0000-4000-8000-000000000401")
    consumer_artifacts = await source.artifacts("00000000-0000-4000-8000-000000000402")
    assert producer_artifacts.outputs[0].artifact.version == "v0"
    assert consumer_artifacts.inputs[0].artifact.version == "v0"
    assert consumer_artifacts.inputs[0].artifact.digest == producer_artifacts.outputs[0].artifact.digest


def test_lab_executes_synthetic_job_through_wandb_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = FakeWandbRun()
    monkeypatch.setattr("posttrain_tracking_wandb.adapter.wandb.init", lambda **_: run)
    backend = WandbBackend(WandbSettings(entity="team", project="tests"))

    result = execute_run_tracked(
        _spec(),
        lambda context: context.metric("train/loss", 1.0, step=0) or "done",
        backend=backend,
    )

    assert result == "done"
    assert run.summary["posttrain/status"] == "succeeded"
    assert run.exit_codes == [0]


def test_screen_and_qualification_packages_resolve_catalog_provenance() -> None:
    screen = resolve_screen_package(open_catalog(scope=QWEN_FOUNDATION_SCREEN.project_id), QWEN_FOUNDATION_SCREEN)
    qualification = resolve_qualification_package(
        open_catalog(scope=GSM8K_QUALIFICATION.project_id),
        GSM8K_QUALIFICATION,
    )
    request = qualification.request(
        EvaluationEndpoint("http://127.0.0.1:8000/v1", screen.request.inference.model.base.repo_id)
    )

    assert screen.request.inference.id == QWEN_FOUNDATION_SCREEN.bindings["screen_inference"].id
    assert request.plan.id == "general-smoke-v1"
    assert request.resolved_budget == (1, 1, 1)
    assert all(screen.snapshot[name]["source_layer"] == "base" for name in QWEN_FOUNDATION_SCREEN.bindings)  # type: ignore[index]


def test_catalog_exposes_typed_training_settings_and_reproducibility_digests() -> None:
    catalog = open_catalog(scope="projects/gsm8k")
    sft_settings = catalog.resolve(CatalogRef("training", QWEN35_SFT_SMOKE.id)).value
    training = catalog.resolve(CatalogRef("training", QWEN35_TRL_QLORA.id)).value
    quantization = catalog.resolve(CatalogRef("quantization", QWEN_35_2B_AWQ_4BIT.id)).value
    snapshot = resolved_snapshot(
        catalog,
        CatalogRef("training", QWEN35_TRL_QLORA.id),
        CatalogRef("quantization", QWEN_35_2B_AWQ_4BIT.id),
    )

    assert isinstance(sft_settings, SFTSettings)
    assert isinstance(training, TrainingBinding)
    assert isinstance(quantization, QuantizationPlan)
    assert len(str(snapshot[f"training/{QWEN35_TRL_QLORA.id}"]["parameter_update_digest"])) == 64  # type: ignore[index]


def test_verifiers_rollout_query_rejects_truncated_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    from trackio.sqlite_storage import SQLiteStorage

    payload = {
        "nodes": [
            {"message": {"role": "user", "content": "question"}},
            {"message": {"role": "assistant", "content": "wrong\n#### 3"}},
        ]
    }
    metadata = {"is_completed": True, "is_truncated": False, "error_type": None, "task_index": 2, "reward": 0.0}
    monkeypatch.setattr(
        SQLiteStorage,
        "get_traces",
        lambda *args, **kwargs: [{"external_id": "trace-1", "payload": payload, "metadata": metadata}],
    )

    assert read_verifiers_rollout("project", "run-1", "trace-1").reward == 0.0
    metadata["is_truncated"] = True
    with pytest.raises(ValueError, match="completed, untruncated, and error-free"):
        read_verifiers_rollout("project", "run-1", "trace-1")
