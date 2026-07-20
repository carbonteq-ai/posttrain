from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import trackio
from posttrain.common import (
    ExecutionContext,
    Invocation,
    Job,
    JobAction,
    LocalArtifactRef,
    ProducedArtifact,
    RunAttempt,
    TraceObservation,
    TrackioArtifactRef,
)
from posttrain.common.profiles import LFM_25_12B_THINKING, QWEN_35_2B
from posttrain.eval.programs import GENERAL_SMOKE
from posttrain.serve import (
    LFM25_VLLM,
    QWEN35_VLLM_TEXT,
    QWEN35_VLLM_TURBOQUANT_K8,
    BenchmarkCell,
    BenchmarkRequest,
    Endpoint,
    GenerationResult,
    LaunchRequest,
    ProbeResult,
)
from posttrain_lab.execution import ArtifactInput, AttemptSpec, execute, execute_tracked
from posttrain_lab.jobs import foundation_screening as foundation_job
from posttrain_lab.jobs.foundation_screening import (
    ManagedEvaluationRequest,
    evaluation_action,
    foundation_screening_job,
    run_managed_evaluation,
    serving_benchmark_action,
)
from posttrain_lab.tracking import TrackioObserver
from posttrain_lab.tracking import verifiers_rollout as read_verifiers_rollout

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


def spec() -> AttemptSpec:
    return AttemptSpec(
        job=Job("tests/example", REVISION, "Example"),
        action=JobAction("tests/example", "evaluate", "evaluation"),
        invocation=Invocation("00000000-0000-4000-8000-000000000001"),
        attempt=RunAttempt("00000000-0000-4000-8000-000000000002", 1),
        inputs={"model": "qwen3.5-2b"},
        source_metadata={"git_revision": REVISION, "git_dirty": False},
    )


def test_execute_uses_ephemeral_workspace() -> None:
    observed: Path | None = None

    def operation(context: ExecutionContext) -> str:
        nonlocal observed
        observed = context.workspace
        assert context.workspace.is_dir()
        context.metric("test/value", 1)
        return "result"

    assert execute(spec(), operation) == "result"
    assert observed is not None
    assert not observed.exists()


def test_trackio_observer_projects_verifiers_trace() -> None:
    run = FakeRun()
    observer = TrackioObserver(run)
    observer.trace(
        TraceObservation(
            trace_type="verifiers",
            external_id="trace-1",
            payload={"id": "trace-1", "version": 1, "nodes": []},
            attributes={"suite": "gsm8k"},
        )
    )
    logged = run.logs[0][0]["traces/verifiers"]
    assert isinstance(logged, trackio.VerifiersTrace)
    assert logged.trace_id == "trace-1"
    assert logged.metadata["suite"] == "gsm8k"


def test_trackio_observer_verifies_and_stages_artifact(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    output.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    artifact = ProducedArtifact(
        name="evaluation/native-result",
        kind="evaluation",
        reference=LocalArtifactRef(output, digest),
    )
    run = FakeRun()
    TrackioObserver(run).artifact(artifact)
    assert run.artifacts[0].name == "evaluation-native-result"
    assert run.artifacts[0].metadata["logical_name"] == artifact.name


def test_execute_tracked_preserves_attempt_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    run = FakeRun()
    init_calls: list[dict[str, Any]] = []

    def fake_init(**kwargs: Any) -> FakeRun:
        init_calls.append(kwargs)
        return run

    attempt = spec()
    monkeypatch.setattr(trackio, "init", fake_init)
    assert execute_tracked(attempt, lambda context: context.attempt.id, project="global-models") == attempt.attempt.id
    assert init_calls[0]["group"] == "tests/example"
    assert init_calls[0]["config"]["invocation_id"] == attempt.invocation.id
    assert init_calls[0]["config"]["attempt_id"] == attempt.attempt.id
    assert run.finished == 1
    assert any(values.get("run/status") == "complete" for values, _ in run.logs)


def test_execute_tracked_finishes_failed_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    run = FakeRun()
    monkeypatch.setattr(trackio, "init", lambda **kwargs: run)

    def broken(context: ExecutionContext) -> None:
        del context
        raise RuntimeError("broken")

    with pytest.raises(RuntimeError, match="broken"):
        execute_tracked(spec(), broken, project="job-tests")

    assert run.finished == 1
    assert any(values.get("run/status") == "failed" for values, _ in run.logs)


def test_execute_tracked_materializes_and_records_input_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    run = FakeRun()
    monkeypatch.setattr(trackio, "init", lambda **kwargs: run)
    base = spec()
    attempt = AttemptSpec(
        job=base.job,
        action=base.action,
        invocation=base.invocation,
        attempt=base.attempt,
        artifacts={
            "model_adapter": ArtifactInput(
                TrackioArtifactRef(
                    project="job-tests",
                    name="training-qwen3.5-2b-sft-adapter",
                    version="v0",
                ),
                "model-adapter",
            )
        },
    )

    def operation(context: ExecutionContext) -> str:
        adapter = context.input_artifact("model_adapter")
        assert adapter.path.joinpath("adapter_config.json").is_file()
        return adapter.digest

    digest = execute_tracked(attempt, operation, project="job-tests")

    assert len(digest) == 64
    assert run.used_artifacts == [("training-qwen3.5-2b-sft-adapter:v0", "model-adapter")]


def test_verifiers_rollout_query_rejects_truncated_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    from trackio.sqlite_storage import SQLiteStorage

    payload = {
        "nodes": [
            {"message": {"role": "user", "content": "question"}},
            {"message": {"role": "assistant", "content": "wrong\n#### 3"}},
        ]
    }
    metadata = {
        "is_completed": True,
        "is_truncated": False,
        "error_type": None,
        "task_index": 2,
        "reward": 0.0,
    }
    monkeypatch.setattr(
        SQLiteStorage,
        "get_traces",
        lambda *args, **kwargs: [{"external_id": "trace-1", "payload": payload, "metadata": metadata}],
    )

    rollout = read_verifiers_rollout("project", "run-1", "trace-1")

    assert rollout.task_index == 2
    assert rollout.response == "wrong\n#### 3"
    assert rollout.reward == 0.0

    metadata["is_truncated"] = True
    with pytest.raises(ValueError, match="completed, untruncated, and error-free"):
        read_verifiers_rollout("project", "run-1", "trace-1")


def test_foundation_screening_action_has_stable_job_owned_identity() -> None:
    request = BenchmarkRequest(
        model=QWEN_35_2B,
        profile=QWEN35_VLLM_TEXT,
        cell=BenchmarkCell("smoke", "short", 1_024, 1, 128, 32, 1, 1),
    )
    job = foundation_screening_job(REVISION)
    action = serving_benchmark_action(request)
    assert action.job_id == job.id
    assert action.id == "serve/qwen3.5-2b/short-ctx1024-c1"


def test_managed_eval_action_is_one_environment_cell() -> None:
    request = ManagedEvaluationRequest(
        LaunchRequest(QWEN_35_2B, QWEN35_VLLM_TURBOQUANT_K8),
        GENERAL_SMOKE,
        "math-gsm8k",
        context_window=8_192,
    )
    action = evaluation_action(request)
    assert action.id == "eval/general/qwen3.5-2b/math-gsm8k"
    assert action.kind == "general-evaluation"


def test_managed_eval_composes_endpoint_without_leaking_serve_into_eval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    endpoint = Endpoint("http://model.test/v1", QWEN_35_2B.artifact.repo_id)

    @contextmanager
    def fake_launch(*args: Any, **kwargs: Any) -> Iterator[Endpoint]:
        del args, kwargs
        yield endpoint

    observed: list[Any] = []
    monkeypatch.setattr(foundation_job, "launch", fake_launch)
    monkeypatch.setattr(
        foundation_job,
        "probe",
        lambda *args, **kwargs: ProbeResult(True, True, 0.1, (endpoint.model,)),
    )
    monkeypatch.setattr(foundation_job, "evaluate", lambda context, request: observed.append(request) or "done")
    managed = ManagedEvaluationRequest(
        LaunchRequest(QWEN_35_2B, QWEN35_VLLM_TURBOQUANT_K8),
        GENERAL_SMOKE,
        "math-gsm8k",
        context_window=8_192,
    )

    assert run_managed_evaluation(
        ExecutionContext(
            job=spec().job,
            action=spec().action,
            invocation=spec().invocation,
            attempt=spec().attempt,
            workspace=tmp_path.resolve(),
        ),
        managed,
    ) == "done"
    assert observed[0].target.base_url == endpoint.base_url
    assert observed[0].program is GENERAL_SMOKE


def test_online_smoke_rejects_a_reasoning_only_truncated_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    endpoint = Endpoint("http://model.test/v1", LFM_25_12B_THINKING.artifact.repo_id)

    @contextmanager
    def fake_launch(*args: Any, **kwargs: Any) -> Iterator[Endpoint]:
        del args, kwargs
        yield endpoint

    monkeypatch.setattr(foundation_job, "launch", fake_launch)
    monkeypatch.setattr(
        foundation_job, "probe", lambda *args, **kwargs: ProbeResult(True, True, 0.1, (endpoint.model,))
    )
    monkeypatch.setattr(
        foundation_job,
        "generate",
        lambda *args, **kwargs: GenerationResult("", "thinking", (), 4, 16, 0.2, 0.1, "length", ()),
    )

    with pytest.raises(RuntimeError, match="no final answer"):
        foundation_job.run_online_smoke(
            ExecutionContext(
                job=spec().job,
                action=spec().action,
                invocation=spec().invocation,
                attempt=spec().attempt,
                workspace=tmp_path.resolve(),
            ),
            LaunchRequest(LFM_25_12B_THINKING, LFM25_VLLM),
        )
