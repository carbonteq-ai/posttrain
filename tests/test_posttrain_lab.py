from __future__ import annotations

import hashlib
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
)
from posttrain_lab.execution import AttemptSpec, execute, execute_tracked
from posttrain_lab.tracking import TrackioObserver

REVISION = "a" * 40


class FakeRun:
    def __init__(self) -> None:
        self.logs: list[tuple[dict[str, Any], int | None]] = []
        self.artifacts: list[trackio.Artifact] = []
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
