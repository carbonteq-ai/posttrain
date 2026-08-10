"""Tests for durable artifact resolution before workspace cleanup."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import pytest
from posttrain.common import (
    ContractError,
    LocalArtifactRef,
    OperationCancelled,
    ProducedArtifact,
    PublishedArtifact,
    StoredArtifactRef,
)
from posttrain.tracking import ArtifactInput, RunOutcome, RunSpec
from posttrain.work import execute_run_tracked, execute_run_tracked_finalized


class PublishingRun:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.outcomes: list[RunOutcome] = []
        self._published: list[PublishedArtifact] = []
        self._source_paths: list[Path] = []

    def materialize_inputs(
        self,
        inputs: Mapping[str, ArtifactInput],
        root: Path,
    ) -> Mapping[str, LocalArtifactRef]:
        del inputs, root
        return {}

    def event(self, observation) -> None:
        del observation

    def metric(self, observation) -> None:
        del observation

    def metrics(self, observation) -> None:
        del observation

    def trace(self, observation) -> None:
        del observation

    def artifact(self, artifact: ProducedArtifact) -> None:
        assert isinstance(artifact.reference, LocalArtifactRef)
        assert artifact.reference.path.exists()
        self._source_paths.append(artifact.reference.path)
        self._published.append(
            PublishedArtifact(
                logical_name=artifact.name,
                kind=artifact.kind,
                reference=StoredArtifactRef(
                    provider="test",
                    namespace="tests",
                    name=artifact.name,
                    version="v0",
                    digest=artifact.reference.digest,
                ),
                required=artifact.required,
                size_bytes=artifact.reference.path.stat().st_size,
                role=artifact.role,
            )
        )

    def published_artifacts(self) -> tuple[PublishedArtifact, ...]:
        assert all(path.exists() for path in self._source_paths)
        return tuple(self._published)

    def flush_artifacts(self, timeout: float | None = None) -> tuple[PublishedArtifact, ...]:
        del timeout
        if "published_artifacts" in self.__dict__ and self.__dict__["published_artifacts"] is None:
            return ()
        return tuple(self._published)

    def finish(self, outcome: RunOutcome) -> None:
        self.outcomes.append(outcome)


class PublishingBackend:
    def __init__(self) -> None:
        self.tracked: PublishingRun | None = None

    def start_run(self, spec: RunSpec) -> PublishingRun:
        self.tracked = PublishingRun(spec.run_id)
        return self.tracked


def _spec() -> RunSpec:
    return RunSpec(
        project_id="tests",
        work_package_id="train/finalize",
        stage="train",
        job_kind="train.sft",
        job_definition_version="train/sft@1",
    )


def test_finalized_execution_returns_exact_identity_before_cleanup(
    tmp_path: Path,
) -> None:
    backend = PublishingBackend()
    source_path: Path | None = None

    def operation(context):
        nonlocal source_path
        path = context.workspace / "model.bin"
        source_path = path
        path.write_bytes(b"weights")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        context.artifact(
            ProducedArtifact(
                "model/final",
                "model",
                LocalArtifactRef(path.resolve(), digest),
            )
        )
        return "trained"

    result = execute_run_tracked_finalized(
        _spec(),
        operation,
        backend=backend,
        scratch_root=tmp_path,
    )

    assert result.value == "trained"
    assert len(result.published_artifacts) == 1
    published = result.published_artifacts[0]
    assert published.logical_name == "model/final"
    assert published.reference.version == "v0"
    assert published.reference.digest == hashlib.sha256(b"weights").hexdigest()
    assert source_path is not None and not source_path.exists()
    assert backend.tracked is not None
    assert backend.tracked.outcomes[-1].status == "succeeded"


@pytest.mark.parametrize(
    "executor",
    [execute_run_tracked, execute_run_tracked_finalized],
    ids=["tracked", "tracked-finalized"],
)
def test_failure_artifacts_are_flushed_before_workspace_cleanup(
    tmp_path: Path,
    executor,
) -> None:
    class DeferredPublishingRun(PublishingRun):
        def __init__(self, run_id: str) -> None:
            super().__init__(run_id)
            self.pending: list[ProducedArtifact] = []
            self.flush_source_states: list[bool] = []

        def artifact(self, artifact: ProducedArtifact) -> None:
            assert isinstance(artifact.reference, LocalArtifactRef)
            self.pending.append(artifact)

        def flush_artifacts(self, timeout: float | None = None) -> tuple[PublishedArtifact, ...]:
            del timeout
            committed: list[PublishedArtifact] = []
            while self.pending:
                artifact = self.pending.pop(0)
                assert isinstance(artifact.reference, LocalArtifactRef)
                path = artifact.reference.path
                self.flush_source_states.append(path.exists())
                if not path.exists():
                    raise FileNotFoundError(path)
                published = PublishedArtifact(
                    logical_name=artifact.name,
                    kind=artifact.kind,
                    reference=StoredArtifactRef(
                        provider="test",
                        namespace="tests",
                        name=artifact.name,
                        version="v0",
                        digest=artifact.reference.digest,
                    ),
                    required=artifact.required,
                    size_bytes=path.stat().st_size,
                    role=artifact.role,
                )
                self._published.append(published)
                committed.append(published)
            return tuple(committed)

        def finish(self, outcome: RunOutcome) -> None:
            # Trackio drains pending asynchronous uploads during finish. This
            # second drain must be empty because execution flushed while the
            # ephemeral source path still existed.
            self.flush_artifacts(timeout=30)
            super().finish(outcome)

    class DeferredPublishingBackend:
        def __init__(self) -> None:
            self.tracked: DeferredPublishingRun | None = None

        def start_run(self, spec: RunSpec) -> DeferredPublishingRun:
            self.tracked = DeferredPublishingRun(spec.run_id)
            return self.tracked

    backend = DeferredPublishingBackend()
    source_path: Path | None = None

    def fail_after_artifact(context) -> None:
        nonlocal source_path
        path = context.workspace / "verifiers-traces.jsonl"
        source_path = path
        path.write_text('{"id":"trace-1"}\n', encoding="utf-8")
        context.artifact(
            ProducedArtifact(
                "training/rollouts/verifiers-traces",
                "evaluation-traces",
                LocalArtifactRef(path.resolve(), hashlib.sha256(path.read_bytes()).hexdigest()),
            )
        )
        raise RuntimeError("all rollouts were truncated")

    with pytest.raises(RuntimeError, match="all rollouts were truncated"):
        executor(
            _spec(),
            fail_after_artifact,
            backend=backend,
            scratch_root=tmp_path,
        )

    assert backend.tracked is not None
    assert backend.tracked.flush_source_states == [True]
    assert backend.tracked.pending == []
    assert backend.tracked.outcomes[-1].status == "failed"
    assert source_path is not None and not source_path.exists()


def test_finalization_failure_marks_run_failed(tmp_path: Path) -> None:
    class BackendWithoutResolver:
        def __init__(self) -> None:
            self.tracked: PublishingRun | None = None

        def start_run(self, spec: RunSpec) -> PublishingRun:
            self.tracked = PublishingRun(spec.run_id)
            self.tracked.__dict__["published_artifacts"] = None
            return self.tracked

    backend = BackendWithoutResolver()

    with pytest.raises(
        ContractError,
        match="cannot resolve committed output artifacts",
    ):
        execute_run_tracked_finalized(
            _spec(),
            lambda context: "done",
            backend=backend,
            scratch_root=tmp_path,
        )

    assert backend.tracked is not None
    assert backend.tracked.outcomes[-1].status == "failed"


def test_cooperative_cancellation_finishes_tracking_as_cancelled(
    tmp_path: Path,
) -> None:
    backend = PublishingBackend()

    with pytest.raises(OperationCancelled, match="stop requested"):
        execute_run_tracked_finalized(
            _spec(),
            lambda context: (_ for _ in ()).throw(OperationCancelled("stop requested")),
            backend=backend,
            scratch_root=tmp_path,
        )

    assert backend.tracked is not None
    assert [outcome.status for outcome in backend.tracked.outcomes] == ["cancelled"]


def test_required_role_must_be_published_once(tmp_path: Path) -> None:
    backend = PublishingBackend()
    spec = RunSpec(
        project_id="tests",
        work_package_id="train/finalize",
        stage="train",
        job_kind="train.sft",
        job_definition_version="train/sft@1",
        required_artifact_roles=("model",),
    )

    with pytest.raises(
        ContractError,
        match="required artifact role 'model' resolved 0 outputs",
    ):
        execute_run_tracked_finalized(
            spec,
            lambda context: "done",
            backend=backend,
            scratch_root=tmp_path,
        )

    assert backend.tracked is not None
    assert backend.tracked.outcomes[-1].status == "failed"


def test_duplicate_required_role_fails_unambiguous_finalization(
    tmp_path: Path,
) -> None:
    backend = PublishingBackend()
    spec = RunSpec(
        project_id="tests",
        work_package_id="train/finalize",
        stage="train",
        job_kind="train.sft",
        job_definition_version="train/sft@1",
        required_artifact_roles=("model",),
    )

    def operation(context) -> str:
        for index in range(2):
            path = context.workspace / f"model-{index}.bin"
            path.write_bytes(f"weights-{index}".encode())
            context.artifact(
                ProducedArtifact(
                    name=f"model/{index}",
                    kind="model-adapter",
                    reference=LocalArtifactRef(
                        path.resolve(),
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                    ),
                    role="model",
                )
            )
        return "done"

    with pytest.raises(
        ContractError,
        match="required artifact role 'model' resolved 2 outputs",
    ):
        execute_run_tracked_finalized(
            spec,
            operation,
            backend=backend,
            scratch_root=tmp_path,
        )
