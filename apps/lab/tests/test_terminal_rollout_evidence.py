"""Composition proof for terminal Verifiers evidence in local Trackio storage.

The train package owns the bridge and the Trackio package owns provider I/O.
This test deliberately lives in the reference composition host so it can prove
their real boundary without making either reusable package depend on the other.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import trackio.context_vars as context_vars
from posttrain.tracking import RunError, RunOutcome, RunSpec
from posttrain.train import PolicySampling, PolicyTurnRequest, PolicyTurnResult, RolloutBatch
from posttrain.train.integrations import VerifiersEnvironmentRolloutBridge
from posttrain.train.integrations.verifiers import VerifiersRolloutFailure
from posttrain_tracking_trackio import TrackioBackend, TrackioDataSource, TrackioSettings

pytest.importorskip("verifiers")

from verifiers.v1 import TaskData


@pytest.fixture
def trackio_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Isolate the real Trackio SQLite backend for this composition test."""

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


class _UnusedGenerator:
    async def generate(self, request: PolicyTurnRequest) -> PolicyTurnResult:
        del request
        raise AssertionError("the controlled environment never calls the policy")


class _FailedTerminalTrace:
    """A native terminal error with no trainable branch or scalar reward."""

    id = "trace-terminal-failure"
    branches: tuple[()] = ()
    agent = None
    error = SimpleNamespace(type="HarnessError", message="generator unavailable")

    def __init__(self) -> None:
        self._run: dict[str, object] = {}
        self._info: dict[str, object] = {}

    def stamp(self, *, run: Any, environment_id: str, task_index: int, example_id: str) -> None:
        self._run = {"type": "train", "id": str(run.id), "step": int(run.step)}
        self._info = {
            "environment_id": environment_id,
            "task_index": task_index,
            "example_id": example_id,
        }

    def to_record(self) -> dict[str, object]:
        return {
            "id": self.id,
            "version": 1,
            "run": self._run,
            "info": self._info,
            "errors": [{"type": "HarnessError", "message": "generator unavailable"}],
            "is_completed": False,
            "stop_condition": None,
            "calls": [],
            "nodes": [],
            "rewards": {},
        }


class _FailedTerminalEpisode:
    async def run(self) -> list[_FailedTerminalTrace]:
        return [_FailedTerminalTrace()]


class _FailedTerminalEnvironment:
    @asynccontextmanager
    async def serving(self) -> AsyncIterator[None]:
        yield

    def episode(self, _task: object, _context: object, n: int = 1) -> _FailedTerminalEpisode:
        assert n == 1
        return _FailedTerminalEpisode()


def _spec(run_id: str) -> RunSpec:
    return RunSpec(
        project_id="terminal-evidence",
        work_package_id="train/terminal-evidence",
        stage="train",
        run_id=run_id,
        job_kind="train.grpo",
        job_definition_version="train/grpo@1",
        resolved_inputs={"environment": {"selection_id": "environments/test@1"}},
        source_metadata={"revision": "a" * 40},
    )


def test_failed_terminal_trace_is_queryable_before_trackio_run_finishes(trackio_dir: Path) -> None:
    """A failed native trace is durable before the bridge raises its typed error."""

    project = "terminal-evidence-sqlite"
    run_id = "00000000-0000-4000-8000-000000000901"
    tracked = TrackioBackend(TrackioSettings(project=project, auto_log_gpu=False, auto_log_cpu=False)).start_run(
        _spec(run_id)
    )
    source = TrackioDataSource(project)
    bridge = VerifiersEnvironmentRolloutBridge(
        dataset_id="terminal-evidence-v1",
        revision="revision",
        tasks={7: SimpleNamespace(data=TaskData(idx=7, prompt="controlled failure"))},
        environment_factory=_FailedTerminalEnvironment,
        trace_path=trackio_dir / "native-traces.jsonl",
        environment_id="terminal-evidence-v1",
        run_id=run_id,
        sampling=PolicySampling(max_tokens=32),
    )
    observed_before_finish = False
    callback_errors: list[BaseException] = []

    async def publish(trace: Any) -> None:
        nonlocal observed_before_finish
        try:
            tracked.trace(trace)
            # The local Trackio writer is intentionally asynchronous (its
            # bounded sender wakes every 0.5 s).  Poll the independent reader
            # rather than flushing or finishing the run, which is the actual
            # producer/consumer behavior we need to qualify.
            deadline = asyncio.get_running_loop().time() + 2.0
            record = None
            while record is None and asyncio.get_running_loop().time() < deadline:
                try:
                    record = await source.get_trace(run_id, trace.external_id)
                except (LookupError, ValueError):
                    # The first emitted record also creates the local project
                    # database, so the reader may race that initialization.
                    await asyncio.sleep(0.05)
                else:
                    if record is None:
                        await asyncio.sleep(0.05)
            assert record is not None
            assert record.external_id == "trace-terminal-failure"
            assert record.attributes["has_error"] is True
            assert record.attributes["is_truncated"] is False
            assert record.payload["errors"] == [{"type": "HarnessError", "message": "generator unavailable"}]
        except BaseException as error:
            callback_errors.append(error)
        else:
            observed_before_finish = True

    with pytest.raises(VerifiersRolloutFailure, match="harness or environment error"):
        asyncio.run(
            bridge.run_observed(
                RolloutBatch(example_ids=("train/000007",), step=0, model_id="model-profile-v1"),
                _UnusedGenerator(),
                on_completed=publish,
            )
        )

    assert callback_errors == []
    assert observed_before_finish is True
    [evidence] = bridge.evidence().metrics
    assert evidence.values["train/rl/rollouts_requested"] == 1
    assert evidence.values["train/rl/rollouts_attempted"] == 1
    assert evidence.values["train/rl/rollouts_failed"] == 1
    assert evidence.values["train/rl/rollouts_unscorable"] == 1
    assert "train/rl/reward_std" not in evidence.values
    assert "train/rl/group_zero_variance_fraction" not in evidence.values

    finished = datetime.now(UTC)
    tracked.finish(
        RunOutcome(
            "failed",
            finished,
            finished,
            RunError("VerifiersRolloutFailure", "controlled terminal rollout failure"),
        )
    )
    record_after_finish = asyncio.run(source.get_trace(run_id, "trace-terminal-failure"))
    assert record_after_finish is not None
