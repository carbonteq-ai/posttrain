"""Shared fixtures: a Trackio project written by Posttrain's own writer.

The project SQL these rely on needs carbonteq-trackio 0.31.5.post14.dev29 or later.
By default the project is local SQLite storage. Set POSTTRAIN_TEST_TRACKIO_SERVER_URL
(and TRACKIO_WRITE_TOKEN) to write a fresh project to a Doris-backed Trackio server
instead, which is how the semantic layer is qualified on Doris.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import trackio
import trackio.context_vars as context_vars
from posttrain.common import (
    EventObservation,
    MetricBatchObservation,
    MetricObservation,
    TraceFactSet,
    TraceObservation,
)
from posttrain.tracking import RunError, RunOutcome, RunSpec
from posttrain_tracking_trackio import TrackioBackend, TrackioDataSource, TrackioSettings

HAS_PROJECT_SQL = callable(getattr(trackio.Api, "project_sql", None))

SERVER_URL = os.environ.get("POSTTRAIN_TEST_TRACKIO_SERVER_URL") or None
PROJECT = f"semantic-sql-{datetime.now(UTC):%Y%m%d%H%M%S}" if SERVER_URL else "semantic-sql"
ENGINE = "doris" if SERVER_URL else "sqlite"
STARTED = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)


def _spec(run_id: str, job_kind: str, learning_rate: float) -> RunSpec:
    return RunSpec(
        project_id=PROJECT,
        work_package_id=f"train/{job_kind}",
        stage="train",
        run_id=run_id,
        job_kind=job_kind,
        job_definition_version=f"{job_kind}@1",
        resolved_inputs={
            "model": {"selection_id": "models/lfm@bf16"},
            "settings": {"selection_id": "s", "resolved": {"learning_rate": learning_rate, "beta": 0.005}},
        },
        source_metadata={},
    )


def _trace(index: int, *, step: int, task: str, reward: float, truncated: bool) -> TraceObservation:
    payload: dict[str, Any] = {
        "id": f"rollout-{index}",
        "version": 2,
        "agent": {"model": "org/model"},
        "task": {"type": "ExampleTask", "data": {"idx": index}},
        "nodes": [
            {"message": {"role": "user", "content": "q"}},
            {"parent": 0, "message": {"role": "assistant", "content": "a"}},
        ],
        "calls": [],
        "rewards": {"correct": reward},
        "metrics": {},
        "errors": [],
        "stop_condition": "max_turns" if truncated else "agent_completed",
        "is_completed": not truncated,
    }
    ending = "turn_limit" if truncated else "completed"
    dimensions: dict[str, str | int | bool] = {
        "task_id": task,
        "rollout_step": step,
        "is_truncated": truncated,
        "has_error": False,
    }
    # rollout-0 and rollout-1 have the ending as a fact (rollout-1 only there), rollout-3 only as a
    # trace attribute (facts projected before the fact existed), and rollout-2 predates the label.
    if index in (0, 1):
        dimensions["episode_ending"] = ending
    facts = TraceFactSet(
        namespace="verifiers.trace",
        calculator_version="test.v1",
        dimensions=dimensions,
        measures={"task_reward": reward, "model_output_tokens": 100.0 * (index + 1)},
    )
    attributes = {"episode_ending": ending} if index in (0, 3) else {}
    return TraceObservation("verifiers", f"rollout-{index}", payload, attributes=attributes, facts=(facts,))


@pytest.fixture(scope="session")
def trackio_engine() -> str:
    """The storage engine the trackio_project fixture's SQL runs on."""

    return ENGINE


@pytest.fixture(scope="session")
def trackio_project(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TrackioDataSource]:
    if not HAS_PROJECT_SQL:
        pytest.skip("the installed carbonteq-trackio has no project SQL (needs 0.31.5.post14.dev29)")
    directory = tmp_path_factory.mktemp("trackio")
    patches = pytest.MonkeyPatch()
    for module in ("trackio", "trackio.sqlite_storage", "trackio.utils"):
        patches.setattr(f"{module}.TRACKIO_DIR", directory)
    patches.setattr("trackio.bucket_storage.TRACKIO_DIR", directory)
    patches.setattr("trackio.utils.ARTIFACTS_DIR", directory / "artifacts")
    context_vars.current_run.set(None)
    context_vars.current_project.set(None)
    context_vars.current_server.set(None)
    backend = TrackioBackend(TrackioSettings(project=PROJECT, server_url=SERVER_URL))

    grpo = backend.start_run(_spec("grpo-a", "train.grpo", 5e-5))
    batches_by_step = {
        1: ((1, 250.0, 48.0), (2, 60.0, 40.0), (1, 50.0, 48.0)),
        2: ((1, 80.0, 48.0),),
        3: ((1, 90.0, 48.0),),
    }
    for step, (seconds, entropy) in enumerate(((400.0, 0.18), (100.0, 0.2), (120.0, 0.25)), start=1):
        # Recorded the old way: rollout time and counts once per rollout batch. Batching
        # restarted within update 1, so two of its batches share an ordinal and a count.
        for ordinal, batch_seconds, attempted in batches_by_step[step]:
            grpo.metrics(
                MetricBatchObservation(
                    {"train/rl/time/rollout_seconds": batch_seconds, "train/rl/rollouts_attempted": attempted},
                    step=step,
                    attributes={"rollout_batch_ordinal": ordinal},
                )
            )
        grpo.metrics(
            MetricBatchObservation({"train/step_time_seconds": seconds, "train/rl/entropy": entropy}, step=step)
        )
        # Resolved policy-update counters, one point per applied update. The
        # applied/attempt counters are cumulative; update 2 retried one overflowed attempt.
        grpo.metrics(
            MetricBatchObservation(
                {
                    "train/rl/applied_optimizer_updates": float(step),
                    "train/rl/optimizer_attempts": float(step if step == 1 else step + 1),
                    "train/rl/selected_policy_actions": 100.0 * step,
                    "train/rl/selected_kl_actions": 50.0 * step,
                    "train/rl/advantage_nonzero_fraction": (1.0, 0.5, 0.0)[step - 1],
                },
                step=step,
                attributes={"measurement_scope": "resolved-applied-update"},
            )
        )
    # Replayed from traces when the run finished, stored at later provider steps.
    for update, value in ((1, 1.0), (2, 0.5)):
        grpo.metric(
            MetricObservation(
                "train/rl/tool_call_frequency",
                value,
                step=10 + update,
                attributes={"observation_source": "verifiers", "source_step": update},
            )
        )
    for index, (step, task, reward, truncated) in enumerate(
        ((1, "t1", 1.0, False), (1, "t2", 0.0, True), (2, "t1", 0.5, False), (2, "t2", 0.25, False))
    ):
        grpo.trace(_trace(index, step=step, task=task, reward=reward, truncated=truncated))
    grpo.finish(RunOutcome("succeeded", STARTED, STARTED + timedelta(seconds=620)))

    sampo = backend.start_run(_spec("sampo-b", "train.sampo", 1e-4))
    sampo.metrics(MetricBatchObservation({"train/step_time_seconds": 200.0}, step=1))
    sampo.event(EventObservation("runtime_phase_failed", STARTED, {"phase": "actor_update", "logical_step": 2}))
    sampo.finish(
        RunOutcome(
            "failed", STARTED, STARTED + timedelta(seconds=300), error=RunError("OutOfMemoryError", "operation failed")
        )
    )
    yield TrackioDataSource(PROJECT, server_url=SERVER_URL)
    patches.undo()
