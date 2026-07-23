from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import pytest
from posttrain.tracking import RunOutcomeStatus
from posttrain_tracking_wandb import WandbBackend, WandbDataSource, WandbSettings

from packages.tracking.tests.conformance import (
    artifact_input,
    assert_conformance_snapshot,
    conformance_spec,
    emit_conformance_run,
    logical_snapshot,
    terminal_outcome,
)


def _settings() -> WandbSettings:
    missing = [
        name for name in ("WANDB_API_KEY", "WANDB_ENTITY", "POSTTRAIN_WANDB_TEST_PROJECT") if not os.environ.get(name)
    ]
    if missing:
        pytest.skip(f"missing required W&B conformance variables: {', '.join(missing)}")
    return WandbSettings(
        entity=os.environ["WANDB_ENTITY"],
        project=os.environ["POSTTRAIN_WANDB_TEST_PROJECT"],
        base_url=os.environ.get("WANDB_BASE_URL"),
        tags=("posttrain", "posttrain-conformance"),
    )


async def _wait_for_snapshot(
    settings: WandbSettings,
    run_id: str,
    *,
    expected_status: RunOutcomeStatus,
    minimum_artifacts: int,
    timeout: float = 90,
) -> tuple[WandbDataSource, dict]:
    deadline = time.monotonic() + timeout
    while True:
        source = WandbDataSource(settings)
        try:
            snapshot = await logical_snapshot(source, run_id)
            if snapshot["status"] == expected_status and len(snapshot["artifacts"]) >= minimum_artifacts:
                return source, snapshot
        except Exception:
            pass
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"W&B run {run_id} did not expose status {expected_status} and {minimum_artifacts} artifact edges"
            )
        time.sleep(2)


@pytest.mark.network
@pytest.mark.asyncio
async def test_real_wandb_shared_conformance_and_artifact_lineage(tmp_path: Path) -> None:
    settings = _settings()
    backend = WandbBackend(settings)

    producer_id = f"pt-{uuid.uuid4().hex}"
    producer = backend.start_run(conformance_spec(producer_id))
    emit_conformance_run(producer, tmp_path / "producer" / "adapter.bin")
    producer_source, _ = await _wait_for_snapshot(
        settings,
        producer_id,
        expected_status="succeeded",
        minimum_artifacts=1,
    )
    producer_output = (await producer_source.artifacts(producer_id)).outputs[0]

    consumer_id = f"pt-{uuid.uuid4().hex}"
    input_value = artifact_input(producer_output.artifact)
    consumer = backend.start_run(conformance_spec(consumer_id, artifacts={"starting_model": input_value}))
    materialized = consumer.materialize_inputs({"starting_model": input_value}, tmp_path / "consumer" / "inputs")
    assert next(materialized["starting_model"].path.rglob("adapter.bin")).read_bytes() == b"adapter"
    emit_conformance_run(consumer, tmp_path / "consumer" / "adapter.bin")

    _, snapshot = await _wait_for_snapshot(
        settings,
        consumer_id,
        expected_status="succeeded",
        minimum_artifacts=2,
    )
    assert_conformance_snapshot(
        snapshot,
        run_id=consumer_id,
        status="succeeded",
        expect_input=True,
    )


@pytest.mark.network
@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["partial", "failed", "cancelled", "unsupported"])
async def test_real_wandb_terminal_outcomes(status: RunOutcomeStatus) -> None:
    settings = _settings()
    run_id = f"pt-{uuid.uuid4().hex}"
    tracked = WandbBackend(settings).start_run(conformance_spec(run_id))
    outcome = terminal_outcome(status)
    tracked.finish(outcome)
    tracked.finish(outcome)

    deadline = time.monotonic() + 60
    while True:
        source = WandbDataSource(settings)
        try:
            detail = await source.get_run(run_id)
            if detail.summary.status == status:
                break
        except Exception:
            pass
        if time.monotonic() >= deadline:
            raise AssertionError(f"W&B run {run_id} did not reach canonical status {status}")
        time.sleep(2)
    assert detail.summary.error is not None if status == "failed" else detail.summary.error is None
