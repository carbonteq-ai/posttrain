"""Tests for provider-neutral tracking lifecycle contracts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from posttrain.common import ContractError, JsonValue
from posttrain.tracking import (
    ArtifactInput,
    EventRecord,
    RunError,
    RunOutcome,
    RunSpec,
    StoredArtifactRef,
    phase_at,
    runtime_phase_intervals,
)

NOW = datetime(2026, 7, 22, tzinfo=UTC)


def test_run_spec_is_immutable_and_provider_neutral() -> None:
    reference = StoredArtifactRef(
        provider="trackio",
        namespace="posttrain-tests",
        name="training-qwen-adapter",
        version="v1",
        provider_metadata={"storage_id": "artifact-1"},
    )
    resolved: dict[str, JsonValue] = {"model": {"selection_id": "models/qwen@bf16"}}
    spec = RunSpec(
        project_id="tests",
        work_package_id="train/qwen",
        stage="train",
        run_id="00000000-0000-4000-8000-000000000010",
        job_kind="train.sft",
        job_definition_version="train/sft@1",
        resolved_inputs=resolved,
        artifacts={"model_adapter": ArtifactInput(reference, "model-adapter")},
    )

    resolved["model"] = {"selection_id": "changed"}

    assert spec.resolved_inputs["model"] == {"selection_id": "models/qwen@bf16"}
    assert spec.artifacts["model_adapter"].reference.provider == "trackio"
    with pytest.raises(TypeError):
        spec.artifacts["other"] = spec.artifacts["model_adapter"]  # type: ignore[index]


def test_stored_artifact_requires_an_immutable_provider_identity() -> None:
    with pytest.raises(ContractError, match="provider"):
        StoredArtifactRef("Weights & Biases", "entity/project", "adapter", "v1")
    with pytest.raises(ContractError, match="immutable version"):
        StoredArtifactRef("wandb", "entity/project", "adapter", "")


def test_run_outcome_validates_failure_and_duration() -> None:
    outcome = RunOutcome(
        "failed",
        NOW,
        NOW + timedelta(seconds=3),
        RunError("RuntimeError", "training failed safely"),
    )

    assert outcome.duration_seconds == 3
    with pytest.raises(ContractError, match="require safe error"):
        RunOutcome("failed", NOW, NOW)
    with pytest.raises(ContractError, match="finish before"):
        RunOutcome("succeeded", NOW, NOW - timedelta(seconds=1))


def test_runtime_phases_pair_nested_events_and_select_the_most_specific_phase() -> None:
    events = (
        EventRecord(
            name="runtime_phase_started",
            occurred_at=NOW,
            attributes={"phase": "operation", "phase_id": "outer"},
        ),
        EventRecord(
            name="runtime_phase_started",
            occurred_at=NOW + timedelta(seconds=2),
            attributes={"phase": "rollout", "phase_id": "inner"},
        ),
        EventRecord(
            name="runtime_phase_completed",
            occurred_at=NOW + timedelta(seconds=6),
            attributes={"phase": "rollout", "phase_id": "inner"},
        ),
        EventRecord(
            name="runtime_phase_completed",
            occurred_at=NOW + timedelta(seconds=10),
            attributes={"phase": "operation", "phase_id": "outer"},
        ),
    )

    projection = runtime_phase_intervals(events, window_finished_at=NOW + timedelta(seconds=10))

    assert projection.issues == ()
    assert tuple(interval.phase for interval in projection.intervals) == ("operation", "rollout")
    assert phase_at(projection.intervals, NOW + timedelta(seconds=4)).phase == "rollout"  # type: ignore[union-attr]
    assert phase_at(projection.intervals, NOW + timedelta(seconds=8)).phase == "operation"  # type: ignore[union-attr]


def test_runtime_phases_keep_incomplete_and_malformed_evidence_explicit() -> None:
    events = (
        EventRecord(
            name="runtime_phase_started",
            occurred_at=NOW,
            attributes={"phase": "actor_update", "phase_id": "open"},
        ),
        EventRecord(
            name="runtime_phase_completed",
            occurred_at=NOW + timedelta(seconds=1),
            attributes={"phase": "rollout", "phase_id": "missing"},
        ),
    )

    projection = runtime_phase_intervals(events, window_finished_at=NOW + timedelta(seconds=3))

    assert projection.intervals[0].status == "incomplete"
    assert projection.intervals[0].finished_at == NOW + timedelta(seconds=3)
    assert len(projection.issues) == 2
