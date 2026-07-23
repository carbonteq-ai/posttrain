"""Shared logical fixtures and assertions for concrete tracking backends."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from posttrain.common import (
    EventObservation,
    LocalArtifactRef,
    MetricBatchObservation,
    MetricObservation,
    ProducedArtifact,
    TraceObservation,
)
from posttrain.tracking import (
    ArtifactInput,
    RunDataSource,
    RunError,
    RunOutcome,
    RunOutcomeStatus,
    RunSpec,
    StoredArtifactRef,
    TraceQuery,
    TrackedRun,
)

STARTED = datetime(2026, 7, 22, 3, 0, tzinfo=UTC)


def conformance_spec(
    run_id: str,
    *,
    artifacts: Mapping[str, ArtifactInput] | None = None,
) -> RunSpec:
    return RunSpec(
        project_id="conformance",
        work_package_id="train/shared-fixture",
        stage="train",
        run_id=run_id,
        job_kind="train.sft",
        job_definition_version="train/sft@1",
        resolved_inputs={
            "model": {
                "selection_id": "models/qwen@bf16",
                "runtime": {"dtype": "bfloat16", "gradient_checkpointing": True},
            }
        },
        source_metadata={"revision": "a" * 40, "dirty": False},
        artifacts=artifacts or {},
    )


def emit_conformance_run(
    tracked: TrackedRun,
    output_path: Path,
    status: RunOutcomeStatus = "succeeded",
) -> RunOutcome:
    tracked.event(EventObservation("operation_started", STARTED, {"phase": "train"}))
    tracked.metric(MetricObservation("train/loss", 2.0, 0, {"split": "train"}))
    tracked.metrics(
        MetricBatchObservation(
            {"train/loss": 1.0, "train/tokens_per_s": 42.0},
            1,
            {"split": "train"},
        )
    )
    tracked.trace(
        TraceObservation(
            "conversation",
            "trace-1",
            {"messages": [{"role": "assistant", "content": "done"}]},
            {"split": "test"},
        )
    )
    tracked.trace(
        TraceObservation(
            "verifiers",
            "rollout-1",
            {
                "id": "rollout-1",
                "version": 2,
                "nodes": [{"message": {"role": "assistant", "content": "4"}}],
                "rewards": {"correct": 1.0},
                "is_completed": True,
            },
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    error = RunError("RuntimeError", "safe failure") if status == "failed" else None
    outcome = RunOutcome(status, STARTED, STARTED + timedelta(seconds=5), error)
    tracked.finish(outcome)
    tracked.finish(outcome)
    return outcome


def terminal_outcome(status: RunOutcomeStatus) -> RunOutcome:
    error = RunError("RuntimeError", "safe failure") if status == "failed" else None
    return RunOutcome(status, STARTED, STARTED + timedelta(seconds=1), error)


def artifact_input(reference: Any) -> ArtifactInput:
    return ArtifactInput(
        StoredArtifactRef(
            reference.provider,
            reference.namespace,
            reference.name,
            reference.version,
            reference.digest,
        ),
        "model-adapter",
    )


async def logical_snapshot(source: RunDataSource, run_id: str) -> dict[str, Any]:
    detail = await source.get_run(run_id)
    series = await source.metric_series(run_id, ("train/loss", "train/tokens_per_s"))
    traces = await source.traces(run_id, TraceQuery(limit=100))
    artifacts = await source.artifacts(run_id)
    return {
        "identity": {
            "run_id": detail.summary.run_id,
            "project_id": detail.summary.project_id,
            "work_package_id": detail.summary.work_package_id,
            "stage": detail.summary.stage,
            "job_kind": detail.summary.job_kind,
            "job_definition_version": detail.summary.job_definition_version,
        },
        "status": detail.summary.status,
        "error": (
            None
            if detail.summary.error is None
            else {
                "type": detail.summary.error.type,
                "message": detail.summary.error.message,
            }
        ),
        "resolved_inputs": detail.resolved_inputs,
        "source_metadata": detail.source_metadata,
        "events": [{"name": event.name, "attributes": event.attributes} for event in detail.events],
        "metrics": {item.name: [(point.step, point.value) for point in item.points] for item in series},
        "traces": sorted(
            (
                trace.trace_type,
                trace.external_id,
                trace.payload,
                trace.attributes,
            )
            for trace in traces.items
        ),
        "artifacts": sorted((item.direction, item.logical_name, item.kind) for item in artifacts.items),
    }


def assert_conformance_snapshot(
    snapshot: Mapping[str, Any],
    *,
    run_id: str,
    status: RunOutcomeStatus,
    expect_input: bool = False,
) -> None:
    assert snapshot["identity"] == {
        "run_id": run_id,
        "project_id": "conformance",
        "work_package_id": "train/shared-fixture",
        "stage": "train",
        "job_kind": "train.sft",
        "job_definition_version": "train/sft@1",
    }
    assert snapshot["status"] == status
    assert snapshot["error"] == ({"type": "RuntimeError", "message": "safe failure"} if status == "failed" else None)
    assert snapshot["resolved_inputs"]["model"]["runtime"] == {
        "dtype": "bfloat16",
        "gradient_checkpointing": True,
    }
    assert snapshot["source_metadata"] == {"revision": "a" * 40, "dirty": False}
    assert snapshot["events"] == [{"name": "operation_started", "attributes": {"phase": "train"}}]
    assert snapshot["metrics"] == {
        "train/loss": [(0, 2.0), (1, 1.0)],
        "train/tokens_per_s": [(1, 42.0)],
    }
    trace_identities = {(item[0], item[1]) for item in snapshot["traces"]}
    assert trace_identities == {
        ("conversation", "trace-1"),
        ("verifiers", "rollout-1"),
    }, snapshot["traces"]
    expected_artifacts = [("output", "training/qwen-adapter", "model-adapter")]
    if expect_input:
        expected_artifacts.insert(0, ("input", "training/qwen-adapter", "model-adapter"))
    assert snapshot["artifacts"] == expected_artifacts


__all__ = [
    "STARTED",
    "artifact_input",
    "assert_conformance_snapshot",
    "conformance_spec",
    "emit_conformance_run",
    "logical_snapshot",
    "terminal_outcome",
]
