"""Tests for the provider-neutral dataset preparation operation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from posttrain.common import (
    EventObservation,
    LocalArtifactRef,
    MetricBatchObservation,
    MetricObservation,
    ProducedArtifact,
    RunContext,
    TraceFactUpdateObservation,
    TraceObservation,
)
from posttrain.data import (
    DatasetPrepareRequest,
    PreferenceDataset,
    PreferenceExample,
    SupervisedDataset,
    SupervisedExample,
    prepare,
)


@dataclass
class RecordingObserver:
    events: list[EventObservation] = field(default_factory=list)
    metric_batches: list[MetricBatchObservation] = field(default_factory=list)
    artifacts: list[ProducedArtifact] = field(default_factory=list)

    def event(self, observation: EventObservation) -> None:
        self.events.append(observation)

    def metric(self, observation: MetricObservation) -> None:
        del observation

    def metrics(self, observation: MetricBatchObservation) -> None:
        self.metric_batches.append(observation)

    def trace(self, observation: TraceObservation) -> None:
        del observation

    def trace_fact_update(self, observation: TraceFactUpdateObservation) -> None:
        del observation

    def artifact(self, artifact: ProducedArtifact) -> None:
        self.artifacts.append(artifact)


def _context(tmp_path: Path, observer: RecordingObserver, run_id: str) -> RunContext:
    return RunContext(
        project_id="data-api-test",
        work_package_id="train/prepare",
        run_id=run_id,
        job_kind="data.prepare",
        job_definition_version="data/canonicalize@1",
        workspace=(tmp_path / run_id).resolve(),
        observer=observer,
    )


def _supervised() -> SupervisedDataset:
    return SupervisedDataset(
        id="datasets/supervised",
        revision="source-revision",
        examples=(
            SupervisedExample(
                id="example-1",
                messages=(
                    {"role": "user", "content": "What is 1 + 1?"},
                    {"role": "assistant", "content": "2"},
                ),
                trainable_message_indices=(1,),
                metadata={"source_row": 0},
            ),
        ),
        metadata={
            "source_kind": "fixture",
            "content_sha256": "a" * 64,
            "materialized_path": "/host-specific/path",
        },
    )


def _preference() -> PreferenceDataset:
    return PreferenceDataset(
        id="datasets/preference",
        revision="source-revision",
        examples=(
            PreferenceExample(
                id="pair-1",
                prompt=({"role": "user", "content": "Pick one."},),
                chosen=({"role": "assistant", "content": "A"},),
                rejected=({"role": "assistant", "content": "B"},),
                chosen_score=1.0,
                rejected_score=0.0,
            ),
        ),
        metadata={"source_kind": "fixture"},
    )


def test_prepare_emits_deterministic_supervised_dataset_artifact(
    tmp_path: Path,
) -> None:
    first_observer = RecordingObserver()
    second_observer = RecordingObserver()

    first = prepare(
        _context(tmp_path, first_observer, "run-1"),
        DatasetPrepareRequest(_supervised()),
    )
    second = prepare(
        _context(tmp_path, second_observer, "run-2"),
        DatasetPrepareRequest(_supervised()),
    )

    assert first.content_sha256 == second.content_sha256
    first_reference = first.native_artifact.reference
    second_reference = second.native_artifact.reference
    assert isinstance(first_reference, LocalArtifactRef)
    assert isinstance(second_reference, LocalArtifactRef)
    assert first_reference.digest == second_reference.digest
    assert first.num_examples == 1
    assert first.native_artifact.kind == "dataset"
    assert first.native_artifact.role == "dataset"
    assert first.native_artifact.metadata["source_content_sha256"] == "a" * 64
    assert "materialized_path" not in first.native_artifact.metadata
    reference = first_reference
    manifest = json.loads((reference.path / "manifest.json").read_text())
    assert manifest == {
        "schema_version": 1,
        "dataset_id": "datasets/supervised",
        "dataset_revision": "source-revision",
        "dataset_kind": "supervised",
        "dataset_schema_version": 1,
        "content_sha256": first.content_sha256,
        "examples": 1,
        "size_bytes": first.size_bytes,
        "data": "data.jsonl",
    }
    rows = [json.loads(line) for line in (reference.path / "data.jsonl").read_text().splitlines()]
    assert rows[0]["id"] == "example-1"
    assert rows[0]["trainable_message_indices"] == [1]
    assert [event.name for event in first_observer.events] == [
        "data_prepare_started",
        "data_prepare_completed",
    ]
    assert first_observer.metric_batches[0].values == {
        "data/examples": 1.0,
        "data/bytes": float(first.size_bytes),
    }
    assert first_observer.artifacts[0].name == first.native_artifact.name
    assert first_observer.artifacts[0].reference == first.native_artifact.reference
    assert first_observer.artifacts[0].metadata["run_id"] == "run-1"


def test_prepare_canonicalizes_preference_dataset(tmp_path: Path) -> None:
    observer = RecordingObserver()

    result = prepare(
        _context(tmp_path, observer, "preference-run"),
        DatasetPrepareRequest(_preference()),
    )

    assert result.descriptor.kind == "preference"
    reference = result.native_artifact.reference
    assert isinstance(reference, LocalArtifactRef)
    row = json.loads((reference.path / "data.jsonl").read_text())
    assert row["id"] == "pair-1"
    assert row["chosen"][0]["content"] == "A"
    assert row["rejected"][0]["content"] == "B"
    assert result.native_artifact.metadata["dataset_kind"] == "preference"
