"""Focused tests for rendered DPO population evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from posttrain.common import (
    EventObservation,
    MetricBatchObservation,
    MetricObservation,
    ProducedArtifact,
    RunContext,
    TraceFactUpdateObservation,
    TraceObservation,
)
from posttrain.data import PreferenceDataset, PreferenceExample
from posttrain.train.backends.trl.dpo import _emit_preference_profile
from posttrain.train.rendering import RenderedPreferenceExample


@dataclass
class CaptureObserver:
    metrics_seen: list[MetricBatchObservation] = field(default_factory=list)

    def event(self, observation: EventObservation) -> None:
        del observation

    def metric(self, observation: MetricObservation) -> None:
        self.metrics_seen.append(MetricBatchObservation({observation.name: observation.value}, observation.step))

    def metrics(self, observation: MetricBatchObservation) -> None:
        self.metrics_seen.append(observation)

    def trace(self, observation: TraceObservation) -> None:
        del observation

    def trace_fact_update(self, observation: TraceFactUpdateObservation) -> None:
        del observation

    def artifact(self, artifact: ProducedArtifact) -> None:
        del artifact


def _context(tmp_path: Path, observer: CaptureObserver) -> RunContext:
    return RunContext(
        project_id="projects/test",
        work_package_id="work-packages/dpo-observability",
        run_id="runs/dpo-observability",
        job_kind="train.dpo",
        job_definition_version="1",
        workspace=tmp_path,
        observer=observer,
    )


def _pair(example_id: str, *, scored: bool) -> PreferenceExample:
    return PreferenceExample(
        example_id,
        ({"role": "user", "content": "Prompt"},),
        ({"role": "assistant", "content": "Chosen"},),
        ({"role": "assistant", "content": "Rejected"},),
        chosen_score=3.0 if scored else None,
        rejected_score=1.0 if scored else None,
    )


def test_rendered_preference_profile_exposes_length_bias_and_score_coverage(
    tmp_path: Path,
) -> None:
    observer = CaptureObserver()
    dataset = PreferenceDataset(
        "dataset/preferences",
        "revision-1",
        (_pair("pair/one", scored=True), _pair("pair/two", scored=False)),
    )
    rendered = (
        RenderedPreferenceExample("pair/one", (1, 2, 3), (4, 5, 6, 7), (8, 9)),
        RenderedPreferenceExample("pair/two", (1,), (2,), (3, 4)),
    )

    _emit_preference_profile(_context(tmp_path, observer), rendered, dataset, max_length=10)

    values = observer.metrics_seen[-1].values
    assert values["train/data/preference_pairs"] == 2
    assert values["train/data/prompt_tokens_mean"] == 2
    assert values["train/data/chosen_tokens_mean"] == 2.5
    assert values["train/data/rejected_tokens_mean"] == 2
    assert values["train/data/prompt_tokens_p95"] == 3
    assert values["train/data/chosen_tokens_p95"] == 4
    assert values["train/data/rejected_tokens_p95"] == 2
    assert values["train/data/max_length_headroom_min"] == 3
    assert values["train/data/chosen_longer_fraction"] == 0.5
    assert values["train/data/preference_score_coverage"] == 0.5
    assert values["train/data/preference_score_margin_mean"] == 2
    assert values["train/data/max_length_utilization"] == 0.5
