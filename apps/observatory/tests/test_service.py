"""Tests for Observatory telemetry and application projections."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from posttrain.common import JsonValue
from posttrain.tracking import (
    ArtifactSet,
    MetricPoint,
    MetricSeries,
    RunDetail,
    RunQuery,
    RunSummary,
    TracePage,
    TrackingCapabilities,
)
from posttrain_observatory import (
    DEFAULT_TELEMETRY_DEFINITIONS,
    ChartDefinition,
    EvidenceRequirementDefinition,
    JobTelemetryDefinition,
    ObservatoryService,
    SummaryFieldDefinition,
)
from posttrain_observatory.cli import main
from posttrain_observatory.fixtures import FixtureRunDataSource
from pydantic import ValidationError

NOW = datetime(2026, 7, 22, tzinfo=UTC)


class FakeRunDataSource:
    def __init__(self, details: dict[str, RunDetail], series: dict[str, dict[str, MetricSeries]]) -> None:
        self.details = details
        self.series = series
        self.capabilities = TrackingCapabilities(provider="fixture", live_traces=True)

    async def list_runs(self, query: RunQuery) -> tuple[RunSummary, ...]:
        return tuple(detail.summary for detail in self.details.values())[: query.limit]

    async def get_run(self, run_id: str) -> RunDetail:
        return self.details[run_id]

    async def metric_series(self, run_id: str, names: tuple[str, ...]) -> tuple[MetricSeries, ...]:
        values = self.series[run_id]
        return tuple(values.get(name, MetricSeries(name=name)) for name in names)

    async def traces(self, run_id: str, query: object) -> TracePage:
        del run_id, query
        return TracePage(live=True)

    async def artifacts(self, run_id: str) -> ArtifactSet:
        del run_id
        return ArtifactSet()


def _summary(run_id: str, job_kind: str = "train.sft") -> RunSummary:
    return RunSummary(
        provider="fixture",
        provider_run_id=f"provider-{run_id}",
        run_id=run_id,
        display_name=run_id,
        project_id="tests",
        work_package_id="train/example",
        stage="train",
        job_kind=job_kind,
        job_definition_version=f"{job_kind}@1",
        status="succeeded",
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=10),
    )


def _source(*, include_loss: bool = True) -> FakeRunDataSource:
    run_id = "runs/sft-one"
    details = {
        run_id: RunDetail(
            summary=_summary(run_id),
            metric_names=("train/loss", "train/learning_rate", "vendor/raw"),
            trace_count=2,
        )
    }
    values = {
        "train/learning_rate": MetricSeries(name="train/learning_rate", points=(MetricPoint(value=0.001, step=1),)),
        "vendor/raw": MetricSeries(name="vendor/raw", points=(MetricPoint(value=99, step=1),)),
    }
    if include_loss:
        values["train/loss"] = MetricSeries(
            name="train/loss",
            points=(MetricPoint(value=1.0, step=1), MetricPoint(value=0.5, step=2)),
        )
    return FakeRunDataSource(details, {run_id: values})


def test_telemetry_models_reject_surface_specific_drift() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SummaryFieldDefinition.model_validate(
            {"key": "loss", "label": "Loss", "metric": "train/loss", "frontend_only": True}
        )

    with pytest.raises(ValidationError, match="need a condition"):
        EvidenceRequirementDefinition(
            key="validation",
            label="Validation",
            level="conditional",
            metrics=("train/validation/loss",),
            reason="Validation evidence.",
        )
    with pytest.raises(ValidationError, match="only conditional"):
        EvidenceRequirementDefinition(
            key="loss",
            label="Loss",
            level="required",
            condition="validation_configured",
            metrics=("train/loss",),
            reason="Core objective evidence.",
        )


def test_distillation_telemetry_is_strict_and_trace_aware() -> None:
    definition = DEFAULT_TELEMETRY_DEFINITIONS["train.distill"]

    assert definition.display_name == "On-policy distillation"
    assert definition.metric_names == {
        "train/distill/loss",
        "train/distill/reverse_kl",
        "train/distill/scored_tokens",
        "train/distill/teacher_latency_ms",
        "train/distill/teacher_failures",
    }
    assert tuple(section.trace_type for section in definition.trace_sections) == ("verifiers",)
    assert {field.key for field in definition.summary_fields if field.required} == {
        "final_loss",
        "reverse_kl",
        "scored_tokens",
        "teacher_failures",
    }
    assert {item.metric for item in definition.metric_help} == definition.metric_names
    with pytest.raises(ValidationError, match="comparison keys"):
        JobTelemetryDefinition(
            job_kind="train.test",
            display_name="Test",
            summary_fields=(SummaryFieldDefinition(key="loss", label="Loss", metric="train/loss"),),
            charts=(ChartDefinition(key="loss", title="Loss", metrics=("train/loss",)),),
            metric_help=(DEFAULT_TELEMETRY_DEFINITIONS["train.sft"].metric_help[0],),
            comparison_keys=("unknown",),
        )


def test_dpo_telemetry_answers_pair_policy_stability_and_data_questions() -> None:
    definition = DEFAULT_TELEMETRY_DEFINITIONS["train.dpo"]

    assert tuple(chart.key for chart in definition.charts) == (
        "preferences",
        "policy",
        "objective",
        "stability",
        "efficiency",
    )
    assert all(chart.question for chart in definition.charts)
    assert {
        "train/logps/chosen",
        "train/logps/rejected",
        "train/entropy",
        "train/grad_norm",
        "train/data/chosen_longer_fraction",
        "train/data/preference_score_coverage",
    }.issubset(definition.metric_names)
    assert {item.metric for item in definition.metric_help} == definition.metric_names
    assert {item.level for item in definition.evidence_requirements} == {
        "required",
        "conditional",
        "diagnostic",
    }


def _dpo_source(*, missing: str | None = None, validation_configured: bool = False) -> FakeRunDataSource:
    run_id = "runs/dpo-one"
    definition = DEFAULT_TELEMETRY_DEFINITIONS["train.dpo"]
    required = {
        metric
        for requirement in definition.evidence_requirements
        if requirement.level == "required"
        for metric in requirement.metrics
    }
    values = {
        name: MetricSeries(
            name=name, points=(MetricPoint(value=0.0 if name.endswith("score_coverage") else 1.0, step=1),)
        )
        for name in required
        if name != missing
    }
    resolved_inputs: dict[str, JsonValue] = (
        {"validation_dataset": {"selection_id": "dataset/held-out", "revision": "v1"}} if validation_configured else {}
    )
    details = {
        run_id: RunDetail(
            summary=_summary(run_id, "train.dpo"),
            metric_names=tuple(values),
            resolved_inputs=resolved_inputs,
        )
    }
    return FakeRunDataSource(details, {run_id: values})


@pytest.mark.asyncio
async def test_dpo_completeness_distinguishes_core_conditional_and_research_evidence() -> None:
    complete = await ObservatoryService(_dpo_source()).get_run_view("runs/dpo-one")
    insufficient = await ObservatoryService(_dpo_source(missing="train/logps/chosen")).get_run_view("runs/dpo-one")
    partial = await ObservatoryService(_dpo_source(validation_configured=True)).get_run_view("runs/dpo-one")

    assert complete.completeness.state == "complete"
    assert complete.trace_evaluation_enabled is False
    assert complete.completeness.research_ready is False
    assert insufficient.completeness.state == "insufficient"
    assert "evidence-policy_movement" in {alert.id for alert in insufficient.alerts}
    validation = next(item for item in partial.completeness.requirements if item.key == "held_out_preferences")
    assert validation.state == "missing"
    assert partial.completeness.state == "partial"
    assert (
        next(item for item in complete.completeness.requirements if item.key == "held_out_preferences").state
        == "not_applicable"
    )


@pytest.mark.asyncio
async def test_run_view_is_job_aware_and_preserves_missing_evidence() -> None:
    available = await ObservatoryService(_source()).get_run_view("runs/sft-one")
    missing = await ObservatoryService(_source(include_loss=False)).get_run_view("runs/sft-one")
    available_values = {value.key: value for value in available.summary}
    missing_values = {value.key: value for value in missing.summary}
    assert available_values["final_loss"].value == 0.5
    assert available.trace_count == 2
    assert available.trace_evaluation_enabled is False
    assert missing_values["final_loss"].state == "missing"
    assert "missing-final_loss" in {alert.id for alert in missing.alerts}


@pytest.mark.asyncio
async def test_trace_navigation_follows_job_telemetry_definition() -> None:
    source = _source()
    source.details["runs/grpo-one"] = RunDetail(
        summary=_summary("runs/grpo-one", "train.grpo"),
        trace_count=0,
    )
    source.series["runs/grpo-one"] = {}

    grpo = await ObservatoryService(source).get_run_view("runs/grpo-one")

    assert grpo.trace_evaluation_enabled is True
    assert grpo.grpo is not None
    assert grpo.completeness.research_ready is False
    assert "missing-grpo-traces" in {alert.id for alert in grpo.alerts}


@pytest.mark.asyncio
async def test_grpo_projection_exposes_population_and_selection_aware_completeness() -> None:
    view = await ObservatoryService(FixtureRunDataSource()).get_run_view("runs/grpo-silver-pine")

    assert view.schema_version == 2
    assert view.grpo is not None
    assert view.grpo.rollout_population.attempted.value == 96
    assert view.grpo.rollout_population.failed.value == 0
    assert view.grpo.acceleration.mtp_selected is False
    assert view.grpo.acceleration.quantized_kv_cache_selected is False
    assert view.grpo.acceleration.speculative_acceptance.state == "missing"
    assert view.completeness.state == "complete"
    assert view.completeness.research_ready is True
    assert next(item for item in view.completeness.requirements if item.key == "mtp_runtime").state == "not_applicable"


@pytest.mark.asyncio
async def test_delta_is_projection_shaped_and_cursor_is_deterministic() -> None:
    source = _source()
    service = ObservatoryService(source)
    first = await service.get_run_delta("runs/sft-one")
    source.series["runs/sft-one"]["train/loss"] = MetricSeries(
        name="train/loss",
        points=(
            MetricPoint(value=1.0, step=1),
            MetricPoint(value=0.5, step=2),
            MetricPoint(value=0.25, step=3),
        ),
    )
    second = await service.get_run_delta("runs/sft-one", first.cursor)
    unchanged = await service.get_run_delta("runs/sft-one", second.cursor)
    assert [change.key for change in second.changed_summary] == ["final_loss"]
    assert [tip.metric for tip in second.series_tips] == ["train/loss"]
    assert unchanged.changed_summary == ()


@pytest.mark.asyncio
async def test_compare_runs_rejects_different_job_kinds() -> None:
    source = _source()
    source.details["runs/eval-one"] = RunDetail(summary=_summary("runs/eval-one", "eval.general"))
    source.series["runs/eval-one"] = {}
    comparison = await ObservatoryService(source).compare_runs(("runs/sft-one", "runs/eval-one"))
    assert comparison.state == "incomparable"
    assert comparison.reason is not None and "different job kinds" in comparison.reason


def test_cli_exposes_the_same_telemetry_schema(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["schema", "train.sft"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["job_kind"] == "train.sft"
    assert payload["summary_fields"][0]["key"] == "final_loss"
