"""Tests for Observatory telemetry and application projections."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from posttrain.common import JsonValue
from posttrain.tracking import (
    ArtifactIntegrityResult,
    ArtifactSet,
    MetricPoint,
    MetricSeries,
    RunDetail,
    RunQuery,
    RunSummary,
    TraceAggregateResult,
    TraceFactsQuery,
    TracePage,
    TrackingCapabilities,
)
from posttrain_observatory import (
    DEFAULT_TELEMETRY_DEFINITIONS,
    ChartDefinition,
    EvidenceRequirementDefinition,
    JobTelemetryDefinition,
    MetricSeriesQuery,
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

    async def aggregate_trace_facts(self, run_id: str, query: TraceFactsQuery) -> TraceAggregateResult:
        del run_id, query
        return TraceAggregateResult(state="unsupported")

    async def artifacts(self, run_id: str) -> ArtifactSet:
        del run_id
        return ArtifactSet()

    async def verify_artifact(self, reference, *, deep: bool = False) -> ArtifactIntegrityResult:
        del reference
        return ArtifactIntegrityResult("unsupported", deep=deep)


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


@pytest.mark.asyncio
async def test_metric_series_uses_replay_source_steps_and_preserves_distinct_waves() -> None:
    run_id = "runs/replayed"
    details = {
        run_id: RunDetail(
            summary=_summary(run_id, "train.sampo"),
            metric_names=("train/rl/reward_std", "train/step_time_seconds"),
        )
    }
    replay_attributes = {"observation_source": "verifiers", "source_step": 0}
    series = {
        "train/rl/reward_std": MetricSeries(
            name="train/rl/reward_std",
            points=(
                MetricPoint(value=0.2, step=1),
                MetricPoint(value=0.1, step=56, attributes=replay_attributes),
                MetricPoint(
                    value=0.3,
                    step=57,
                    attributes={"observation_source": "verifiers", "source_step": 1},
                ),
            ),
        ),
        "train/step_time_seconds": MetricSeries(
            name="train/step_time_seconds",
            points=(
                MetricPoint(value=10.0, step=1),
                MetricPoint(value=10.01, step=1),
            ),
        ),
    }

    result = await ObservatoryService(FakeRunDataSource(details, {run_id: series})).get_metric_series(
        run_id,
        MetricSeriesQuery(names=("train/rl/reward_std", "train/step_time_seconds")),
    )

    assert [point.step for point in result.series[0].points] == [0, 1]
    assert [point.value for point in result.series[0].points] == [0.1, 0.3]
    assert [point.value for point in result.series[1].points] == [10.0, 10.01]


@pytest.mark.asyncio
async def test_metric_series_keeps_native_steps_outside_partial_replay() -> None:
    run_id = "runs/partial-replay"
    details = {
        run_id: RunDetail(
            summary=_summary(run_id, "train.grpo"),
            metric_names=("train/rl/reward_std",),
        )
    }
    series = {
        "train/rl/reward_std": MetricSeries(
            name="train/rl/reward_std",
            points=(
                MetricPoint(value=0.2, step=1),
                MetricPoint(value=0.3, step=2),
                MetricPoint(
                    value=0.4,
                    step=99,
                    attributes={"observation_source": "verifiers", "source_step": 3},
                ),
            ),
        )
    }

    result = await ObservatoryService(FakeRunDataSource(details, {run_id: series})).get_metric_series(
        run_id,
        MetricSeriesQuery(names=("train/rl/reward_std",)),
    )

    assert [point.step for point in result.series[0].points] == [1, 2, 3]
    assert [point.value for point in result.series[0].points] == [0.2, 0.3, 0.4]


@pytest.mark.asyncio
async def test_metric_series_orders_native_provider_history_by_logical_step() -> None:
    run_id = "runs/out-of-order"
    details = {
        run_id: RunDetail(
            summary=_summary(run_id),
            metric_names=("train/loss",),
        )
    }
    series = {
        "train/loss": MetricSeries(
            name="train/loss",
            points=(
                MetricPoint(value=2.4, step=13),
                MetricPoint(value=1.8, step=2),
                MetricPoint(value=1.3, step=14),
                MetricPoint(value=4.1, step=0),
            ),
        )
    }

    service = ObservatoryService(FakeRunDataSource(details, {run_id: series}))
    result = await service.get_metric_series(
        run_id,
        MetricSeriesQuery(names=("train/loss",)),
    )

    assert [point.step for point in result.series[0].points] == [0, 2, 13, 14]
    assert result.series[0].points[-1].value == 1.3
    view = await service.get_run_view(run_id)
    latest_loss = next(item for item in view.summary if item.metric == "train/loss")
    assert latest_loss.value == 1.3


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


def test_sampo_smoke_and_data_prepare_telemetry_match_framework_emissions() -> None:
    sampo = DEFAULT_TELEMETRY_DEFINITIONS["train.sampo"]
    smoke = DEFAULT_TELEMETRY_DEFINITIONS["serve.smoke"]
    prepare = DEFAULT_TELEMETRY_DEFINITIONS["data.prepare"]

    assert {
        "train/rl/episode_advantage_mean",
        "train/rl/turn_advantage_mean",
        "train/rl/anchor_group_size_mean",
        "train/rl/sparse_reward_projection_fraction",
    }.issubset(sampo.metric_names)
    assert tuple(section.trace_type for section in sampo.trace_sections) == ("verifiers",)
    assert smoke.metric_names == {
        "serve/probe_latency_seconds",
        "serve/probe_healthy",
        "serve/probe_model_available",
    }
    assert {role.kind for role in smoke.artifact_roles} == {"serving-log"}
    assert prepare.metric_names == {"data/examples", "data/bytes"}
    assert {role.kind for role in prepare.artifact_roles} == {"dataset"}
    for definition in (sampo, smoke, prepare):
        assert {item.metric for item in definition.metric_help} == definition.metric_names


def _registered_job_source(
    job_kind: str,
    values: dict[str, float],
    *,
    trace_count: int = 0,
) -> FakeRunDataSource:
    run_id = f"runs/{job_kind.replace('.', '-')}"
    series = {
        name: MetricSeries(name=name, points=(MetricPoint(value=value, step=1),)) for name, value in values.items()
    }
    return FakeRunDataSource(
        {
            run_id: RunDetail(
                summary=_summary(run_id, job_kind),
                metric_names=tuple(series),
                trace_count=trace_count,
            )
        },
        {run_id: series},
    )


@pytest.mark.parametrize(
    ("job_kind", "trace_count"),
    (("train.sampo", 2), ("train.distill", 2), ("serve.smoke", 0), ("data.prepare", 0)),
)
@pytest.mark.asyncio
async def test_registered_job_kinds_resolve_to_first_class_metric_views(
    job_kind: str,
    trace_count: int,
) -> None:
    definition = DEFAULT_TELEMETRY_DEFINITIONS[job_kind]
    values = {metric: 1.0 for metric in definition.metric_names}
    values["train/rl/rollouts_failed"] = 0.0
    values["train/rl/rollouts_unscorable"] = 0.0
    values["train/distill/teacher_failures"] = 0.0
    source = _registered_job_source(job_kind, values, trace_count=trace_count)
    run_id = f"runs/{job_kind.replace('.', '-')}"

    response = await ObservatoryService(source).get_run_view_response(run_id, mode="job")

    assert response.resolved_mode == "job"
    assert response.fallback_reason is None
    assert response.view.view_kind == "job.metrics"
    assert response.view.completeness.state == "complete"
    assert {item.metric for item in response.view.summary} == {field.metric for field in definition.summary_fields}
    assert response.view.trace_evaluation_enabled is bool(definition.trace_sections)


@pytest.mark.asyncio
async def test_distillation_projection_requires_traces_and_surfaces_teacher_failures() -> None:
    definition = DEFAULT_TELEMETRY_DEFINITIONS["train.distill"]
    values = {metric: 1.0 for metric in definition.metric_names}
    source = _registered_job_source("train.distill", values)

    view = await ObservatoryService(source).get_run_view("runs/train-distill")

    assert view.completeness.state == "complete"
    assert view.completeness.research_ready is False
    assert {"distill-teacher-failures", "missing-distill-traces"}.issubset({alert.id for alert in view.alerts})


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


def test_grpo_policy_optimization_unifies_learning_signal_and_update_control() -> None:
    definition = DEFAULT_TELEMETRY_DEFINITIONS["train.grpo"]
    optimization = next(chart for chart in definition.charts if chart.key == "optimization")

    assert optimization.question == (
        "Does reward improve while groups retain relative signal and policy updates remain controlled?"
    )
    assert optimization.metrics == (
        "train/rl/reward_mean",
        "train/rl/reward_std",
        "train/rl/group_zero_variance_fraction",
        "train/rl/policy_loss",
        "train/rl/entropy",
        "train/rl/kl",
        "train/rl/clip_fraction",
        "train/rl/clip_fraction_low",
        "train/rl/clip_fraction_high",
    )
    assert [chart.key for chart in definition.charts][-3:] == [
        "dynamic_sampling",
        "active_sampling_yield",
        "active_sampling_population",
    ]


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
    assert view.grpo.rollout_population.requested.state == "missing"
    assert view.grpo.rollout_population.attempted.value == 96
    assert view.grpo.rollout_population.failed.value == 0
    assert view.grpo.rollout_population.missing.state == "missing"
    assert view.grpo.acceleration.mtp_selected is False
    assert view.grpo.acceleration.quantized_kv_cache_selected is False
    assert view.grpo.acceleration.speculative_acceptance.state == "missing"
    assert view.completeness.state == "complete"
    assert view.completeness.research_ready is True
    assert next(item for item in view.completeness.requirements if item.key == "mtp_runtime").state == "not_applicable"


@pytest.mark.asyncio
async def test_olmo3_active_sampling_is_exposed_as_conditional_evidence() -> None:
    run_id = "runs/olmo3-active-sampling"
    active_metrics = {
        "train/rl/active_sampling_generation_rounds": 2.0,
        "train/rl/active_sampling_retained_fraction": 0.875,
        "train/rl/active_sampling_generated_rows": 144.0,
        "train/rl/active_sampling_candidate_groups_reserved": 128.0,
        "train/rl/active_sampling_candidate_groups_generated": 144.0,
        "train/rl/active_sampling_candidate_groups_retained": 126.0,
        "train/rl/active_sampling_candidate_groups_unused": 0.0,
    }
    details = {
        run_id: RunDetail(
            summary=_summary(run_id, "train.grpo"),
            # Earlier runs did not retain algorithm selection in their snapshot.
            # Native active-sampling evidence must still activate its audit.
            resolved_inputs={"settings": {}},
            metric_names=tuple(active_metrics),
        )
    }
    series = {
        name: MetricSeries(name=name, points=(MetricPoint(value=value, step=1),))
        for name, value in active_metrics.items()
    }

    view = await ObservatoryService(FakeRunDataSource(details, {run_id: series})).get_run_view(run_id)

    chart_keys = [chart.key for chart in view.charts]
    assert {"active_sampling_yield", "active_sampling_population"}.issubset(chart_keys)
    assert chart_keys[-2:] == ["active_sampling_yield", "active_sampling_population"]
    requirement = next(item for item in view.completeness.requirements if item.key == "olmo3_active_sampling")
    assert requirement.state == "available"
    assert requirement.missing_metrics == ()


@pytest.mark.asyncio
async def test_grpo_tool_environment_category_activates_tool_evidence() -> None:
    source = FixtureRunDataSource()
    run_id = "runs/grpo-silver-pine"
    baseline = await ObservatoryService(source).get_run_view(run_id)
    detail = source._details[run_id]  # noqa: SLF001 - deterministic fixture mutation
    source._details[run_id] = detail.model_copy(  # noqa: SLF001
        update={"resolved_inputs": {**detail.resolved_inputs, "environment_category": "agentic-tool-use"}}
    )

    view = await ObservatoryService(source).get_run_view(run_id)

    tool_requirement = next(item for item in view.completeness.requirements if item.key == "tool_behavior")
    assert tool_requirement.state == "missing"
    assert view.completeness.conditional_active == baseline.completeness.conditional_active + 1
    assert view.completeness.research_ready is False


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


@pytest.mark.asyncio
async def test_compare_evaluations_requires_the_same_population_but_allows_model_changes() -> None:
    source = _source()
    first = "runs/eval-first"
    second = "runs/eval-second"

    def inputs(environment: str, model: str) -> dict[str, JsonValue]:
        return {
            "evaluation_plan": {"selection_id": "plans/ifeval", "revision": "1"},
            "environment": {
                "selection_id": environment,
                "revision": "env-rev",
                "resolved": {
                    "activation": {
                        "config": {"taskset": {"dataset_repo": "google/IFEval", "split": "train", "order_seed": 0}}
                    },
                    "source_revision": environment,
                    "reward_components": ["strict_prompt_accuracy"],
                },
            },
            "model": {"selection_id": model, "revision": "model-rev"},
        }

    source.details[first] = RunDetail(
        summary=_summary(first, "eval.general"), resolved_inputs=inputs("env/ifeval", "models/a")
    )
    source.details[second] = RunDetail(
        summary=_summary(second, "eval.general"), resolved_inputs=inputs("env/ifeval", "models/b")
    )
    source.series[first] = {}
    source.series[second] = {}
    comparable = await ObservatoryService(source).compare_runs((first, second))
    assert comparable.state == "comparable"
    assert comparable.rows[0].context["model"] == "models/a"
    source.details[second] = source.details[second].model_copy(
        update={"resolved_inputs": inputs("env/reasoning", "models/b")}
    )
    incomparable = await ObservatoryService(source).compare_runs((first, second))
    assert incomparable.state == "incomparable"
    assert incomparable.reason is not None and "evaluation populations" in incomparable.reason


def test_cli_exposes_the_same_telemetry_schema(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["schema", "train.sft"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["job_kind"] == "train.sft"
    assert payload["summary_fields"][0]["key"] == "final_loss"
