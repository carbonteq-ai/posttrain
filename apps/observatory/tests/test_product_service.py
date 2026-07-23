"""End-to-end logical-model tests for Observatory product views."""

from __future__ import annotations

import pytest
from posttrain_observatory import (
    FixtureRunDataSource,
    FixtureSemanticSummaryProvider,
    GenericRunView,
    MetricSeriesQuery,
    ObservatoryService,
    RunLocator,
    SemanticSummaryRequest,
)


@pytest.fixture
def service() -> ObservatoryService:
    return ObservatoryService(
        {"fixture": FixtureRunDataSource()},
        semantic_provider=FixtureSemanticSummaryProvider(),
    )


@pytest.mark.asyncio
async def test_unknown_job_falls_back_and_redacts_config(service: ObservatoryService) -> None:
    locator = RunLocator(source_id="fixture", run_id="runs/custom-orbit")
    response = await service.get_run_view_response(locator)
    assert response.resolved_mode == "generic"
    assert response.fallback_reason is not None
    assert isinstance(response.view, GenericRunView)
    assert response.view.resolved_inputs["training"]["api_key"] == "[REDACTED]"  # type: ignore[index]
    series = await service.get_metric_series(locator, MetricSeriesQuery(names=("custom/quality",), max_points=10))
    assert series.series[0].points[-1].value == 0.59


@pytest.mark.asyncio
async def test_eval_population_is_trace_derived_and_drillable(service: ObservatoryService) -> None:
    locator = RunLocator(source_id="fixture", run_id="runs/eval-violet-river")
    response = await service.get_run_view_response(locator)
    assert response.view.view_kind == "job.evaluation"
    evaluation = await service.get_trace_evaluation_view(locator)
    assert evaluation.state == "complete"
    assert evaluation.included == 12
    detail = await service.get_trace_detail(locator, evaluation.traces[0].external_id)
    assert detail.reward_components
    assert detail.transcript[0]["role"] == "user"


@pytest.mark.asyncio
async def test_semantic_summary_is_explicit_cited_and_cached(service: ObservatoryService) -> None:
    locator = RunLocator(source_id="fixture", run_id="runs/sft-calm-harbor")
    response = await service.get_run_view_response(locator)
    assert response.view.view_kind == "job.metrics"
    assert response.view.resolved_inputs["training"]["api_key"] == "[REDACTED]"  # type: ignore[index]
    assert response.view.execution_targets[0].selection_id == "targets/fixture-cuda-24gb"
    assert response.view.execution_targets[0].aggregate_memory_bytes == 24 * 1024**3
    first = await service.summarize_run(locator, SemanticSummaryRequest())
    second = await service.summarize_run(locator, SemanticSummaryRequest())
    assert first.status == "ready"
    assert first == second
    assert first.summary is not None
    assert first.summary.claims[0].citations[0].evidence_id == "run:status"


@pytest.mark.asyncio
async def test_work_package_uses_source_qualified_lineage(service: ObservatoryService) -> None:
    view = await service.get_work_package_view("train/reward-v2", project_id="projects/automation-agent")
    assert view.description is not None
    assert "candidate model-improvement techniques" in view.description
    assert {group.job_kind for group in view.job_groups} >= {"train.sft", "train.dpo", "train.grpo"}
    sft = next(group for group in view.job_groups if group.job_kind == "train.sft")
    assert sft.definitions[0].id == "train.sft@1"
    assert sft.definitions[0].description is not None
    assert "supervised demonstrations" in sft.definitions[0].description
    assert all(locator.source_id == "fixture" for locator, _ in view.lineage)


@pytest.mark.asyncio
async def test_system_metrics_are_cross_job_and_keep_missing_evidence_explicit(
    service: ObservatoryService,
) -> None:
    locator = RunLocator(source_id="fixture", run_id="runs/sft-calm-harbor")
    system = await service.get_system_metrics(locator)
    assert system.state == "available"
    assert system.window_finished_at is not None
    assert system.window_started_at < system.window_finished_at
    assert system.sample_count == 6
    assert {group.key for group in system.groups} == {"compute", "memory", "runtime"}
    assert next(value for value in system.summary if value.key == "gpu_utilization").value == 77
    assert next(value for value in system.summary if value.key == "traces_dropped").state == "missing"
    assert system.phase_state == "available"
    assert system.vram_capacity_state == "available"
    assert system.vram_capacity_bytes == 24 * 1024**3
    assert system.vram_observed_peak_bytes == 15_300_000_000
    assert system.execution_targets[0].roles == ("training",)
    assert any(interval.phase == "operation" for interval in system.phase_intervals)
    assert system.unclassified_sample_count == 0
    assert [phase.phase for phase in system.phase_summary] == [
        "model_loading",
        "actor_update",
        "artifact_export",
    ]
    actor_update = next(phase for phase in system.phase_summary if phase.phase == "actor_update")
    actor_vram = next(metric for metric in actor_update.metrics if metric.metric == "system/gpu_vram_used_bytes")
    assert actor_update.sample_count == 4
    assert actor_vram.mean == 14_775_000_000
    assert actor_vram.peak == 15_300_000_000


@pytest.mark.asyncio
async def test_nested_runtime_phases_do_not_double_count_host_samples(
    service: ObservatoryService,
) -> None:
    system = await service.get_system_metrics(RunLocator(source_id="fixture", run_id="runs/grpo-silver-pine"))
    rollout = next(phase for phase in system.phase_summary if phase.phase == "rollout")
    actor_update = next(phase for phase in system.phase_summary if phase.phase == "actor_update")
    assert rollout.occurrences == 2
    assert rollout.duration_s == 240
    assert actor_update.duration_s == 240
    assert rollout.sample_count == 2
    assert actor_update.sample_count == 2
    assert (
        sum(phase.sample_count for phase in system.phase_summary if phase.phase != "operation") == system.sample_count
    )
