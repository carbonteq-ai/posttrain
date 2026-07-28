"""End-to-end logical-model tests for Observatory product views."""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta

import pytest
from posttrain.tracking import MetricPoint, MetricSeries, TracePage, TraceRecord
from posttrain_observatory import (
    FixtureRunDataSource,
    FixtureSemanticSummaryProvider,
    GenericRunView,
    MetricSeriesQuery,
    ObservatoryService,
    RunLocator,
    SemanticSummaryRequest,
    ServingBenchmarkRunView,
)
from posttrain_observatory.service import _inference_timing_summary
from posttrain_observatory.traces import trace_evaluation_view


@pytest.fixture
def service() -> ObservatoryService:
    return ObservatoryService(
        {"fixture": FixtureRunDataSource()},
        semantic_provider=FixtureSemanticSummaryProvider(),
    )


def _serving_comparison_source() -> FixtureRunDataSource:
    source = FixtureRunDataSource()
    template_id = "runs/serve-cedar-point"
    detail = source._details[template_id]  # noqa: SLF001 - deterministic fixture construction
    template_metrics = source._metrics.pop(template_id)  # noqa: SLF001
    template_artifacts = source._artifacts.pop(template_id)  # noqa: SLF001
    source._details.pop(template_id)  # noqa: SLF001
    source._traces.pop(template_id)  # noqa: SLF001

    scenarios = (
        ("eligible", 58.0, 0.70, 32_768, "models/eligible", "inference/eligible", "eligible"),
        ("latency", 71.0, 1.40, 32_768, "models/latency", "inference/latency", "latency_constrained"),
        ("context", 62.0, 0.65, 8_192, "models/context", "inference/context", "context_failed"),
        ("below", 46.0, 0.60, 32_768, "models/below", "inference/below", "below_capacity"),
    )
    for scenario_index, (name, final_tps, ttft, context, model_id, binding_id, _) in enumerate(scenarios):
        run_id = f"runs/serve-{name}"
        summary = detail.summary.model_copy(
            update={
                "provider_run_id": f"fixture-{run_id}",
                "run_id": run_id,
                "display_name": f"Serving {name}",
                "started_at": detail.summary.started_at + timedelta(minutes=scenario_index),
            }
        )
        resolved = deepcopy(detail.resolved_inputs)
        resolved["model"]["selection_id"] = model_id  # type: ignore[index]
        resolved["screen_inference"]["selection_id"] = binding_id  # type: ignore[index]
        resolved["workload"]["resolved"]["saturation_state"] = "saturated"  # type: ignore[index]
        concurrencies = (1.0, 2.0, 4.0)
        point_tps = (final_tps * 0.55, final_tps * 0.78, final_tps)

        def metric_series(metric_name: str, values: tuple[float, ...]) -> MetricSeries:
            return MetricSeries(
                name=metric_name,
                points=tuple(MetricPoint(value=value, step=step) for step, value in enumerate(values)),
            )

        metrics = {
            "serve/run/concurrency": metric_series("serve/run/concurrency", concurrencies),
            "serve/run/context_tokens": metric_series(
                "serve/run/context_tokens",
                (float(context),) * 3,
            ),
            "serve/run/requests_attempted": metric_series(
                "serve/run/requests_attempted",
                (2.0,) * 3,
            ),
            "serve/run/requests_measured": metric_series(
                "serve/run/requests_measured",
                (2.0,) * 3,
            ),
            "serve/run/requests_failed": metric_series(
                "serve/run/requests_failed",
                (0.0,) * 3,
            ),
            "serve/run/requests_unsupported": metric_series(
                "serve/run/requests_unsupported",
                (0.0,) * 3,
            ),
            "serve/run/output_tokens_measured": metric_series(
                "serve/run/output_tokens_measured",
                (116.0,) * 3,
            ),
            "serve/run/measurement_duration_s": metric_series(
                "serve/run/measurement_duration_s",
                tuple(116.0 / tps for tps in point_tps),
            ),
            "serve/backend/peak_vram_bytes": metric_series(
                "serve/backend/peak_vram_bytes",
                tuple((6.8 + 0.1 * step + 0.05 * scenario_index) * 1024**3 for step in range(3)),
            ),
        }
        traces = tuple(
            TraceRecord(
                trace_type="inference",
                external_id=f"{name}-{sweep_index}-{request_index}",
                payload={
                    "sweep_index": sweep_index,
                    "concurrency": int(concurrency),
                    "warmup": False,
                    "input_tokens": 512,
                    "output_tokens": 58,
                    "ttft_seconds": ttft,
                    "tpot_seconds": 0.015,
                    "error_class": None,
                },
            )
            for sweep_index, concurrency in enumerate(concurrencies)
            for request_index in range(2)
        )
        source._details[run_id] = detail.model_copy(  # noqa: SLF001
            update={
                "summary": summary,
                "resolved_inputs": resolved,
                "metric_names": tuple(metrics),
                "trace_count": len(traces),
            }
        )
        source._metrics[run_id] = metrics  # noqa: SLF001
        source._traces[run_id] = traces  # noqa: SLF001
        source._artifacts[run_id] = template_artifacts  # noqa: SLF001
    assert template_metrics
    return source


@pytest.mark.asyncio
async def test_inference_timing_summary_uses_request_grain_without_summing_stages() -> None:
    class TraceSource:
        async def traces(self, run_id: str, query) -> TracePage:
            assert run_id == "serve-run"
            assert query.trace_type == "inference"
            return TracePage(
                items=(
                    TraceRecord(
                        trace_type="inference",
                        external_id="request-1",
                        payload={
                            "queue_seconds": 0.001,
                            "prefill_seconds": 0.010,
                            "decode_seconds": 0.200,
                            "engine_e2e_seconds": 0.211,
                        },
                    ),
                    TraceRecord(
                        trace_type="inference",
                        external_id="request-2",
                        payload={
                            "queue_seconds": 0.003,
                            "prefill_seconds": 0.014,
                            "decode_seconds": 0.240,
                            "engine_e2e_seconds": 0.257,
                        },
                    ),
                )
            )

    summary = await _inference_timing_summary(TraceSource(), "serve-run")  # type: ignore[arg-type]

    assert summary is not None
    assert summary.requests == 2
    prefill = next(stage for stage in summary.stages if stage.stage == "prefill")
    assert prefill.p50_ms == 14
    assert prefill.mean_ms == 12


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
async def test_serving_benchmark_has_constraint_relative_job_view(service: ObservatoryService) -> None:
    locator = RunLocator(source_id="fixture", run_id="runs/serve-cedar-point")

    response = await service.get_run_view_response(locator)

    assert response.resolved_mode == "job"
    assert response.fallback_reason is None
    assert isinstance(response.view, ServingBenchmarkRunView)
    assert response.view.view_kind == "job.serving"
    assert response.view.eligibility.state == "eligible"
    assert response.view.eligibility.calculator_version == "serving-capacity-v1"
    assert response.view.selected_point is not None
    assert response.view.selected_point.aggregate_output_tps == 58
    assert response.view.selected_point.concurrency == 4
    assert {requirement.state for requirement in response.view.requirements} == {"pass"}
    assert response.view.population.correctness_scored is False
    assert response.view.population.output_length_policy == "fixed"
    assert response.view.population.output_target_hit_rate == 1
    scheduler = next(group for group in response.view.runtime_settings if group.key == "scheduler")
    assert next(setting for setting in scheduler.settings if setting.key == "max_num_batched_tokens").value == 4096
    additional = next(group for group in response.view.runtime_settings if group.key == "additional")
    assert next(setting for setting in additional.settings if setting.key == "experimental_scheduler").value == "async"


@pytest.mark.asyncio
async def test_serving_capacity_work_package_lists_cross_run_points(service: ObservatoryService) -> None:
    view = await service.get_serving_capacity_view(
        "screen/serving-capacity-v1",
        project_id="projects/automation-agent",
        source_id="fixture",
    )

    assert view.methodology == "cross_run_compatibility"
    assert "historical single-point runs" in view.explanation
    assert len(view.rows) == 1
    assert view.rows[0].point.concurrency == 4
    assert view.rows[0].point.aggregate_output_tps == 58
    assert view.rows[0].eligibility.state == "eligible"


@pytest.mark.asyncio
async def test_serving_capacity_work_package_keeps_all_states_and_computes_pareto() -> None:
    view = await ObservatoryService({"fixture": _serving_comparison_source()}).get_serving_capacity_view(
        "screen/serving-capacity-v1",
        project_id="projects/automation-agent",
        source_id="fixture",
    )

    assert view.methodology == "strict_pareto"
    assert {contender.eligibility.state for contender in view.contenders} == {
        "eligible",
        "latency_constrained",
        "context_failed",
        "below_capacity",
    }
    assert all(contender.comparable for contender in view.contenders)
    assert len(view.pareto) == 1
    assert view.pareto[0].inference_binding_id == "inference/eligible"
    assert next(
        contender for contender in view.contenders if contender.inference_binding_id == "inference/eligible"
    ).pareto_member


@pytest.mark.asyncio
async def test_serving_capacity_marks_target_mismatch_incomparable() -> None:
    source = _serving_comparison_source()
    detail = source._details["runs/serve-latency"]  # noqa: SLF001
    resolved = deepcopy(detail.resolved_inputs)
    resolved["target"]["selection_id"] = "targets/other-gpu"  # type: ignore[index]
    source._details["runs/serve-latency"] = detail.model_copy(  # noqa: SLF001
        update={"resolved_inputs": resolved}
    )

    view = await ObservatoryService({"fixture": source}).get_serving_capacity_view(
        "screen/serving-capacity-v1",
        project_id="projects/automation-agent",
        source_id="fixture",
    )

    mismatched = next(
        contender for contender in view.contenders if contender.inference_binding_id == "inference/latency"
    )
    assert mismatched.comparable is False
    assert mismatched.comparability_reason is not None
    assert "execution target" in mismatched.comparability_reason
    assert mismatched.eligibility.state == "latency_constrained"


@pytest.mark.asyncio
async def test_serving_run_retains_resource_boundary_after_valid_points() -> None:
    source = _serving_comparison_source()
    run_id = "runs/serve-eligible"
    metrics = source._metrics[run_id]  # noqa: SLF001

    def append(name: str, value: float) -> None:
        current = metrics[name]
        metrics[name] = MetricSeries(
            name=name,
            points=(*current.points, MetricPoint(value=value, step=3)),
        )

    append("serve/run/concurrency", 8)
    metrics["serve/run/requests_attempted"] = MetricSeries(
        name="serve/run/requests_attempted",
        points=(*metrics["serve/run/requests_attempted"].points, MetricPoint(value=0, step=3)),
    )
    metrics["serve/run/point_resource_exhausted"] = MetricSeries(
        name="serve/run/point_resource_exhausted",
        points=(
            MetricPoint(value=0, step=0),
            MetricPoint(value=0, step=1),
            MetricPoint(value=0, step=2),
            MetricPoint(value=1, step=3),
        ),
    )
    detail = source._details[run_id]  # noqa: SLF001
    resolved = deepcopy(detail.resolved_inputs)
    resolved["workload"]["resolved"]["saturation_state"] = "unknown"  # type: ignore[index]
    source._details[run_id] = detail.model_copy(  # noqa: SLF001
        update={"metric_names": tuple(metrics), "resolved_inputs": resolved}
    )

    response = await ObservatoryService({"fixture": source}).get_run_view_response(
        RunLocator(source_id="fixture", run_id=run_id)
    )

    assert isinstance(response.view, ServingBenchmarkRunView)
    assert tuple(point.concurrency for point in response.view.operating_points) == (1, 2, 4, 8)
    boundary = response.view.operating_points[-1]
    assert boundary.terminal_status == "resource_exhausted"
    assert boundary.evidence_state == "partial"
    assert response.view.selected_point is not None
    assert response.view.selected_point.concurrency == 4
    assert response.view.eligibility.state == "eligible"


@pytest.mark.asyncio
async def test_serving_benchmark_projects_one_run_level_concurrency_sweep() -> None:
    source = FixtureRunDataSource()
    run_id = "runs/serve-cedar-point"
    metrics = source._metrics[run_id]  # noqa: SLF001 - deterministic fixture mutation
    workload = source._details[run_id].resolved_inputs["workload"]  # noqa: SLF001
    assert isinstance(workload, dict)
    resolved_workload = workload["resolved"]
    assert isinstance(resolved_workload, dict)
    resolved_workload["saturation_state"] = "unsaturated"

    def series(name: str, values: tuple[float, ...]) -> MetricSeries:
        return MetricSeries(
            name=name,
            points=tuple(MetricPoint(value=value, step=index) for index, value in enumerate(values)),
        )

    metrics.update(
        {
            "serve/concurrency": series("serve/concurrency", (1, 2, 4)),
            "serve/requests": series("serve/requests", (2, 2, 2)),
            "serve/output_tokens": series("serve/output_tokens", (116, 116, 116)),
            "serve/elapsed_seconds": series("serve/elapsed_seconds", (2, 1, 0.8)),
            "serve/context_window": series("serve/context_window", (32_768, 32_768, 32_768)),
            "serve/peak_gpu_memory_gib": series("serve/peak_gpu_memory_gib", (6.8, 7.0, 7.2)),
        }
    )
    source._traces[run_id] = tuple(  # noqa: SLF001 - deterministic fixture mutation
        TraceRecord(
            trace_type="inference",
            external_id=f"request-{sweep_index}-{request_index}",
            payload={
                "sweep_index": sweep_index,
                "concurrency": concurrency,
                "warmup": False,
                "input_tokens": 512,
                "output_tokens": 58,
                "ttft_seconds": 0.2 + 0.05 * sweep_index,
                "tpot_seconds": 0.015,
                "error_class": None,
            },
        )
        for sweep_index, concurrency in enumerate((1, 2, 4))
        for request_index in range(2)
    )

    response = await ObservatoryService({"fixture": source}).get_run_view_response(
        RunLocator(source_id="fixture", run_id=run_id)
    )

    assert isinstance(response.view, ServingBenchmarkRunView)
    assert tuple(point.concurrency for point in response.view.operating_points) == (1, 2, 4)
    assert response.view.selected_point is not None
    assert response.view.selected_point.concurrency == 4
    assert response.view.selected_point.aggregate_output_tps == 145
    assert response.view.eligibility.state == "unsaturated"
    assert response.view.eligibility.label == "Sweep passes; saturation not reached"


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
async def test_eval_population_projects_verifiers_v1_wire_trace_fields() -> None:
    source = FixtureRunDataSource()
    run_id = "runs/eval-wire-trace"
    source._traces[run_id] = (  # noqa: SLF001 - deterministic fixture mutation
        TraceRecord(
            trace_type="verifiers",
            external_id="trace-complete",
            payload={
                "rewards": {"correct": 1.0},
                "metrics": {},
                "info": {
                    "task": {
                        "type": "GSM8KTask",
                        "data": {
                            "idx": 1,
                            "prompt": "must not become a slice label",
                            "answer": "42",
                        },
                    }
                },
                "is_completed": True,
                "stop_condition": "agent_completed",
                "errors": [],
                "nodes": [
                    {
                        "sampled": True,
                        "message": {"role": "assistant", "tool_calls": [{"name": "calculator"}]},
                    }
                ],
                "calls": [{"finish_reason": "stop", "error": None}],
            },
        ),
        TraceRecord(
            trace_type="verifiers",
            external_id="trace-truncated",
            payload={
                "rewards": {"correct": 0.0},
                "info": {"example_id": "gsm8k-2"},
                "is_completed": False,
                "stop_condition": "max_output_tokens",
                "errors": [{"type": "model_timeout", "message": "redacted in summary"}],
                "nodes": [],
                "calls": [],
            },
        ),
    )

    evaluation = await trace_evaluation_view(source, run_id, expected=2)

    assert evaluation.state == "complete"
    assert evaluation.mean_reward == 0.5
    assert evaluation.success_rate == 0.5
    assert evaluation.failures == 1
    assert evaluation.truncated == 1
    assert evaluation.traces[0].task == "GSM8KTask:1"
    assert evaluation.traces[0].tool_calls == 1
    assert evaluation.traces[1].error == "model_timeout"


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
    assert [(phase.phase, phase.group) for phase in system.phase_summary] == [
        ("model_loading", "startup"),
        ("actor_update", "training"),
        ("artifact_export", "finalization"),
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
