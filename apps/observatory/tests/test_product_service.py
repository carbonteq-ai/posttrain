"""End-to-end logical-model tests for Observatory product views."""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from typing import cast

import pytest
from posttrain.common import JsonValue
from posttrain.tracking import (
    MetricPoint,
    MetricSeries,
    TraceAggregateBucket,
    TraceAggregateResult,
    TracePage,
    TraceRecord,
)
from posttrain_observatory import (
    EvaluationBreakdownSpec,
    EvaluationFacetSpec,
    EvaluationMetadata,
    FixtureRunDataSource,
    FixtureSemanticSummaryProvider,
    GenericRunView,
    MetricSeriesQuery,
    ObservatoryService,
    RunLocator,
    SemanticSummaryRequest,
    ServingBenchmarkRunView,
)
from posttrain_observatory.models import EvaluationSuccessDefinition
from posttrain_observatory.redaction import RedactionPolicy
from posttrain_observatory.service import (
    _evaluation_expected_traces,
    _evaluation_metadata,
    _inference_timing_summary,
)
from posttrain_observatory.traces import project_trace, rollout_behavior_view, trace_evaluation_view


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
async def test_trace_population_uses_the_job_telemetry_trace_type() -> None:
    source = FixtureRunDataSource()
    service = ObservatoryService({"fixture": source})

    evaluation = await service.get_trace_evaluation_view(
        RunLocator(source_id="fixture", run_id="runs/serve-cedar-point")
    )

    assert evaluation.included > 0
    assert {trace.trace_type for trace in evaluation.traces} == {"inference"}


@pytest.mark.asyncio
async def test_trace_summary_page_is_provider_bounded_and_cursor_driven() -> None:
    source = FixtureRunDataSource()
    service = ObservatoryService({"fixture": source})
    locator = RunLocator(source_id="fixture", run_id="runs/eval-violet-river")

    first = await service.get_trace_summary_page(locator, limit=3)
    second = await service.get_trace_summary_page(locator, cursor=first.next_cursor, limit=3)

    assert first.total == 12
    assert len(first.items) == 3
    assert first.next_cursor == "3"
    assert len(second.items) == 3
    assert second.next_cursor == "6"
    assert {item.external_id for item in first.items}.isdisjoint(item.external_id for item in second.items)


@pytest.mark.asyncio
async def test_trace_evaluation_can_omit_trace_rows_without_losing_aggregates() -> None:
    service = ObservatoryService({"fixture": FixtureRunDataSource()})
    locator = RunLocator(source_id="fixture", run_id="runs/eval-violet-river")

    evaluation = await service.get_trace_evaluation_view(locator, include_traces=False)

    assert evaluation.included == 12
    assert evaluation.scored == 12
    assert evaluation.mean_reward is not None
    assert evaluation.traces == ()


@pytest.mark.asyncio
async def test_trace_page_reuses_run_metadata_loaded_for_overview() -> None:
    class CountingSource(FixtureRunDataSource):
        get_run_calls = 0

        async def get_run(self, run_id: str):
            self.get_run_calls += 1
            return await super().get_run(run_id)

    source = CountingSource()
    service = ObservatoryService({"fixture": source})
    locator = RunLocator(source_id="fixture", run_id="runs/grpo-silver-pine")

    await service.get_run_view_response(locator)
    page = await service.get_trace_summary_page(locator, limit=2)

    assert len(page.items) == 2
    assert source.get_run_calls == 1


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
                        "message": {
                            "role": "assistant",
                            "content": "42",
                            "reasoning_content": "Check the arithmetic.",
                            "tool_calls": [{"name": "calculator"}],
                        },
                    }
                ],
                "calls": [
                    {
                        "finish_reason": "stop",
                        "error": None,
                        "usage": {"prompt_tokens": 8, "completion_tokens": 12, "reasoning_tokens": 5},
                        "time": {"start": 10.0, "end": 12.5},
                    }
                ],
            },
        ),
        TraceRecord(
            trace_type="verifiers",
            external_id="trace-truncated",
            payload={
                "rewards": {"correct": 0.0},
                "info": {"example_id": "gsm8k-2"},
                "is_completed": False,
                "errors": [{"type": "model_timeout", "message": "redacted in summary"}],
                "nodes": [],
            },
            attributes={"is_truncated": True},
        ),
    )

    evaluation = await trace_evaluation_view(source, run_id, expected=2)

    assert evaluation.state == "complete"
    assert evaluation.mean_reward == 0.5
    assert evaluation.success_rate == 0.5
    assert evaluation.scored == 2
    assert evaluation.passed == 1
    assert evaluation.pass_scored == 2
    assert evaluation.failures == 1
    assert evaluation.truncated == 1
    assert evaluation.traces[0].task == "GSM8KTask:1"
    assert evaluation.traces[0].prompt_preview == "must not become a slice label"
    assert evaluation.traces[0].task_metadata is not None
    assert evaluation.traces[0].task_metadata.label == "GSM8K Task 1"
    assert evaluation.traces[0].tool_calls == 1
    assert evaluation.traces[0].response_tokens == 7
    assert evaluation.traces[0].input_tokens == 8
    assert evaluation.traces[0].completion_tokens == 12
    assert evaluation.traces[0].tokens == 12
    assert evaluation.traces[0].latency_ms == 2500
    assert evaluation.performance.latency_ms is not None
    assert evaluation.performance.latency_ms.model_dump() == {
        "samples": 1,
        "mean": 2500.0,
        "p50": 2500.0,
        "p95": 2500.0,
        "maximum": 2500.0,
    }
    assert evaluation.performance.completion_tokens is not None
    assert evaluation.performance.completion_tokens.p50 == 12
    assert evaluation.performance.thinking_tokens is not None
    assert evaluation.performance.thinking_tokens.p50 == 5
    assert evaluation.traces[0].response_chars == 2
    assert evaluation.traces[0].thinking_chars == len("Check the arithmetic.")
    assert evaluation.traces[0].model_calls == 1
    assert evaluation.traces[0].outcome == "pass"
    assert evaluation.traces[1].error == "model_timeout"
    assert evaluation.traces[1].outcome == "error"
    detail = project_trace(source._traces[run_id][0], RedactionPolicy())  # noqa: SLF001 - deterministic fixture inspection
    assert detail.transcript[0]["reasoning_content"] == "Check the arithmetic."


def test_trace_summary_projects_bounded_provider_scalars() -> None:
    detail = project_trace(
        TraceRecord(
            trace_type="verifiers",
            external_id="trace-bounded",
            payload={
                "num_tool_calls": 0,
                "num_model_calls": 1,
                "input_tokens": 3172,
                "completion_tokens": 825,
                "latency_ms": 48617.406,
            },
        ),
        RedactionPolicy(),
    )

    assert detail.summary.tool_calls == 0
    assert detail.summary.model_calls == 1
    assert detail.summary.input_tokens == 3172
    assert detail.summary.completion_tokens == 825
    assert detail.summary.tokens == 825
    assert detail.summary.latency_ms == 48617.406


def test_ifeval_task_metadata_uses_instruction_families_not_numeric_key() -> None:
    record = TraceRecord(
        trace_type="verifiers",
        external_id="ifeval-trace",
        payload={
            "task": {
                "type": "IFEvalTask",
                "data": {
                    "idx": 13,
                    "name": "ifeval-13",
                    "instruction_id_list": [
                        "detectable_format:json_format",
                        "keywords:existence",
                    ],
                    "source_repo": "google/IFEval",
                    "source_revision": "9" * 40,
                    "source_split": "train",
                },
            },
            "rewards": {"strict_prompt_accuracy": 1.0},
            "metrics": {"strict_prompt_accuracy": 1.0},
            "is_completed": True,
            "nodes": [],
            "calls": [],
        },
    )

    detail = project_trace(record, RedactionPolicy())
    metadata = detail.summary.task_metadata
    assert metadata is not None
    assert metadata.key == "IFEvalTask:ifeval-13"
    assert metadata.label == "Detectable Format + Keywords · IFEval 13"
    assert metadata.category == "Detectable Format + Keywords"
    assert metadata.instruction_ids == (
        "detectable_format:json_format",
        "keywords:existence",
    )
    assert metadata.instruction_families == ("detectable_format", "keywords")
    assert [(facet.dimension, facet.value) for facet in metadata.facets] == [
        ("instruction_family", "detectable_format"),
        ("instruction_family", "keywords"),
    ]


@pytest.mark.asyncio
async def test_ifeval_evaluation_exposes_multi_label_instruction_facets() -> None:
    source = FixtureRunDataSource()
    source._traces["runs/ifeval-facets"] = (  # noqa: SLF001 - deterministic fixture construction
        TraceRecord(
            trace_type="verifiers",
            external_id="ifeval-facet-trace",
            payload={
                "task": {
                    "type": "IFEvalTask",
                    "data": {
                        "idx": 13,
                        "name": "ifeval-13",
                        "instruction_id_list": ["detectable_format:json_format", "keywords:existence"],
                    },
                },
                "rewards": {"strict_prompt_accuracy": 1.0},
                "metrics": {"strict_prompt_accuracy": 1.0},
                "is_completed": True,
                "nodes": [],
                "calls": [],
            },
        ),
    )

    evaluation = await trace_evaluation_view(source, "runs/ifeval-facets", expected=1)

    assert [(facet.key, facet.label, facet.count) for facet in evaluation.facets] == [
        ("instruction_family:detectable_format", "Detectable Format", 1),
        ("instruction_family:keywords", "Keywords", 1),
    ]


@pytest.mark.asyncio
async def test_native_environment_semantics_become_generic_evaluation_facets() -> None:
    source = FixtureRunDataSource()
    source._traces["runs/native-facets"] = (  # noqa: SLF001 - deterministic fixture construction
        TraceRecord(
            trace_type="verifiers",
            external_id="reasoning-gym-trace",
            payload={
                "task": {
                    "type": "ReasoningGymTask",
                    "data": {"idx": 0, "name": "eval:countdown:0", "generator": "countdown"},
                },
                "rewards": {"native_reward": 1.0},
                "nodes": [],
                "calls": [],
            },
        ),
        TraceRecord(
            trace_type="verifiers",
            external_id="math-python-trace",
            payload={
                "task": {
                    "type": "MathPythonTask",
                    "data": {
                        "idx": 1,
                        "name": "test:1",
                        "problem_type": "algebra",
                        "level": "Level 3",
                    },
                },
                "rewards": {"math_reward": 1.0},
                "nodes": [],
                "calls": [],
            },
        ),
        TraceRecord(
            trace_type="verifiers",
            external_id="mmlu-pro-trace",
            payload={
                "task": {
                    "type": "MMLUProTask",
                    "data": {"idx": 2, "name": "mmlu-pro-2", "category": "computer science"},
                },
                "rewards": {"answer_correct": 1.0},
                "nodes": [],
                "calls": [],
            },
        ),
    )

    evaluation = await trace_evaluation_view(source, "runs/native-facets", expected=3)

    assert [(facet.key, facet.dimension_label, facet.count) for facet in evaluation.facets] == [
        ("category:computer science", "Category", 1),
        ("difficulty:Level 3", "Difficulty", 1),
        ("generator:countdown", "Generator", 1),
        ("problem_type:algebra", "Problem type", 1),
    ]


@pytest.mark.asyncio
async def test_resolved_environment_observation_selects_reward_pass_rate_and_facets() -> None:
    source = FixtureRunDataSource()
    source._traces["runs/declared-observation"] = (  # noqa: SLF001 - deterministic fixture construction
        TraceRecord(
            trace_type="verifiers",
            external_id="automationbench-trace",
            payload={
                "task": {
                    "type": "AutomationBenchTask",
                    "data": {"idx": 0, "name": "simple-0", "domain": "simple"},
                },
                "rewards": {"partial_credit": 0.5, "task_completed_correctly": 1.0},
                "metrics": {"partial_credit": 0.5, "task_completed_correctly": 1.0},
                "nodes": [],
                "calls": [],
            },
        ),
    )
    metadata = EvaluationMetadata(
        key="automationbench-v1",
        label="AutomationBench",
        primary_metric="partial_credit",
        primary_metric_label="Partial credit",
        pass_rate_metric="task_completed_correctly",
        facet_specs=(EvaluationFacetSpec(field="domain", dimension="domain", label="Domain"),),
    )

    evaluation = await trace_evaluation_view(source, "runs/declared-observation", expected=1, metadata=metadata)

    assert evaluation.mean_reward == 0.5
    assert evaluation.success_rate == 1.0
    assert evaluation.traces[0].outcome == "pass"
    assert [(facet.key, facet.label, facet.dimension_label) for facet in evaluation.facets] == [
        ("domain:simple", "Simple", "Domain"),
    ]


@pytest.mark.asyncio
async def test_declared_compound_breakdown_preserves_dimensions_and_coverage() -> None:
    source = FixtureRunDataSource()
    source._traces["runs/math-compound"] = tuple(  # noqa: SLF001 - deterministic fixture construction
        TraceRecord(
            trace_type="verifiers",
            external_id=external_id,
            payload={
                "task": {
                    "type": "MathPythonTask",
                    "data": {"idx": index, "name": external_id, "problem_type": problem_type, "level": level},
                },
                "rewards": cast(dict[str, JsonValue], rewards),
                "metrics": cast(dict[str, JsonValue], metrics),
                "error": error,
                "is_truncated": truncated,
                "nodes": [],
                "calls": [],
            },
        )
        for index, (external_id, problem_type, level, rewards, metrics, error, truncated) in enumerate(
            (
                (
                    "algebra-l1-pass",
                    "Algebra",
                    "Level 1",
                    {"math_reward": 1.0},
                    {"symbolic_correctness": 1.0},
                    None,
                    False,
                ),
                (
                    "algebra-l2-fail",
                    "Algebra",
                    "Level 2",
                    {"math_reward": 0.0},
                    {"symbolic_correctness": 0.0},
                    None,
                    False,
                ),
                ("geometry-l1-error", "Geometry", "Level 1", {}, {}, "ProviderError", False),
                ("geometry-l2-truncated", "Geometry", "Level 2", {}, {}, None, True),
            )
        )
    )
    metadata = EvaluationMetadata(
        key="math-python-v1",
        label="Math Python",
        primary_metric="math_reward",
        success_definition=EvaluationSuccessDefinition(
            id="symbolic-correctness",
            label="Symbolically correct",
            namespace="metric",
            signal="symbolic_correctness",
            operator="eq",
            value=1.0,
        ),
        facet_specs=(
            EvaluationFacetSpec(field="problem_type", dimension="problem_type", label="Problem type"),
            EvaluationFacetSpec(field="level", dimension="difficulty", label="Difficulty"),
        ),
        breakdown_specs=(
            EvaluationBreakdownSpec(
                id="problem-type-by-difficulty",
                label="Problem type × difficulty",
                dimensions=("problem_type", "difficulty"),
            ),
        ),
    )

    evaluation = await trace_evaluation_view(source, "runs/math-compound", expected=4, metadata=metadata)

    assert len(evaluation.breakdowns) == 1
    breakdown = evaluation.breakdowns[0]
    assert breakdown.dimensions == ("problem_type", "difficulty")
    assert breakdown.excluded == 0
    assert [
        (group.label, group.count, group.scored, group.failures, group.truncated) for group in breakdown.groups
    ] == [
        ("Algebra · Level 1", 1, 1, 0, 0),
        ("Algebra · Level 2", 1, 1, 0, 0),
        ("Geometry · Level 1", 1, 0, 1, 0),
        ("Geometry · Level 2", 1, 0, 0, 1),
    ]
    assert breakdown.groups[0].success_rate == 1.0
    assert breakdown.groups[1].success_rate == 0.0


def test_versioned_contract_is_authoritative_over_current_environment_observation() -> None:
    metadata = _evaluation_metadata(
        {
            "environment": {
                "selection_id": "env/ifeval",
                "resolved": {
                    "category": "instruction-following",
                    "package": "ifeval-v1",
                    "source_revision": "legacy-source",
                    "observation": {
                        "primary_metric": "wrong_current_catalog_metric",
                        "pass_rate_metric": "wrong_current_catalog_metric",
                    },
                },
            },
            "evaluation": {
                "contract": {"id": "posttrain.eval.verifiers-observation", "schema_version": 1},
                "environment": {
                    "package": "ifeval-v1",
                    "category": "instruction-following",
                    "source_revision": "pinned-source",
                },
                "population": {
                    "taskset": {
                        "dataset_repo": "google/IFEval",
                        "dataset_revision": "pinned-dataset",
                        "split": "train",
                    }
                },
                "signal_manifest": {
                    "reward_components": ["strict_prompt_accuracy"],
                    "observation": {
                        "primary_metric": "strict_prompt_accuracy",
                        "pass_rate_metric": "strict_prompt_accuracy",
                        "facets": [],
                    },
                },
            },
        }
    )

    assert metadata is not None
    assert metadata.contract_state == "versioned"
    assert metadata.primary_metric == "strict_prompt_accuracy"
    assert metadata.pass_rate_metric == "strict_prompt_accuracy"
    assert metadata.dataset == "google/IFEval"
    assert metadata.dataset_revision == "pinned-dataset"
    assert metadata.source_revision == "pinned-source"


def test_versioned_contract_normalizes_math_python_dataset_identity() -> None:
    metadata = _evaluation_metadata(
        {
            "environment": {
                "selection_id": "math-python",
                "resolved": {
                    "package": "math-python-v1",
                    "category": "math-tool-use",
                    "source_revision": "pinned-source",
                },
            },
            "evaluation": {
                "contract": {"id": "posttrain.eval.verifiers-observation", "schema_version": 2},
                "environment": {
                    "package": "math-python-v1",
                    "category": "math-tool-use",
                    "source_revision": "pinned-source",
                },
                "population": {
                    "taskset": {
                        "repository": "DigitalLearningGmbH/MATH-lighteval",
                        "revision": "pinned-dataset",
                        "split": "test",
                    }
                },
                "signal_manifest": {"reward_components": [], "observation": {}},
                "plan": {"success": None},
            },
        }
    )

    assert metadata is not None
    assert metadata.dataset == "DigitalLearningGmbH/MATH-lighteval"
    assert metadata.dataset_revision == "pinned-dataset"
    assert metadata.split == "test"


def test_versioned_evaluation_uses_frozen_population_for_expected_trace_count() -> None:
    inputs = {
        "evaluation": {
            "contract": {"id": "posttrain.eval.verifiers-observation", "schema_version": 3},
            "environment": {},
            "population": {"num_tasks": 500, "num_rollouts": 2},
            "signal_manifest": {},
            "plan": {},
            "native_evidence": {},
        }
    }

    assert _evaluation_expected_traces(inputs, fallback=9) == 1000


@pytest.mark.asyncio
async def test_v2_success_predicate_and_reward_components_are_projected_from_run_contract() -> None:
    inputs = {
        "environment": {
            "selection_id": "reasoning-gym",
            "resolved": {
                "package": "reasoning-gym-v1",
                "category": "procedural-reasoning",
                "source_revision": "pinned-source",
                "reward_components": ["native_reward"],
                "observation": {"primary_metric": "native_reward", "facets": []},
            },
        },
        "evaluation": {
            "contract": {"id": "posttrain.eval.verifiers-observation", "schema_version": 2},
            "environment": {
                "package": "reasoning-gym-v1",
                "category": "procedural-reasoning",
                "source_revision": "pinned-source",
            },
            "population": {"taskset": {"split": "eval"}},
            "signal_manifest": {
                "reward_components": ["native_reward"],
                "observation": {"primary_metric": "native_reward", "facets": []},
            },
            "plan": {
                "success": {
                    "id": "full-credit-solution",
                    "label": "Full-credit solution",
                    "source": {"namespace": "metric", "name": "native_score"},
                    "predicate": {"operator": "gte", "value": 0.99, "upper": None, "tolerance": 0.0},
                    "missing": "error",
                }
            },
            "native_evidence": {"schema_id": "verifiers.trace", "schema_version": "v1"},
        },
    }
    metadata = _evaluation_metadata(inputs)
    assert metadata is not None
    assert metadata.success_definition is not None
    assert metadata.pass_rate_basis == "metric.native_score gte 0.99"

    source = FixtureRunDataSource()
    source._traces["runs/reasoning-v2"] = (  # noqa: SLF001 - deterministic fixture construction
        TraceRecord(
            trace_type="verifiers",
            external_id="pass",
            payload={
                "rewards": {"native_reward": 1.0, "format_bonus": 0.2},
                "metrics": {"native_score": 1.0},
                "nodes": [],
                "calls": [],
            },
        ),
        TraceRecord(
            trace_type="verifiers",
            external_id="fail",
            payload={
                "rewards": {"native_reward": 0.5, "format_bonus": 0.1},
                "metrics": {"native_score": 0.5},
                "nodes": [],
                "calls": [],
            },
        ),
        TraceRecord(
            trace_type="verifiers",
            external_id="truncated",
            payload={
                "rewards": {"native_reward": 0.0},
                "metrics": {"native_score": 0.0},
                "truncated": True,
                "nodes": [],
                "calls": [],
            },
        ),
    )

    evaluation = await trace_evaluation_view(source, "runs/reasoning-v2", expected=3, metadata=metadata)

    assert evaluation.success_rate == 0.5
    assert evaluation.passed == 1
    assert evaluation.pass_scored == 2
    assert [trace.outcome for trace in evaluation.traces] == ["pass", "review", "truncated"]
    assert evaluation.traces[2].success is None
    assert evaluation.traces[0].reward_components == {"native_reward": 1.0, "format_bonus": 0.2}
    assert evaluation.traces[0].native_metrics == {"native_score": 1.0}


def test_unsupported_contract_does_not_infer_metric_from_legacy_observation() -> None:
    metadata = _evaluation_metadata(
        {
            "environment": {
                "selection_id": "env/future",
                "resolved": {
                    "observation": {"primary_metric": "reward", "pass_rate_metric": "reward"},
                    "reward_components": ["reward"],
                },
            },
            "evaluation": {"contract": {"id": "posttrain.eval.verifiers-observation", "schema_version": 99}},
        }
    )

    assert metadata is not None
    assert metadata.contract_state == "unsupported"
    assert metadata.primary_metric is None
    assert metadata.pass_rate_metric is None
    assert metadata.metrics == ()


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
async def test_rollout_behavior_uses_recorded_optimizer_steps_and_marks_unattributed_traces() -> None:
    class TraceSource:
        async def aggregate_trace_facts(self, run_id: str, query) -> TraceAggregateResult:
            assert run_id == "train-run"
            assert query.trace_type == "verifiers"
            return TraceAggregateResult(
                state="available",
                buckets=(
                    TraceAggregateBucket(
                        dimensions={"rollout_step": 4},
                        trace_count=2,
                        values={
                            "mean_thinking_tokens": None,
                            "mean_model_output_tokens": 120.0,
                            "mean_tool_calls": 2.0,
                        },
                    ),
                    TraceAggregateBucket(
                        dimensions={"rollout_step": 5},
                        trace_count=1,
                        values={
                            "mean_thinking_tokens": None,
                            "mean_model_output_tokens": 80.0,
                            "mean_tool_calls": 0.0,
                        },
                    ),
                    TraceAggregateBucket(
                        dimensions={"rollout_step": None},
                        trace_count=1,
                        values={
                            "mean_thinking_tokens": None,
                            "mean_model_output_tokens": 999.0,
                            "mean_tool_calls": 9.0,
                        },
                    ),
                ),
            )

        async def traces(self, run_id: str, query) -> TracePage:
            raise AssertionError("rollout behavior must not scan native payloads")

    view = await rollout_behavior_view(TraceSource(), "train-run", expected=4)  # type: ignore[arg-type]

    assert view.state == "complete"
    assert view.included == 3
    assert view.unattributed == 1
    assert [(point.step, point.rollouts) for point in view.points] == [(4, 2), (5, 1)]
    assert view.points[0].output_tokens == 120
    assert view.points[0].tool_calls == 2
    assert view.points[0].thinking_tokens is None


@pytest.mark.asyncio
async def test_rollout_behavior_reads_exact_qwen_thinking_tokens_from_persisted_facts() -> None:
    class TraceSource:
        async def aggregate_trace_facts(self, run_id: str, query) -> TraceAggregateResult:
            assert run_id == "train-run"
            return TraceAggregateResult(
                state="available",
                buckets=(
                    TraceAggregateBucket(
                        dimensions={"rollout_step": 4},
                        trace_count=1,
                        values={
                            "mean_thinking_tokens": 3.0,
                            "mean_model_output_tokens": 8.0,
                            "mean_tool_calls": 0.0,
                        },
                    ),
                ),
            )

        async def traces(self, run_id: str, query) -> TracePage:
            raise AssertionError("rollout behavior must not scan native payloads")

    view = await rollout_behavior_view(TraceSource(), "train-run", expected=1)  # type: ignore[arg-type]

    assert view.points[0].thinking_tokens == 3
    assert view.points[0].output_tokens == 8


@pytest.mark.asyncio
async def test_rollout_behavior_prefers_the_provider_trace_fact_aggregate() -> None:
    class AggregateSource:
        async def aggregate_trace_facts(self, run_id: str, query):
            assert run_id == "train-run"
            assert query.group_by == ("rollout_step",)
            return TraceAggregateResult(
                state="available",
                buckets=(
                    TraceAggregateBucket(
                        dimensions={"rollout_step": 7},
                        trace_count=3,
                        values={
                            "mean_thinking_tokens": 11.0,
                            "mean_model_output_tokens": 21.0,
                            "mean_tool_calls": 1.0,
                        },
                        coverage={
                            "mean_thinking_tokens": 3,
                            "mean_model_output_tokens": 3,
                            "mean_tool_calls": 3,
                        },
                    ),
                ),
            )

        async def traces(self, run_id: str, query):
            raise AssertionError("payload scan must not run when trace facts are available")

    view = await rollout_behavior_view(AggregateSource(), "train-run", expected=3)  # type: ignore[arg-type]

    assert view.state == "complete"
    assert view.scanned == 3
    assert view.points[0].step == 7
    assert view.points[0].thinking_tokens == 11
    assert view.points[0].output_tokens == 21


@pytest.mark.asyncio
async def test_rollout_behavior_service_caches_the_fact_aggregate() -> None:
    class CountingSource(FixtureRunDataSource):
        aggregate_calls = 0

        async def aggregate_trace_facts(self, run_id: str, query):
            self.aggregate_calls += 1
            return await super().aggregate_trace_facts(run_id, query)

        async def traces(self, run_id: str, query):
            raise AssertionError("rollout behavior must not scan native payloads")

    source = CountingSource()
    service = ObservatoryService({"fixture": source})
    locator = RunLocator(source_id="fixture", run_id="runs/grpo-silver-pine")

    first = await service.get_rollout_behavior_view(locator)
    calls_after_first = source.aggregate_calls
    second = await service.get_rollout_behavior_view(locator)

    assert first == second
    assert calls_after_first > 0
    assert source.aggregate_calls == calls_after_first


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
    assert system.backend_runtime is not None
    assert system.backend_runtime.rollout_samples == 6
    assert system.backend_runtime.rollout_tokens_per_second_latest == 171
    assert system.backend_runtime.rollout_tokens_per_second_mean == pytest.approx(157.1666666667)
    assert system.backend_runtime.rollout_seconds_latest == 14
    assert system.backend_runtime.rollouts_per_prompt == 4
    assert system.backend_runtime.mtp_selected is False
