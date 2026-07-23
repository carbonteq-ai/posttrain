"""Provider-neutral Observatory query, projection, trace, and analysis service."""

from __future__ import annotations

import asyncio
import base64
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from statistics import fmean
from typing import Any, cast

from posttrain.common import JsonValue
from posttrain.tracking import EventRecord, MetricSeries, RunDataSource, RunQuery

from .execution_targets import execution_target_capacity, execution_target_contexts
from .models import (
    ChartView,
    ComparisonRow,
    EvaluationRunView,
    EvidenceCompleteness,
    EvidenceRequirement,
    GenericRunView,
    GRPOAccelerationEvidence,
    GRPOProjection,
    GRPORolloutPopulation,
    LocatedRunSummary,
    MetricCatalog,
    MetricNamespace,
    MetricSeriesQuery,
    MetricSeriesSet,
    RunAlert,
    RunComparison,
    RunDelta,
    RunLocator,
    RunView,
    RunViewResponse,
    SemanticSummaryRequest,
    SemanticSummaryResult,
    SeriesTip,
    SourceSummary,
    SummaryChange,
    SummaryValue,
    SystemMetricGroup,
    SystemMetricSummary,
    SystemMetricsView,
    TraceDetail,
    TraceEvaluationView,
    ViewMode,
    WorkPackageView,
)
from .redaction import RedactionPolicy
from .runtime_phases import project_runtime_phases
from .semantic import SemanticAnalysisService, SemanticSummaryProvider
from .sources import RunSourceRegistry
from .telemetry import (
    DEFAULT_TELEMETRY_DEFINITIONS,
    EvidenceCondition,
    HealthRuleDefinition,
    JobTelemetryDefinition,
)
from .traces import get_trace_detail as load_trace_detail
from .traces import trace_evaluation_view


def _reduce(series: MetricSeries, reducer: str) -> float | None:
    values = [point.value for point in series.points]
    if not values:
        return None
    reducers = {
        "last": lambda: values[-1],
        "min": lambda: min(values),
        "max": lambda: max(values),
        "mean": lambda: fmean(values),
        "sum": lambda: sum(values),
    }
    try:
        return reducers[reducer]()
    except KeyError as error:
        raise ValueError(f"unsupported telemetry reducer: {reducer}") from error


def _metric_summary(
    series: Mapping[str, MetricSeries],
    *,
    key: str,
    label: str,
    metric: str,
    reducer: str = "last",
    unit: str | None = None,
) -> SummaryValue:
    value = _reduce(series.get(metric, MetricSeries(name=metric)), reducer)
    return SummaryValue(
        key=key,
        label=label,
        metric=metric,
        state="available" if value is not None else "missing",
        value=value,
        unit=unit,
    )


def _grpo_projection(
    resolved_inputs: Mapping[str, JsonValue],
    series: Mapping[str, MetricSeries],
) -> GRPOProjection:
    return GRPOProjection(
        rollout_population=GRPORolloutPopulation(
            attempted=_metric_summary(
                series, key="attempted", label="Attempted", metric="train/rl/rollouts_attempted", reducer="sum"
            ),
            completed=_metric_summary(
                series, key="completed", label="Completed", metric="train/rl/rollouts_completed", reducer="sum"
            ),
            failed=_metric_summary(
                series, key="failed", label="Failed", metric="train/rl/rollouts_failed", reducer="sum"
            ),
            truncated=_metric_summary(
                series, key="truncated", label="Truncated", metric="train/rl/rollouts_truncated", reducer="sum"
            ),
            unscorable=_metric_summary(
                series, key="unscorable", label="Unscorable", metric="train/rl/rollouts_unscorable", reducer="sum"
            ),
        ),
        acceleration=GRPOAccelerationEvidence(
            mtp_selected=_condition_active("mtp_rollout_enabled", resolved_inputs, series),
            quantized_kv_cache_selected=_condition_active("quantized_kv_cache", resolved_inputs, series),
            speculative_acceptance=_metric_summary(
                series,
                key="speculative_acceptance",
                label="MTP acceptance",
                metric="serve/backend/speculative_acceptance_rate",
                unit="ratio",
            ),
            accepted_speculative_length=_metric_summary(
                series,
                key="accepted_speculative_length",
                label="Accepted length",
                metric="serve/backend/speculative_accepted_length",
                unit="tokens",
            ),
            kv_cache_peak_usage=_metric_summary(
                series,
                key="kv_cache_peak_usage",
                label="Peak KV-cache usage",
                metric="serve/backend/kv_cache_peak_usage_ratio",
                unit="ratio",
            ),
        ),
    )


def _threshold_fired(value: float, rule: HealthRuleDefinition) -> bool:
    if rule.threshold is None or rule.operator is None:
        raise ValueError(f"threshold rule {rule.id!r} is incomplete")
    return {
        "gt": value > rule.threshold,
        "gte": value >= rule.threshold,
        "lt": value < rule.threshold,
        "lte": value <= rule.threshold,
        "eq": value == rule.threshold,
    }[rule.operator]


def _cursor_payload(view: RunView) -> dict[str, JsonValue]:
    tips: dict[str, JsonValue] = {}
    for chart in view.charts:
        for series in chart.series:
            if series.points:
                point = series.points[-1]
                tips[series.name] = {"step": point.step, "value": point.value}
    return {
        "run_id": view.run.run_id,
        "status": view.run.status,
        "summary": {value.key: {"state": value.state, "value": value.value} for value in view.summary},
        "alerts": [alert.id for alert in view.alerts],
        "tips": tips,
    }


def _encode_cursor(payload: Mapping[str, JsonValue]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: str | None, run_id: str) -> dict[str, Any]:
    if cursor is None:
        return {"run_id": run_id, "status": None, "summary": {}, "alerts": [], "tips": {}}
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        value = json.loads(base64.urlsafe_b64decode(padded).decode())
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("run delta cursor is invalid") from error
    if not isinstance(value, dict) or value.get("run_id") != run_id:
        raise ValueError("run delta cursor does not belong to this run")
    for key, expected in (("summary", dict), ("alerts", list), ("tips", dict)):
        if not isinstance(value.get(key), expected):
            raise ValueError("run delta cursor payload is invalid")
    return value


def _downsample(series: MetricSeries, maximum: int) -> tuple[MetricSeries, bool]:
    points = series.points
    if len(points) <= maximum:
        return series, False
    bucket_count = max(1, maximum // 2)
    selected = []
    for index in range(bucket_count):
        start = index * len(points) // bucket_count
        end = max(start + 1, (index + 1) * len(points) // bucket_count)
        bucket = points[start:end]
        low = min(bucket, key=lambda point: point.value)
        high = max(bucket, key=lambda point: point.value)
        candidates = [low] if low == high else [low, high]
        selected.extend(sorted(candidates, key=lambda point: point.step if point.step is not None else -1))
    return MetricSeries(name=series.name, points=tuple(selected[:maximum])), True


def _config_values(value: JsonValue, key: str) -> tuple[JsonValue, ...]:
    values: list[JsonValue] = []
    if isinstance(value, dict):
        for item_key, item_value in value.items():
            if item_key == key:
                values.append(item_value)
            values.extend(_config_values(item_value, key))
    elif isinstance(value, list):
        for item in value:
            values.extend(_config_values(item, key))
    return tuple(values)


def _condition_active(
    condition: EvidenceCondition,
    resolved_inputs: Mapping[str, JsonValue],
    series: Mapping[str, MetricSeries],
) -> bool:
    if condition == "validation_configured":
        return bool(_config_values(dict(resolved_inputs), "validation_dataset")) or bool(
            _config_values(dict(resolved_inputs), "validation_dataset_id")
        )
    if condition == "gradient_clipping_enabled":
        configured = _config_values(dict(resolved_inputs), "max_grad_norm")
        return any(isinstance(value, int | float) and value > 0 for value in configured) or bool(
            series.get("train/gradient_clipped", MetricSeries(name="train/gradient_clipped")).points
        )
    if condition == "source_scores_available":
        coverage = series.get(
            "train/data/preference_score_coverage",
            MetricSeries(name="train/data/preference_score_coverage"),
        )
        return bool(coverage.points and coverage.points[-1].value > 0)
    if condition == "distributed":
        return any(
            isinstance(value, int | float) and value > 1
            for value in _config_values(dict(resolved_inputs), "world_size")
        )
    if condition == "quantized_update":
        kinds = {str(value).lower() for value in _config_values(dict(resolved_inputs), "parameter_update_kind")}
        return bool(kinds & {"qlora", "quantization-aware"})
    if condition == "packing_enabled":
        return any(value is True for value in _config_values(dict(resolved_inputs), "packing"))
    if condition == "reference_kl_enabled":
        return any(
            isinstance(value, int | float) and not isinstance(value, bool) and value > 0
            for value in _config_values(dict(resolved_inputs), "beta")
        )
    if condition == "decoupled_rollout":
        return any(
            isinstance(value, str) and value.split("@", 1)[0].lower() == "vllm"
            for value in _config_values(dict(resolved_inputs), "backend")
        )
    if condition == "asynchronous_rollout":
        return any(
            isinstance(value, str) and value.lower() == "async"
            for value in _config_values(dict(resolved_inputs), "mode")
        )
    if condition == "mtp_rollout_enabled":
        speculative = _config_values(dict(resolved_inputs), "speculative_config")
        selected = any(
            isinstance(value, Mapping) and str(value.get("method", "")).lower() == "mtp" for value in speculative
        )
        methods = _config_values(dict(resolved_inputs), "speculative_method")
        return selected or any(isinstance(value, str) and value.lower() == "mtp" for value in methods)
    if condition == "quantized_kv_cache":
        return any(
            isinstance(value, str) and value.lower().startswith("turboquant_")
            for value in _config_values(dict(resolved_inputs), "kv_cache_dtype")
        )
    if condition == "tool_environment":
        return any(bool(value) for value in _config_values(dict(resolved_inputs), "tools")) or any(
            value is True for value in _config_values(dict(resolved_inputs), "tool_environment")
        )
    raise ValueError(f"unknown evidence condition: {condition}")


def _evidence_completeness(
    definition: JobTelemetryDefinition,
    resolved_inputs: Mapping[str, JsonValue],
    series: Mapping[str, MetricSeries],
    *,
    trace_count: int = 0,
) -> EvidenceCompleteness:
    requirements: list[EvidenceRequirement] = []
    if definition.evidence_requirements:
        for requirement in definition.evidence_requirements:
            active = requirement.level != "conditional" or _condition_active(
                cast(EvidenceCondition, requirement.condition), resolved_inputs, series
            )
            missing = tuple(
                metric for metric in requirement.metrics if not series.get(metric, MetricSeries(name=metric)).points
            )
            if not active:
                state = "not_applicable"
                condition_label = (
                    requirement.condition.replace("_", " ") if requirement.condition is not None else "not selected"
                )
                reason = f"Condition {condition_label} is not active."
                missing = ()
            elif missing:
                state = "missing"
                reason = f"Missing {', '.join(missing)}. {requirement.reason}"
            else:
                state = "available"
                reason = requirement.reason
            requirements.append(
                EvidenceRequirement(
                    key=requirement.key,
                    label=requirement.label,
                    level=requirement.level,
                    state=state,
                    metrics=requirement.metrics,
                    missing_metrics=missing,
                    reason=reason,
                )
            )
    else:
        for field in definition.summary_fields:
            if not field.required:
                continue
            available = bool(series.get(field.metric, MetricSeries(name=field.metric)).points)
            requirements.append(
                EvidenceRequirement(
                    key=field.key,
                    label=field.label,
                    level="required",
                    state="available" if available else "missing",
                    metrics=(field.metric,),
                    missing_metrics=() if available else (field.metric,),
                    reason="Required summary telemetry for this job view.",
                )
            )
    required = [item for item in requirements if item.level == "required"]
    conditional = [item for item in requirements if item.level == "conditional" and item.state != "not_applicable"]
    missing_required = any(item.state == "missing" for item in required)
    missing_conditional = any(item.state == "missing" for item in conditional)
    state = "insufficient" if missing_required else "partial" if missing_conditional else "complete"
    validation = next((item for item in requirements if item.key == "held_out_preferences"), None)
    if definition.job_kind == "train.grpo":
        research_ready = state == "complete" and trace_count > 0
    else:
        research_ready = state == "complete" and validation is not None and validation.state == "available"
    return EvidenceCompleteness(
        state=state,
        research_ready=research_ready,
        required_available=sum(item.state == "available" for item in required),
        required_total=len(required),
        conditional_available=sum(item.state == "available" for item in conditional),
        conditional_active=len(conditional),
        requirements=tuple(requirements),
    )


class ObservatoryService:
    """Single application service consumed by Python, HTTP, MCP, CLI, and UI."""

    def __init__(
        self,
        source: RunDataSource | Mapping[str, RunDataSource],
        definitions: Mapping[str, JobTelemetryDefinition] = DEFAULT_TELEMETRY_DEFINITIONS,
        *,
        semantic_provider: SemanticSummaryProvider | None = None,
        redaction: RedactionPolicy | None = None,
    ) -> None:
        sources = {"default": source} if not isinstance(source, Mapping) else dict(source)
        self.registry = RunSourceRegistry(cast(Mapping[str, RunDataSource], sources))
        self._definitions = dict(definitions)
        self._redaction = redaction or RedactionPolicy()
        self._semantic = SemanticAnalysisService(semantic_provider)

    def _locator(self, value: str | RunLocator) -> RunLocator:
        return (
            value if isinstance(value, RunLocator) else RunLocator(source_id=self.registry.source_ids[0], run_id=value)
        )

    def get_job_telemetry_schema(self, job_kind: str) -> JobTelemetryDefinition:
        try:
            return self._definitions[job_kind]
        except KeyError as error:
            available = ", ".join(sorted(self._definitions)) or "none"
            raise KeyError(f"job telemetry is not defined for {job_kind!r}; available: {available}") from error

    def list_job_kinds(self) -> tuple[JobTelemetryDefinition, ...]:
        return tuple(self._definitions[key] for key in sorted(self._definitions))

    async def list_sources(self) -> tuple[SourceSummary, ...]:
        return await self.registry.sources()

    async def list_runs(self, query: RunQuery | None = None) -> tuple[LocatedRunSummary, ...]:
        return await self.registry.list_runs(query or RunQuery())

    async def get_run_view(
        self,
        run: str | RunLocator,
        mode: ViewMode = "auto",
        metrics: tuple[str, ...] = (),
    ) -> RunView:
        """Compatibility Python view for registered metric schemas.

        New consumers should use ``get_run_view_response`` to receive strict
        job-evaluation and generic variants.
        """
        locator = self._locator(run)
        detail = await self.registry.resolve(locator).get_run(locator.run_id)
        definition = self.get_job_telemetry_schema(detail.summary.job_kind)
        if mode == "generic":
            raise ValueError("use get_run_view_response for generic mode")
        return await self._metric_job_view(locator, definition)

    async def get_run_view_response(
        self,
        run: str | RunLocator,
        mode: ViewMode = "auto",
        metrics: tuple[str, ...] = (),
    ) -> RunViewResponse:
        locator = self._locator(run)
        source = self.registry.resolve(locator)
        detail = await source.get_run(locator.run_id)
        definition = self._definitions.get(detail.summary.job_kind)
        if mode == "job" and definition is None:
            raise LookupError(f"job view is unavailable for {detail.summary.job_kind!r}")
        if mode == "generic" or definition is None:
            generic = await self._generic_view(locator, metrics)
            reason = None if mode == "generic" else f"No job view is registered for {detail.summary.job_kind}."
            return RunViewResponse(
                requested_mode=mode,
                resolved_mode="generic",
                fallback_reason=reason,
                view=generic,
            )
        metric_view = await self._metric_job_view(locator, definition)
        if detail.summary.job_kind.startswith("eval."):
            evaluation = await trace_evaluation_view(source, locator.run_id, expected=detail.trace_count or None)
            view: RunView | EvaluationRunView = EvaluationRunView(
                schema_version=metric_view.schema_version,
                locator=locator,
                run=metric_view.run,
                summary=metric_view.summary,
                charts=metric_view.charts,
                metric_help=metric_view.metric_help,
                completeness=metric_view.completeness,
                alerts=metric_view.alerts,
                evaluation=evaluation,
                artifacts=metric_view.artifacts,
                execution_targets=metric_view.execution_targets,
                resolved_inputs=metric_view.resolved_inputs,
                source_metadata=metric_view.source_metadata,
                trace_evaluation_enabled=metric_view.trace_evaluation_enabled,
                capabilities=metric_view.capabilities,
            )
        else:
            view = metric_view
        return RunViewResponse(requested_mode=mode, resolved_mode="job", view=view)

    async def _metric_job_view(self, locator: RunLocator, definition: JobTelemetryDefinition) -> RunView:
        source = self.registry.resolve(locator)
        detail = await source.get_run(locator.run_id)
        names = tuple(sorted(definition.metric_names))
        series_values, artifacts = await asyncio.gather(
            source.metric_series(locator.run_id, names), source.artifacts(locator.run_id)
        )
        by_name = {series.name: series for series in series_values}
        summary = tuple(
            SummaryValue(
                key=field.key,
                label=field.label,
                metric=field.metric,
                state=(
                    "available"
                    if (value := _reduce(by_name.get(field.metric, MetricSeries(name=field.metric)), field.reducer))
                    is not None
                    else "missing"
                ),
                value=value,
                unit=field.unit,
            )
            for field in definition.summary_fields
        )
        charts = tuple(
            ChartView(
                key=chart.key,
                title=chart.title,
                question=chart.question,
                series=tuple(by_name.get(name, MetricSeries(name=name)) for name in chart.metrics),
            )
            for chart in definition.charts
            if any(by_name.get(name, MetricSeries(name=name)).points for name in chart.metrics)
        )
        completeness = _evidence_completeness(
            definition,
            detail.resolved_inputs,
            by_name,
            trace_count=detail.trace_count,
        )
        execution_targets = execution_target_contexts(detail.resolved_inputs)
        return RunView(
            schema_version=definition.schema_version,
            locator=locator,
            run=detail.summary,
            summary=summary,
            charts=charts,
            metric_help=definition.metric_help,
            completeness=completeness,
            grpo=(_grpo_projection(detail.resolved_inputs, by_name) if definition.job_kind == "train.grpo" else None),
            alerts=self._alerts(
                detail.summary.status,
                definition,
                summary,
                by_name,
                completeness,
                trace_count=detail.trace_count,
            ),
            artifacts=artifacts,
            execution_targets=execution_targets,
            resolved_inputs=self._redaction.mapping(detail.resolved_inputs),
            source_metadata=self._redaction.mapping(detail.source_metadata),
            trace_count=detail.trace_count,
            trace_evaluation_enabled=bool(definition.trace_sections),
            capabilities=source.capabilities,
        )

    async def _generic_view(self, locator: RunLocator, metrics: tuple[str, ...]) -> GenericRunView:
        source = self.registry.resolve(locator)
        detail, artifacts = await asyncio.gather(source.get_run(locator.run_id), source.artifacts(locator.run_id))
        catalog = self._catalog(detail.metric_names)
        selected = None
        if metrics:
            selected = await self.get_metric_series(locator, MetricSeriesQuery(names=metrics))
        events = tuple(
            EventRecord(
                name=event.name,
                occurred_at=event.occurred_at,
                attributes=self._redaction.mapping(event.attributes),
            )
            for event in detail.events
        )
        return GenericRunView(
            locator=locator,
            run=detail.summary,
            metric_catalog=catalog,
            selected_series=selected,
            events=events,
            artifacts=artifacts,
            execution_targets=execution_target_contexts(detail.resolved_inputs),
            resolved_inputs=self._redaction.mapping(detail.resolved_inputs),
            source_metadata=self._redaction.mapping(detail.source_metadata),
            trace_count=detail.trace_count,
            trace_evaluation_enabled=detail.trace_count > 0,
            capabilities=source.capabilities,
        )

    @staticmethod
    def _catalog(names: tuple[str, ...]) -> MetricCatalog:
        grouped: dict[str, list[str]] = defaultdict(list)
        for name in sorted(set(names)):
            grouped[name.split("/", 1)[0]].append(name)
        return MetricCatalog(
            namespaces=tuple(
                MetricNamespace(name=namespace, metrics=tuple(values)) for namespace, values in sorted(grouped.items())
            ),
            total=sum(len(values) for values in grouped.values()),
        )

    async def list_run_metrics(self, run: str | RunLocator) -> MetricCatalog:
        locator = self._locator(run)
        detail = await self.registry.resolve(locator).get_run(locator.run_id)
        return self._catalog(detail.metric_names)

    async def get_system_metrics(self, run: str | RunLocator) -> SystemMetricsView:
        """Return the same canonical runtime evidence for every job kind."""

        locator = self._locator(run)
        source = self.registry.resolve(locator)
        detail = await source.get_run(locator.run_id)
        cards = (
            (
                "gpu_utilization",
                "GPU utilization",
                "system/gpu_utilization",
                "%",
                "Share of the sampling interval during which the accelerator was executing work.",
                "Sustained high utilization usually means the accelerator is being kept busy; low values should be correlated with data loading, communication, and step-time evidence.",
                "A device average can hide imbalance between ranks.",
            ),
            (
                "gpu_memory",
                "GPU memory",
                "system/gpu_vram_used_bytes",
                "bytes",
                "Accelerator memory currently occupied by the training process.",
                "Approaching device capacity increases out-of-memory risk and may constrain batch or sequence length.",
                "Allocated, reserved, and device-wide memory are different measures; this card shows the recorded provider series.",
            ),
            (
                "cpu_utilization",
                "CPU utilization",
                "system/cpu_percent",
                "%",
                "CPU capacity used by the training process or host during the sampling interval.",
                "High CPU use paired with low GPU use can indicate input preparation or orchestration bottlenecks.",
                "Provider definitions may report one-core or whole-host percentages.",
            ),
            (
                "process_memory",
                "Process memory",
                "system/process_rss_bytes",
                "bytes",
                "Resident host memory occupied by the training process.",
                "A steadily rising value can indicate caching pressure or a memory leak.",
                None,
            ),
            (
                "wall_time",
                "Wall time",
                "system/wall_time_s",
                "s",
                "Elapsed wall-clock time since the run started.",
                "Use it to align runtime events and estimate time-to-completion from comparable runs.",
                None,
            ),
            (
                "traces_dropped",
                "Dropped traces",
                "tracking/traces_dropped",
                "traces",
                "Trace records the observer could not persist or synchronize.",
                "Zero is expected; any non-zero value means trace-backed evidence is incomplete.",
                "Metric series and artifacts may still be complete even when traces were dropped.",
            ),
        )
        requested = tuple(
            sorted(
                {
                    *(name for name in detail.metric_names if name.startswith(("system/", "tracking/"))),
                    *(metric for _, _, metric, *_ in cards),
                }
            )
        )
        series_values = await source.metric_series(locator.run_id, requested)
        by_name = {series.name: series for series in series_values}
        sample_count = max(
            (len(series.points) for name, series in by_name.items() if name.startswith("system/")),
            default=0,
        )
        summary = tuple(
            SystemMetricSummary(
                key=key,
                label=label,
                metric=metric,
                value=(values.points[-1].value if (values := by_name.get(metric)) and values.points else None),
                unit=unit,
                state="available" if metric in by_name and by_name[metric].points else "missing",
                description=description,
                interpretation=interpretation,
                caveat=caveat,
            )
            for key, label, metric, unit, description, interpretation, caveat in cards
        )
        groups = []
        for key, title, names in (
            (
                "compute",
                "Compute utilization",
                ("system/gpu_utilization", "system/cpu_percent"),
            ),
            (
                "memory",
                "Memory pressure",
                ("system/gpu_vram_used_bytes", "system/process_rss_bytes"),
            ),
            (
                "runtime",
                "Runtime and observer health",
                ("system/wall_time_s", "tracking/traces_written", "tracking/traces_dropped"),
            ),
        ):
            group_series = tuple(by_name[name] for name in names if name in by_name)
            if group_series:
                groups.append(SystemMetricGroup(key=key, title=title, series=group_series))
        execution_targets = execution_target_contexts(detail.resolved_inputs)
        capacity_state, capacity_bytes = execution_target_capacity(execution_targets)
        phase_projection = project_runtime_phases(detail, by_name)
        return SystemMetricsView(
            locator=locator,
            state="available" if sample_count else "unavailable",
            window_started_at=detail.summary.started_at,
            window_finished_at=detail.summary.finished_at,
            sample_count=sample_count,
            summary=summary,
            groups=tuple(groups),
            missing=tuple(item.metric for item in summary if item.state == "missing"),
            phase_state=phase_projection.state,
            phase_intervals=phase_projection.intervals,
            phase_segments=phase_projection.segments,
            phase_summary=phase_projection.summary,
            phase_issues=phase_projection.issues,
            unclassified_sample_count=phase_projection.unclassified_sample_count,
            execution_targets=execution_targets,
            vram_capacity_state=capacity_state,
            vram_capacity_bytes=capacity_bytes,
            vram_observed_peak_bytes=phase_projection.vram_observed_peak_bytes,
            capabilities=source.capabilities,
        )

    async def get_metric_series(self, run: str | RunLocator, query: MetricSeriesQuery) -> MetricSeriesSet:
        locator = self._locator(run)
        source = self.registry.resolve(locator)
        detail = await source.get_run(locator.run_id)
        unknown = set(query.names) - set(detail.metric_names)
        if unknown:
            raise ValueError(f"unknown metric names: {', '.join(sorted(unknown))}")
        raw = await source.metric_series(locator.run_id, query.names)
        filtered = []
        requested = 0
        changed = False
        for series in raw:
            points = tuple(
                point
                for point in series.points
                if (query.start_step is None or point.step is None or point.step >= query.start_step)
                and (query.end_step is None or point.step is None or point.step <= query.end_step)
            )
            requested += len(points)
            value, downsampled = _downsample(MetricSeries(name=series.name, points=points), query.max_points)
            filtered.append(value)
            changed = changed or downsampled
        return MetricSeriesSet(
            series=tuple(filtered),
            downsampled=changed,
            requested_points=requested,
            returned_points=sum(len(series.points) for series in filtered),
        )

    async def get_run_alerts(self, run: str | RunLocator) -> tuple[RunAlert, ...]:
        view = (await self.get_run_view_response(run)).view
        return view.alerts if not isinstance(view, GenericRunView) else ()

    async def get_run_delta(self, run: str | RunLocator, cursor: str | None = None) -> RunDelta:
        locator = self._locator(run)
        view = (await self.get_run_view_response(locator)).view
        if not isinstance(view, RunView):
            return RunDelta(
                run_id=locator.run_id,
                status=view.run.status,
                changed_summary=(),
                alerts_added=(),
                alerts_removed=(),
                series_tips=(),
                cursor=_encode_cursor({"run_id": locator.run_id, "summary": {}, "alerts": [], "tips": {}}),
            )
        definition = self.get_job_telemetry_schema(view.run.job_kind)
        previous = _decode_cursor(cursor, locator.run_id)
        current = _cursor_payload(view)
        changes = []
        for value in view.summary:
            old = previous["summary"].get(value.key)
            if old != {"state": value.state, "value": value.value}:
                changes.append(
                    SummaryChange(
                        key=value.key,
                        previous=old.get("value") if isinstance(old, dict) else None,
                        current=value.value,
                        state=value.state,
                    )
                )
        previous_alerts = {str(value) for value in previous["alerts"]}
        current_alerts = {alert.id: alert for alert in view.alerts}
        tips: dict[str, SeriesTip] = {}
        for chart in view.charts:
            for series in chart.series:
                if series.name not in definition.delta_tip_metrics or not series.points:
                    continue
                point = series.points[-1]
                if previous["tips"].get(series.name) != {"step": point.step, "value": point.value}:
                    tips[series.name] = SeriesTip(metric=series.name, step=point.step, value=point.value)
        return RunDelta(
            run_id=locator.run_id,
            status=view.run.status,
            changed_summary=tuple(changes),
            alerts_added=tuple(current_alerts[key] for key in sorted(set(current_alerts) - previous_alerts)),
            alerts_removed=tuple(sorted(previous_alerts - set(current_alerts))),
            series_tips=tuple(tips[name] for name in definition.delta_tip_metrics if name in tips)[:5],
            cursor=_encode_cursor(current),
        )

    async def compare_runs(self, runs: tuple[str | RunLocator, ...]) -> RunComparison:
        if not runs:
            return RunComparison(state="incomparable", columns=(), rows=(), reason="no runs were supplied")
        locators = tuple(self._locator(run) for run in runs)
        responses = await asyncio.gather(*(self.get_run_view_response(locator, mode="job") for locator in locators))
        views = tuple(response.view for response in responses)
        if any(not isinstance(view, RunView | EvaluationRunView) for view in views):
            return RunComparison(state="incomparable", columns=(), rows=(), reason="a job view is unavailable")
        job_views = cast(tuple[RunView | EvaluationRunView, ...], views)
        job_kinds = {view.run.job_kind for view in job_views}
        versions = {view.schema_version for view in job_views}
        if len(job_kinds) != 1 or len(versions) != 1:
            return RunComparison(
                state="incomparable",
                columns=(),
                rows=(),
                reason="runs use different job kinds or view schema versions",
            )
        job_kind = next(iter(job_kinds))
        definition = self.get_job_telemetry_schema(job_kind)
        rows = []
        for locator, view in zip(locators, job_views, strict=True):
            values = {value.key: value for value in view.summary}
            rows.append(
                ComparisonRow(
                    locator=locator,
                    run_id=view.run.run_id,
                    values={key: values[key].value for key in definition.comparison_keys},
                    states={key: values[key].state for key in definition.comparison_keys},
                )
            )
        return RunComparison(
            job_kind=job_kind,
            state="comparable",
            columns=definition.comparison_keys,
            rows=tuple(rows),
        )

    async def get_trace_evaluation_view(self, run: str | RunLocator) -> TraceEvaluationView:
        locator = self._locator(run)
        source = self.registry.resolve(locator)
        detail = await source.get_run(locator.run_id)
        return await trace_evaluation_view(source, locator.run_id, expected=detail.trace_count or None)

    async def get_trace_detail(self, run: str | RunLocator, trace_id: str) -> TraceDetail:
        locator = self._locator(run)
        return await load_trace_detail(self.registry.resolve(locator), locator.run_id, trace_id, self._redaction)

    async def get_work_package_view(
        self,
        work_package_id: str,
        *,
        project_id: str | None = None,
        source_id: str | None = None,
    ) -> WorkPackageView:
        return await self.registry.work_package_view(
            work_package_id,
            project_id=project_id,
            source_id=source_id,
        )

    async def summarize_run(self, run: str | RunLocator, request: SemanticSummaryRequest) -> SemanticSummaryResult:
        response = await self.get_run_view_response(
            run,
            mode="generic" if request.scope == "metrics" else "auto",
            metrics=request.metric_names,
        )
        return await self._semantic.summarize(response, request)

    @staticmethod
    def _alerts(
        status: str,
        definition: JobTelemetryDefinition,
        summary: tuple[SummaryValue, ...],
        series: Mapping[str, MetricSeries],
        completeness: EvidenceCompleteness,
        *,
        trace_count: int = 0,
    ) -> tuple[RunAlert, ...]:
        alerts: list[RunAlert] = []
        if status == "failed":
            alerts.append(RunAlert(id="run-failed", severity="error", message="The run failed."))
        elif status in {"partial", "cancelled", "unsupported"}:
            alerts.append(
                RunAlert(id=f"run-{status}", severity="warning", message=f"The run finished with status {status}.")
            )
        if definition.job_kind == "train.grpo" and trace_count == 0:
            alerts.append(
                RunAlert(
                    id="missing-grpo-traces",
                    severity="error",
                    message="GRPO rollout traces are missing, so reward aggregates cannot be audited.",
                    field="traces",
                )
            )
        by_key = {value.key: value for value in summary}
        for field in definition.summary_fields:
            if field.required and by_key[field.key].state == "missing":
                alerts.append(
                    RunAlert(
                        id=f"missing-{field.key}",
                        severity="warning",
                        message=f"Required telemetry is missing: {field.label}.",
                        field=field.key,
                    )
                )
        for requirement in completeness.requirements:
            if requirement.state != "missing" or requirement.level == "diagnostic":
                continue
            alerts.append(
                RunAlert(
                    id=f"evidence-{requirement.key}",
                    severity="error" if requirement.level == "required" else "warning",
                    message=f"{requirement.label} evidence is incomplete.",
                    field=requirement.key,
                )
            )
        for rule in definition.health_rules:
            values = [point.value for point in series.get(rule.metric, MetricSeries(name=rule.metric)).points]
            fired = (
                any(not math.isfinite(value) for value in values)
                if rule.kind == "non_finite"
                else bool(values and _threshold_fired(values[-1], rule))
            )
            if fired:
                alerts.append(RunAlert(id=rule.id, severity=rule.severity, message=rule.message, field=rule.metric))
        return tuple(sorted(alerts, key=lambda alert: alert.id))


RunViewService = ObservatoryService

__all__ = ["ObservatoryService", "RunViewService"]
