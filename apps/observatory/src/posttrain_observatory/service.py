"""Provider-neutral Observatory query, projection, trace, and analysis service."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Literal, cast

from posttrain.common import JsonValue
from posttrain.tracking import (
    EventRecord,
    MetricPoint,
    MetricSeries,
    RunDataSource,
    RunDetail,
    RunQuery,
    TraceQuery,
)

from .discovery import TrackioSourceDiscovery
from .evaluation_contracts import read_evaluation_contract
from .execution_targets import execution_target_capacity, execution_target_contexts
from .models import (
    BackendRuntimeSummary,
    ChartView,
    ComparisonRow,
    EvaluationBreakdownSpec,
    EvaluationFacetSpec,
    EvaluationMetadata,
    EvaluationMetricDefinition,
    EvaluationRunView,
    EvaluationSuccessDefinition,
    EvidenceCompleteness,
    EvidenceRequirement,
    GenericRunView,
    GRPOAccelerationEvidence,
    GRPOProjection,
    GRPORolloutPopulation,
    InferenceTimingStageSummary,
    InferenceTimingSummary,
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
    ServingBenchmarkRunView,
    ServingCapacityRunRow,
    ServingCapacityWorkPackageView,
    ServingContenderView,
    ServingParetoPoint,
    SourceRefreshStatus,
    SourceSummary,
    SummaryChange,
    SummaryValue,
    SystemMetricGroup,
    SystemMetricSummary,
    SystemMetricsView,
    TraceDetail,
    TraceEvaluationView,
    TraceSummaryPage,
    ViewMode,
    WorkPackageView,
)
from .redaction import RedactionPolicy
from .runtime_phases import project_runtime_phases
from .semantic import SemanticAnalysisService, SemanticSummaryProvider
from .serving_capacity import project_serving_benchmark
from .sources import RunSourceRegistry
from .telemetry import (
    DEFAULT_TELEMETRY_DEFINITIONS,
    EvidenceCondition,
    HealthRuleDefinition,
    JobTelemetryDefinition,
)
from .traces import get_trace_detail as load_trace_detail
from .traces import trace_evaluation_view, trace_summary_page


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


def _selection_identity(value: object) -> tuple[str | None, str | None]:
    if not isinstance(value, Mapping):
        return None, None
    selection_id = value.get("selection_id")
    revision = value.get("revision")
    return (
        selection_id if isinstance(selection_id, str) else None,
        revision if isinstance(revision, str) else None,
    )


def _evaluation_population_identity(view: EvaluationRunView | RunView) -> dict[str, str]:
    """Return the immutable population identity required for evaluation comparison."""

    inputs = view.resolved_inputs
    contract = read_evaluation_contract(inputs)
    environment = inputs.get("environment")
    resolved = environment.get("resolved") if isinstance(environment, Mapping) else None
    activation = resolved.get("activation") if isinstance(resolved, Mapping) else None
    config = activation.get("config") if isinstance(activation, Mapping) else None
    taskset = config.get("taskset") if isinstance(config, Mapping) else None
    if contract.state == "versioned":
        contract_taskset = contract.population.get("taskset")
        if isinstance(contract_taskset, Mapping):
            taskset = contract_taskset
        manifest = contract.signal_manifest
        reward_components = manifest.get("reward_components")
        observation = manifest.get("observation")
        aggregation = contract.plan.get("aggregation")
        comparison = contract.plan.get("comparison")
        success = contract.plan.get("success")
    else:
        reward_components = resolved.get("reward_components") if isinstance(resolved, Mapping) else None
        observation = resolved.get("observation") if isinstance(resolved, Mapping) else None
        aggregation = {}
        comparison = {}
        success = {}
    taskset_json = json.dumps(taskset or {}, sort_keys=True, separators=(",", ":"))
    reward_json = json.dumps(reward_components or [], sort_keys=True, separators=(",", ":"))
    observation_json = json.dumps(observation or {}, sort_keys=True, separators=(",", ":"))
    source_revision = (
        contract.environment.get("source_revision")
        if contract.state == "versioned"
        else (resolved.get("source_revision") if isinstance(resolved, Mapping) else None)
    )
    package = (
        contract.environment.get("package")
        if contract.state == "versioned"
        else (resolved.get("package") if isinstance(resolved, Mapping) else None)
    )
    return {
        "job_kind": view.run.job_kind,
        # The plan and environment selection IDs are orchestration wrappers and
        # may differ between model qualification packages. Their immutable
        # taskset, source revision, and native metric schema are the population.
        "environment_source_revision": source_revision if isinstance(source_revision, str) else "missing",
        "environment_package": package if isinstance(package, str) else "missing",
        "taskset": taskset_json,
        "reward_components": reward_json,
        # Score and facet semantics are part of the logical evaluation
        # population.  Runs cannot be compared when they interpret the same
        # raw trace values through different environment declarations.
        "observation": observation_json,
        "aggregation": json.dumps(aggregation or {}, sort_keys=True, separators=(",", ":")),
        "comparison": json.dumps(comparison or {}, sort_keys=True, separators=(",", ":")),
        "success": json.dumps(success or {}, sort_keys=True, separators=(",", ":")),
        "evaluation_contract": json.dumps(
            {
                "id": contract.contract_id,
                "version": contract.contract_version,
                "state": contract.state,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def _evaluation_population_key(view: EvaluationRunView | RunView) -> str:
    identity = json.dumps(_evaluation_population_identity(view), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(identity.encode()).hexdigest()


def _humanize_name(value: str) -> str:
    label = " ".join(part.capitalize() for part in value.replace("_", " ").replace("-", " ").split())
    for source, target in (("Ifeval", "IFEval"), ("Gsm8k", "GSM8K"), ("Mmlu", "MMLU")):
        label = label.replace(source, target)
    return label


def _string_value(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _evaluation_success(value: object) -> EvaluationSuccessDefinition | None:
    if not isinstance(value, Mapping):
        return None
    source = value.get("source")
    predicate = value.get("predicate")
    if not isinstance(source, Mapping) or not isinstance(predicate, Mapping):
        return None
    success_id = _string_value(value.get("id"))
    label = _string_value(value.get("label"))
    namespace = _string_value(source.get("namespace"))
    signal = _string_value(source.get("name"))
    operator = _string_value(predicate.get("operator"))
    threshold = predicate.get("value")
    upper = predicate.get("upper")
    tolerance = predicate.get("tolerance", 0.0)
    missing = _string_value(value.get("missing")) or "error"
    if (
        success_id is None
        or label is None
        or namespace not in {"reward", "metric"}
        or signal is None
        or operator not in {"eq", "gt", "gte", "lt", "lte", "between"}
        or not isinstance(threshold, int | float)
        or isinstance(threshold, bool)
        or (upper is not None and (not isinstance(upper, int | float) or isinstance(upper, bool)))
        or not isinstance(tolerance, int | float)
        or isinstance(tolerance, bool)
        or missing not in {"error", "exclude"}
    ):
        return None
    return EvaluationSuccessDefinition(
        id=success_id,
        label=label,
        namespace=cast(Literal["reward", "metric"], namespace),
        signal=signal,
        operator=cast(Literal["eq", "gt", "gte", "lt", "lte", "between"], operator),
        value=float(threshold),
        upper=float(upper) if upper is not None else None,
        tolerance=float(tolerance),
        missing=cast(Literal["error", "exclude"], missing),
    )


def _evaluation_metadata(inputs: Mapping[str, JsonValue]) -> EvaluationMetadata | None:
    contract = read_evaluation_contract(inputs)
    environment = inputs.get("environment")
    resolved = environment.get("resolved") if isinstance(environment, Mapping) else None
    if not isinstance(resolved, Mapping):
        return None
    taskset = resolved.get("activation")
    config = taskset.get("config") if isinstance(taskset, Mapping) else None
    taskset = config.get("taskset") if isinstance(config, Mapping) else None
    taskset = taskset if isinstance(taskset, Mapping) else {}
    contract_environment = contract.environment
    contract_population = contract.population
    contract_taskset = contract_population.get("taskset")
    contract_dataset = contract_population.get("dataset")
    if contract.state == "versioned" and isinstance(contract_taskset, Mapping):
        taskset = contract_taskset
    package = _string_value(contract_environment.get("package")) or _string_value(resolved.get("package"))
    category = _string_value(contract_environment.get("category")) or _string_value(resolved.get("category"))
    dataset_identity = contract_dataset if isinstance(contract_dataset, Mapping) else {}
    dataset = (
        _string_value(dataset_identity.get("id"))
        or _string_value(taskset.get("dataset_repo"))
        or _string_value(taskset.get("repository"))
    )
    dataset_revision = (
        _string_value(dataset_identity.get("revision"))
        or _string_value(taskset.get("dataset_revision"))
        or _string_value(taskset.get("revision"))
    )
    split = _string_value(dataset_identity.get("split")) or _string_value(taskset.get("split"))
    source_revision = _string_value(contract_environment.get("source_revision")) or _string_value(
        resolved.get("source_revision")
    )
    if contract.state == "versioned":
        manifest = contract.signal_manifest
        reward_components = manifest.get("reward_components")
        observation = manifest.get("observation")
    elif contract.state == "legacy":
        reward_components = resolved.get("reward_components")
        observation = resolved.get("observation")
    else:
        reward_components = None
        observation = None
    rewards = (
        tuple(value for value in reward_components if isinstance(value, str))
        if isinstance(reward_components, list)
        else ()
    )
    observation = observation if isinstance(observation, Mapping) else {}
    configured_primary = _string_value(observation.get("primary_metric"))
    configured_primary_label = _string_value(observation.get("primary_metric_label"))
    configured_pass_rate = _string_value(observation.get("pass_rate_metric"))
    success_definition = (
        _evaluation_success(contract.plan.get("success"))
        if contract.contract_version is not None and contract.contract_version >= 2
        else None
    )
    raw_facets = observation.get("facets")
    facet_specs = (
        tuple(
            EvaluationFacetSpec(
                field=field,
                dimension=dimension,
                label=label,
                transform=cast(
                    Literal["identity", "prefix_before_colon"],
                    transform if transform in {"identity", "prefix_before_colon"} else "identity",
                ),
            )
            for item in raw_facets
            if isinstance(item, Mapping)
            if (field := _string_value(item.get("field"))) is not None
            if (dimension := _string_value(item.get("dimension"))) is not None
            if (label := _string_value(item.get("label"))) is not None
            for transform in [_string_value(item.get("transform")) or "identity"]
        )
        if isinstance(raw_facets, list)
        else ()
    )
    raw_breakdowns = contract.plan.get("breakdowns") if contract.contract_version == 3 else None
    breakdown_specs = (
        tuple(
            EvaluationBreakdownSpec(
                id=breakdown_id,
                label=label,
                dimensions=cast(tuple[str, str], tuple(dimensions)),
                presentation="matrix",
                multi_value=cast(
                    Literal["reject", "cross"],
                    multi_value if multi_value in {"reject", "cross"} else "reject",
                ),
                missing=cast(
                    Literal["exclude", "bucket"],
                    missing if missing in {"exclude", "bucket"} else "exclude",
                ),
            )
            for item in raw_breakdowns
            if isinstance(item, Mapping)
            if (breakdown_id := _string_value(item.get("id"))) is not None
            if (label := _string_value(item.get("label"))) is not None
            if isinstance((dimensions := item.get("dimensions")), list)
            and len(dimensions) == 2
            and all(isinstance(dimension, str) and dimension for dimension in dimensions)
            for multi_value in [_string_value(item.get("multi_value")) or "reject"]
            for missing in [_string_value(item.get("missing")) or "exclude"]
        )
        if isinstance(raw_breakdowns, list)
        else ()
    )
    primary = configured_primary or (rewards[0] if rewards else None)
    # A pass rate is a verifier claim, not a property Observatory may infer
    # from a continuous reward or a transport-level completion flag.
    pass_rate_metric = success_definition.signal if success_definition is not None else configured_pass_rate
    metric_names = tuple(
        dict.fromkeys((*rewards, *(value for value in (configured_primary, pass_rate_metric) if value)))
    )
    metrics = tuple(
        EvaluationMetricDefinition(
            name=name,
            label=(
                success_definition.label
                if success_definition is not None and name == success_definition.signal
                else _humanize_name(name)
            ),
            role=(
                "primary_reward" if name == primary else "success" if name == pass_rate_metric else "reward_component"
            ),
        )
        for name in metric_names
    )
    selection_id, _ = _selection_identity(environment)
    key = package or selection_id or "evaluation"
    return EvaluationMetadata(
        key=key,
        label=_humanize_name(package or selection_id or "evaluation"),
        category=category,
        package=package,
        dataset=dataset,
        dataset_revision=dataset_revision,
        split=split,
        source_revision=source_revision,
        primary_metric=primary,
        primary_metric_label=configured_primary_label or (_humanize_name(primary) if primary else None),
        pass_rate_metric=pass_rate_metric,
        pass_rate_basis=(
            f"{success_definition.namespace}.{success_definition.signal} "
            f"{success_definition.operator} {success_definition.value:g}"
            if success_definition is not None
            else "configured binary metric"
            if configured_pass_rate
            else None
        ),
        success_definition=success_definition,
        contract_id=contract.contract_id,
        contract_version=contract.contract_version,
        contract_state=contract.state,
        facet_specs=facet_specs,
        breakdown_specs=breakdown_specs,
        metrics=metrics,
    )


def _evaluation_expected_traces(
    inputs: Mapping[str, JsonValue],
    *,
    fallback: int | None,
) -> int | None:
    """Return the frozen population size instead of a lagging live trace count."""

    contract = read_evaluation_contract(inputs)
    if contract.state == "versioned":
        num_tasks = contract.population.get("num_tasks")
        num_rollouts = contract.population.get("num_rollouts", 1)
        if (
            isinstance(num_tasks, int)
            and not isinstance(num_tasks, bool)
            and num_tasks > 0
            and isinstance(num_rollouts, int)
            and not isinstance(num_rollouts, bool)
            and num_rollouts > 0
        ):
            return num_tasks * num_rollouts
    return fallback if fallback is not None and fallback > 0 else None


@dataclass(frozen=True, slots=True)
class _TraceReadContext:
    detail: RunDetail
    trace_type: str
    expires_at: float


def _comparison_context(view: EvaluationRunView | RunView) -> dict[str, JsonValue]:
    inputs = view.resolved_inputs
    model_id, model_revision = _selection_identity(inputs.get("model"))
    inference = inputs.get("evaluation_inference") or inputs.get("inference")
    inference_id, inference_revision = _selection_identity(inference)
    environment_id, environment_revision = _selection_identity(inputs.get("environment"))
    return {
        "model": model_id,
        "model_revision": model_revision,
        "inference": inference_id,
        "inference_revision": inference_revision,
        "environment": environment_id,
        "environment_revision": environment_revision,
    }


def _logical_metric_series(series: MetricSeries) -> MetricSeries:
    """Project replayed evidence onto its source step without double plotting.

    Isolated environments replay trace-derived metrics during finalization.
    Their provider step is append-only storage position, while ``source_step``
    is the optimizer step they describe. Replay is authoritative when present.
    For ordinary same-step measurements, collapse only numerically equivalent
    duplicates and preserve meaningfully different observations.
    """

    replay: list[tuple[MetricPoint, int]] = []
    for point in series.points:
        source_step = point.attributes.get("source_step")
        if (
            point.attributes.get("observation_source") == "verifiers"
            and isinstance(source_step, int)
            and not isinstance(source_step, bool)
            and source_step >= 0
        ):
            replay.append((point, source_step))
    if replay:
        replay_steps = {source_step for _, source_step in replay}
        # Replay is authoritative only for the optimizer steps it covers. Keep
        # native points for other steps so a partially finalized run cannot
        # silently lose its optimizer movement. Multiple replay points for one
        # source step are intentional: they represent rollout waves and must
        # remain available to population reducers.
        native = [
            point
            for point in series.points
            if point.attributes.get("observation_source") != "verifiers" and point.step not in replay_steps
        ]
        projected = [point.model_copy(update={"step": source_step}) for point, source_step in replay]
        return MetricSeries(
            name=series.name,
            points=tuple(sorted((*native, *projected), key=lambda point: point.step if point.step is not None else -1)),
        )

    retained: list[MetricPoint] = []
    for point in series.points:
        if point.step is not None and any(
            existing.step == point.step and existing.value == point.value for existing in retained
        ):
            continue
        retained.append(point)
    # Provider history is append-only, but readers are not required to return
    # rows in logical-step order. Sorting here keeps reducers (especially
    # ``last``) and every presentation surface on the same timeline.
    retained.sort(key=lambda point: point.step if point.step is not None else -1)
    return MetricSeries(name=series.name, points=tuple(retained))


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile + 0.5)))
    return ordered[index]


async def _inference_timing_summary(
    source: RunDataSource,
    run_id: str,
) -> InferenceTimingSummary | None:
    values: dict[str, list[float]] = {
        "queue": [],
        "prefill": [],
        "decode": [],
        "engine_e2e": [],
    }
    cursor: str | None = None
    requests = 0
    while True:
        page = await source.traces(
            run_id,
            TraceQuery(trace_type="inference", cursor=cursor, limit=1000),
        )
        for trace in page.items:
            request_has_timing = False
            for stage in values:
                value = trace.payload.get(f"{stage}_seconds")
                if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                    values[stage].append(float(value) * 1000)
                    request_has_timing = True
            requests += int(request_has_timing)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    if requests == 0:
        return None
    labels = {
        "queue": "Queue",
        "prefill": "Prefill",
        "decode": "Decode",
        "engine_e2e": "Engine end-to-end",
    }
    stages = tuple(
        InferenceTimingStageSummary(
            stage=stage,  # type: ignore[arg-type]
            label=labels[stage],
            samples=len(stage_values),
            mean_ms=fmean(stage_values),
            p50_ms=_percentile(stage_values, 0.50),
            p95_ms=_percentile(stage_values, 0.95),
        )
        for stage, stage_values in values.items()
        if stage_values
    )
    return InferenceTimingSummary(requests=requests, stages=stages)


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
    selected = selected[:maximum]
    # Preserve the terminal observation. Without this, a long campaign can
    # render a chart whose apparent latest point predates the actual run end.
    if points[-1] not in selected:
        selected[-1] = points[-1]
    selected.sort(key=lambda point: point.step if point.step is not None else -1)
    return MetricSeries(name=series.name, points=tuple(selected)), True


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


def _config_positive_int(resolved_inputs: Mapping[str, JsonValue], key: str) -> int | None:
    for value in _config_values(dict(resolved_inputs), key):
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


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
        categories = _config_values(dict(resolved_inputs), "environment_category")
        return (
            any(bool(value) for value in _config_values(dict(resolved_inputs), "tools"))
            or any(value is True for value in _config_values(dict(resolved_inputs), "tool_environment"))
            or any(isinstance(value, str) and "tool" in value.lower().split("-") for value in categories)
        )
    if condition == "dapo_algorithm_enabled":
        return any(
            isinstance(value, str) and value.lower() == "dapo"
            for value in _config_values(dict(resolved_inputs), "algorithm")
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
    if definition.job_kind in {"train.grpo", "train.sampo", "train.distill"}:
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
        source: RunDataSource | Mapping[str, RunDataSource] | RunSourceRegistry,
        definitions: Mapping[str, JobTelemetryDefinition] = DEFAULT_TELEMETRY_DEFINITIONS,
        *,
        semantic_provider: SemanticSummaryProvider | None = None,
        redaction: RedactionPolicy | None = None,
        source_discovery: TrackioSourceDiscovery | None = None,
    ) -> None:
        if isinstance(source, RunSourceRegistry):
            self.registry = source
        else:
            sources = {"default": source} if not isinstance(source, Mapping) else dict(source)
            self.registry = RunSourceRegistry(cast(Mapping[str, RunDataSource], sources))
        self._definitions = dict(definitions)
        self._redaction = redaction or RedactionPolicy()
        self._semantic = SemanticAnalysisService(semantic_provider)
        self._source_discovery = source_discovery
        self._trace_read_contexts: dict[tuple[str, str], _TraceReadContext] = {}

    def _locator(self, value: str | RunLocator) -> RunLocator:
        if isinstance(value, RunLocator):
            return value
        if not self.registry.source_ids:
            raise LookupError("Observatory has no configured sources")
        return RunLocator(source_id=self.registry.source_ids[0], run_id=value)

    def _remember_trace_read_context(
        self,
        locator: RunLocator,
        detail: RunDetail,
    ) -> _TraceReadContext:
        definition = self._definitions.get(detail.summary.job_kind)
        trace_type = (
            definition.trace_sections[0].trace_type
            if definition is not None and definition.trace_sections
            else "verifiers"
        )
        context = _TraceReadContext(
            detail=detail,
            trace_type=trace_type,
            expires_at=time.monotonic() + (60.0 if detail.summary.status == "running" else 300.0),
        )
        key = (locator.source_id, locator.run_id)
        self._trace_read_contexts[key] = context
        if len(self._trace_read_contexts) > 512:
            self._trace_read_contexts.pop(next(iter(self._trace_read_contexts)))
        return context

    async def _trace_read_context(self, locator: RunLocator) -> _TraceReadContext:
        key = (locator.source_id, locator.run_id)
        cached = self._trace_read_contexts.get(key)
        if cached is not None and cached.expires_at > time.monotonic():
            return cached
        source = self.registry.resolve(locator)
        detail = await source.get_run(locator.run_id)
        return self._remember_trace_read_context(locator, detail)

    async def start_source_discovery(self) -> None:
        if self._source_discovery is not None:
            await self._source_discovery.start()

    async def stop_source_discovery(self) -> None:
        if self._source_discovery is not None:
            await self._source_discovery.stop()

    async def refresh_sources(self) -> SourceRefreshStatus:
        if self._source_discovery is None:
            return SourceRefreshStatus(enabled=False, state="disabled")
        return await self._source_discovery.refresh()

    def source_refresh_status(self) -> SourceRefreshStatus:
        if self._source_discovery is None:
            return SourceRefreshStatus(enabled=False, state="disabled")
        return self._source_discovery.status()

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

    async def locate_run(self, run_id: str) -> tuple[LocatedRunSummary, ...]:
        return await self.registry.locate_run(run_id)

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
        self._remember_trace_read_context(locator, detail)
        definition = self.get_job_telemetry_schema(detail.summary.job_kind)
        if mode == "generic":
            raise ValueError("use get_run_view_response for generic mode")
        return await self._metric_job_view(locator, definition, detail=detail)

    async def get_run_view_response(
        self,
        run: str | RunLocator,
        mode: ViewMode = "auto",
        metrics: tuple[str, ...] = (),
    ) -> RunViewResponse:
        locator = self._locator(run)
        source = self.registry.resolve(locator)
        detail = await source.get_run(locator.run_id)
        self._remember_trace_read_context(locator, detail)
        definition = self._definitions.get(detail.summary.job_kind)
        if mode == "job" and definition is None:
            raise LookupError(f"job view is unavailable for {detail.summary.job_kind!r}")
        if mode == "generic" or definition is None:
            generic = await self._generic_view(locator, metrics, detail=detail)
            reason = None if mode == "generic" else f"No job view is registered for {detail.summary.job_kind}."
            return RunViewResponse(
                requested_mode=mode,
                resolved_mode="generic",
                fallback_reason=reason,
                view=generic,
            )
        if detail.summary.job_kind == "serve.benchmark":
            view: RunView | EvaluationRunView | ServingBenchmarkRunView = await project_serving_benchmark(
                locator,
                source,
                detail,
                self._redaction,
            )
        else:
            metric_view = await self._metric_job_view(locator, definition, detail=detail)
            if detail.summary.job_kind.startswith("eval."):
                evaluation = await trace_evaluation_view(
                    source,
                    locator.run_id,
                    expected=_evaluation_expected_traces(
                        metric_view.resolved_inputs,
                        fallback=detail.trace_count or None,
                    ),
                    metadata=_evaluation_metadata(metric_view.resolved_inputs),
                )
                view = EvaluationRunView(
                    schema_version=metric_view.schema_version,
                    locator=locator,
                    run=metric_view.run,
                    summary=metric_view.summary,
                    charts=metric_view.charts,
                    metric_help=metric_view.metric_help,
                    completeness=metric_view.completeness,
                    alerts=metric_view.alerts,
                    evaluation=evaluation,
                    comparison_key=_evaluation_population_key(metric_view),
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

    async def _metric_job_view(
        self,
        locator: RunLocator,
        definition: JobTelemetryDefinition,
        *,
        detail: RunDetail | None = None,
    ) -> RunView:
        source = self.registry.resolve(locator)
        if detail is None:
            detail = await source.get_run(locator.run_id)
        names = tuple(sorted(definition.metric_names))
        series_values, artifacts = await asyncio.gather(
            source.metric_series(locator.run_id, names), source.artifacts(locator.run_id)
        )
        series_values = tuple(_logical_metric_series(series) for series in series_values)
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

    async def _generic_view(
        self,
        locator: RunLocator,
        metrics: tuple[str, ...],
        *,
        detail: RunDetail | None = None,
    ) -> GenericRunView:
        source = self.registry.resolve(locator)
        if detail is None:
            detail, artifacts = await asyncio.gather(source.get_run(locator.run_id), source.artifacts(locator.run_id))
        else:
            artifacts = await source.artifacts(locator.run_id)
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
                    *(
                        name
                        for name in detail.metric_names
                        if name.startswith(("system/", "tracking/", "serve/backend/"))
                    ),
                    *(
                        name
                        for name in (
                            "train/rl/rollout_tokens_per_second",
                            "train/rl/time/rollout_seconds",
                        )
                        if name in detail.metric_names
                    ),
                    *(metric for _, _, metric, *_ in cards),
                }
            )
        )
        series_values = await source.metric_series(locator.run_id, requested)
        series_values = tuple(_logical_metric_series(series) for series in series_values)
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
            (
                "inference_backend",
                "Inference backend pressure",
                (
                    "serve/backend/kv_cache_usage_ratio",
                    "serve/backend/running_requests",
                    "serve/backend/waiting_requests",
                ),
            ),
        ):
            group_series = tuple(by_name[name] for name in names if name in by_name)
            if group_series:
                groups.append(SystemMetricGroup(key=key, title=title, series=group_series))
        execution_targets = execution_target_contexts(detail.resolved_inputs)
        capacity_state, capacity_bytes = execution_target_capacity(execution_targets)
        phase_projection = project_runtime_phases(detail, by_name)
        kv_usage = by_name.get("serve/backend/kv_cache_usage_ratio")
        kv_peak_series = by_name.get("serve/backend/kv_cache_peak_usage_ratio")
        kv_capacity = _reduce(
            by_name.get(
                "serve/backend/kv_cache_capacity_tokens",
                MetricSeries(name="serve/backend/kv_cache_capacity_tokens"),
            ),
            "last",
        )
        kv_peak = _reduce(
            kv_peak_series or MetricSeries(name="serve/backend/kv_cache_peak_usage_ratio"),
            "max",
        )
        if kv_peak is None and kv_usage is not None:
            kv_peak = _reduce(kv_usage, "max")
        mtp_acceptance = by_name.get("serve/backend/speculative_acceptance_rate")
        mtp_length = by_name.get("serve/backend/speculative_accepted_length")
        rollout_throughput = by_name.get("train/rl/rollout_tokens_per_second")
        rollout_seconds = by_name.get("train/rl/time/rollout_seconds")
        mtp_selected = _condition_active("mtp_rollout_enabled", detail.resolved_inputs, by_name)
        environment_concurrency = _config_positive_int(detail.resolved_inputs, "max_concurrent")
        inference_sequence_cap = _config_positive_int(detail.resolved_inputs, "max_num_seqs")
        rollouts_per_prompt = _config_positive_int(detail.resolved_inputs, "num_generations")
        rollouts_per_update = _config_positive_int(detail.resolved_inputs, "global_batch_size")
        backend_runtime = (
            BackendRuntimeSummary(
                kv_cache_capacity_tokens=kv_capacity,
                kv_cache_peak_usage_ratio=kv_peak,
                kv_cache_samples=max(
                    len(kv_usage.points) if kv_usage is not None else 0,
                    len(kv_peak_series.points) if kv_peak_series is not None else 0,
                ),
                mtp_selected=mtp_selected,
                mtp_acceptance_rate=_reduce(
                    mtp_acceptance or MetricSeries(name="serve/backend/speculative_acceptance_rate"),
                    "last",
                ),
                mtp_accepted_length=_reduce(
                    mtp_length or MetricSeries(name="serve/backend/speculative_accepted_length"),
                    "last",
                ),
                mtp_samples=max(
                    len(mtp_acceptance.points) if mtp_acceptance is not None else 0,
                    len(mtp_length.points) if mtp_length is not None else 0,
                ),
                rollout_tokens_per_second_latest=_reduce(
                    rollout_throughput or MetricSeries(name="train/rl/rollout_tokens_per_second"),
                    "last",
                ),
                rollout_tokens_per_second_mean=_reduce(
                    rollout_throughput or MetricSeries(name="train/rl/rollout_tokens_per_second"),
                    "mean",
                ),
                rollout_seconds_latest=_reduce(
                    rollout_seconds or MetricSeries(name="train/rl/time/rollout_seconds"),
                    "last",
                ),
                rollout_seconds_mean=_reduce(
                    rollout_seconds or MetricSeries(name="train/rl/time/rollout_seconds"),
                    "mean",
                ),
                rollout_samples=max(
                    len(rollout_throughput.points) if rollout_throughput is not None else 0,
                    len(rollout_seconds.points) if rollout_seconds is not None else 0,
                ),
                environment_concurrency=environment_concurrency,
                inference_sequence_cap=inference_sequence_cap,
                rollouts_per_prompt=rollouts_per_prompt,
                rollouts_per_update=rollouts_per_update,
            )
            if any(
                value is not None
                for value in (
                    kv_capacity,
                    kv_peak,
                    mtp_acceptance,
                    mtp_length,
                    rollout_throughput,
                    rollout_seconds,
                    environment_concurrency,
                    inference_sequence_cap,
                    rollouts_per_prompt,
                    rollouts_per_update,
                )
            )
            or mtp_selected
            else None
        )
        inference_timing = (
            await _inference_timing_summary(source, locator.run_id)
            if detail.summary.job_kind == "serve.benchmark" and detail.trace_count
            else None
        )
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
            inference_timing=inference_timing,
            backend_runtime=backend_runtime,
            capabilities=source.capabilities,
        )

    async def get_metric_series(self, run: str | RunLocator, query: MetricSeriesQuery) -> MetricSeriesSet:
        locator = self._locator(run)
        source = self.registry.resolve(locator)
        detail = await source.get_run(locator.run_id)
        unknown = set(query.names) - set(detail.metric_names)
        if unknown:
            raise ValueError(f"unknown metric names: {', '.join(sorted(unknown))}")
        raw = tuple(
            _logical_metric_series(series) for series in await source.metric_series(locator.run_id, query.names)
        )
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
        evaluation_views = tuple(view for view in job_views if isinstance(view, EvaluationRunView))
        if evaluation_views and len(evaluation_views) != len(job_views):
            return RunComparison(
                state="incomparable",
                columns=(),
                rows=(),
                reason="evaluation runs cannot be compared with non-evaluation job views",
            )
        basis: tuple[str, ...] = ()
        if evaluation_views:
            identities = {_evaluation_population_key(view) for view in evaluation_views}
            if len(identities) != 1:
                return RunComparison(
                    job_kind=next(iter(job_kinds)),
                    state="incomparable",
                    columns=(),
                    rows=(),
                    reason=(
                        "runs use different evaluation populations; dataset/task identity, split, seed, "
                        "environment source, and native metric schema must match"
                    ),
                    basis=(
                        "job kind",
                        "dataset/task selection",
                        "split/subset/seed",
                        "environment source",
                        "native metric schema",
                    ),
                )
            basis = (
                "job kind",
                "dataset/task selection",
                "split/subset/seed",
                "environment source",
                "native metric schema",
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
                    context=(_comparison_context(view) if isinstance(view, EvaluationRunView) else {}),
                )
            )
        return RunComparison(
            job_kind=job_kind,
            state="comparable",
            columns=definition.comparison_keys,
            rows=tuple(rows),
            basis=basis,
        )

    async def get_trace_evaluation_view(
        self,
        run: str | RunLocator,
        *,
        include_traces: bool = True,
    ) -> TraceEvaluationView:
        locator = self._locator(run)
        source = self.registry.resolve(locator)
        context = await self._trace_read_context(locator)
        detail = context.detail
        return await trace_evaluation_view(
            source,
            locator.run_id,
            expected=_evaluation_expected_traces(
                detail.resolved_inputs,
                fallback=detail.trace_count or None,
            ),
            trace_type=context.trace_type,
            metadata=_evaluation_metadata(detail.resolved_inputs),
            include_traces=include_traces,
        )

    async def get_trace_summary_page(
        self,
        run: str | RunLocator,
        *,
        cursor: str | None = None,
        limit: int = 100,
    ) -> TraceSummaryPage:
        locator = self._locator(run)
        source = self.registry.resolve(locator)
        context = await self._trace_read_context(locator)
        detail = context.detail
        return await trace_summary_page(
            source,
            locator.run_id,
            total=detail.trace_count,
            cursor=cursor,
            limit=limit,
            trace_type=context.trace_type,
            metadata=(
                _evaluation_metadata(detail.resolved_inputs) if detail.summary.job_kind.startswith("eval.") else None
            ),
        )

    async def get_run_comparison_key(self, run: str | RunLocator) -> tuple[str, str] | None:
        """Return the lightweight population key used to populate Compare."""

        locator = self._locator(run)
        source = self.registry.resolve(locator)
        detail = await source.get_run(locator.run_id)
        definition = self._definitions.get(detail.summary.job_kind)
        if definition is None or not detail.summary.job_kind.startswith("eval."):
            return None
        metric_view = await self._metric_job_view(locator, definition)
        return detail.summary.job_kind, _evaluation_population_key(metric_view)

    async def get_trace_detail(self, run: str | RunLocator, trace_id: str) -> TraceDetail:
        locator = self._locator(run)
        source = self.registry.resolve(locator)
        run_detail = (await self._trace_read_context(locator)).detail
        metadata = (
            _evaluation_metadata(run_detail.resolved_inputs)
            if run_detail.summary.job_kind.startswith("eval.")
            else None
        )
        return await load_trace_detail(
            source,
            locator.run_id,
            trace_id,
            self._redaction,
            metadata=metadata,
        )

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

    async def get_serving_capacity_view(
        self,
        work_package_id: str,
        *,
        project_id: str | None = None,
        source_id: str | None = None,
    ) -> ServingCapacityWorkPackageView:
        located = await self.registry.list_runs(
            RunQuery(
                project_id=project_id,
                work_package_id=work_package_id,
                job_kinds=("serve.benchmark",),
                limit=1000,
            )
        )
        if source_id is not None:
            self.registry.resolve(RunLocator(source_id=source_id, run_id="source-probe"))
            located = tuple(item for item in located if item.locator.source_id == source_id)

        async def project(item: LocatedRunSummary) -> ServingBenchmarkRunView:
            source = self.registry.resolve(item.locator)
            detail = await source.get_run(item.locator.run_id)
            return await project_serving_benchmark(
                item.locator,
                source,
                detail,
                self._redaction,
                include_request_traces=detail.trace_count > 0,
                include_artifacts=False,
            )

        views = await asyncio.gather(*(project(item) for item in located))
        strict_candidates = [
            (item, view)
            for item, view in zip(located, views, strict=True)
            if len(view.operating_points) > 1
            and view.population.cohort == "representative"
            and view.eligibility.requirements_digest is not None
            and view.execution_target_id is not None
            and view.workload_id is not None
            and view.population.corpus_digest is not None
        ]
        reference = strict_candidates[0] if strict_candidates else None
        reference_view = reference[1] if reference is not None else None

        contenders: list[ServingContenderView] = []
        for item, view in zip(located, views, strict=True):
            mismatch: list[str] = []
            if reference_view is None:
                mismatch.append("no decision-grade multi-point representative sweep is available")
            else:
                comparisons = (
                    (
                        "requirements",
                        view.eligibility.requirements_digest,
                        reference_view.eligibility.requirements_digest,
                    ),
                    ("execution target", view.execution_target_id, reference_view.execution_target_id),
                    ("workload", view.workload_id, reference_view.workload_id),
                    (
                        "corpus",
                        view.population.corpus_digest,
                        reference_view.population.corpus_digest,
                    ),
                    (
                        "calculator",
                        view.eligibility.calculator_version,
                        reference_view.eligibility.calculator_version,
                    ),
                    ("cohort", view.population.cohort, "representative"),
                )
                mismatch.extend(
                    label for label, actual, expected in comparisons if actual is None or actual != expected
                )
                if len(view.operating_points) < 2:
                    mismatch.append("run contains only one operating point")
            contenders.append(
                ServingContenderView(
                    locator=item.locator,
                    run_key=item.run_key,
                    display_name=item.run.display_name,
                    started_at=item.run.started_at,
                    model_variant_id=view.model_variant_id,
                    inference_binding_id=view.inference_binding_id,
                    inference_backend=view.inference_backend,
                    workload_id=view.workload_id,
                    corpus_digest=view.population.corpus_digest,
                    execution_target_id=view.execution_target_id,
                    requirements_digest=view.eligibility.requirements_digest,
                    calculator_version=view.eligibility.calculator_version,
                    comparable=not mismatch,
                    comparability_reason=(
                        None if not mismatch else "Incomparable: " + ", ".join(dict.fromkeys(mismatch)) + "."
                    ),
                    selected_point=view.selected_point,
                    eligibility=view.eligibility,
                )
            )

        pareto_candidates = [
            contender
            for contender in contenders
            if contender.comparable
            and contender.eligibility.state == "eligible"
            and contender.selected_point is not None
            and contender.selected_point.aggregate_output_tps is not None
            and contender.selected_point.p95_ttft_ms is not None
            and contender.selected_point.peak_vram_bytes is not None
        ]

        def dominates(left: ServingContenderView, right: ServingContenderView) -> bool:
            left_point = cast(Any, left.selected_point)
            right_point = cast(Any, right.selected_point)
            no_worse = (
                left_point.aggregate_output_tps >= right_point.aggregate_output_tps
                and left_point.p95_ttft_ms <= right_point.p95_ttft_ms
                and left_point.peak_vram_bytes <= right_point.peak_vram_bytes
            )
            strictly_better = (
                left_point.aggregate_output_tps > right_point.aggregate_output_tps
                or left_point.p95_ttft_ms < right_point.p95_ttft_ms
                or left_point.peak_vram_bytes < right_point.peak_vram_bytes
            )
            return no_worse and strictly_better

        pareto_keys = {
            contender.run_key
            for contender in pareto_candidates
            if not any(
                other.run_key != contender.run_key and dominates(other, contender) for other in pareto_candidates
            )
        }
        contenders = [
            contender.model_copy(update={"pareto_member": contender.run_key in pareto_keys}) for contender in contenders
        ]
        pareto = tuple(
            ServingParetoPoint(
                run_key=contender.run_key,
                model_variant_id=contender.model_variant_id,
                inference_binding_id=contender.inference_binding_id,
                aggregate_output_tps=cast(float, contender.selected_point.aggregate_output_tps),
                p95_ttft_ms=cast(float, contender.selected_point.p95_ttft_ms),
                peak_vram_bytes=int(cast(float, contender.selected_point.peak_vram_bytes)),
            )
            for contender in contenders
            if contender.pareto_member and contender.selected_point is not None
        )
        rows = []
        for item, view in zip(located, views, strict=True):
            for point in view.operating_points:
                rows.append(
                    ServingCapacityRunRow(
                        locator=item.locator,
                        run_key=item.run_key,
                        display_name=item.run.display_name,
                        started_at=item.run.started_at,
                        model_variant_id=view.model_variant_id,
                        inference_binding_id=view.inference_binding_id,
                        inference_backend=view.inference_backend,
                        workload_id=view.workload_id,
                        execution_target_id=view.execution_target_id,
                        requirements_digest=view.eligibility.requirements_digest,
                        point=point,
                        point_state=(
                            "incomplete"
                            if point.evidence_state != "complete"
                            else "valid"
                            if point.valid
                            else "constraint_failed"
                        ),
                        point_label=(
                            f"{point.terminal_status.replace('_', ' ').title()} boundary"
                            if point.terminal_status is not None
                            else "Incomplete evidence"
                            if point.evidence_state != "complete"
                            else "Valid point"
                            if point.valid
                            else "Constraint missed"
                        ),
                        eligibility=view.eligibility,
                    )
                )
        rows.sort(
            key=lambda row: (
                row.inference_binding_id or "",
                row.point.concurrency,
                row.started_at,
            )
        )
        projects = {item.run.project_id for item in located}
        return ServingCapacityWorkPackageView(
            project_id=project_id or (next(iter(projects)) if len(projects) == 1 else None),
            work_package_id=work_package_id,
            methodology="strict_pareto" if reference_view is not None else "cross_run_compatibility",
            explanation=(
                "Comparable contenders use the same representative workload, corpus, project requirements, "
                "execution target, and calculator. Ineligible and incomparable runs remain visible."
                if reference_view is not None
                else "Compatibility projection across historical single-point runs. "
                "Rows are grouped by inference binding and are not treated as one decision-grade sweep."
            ),
            requirements=reference_view.requirements if reference_view is not None else (),
            execution_target_id=reference_view.execution_target_id if reference_view is not None else None,
            workload_id=reference_view.workload_id if reference_view is not None else None,
            corpus_digest=reference_view.population.corpus_digest if reference_view is not None else None,
            requirements_digest=(
                reference_view.eligibility.requirements_digest if reference_view is not None else None
            ),
            calculator_version=(reference_view.eligibility.calculator_version if reference_view is not None else None),
            contenders=tuple(contenders),
            pareto=pareto,
            rows=tuple(rows),
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
        if definition.job_kind in {"train.grpo", "train.sampo", "train.distill"} and trace_count == 0:
            technique = definition.job_kind.removeprefix("train.")
            alerts.append(
                RunAlert(
                    id=f"missing-{technique}-traces",
                    severity="error",
                    message=(
                        f"{definition.display_name} rollout traces are missing, "
                        "so aggregate training evidence cannot be audited."
                    ),
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
