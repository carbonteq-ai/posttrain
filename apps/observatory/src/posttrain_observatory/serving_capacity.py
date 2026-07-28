"""First-party serving-capacity projection for ``serve.benchmark`` runs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from statistics import fmean
from typing import Literal, cast

from posttrain.common import JsonValue
from posttrain.tracking import ArtifactSet, MetricSeries, RunDataSource, RunDetail, TraceQuery

from .execution_targets import execution_target_contexts
from .models import (
    BenchmarkPopulationView,
    RunAlert,
    RunLocator,
    RuntimeSettingGroup,
    RuntimeSettingValue,
    ServingBenchmarkRunView,
    ServingEligibility,
    ServingOperatingPoint,
    ServingRequirementView,
)
from .redaction import RedactionPolicy

CALCULATOR_VERSION = "serving-capacity-v1"

_SERVING_METRICS = (
    "serve/run/concurrency",
    "serve/run/context_tokens",
    "serve/run/corpus_records_measured",
    "serve/run/requests_attempted",
    "serve/run/requests_measured",
    "serve/run/requests_failed",
    "serve/run/requests_unsupported",
    "serve/run/point_resource_exhausted",
    "serve/run/point_unsupported",
    "serve/run/point_failed",
    "serve/run/input_tokens_measured",
    "serve/run/output_tokens_measured",
    "serve/run/measurement_duration_s",
    "serve/backend/peak_vram_bytes",
    "serve/concurrency",
    "serve/context_window",
    "serve/corpus_records_measured",
    "serve/elapsed_seconds",
    "serve/input_tokens_mean",
    "serve/input_tokens_p95",
    "serve/output_token_throughput",
    "serve/output_tokens",
    "serve/p50_tpot",
    "serve/p50_ttft",
    "serve/p95_tpot",
    "serve/p95_ttft",
    "serve/peak_gpu_memory_gib",
    "serve/requests",
    "serve/backend/kv_cache_peak_usage_ratio",
)

_METRIC_ALIASES = {
    "serve/concurrency": ("serve/run/concurrency", "serve/concurrency"),
    "serve/context_window": ("serve/run/context_tokens", "serve/context_window"),
    "serve/corpus_records_measured": (
        "serve/run/corpus_records_measured",
        "serve/corpus_records_measured",
    ),
    "serve/requests": ("serve/run/requests_attempted", "serve/requests"),
    "serve/requests_measured": ("serve/run/requests_measured",),
    "serve/requests_failed": ("serve/run/requests_failed",),
    "serve/requests_unsupported": ("serve/run/requests_unsupported",),
    "serve/input_tokens": ("serve/run/input_tokens_measured", "serve/input_tokens"),
    "serve/output_tokens": ("serve/run/output_tokens_measured", "serve/output_tokens"),
    "serve/elapsed_seconds": (
        "serve/run/measurement_duration_s",
        "serve/elapsed_seconds",
    ),
    "serve/peak_vram_bytes": ("serve/backend/peak_vram_bytes",),
}

_RUNTIME_PRESENTATION = (
    (
        "model_context",
        "Model & context",
        (
            ("dtype", "Runtime dtype", None, "primary"),
            ("max_model_len", "Maximum model length", "tokens", "primary"),
            ("load_format", "Load format", None, "advanced"),
        ),
    ),
    (
        "parallelism",
        "Parallelism",
        (
            ("tensor_parallel_size", "Tensor parallel size", None, "primary"),
            ("pipeline_parallel_size", "Pipeline parallel size", None, "advanced"),
        ),
    ),
    (
        "memory_cache",
        "Memory & cache",
        (
            ("gpu_memory_utilization", "GPU memory reservation", "ratio", "primary"),
            ("kv_cache_dtype", "KV-cache dtype", None, "primary"),
            ("enable_prefix_caching", "Prefix caching", None, "advanced"),
        ),
    ),
    (
        "scheduler",
        "Scheduler & batching",
        (
            ("max_num_seqs", "Maximum sequences", None, "primary"),
            ("max_num_batched_tokens", "Maximum batched tokens", "tokens", "primary"),
            ("enable_chunked_prefill", "Chunked prefill", None, "advanced"),
        ),
    ),
    (
        "acceleration",
        "Acceleration",
        (
            ("enforce_eager", "Eager execution", None, "primary"),
            ("speculative_config", "Speculative decoding", None, "primary"),
        ),
    ),
)


def _selection(inputs: Mapping[str, JsonValue], key: str) -> tuple[str | None, Mapping[str, JsonValue]]:
    value = inputs.get(key)
    if not isinstance(value, Mapping):
        return None, {}
    resolved = value.get("resolved")
    detail = cast(Mapping[str, JsonValue], resolved) if isinstance(resolved, Mapping) else value
    selection_id = value.get("selection_id", value.get("id"))
    return (selection_id if isinstance(selection_id, str) else None), detail


def _number(value: JsonValue | None) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _integer(value: JsonValue | None) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number >= 0 else None


def _string(value: JsonValue | None) -> str | None:
    return value if isinstance(value, str) else None


def _last(series: Mapping[str, MetricSeries], name: str) -> float | None:
    for candidate in _METRIC_ALIASES.get(name, (name,)):
        points = series.get(candidate, MetricSeries(name=candidate)).points
        if points:
            return points[-1].value
    return None


def _at_sweep(series: Mapping[str, MetricSeries], name: str, sweep_index: int) -> float | None:
    for candidate in _METRIC_ALIASES.get(name, (name,)):
        points = series.get(candidate, MetricSeries(name=candidate)).points
        for point in points:
            if point.step == sweep_index:
                return point.value
        if len(points) == 1:
            return points[0].value
        if sweep_index < len(points):
            return points[sweep_index].value
    return None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.5)))
    return ordered[index]


@dataclass(slots=True)
class _RequestAccumulator:
    attempted: int = 0
    completed: int = 0
    failed: int = 0
    output_tokens: int = 0
    output_target_hits: int = 0
    input_token_counts: list[float] = field(default_factory=list)
    output_token_counts: list[float] = field(default_factory=list)
    ttft_ms: list[float] = field(default_factory=list)
    tpot_ms: list[float] = field(default_factory=list)


async def _request_evidence(
    source: RunDataSource,
    run_id: str,
    output_token_target: int | None,
) -> dict[int, tuple[dict[str, float | int | None], bool]]:
    by_sweep: defaultdict[int, _RequestAccumulator] = defaultdict(_RequestAccumulator)
    cursor: str | None = None
    while True:
        page = await source.traces(run_id, TraceQuery(trace_type="inference", cursor=cursor, limit=1000))
        for trace in page.items:
            if trace.payload.get("warmup") is True:
                continue
            sweep_index = _integer(trace.payload.get("sweep_index")) or 0
            evidence = by_sweep[sweep_index]
            evidence.attempted += 1
            error_class = trace.payload.get("error_class")
            if isinstance(error_class, str) and error_class:
                evidence.failed += 1
            else:
                evidence.completed += 1
            input_count = _integer(trace.payload.get("input_tokens"))
            output_count = _integer(trace.payload.get("output_tokens"))
            if input_count is not None:
                evidence.input_token_counts.append(float(input_count))
            if output_count is not None:
                evidence.output_token_counts.append(float(output_count))
                evidence.output_tokens += output_count
                if output_token_target is not None and output_count >= output_token_target:
                    evidence.output_target_hits += 1
            ttft = _number(trace.payload.get("ttft_seconds"))
            tpot = _number(trace.payload.get("tpot_seconds"))
            if ttft is not None and ttft >= 0:
                evidence.ttft_ms.append(ttft * 1000)
            if tpot is not None and tpot >= 0:
                evidence.tpot_ms.append(tpot * 1000)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    return {
        sweep_index: (
            {
                "attempted": evidence.attempted,
                "completed": evidence.completed,
                "failed": evidence.failed,
                "output_tokens": evidence.output_tokens,
                "input_tokens_mean": (fmean(evidence.input_token_counts) if evidence.input_token_counts else None),
                "input_tokens_p95": _percentile(evidence.input_token_counts, 0.95),
                "output_tokens_mean": (fmean(evidence.output_token_counts) if evidence.output_token_counts else None),
                "output_tokens_p95": _percentile(evidence.output_token_counts, 0.95),
                "output_target_hit_rate": (
                    evidence.output_target_hits / evidence.completed
                    if output_token_target is not None
                    and evidence.completed
                    and len(evidence.output_token_counts) == evidence.completed
                    else None
                ),
                "p50_ttft_ms": _percentile(evidence.ttft_ms, 0.50),
                "p95_ttft_ms": _percentile(evidence.ttft_ms, 0.95),
                "p50_tpot_ms": _percentile(evidence.tpot_ms, 0.50),
                "p95_tpot_ms": _percentile(evidence.tpot_ms, 0.95),
            },
            evidence.attempted > 0
            and len(evidence.ttft_ms) == evidence.completed
            and len(evidence.tpot_ms) == evidence.completed,
        )
        for sweep_index, evidence in by_sweep.items()
    }


def _requirement(
    *,
    key: str,
    label: str,
    operator: Literal["gte", "lte"],
    threshold: float | None,
    measured: float | None,
    unit: str,
) -> ServingRequirementView:
    state = "unavailable"
    margin = None
    if threshold is not None and measured is not None:
        margin = measured - threshold if operator == "gte" else threshold - measured
        state = "pass" if margin >= 0 else "fail"
    comparison = "at least" if operator == "gte" else "at most"
    if threshold is None:
        explanation = "This product constraint was not recorded with the run."
    elif measured is None:
        explanation = f"The run did not retain evidence needed to verify {comparison} {threshold:g} {unit}."
    elif state == "pass":
        explanation = f"Measured {measured:g} {unit}; requirement is {comparison} {threshold:g} {unit}."
    else:
        explanation = f"Measured {measured:g} {unit}; this misses the {comparison} {threshold:g} {unit} requirement."
    return ServingRequirementView(
        key=key,
        label=label,
        operator=operator,
        threshold=threshold,
        measured=measured,
        margin=margin,
        unit=unit,
        state=state,
        explanation=explanation,
    )


def _runtime_settings(
    backend: str | None,
    engine: Mapping[str, JsonValue],
    redaction: RedactionPolicy,
) -> tuple[RuntimeSettingGroup, ...]:
    redacted = redaction.mapping(engine)
    consumed: set[str] = set()
    groups: list[RuntimeSettingGroup] = []
    for group_key, label, definitions in _RUNTIME_PRESENTATION:
        settings: list[RuntimeSettingValue] = []
        for key, setting_label, unit, importance in definitions:
            if key not in redacted:
                continue
            consumed.add(key)
            value = redacted[key]
            settings.append(
                RuntimeSettingValue(
                    key=key,
                    label=setting_label,
                    value=value,
                    unit=unit,
                    state="redacted" if value == "[REDACTED]" else "available",
                    importance=cast(Literal["primary", "advanced", "additional"], importance),
                )
            )
        if settings:
            groups.append(RuntimeSettingGroup(key=group_key, label=label, settings=tuple(settings)))
    additional = tuple(
        RuntimeSettingValue(
            key=key,
            label=key.replace("_", " ").title(),
            value=value,
            state="redacted" if value == "[REDACTED]" else "available",
            importance="additional",
        )
        for key, value in sorted(redacted.items())
        if key not in consumed
    )
    if backend is not None:
        groups.insert(
            0,
            RuntimeSettingGroup(
                key="backend",
                label="Backend",
                settings=(
                    RuntimeSettingValue(
                        key="backend",
                        label="Serving backend",
                        value=backend,
                        state="available",
                        importance="primary",
                    ),
                ),
            ),
        )
    if additional:
        groups.append(RuntimeSettingGroup(key="additional", label="Additional backend settings", settings=additional))
    return tuple(groups)


def _eligibility(
    requirements: tuple[ServingRequirementView, ...],
    point: ServingOperatingPoint,
    *,
    point_count: int,
    requirements_digest: str | None,
    saturated: bool,
) -> ServingEligibility:
    failed = {requirement.key for requirement in requirements if requirement.state == "fail"}
    missing = [requirement.label for requirement in requirements if requirement.state == "unavailable"]
    selected = point.sweep_index if point.valid else None
    if "context" in failed:
        state, label, reason = (
            "context_failed",
            "Context failed",
            "The configured runtime did not meet the required context allocation.",
        )
    elif "failure_rate" in failed:
        state, label, reason = (
            "reliability_constrained",
            "Reliability constrained",
            "Request failures exceed the product reliability limit.",
        )
    elif failed & {"p95_ttft", "p95_tpot"}:
        state, label, reason = (
            "latency_constrained",
            "Latency constrained",
            "The measured operating point is fast in aggregate but violates a product latency limit.",
        )
    elif "output_tps" in failed:
        state, label, reason = (
            "below_capacity",
            "Below capacity",
            "The measured operating point does not reach the required sustained output throughput.",
        )
    elif missing or point.evidence_state != "complete":
        state, label, reason = (
            "insufficient_evidence",
            "Insufficient evidence",
            "A decision-grade result needs complete request traces and every configured constraint.",
        )
    elif not saturated:
        if point_count > 1:
            state, label, reason = (
                "unsaturated",
                "Sweep passes; saturation not reached",
                "The selected operating point passes the recorded constraints, but throughput is still improving at the configured concurrency ceiling.",
            )
        else:
            state, label, reason = (
                "unsaturated",
                "Point passes; sweep incomplete",
                "This operating point passes the recorded constraints, but a single point does not prove the hardware saturation boundary.",
            )
    else:
        state, label, reason = (
            "eligible",
            "Eligible",
            "The saturated sweep contains a complete operating point that passes every recorded product constraint.",
        )
    return ServingEligibility(
        state=cast(
            Literal[
                "eligible",
                "below_capacity",
                "latency_constrained",
                "reliability_constrained",
                "context_failed",
                "unsaturated",
                "insufficient_evidence",
            ],
            state,
        ),
        label=label,
        reason=reason,
        requirements_digest=requirements_digest,
        saturation_state="saturated" if saturated else "unsaturated",
        selected_sweep_index=selected,
    )


def _project_operating_point(
    series: Mapping[str, MetricSeries],
    *,
    sweep_index: int,
    request_evidence: Mapping[str, float | int | None],
    complete_latencies: bool,
    requests: Mapping[str, JsonValue],
    serving_requirements: Mapping[str, JsonValue],
) -> tuple[ServingOperatingPoint, tuple[ServingRequirementView, ...]]:
    metric_attempted = int(_at_sweep(series, "serve/requests", sweep_index) or 0)
    metric_completed = int(_at_sweep(series, "serve/requests_measured", sweep_index) or 0)
    metric_failed = int(_at_sweep(series, "serve/requests_failed", sweep_index) or 0)
    metric_unsupported = int(_at_sweep(series, "serve/requests_unsupported", sweep_index) or 0)
    terminal_status: Literal["resource_exhausted", "unsupported", "failed"] | None = None
    if _at_sweep(series, "serve/run/point_resource_exhausted", sweep_index):
        terminal_status = "resource_exhausted"
    elif _at_sweep(series, "serve/run/point_unsupported", sweep_index):
        terminal_status = "unsupported"
    elif _at_sweep(series, "serve/run/point_failed", sweep_index):
        terminal_status = "failed"
    trace_attempted = int(request_evidence["attempted"] or 0)
    attempted = trace_attempted or metric_attempted
    completed = (
        int(request_evidence["completed"] or 0)
        if trace_attempted
        else metric_completed or max(0, metric_attempted - metric_failed - metric_unsupported)
    )
    failed = int(request_evidence["failed"] or 0) if trace_attempted else metric_failed
    output_tokens = (
        int(request_evidence["output_tokens"] or 0)
        if trace_attempted
        else _integer(_at_sweep(series, "serve/output_tokens", sweep_index))
    )
    measurement_seconds = _at_sweep(series, "serve/elapsed_seconds", sweep_index)
    aggregate_output_tps = (
        output_tokens / measurement_seconds
        if output_tokens is not None and measurement_seconds is not None and measurement_seconds > 0
        else _at_sweep(series, "serve/output_token_throughput", sweep_index)
    )
    failure_rate = failed / attempted if attempted else None

    def timing(key: str, metric: str) -> float | None:
        value = cast(float | None, request_evidence[key])
        if value is not None:
            return value
        recorded = _at_sweep(series, metric, sweep_index)
        return recorded * 1000 if recorded is not None else None

    p50_ttft = timing("p50_ttft_ms", "serve/p50_ttft")
    p95_ttft = timing("p95_ttft_ms", "serve/p95_ttft")
    p50_tpot = timing("p50_tpot_ms", "serve/p50_tpot")
    p95_tpot = timing("p95_tpot_ms", "serve/p95_tpot")
    context_tokens = _integer(_at_sweep(series, "serve/context_window", sweep_index)) or _integer(
        requests.get("context_window")
    )
    concurrency = _integer(_at_sweep(series, "serve/concurrency", sweep_index)) or 1
    complete_requests = (
        trace_attempted > 0 and trace_attempted == (metric_attempted or trace_attempted) and complete_latencies
    )
    evidence_state = (
        "complete" if complete_requests else "partial" if terminal_status is not None else "legacy_single_point"
    )
    peak_bytes = _at_sweep(series, "serve/peak_vram_bytes", sweep_index)
    peak_gib = _at_sweep(series, "serve/peak_gpu_memory_gib", sweep_index)
    point = ServingOperatingPoint(
        sweep_index=sweep_index,
        concurrency=concurrency,
        context_tokens=context_tokens,
        attempted_requests=attempted,
        completed_requests=completed,
        failed_requests=failed,
        output_tokens=output_tokens,
        input_tokens_mean=cast(float | None, request_evidence["input_tokens_mean"])
        or _at_sweep(series, "serve/input_tokens_mean", sweep_index),
        input_tokens_p95=cast(float | None, request_evidence["input_tokens_p95"])
        or _at_sweep(series, "serve/input_tokens_p95", sweep_index),
        output_tokens_mean=cast(float | None, request_evidence["output_tokens_mean"])
        or (output_tokens / completed if output_tokens is not None and completed else None),
        output_tokens_p95=cast(float | None, request_evidence["output_tokens_p95"]),
        measurement_seconds=measurement_seconds,
        aggregate_output_tps=aggregate_output_tps,
        failure_rate=failure_rate,
        p50_ttft_ms=p50_ttft,
        p95_ttft_ms=p95_ttft,
        p50_tpot_ms=p50_tpot,
        p95_tpot_ms=p95_tpot,
        peak_vram_bytes=(
            int(peak_bytes) if peak_bytes is not None else int(peak_gib * 1024**3) if peak_gib is not None else None
        ),
        kv_cache_peak_usage_ratio=_at_sweep(
            series,
            "serve/backend/kv_cache_peak_usage_ratio",
            sweep_index,
        ),
        evidence_state=evidence_state,
        terminal_status=terminal_status,
        valid=False,
    )
    requirements = (
        _requirement(
            key="context",
            label="Context allocation",
            operator="gte",
            threshold=_number(serving_requirements.get("required_context_tokens")),
            measured=float(context_tokens) if context_tokens is not None else None,
            unit="tokens",
        ),
        _requirement(
            key="output_tps",
            label="Sustained aggregate output throughput",
            operator="gte",
            threshold=_number(serving_requirements.get("min_sustained_output_tokens_per_second")),
            measured=aggregate_output_tps,
            unit="tokens/s",
        ),
        _requirement(
            key="p95_ttft",
            label="p95 time to first token",
            operator="lte",
            threshold=_number(serving_requirements.get("max_p95_ttft_ms")),
            measured=p95_ttft,
            unit="ms",
        ),
        _requirement(
            key="p95_tpot",
            label="p95 time per output token",
            operator="lte",
            threshold=_number(serving_requirements.get("max_p95_tpot_ms")),
            measured=p95_tpot,
            unit="ms/token",
        ),
        _requirement(
            key="failure_rate",
            label="Request failure rate",
            operator="lte",
            threshold=_number(serving_requirements.get("max_failure_rate")),
            measured=failure_rate,
            unit="ratio",
        ),
    )
    violations = tuple(requirement.key for requirement in requirements if requirement.state != "pass")
    operating_violations = tuple(key for key in violations if key != "output_tps")
    return (
        point.model_copy(
            update={
                "valid": not operating_violations and evidence_state == "complete",
                "violations": violations,
            }
        ),
        requirements,
    )


def _saturation_reached(
    points: tuple[ServingOperatingPoint, ...],
    workload: Mapping[str, JsonValue],
) -> bool:
    if workload.get("saturation_state") == "saturated":
        return True
    seen_valid = False
    for point in points:
        if point.valid:
            seen_valid = True
        elif seen_valid and point.terminal_status in {"resource_exhausted", "unsupported"}:
            return True
        elif seen_valid and point.evidence_state == "complete":
            return True
    valid_tps = [
        point.aggregate_output_tps for point in points if point.valid and point.aggregate_output_tps is not None
    ]
    plateau_intervals = _integer(workload.get("plateau_intervals")) or 2
    plateau_ratio = _number(workload.get("plateau_improvement_ratio")) or 0.05
    if len(valid_tps) < plateau_intervals + 1:
        return False
    recent = valid_tps[-(plateau_intervals + 1) :]
    improvements = [(current - previous) / previous for previous, current in pairwise(recent) if previous > 0]
    return len(improvements) == plateau_intervals and all(improvement <= plateau_ratio for improvement in improvements)


async def project_serving_benchmark(
    locator: RunLocator,
    source: RunDataSource,
    detail: RunDetail,
    redaction: RedactionPolicy,
    *,
    include_request_traces: bool = True,
    include_artifacts: bool = True,
) -> ServingBenchmarkRunView:
    series_values, artifacts = await _gather_serving_evidence(
        source,
        locator.run_id,
        include_artifacts=include_artifacts,
    )
    series = {item.name: item for item in series_values}
    model_id, model = _selection(detail.resolved_inputs, "model")
    inference_id, inference = _selection(detail.resolved_inputs, "screen_inference")
    if not inference:
        inference_id, inference = _selection(detail.resolved_inputs, "inference")
    workload_id, workload = _selection(detail.resolved_inputs, "workload")
    target_id, _target = _selection(detail.resolved_inputs, "target")
    _brief_id, brief = _selection(detail.resolved_inputs, "project_brief")

    serving = brief.get("serving")
    serving_requirements = cast(Mapping[str, JsonValue], serving) if isinstance(serving, Mapping) else {}
    request_spec = workload.get("requests")
    requests = cast(Mapping[str, JsonValue], request_spec) if isinstance(request_spec, Mapping) else {}
    output_token_budget = _integer(requests.get("output_tokens"))
    corpus_spec = requests.get("corpus")
    corpus = cast(Mapping[str, JsonValue], corpus_spec) if isinstance(corpus_spec, Mapping) else {}
    engine_spec = inference.get("engine")
    engine = cast(Mapping[str, JsonValue], engine_spec) if isinstance(engine_spec, Mapping) else {}
    if include_request_traces:
        request_evidence_by_sweep = await _request_evidence(
            source,
            locator.run_id,
            output_token_budget,
        )
    else:
        request_evidence_by_sweep = {}
    empty_request_evidence: dict[str, float | int | None] = {
        "attempted": 0,
        "completed": 0,
        "failed": 0,
        "output_tokens": 0,
        "input_tokens_mean": None,
        "input_tokens_p95": None,
        "output_tokens_mean": None,
        "output_tokens_p95": None,
        "output_target_hit_rate": None,
        "p50_ttft_ms": None,
        "p95_ttft_ms": None,
        "p50_tpot_ms": None,
        "p95_tpot_ms": None,
    }
    concurrency_points = next(
        (
            series[name].points
            for name in _METRIC_ALIASES["serve/concurrency"]
            if name in series and series[name].points
        ),
        (),
    )
    metric_sweep_indices = {
        point.step if point.step is not None else index for index, point in enumerate(concurrency_points)
    }
    sweep_indices = sorted(
        {*metric_sweep_indices, *request_evidence_by_sweep}
        if len(concurrency_points) > 1
        else set(request_evidence_by_sweep) or metric_sweep_indices
    )
    if not sweep_indices:
        sweep_indices = [0]
    points = []
    requirements_by_sweep: dict[int, tuple[ServingRequirementView, ...]] = {}
    for sweep_index in sweep_indices:
        point_evidence, complete_latencies = request_evidence_by_sweep.get(
            sweep_index,
            (empty_request_evidence, False),
        )
        point, point_requirements = _project_operating_point(
            series,
            sweep_index=sweep_index,
            request_evidence=point_evidence,
            complete_latencies=complete_latencies,
            requests=requests,
            serving_requirements=serving_requirements,
        )
        points.append(point)
        requirements_by_sweep[sweep_index] = point_requirements
    valid_points = [point for point in points if point.valid]
    selected_point = max(
        valid_points,
        key=lambda point: point.aggregate_output_tps or 0,
        default=None,
    )
    decision_point = selected_point or max(
        points,
        key=lambda point: point.aggregate_output_tps or 0,
    )
    requirements = requirements_by_sweep[decision_point.sweep_index]
    saturated = _saturation_reached(tuple(points), workload)
    requirements_digest = brief.get("digest")
    digest = requirements_digest if isinstance(requirements_digest, str) else None
    eligibility = _eligibility(
        requirements,
        decision_point,
        point_count=len(points),
        requirements_digest=digest,
        saturated=saturated,
    )
    alerts = (
        ()
        if all(point.evidence_state == "complete" for point in points)
        else (
            RunAlert(
                id="serving-incomplete-request-evidence",
                severity="warning",
                message="Decision-grade serving evidence requires the complete measured inference-trace population.",
                field="trace_count",
            ),
        )
    )
    backend = inference.get("backend")
    backend_name = backend if isinstance(backend, str) else None
    renderer = inference.get("renderer", model.get("renderer"))
    return ServingBenchmarkRunView(
        schema_version=1,
        locator=locator,
        run=detail.summary,
        question="Does this model and serving configuration satisfy the product envelope on the fixed hardware profile?",
        eligibility=eligibility,
        requirements=requirements,
        operating_points=tuple(points),
        selected_point=selected_point,
        model_variant_id=model_id,
        inference_binding_id=inference_id,
        inference_backend=backend_name,
        workload_id=workload_id,
        execution_target_id=target_id,
        runtime_settings=_runtime_settings(backend_name, engine, redaction),
        population=BenchmarkPopulationView(
            cohort=_string(requests.get("cohort")),
            corpus_id=_string(corpus.get("id")),
            corpus_revision=_string(corpus.get("revision")),
            corpus_digest=_string(corpus.get("digest")),
            suite_id=_string(requests.get("suite_id")),
            shape_id=_string(requests.get("shape_id")),
            renderer=renderer if isinstance(renderer, str) else None,
            requested_records=_integer(requests.get("record_count")),
            measured_records=_integer(
                _at_sweep(
                    series,
                    "serve/corpus_records_measured",
                    decision_point.sweep_index,
                )
            )
            or decision_point.attempted_requests,
            input_tokens_mean=decision_point.input_tokens_mean,
            input_tokens_p95=decision_point.input_tokens_p95,
            output_token_budget=output_token_budget,
            output_length_policy="fixed",
            output_target_hit_rate=cast(
                float | None,
                request_evidence_by_sweep.get(
                    decision_point.sweep_index,
                    (empty_request_evidence, False),
                )[0]["output_target_hit_rate"],
            ),
        ),
        alerts=alerts,
        artifacts=artifacts,
        execution_targets=execution_target_contexts(detail.resolved_inputs),
        resolved_inputs=redaction.mapping(detail.resolved_inputs),
        source_metadata=redaction.mapping(detail.source_metadata),
        trace_count=detail.trace_count,
        capabilities=source.capabilities,
    )


async def _gather_serving_evidence(
    source: RunDataSource,
    run_id: str,
    *,
    include_artifacts: bool,
) -> tuple[tuple[MetricSeries, ...], ArtifactSet]:
    # Local import avoids making the pure calculator depend on a concrete
    # tracking provider while still issuing both independent reads together.
    import asyncio

    if include_artifacts:
        series, artifacts = await asyncio.gather(
            source.metric_series(run_id, _SERVING_METRICS),
            source.artifacts(run_id),
        )
        return series, artifacts
    return await source.metric_series(run_id, _SERVING_METRICS), ArtifactSet()


def _seconds_metric_ms(series: Mapping[str, MetricSeries], name: str) -> float | None:
    value = _last(series, name)
    return value * 1000 if value is not None else None


__all__ = ["CALCULATOR_VERSION", "project_serving_benchmark"]
