"""Strict product-facing models shared by Observatory transports and UI."""

from __future__ import annotations

import base64
import json
from datetime import datetime
from typing import Annotated, Literal

from posttrain.common import JsonValue
from posttrain.tracking import (
    ArtifactLink,
    ArtifactSet,
    EventRecord,
    MetricSeries,
    RunStatus,
    RunSummary,
    TrackingCapabilities,
)
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, WithJsonSchema

type EvidenceState = Literal["available", "missing", "incomparable", "not_run", "stale", "reused_from_framework"]
type AlertSeverity = Literal["info", "warning", "error"]
type ViewMode = Literal["auto", "job", "generic"]
type EvidenceRequirementLevel = Literal["required", "conditional", "diagnostic"]
type EvidenceRequirementState = Literal["available", "missing", "not_applicable"]
type EvidenceCompletenessState = Literal["complete", "partial", "insufficient"]
type ServingRequirementState = Literal["pass", "fail", "unavailable"]
type TraceOutcome = Literal["pass", "review", "scored", "error", "truncated", "unknown"]
type ServingEligibilityState = Literal[
    "eligible",
    "below_capacity",
    "latency_constrained",
    "reliability_constrained",
    "context_failed",
    "unsaturated",
    "insufficient_evidence",
]


def _tuple_from_json(value: object) -> object:
    """Accept the JSON array representation while preserving tuple contracts."""

    return tuple(value) if isinstance(value, list) else value


type StringTuple = Annotated[tuple[str, ...], BeforeValidator(_tuple_from_json)]
type JsonPayload = Annotated[
    JsonValue,
    WithJsonSchema({"description": "Any JSON-compatible value."}),
]


class ObservatoryModel(BaseModel):
    """Strict immutable base for product contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RunLocator(ObservatoryModel):
    source_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)

    @property
    def key(self) -> str:
        raw = json.dumps([self.source_id, self.run_id], separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    @classmethod
    def from_key(cls, key: str) -> RunLocator:
        try:
            padded = key + "=" * (-len(key) % 4)
            value = json.loads(base64.urlsafe_b64decode(padded).decode())
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("run key is invalid") from error
        if not isinstance(value, list) or len(value) != 2 or not all(isinstance(item, str) for item in value):
            raise ValueError("run key is invalid")
        return cls(source_id=value[0], run_id=value[1])


class SourceSummary(ObservatoryModel):
    source_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    state: Literal["healthy", "unavailable"]
    message: str | None = None
    capabilities: TrackingCapabilities | None = None


class SourceRefreshStatus(ObservatoryModel):
    enabled: bool
    state: Literal["disabled", "pending", "refreshing", "succeeded", "failed"]
    last_attempt_at: datetime | None = None
    last_success_at: datetime | None = None
    error: str | None = None
    discovered_source_ids: StringTuple = ()


class LocatedRunSummary(ObservatoryModel):
    locator: RunLocator
    run_key: str = Field(min_length=1)
    run: RunSummary
    alert_count: int = Field(default=0, ge=0)


class SummaryValue(ObservatoryModel):
    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    metric: str | None = Field(default=None, min_length=1)
    state: EvidenceState
    value: JsonPayload = None
    unit: str | None = None


class MetricHelp(ObservatoryModel):
    """Human-readable semantics for one provider-neutral metric name."""

    metric: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    interpretation: str = Field(min_length=1)
    caveat: str | None = Field(default=None, min_length=1)
    unit: str | None = None


class SeriesTip(ObservatoryModel):
    metric: str = Field(min_length=1)
    step: int | None = Field(default=None, ge=0)
    value: float


class ChartView(ObservatoryModel):
    key: str = Field(min_length=1)
    title: str = Field(min_length=1)
    question: str | None = Field(default=None, min_length=1)
    series: tuple[MetricSeries, ...]


class EvidenceRequirement(ObservatoryModel):
    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    level: EvidenceRequirementLevel
    state: EvidenceRequirementState
    metrics: StringTuple = Field(min_length=1)
    missing_metrics: StringTuple = ()
    reason: str | None = Field(default=None, min_length=1)


class EvidenceCompleteness(ObservatoryModel):
    state: EvidenceCompletenessState
    research_ready: bool
    required_available: int = Field(ge=0)
    required_total: int = Field(ge=0)
    conditional_available: int = Field(ge=0)
    conditional_active: int = Field(ge=0)
    requirements: tuple[EvidenceRequirement, ...]


class GRPORolloutPopulation(ObservatoryModel):
    requested: SummaryValue
    attempted: SummaryValue
    completed: SummaryValue
    failed: SummaryValue
    truncated: SummaryValue
    unscorable: SummaryValue
    missing: SummaryValue


class GRPOAccelerationEvidence(ObservatoryModel):
    mtp_selected: bool
    quantized_kv_cache_selected: bool
    speculative_acceptance: SummaryValue
    accepted_speculative_length: SummaryValue
    kv_cache_peak_usage: SummaryValue


class GRPOProjection(ObservatoryModel):
    """GRPO-specific evidence consumed identically by HTTP, MCP, Python, and UI."""

    rollout_population: GRPORolloutPopulation
    acceleration: GRPOAccelerationEvidence


class RunAlert(ObservatoryModel):
    id: str = Field(min_length=1)
    severity: AlertSeverity
    message: str = Field(min_length=1)
    field: str | None = None


class MetricNamespace(ObservatoryModel):
    name: str = Field(min_length=1)
    metrics: tuple[str, ...]


class MetricCatalog(ObservatoryModel):
    namespaces: tuple[MetricNamespace, ...]
    total: int = Field(ge=0)


class MetricSeriesQuery(ObservatoryModel):
    names: StringTuple = Field(min_length=1, max_length=12)
    start_step: int | None = Field(default=None, ge=0)
    end_step: int | None = Field(default=None, ge=0)
    max_points: int = Field(default=400, ge=10, le=2000)


class MetricSeriesSet(ObservatoryModel):
    series: tuple[MetricSeries, ...]
    downsampled: bool = False
    requested_points: int = Field(default=0, ge=0)
    returned_points: int = Field(default=0, ge=0)


class ExecutionTargetContext(ObservatoryModel):
    """Immutable hardware context retained with the run's resolved selections."""

    selection_id: str = Field(min_length=1)
    revision: str | None = None
    roles: StringTuple = ()
    device_class: str | None = Field(default=None, min_length=1)
    device_count: int | None = Field(default=None, ge=1)
    memory_bytes_per_device: float | None = Field(default=None, gt=0)
    aggregate_memory_bytes: float | None = Field(default=None, gt=0)
    placement: dict[str, JsonPayload] = Field(default_factory=dict)
    host_constraints: dict[str, JsonPayload] = Field(default_factory=dict)
    state: Literal["complete", "partial"]


class RunView(ObservatoryModel):
    """Registered metric-job projection. Kept as the stable Python name."""

    view_kind: Literal["job.metrics"] = "job.metrics"
    schema_version: int = Field(ge=1)
    locator: RunLocator | None = None
    run: RunSummary
    summary: tuple[SummaryValue, ...]
    charts: tuple[ChartView, ...]
    metric_help: tuple[MetricHelp, ...]
    completeness: EvidenceCompleteness
    grpo: GRPOProjection | None = None
    alerts: tuple[RunAlert, ...]
    artifacts: ArtifactSet
    execution_targets: tuple[ExecutionTargetContext, ...] = ()
    resolved_inputs: dict[str, JsonPayload] = Field(default_factory=dict)
    source_metadata: dict[str, JsonPayload] = Field(default_factory=dict)
    trace_count: int = Field(ge=0)
    trace_evaluation_enabled: bool
    capabilities: TrackingCapabilities


class GenericRunView(ObservatoryModel):
    view_kind: Literal["generic"] = "generic"
    schema_version: int = 1
    locator: RunLocator
    run: RunSummary
    metric_catalog: MetricCatalog
    selected_series: MetricSeriesSet | None = None
    events: tuple[EventRecord, ...]
    artifacts: ArtifactSet
    execution_targets: tuple[ExecutionTargetContext, ...] = ()
    resolved_inputs: dict[str, JsonPayload]
    source_metadata: dict[str, JsonPayload]
    trace_count: int = Field(ge=0)
    trace_evaluation_enabled: bool
    capabilities: TrackingCapabilities


class ServingRequirementView(ObservatoryModel):
    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    operator: Literal["gte", "lte"]
    threshold: float | None = None
    measured: float | None = None
    margin: float | None = None
    unit: str = Field(min_length=1)
    state: ServingRequirementState
    explanation: str = Field(min_length=1)


class ServingOperatingPoint(ObservatoryModel):
    sweep_index: int = Field(ge=0)
    concurrency: int = Field(ge=1)
    context_tokens: int | None = Field(default=None, ge=1)
    attempted_requests: int = Field(ge=0)
    completed_requests: int = Field(ge=0)
    failed_requests: int = Field(ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    input_tokens_mean: float | None = Field(default=None, ge=0)
    input_tokens_p95: float | None = Field(default=None, ge=0)
    output_tokens_mean: float | None = Field(default=None, ge=0)
    output_tokens_p95: float | None = Field(default=None, ge=0)
    measurement_seconds: float | None = Field(default=None, gt=0)
    aggregate_output_tps: float | None = Field(default=None, ge=0)
    failure_rate: float | None = Field(default=None, ge=0, le=1)
    p50_ttft_ms: float | None = Field(default=None, ge=0)
    p95_ttft_ms: float | None = Field(default=None, ge=0)
    p50_tpot_ms: float | None = Field(default=None, ge=0)
    p95_tpot_ms: float | None = Field(default=None, ge=0)
    peak_vram_bytes: float | None = Field(default=None, ge=0)
    kv_cache_peak_usage_ratio: float | None = Field(default=None, ge=0, le=1)
    evidence_state: Literal["complete", "partial", "legacy_single_point"]
    terminal_status: Literal["resource_exhausted", "unsupported", "failed"] | None = None
    valid: bool
    violations: StringTuple = ()


class ServingEligibility(ObservatoryModel):
    state: ServingEligibilityState
    label: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    calculator_version: Literal["serving-capacity-v1"] = "serving-capacity-v1"
    requirements_digest: str | None = Field(default=None, min_length=1)
    saturation_state: Literal["saturated", "unsaturated", "unknown"]
    selected_sweep_index: int | None = Field(default=None, ge=0)


class RuntimeSettingValue(ObservatoryModel):
    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value: JsonPayload = None
    unit: str | None = None
    state: Literal["available", "missing", "redacted"]
    importance: Literal["primary", "advanced", "additional"] = "advanced"


class RuntimeSettingGroup(ObservatoryModel):
    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    settings: tuple[RuntimeSettingValue, ...]


class BenchmarkPopulationView(ObservatoryModel):
    cohort: str | None = Field(default=None, min_length=1)
    corpus_id: str | None = Field(default=None, min_length=1)
    corpus_revision: str | None = Field(default=None, min_length=1)
    corpus_digest: str | None = Field(default=None, min_length=1)
    suite_id: str | None = Field(default=None, min_length=1)
    shape_id: str | None = Field(default=None, min_length=1)
    renderer: str | None = Field(default=None, min_length=1)
    requested_records: int | None = Field(default=None, ge=0)
    measured_records: int | None = Field(default=None, ge=0)
    input_tokens_mean: float | None = Field(default=None, ge=0)
    input_tokens_p95: float | None = Field(default=None, ge=0)
    output_token_budget: int | None = Field(default=None, ge=1)
    output_length_policy: Literal["fixed", "maximum", "unknown"] = "unknown"
    output_target_hit_rate: float | None = Field(default=None, ge=0, le=1)
    correctness_scored: Literal[False] = False


class ServingBenchmarkRunView(ObservatoryModel):
    view_kind: Literal["job.serving"] = "job.serving"
    schema_version: int = Field(ge=1)
    locator: RunLocator
    run: RunSummary
    question: str = Field(min_length=1)
    eligibility: ServingEligibility
    requirements: tuple[ServingRequirementView, ...]
    operating_points: tuple[ServingOperatingPoint, ...]
    selected_point: ServingOperatingPoint | None = None
    model_variant_id: str | None = Field(default=None, min_length=1)
    inference_binding_id: str | None = Field(default=None, min_length=1)
    inference_backend: str | None = Field(default=None, min_length=1)
    workload_id: str | None = Field(default=None, min_length=1)
    execution_target_id: str | None = Field(default=None, min_length=1)
    runtime_settings: tuple[RuntimeSettingGroup, ...]
    population: BenchmarkPopulationView
    alerts: tuple[RunAlert, ...]
    artifacts: ArtifactSet
    execution_targets: tuple[ExecutionTargetContext, ...] = ()
    resolved_inputs: dict[str, JsonPayload] = Field(default_factory=dict)
    source_metadata: dict[str, JsonPayload] = Field(default_factory=dict)
    trace_count: int = Field(ge=0)
    trace_evaluation_enabled: bool = False
    capabilities: TrackingCapabilities


class ServingCapacityRunRow(ObservatoryModel):
    locator: RunLocator
    run_key: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    started_at: datetime
    model_variant_id: str | None = Field(default=None, min_length=1)
    inference_binding_id: str | None = Field(default=None, min_length=1)
    inference_backend: str | None = Field(default=None, min_length=1)
    workload_id: str | None = Field(default=None, min_length=1)
    execution_target_id: str | None = Field(default=None, min_length=1)
    requirements_digest: str | None = Field(default=None, min_length=1)
    point: ServingOperatingPoint
    point_state: Literal["valid", "constraint_failed", "incomplete"]
    point_label: str = Field(min_length=1)
    eligibility: ServingEligibility


class ServingContenderView(ObservatoryModel):
    locator: RunLocator
    run_key: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    started_at: datetime
    model_variant_id: str | None = Field(default=None, min_length=1)
    inference_binding_id: str | None = Field(default=None, min_length=1)
    inference_backend: str | None = Field(default=None, min_length=1)
    workload_id: str | None = Field(default=None, min_length=1)
    corpus_digest: str | None = Field(default=None, min_length=1)
    execution_target_id: str | None = Field(default=None, min_length=1)
    requirements_digest: str | None = Field(default=None, min_length=1)
    calculator_version: str = Field(min_length=1)
    comparable: bool
    comparability_reason: str | None = None
    selected_point: ServingOperatingPoint | None = None
    eligibility: ServingEligibility
    pareto_member: bool = False


class ServingParetoPoint(ObservatoryModel):
    run_key: str = Field(min_length=1)
    model_variant_id: str | None = Field(default=None, min_length=1)
    inference_binding_id: str | None = Field(default=None, min_length=1)
    aggregate_output_tps: float = Field(ge=0)
    p95_ttft_ms: float = Field(ge=0)
    peak_vram_bytes: int = Field(ge=0)


class ServingCapacityWorkPackageView(ObservatoryModel):
    schema_version: int = 1
    project_id: str | None = None
    work_package_id: str = Field(min_length=1)
    methodology: Literal[
        "strict_pareto",
        "single_run_sweep",
        "cross_run_compatibility",
    ] = "cross_run_compatibility"
    explanation: str = Field(min_length=1)
    requirements: tuple[ServingRequirementView, ...] = ()
    execution_target_id: str | None = Field(default=None, min_length=1)
    workload_id: str | None = Field(default=None, min_length=1)
    corpus_digest: str | None = Field(default=None, min_length=1)
    requirements_digest: str | None = Field(default=None, min_length=1)
    calculator_version: str | None = Field(default=None, min_length=1)
    contenders: tuple[ServingContenderView, ...] = ()
    pareto: tuple[ServingParetoPoint, ...] = ()
    rows: tuple[ServingCapacityRunRow, ...]


class RewardComponent(ObservatoryModel):
    name: str = Field(min_length=1)
    value: float


class EvaluationMetricDefinition(ObservatoryModel):
    """The native metric vocabulary advertised by an evaluation."""

    name: str = Field(min_length=1)
    label: str = Field(min_length=1)
    role: Literal["primary_reward", "reward_component", "diagnostic", "success"]


class EvaluationFacetSpec(ObservatoryModel):
    """Resolved environment declaration for one native task-data facet."""

    field: str = Field(min_length=1)
    dimension: str = Field(min_length=1)
    label: str = Field(min_length=1)
    transform: Literal["identity", "prefix_before_colon"] = "identity"


class EvaluationBreakdownSpec(ObservatoryModel):
    """Run-owned compound grouping over declared facet dimensions."""

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    dimensions: tuple[str, str]
    presentation: Literal["matrix"] = "matrix"
    multi_value: Literal["reject", "cross"] = "reject"
    missing: Literal["exclude", "bucket"] = "exclude"


class EvaluationSuccessDefinition(ObservatoryModel):
    """Normalized run-owned predicate used to calculate semantic pass/fail."""

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    namespace: Literal["reward", "metric"]
    signal: str = Field(min_length=1)
    operator: Literal["eq", "gt", "gte", "lt", "lte", "between"]
    value: float
    upper: float | None = None
    tolerance: float = Field(default=0.0, ge=0)
    missing: Literal["error", "exclude"] = "error"


class TaskFacet(ObservatoryModel):
    """One native semantic attribute attached to an evaluation task.

    Environments own the values; Observatory only retains their stable
    dimension/value identity and a human-readable presentation.
    """

    key: str = Field(min_length=1)
    dimension: str = Field(min_length=1)
    dimension_label: str = Field(min_length=1)
    value: str = Field(min_length=1)
    label: str = Field(min_length=1)


class TaskSliceMetadata(ObservatoryModel):
    """Stable, human-readable identity for one evaluation task slice.

    ``key`` remains the source task identity (which may be numeric, as in
    IFEval). Semantic facets are optional and must not be confused with that
    identity.
    """

    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str | None = None
    category: str | None = None
    instruction_ids: tuple[str, ...] = ()
    instruction_families: tuple[str, ...] = ()
    facets: tuple[TaskFacet, ...] = ()
    dataset: str | None = None
    dataset_revision: str | None = None
    split: str | None = None
    seed: int | None = None
    index: int | None = Field(default=None, ge=0)


class EvaluationMetadata(ObservatoryModel):
    """Evaluation identity and metric semantics used to build the Overview."""

    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    category: str | None = None
    package: str | None = None
    dataset: str | None = None
    dataset_revision: str | None = None
    split: str | None = None
    source_revision: str | None = None
    primary_metric: str | None = None
    primary_metric_label: str | None = None
    pass_rate_metric: str | None = None
    pass_rate_basis: str | None = None
    success_definition: EvaluationSuccessDefinition | None = None
    contract_id: str | None = None
    contract_version: int | None = Field(default=None, ge=1)
    contract_state: Literal["versioned", "legacy", "unsupported"] = "legacy"
    facet_specs: tuple[EvaluationFacetSpec, ...] = ()
    breakdown_specs: tuple[EvaluationBreakdownSpec, ...] = ()
    metrics: tuple[EvaluationMetricDefinition, ...] = ()


class TraceSummary(ObservatoryModel):
    external_id: str = Field(min_length=1)
    trace_type: str = Field(min_length=1)
    prompt_preview: str | None = None
    task: str | None = None
    task_label: str | None = None
    task_metadata: TaskSliceMetadata | None = None
    reward: float | None = None
    success: bool | None = None
    outcome: TraceOutcome = "unknown"
    truncated: bool = False
    error: str | None = None
    tool_calls: int | None = Field(default=None, ge=0)
    model_calls: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    tokens: int | None = Field(default=None, ge=0)
    response_tokens: int | None = Field(default=None, ge=0)
    response_chars: int | None = Field(default=None, ge=0)
    thinking_tokens: int | None = Field(default=None, ge=0)
    thinking_chars: int | None = Field(default=None, ge=0)
    reward_components: dict[str, float] = Field(default_factory=dict)
    native_metrics: dict[str, float] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)


class TraceDetail(ObservatoryModel):
    summary: TraceSummary
    reward_components: tuple[RewardComponent, ...] = ()
    transcript: tuple[dict[str, JsonPayload], ...] = ()
    attributes: dict[str, JsonPayload] = Field(default_factory=dict)
    raw: dict[str, JsonPayload] = Field(default_factory=dict)
    projection_warning: str | None = None


class TraceSummaryPage(ObservatoryModel):
    """One bounded page of trace summaries for interactive investigation."""

    items: tuple[TraceSummary, ...] = ()
    next_cursor: str | None = None
    total: int = Field(ge=0)
    live: bool = False


class EvaluationSlice(ObservatoryModel):
    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str | None = None
    metadata: TaskSliceMetadata | None = None
    count: int = Field(ge=0)
    mean_reward: float | None = None
    success_rate: float | None = None


class EvaluationFacet(ObservatoryModel):
    """Multi-label aggregate used to explain a task population by capability."""

    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    dimension: str = Field(min_length=1)
    dimension_label: str = Field(min_length=1)
    count: int = Field(ge=0)
    mean_reward: float | None = None
    success_rate: float | None = None


class EvaluationBreakdownValue(ObservatoryModel):
    """One structured dimension value inside a compound group."""

    dimension: str = Field(min_length=1)
    dimension_label: str = Field(min_length=1)
    value: str = Field(min_length=1)
    label: str = Field(min_length=1)


class EvaluationBreakdownGroup(ObservatoryModel):
    """Coverage-aware aggregate for one compound facet combination."""

    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    values: tuple[EvaluationBreakdownValue, ...]
    count: int = Field(ge=0)
    scored: int = Field(ge=0)
    failures: int = Field(ge=0)
    truncated: int = Field(ge=0)
    mean_reward: float | None = None
    success_rate: float | None = None


class EvaluationBreakdown(ObservatoryModel):
    """One declared compound report with structured groups."""

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    dimensions: tuple[str, str]
    dimension_labels: tuple[str, str]
    presentation: Literal["matrix"] = "matrix"
    groups: tuple[EvaluationBreakdownGroup, ...] = ()
    excluded: int = Field(default=0, ge=0)


class EvaluationDistribution(ObservatoryModel):
    """Observed distribution for one request-level evaluation signal."""

    samples: int = Field(ge=1)
    mean: float = Field(ge=0)
    p50: float = Field(ge=0)
    p95: float = Field(ge=0)
    maximum: float = Field(ge=0)


class EvaluationPerformance(ObservatoryModel):
    """Efficiency evidence computed from the same trace population as quality."""

    latency_ms: EvaluationDistribution | None = None
    completion_tokens: EvaluationDistribution | None = None
    thinking_tokens: EvaluationDistribution | None = None
    tool_calls: EvaluationDistribution | None = None


class TraceEvaluationView(ObservatoryModel):
    state: Literal["complete", "partial", "unavailable"]
    metadata: EvaluationMetadata | None = None
    scanned: int = Field(ge=0)
    expected: int | None = Field(default=None, ge=0)
    included: int = Field(ge=0)
    scored: int = Field(default=0, ge=0)
    mean_reward: float | None = None
    success_rate: float | None = None
    passed: int = Field(default=0, ge=0)
    pass_scored: int = Field(default=0, ge=0)
    failures: int = Field(default=0, ge=0)
    truncated: int = Field(default=0, ge=0)
    slices: tuple[EvaluationSlice, ...] = ()
    facets: tuple[EvaluationFacet, ...] = ()
    breakdowns: tuple[EvaluationBreakdown, ...] = ()
    performance: EvaluationPerformance = Field(default_factory=EvaluationPerformance)
    traces: tuple[TraceSummary, ...] = ()
    next_cursor: str | None = None
    live: bool = False


class SystemMetricSummary(ObservatoryModel):
    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    value: float | None = None
    unit: str | None = None
    state: EvidenceState
    description: str = Field(min_length=1)
    interpretation: str = Field(min_length=1)
    caveat: str | None = Field(default=None, min_length=1)


class SystemMetricGroup(ObservatoryModel):
    key: str = Field(min_length=1)
    title: str = Field(min_length=1)
    series: tuple[MetricSeries, ...]


class PhaseMetricAggregate(ObservatoryModel):
    metric: str = Field(min_length=1)
    label: str = Field(min_length=1)
    unit: str | None = None
    mean: float
    peak: float
    minimum: float
    samples: int = Field(ge=1)


class RuntimePhaseSegment(ObservatoryModel):
    phase: str = Field(min_length=1)
    phase_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    group: str = Field(min_length=1)
    group_label: str = Field(min_length=1)
    status: Literal["running", "completed", "failed", "incomplete", "unclassified"]
    started_at: datetime
    finished_at: datetime
    start_offset_s: float = Field(ge=0)
    end_offset_s: float = Field(ge=0)
    duration_s: float = Field(ge=0)
    sample_count: int = Field(ge=0)
    metrics: tuple[PhaseMetricAggregate, ...] = ()


class RuntimePhaseIntervalView(ObservatoryModel):
    phase: str = Field(min_length=1)
    phase_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    group: str = Field(min_length=1)
    group_label: str = Field(min_length=1)
    status: Literal["running", "completed", "failed", "incomplete"]
    started_at: datetime
    finished_at: datetime
    start_offset_s: float = Field(ge=0)
    end_offset_s: float = Field(ge=0)
    duration_s: float = Field(ge=0)


class RuntimePhaseSummary(ObservatoryModel):
    phase: str = Field(min_length=1)
    label: str = Field(min_length=1)
    group: str = Field(min_length=1)
    group_label: str = Field(min_length=1)
    duration_s: float = Field(ge=0)
    occurrences: int = Field(ge=1)
    sample_count: int = Field(ge=0)
    metrics: tuple[PhaseMetricAggregate, ...] = ()


class InferenceTimingStageSummary(ObservatoryModel):
    stage: Literal["queue", "prefill", "decode", "engine_e2e"]
    label: str = Field(min_length=1)
    samples: int = Field(ge=1)
    mean_ms: float = Field(ge=0)
    p50_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)


class InferenceTimingSummary(ObservatoryModel):
    requests: int = Field(ge=1)
    stages: tuple[InferenceTimingStageSummary, ...]


class BackendRuntimeSummary(ObservatoryModel):
    kv_cache_capacity_tokens: float | None = Field(default=None, gt=0)
    kv_cache_peak_usage_ratio: float | None = Field(default=None, ge=0, le=1)
    kv_cache_samples: int = Field(default=0, ge=0)
    mtp_selected: bool = False
    mtp_acceptance_rate: float | None = Field(default=None, ge=0, le=1)
    mtp_accepted_length: float | None = Field(default=None, ge=0)
    mtp_samples: int = Field(default=0, ge=0)
    rollout_tokens_per_second_latest: float | None = Field(default=None, ge=0)
    rollout_tokens_per_second_mean: float | None = Field(default=None, ge=0)
    rollout_seconds_latest: float | None = Field(default=None, ge=0)
    rollout_seconds_mean: float | None = Field(default=None, ge=0)
    rollout_samples: int = Field(default=0, ge=0)
    environment_concurrency: int | None = Field(default=None, ge=1)
    inference_sequence_cap: int | None = Field(default=None, ge=1)
    rollouts_per_prompt: int | None = Field(default=None, ge=1)
    rollouts_per_update: int | None = Field(default=None, ge=1)


class SystemMetricsView(ObservatoryModel):
    locator: RunLocator
    state: Literal["available", "unavailable"]
    window_started_at: datetime
    window_finished_at: datetime | None = None
    sample_count: int = Field(ge=0)
    summary: tuple[SystemMetricSummary, ...]
    groups: tuple[SystemMetricGroup, ...]
    missing: tuple[str, ...]
    phase_state: Literal["available", "partial", "unavailable"] = "unavailable"
    phase_intervals: tuple[RuntimePhaseIntervalView, ...] = ()
    phase_segments: tuple[RuntimePhaseSegment, ...] = ()
    phase_summary: tuple[RuntimePhaseSummary, ...] = ()
    phase_issues: tuple[str, ...] = ()
    unclassified_sample_count: int = Field(default=0, ge=0)
    execution_targets: tuple[ExecutionTargetContext, ...] = ()
    vram_capacity_state: Literal["available", "ambiguous", "unavailable"] = "unavailable"
    vram_capacity_bytes: float | None = Field(default=None, gt=0)
    vram_observed_peak_bytes: float | None = Field(default=None, gt=0)
    inference_timing: InferenceTimingSummary | None = None
    backend_runtime: BackendRuntimeSummary | None = None
    capabilities: TrackingCapabilities


class EvaluationRunView(ObservatoryModel):
    view_kind: Literal["job.evaluation"] = "job.evaluation"
    schema_version: int = Field(ge=1)
    locator: RunLocator
    run: RunSummary
    summary: tuple[SummaryValue, ...]
    charts: tuple[ChartView, ...]
    metric_help: tuple[MetricHelp, ...]
    completeness: EvidenceCompleteness
    alerts: tuple[RunAlert, ...]
    evaluation: TraceEvaluationView
    comparison_key: str = Field(min_length=1)
    artifacts: ArtifactSet
    execution_targets: tuple[ExecutionTargetContext, ...] = ()
    resolved_inputs: dict[str, JsonPayload] = Field(default_factory=dict)
    source_metadata: dict[str, JsonPayload] = Field(default_factory=dict)
    trace_evaluation_enabled: bool = True
    capabilities: TrackingCapabilities


type RunViewVariant = Annotated[
    RunView | EvaluationRunView | ServingBenchmarkRunView | GenericRunView,
    Field(discriminator="view_kind"),
]


class RunViewResponse(ObservatoryModel):
    requested_mode: ViewMode
    resolved_mode: Literal["job", "generic"]
    fallback_reason: str | None = None
    view: RunViewVariant


class SummaryChange(ObservatoryModel):
    key: str = Field(min_length=1)
    previous: JsonPayload = None
    current: JsonPayload = None
    state: EvidenceState


class RunDelta(ObservatoryModel):
    run_id: str = Field(min_length=1)
    status: RunStatus
    changed_summary: tuple[SummaryChange, ...]
    alerts_added: tuple[RunAlert, ...]
    alerts_removed: tuple[str, ...]
    series_tips: tuple[SeriesTip, ...]
    cursor: str = Field(min_length=1)


class ComparisonRow(ObservatoryModel):
    locator: RunLocator | None = None
    run_id: str = Field(min_length=1)
    values: dict[str, JsonPayload]
    states: dict[str, EvidenceState]
    context: dict[str, JsonPayload] = Field(default_factory=dict)


class RunComparison(ObservatoryModel):
    job_kind: str | None = None
    state: Literal["comparable", "incomparable"]
    columns: tuple[str, ...]
    rows: tuple[ComparisonRow, ...]
    reason: str | None = None
    basis: StringTuple = ()


class WorkPackageRun(ObservatoryModel):
    locator: RunLocator
    run_key: str
    run: RunSummary
    metric_names: tuple[str, ...]
    job_definition_description: str | None = None


class JobDefinitionSummary(ObservatoryModel):
    id: str = Field(min_length=1)
    description: str | None = None


class JobKindGroup(ObservatoryModel):
    job_kind: str
    run_keys: tuple[str, ...]
    statuses: tuple[RunStatus, ...]
    definitions: tuple[JobDefinitionSummary, ...] = ()


class WorkPackageView(ObservatoryModel):
    project_id: str | None
    work_package_id: str
    description: str | None = None
    runs: tuple[WorkPackageRun, ...]
    job_groups: tuple[JobKindGroup, ...]
    lineage: tuple[tuple[RunLocator, ArtifactLink], ...]


class ExportRequest(ObservatoryModel):
    run_keys: StringTuple = ()
    work_package_id: str | None = None
    project_id: str | None = None
    source_id: str | None = None
    view: Literal["runs", "serving_capacity"] = "runs"
    format: Literal["json", "csv"] = "json"


class EvidenceCitation(ObservatoryModel):
    evidence_id: str = Field(min_length=1)


class SemanticEvidenceItem(ObservatoryModel):
    evidence_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value: JsonPayload


class SemanticEvidenceBundle(ObservatoryModel):
    fingerprint: str = Field(min_length=1)
    scope: Literal["run", "metrics", "evaluation", "trace", "comparison"]
    job_kind: str | None = None
    completeness: Literal["complete", "partial", "unknown"] = "unknown"
    items: tuple[SemanticEvidenceItem, ...]


class SemanticClaim(ObservatoryModel):
    kind: Literal["observation", "inference", "hypothesis"]
    text: str = Field(min_length=1)
    citations: tuple[EvidenceCitation, ...] = Field(min_length=1)


class SemanticProvenance(ObservatoryModel):
    provider: str
    model: str
    prompt_version: str
    evidence_fingerprint: str
    generated_at: datetime


class SemanticSummary(ObservatoryModel):
    title: str
    overview: str
    claims: tuple[SemanticClaim, ...]
    limitations: tuple[str, ...] = ()
    provenance: SemanticProvenance


class SemanticSummaryRequest(ObservatoryModel):
    scope: Literal["run", "metrics", "evaluation", "trace"] = "run"
    metric_names: StringTuple = ()
    trace_id: str | None = None


class SemanticSummaryResult(ObservatoryModel):
    status: Literal["disabled", "unavailable", "ready", "stale", "failed"]
    summary: SemanticSummary | None = None
    message: str | None = None


class ErrorResponse(ObservatoryModel):
    code: str
    message: str
    request_id: str


__all__ = [
    "AlertSeverity",
    "BackendRuntimeSummary",
    "BenchmarkPopulationView",
    "ChartView",
    "ComparisonRow",
    "EvaluationBreakdown",
    "EvaluationBreakdownGroup",
    "EvaluationBreakdownSpec",
    "EvaluationBreakdownValue",
    "EvaluationMetadata",
    "EvaluationMetricDefinition",
    "ErrorResponse",
    "EvaluationRunView",
    "EvaluationSlice",
    "EvaluationFacet",
    "EvidenceCitation",
    "EvidenceCompleteness",
    "EvidenceCompletenessState",
    "EvidenceRequirement",
    "EvidenceRequirementLevel",
    "EvidenceRequirementState",
    "EvidenceState",
    "ExportRequest",
    "GenericRunView",
    "GRPOAccelerationEvidence",
    "GRPOProjection",
    "GRPORolloutPopulation",
    "JobDefinitionSummary",
    "JobKindGroup",
    "InferenceTimingStageSummary",
    "InferenceTimingSummary",
    "LocatedRunSummary",
    "MetricCatalog",
    "MetricNamespace",
    "MetricSeriesQuery",
    "MetricSeriesSet",
    "ObservatoryModel",
    "RewardComponent",
    "RuntimeSettingGroup",
    "RuntimeSettingValue",
    "RunAlert",
    "RunComparison",
    "RunDelta",
    "RunLocator",
    "RunView",
    "RunViewResponse",
    "RunViewVariant",
    "SemanticClaim",
    "SemanticEvidenceBundle",
    "SemanticEvidenceItem",
    "SemanticProvenance",
    "SemanticSummary",
    "SemanticSummaryRequest",
    "SemanticSummaryResult",
    "SeriesTip",
    "ServingBenchmarkRunView",
    "ServingCapacityRunRow",
    "ServingCapacityWorkPackageView",
    "ServingEligibility",
    "ServingEligibilityState",
    "ServingOperatingPoint",
    "ServingRequirementState",
    "ServingRequirementView",
    "SourceSummary",
    "SourceRefreshStatus",
    "SummaryChange",
    "SummaryValue",
    "TaskSliceMetadata",
    "SystemMetricGroup",
    "SystemMetricSummary",
    "SystemMetricsView",
    "TraceDetail",
    "TraceEvaluationView",
    "TraceSummary",
    "TraceSummaryPage",
    "ViewMode",
    "WorkPackageRun",
    "WorkPackageView",
]
