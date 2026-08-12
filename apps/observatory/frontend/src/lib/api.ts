export type RunItem = {
  locator: { source_id: string; run_id: string };
  run_key: string;
  alert_count: number;
  run: {
    run_id: string;
    display_name: string;
    project_id: string;
    work_package_id: string;
    stage: string;
    job_kind: string;
    job_definition_version: string;
    status: string;
    provider: string;
    started_at: string;
    finished_at: string | null;
  };
};

export type MetricPoint = {
  value: number;
  step: number | null;
  observed_at?: string | null;
  attributes?: Record<string, unknown>;
};
export type MetricSeries = { name: string; points: MetricPoint[] };

export type ExecutionTargetContext = {
  selection_id: string;
  revision: string | null;
  roles: string[];
  device_class: string | null;
  device_count: number | null;
  memory_bytes_per_device: number | null;
  aggregate_memory_bytes: number | null;
  placement: Record<string, unknown>;
  host_constraints: Record<string, unknown>;
  state: 'complete' | 'partial';
};
export type SummaryMetric = {
  key: string;
  label: string;
  metric: string | null;
  state: string;
  value: unknown;
  unit: string | null;
};

export type MetricHelp = {
  metric: string;
  label: string;
  description: string;
  interpretation: string;
  caveat: string | null;
  unit: string | null;
};

export type ServingRequirement = {
  key: string;
  label: string;
  operator: 'gte' | 'lte';
  threshold: number | null;
  measured: number | null;
  margin: number | null;
  unit: string;
  state: 'pass' | 'fail' | 'unavailable';
  explanation: string;
};

export type ServingOperatingPoint = {
  sweep_index: number;
  concurrency: number;
  context_tokens: number | null;
  attempted_requests: number;
  completed_requests: number;
  failed_requests: number;
  output_tokens: number | null;
  input_tokens_mean: number | null;
  input_tokens_p95: number | null;
  output_tokens_mean: number | null;
  output_tokens_p95: number | null;
  measurement_seconds: number | null;
  aggregate_output_tps: number | null;
  failure_rate: number | null;
  p50_ttft_ms: number | null;
  p95_ttft_ms: number | null;
  p50_tpot_ms: number | null;
  p95_tpot_ms: number | null;
  peak_vram_bytes: number | null;
  kv_cache_peak_usage_ratio: number | null;
  evidence_state: 'complete' | 'partial' | 'legacy_single_point';
  terminal_status: 'resource_exhausted' | 'unsupported' | 'failed' | null;
  valid: boolean;
  violations: string[];
};

export type RuntimeSettingGroup = {
  key: string;
  label: string;
  settings: Array<{
    key: string;
    label: string;
    value: unknown;
    unit: string | null;
    state: 'available' | 'missing' | 'redacted';
    importance: 'primary' | 'advanced' | 'additional';
  }>;
};

export type ServingCapacityEligibility = {
  state: 'eligible' | 'below_capacity' | 'latency_constrained' | 'reliability_constrained' | 'context_failed' | 'unsaturated' | 'insufficient_evidence';
  label: string;
  reason: string;
  calculator_version: 'serving-capacity-v1';
  requirements_digest: string | null;
  saturation_state: 'saturated' | 'unsaturated' | 'unknown';
  selected_sweep_index: number | null;
};

export type ServingCapacityWorkPackage = {
  schema_version: number;
  project_id: string | null;
  work_package_id: string;
  methodology: 'strict_pareto' | 'single_run_sweep' | 'cross_run_compatibility';
  explanation: string;
  requirements: ServingRequirement[];
  execution_target_id: string | null;
  workload_id: string | null;
  corpus_digest: string | null;
  requirements_digest: string | null;
  calculator_version: string | null;
  contenders: Array<{
    locator: RunItem['locator'];
    run_key: string;
    display_name: string;
    started_at: string;
    model_variant_id: string | null;
    inference_binding_id: string | null;
    inference_backend: string | null;
    workload_id: string | null;
    corpus_digest: string | null;
    execution_target_id: string | null;
    requirements_digest: string | null;
    calculator_version: string;
    comparable: boolean;
    comparability_reason: string | null;
    selected_point: ServingOperatingPoint | null;
    eligibility: ServingCapacityEligibility;
    pareto_member: boolean;
  }>;
  pareto: Array<{
    run_key: string;
    model_variant_id: string | null;
    inference_binding_id: string | null;
    aggregate_output_tps: number;
    p95_ttft_ms: number;
    peak_vram_bytes: number;
  }>;
  rows: Array<{
    locator: RunItem['locator'];
    run_key: string;
    display_name: string;
    started_at: string;
    model_variant_id: string | null;
    inference_binding_id: string | null;
    inference_backend: string | null;
    workload_id: string | null;
    execution_target_id: string | null;
    requirements_digest: string | null;
    point: ServingOperatingPoint;
    point_state: 'valid' | 'constraint_failed' | 'incomplete';
    point_label: string;
    eligibility: ServingCapacityEligibility;
  }>;
};

export type Artifact = {
  direction: string;
  logical_name: string;
  kind: string;
  artifact: {
    provider: string;
    namespace: string;
    name: string;
    version: string;
    digest: string | null;
    provider_metadata: Record<string, unknown>;
  };
};

export type TraceSummary = {
  external_id: string;
  trace_type: string;
  prompt_preview: string | null;
  task: string | null;
  task_label: string | null;
  task_metadata: {
    key: string;
    label: string;
    description: string | null;
    category: string | null;
    instruction_ids: string[];
    instruction_families: string[];
    facets: Array<{
      key: string;
      dimension: string;
      dimension_label: string;
      value: string;
      label: string;
    }>;
    dataset: string | null;
    dataset_revision: string | null;
    split: string | null;
    seed: number | null;
    index: number | null;
  } | null;
  reward: number | null;
  success: boolean | null;
  outcome: 'pass' | 'review' | 'scored' | 'error' | 'truncated' | 'unknown';
  truncated: boolean;
  error: string | null;
  tool_calls: number | null;
  model_calls: number | null;
  input_tokens: number | null;
  completion_tokens: number | null;
  latency_ms: number | null;
  tokens: number | null;
  response_tokens: number | null;
  response_chars: number | null;
  thinking_tokens: number | null;
  thinking_chars: number | null;
  reward_components: Record<string, number>;
  native_metrics: Record<string, number>;
  metrics: Record<string, number>;
};

export type TraceEvaluation = {
  state: 'complete' | 'partial' | 'unavailable';
  metadata: {
    key: string;
    label: string;
    category: string | null;
    package: string | null;
    dataset: string | null;
    dataset_revision: string | null;
    split: string | null;
    source_revision: string | null;
    primary_metric: string | null;
    primary_metric_label: string | null;
    pass_rate_metric: string | null;
    pass_rate_basis: string | null;
    success_definition: {
      id: string;
      label: string;
      namespace: 'reward' | 'metric';
      signal: string;
      operator: 'eq' | 'gt' | 'gte' | 'lt' | 'lte' | 'between';
      value: number;
      upper: number | null;
      tolerance: number;
      missing: 'error' | 'exclude';
    } | null;
    facet_specs: Array<{
      field: string;
      dimension: string;
      label: string;
      transform: 'identity' | 'prefix_before_colon';
    }>;
    breakdown_specs: Array<{
      id: string;
      label: string;
      dimensions: [string, string];
      presentation: 'matrix';
      multi_value: 'reject' | 'cross';
      missing: 'exclude' | 'bucket';
    }>;
    metrics: Array<{ name: string; label: string; role: 'primary_reward' | 'reward_component' | 'diagnostic' | 'success' }>;
  } | null;
  scanned: number;
  expected: number | null;
  included: number;
  scored: number;
  mean_reward: number | null;
  success_rate: number | null;
  passed?: number;
  pass_scored?: number;
  failures: number;
  truncated: number;
  slices: Array<{
    key: string;
    label: string;
    description: string | null;
    metadata: TraceSummary['task_metadata'];
    count: number;
    mean_reward: number | null;
    success_rate: number | null;
  }>;
  facets: Array<{
    key: string;
    label: string;
    dimension: string;
    dimension_label: string;
    count: number;
    mean_reward: number | null;
    success_rate: number | null;
  }>;
  breakdowns: Array<{
    id: string;
    label: string;
    dimensions: [string, string];
    dimension_labels: [string, string];
    presentation: 'matrix';
    groups: Array<{
      key: string;
      label: string;
      values: Array<{
        dimension: string;
        dimension_label: string;
        value: string;
        label: string;
      }>;
      count: number;
      scored: number;
      failures: number;
      truncated: number;
      mean_reward: number | null;
      success_rate: number | null;
    }>;
    excluded: number;
  }>;
  performance?: {
    latency_ms: EvaluationDistribution | null;
    completion_tokens: EvaluationDistribution | null;
    thinking_tokens: EvaluationDistribution | null;
    tool_calls: EvaluationDistribution | null;
  };
  traces: TraceSummary[];
  next_cursor: string | null;
  live: boolean;
};

export type TraceSummaryPage = {
  items: TraceSummary[];
  next_cursor: string | null;
  total: number;
  live: boolean;
};

export type EvaluationDistribution = {
  samples: number;
  mean: number;
  p50: number;
  p95: number;
  maximum: number;
};

export type TraceDetail = {
  summary: TraceSummary;
  reward_components: Array<{ name: string; value: number }>;
  transcript: Array<Record<string, unknown>>;
  attributes: Record<string, unknown>;
  raw: Record<string, unknown>;
  projection_warning: string | null;
};

export type RunComparison = {
  job_kind: string | null;
  state: 'comparable' | 'incomparable';
  columns: string[];
  rows: Array<{
    locator: RunItem['locator'] | null;
    run_id: string;
    values: Record<string, unknown>;
    states: Record<string, string>;
    context: Record<string, unknown>;
  }>;
  reason: string | null;
  basis: string[];
};

export type RunComparisonKey = { job_kind: string | null; comparison_key: string | null };

export type DistillationPairing = {
  studentModel: string;
  teacherModel: string;
  scoredTokens: number | null;
  teacherLatencyMs: number | null;
  teacherFailures: number | null;
  batchIds: string[];
};

export type RunView = {
  requested_mode: 'auto' | 'job' | 'generic';
  resolved_mode: 'job' | 'generic';
  fallback_reason: string | null;
  view: {
    schema_version?: number;
    view_kind: 'job.metrics' | 'job.evaluation' | 'job.serving' | 'generic';
    run: RunItem['run'];
    summary?: SummaryMetric[];
    charts?: Array<{ key: string; title: string; question: string | null; series: MetricSeries[] }>;
    metric_help?: MetricHelp[];
    completeness?: {
      state: 'complete' | 'partial' | 'insufficient';
      research_ready: boolean;
      required_available: number;
      required_total: number;
      conditional_available: number;
      conditional_active: number;
      requirements: Array<{
        key: string;
        label: string;
        level: 'required' | 'conditional' | 'diagnostic';
        state: 'available' | 'missing' | 'not_applicable';
        metrics: string[];
        missing_metrics: string[];
        reason: string | null;
      }>;
    };
    grpo?: {
      rollout_population: {
        requested: SummaryMetric;
        attempted: SummaryMetric;
        completed: SummaryMetric;
        failed: SummaryMetric;
        truncated: SummaryMetric;
        unscorable: SummaryMetric;
        missing: SummaryMetric;
      };
      acceleration: {
        mtp_selected: boolean;
        quantized_kv_cache_selected: boolean;
        speculative_acceptance: SummaryMetric;
        accepted_speculative_length: SummaryMetric;
        kv_cache_peak_usage: SummaryMetric;
      };
    } | null;
    alerts?: Array<{ id: string; severity: string; message: string; field: string | null }>;
    metric_catalog?: { namespaces: Array<{ name: string; metrics: string[] }>; total: number };
    selected_series?: {
      series: MetricSeries[];
      downsampled: boolean;
      requested_points: number;
      returned_points: number;
    } | null;
    evaluation?: TraceEvaluation;
    comparison_key?: string;
    question?: string;
    eligibility?: {
      state: 'eligible' | 'below_capacity' | 'latency_constrained' | 'reliability_constrained' | 'context_failed' | 'unsaturated' | 'insufficient_evidence';
      label: string;
      reason: string;
      calculator_version: 'serving-capacity-v1';
      requirements_digest: string | null;
      saturation_state: 'saturated' | 'unsaturated' | 'unknown';
      selected_sweep_index: number | null;
    };
    requirements?: ServingRequirement[];
    operating_points?: ServingOperatingPoint[];
    selected_point?: ServingOperatingPoint | null;
    model_variant_id?: string | null;
    inference_binding_id?: string | null;
    inference_backend?: string | null;
    workload_id?: string | null;
    execution_target_id?: string | null;
    runtime_settings?: RuntimeSettingGroup[];
    population?: {
      cohort: string | null;
      corpus_id: string | null;
      corpus_revision: string | null;
      corpus_digest: string | null;
      suite_id: string | null;
      shape_id: string | null;
      renderer: string | null;
      requested_records: number | null;
      measured_records: number | null;
      input_tokens_mean: number | null;
      input_tokens_p95: number | null;
      output_token_budget: number | null;
      output_length_policy: 'fixed' | 'maximum' | 'unknown';
      output_target_hit_rate: number | null;
      correctness_scored: false;
    };
    artifacts: { items: Artifact[] };
    execution_targets?: ExecutionTargetContext[];
    resolved_inputs?: Record<string, unknown>;
    source_metadata?: Record<string, unknown>;
    events?: Array<{ name: string; occurred_at: string; attributes: Record<string, unknown> }>;
    trace_count?: number;
    trace_evaluation_enabled: boolean;
  };
};

export type WorkPackageView = {
  project_id: string | null;
  work_package_id: string;
  description: string | null;
  runs: Array<{
    locator: RunItem['locator'];
    run_key: string;
    run: RunItem['run'];
    metric_names: string[];
    job_definition_description: string | null;
  }>;
  job_groups: Array<{
    job_kind: string;
    run_keys: string[];
    statuses: string[];
    definitions: Array<{ id: string; description: string | null }>;
  }>;
  lineage: Array<[RunItem['locator'], Artifact]>;
};

export type SourceRefreshStatus = {
  enabled: boolean;
  state: 'disabled' | 'pending' | 'refreshing' | 'succeeded' | 'failed';
  last_attempt_at: string | null;
  last_success_at: string | null;
  error: string | null;
  discovered_source_ids: string[];
};

export type SystemMetrics = {
  state: 'available' | 'unavailable';
  window_started_at: string;
  window_finished_at: string | null;
  sample_count: number;
  summary: Array<{
    key: string;
    label: string;
    metric: string;
    value: number | null;
    unit: string | null;
    state: string;
    description: string;
    interpretation: string;
    caveat: string | null;
  }>;
  groups: Array<{ key: string; title: string; series: MetricSeries[] }>;
  missing: string[];
  phase_state: 'available' | 'partial' | 'unavailable';
  phase_intervals: RuntimePhaseInterval[];
  phase_segments: RuntimePhaseSegment[];
  phase_summary: RuntimePhaseSummary[];
  phase_issues: string[];
  unclassified_sample_count: number;
  execution_targets: ExecutionTargetContext[];
  vram_capacity_state: 'available' | 'ambiguous' | 'unavailable';
  vram_capacity_bytes: number | null;
  vram_observed_peak_bytes: number | null;
  inference_timing: {
    requests: number;
    stages: Array<{
      stage: 'queue' | 'prefill' | 'decode' | 'engine_e2e';
      label: string;
      samples: number;
      mean_ms: number;
      p50_ms: number;
      p95_ms: number;
    }>;
  } | null;
  backend_runtime: {
    kv_cache_capacity_tokens: number | null;
    kv_cache_peak_usage_ratio: number | null;
    kv_cache_samples: number;
    mtp_selected: boolean;
    mtp_acceptance_rate: number | null;
    mtp_accepted_length: number | null;
    mtp_samples: number;
    rollout_tokens_per_second_latest: number | null;
    rollout_tokens_per_second_mean: number | null;
    rollout_seconds_latest: number | null;
    rollout_seconds_mean: number | null;
    rollout_samples: number;
    environment_concurrency: number | null;
    inference_sequence_cap: number | null;
    rollouts_per_prompt: number | null;
    rollouts_per_update: number | null;
  } | null;
};

export type PhaseMetricAggregate = {
  metric: string;
  label: string;
  unit: string | null;
  mean: number;
  peak: number;
  minimum: number;
  samples: number;
};

export type RuntimePhaseSegment = {
  phase: string;
  phase_id: string;
  label: string;
  group: string;
  group_label: string;
  status: 'running' | 'completed' | 'failed' | 'incomplete' | 'unclassified';
  started_at: string;
  finished_at: string;
  start_offset_s: number;
  end_offset_s: number;
  duration_s: number;
  sample_count: number;
  metrics: PhaseMetricAggregate[];
};

export type RuntimePhaseInterval = {
  phase: string;
  phase_id: string;
  label: string;
  group: string;
  group_label: string;
  status: 'running' | 'completed' | 'failed' | 'incomplete';
  started_at: string;
  finished_at: string;
  start_offset_s: number;
  end_offset_s: number;
  duration_s: number;
};

export type RuntimePhaseSummary = {
  phase: string;
  label: string;
  group: string;
  group_label: string;
  duration_s: number;
  occurrences: number;
  sample_count: number;
  metrics: PhaseMetricAggregate[];
};

export type SemanticResult = {
  status: string;
  message?: string;
  summary?: {
    title: string;
    overview: string;
    claims: Array<{
      kind: string;
      text: string;
      citations: Array<{ evidence_id: string }>;
    }>;
    limitations: string[];
    provenance: {
      provider: string;
      model: string;
      prompt_version: string;
      generated_at: string;
    };
  };
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as {
      message?: string;
      detail?: string;
    };
    throw new Error(body.message ?? body.detail ?? `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  runs: () => request<RunItem[]>('/api/v1/runs'),
  refreshSources: () => request<SourceRefreshStatus>('/api/v1/sources/refresh', { method: 'POST' }),
  workPackage: (workPackageId: string, projectId: string, sourceId: string) => {
    const path = workPackageId.split('/').map(encodeURIComponent).join('/');
    const query = new URLSearchParams({ project_id: projectId, source_id: sourceId });
    return request<WorkPackageView>(`/api/v1/work-packages/${path}?${query}`);
  },
  servingCapacity: (workPackageId: string, projectId: string, sourceId: string) => {
    const path = workPackageId.split('/').map(encodeURIComponent).join('/');
    const query = new URLSearchParams({ project_id: projectId, source_id: sourceId });
    return request<ServingCapacityWorkPackage>(`/api/v1/serving-capacity/work-packages/${path}?${query}`);
  },
  view: (key: string, mode = 'auto', metrics: string[] = []) => {
    const query = new URLSearchParams({ mode });
    metrics.forEach((metric) => query.append('metric', metric));
    return request<RunView>(`/api/v1/runs/${key}/view?${query}`);
  },
  comparisonKey: (key: string) => request<RunComparisonKey>(`/api/v1/runs/${key}/comparison-key`),
  system: (key: string) => request<SystemMetrics>(`/api/v1/runs/${key}/system-metrics`),
  evaluation: (key: string, includeTraces = true) => {
    const query = new URLSearchParams({ include_traces: String(includeTraces) });
    return request<TraceEvaluation>(`/api/v1/runs/${key}/traces-evaluation?${query}`);
  },
  tracePage: (key: string, cursor: string | null = null, limit = 100) => {
    const query = new URLSearchParams({ limit: String(limit) });
    if (cursor) query.set('cursor', cursor);
    return request<TraceSummaryPage>(`/api/v1/runs/${key}/traces?${query}`);
  },
  trace: (key: string, traceId: string) =>
    request<TraceDetail>(`/api/v1/runs/${key}/traces/${encodeURIComponent(traceId)}`),
  compare: (runKeys: string[]) =>
    request<RunComparison>('/api/v1/runs/compare', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ run_keys: runKeys }),
    }),
  summarize: (key: string) =>
    request<SemanticResult>(`/api/v1/runs/${key}/semantic-summary`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ scope: 'run', metric_names: [], trace_id: null }),
    }),
};
