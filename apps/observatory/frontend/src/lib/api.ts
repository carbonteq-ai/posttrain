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
  task: string | null;
  reward: number | null;
  success: boolean | null;
  truncated: boolean;
  error: string | null;
  tool_calls: number | null;
  latency_ms: number | null;
  tokens: number | null;
};

export type TraceEvaluation = {
  state: 'complete' | 'partial' | 'unavailable';
  scanned: number;
  expected: number | null;
  included: number;
  mean_reward: number | null;
  success_rate: number | null;
  failures: number;
  truncated: number;
  slices: Array<{
    key: string;
    count: number;
    mean_reward: number | null;
    success_rate: number | null;
  }>;
  traces: TraceSummary[];
  next_cursor: string | null;
  live: boolean;
};

export type TraceDetail = {
  summary: TraceSummary;
  reward_components: Array<{ name: string; value: number }>;
  transcript: Array<Record<string, unknown>>;
  attributes: Record<string, unknown>;
  raw: Record<string, unknown>;
  projection_warning: string | null;
};

export type RunView = {
  requested_mode: 'auto' | 'job' | 'generic';
  resolved_mode: 'job' | 'generic';
  fallback_reason: string | null;
  view: {
    schema_version?: number;
    view_kind: 'job.metrics' | 'job.evaluation' | 'generic';
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
        attempted: SummaryMetric;
        completed: SummaryMetric;
        failed: SummaryMetric;
        truncated: SummaryMetric;
        unscorable: SummaryMetric;
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
  workPackage: (workPackageId: string, projectId: string, sourceId: string) => {
    const path = workPackageId.split('/').map(encodeURIComponent).join('/');
    const query = new URLSearchParams({ project_id: projectId, source_id: sourceId });
    return request<WorkPackageView>(`/api/v1/work-packages/${path}?${query}`);
  },
  view: (key: string, mode = 'auto', metrics: string[] = []) => {
    const query = new URLSearchParams({ mode });
    metrics.forEach((metric) => query.append('metric', metric));
    return request<RunView>(`/api/v1/runs/${key}/view?${query}`);
  },
  system: (key: string) => request<SystemMetrics>(`/api/v1/runs/${key}/system-metrics`),
  evaluation: (key: string) =>
    request<TraceEvaluation>(`/api/v1/runs/${key}/traces-evaluation`),
  trace: (key: string, traceId: string) =>
    request<TraceDetail>(`/api/v1/runs/${key}/traces/${encodeURIComponent(traceId)}`),
  summarize: (key: string) =>
    request<SemanticResult>(`/api/v1/runs/${key}/semantic-summary`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ scope: 'run', metric_names: [], trace_id: null }),
    }),
};
