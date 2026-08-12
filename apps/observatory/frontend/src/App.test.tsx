import { act, cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./components/EvidenceChart', () => ({
  EvidenceChart: ({ ariaLabel, showLegend = true, hoveredStep, onHoverStep, xDomain, xOrigin, xRange, series }: {
    ariaLabel: string;
    showLegend?: boolean;
    hoveredStep?: number | null;
    onHoverStep?: (step: number | null) => void;
    xDomain?: string;
    xOrigin?: string;
    xRange?: readonly [number, number];
    series: Array<{ points: unknown[] }>;
  }) => <div
    role="img"
    aria-label={ariaLabel}
    data-show-legend={String(showLegend)}
    data-hovered-step={hoveredStep ?? ''}
    data-x-domain={xDomain ?? 'logical-step'}
    data-x-origin={xOrigin ?? ''}
    data-x-min={xRange?.[0] ?? ''}
    data-x-max={xRange?.[1] ?? ''}
    data-point-count={Math.max(0, ...series.map((item) => item.points.length))}
    onMouseEnter={() => onHoverStep?.(1)}
    onMouseLeave={() => onHoverStep?.(null)}
  />,
}));
vi.mock('./components/EvaluationCharts', () => ({
  EvaluationCharts: () => <div role="img" aria-label="Evaluation population charts" />,
}));

import App, { limitSystemSeriesToRecentWindow } from './App';

const run = {
  locator: { source_id: 'fixture', run_id: 'runs/sft' },
  run_key: 'run-key',
  alert_count: 0,
  run: {
    run_id: 'runs/sft',
    display_name: 'SFT calm harbor',
    project_id: 'projects/demo',
    work_package_id: 'train/demo',
    stage: 'train',
    job_kind: 'train.sft',
    job_definition_version: 'train.sft@1',
    status: 'succeeded',
    provider: 'fixture',
    started_at: '2026-07-22T04:00:00Z',
    finished_at: '2026-07-22T04:12:00Z',
  },
};

const view = {
  requested_mode: 'auto',
  resolved_mode: 'job',
  fallback_reason: null,
  view: {
    view_kind: 'job.metrics',
    run: run.run,
    summary: [{ key: 'final_loss', label: 'Final loss', metric: 'train/loss', state: 'available', value: 0.39, unit: null }],
    charts: [{ key: 'loss', title: 'Optimization', series: [{ name: 'train/loss', points: [{ value: 0.39, step: 10 }] }] }],
    metric_help: [{
      metric: 'train/loss',
      label: 'Training loss',
      description: 'Mean supervised objective on the current optimization batch.',
      interpretation: 'A sustained decrease means the model is fitting the rendered training targets.',
      caveat: 'Compare it with held-out loss before judging generalization.',
      unit: null,
    }],
    alerts: [],
    artifacts: { items: [
      {
        direction: 'input',
        logical_name: 'models/base@v1',
        kind: 'model-weights',
        artifact: {
          provider: 'fixture',
          namespace: 'demo',
          name: 'base-model',
          version: 'v1',
          digest: 'input-digest',
          provider_metadata: { run_id: 'runs/producer', job_kind: 'train.sft' },
        },
      },
      {
        direction: 'output',
        logical_name: 'models/sft@v2',
        kind: 'model-adapter',
        artifact: {
          provider: 'fixture',
          namespace: 'demo',
          name: 'sft-adapter',
          version: 'v2',
          digest: 'output-digest',
          provider_metadata: { run_id: 'runs/sft', job_kind: 'train.sft' },
        },
      },
    ] },
    execution_targets: [{
      selection_id: 'targets/cuda-8gb',
      revision: '1',
      roles: ['training'],
      device_class: 'nvidia-cuda',
      device_count: 1,
      memory_bytes_per_device: 8 * 1024 ** 3,
      aggregate_memory_bytes: 8 * 1024 ** 3,
      placement: { world_size: 1 },
      host_constraints: {},
      state: 'complete',
    }],
    resolved_inputs: {
      model: {
        selection_id: 'models/qwen-demo',
        revision: 'model-revision-1234567890',
        source_layer: 'catalog',
        resolved: { family: 'qwen', form: 'adapter', weight_precision: 'bf16' },
      },
      dataset: {
        selection_id: 'datasets/sft-train',
        revision: 'train-v1',
        source_layer: 'project',
        resolved: { kind: 'sft', num_examples: 120, metadata: { split: 'train', partition: 'training' } },
      },
      validation_dataset: {
        selection_id: 'datasets/sft-validation',
        revision: 'validation-v1',
        source_layer: 'project',
        resolved: { kind: 'sft', num_examples: 20, metadata: { split: 'train', partition: 'validation' } },
      },
      settings: {
        selection_id: 'settings/sft',
        revision: 'v4',
        resolved: { max_steps: 60, learning_rate: 0.0001, use_liger_kernel: true },
      },
      training: {
        selection_id: 'training/qlora',
        revision: 'v1',
        resolved: { backend: 'trl', parameter_update_kind: 'qlora', parameter_update: { kind: 'qlora', rank: 16, alpha: 32, quant_type: 'nf4' }, runtime: { global_batch_size: 8 } },
      },
      job_definition: { id: 'train.sft@1', kind: 'train.sft', description: 'Fit supervised demonstrations with validation-aware SFT.' },
      work_package: { project_id: 'projects/demo', work_package_id: 'train/demo', stage: 'train', description: 'Establish a supervised baseline for the demonstration task.' },
    },
    source_metadata: {
      git_revision: '0123456789abcdef',
      git_dirty: false,
    },
    trace_count: 0,
    trace_evaluation_enabled: false,
  },
};

const workPackageView = {
  project_id: 'projects/demo',
  work_package_id: 'train/demo',
  description: 'Establish a supervised baseline for the demonstration task.',
  runs: [{ locator: run.locator, run_key: run.run_key, run: run.run, metric_names: ['train/loss'], job_definition_description: 'Fit supervised demonstrations with validation-aware SFT.' }],
  job_groups: [{ job_kind: 'train.sft', run_keys: [run.run_key], statuses: ['succeeded'], definitions: [{ id: 'train.sft@1', description: 'Fit supervised demonstrations with validation-aware SFT.' }] }],
  lineage: view.view.artifacts.items.map((artifact) => [run.locator, artifact]),
};

const dpoRun = {
  ...run,
  run: {
    ...run.run,
    run_id: 'runs/dpo',
    display_name: 'DPO amber field',
    work_package_id: 'train/dpo-demo',
    job_kind: 'train.dpo',
    job_definition_version: 'train.dpo@1',
  },
};

const dpoView = {
  requested_mode: 'auto',
  resolved_mode: 'job',
  fallback_reason: null,
  view: {
    view_kind: 'job.metrics',
    run: dpoRun.run,
    summary: [
      { key: 'reward_margin', label: 'Reward margin', metric: 'train/rewards/margins', state: 'available', value: 1.4, unit: null },
      { key: 'preference_accuracy', label: 'Pair ordering accuracy', metric: 'train/rewards/accuracies', state: 'available', value: 0.75, unit: 'ratio' },
      { key: 'chosen_reward', label: 'Chosen reward', metric: 'train/rewards/chosen', state: 'available', value: 0.8, unit: null },
      { key: 'rejected_reward', label: 'Rejected reward', metric: 'train/rewards/rejected', state: 'available', value: -0.6, unit: null },
      { key: 'chosen_logp', label: 'Chosen log probability', metric: 'train/logps/chosen', state: 'available', value: -12, unit: null },
      { key: 'grad_norm', label: 'Gradient norm', metric: 'train/grad_norm', state: 'available', value: 0.9, unit: null },
      { key: 'entropy', label: 'Token entropy', metric: 'train/entropy', state: 'available', value: 0.5, unit: null },
    ],
    charts: [{
      key: 'preferences',
      title: 'Pair ordering',
      question: 'Is the policy consistently ranking chosen completions above rejected ones?',
      series: [{ name: 'train/rewards/margins', points: [{ value: 1.4, step: 10 }] }],
    }],
    metric_help: [],
    completeness: {
      state: 'partial',
      research_ready: false,
      required_available: 5,
      required_total: 6,
      conditional_available: 0,
      conditional_active: 1,
      requirements: [{
        key: 'runtime',
        label: 'Effective runtime',
        level: 'required',
        state: 'missing',
        metrics: ['train/non_padding_tokens_per_second'],
        missing_metrics: ['train/non_padding_tokens_per_second'],
        reason: 'Missing train/non_padding_tokens_per_second. Provides effective throughput evidence.',
      }],
    },
    alerts: [],
    artifacts: { items: [] },
    resolved_inputs: {
      dataset: { selection_id: 'dataset/preferences', revision: 'v1', resolved: { num_examples: 128 } },
      settings: { selection_id: 'settings/dpo', revision: 'v1', resolved: { beta: 0.1 } },
      training: { selection_id: 'training/qlora', revision: 'v1', resolved: { parameter_update_kind: 'qlora' } },
    },
    trace_count: 0,
    trace_evaluation_enabled: false,
  },
};

const genericMetricNames = ['train/loss', 'train/learning_rate', 'train/grad_norm', 'train/entropy'];

function genericView(selected: string[] = []) {
  return {
    requested_mode: 'generic',
    resolved_mode: 'generic',
    fallback_reason: null,
    view: {
      view_kind: 'generic',
      run: run.run,
      metric_catalog: { namespaces: [{ name: 'train', metrics: genericMetricNames }], total: genericMetricNames.length },
      selected_series: selected.length ? {
        series: selected.map((name, index) => ({ name, points: [{ value: index + 0.25, step: 1 }, { value: index + 0.5, step: 2 }] })),
        downsampled: false,
        requested_points: selected.length * 2,
        returned_points: selected.length * 2,
      } : null,
      events: [],
      artifacts: { items: [] },
      resolved_inputs: {},
      source_metadata: {},
      trace_count: 0,
      trace_evaluation_enabled: false,
    },
  };
}

function metricJob(
  jobKind: string,
  displayName: string,
  summary: Array<{ key: string; label: string; metric: string; value: number; unit: string | null }>,
  resolvedInputs: Record<string, unknown>,
  traceAware = false,
) {
  const jobRun = {
    ...run,
    locator: { source_id: 'fixture', run_id: `runs/${jobKind.replace('.', '-')}` },
    run_key: `${jobKind}-key`,
    run: {
      ...run.run,
      run_id: `runs/${jobKind.replace('.', '-')}`,
      display_name: displayName,
      work_package_id: `${jobKind.split('.')[0]}/${jobKind.replace('.', '-')}`,
      job_kind: jobKind,
      job_definition_version: `${jobKind}@1`,
    },
  };
  const jobView = {
    requested_mode: 'auto',
    resolved_mode: 'job',
    fallback_reason: null,
    view: {
      view_kind: 'job.metrics',
      run: jobRun.run,
      summary: summary.map((item) => ({ ...item, state: 'available' })),
      charts: [{
        key: 'evidence',
        title: 'Recorded evidence',
        question: 'What did this job record?',
        series: summary.map((item) => ({ name: item.metric, points: [{ value: item.value, step: 1 }] })),
      }],
      metric_help: [],
      completeness: {
        state: 'complete',
        research_ready: traceAware,
        required_available: summary.length,
        required_total: summary.length,
        conditional_available: 0,
        conditional_active: 0,
        requirements: [],
      },
      alerts: [],
      artifacts: { items: [] },
      execution_targets: [],
      resolved_inputs: resolvedInputs,
      source_metadata: {},
      trace_count: traceAware ? 2 : 0,
      trace_evaluation_enabled: traceAware,
    },
  };
  return { jobRun, jobView };
}

const servingRun = {
  ...run,
  locator: { source_id: 'fixture', run_id: 'runs/serve' },
  run_key: 'serve-key',
  run: {
    ...run.run,
    run_id: 'runs/serve',
    display_name: 'Serving capacity point',
    work_package_id: 'screen/serving-capacity',
    stage: 'screen',
    job_kind: 'serve.benchmark',
    job_definition_version: 'serve.benchmark@1',
  },
};

const servingView = {
  requested_mode: 'auto',
  resolved_mode: 'job',
  fallback_reason: null,
  view: {
    view_kind: 'job.serving',
    schema_version: 1,
    run: servingRun.run,
    question: 'Does this model and serving configuration satisfy the product envelope on the fixed hardware profile?',
    eligibility: {
      state: 'unsaturated',
      label: 'Point passes; sweep incomplete',
      reason: 'This operating point passes the recorded constraints, but a single point does not prove the hardware saturation boundary.',
      calculator_version: 'serving-capacity-v1',
      requirements_digest: 'sha256:brief',
      saturation_state: 'unsaturated',
      selected_sweep_index: 0,
    },
    requirements: [
      { key: 'context', label: 'Context allocation', operator: 'gte', threshold: 32768, measured: 32768, margin: 0, unit: 'tokens', state: 'pass', explanation: 'Context passes.' },
      { key: 'output_tps', label: 'Sustained aggregate output throughput', operator: 'gte', threshold: 50, measured: 58, margin: 8, unit: 'tokens/s', state: 'pass', explanation: 'Throughput passes.' },
      { key: 'p95_ttft', label: 'p95 time to first token', operator: 'lte', threshold: 1000, measured: 700, margin: 300, unit: 'ms', state: 'pass', explanation: 'TTFT passes.' },
      { key: 'p95_tpot', label: 'p95 time per output token', operator: 'lte', threshold: 30, measured: 24, margin: 6, unit: 'ms/token', state: 'pass', explanation: 'TPOT passes.' },
      { key: 'failure_rate', label: 'Request failure rate', operator: 'lte', threshold: 0.01, measured: 0, margin: 0.01, unit: 'ratio', state: 'pass', explanation: 'Reliability passes.' },
    ],
    operating_points: [{
      sweep_index: 0,
      concurrency: 4,
      context_tokens: 32768,
      attempted_requests: 128,
      completed_requests: 128,
      failed_requests: 0,
      output_tokens: 16384,
      input_tokens_mean: 612,
      input_tokens_p95: 944,
      output_tokens_mean: 128,
      output_tokens_p95: 128,
      measurement_seconds: 282.48,
      aggregate_output_tps: 58,
      failure_rate: 0,
      p50_ttft_ms: 420,
      p95_ttft_ms: 700,
      p50_tpot_ms: 18,
      p95_tpot_ms: 24,
      peak_vram_bytes: 7.2 * 1024 ** 3,
      kv_cache_peak_usage_ratio: 0.78,
      evidence_state: 'complete',
      valid: true,
      violations: [],
    }],
    selected_point: {
      sweep_index: 0,
      concurrency: 4,
      context_tokens: 32768,
      attempted_requests: 128,
      completed_requests: 128,
      failed_requests: 0,
      output_tokens: 16384,
      input_tokens_mean: 612,
      input_tokens_p95: 944,
      output_tokens_mean: 128,
      output_tokens_p95: 128,
      measurement_seconds: 282.48,
      aggregate_output_tps: 58,
      failure_rate: 0,
      p50_ttft_ms: 420,
      p95_ttft_ms: 700,
      p50_tpot_ms: 18,
      p95_tpot_ms: 24,
      peak_vram_bytes: 7.2 * 1024 ** 3,
      kv_cache_peak_usage_ratio: 0.78,
      evidence_state: 'complete',
      valid: true,
      violations: [],
    },
    model_variant_id: 'models/qwen3.5-0.8b@bf16',
    inference_binding_id: 'inference/qwen-vllm@1',
    inference_backend: 'vllm@0.25.1',
    workload_id: 'workloads/representative@1',
    execution_target_id: 'targets/cuda-8gb',
    runtime_settings: [{
      key: 'scheduler',
      label: 'Scheduler & batching',
      settings: [{ key: 'max_num_seqs', label: 'Maximum sequences', value: 8, unit: null, state: 'available', importance: 'primary' }],
    }],
    population: {
      cohort: 'representative',
      corpus_id: 'general-serving-v1',
      corpus_revision: '1',
      corpus_digest: 'sha256:corpus',
      suite_id: 'general-serving-v1',
      shape_id: 'representative-128out',
      renderer: 'qwen3.5-tools@1',
      requested_records: 128,
      measured_records: 128,
      input_tokens_mean: 612,
      input_tokens_p95: 944,
      output_token_budget: 128,
      output_length_policy: 'fixed',
      output_target_hit_rate: 1,
      correctness_scored: false,
    },
    alerts: [],
    artifacts: { items: [] },
    execution_targets: [{
      selection_id: 'targets/cuda-8gb',
      revision: '1',
      roles: ['screen_inference'],
      device_class: 'nvidia-cuda',
      device_count: 1,
      memory_bytes_per_device: 8 * 1024 ** 3,
      aggregate_memory_bytes: 8 * 1024 ** 3,
      placement: { world_size: 1 },
      host_constraints: {},
      state: 'complete',
    }],
    resolved_inputs: {},
    source_metadata: {},
    trace_count: 128,
    trace_evaluation_enabled: false,
  },
};
const secondServingPoint = {
  ...servingView.view.selected_point,
  sweep_index: 1,
  concurrency: 8,
  output_tokens: 16384,
  measurement_seconds: 182.04,
  aggregate_output_tps: 90,
  p50_ttft_ms: 510,
  p95_ttft_ms: 850,
  p50_tpot_ms: 19,
  p95_tpot_ms: 25,
};
servingView.view.operating_points.push(secondServingPoint);
servingView.view.selected_point = secondServingPoint;
servingView.view.eligibility.label = 'Sweep passes; saturation not reached';
servingView.view.eligibility.reason = 'The selected operating point passes the recorded constraints, but throughput is still improving at the configured concurrency ceiling.';
servingView.view.eligibility.selected_sweep_index = 1;

describe('Observatory React product shell', () => {
  afterEach(cleanup);

  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      const body = path === '/api/v1/runs'
        ? [run]
        : path.startsWith('/api/v1/work-packages/')
          ? workPackageView
        : path.includes('/system-metrics')
          ? {
              state: 'available',
              window_started_at: '2026-07-22T04:00:00Z',
              window_finished_at: '2026-07-22T04:12:00Z',
              sample_count: 6,
              summary: [{ key: 'gpu_utilization', label: 'GPU utilization', metric: 'system/gpu_utilization', value: 77, unit: '%', state: 'available' }],
              groups: [
                { key: 'compute', title: 'Compute utilization', series: [{ name: 'system/gpu_utilization', points: [{ value: 77, step: 10, observed_at: '2026-07-22T04:08:00Z' }] }] },
                { key: 'memory', title: 'Memory pressure', series: [{ name: 'system/gpu_vram_used_bytes', points: [{ value: 5_000_000_000, step: 10, observed_at: '2026-07-22T04:08:00Z' }] }] },
              ],
              missing: [],
              phase_state: 'available',
              phase_intervals: [
                {
                  phase: 'operation',
                  phase_id: 'operation-1',
                  label: 'Operation',
                  group: 'run',
                  group_label: 'Run',
                  status: 'completed',
                  started_at: '2026-07-22T04:00:00Z',
                  finished_at: '2026-07-22T04:12:00Z',
                  start_offset_s: 0,
                  end_offset_s: 720,
                  duration_s: 720,
                },
                {
                  phase: 'model_loading',
                  phase_id: 'model-loading-1',
                  label: 'Model loading',
                  group: 'startup',
                  group_label: 'Startup',
                  status: 'completed',
                  started_at: '2026-07-22T04:00:00Z',
                  finished_at: '2026-07-22T04:02:00Z',
                  start_offset_s: 0,
                  end_offset_s: 120,
                  duration_s: 120,
                },
                {
                  phase: 'actor_update',
                  phase_id: 'actor-update-1',
                  label: 'Actor update',
                  group: 'training',
                  group_label: 'Training',
                  status: 'completed',
                  started_at: '2026-07-22T04:02:00Z',
                  finished_at: '2026-07-22T04:10:00Z',
                  start_offset_s: 120,
                  end_offset_s: 600,
                  duration_s: 480,
                },
              ],
              phase_segments: [
                {
                  phase: 'model_loading',
                  phase_id: 'model-loading-1',
                  label: 'Model loading',
                  group: 'startup',
                  group_label: 'Startup',
                  status: 'completed',
                  started_at: '2026-07-22T04:00:00Z',
                  finished_at: '2026-07-22T04:02:00Z',
                  start_offset_s: 0,
                  end_offset_s: 120,
                  duration_s: 120,
                  sample_count: 1,
                  metrics: [{ metric: 'system/gpu_vram_used_bytes', label: 'GPU memory', unit: 'bytes', mean: 3_000_000_000, peak: 3_000_000_000, minimum: 3_000_000_000, samples: 1 }],
                },
                {
                  phase: 'actor_update',
                  phase_id: 'actor-update-1',
                  label: 'Actor update',
                  group: 'training',
                  group_label: 'Training',
                  status: 'completed',
                  started_at: '2026-07-22T04:02:00Z',
                  finished_at: '2026-07-22T04:10:00Z',
                  start_offset_s: 120,
                  end_offset_s: 600,
                  duration_s: 480,
                  sample_count: 4,
                  metrics: [{ metric: 'system/gpu_vram_used_bytes', label: 'GPU memory', unit: 'bytes', mean: 5_000_000_000, peak: 5_500_000_000, minimum: 4_500_000_000, samples: 4 }],
                },
              ],
              phase_summary: [
                {
                  phase: 'model_loading',
                  label: 'Model loading',
                  group: 'startup',
                  group_label: 'Startup',
                  duration_s: 120,
                  occurrences: 1,
                  sample_count: 1,
                  metrics: [{ metric: 'system/gpu_vram_used_bytes', label: 'GPU memory', unit: 'bytes', mean: 3_000_000_000, peak: 3_000_000_000, minimum: 3_000_000_000, samples: 1 }],
                },
                {
                  phase: 'actor_update',
                  label: 'Actor update',
                  group: 'training',
                  group_label: 'Training',
                  duration_s: 480,
                  occurrences: 1,
                  sample_count: 4,
                  metrics: [{ metric: 'system/gpu_vram_used_bytes', label: 'GPU memory', unit: 'bytes', mean: 5_000_000_000, peak: 5_500_000_000, minimum: 4_500_000_000, samples: 4 }],
                },
                {
                  phase: 'rollout',
                  label: 'Rollout generation',
                  group: 'training',
                  group_label: 'Training',
                  duration_s: 180,
                  occurrences: 3,
                  sample_count: 6,
                  metrics: [
                    { metric: 'system/gpu_utilization', label: 'GPU utilization', unit: '%', mean: 29, peak: 78, minimum: 0, samples: 6 },
                    { metric: 'system/gpu_vram_used_bytes', label: 'GPU memory', unit: 'bytes', mean: 7_000_000_000, peak: 8_200_000_000, minimum: 6_000_000_000, samples: 6 },
                  ],
                },
              ],
              phase_issues: [],
              unclassified_sample_count: 0,
              execution_targets: view.view.execution_targets,
              vram_capacity_state: 'available',
              vram_capacity_bytes: 8 * 1024 ** 3,
              vram_observed_peak_bytes: 5_500_000_000,
              backend_runtime: {
                kv_cache_capacity_tokens: 483_328,
                kv_cache_peak_usage_ratio: 0.3,
                kv_cache_samples: 6,
                mtp_selected: true,
                mtp_acceptance_rate: 0.74,
                mtp_accepted_length: 1.74,
                mtp_samples: 6,
                rollout_tokens_per_second_latest: 115,
                rollout_tokens_per_second_mean: 152,
                rollout_seconds_latest: 708,
                rollout_seconds_mean: 549,
                rollout_samples: 6,
                environment_concurrency: 64,
                inference_sequence_cap: 64,
                rollouts_per_prompt: 4,
                rollouts_per_update: 128,
              },
            }
          : view;
      return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
    }));
  });

  it('limits system chart series to the latest hour without changing the source series', () => {
    const series = [{
      name: 'system/gpu_utilization',
      points: [
        { value: 20, step: 0, observed_at: '2026-08-07T10:00:00Z' },
        { value: 40, step: 1, observed_at: '2026-08-07T11:30:00Z' },
        { value: 60, step: 2, observed_at: '2026-08-07T12:30:00Z' },
      ],
    }];

    const limited = limitSystemSeriesToRecentWindow(series, Date.parse('2026-08-07T12:30:00Z'));

    expect(limited[0].points.map((point) => point.value)).toEqual([40, 60]);
    expect(series[0].points).toHaveLength(3);
  });

  it('presents a useful job question and a dedicated system metrics view', async () => {
    const user = userEvent.setup();
    render(<App />);
    expect(await screen.findByRole('heading', { name: 'Learning & data evidence' })).toBeVisible();
    expect(document.querySelector('[data-theme="light"]')).toBeInTheDocument();
    expect(screen.getByText(/Is held-out loss improving/)).toBeVisible();
    expect(screen.getByText('Fit supervised demonstrations with validation-aware SFT.')).toBeVisible();
    expect(screen.queryByText('Curated view')).not.toBeInTheDocument();
    expect(screen.queryByText('fixture evidence')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Generate analysis' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Traces & evaluation' })).not.toBeInTheDocument();
    const method = screen.getByRole('region', { name: 'Algorithm settings' });
    expect(within(method).getByText('QLORA')).toBeVisible();
    expect(method).toHaveTextContent('Adapter rank16');
    expect(method).toHaveTextContent('Training data120 examples');
    expect(method).toHaveTextContent('Learning rate1.00e-4');
    expect(screen.getByText('targets/cuda-8gb')).toBeVisible();
    expect(screen.getByRole('region', { name: 'Input artifacts' })).toHaveTextContent('models/base@v1');
    const producedEvidence = screen.getByRole('region', { name: 'Produced evidence' });
    expect(producedEvidence).toHaveTextContent('models/sft@v2');
    expect(producedEvidence).not.toHaveTextContent('models/base@v1');
    await user.click(screen.getByRole('button', { name: 'System metrics' }));
    expect(await screen.findByRole('heading', { name: 'System metrics' })).toBeVisible();
    await waitFor(() => expect(screen.getByText('77%')).toBeVisible());
    const telemetryWindow = screen.getByRole('region', { name: 'Run telemetry window' });
    expect(telemetryWindow).toHaveTextContent('Started');
    expect(telemetryWindow).toHaveTextContent('Finished');
    expect(telemetryWindow).toHaveTextContent('Run samples6');
    expect(telemetryWindow).not.toHaveClass('obs-card');
    const chartRange = screen.getByRole('region', { name: 'System chart range' });
    expect(chartRange).toHaveTextContent('Latest observed hour · 1 of 6 run samples');
    expect(within(chartRange).getByRole('button', { name: 'Latest 1h' })).toHaveAttribute('aria-pressed', 'true');
    const windowedComputeChart = screen.getByRole('img', { name: 'Compute utilization over run time' });
    expect(windowedComputeChart).toHaveAttribute('data-x-min', '0');
    expect(windowedComputeChart).toHaveAttribute('data-x-max', '480');
    await user.click(within(chartRange).getByRole('button', { name: 'Full run' }));
    expect(within(chartRange).getByRole('button', { name: 'Full run' })).toHaveAttribute('aria-pressed', 'true');
    expect(chartRange).toHaveTextContent('Full run · 1 samples');
    expect(windowedComputeChart).toHaveAttribute('data-x-min', '0');
    expect(windowedComputeChart).toHaveAttribute('data-x-max', '720');
    const phaseProfile = screen.getByRole('region', { name: 'Runtime phase profile' });
    expect(phaseProfile).toBeVisible();
    expect(screen.getByRole('img', { name: 'Phase GPU memory against declared hardware capacity' })).toBeVisible();
    expect(within(phaseProfile).getByText(/1 × nvidia-cuda · 8.00 GiB per device/)).toBeVisible();
    expect(within(phaseProfile).getAllByText('Actor update').length).toBeGreaterThan(0);
    expect(within(phaseProfile).getByRole('region', { name: 'Startup phases' })).toBeVisible();
    expect(within(phaseProfile).getByRole('region', { name: 'Training phases' })).toBeVisible();
    const inferenceDetails = screen.getByRole('region', { name: 'Inference details' });
    expect(inferenceDetails).toHaveTextContent('Rollout compute');
    expect(inferenceDetails).toHaveTextContent('29.0% average');
    expect(inferenceDetails).toHaveTextContent('MTP acceleration');
    expect(inferenceDetails).toHaveTextContent('74.0% acceptance');
    expect(inferenceDetails).toHaveTextContent('6 step samples');
    expect(inferenceDetails).toHaveTextContent('Environment concurrency64');
    expect(inferenceDetails).toHaveTextContent('vLLM sequence cap64');
    expect(inferenceDetails).toHaveTextContent('Rollouts / update128');
    await user.click(within(phaseProfile).getByRole('tab', { name: 'timeline' }));
    expect(screen.getByRole('img', { name: 'Runtime phase and GPU utilization timeline' })).toBeVisible();
    const computeChart = screen.getByRole('region', { name: 'Compute utilization system chart' });
    const computeHeader = within(computeChart).getByRole('heading', { name: 'Compute utilization' }).parentElement;
    expect(computeHeader).toHaveTextContent('GPU utilization');
    expect(within(computeChart).getByRole('img', { name: 'Compute utilization over run time' }))
      .toHaveAttribute('data-x-domain', 'elapsed-time');
    expect(within(computeChart).getByRole('img', { name: 'Compute utilization over run time' }))
      .toHaveAttribute('data-x-origin', '2026-07-22T04:00:00Z');
  });

  it('explains serving constraints instead of falling back to generic evidence', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      const body = path === '/api/v1/runs'
        ? [servingRun]
        : path.includes('/api/v1/serving-capacity/work-packages/')
          ? {
              schema_version: 1,
              project_id: servingRun.run.project_id,
              work_package_id: servingRun.run.work_package_id,
              methodology: 'cross_run_compatibility',
              explanation: 'Compatibility projection across historical single-point runs.',
              rows: [{
                locator: servingRun.locator,
                run_key: servingRun.run_key,
                display_name: servingRun.run.display_name,
                started_at: servingRun.run.started_at,
                model_variant_id: servingView.view.model_variant_id,
                inference_binding_id: servingView.view.inference_binding_id,
                inference_backend: servingView.view.inference_backend,
                workload_id: servingView.view.workload_id,
                execution_target_id: servingView.view.execution_target_id,
                requirements_digest: servingView.view.eligibility.requirements_digest,
                point: servingView.view.selected_point,
                point_state: 'valid',
                point_label: 'Valid point',
                eligibility: servingView.view.eligibility,
              }],
            }
          : servingView;
      return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
    }));

    render(<App />);

    expect(await screen.findByRole('heading', { name: 'Serving benchmark' })).toBeVisible();
    expect(screen.getByRole('region', { name: 'Serving benchmark decision' })).toHaveTextContent('Sweep passes; saturation not reached');
    expect(screen.getByRole('heading', { name: 'Product constraints' })).toBeVisible();
    expect(screen.getByText('≥ 50 tokens/s')).toBeVisible();
    expect(screen.getByText('58 tokens/s')).toBeVisible();
    expect(await screen.findByRole('heading', { name: 'Concurrency points' })).toBeVisible();
    expect(screen.getByRole('heading', { name: 'Throughput and latency under concurrency' })).toBeVisible();
    expect(screen.getByRole('img', { name: 'Aggregate output tokens per second by concurrency' })).toBeVisible();
    expect(screen.getByRole('img', { name: 'p95 time to first token versus aggregate output tokens per second' })).toBeVisible();
    expect(screen.getByRole('table')).toHaveTextContent('Mean output');
    expect(screen.getByRole('table')).toHaveTextContent('128 tok');
    expect(screen.getByText('Mean response length')).toBeVisible();
    expect(screen.getByText('Fixed-length systems run:', { exact: false })).toBeVisible();
    expect(screen.getByText('Output target hit')).toBeVisible();
    expect(screen.getByRole('heading', { name: 'Serving configuration' })).toBeVisible();
    expect(screen.getByText('Maximum sequences')).toBeVisible();
    expect(screen.getByText('Capacity only; task correctness was not scored.')).toBeVisible();
    expect(screen.queryByText('Generic evidence workspace')).not.toBeInTheDocument();
    expect(screen.queryByText(/No job view is registered/)).not.toBeInTheDocument();
  });

  it('shows resolved run inputs without misrepresenting them as artifact edges', async () => {
    const user = userEvent.setup();
    render(<App />);
    expect(await screen.findByRole('heading', { name: 'Learning & data evidence' })).toBeVisible();
    await user.click(screen.getByRole('button', { name: 'Artifacts & lineage' }));
    expect(await screen.findByRole('heading', { name: 'Run inputs and outputs' })).toBeVisible();
    const runInputs = screen.getByRole('region', { name: 'Run inputs' });
    expect(runInputs).toHaveTextContent('Base model');
    expect(runInputs).toHaveTextContent('models/qwen-demo');
    expect(runInputs).toHaveTextContent('Training dataset');
    expect(runInputs).toHaveTextContent('datasets/sft-train');
    expect(runInputs).toHaveTextContent('Validation dataset');
    expect(runInputs).toHaveTextContent('datasets/sft-validation');
    expect(runInputs).toHaveTextContent('Execution target');
    expect(runInputs).toHaveTextContent('targets/cuda-8gb');
    expect(runInputs).toHaveTextContent('Recorded input artifacts');
    expect(runInputs).toHaveTextContent('models/base@v1');
    const observedRun = screen.getByRole('article', { name: 'Run SFT calm harbor' });
    expect(observedRun).toBeVisible();
    expect(within(observedRun).getByText('succeeded')).toHaveClass('text-[11px]');
    expect(screen.getByRole('region', { name: 'Produced artifacts' })).toHaveTextContent('models/sft@v2');
    expect(screen.getByRole('heading', { name: 'Artifact ledger' })).toBeVisible();
  });

  it('uses one active project and expands runs beneath their work package', async () => {
    const user = userEvent.setup();
    render(<App />);
    expect(await screen.findByRole('heading', { name: 'Learning & data evidence' })).toBeVisible();
    expect(screen.getByRole('button', { name: 'Project: demo' })).toBeVisible();
    expect(screen.queryByRole('button', { name: 'Projects' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Open work package train/demo' })).toBeVisible();
    expect(screen.getByRole('button', { name: 'Select run SFT calm harbor' })).toBeVisible();
    expect(document.querySelector('time[datetime="2026-07-22T04:00:00Z"]')).toBeVisible();

    await user.click(screen.getByRole('button', { name: 'Open work package train/demo' }));
    expect(await screen.findByRole('heading', { name: 'Run groups' })).toBeVisible();
    expect(screen.getByText('Establish a supervised baseline for the demonstration task.')).toBeVisible();
    expect(screen.getByText('Fit supervised demonstrations with validation-aware SFT.')).toBeVisible();
    expect(screen.getByRole('region', { name: 'Work package summary' })).toHaveTextContent('1 succeeded');
    expect(screen.getByRole('heading', { name: 'Artifact lineage' })).toBeVisible();
    expect(screen.getByRole('region', { name: 'Decision record' })).toHaveTextContent('No package conclusion recorded');
  });

  it('orders sidebar evidence by workflow stage and then by run recency', async () => {
    const sidebarRun = (
      runId: string,
      displayName: string,
      workPackageId: string,
      stage: string,
      startedAt: string,
    ) => ({
      ...run,
      locator: { source_id: 'fixture', run_id: runId },
      run_key: `${runId}-key`,
      run: {
        ...run.run,
        run_id: runId,
        display_name: displayName,
        work_package_id: workPackageId,
        stage,
        started_at: startedAt,
      },
    });
    const trainOlder = { ...run, run: { ...run.run, display_name: 'Train older' } };
    const items = [
      sidebarRun('runs/qualify', 'Qualify latest', 'qualify/demo', 'qualify', '2026-07-22T09:00:00Z'),
      trainOlder,
      sidebarRun('runs/screen', 'Screen earliest', 'screen/demo', 'screen', '2026-07-22T03:00:00Z'),
      sidebarRun('runs/train-new', 'Train newer', 'train/demo', 'train', '2026-07-22T06:00:00Z'),
      sidebarRun('runs/train-package-new', 'Other package run', 'train/other', 'train', '2026-07-22T07:00:00Z'),
    ];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const body = String(input) === '/api/v1/runs' ? items : view;
      return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
    }));

    render(<App />);
    expect(await screen.findByRole('heading', { name: 'Learning & data evidence' })).toBeVisible();

    const stageGroups = screen.getAllByRole('region', { name: /^(screen|train|qualify) work packages$/ });
    expect(stageGroups.map((element) => element.getAttribute('aria-label'))).toEqual([
      'screen work packages',
      'train work packages',
      'qualify work packages',
    ]);
    const trainStage = screen.getByRole('region', { name: 'train work packages' });
    expect(within(trainStage).getAllByRole('button', { name: /^Open work package/ }).map((button) => button.getAttribute('aria-label'))).toEqual([
      'Open work package train/other',
      'Open work package train/demo',
    ]);
    const trainDemo = screen.getByRole('region', { name: 'Work package train/demo' });
    expect(within(trainDemo).getAllByRole('button', { name: /^Select run/ }).map((button) => button.getAttribute('aria-label'))).toEqual([
      'Select run Train newer',
      'Select run Train older',
    ]);
    expect([...trainDemo.querySelectorAll('time')].map((element) => element.getAttribute('datetime'))).toEqual([
      '2026-07-22T06:00:00Z',
      '2026-07-22T04:00:00Z',
    ]);
  });

  it('switches project scope instead of mixing project runs', async () => {
    const otherRun = {
      ...run,
      locator: { source_id: 'fixture', run_id: 'runs/other' },
      run_key: 'other-key',
      run: { ...run.run, run_id: 'runs/other', display_name: 'Other run', project_id: 'projects/other', work_package_id: 'train/other' },
    };
    const otherView = { ...view, view: { ...view.view, run: otherRun.run } };
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      const body = path === '/api/v1/runs' ? [run, otherRun] : path.includes('other-key') ? otherView : view;
      return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
    }));
    const user = userEvent.setup();
    render(<App />);
    expect(await screen.findByRole('button', { name: 'Project: demo' })).toBeVisible();
    await user.click(screen.getByRole('button', { name: 'Project: demo' }));
    await user.click(screen.getByRole('option', { name: 'other projects/other' }));

    expect(await screen.findByRole('button', { name: 'Project: other' })).toBeVisible();
    expect(screen.getByRole('button', { name: 'Open work package train/other' })).toBeVisible();
    expect(screen.getByRole('button', { name: 'Select run Other run' })).toBeVisible();
    expect(screen.queryByRole('button', { name: 'Select run SFT calm harbor' })).not.toBeInTheDocument();
  });

  it('switches evidence backends without mixing runs from the same project', async () => {
    const wandbRun = {
      ...run,
      locator: { source_id: 'wandb-cloud', run_id: 'runs/wandb-sft' },
      run_key: 'wandb-key',
      run: {
        ...run.run,
        run_id: 'runs/wandb-sft',
        display_name: 'W&B SFT run',
        provider: 'wandb',
      },
    };
    const wandbView = { ...view, view: { ...view.view, run: wandbRun.run } };
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      const body = path === '/api/v1/runs' ? [run, wandbRun] : path.includes('wandb-key') ? wandbView : view;
      return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
    }));
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByRole('button', { name: 'Backend: Fixture' })).toBeVisible();
    expect(screen.getByRole('button', { name: 'Select run SFT calm harbor' })).toBeVisible();
    expect(screen.queryByRole('button', { name: 'Select run W&B SFT run' })).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Backend: Fixture' }));
    await user.click(screen.getByRole('option', { name: /Weights & Biases.*wandb-cloud/ }));

    expect(await screen.findByRole('button', { name: 'Backend: Weights & Biases' })).toBeVisible();
    expect(screen.getByRole('button', { name: 'Select run W&B SFT run' })).toBeVisible();
    expect(screen.queryByRole('button', { name: 'Select run SFT calm harbor' })).not.toBeInTheDocument();
  });

  it('refreshes all evidence backends from inside the open backend popover', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      const body = path === '/api/v1/runs'
        ? [run]
        : path === '/api/v1/sources/refresh'
          ? {
              enabled: true,
              state: 'succeeded',
              last_attempt_at: '2026-07-30T11:00:00Z',
              last_success_at: '2026-07-30T11:00:00Z',
              error: null,
              discovered_source_ids: ['fixture'],
            }
          : view;
      if (path === '/api/v1/sources/refresh') expect(init?.method).toBe('POST');
      return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
    });
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole('button', { name: 'Backend: Fixture' }));
    await user.click(screen.getByRole('button', { name: 'Refresh evidence backends' }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/v1/sources/refresh', { method: 'POST' }));
    expect(screen.getByText('Evidence backend')).toBeVisible();
    expect(screen.getByRole('button', { name: 'Refresh evidence backends' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Select run SFT calm harbor' })).toBeVisible();
    expect(fetchMock.mock.calls.filter(([input]) => String(input) === '/api/v1/runs')).toHaveLength(2);
  });

  it('shows a safe source refresh failure beneath the backend header', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      const body = path === '/api/v1/runs'
        ? [run]
        : path === '/api/v1/sources/refresh'
          ? {
              enabled: true,
              state: 'failed',
              last_attempt_at: '2026-07-30T11:00:00Z',
              last_success_at: null,
              error: 'Trackio is temporarily unavailable',
              discovered_source_ids: [],
            }
          : view;
      return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
    }));
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole('button', { name: 'Backend: Fixture' }));
    await user.click(screen.getByRole('button', { name: 'Refresh evidence backends' }));

    expect(await screen.findByRole('alert')).toHaveTextContent('Trackio is temporarily unavailable');
    expect(screen.getByText('Evidence backend')).toBeVisible();
    expect(screen.getByRole('button', { name: 'Select run SFT calm harbor' })).toBeVisible();
  });

  it('shows the selected run shell while the backend view is loading', async () => {
    const wandbRun = {
      ...dpoRun,
      locator: { source_id: 'wandb-cloud', run_id: 'runs/wandb-dpo' },
      run_key: 'wandb-dpo-key',
      run: {
        ...dpoRun.run,
        run_id: 'runs/wandb-dpo',
        display_name: 'W&B failed DPO run',
        provider: 'wandb',
        status: 'failed',
      },
    };
    const wandbView = { ...dpoView, view: { ...dpoView.view, run: wandbRun.run, artifacts: { items: [] } } };
    let resolveWandb: ((response: Response) => void) | undefined;
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path === '/api/v1/runs') {
        return new Response(JSON.stringify([run, wandbRun]), { status: 200, headers: { 'content-type': 'application/json' } });
      }
      if (path.includes('wandb-dpo-key')) {
        return await new Promise<Response>((resolve) => { resolveWandb = resolve; });
      }
      return new Response(JSON.stringify(view), { status: 200, headers: { 'content-type': 'application/json' } });
    }));
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByRole('heading', { name: 'Learning & data evidence' })).toBeVisible();
    await user.click(screen.getByRole('button', { name: 'Backend: Fixture' }));
    await user.click(screen.getByRole('option', { name: /Weights & Biases.*wandb-cloud/ }));

    expect(screen.getByRole('region', { name: 'Run summary shell' })).toBeVisible();
    expect(screen.queryByRole('heading', { name: 'Learning & data evidence' })).not.toBeInTheDocument();

    await act(async () => {
      resolveWandb?.(new Response(JSON.stringify(wandbView), { status: 200, headers: { 'content-type': 'application/json' } }));
      await Promise.resolve();
    });
    expect(await screen.findByRole('heading', { name: 'Preference learning evidence' })).toBeVisible();
    expect(screen.getByText('failed')).toBeVisible();
    expect(screen.queryByRole('heading', { name: 'Learning & data evidence' })).not.toBeInTheDocument();
  });

  it('does not let a slower prior run request replace the current run view', async () => {
    const raceDpoRun = {
      ...dpoRun,
      locator: { source_id: 'fixture', run_id: 'runs/dpo' },
      run_key: 'dpo-key',
    };
    const raceDpoView = { ...dpoView, view: { ...dpoView.view, run: raceDpoRun.run } };
    let resolveSft: ((response: Response) => void) | undefined;
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path === '/api/v1/runs') {
        return new Response(JSON.stringify([raceDpoRun, run]), { status: 200, headers: { 'content-type': 'application/json' } });
      }
      if (path.includes('run-key')) {
        return await new Promise<Response>((resolve) => { resolveSft = resolve; });
      }
      return new Response(JSON.stringify(raceDpoView), { status: 200, headers: { 'content-type': 'application/json' } });
    }));
    const user = userEvent.setup();
    render(<App />);
    expect(await screen.findByRole('img', { name: 'Pair ordering metric series for DPO amber field' })).toBeVisible();

    await user.click(screen.getByRole('button', { name: 'Select run SFT calm harbor' }));
    expect(resolveSft).toBeDefined();
    await user.click(screen.getByRole('button', { name: 'Select run DPO amber field' }));

    await act(async () => {
      resolveSft?.(new Response(JSON.stringify(view), { status: 200, headers: { 'content-type': 'application/json' } }));
      await Promise.resolve();
    });
    expect(screen.getByRole('img', { name: 'Pair ordering metric series for DPO amber field' })).toBeVisible();
    expect(screen.queryByRole('img', { name: 'Optimization metric series for DPO amber field' })).not.toBeInTheDocument();
  });

  it('organizes SFT run configuration by the job schema and keeps JSON secondary', async () => {
    const user = userEvent.setup();
    render(<App />);
    expect(await screen.findByRole('heading', { name: 'Learning & data evidence' })).toBeVisible();
    await user.click(screen.getByRole('button', { name: 'Run config' }));

    expect(await screen.findByRole('heading', { name: 'Run configuration' })).toBeVisible();
    expect(screen.getByRole('region', { name: 'Run contract' })).toHaveTextContent('train.sft@1');
    expect(screen.getByRole('heading', { name: 'Training population' })).toBeVisible();
    expect(screen.getByRole('heading', { name: 'Optimization' })).toBeVisible();
    expect(screen.getByRole('heading', { name: 'Package context' })).toBeVisible();
    expect(screen.getByRole('heading', { name: 'Source provenance' })).toBeVisible();
    expect(screen.getByRole('article', { name: 'Training dataset selection' })).toHaveTextContent('datasets/sft-train');
    expect(screen.getByRole('article', { name: 'Validation dataset selection' })).toHaveTextContent('datasets/sft-validation');
    const raw = screen.getByText('View redacted JSON');
    expect(raw.closest('details')).not.toHaveAttribute('open');
  });

  it('renders independently scaled metric cards and allows more than three selections', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path === '/api/v1/runs') {
        return new Response(JSON.stringify([run]), { status: 200, headers: { 'content-type': 'application/json' } });
      }
      if (path.includes('mode=generic')) {
        const selected = new URL(path, 'http://observatory.local').searchParams.getAll('metric');
        return new Response(JSON.stringify(genericView(selected)), { status: 200, headers: { 'content-type': 'application/json' } });
      }
      return new Response(JSON.stringify(view), { status: 200, headers: { 'content-type': 'application/json' } });
    }));
    const user = userEvent.setup();
    render(<App />);
    expect(await screen.findByRole('heading', { name: 'Learning & data evidence' })).toBeVisible();
    await user.click(screen.getByRole('button', { name: 'Metrics' }));
    expect(await screen.findByRole('heading', { name: 'Metric workspace' })).toBeVisible();
    await waitFor(() => expect(screen.getAllByRole('region', { name: /metric card$/ })).toHaveLength(3));
    const fourth = screen.getByRole('checkbox', { name: 'train/entropy' });
    expect(fourth).toBeEnabled();
    await user.click(fourth);
    await waitFor(() => expect(screen.getAllByRole('region', { name: /metric card$/ })).toHaveLength(4));
    for (const chart of screen.getAllByRole('img', { name: /metric series$/ })) {
      expect(chart).toHaveAttribute('data-show-legend', 'false');
    }
    const charts = screen.getAllByRole('img', { name: /metric series$/ });
    await user.hover(charts[0]);
    await waitFor(() => {
      for (const chart of charts) expect(chart).toHaveAttribute('data-hovered-step', '1');
    });
    await user.unhover(charts[0]);
    await waitFor(() => {
      for (const chart of charts) expect(chart).toHaveAttribute('data-hovered-step', '');
    });
  });

  it('explains registered metrics from the shared telemetry schema', async () => {
    const user = userEvent.setup();
    render(<App />);
    expect(await screen.findByRole('heading', { name: 'Learning & data evidence' })).toBeVisible();
    await user.click(screen.getByRole('button', { name: 'About Final loss' }));
    const popover = await screen.findByRole('dialog', { name: 'Final loss metric definition' });
    expect(popover).toHaveTextContent('Mean supervised objective on the current optimization batch.');
    expect(popover).toHaveTextContent('train/loss');
    await user.keyboard('{Escape}');
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Final loss metric definition' })).not.toBeInTheDocument());
  });

  it('organizes DPO evidence around pair ordering, policy movement, and method context', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const body = String(input) === '/api/v1/runs' ? [dpoRun] : dpoView;
      return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
    }));
    const user = userEvent.setup();
    render(<App />);
    expect(await screen.findByRole('heading', { name: 'Preference learning evidence' })).toBeVisible();
    expect(screen.getByText(/without weakening chosen responses/)).toBeVisible();
    const method = screen.getByRole('region', { name: 'Algorithm settings' });
    expect(within(method).getByRole('heading', { name: 'Method context' })).toBeVisible();
    expect(method).toHaveTextContent('Beta0.1');
    expect(method).toHaveTextContent('Preference data128 examples');
    expect(screen.getByText('Is the policy consistently ranking chosen completions above rejected ones?')).toBeVisible();
    expect(screen.getByRole('region', { name: 'DPO evidence completeness' })).toHaveTextContent('partial');
    expect(screen.getByText('Required 5/6')).toBeVisible();
    expect(screen.getByText('Not research ready')).toBeVisible();
    expect(screen.getByText('Effective runtime · 1 metric missing')).toBeVisible();
    expect(screen.queryByRole('button', { name: 'Traces & evaluation' })).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Run config' }));
    expect(await screen.findByRole('heading', { name: 'Preference inputs' })).toBeVisible();
    expect(screen.getByRole('heading', { name: 'Preference optimization' })).toBeVisible();
    expect(screen.getByRole('article', { name: 'Training dataset selection' })).toHaveTextContent('dataset/preferences');
  });

  it('presents SAMPO hierarchy, method settings, and run configuration explicitly', async () => {
    const { jobRun, jobView } = metricJob(
      'train.sampo',
      'SAMPO copper ridge',
      [
        { key: 'reward_mean', label: 'Mean reward', metric: 'train/rl/reward_mean', value: 0.7, unit: null },
        { key: 'episode_advantage', label: 'Episode advantage', metric: 'train/rl/episode_advantage_mean', value: 0.4, unit: null },
        { key: 'turn_advantage', label: 'Turn advantage', metric: 'train/rl/turn_advantage_mean', value: 0.2, unit: null },
      ],
      {
        model: { selection_id: 'models/qwen-sampo', revision: 'v1', resolved: {} },
        environment: { selection_id: 'environments/tool-use', revision: 'v2', resolved: {} },
        settings: {
          selection_id: 'settings/sampo',
          revision: 'v1',
          resolved: {
            num_generations: 4,
            discount_gamma: 0.95,
            step_advantage_weight: 1,
            advantage_normalization: 'mean_std',
            clip_epsilon_low: 0.003,
            clip_epsilon_high: 0.004,
          },
        },
        rollout_inference: { selection_id: 'inference/sampo', revision: 'v1', resolved: {} },
        training: {
          selection_id: 'training/sampo-lora',
          revision: 'v1',
          resolved: { parameter_update_kind: 'lora', runtime: { global_batch_size: 4 } },
        },
      },
      true,
    );
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const body = String(input) === '/api/v1/runs' ? [jobRun] : jobView;
      return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
    }));
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByRole('heading', { name: 'Hierarchical policy-learning evidence' })).toBeVisible();
    expect(screen.getByText(/episode and turn-level credit assignment/)).toBeVisible();
    expect(screen.getByRole('region', { name: 'SAMPO evidence completeness' })).toHaveTextContent('Research ready');
    const method = screen.getByRole('region', { name: 'Algorithm settings' });
    expect(method).toHaveTextContent('Discount gamma0.95');
    expect(method).toHaveTextContent('Turn weight1');
    expect(method).toHaveTextContent('Advantage normalizationMean Std');

    await user.click(screen.getByRole('button', { name: 'Run config' }));
    expect(await screen.findByRole('heading', { name: 'Policy & environment' })).toBeVisible();
    expect(screen.getByRole('heading', { name: 'Hierarchical optimization' })).toBeVisible();
  });

  it.each([
    {
      kind: 'train.distill',
      display: 'Distill quiet lake',
      heading: 'Student learning evidence',
      configHeading: 'Student, teacher & data',
      summary: [
        { key: 'final_loss', label: 'Final loss', metric: 'train/distill/loss', value: 0.3, unit: null },
        { key: 'reverse_kl', label: 'Reverse KL', metric: 'train/distill/reverse_kl', value: 0.1, unit: null },
      ],
      inputs: {
        student: { selection_id: 'models/student', revision: 'v1', resolved: {} },
        teacher: { selection_id: 'models/teacher', revision: 'v1', resolved: {} },
        settings: { selection_id: 'settings/distill', revision: 'v1', resolved: { temperature: 0.8, num_generations: 2 } },
      },
      traceAware: true,
    },
    {
      kind: 'serve.smoke',
      display: 'Serve smoke green field',
      heading: 'Managed endpoint smoke test',
      configHeading: 'Serving binding',
      summary: [
        { key: 'healthy', label: 'Endpoint healthy', metric: 'serve/probe_healthy', value: 1, unit: 'ratio' },
        { key: 'model_available', label: 'Model available', metric: 'serve/probe_model_available', value: 1, unit: 'ratio' },
        { key: 'probe_latency', label: 'Probe latency', metric: 'serve/probe_latency_seconds', value: 0.2, unit: 's' },
      ],
      inputs: { inference: { selection_id: 'inference/smoke', revision: 'v1', resolved: {} } },
      traceAware: false,
    },
    {
      kind: 'data.prepare',
      display: 'Dataset prepare blue field',
      heading: 'Prepared dataset evidence',
      configHeading: 'Dataset input',
      summary: [
        { key: 'examples', label: 'Prepared examples', metric: 'data/examples', value: 128, unit: null },
        { key: 'bytes', label: 'Prepared bytes', metric: 'data/bytes', value: 4096, unit: 'bytes' },
      ],
      inputs: { dataset: { selection_id: 'datasets/prepared', revision: 'v1', resolved: { num_examples: 128 } } },
      traceAware: false,
    },
  ])('uses first-class copy and configuration for $kind', async ({
    kind,
    display,
    heading,
    configHeading,
    summary,
    inputs,
    traceAware,
  }) => {
    const { jobRun, jobView } = metricJob(kind, display, summary, inputs, traceAware);
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const body = String(input) === '/api/v1/runs' ? [jobRun] : jobView;
      return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
    }));
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByRole('heading', { name: heading })).toBeVisible();
    if (kind === 'train.distill') {
      expect(screen.getByText('Student model')).toBeVisible();
      expect(screen.getByText('Teacher model')).toBeVisible();
    }
    if (kind === 'serve.smoke' || kind === 'data.prepare') {
      expect(screen.queryByRole('region', { name: 'Algorithm settings' })).not.toBeInTheDocument();
      expect(screen.queryByText('Training binding')).not.toBeInTheDocument();
    }
    if (kind === 'serve.smoke') expect(screen.getByText('Inference binding')).toBeVisible();
    if (kind === 'data.prepare') expect(screen.getByText('Source dataset')).toBeVisible();
    await user.click(screen.getByRole('button', { name: 'Run config' }));
    expect(await screen.findByRole('heading', { name: configHeading })).toBeVisible();
  });

  it('keeps trace navigation for trace-aware job definitions', async () => {
    const grpoRun = {
      ...run,
      run: {
        ...run.run,
        run_id: 'runs/grpo',
        display_name: 'GRPO silver pine',
        job_kind: 'train.grpo',
        job_definition_version: 'train.grpo@1',
      },
    };
    const grpoView = {
      ...view,
      view: {
        ...view.view,
        run: grpoRun.run,
        trace_count: 0,
        trace_evaluation_enabled: true,
      },
    };
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const body = String(input) === '/api/v1/runs' ? [grpoRun] : grpoView;
      return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
    }));

    render(<App />);

    expect(await screen.findByRole('heading', { name: 'Policy learning evidence' })).toBeVisible();
    expect(screen.getByRole('button', { name: 'Rollouts & rewards' })).toBeVisible();
  });

  it('pages optimization rollouts without requesting the evaluation population scan', async () => {
    const { jobRun, jobView } = metricJob(
      'train.grpo',
      'GRPO paged cedar',
      [{ key: 'reward_mean', label: 'Mean reward', metric: 'train/rl/reward_mean', value: 0.2, unit: null }],
      {},
      true,
    );
    jobView.view.trace_count = 250;
    const trace = (externalId: string) => ({
      external_id: externalId,
      trace_type: 'verifiers',
      prompt_preview: `Prompt ${externalId}`,
      task: 'task-a',
      task_label: 'Task A',
      task_metadata: null,
      reward: 0.25,
      success: null,
      outcome: 'scored',
      truncated: false,
      error: null,
      tool_calls: 1,
      model_calls: 1,
      input_tokens: 10,
      completion_tokens: 20,
      latency_ms: 100,
      tokens: 30,
      response_tokens: 20,
      response_chars: 80,
      thinking_tokens: 5,
      thinking_chars: 20,
      reward_components: { correct: 0.25 },
      native_metrics: {},
      metrics: { correct: 0.25 },
    });
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path === '/api/v1/runs') return new Response(JSON.stringify([jobRun]));
      if (path.includes('/traces-evaluation')) throw new Error('optimization view must not request evaluation aggregation');
      if (path.includes('/traces?')) {
        const second = path.includes('cursor=100');
        return new Response(JSON.stringify({
          items: second ? [trace('rollout-3')] : [trace('rollout-1'), trace('rollout-2')],
          next_cursor: second ? null : '100',
          total: 250,
          live: true,
        }));
      }
      return new Response(JSON.stringify(jobView));
    });
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole('button', { name: 'Rollouts & rewards' }));
    expect(await screen.findByText('2 of 250 loaded')).toBeVisible();
    expect(fetchMock.mock.calls.some(([input]) => String(input).includes('/traces-evaluation'))).toBe(false);

    await user.click(screen.getByRole('button', { name: 'Load 100 more' }));
    expect(await screen.findByText('3 of 250 loaded')).toBeVisible();
  });

  it('groups OPD student trajectories with their teacher scoring evidence', async () => {
    const { jobRun, jobView } = metricJob(
      'train.distill',
      'OPD paired evidence',
      [
        { key: 'scored_tokens', label: 'Scored tokens', metric: 'train/distill/scored_tokens', value: 11803, unit: 'tokens' },
        { key: 'teacher_latency_ms', label: 'Teacher latency', metric: 'train/distill/teacher_latency_ms', value: 1536.69, unit: 'ms' },
        { key: 'teacher_failures', label: 'Teacher failures', metric: 'train/distill/teacher_failures', value: 0, unit: null },
      ],
      {
        student: { selection_id: 'models/gemma4-e2b-it@bf16', resolved: {} },
        teacher: { selection_id: 'models/gemma4-12b-it@bf16', resolved: {} },
      },
      true,
    );
    Object.assign(jobView.view.charts[0].series[0].points[0], {
      attributes: { distillation_batch_id: 'distill-batch-1' },
    });
    const trace = {
      external_id: 'opd-rollout-1', trace_type: 'verifiers', prompt_preview: 'Produce one source-grounded target',
      task: 'scope-opd', task_label: 'Scope OPD', task_metadata: null, reward: 0, success: null,
      outcome: 'scored', truncated: false, error: null, tool_calls: 0, model_calls: 1,
      input_tokens: 3172, completion_tokens: 825, latency_ms: 48617.406, tokens: 825,
      response_tokens: null, response_chars: null, thinking_tokens: null, thinking_chars: null,
      reward_components: {}, native_metrics: {}, metrics: {},
    };
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path === '/api/v1/runs') return new Response(JSON.stringify([jobRun]));
      if (path.includes('/traces?')) return new Response(JSON.stringify({ items: [trace], next_cursor: null, total: 1, live: true }));
      return new Response(JSON.stringify(jobView));
    }));
    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole('button', { name: 'Rollouts & rewards' }));
    const pairing = await screen.findByRole('region', { name: 'Paired distillation evidence' });
    expect(within(pairing).getByText('models/gemma4-e2b-it@bf16')).toBeVisible();
    expect(within(pairing).getByText('models/gemma4-12b-it@bf16')).toBeVisible();
    expect(within(pairing).getByText(/11,803 scored tokens/)).toBeVisible();
    expect(within(pairing).getByText(/1\.5s teacher latency/)).toBeVisible();
    expect(within(pairing).getByText(/0 failures/)).toBeVisible();
    expect(screen.getByRole('heading', { name: 'Student trajectories (1)' })).toBeVisible();
  });

  it('keeps the GRPO decision metrics to one five-item headline group', async () => {
    const { jobRun, jobView } = metricJob(
      'train.grpo',
      'GRPO compact ridge',
      [
        { key: 'reward_mean', label: 'Mean reward', metric: 'train/rl/reward_mean', value: -0.17, unit: null },
        { key: 'entropy', label: 'Policy entropy', metric: 'train/rl/entropy', value: 1.42, unit: null },
        { key: 'zero_variance', label: 'Zero-variance groups', metric: 'train/rl/group_zero_variance_fraction', value: 0.25, unit: 'ratio' },
        { key: 'clip_fraction_low', label: 'Lower clip fraction', metric: 'train/rl/clip_fraction_low', value: 0.03, unit: 'ratio' },
        { key: 'clip_fraction_high', label: 'Upper clip fraction', metric: 'train/rl/clip_fraction_high', value: 0.07, unit: 'ratio' },
        { key: 'grad_norm', label: 'Gradient norm', metric: 'train/grad_norm', value: 0.88, unit: null },
        { key: 'policy_loss', label: 'Policy loss', metric: 'train/rl/policy_loss', value: -0.4, unit: null },
        { key: 'rollout_tps', label: 'Rollout throughput', metric: 'train/rl/rollout_tokens_per_second', value: 115, unit: 'tokens/s' },
      ],
      {
        environment: { selection_id: 'environments/rollout', revision: 'v1', resolved: { max_concurrent: 160 } },
        settings: {
          selection_id: 'settings/dapo',
          revision: 'v1',
          resolved: {
            beta: 0,
            num_generations: 8,
            max_prompt_length: 2048,
            max_completion_length: 1536,
            per_device_batch_size: 4,
            gradient_accumulation_steps: 64,
            learning_rate: 0.00001,
            importance_sampling_mode: 'sequence_truncate',
          },
        },
        training: {
          selection_id: 'training/lora',
          revision: 'v1',
          resolved: { parameter_update_kind: 'lora', runtime: { global_batch_size: 256 } },
        },
        rollout_inference: {
          selection_id: 'inference/rollout',
          revision: 'v1',
          resolved: {
            engine: { max_num_seqs: 160, speculative_config: { method: 'mtp', num_speculative_tokens: 3 } },
            sampling: { temperature: 1, top_p: 1 },
          },
        },
      },
      true,
    );
    const configuredJobView = {
      ...jobView,
      view: {
        ...jobView.view,
        charts: [
          ...jobView.view.charts,
          {
            key: 'rollout_population',
            title: 'Rollout population',
            question: 'What happened to the requested rollout population?',
            series: [{ name: 'train/rl/rollouts_completed', points: [{ value: 256, step: 1 }] }],
          },
        ],
        grpo: {
          rollout_population: {
            requested: { key: 'requested', label: 'Requested', metric: 'train/rl/rollouts_requested', state: 'available', value: 256, unit: null },
            attempted: { key: 'attempted', label: 'Attempted', metric: 'train/rl/rollouts_attempted', state: 'available', value: 256, unit: null },
            completed: { key: 'completed', label: 'Completed', metric: 'train/rl/rollouts_completed', state: 'available', value: 256, unit: null },
            failed: { key: 'failed', label: 'Failed', metric: 'train/rl/rollouts_failed', state: 'available', value: 0, unit: null },
            truncated: { key: 'truncated', label: 'Truncated', metric: 'train/rl/rollouts_truncated', state: 'available', value: 0, unit: null },
            unscorable: { key: 'unscorable', label: 'Unscorable', metric: 'train/rl/rollouts_unscorable', state: 'available', value: 0, unit: null },
            missing: { key: 'missing', label: 'Missing', metric: 'train/rl/rollouts_missing', state: 'available', value: 0, unit: null },
          },
          acceleration: {
            mtp_selected: true,
            quantized_kv_cache_selected: false,
            speculative_acceptance: { key: 'speculative_acceptance', label: 'MTP acceptance', metric: 'serve/backend/speculative_acceptance_rate', state: 'available', value: 0.51, unit: 'ratio' },
            accepted_speculative_length: { key: 'accepted_speculative_length', label: 'Accepted length', metric: 'serve/backend/speculative_accepted_length', state: 'available', value: 2.5, unit: 'tokens' },
            kv_cache_peak_usage: { key: 'kv_cache_peak_usage', label: 'Peak KV-cache usage', metric: 'serve/backend/kv_cache_peak_usage_ratio', state: 'available', value: 0.25, unit: 'ratio' },
          },
        },
      },
    };
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const body = String(input) === '/api/v1/runs' ? [jobRun] : configuredJobView;
      return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
    }));

    render(<App />);

    const headline = await screen.findByRole('group', { name: 'GRPO headline metrics' });
    expect(within(headline).getByText('Mean reward')).toBeVisible();
    expect(within(headline).getByText('Policy entropy')).toBeVisible();
    expect(within(headline).getByText('Usable groups')).toBeVisible();
    expect(within(headline).getByText('75.0%')).toBeVisible();
    expect(within(headline).getByText('Clip pressure')).toBeVisible();
    expect(within(headline).getByText('3.0% / 7.0%')).toBeVisible();
    expect(within(headline).getByText('Gradient norm')).toBeVisible();
    expect(within(headline).queryByText('Policy loss')).not.toBeInTheDocument();
    expect(within(headline).queryByText('Rollout throughput')).not.toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'GRPO rollout population' })).not.toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Rollout population' })).toBeVisible();
    const rolloutSetup = screen.getByRole('region', { name: 'Rollout setup' });
    expect(rolloutSetup).toHaveTextContent('Prompt groups / update32 · derived');
    expect(rolloutSetup).toHaveTextContent('Rollouts / prompt8');
    expect(rolloutSetup).toHaveTextContent('Rollouts / update256');
    expect(rolloutSetup).toHaveTextContent('Environment concurrency160');
    expect(rolloutSetup).toHaveTextContent('Inference sequence cap160');
    expect(rolloutSetup).toHaveTextContent('AccelerationMTP · 3 draft tokens');
    const algorithm = screen.getByRole('region', { name: 'Algorithm settings' });
    expect(algorithm).toHaveTextContent('Actor microbatch4');
    expect(algorithm).toHaveTextContent('Grad accumulation64');
  });
});
