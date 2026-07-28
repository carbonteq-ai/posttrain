import { act, cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./components/EvidenceChart', () => ({
  EvidenceChart: ({ ariaLabel, showLegend = true }: { ariaLabel: string; showLegend?: boolean }) => <div role="img" aria-label={ariaLabel} data-show-legend={String(showLegend)} />,
}));
vi.mock('./components/EvaluationCharts', () => ({
  EvaluationCharts: () => <div role="img" aria-label="Evaluation population charts" />,
}));

import App from './App';

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
              ],
              phase_issues: [],
              unclassified_sample_count: 0,
              execution_targets: view.view.execution_targets,
              vram_capacity_state: 'available',
              vram_capacity_bytes: 8 * 1024 ** 3,
              vram_observed_peak_bytes: 5_500_000_000,
            }
          : view;
      return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
    }));
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
    expect(telemetryWindow).toHaveTextContent('Samples6');
    expect(telemetryWindow).not.toHaveClass('obs-card');
    const phaseProfile = screen.getByRole('region', { name: 'Runtime phase profile' });
    expect(phaseProfile).toBeVisible();
    expect(screen.getByRole('img', { name: 'Phase GPU memory against declared hardware capacity' })).toBeVisible();
    expect(within(phaseProfile).getByText(/1 × nvidia-cuda · 8.00 GiB per device/)).toBeVisible();
    expect(within(phaseProfile).getAllByText('Actor update').length).toBeGreaterThan(0);
    expect(within(phaseProfile).getByRole('region', { name: 'Startup phases' })).toBeVisible();
    expect(within(phaseProfile).getByRole('region', { name: 'Training phases' })).toBeVisible();
    await user.click(within(phaseProfile).getByRole('tab', { name: 'timeline' }));
    expect(screen.getByRole('img', { name: 'Runtime phase and GPU utilization timeline' })).toBeVisible();
    const computeChart = screen.getByRole('region', { name: 'Compute utilization system chart' });
    const computeHeader = within(computeChart).getByRole('heading', { name: 'Compute utilization' }).parentElement;
    expect(computeHeader).toHaveTextContent('GPU utilization');
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

  it('clears prior-provider evidence while the selected backend view is loading', async () => {
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

    expect(screen.getByText('Loading run evidence…')).toBeVisible();
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
});
