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
                  duration_s: 120,
                  occurrences: 1,
                  sample_count: 1,
                  metrics: [{ metric: 'system/gpu_vram_used_bytes', label: 'GPU memory', unit: 'bytes', mean: 3_000_000_000, peak: 3_000_000_000, minimum: 3_000_000_000, samples: 1 }],
                },
                {
                  phase: 'actor_update',
                  label: 'Actor update',
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
    await user.click(within(phaseProfile).getByRole('tab', { name: 'timeline' }));
    expect(screen.getByRole('img', { name: 'Runtime phase overlap and GPU memory timeline' })).toBeVisible();
    const computeChart = screen.getByRole('region', { name: 'Compute utilization system chart' });
    const computeHeader = within(computeChart).getByRole('heading', { name: 'Compute utilization' }).parentElement;
    expect(computeHeader).toHaveTextContent('GPU utilization');
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
