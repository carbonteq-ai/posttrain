import { cleanup, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it } from 'vitest';
import type { ConfigurationReview } from '../../lib/api';
import { ConfigPage, ConfigurationStrip } from './ConfigPage';

const inputs = {
  settings: { selection_id: 'lfm/vortex', revision: '1', resolved: { learning_rate: 1e-5, num_prompts_per_step: 10 } },
  rollout_inference: {
    selection_id: 'inference/lfm-rollout',
    revision: '2',
    resolved: { backend: 'vllm@0.29.1.dev3', engine: { enforce_eager: true, free_cache_engine: true, gpu_memory_utilization: 0.2 } },
  },
};

const review: ConfigurationReview = {
  calculator: 'available',
  calibration: { rollout_seconds: 160, rounds_per_step: 3.5, step_seconds: 850, steps: 20 },
  findings: [
    { code: 'VLLM_EAGER_DISABLES_CUDA_GRAPHS', severity: 'error', source: 'rule', role: 'rollout_inference', path: 'rollout_inference.engine.enforce_eager', value: true, message: 'eager decode was 2.32x slower', hint: 'Remove enforce_eager.', related_paths: [] },
    { code: 'TRL_ROLLOUT_ENGINE_KEYS_IGNORED', severity: 'info', source: 'rule', role: 'rollout_inference', path: 'rollout_inference.engine', value: null, message: 'free_cache_engine is ignored. Acknowledged.', hint: null, related_paths: ['rollout_inference.engine.free_cache_engine'] },
    { code: 'LORA_RL_LEARNING_RATE_BELOW_REFERENCE', severity: 'warning', source: 'rule', role: 'settings', path: 'settings.learning_rate', value: 1e-5, message: 'learning rate below the Tinker recipes', hint: 'Use 4e-05 to 0.00016.', related_paths: [] },
    { code: 'LR_SCHEDULE_DECAYS_WITHIN_SHORT_RUN', severity: 'warning', source: 'rule', role: 'settings', path: 'settings.lr_scheduler_type', value: null, message: 'linear decay over 20 updates', hint: null, related_paths: [] },
  ],
  recommendations: [{
    role: 'rollout_inference',
    state: 'available',
    binding_id: 'inference/lfm-rollout@2',
    hardware: { accelerator: 'RTXPRO6000', memory_gb: 96, gpus: 1 },
    task: { purpose: 'rollout', concurrency: 40, prompt_tokens: 20480, completion_tokens: 4096 },
    settings: [
      { key: 'enforce_eager', suggested: false, current: true, changed: true, reason: 'CUDA graphs stay on' },
      { key: 'gpu_memory_utilization', suggested: 0.14, current: 0.2, changed: true, reason: 'engine share beside the trainer' },
      { key: 'max_num_seqs', suggested: 40, current: 40, changed: false, reason: 'the requested concurrency' },
    ],
    environment: {},
    memory_gb: { weights: 5, kv_cache: 5.25, trainer: 15.7, free: 68 },
    max_concurrency: 552,
    notes: [],
    step: {
      step_sequences: 40, fits: 552, useful: 154, waves: 1, margin_sequences: 114, oversample_groups: 7,
      extra_prompts_per_step: 21, recommended_prompts_per_step: 31, recommended_in_flight: 152, reason: '154 sequences can decode at once',
      options: [
        { label: 'current', prompts_per_step: 10, oversample_groups: 0, rows: 40, waves: 1, relative_step_time: 1, relative_rows_per_second: 1, estimated_step_seconds: 850, estimated_relative_rows_per_second: 1 },
        { label: 'recommended', prompts_per_step: 31, oversample_groups: 7, rows: 152, waves: 1, relative_step_time: 2.43, relative_rows_per_second: 1.56, estimated_step_seconds: 3000, estimated_relative_rows_per_second: 1.07 },
      ],
    },
  }],
};

function renderPage() {
  return render(<ConfigPage
    jobKind="train.grpo"
    jobDefinition="train/trl-grpo@1"
    stage="train"
    schemaVersion={2}
    inputs={inputs}
    source={{}}
    review={review}
    groups={[{ title: 'Rollout & optimization', description: 'Settings.', keys: ['settings', 'rollout_inference'] }]}
    labels={{ settings: 'Optimization settings', rollout_inference: 'Rollout inference' }}
    heading={<h1>Run configuration</h1>}
    sourceFields={null}
  />);
}

afterEach(cleanup);

describe('ConfigPage', () => {
  it('shows findings against the recorded values they are about', () => {
    renderPage();
    const summary = screen.getByRole('region', { name: 'Configuration review' });
    expect(within(summary).getByRole('tab', { name: /1 errors/ })).toBeVisible();
    expect(within(summary).getByRole('tab', { name: /2 warnings/ })).toBeVisible();
    const eager = document.getElementById('cfg-rollout_inference.engine.enforce_eager');
    expect(eager).toHaveTextContent('VLLM_EAGER_DISABLES_CUDA_GRAPHS');
    // A related path carries the finding too, and a value the run did not record still gets a row.
    expect(document.getElementById('cfg-rollout_inference.engine.free_cache_engine')).toHaveTextContent('TRL_ROLLOUT_ENGINE_KEYS_IGNORED');
    expect(document.getElementById('cfg-settings.lr_scheduler_type')).toHaveTextContent('not recorded');
    // The calculator's value sits beside the current one.
    expect(document.getElementById('cfg-rollout_inference.engine.gpu_memory_utilization')).toHaveTextContent('0.14');
  });

  it('offers step options with waves and calibrated step time', () => {
    renderPage();
    const table = screen.getByRole('region', { name: 'Step sizing' });
    expect(table).toHaveTextContent('Oversampled groups');
    expect(table).toHaveTextContent('50.0 min');
    expect(table).toHaveTextContent('1.07×');
  });

  it('opens recommended settings with only the differences by default', async () => {
    const user = userEvent.setup();
    renderPage();
    await user.click(screen.getAllByRole('button', { name: /Recommended settings/ })[0]);
    const dialog = screen.getByRole('dialog', { name: 'Recommended settings' });
    expect(dialog).toHaveTextContent('gpu_memory_utilization');
    expect(dialog).not.toHaveTextContent('max_num_seqs');
    await user.click(within(dialog).getByLabelText('Only differences'));
    expect(dialog).toHaveTextContent('max_num_seqs');
    await user.keyboard('{Escape}');
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('filters settings to flagged and suggested rows', async () => {
    const user = userEvent.setup();
    renderPage();
    await user.click(screen.getByLabelText('Flagged and suggested only'));
    expect(document.getElementById('cfg-settings.num_prompts_per_step')).toBeNull();
    expect(document.getElementById('cfg-settings.learning_rate')).not.toBeNull();
  });

  it('summarizes open findings on the overview', () => {
    render(<ConfigurationStrip review={review} onOpen={() => undefined} />);
    const strip = screen.getByRole('region', { name: 'Configuration review summary' });
    expect(strip).toHaveTextContent('1 error');
    expect(strip).toHaveTextContent('2 warnings');
    expect(strip).not.toHaveTextContent('acknowledged');
  });
});
