import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type {
  ServingCapacityEligibility,
  ServingCapacityWorkPackage,
  ServingOperatingPoint,
} from '../../lib/api';
import { ServingCapacityWorkPackageView } from './ServingCapacityWorkPackage';

const point = (
  aggregateOutputTps: number,
  p95TtftMs: number,
  peakVramBytes: number,
): ServingOperatingPoint => ({
  sweep_index: 2,
  concurrency: 4,
  context_tokens: 32_768,
  attempted_requests: 128,
  completed_requests: 128,
  failed_requests: 0,
  output_tokens: 16_384,
  input_tokens_mean: 512,
  input_tokens_p95: 780,
  output_tokens_mean: 128,
  output_tokens_p95: 128,
  measurement_seconds: 4,
  aggregate_output_tps: aggregateOutputTps,
  failure_rate: 0,
  p50_ttft_ms: p95TtftMs * 0.8,
  p95_ttft_ms: p95TtftMs,
  p50_tpot_ms: 14,
  p95_tpot_ms: 18,
  peak_vram_bytes: peakVramBytes,
  kv_cache_peak_usage_ratio: 0.72,
  evidence_state: 'complete',
  terminal_status: null,
  valid: true,
  violations: [],
});

const eligibility = (
  state: ServingCapacityEligibility['state'],
  label: string,
): ServingCapacityEligibility => ({
  state,
  label,
  reason: label,
  calculator_version: 'serving-capacity-v1',
  requirements_digest: 'requirements-v1',
  saturation_state: 'saturated',
  selected_sweep_index: 2,
});

const view: ServingCapacityWorkPackage = {
  schema_version: 1,
  project_id: 'foundation-models',
  work_package_id: 'screen/serving-capacity-v1',
  methodology: 'strict_pareto',
  explanation: 'Only equivalent requirement, workload, corpus, target, and calculator snapshots are compared.',
  requirements: [],
  execution_target_id: 'targets/local-cuda-8gb',
  workload_id: 'workloads/general-serving-32k-sweep@1',
  corpus_digest: 'corpus-v1',
  requirements_digest: 'requirements-v1',
  calculator_version: 'serving-capacity-v1',
  contenders: [
    {
      locator: { source_id: 'fixture', run_id: 'runs/eligible' },
      run_key: 'fixture:runs/eligible',
      display_name: 'Eligible model',
      started_at: '2026-07-25T09:00:00Z',
      model_variant_id: 'models/eligible',
      inference_binding_id: 'inference/eligible',
      inference_backend: 'vllm',
      workload_id: 'workloads/general-serving-32k-sweep@1',
      corpus_digest: 'corpus-v1',
      execution_target_id: 'targets/local-cuda-8gb',
      requirements_digest: 'requirements-v1',
      calculator_version: 'serving-capacity-v1',
      comparable: true,
      comparability_reason: null,
      selected_point: point(58, 700, 7 * 1024 ** 3),
      eligibility: eligibility('eligible', 'Eligible'),
      pareto_member: true,
    },
    {
      locator: { source_id: 'fixture', run_id: 'runs/latency' },
      run_key: 'fixture:runs/latency',
      display_name: 'Latency constrained model',
      started_at: '2026-07-25T09:05:00Z',
      model_variant_id: 'models/latency',
      inference_binding_id: 'inference/latency',
      inference_backend: 'vllm',
      workload_id: 'workloads/general-serving-32k-sweep@1',
      corpus_digest: 'corpus-v1',
      execution_target_id: 'targets/local-cuda-8gb',
      requirements_digest: 'requirements-v1',
      calculator_version: 'serving-capacity-v1',
      comparable: true,
      comparability_reason: null,
      selected_point: point(71, 1_400, 7.2 * 1024 ** 3),
      eligibility: eligibility('latency_constrained', 'Latency constrained'),
      pareto_member: false,
    },
    {
      locator: { source_id: 'fixture', run_id: 'runs/incomparable' },
      run_key: 'fixture:runs/incomparable',
      display_name: 'Different hardware',
      started_at: '2026-07-25T09:10:00Z',
      model_variant_id: 'models/other-target',
      inference_binding_id: 'inference/other-target',
      inference_backend: 'vllm',
      workload_id: 'workloads/general-serving-32k-sweep@1',
      corpus_digest: 'corpus-v1',
      execution_target_id: 'targets/other-gpu',
      requirements_digest: 'requirements-v1',
      calculator_version: 'serving-capacity-v1',
      comparable: false,
      comparability_reason: 'Different execution target.',
      selected_point: point(90, 500, 12 * 1024 ** 3),
      eligibility: eligibility('eligible', 'Eligible'),
      pareto_member: false,
    },
  ],
  pareto: [{
    run_key: 'fixture:runs/eligible',
    model_variant_id: 'models/eligible',
    inference_binding_id: 'inference/eligible',
    aggregate_output_tps: 58,
    p95_ttft_ms: 700,
    peak_vram_bytes: 7 * 1024 ** 3,
  }],
  rows: [],
};

describe('ServingCapacityWorkPackageView', () => {
  it('keeps constrained and incomparable evidence visible beside the Pareto set', () => {
    const onOpenRun = vi.fn();
    render(<ServingCapacityWorkPackageView view={view} onOpenRun={onOpenRun} />);

    expect(screen.getByRole('img', { name: 'Serving contender throughput versus latency Pareto plot' })).toBeVisible();
    expect(screen.getByText('Strict Pareto')).toBeVisible();
    expect(screen.getByText('Latency constrained')).toBeVisible();
    expect(screen.getByText('Different execution target.')).toBeVisible();
    expect(screen.getByText('Incomparable')).toBeVisible();
    expect(screen.getByText('Pareto member')).toBeVisible();

    const table = screen.getByRole('table');
    expect(within(table).getAllByRole('row')).toHaveLength(4);
    fireEvent.click(screen.getByRole('button', { name: 'inference/eligible' }));
    expect(onOpenRun).toHaveBeenCalledWith('fixture:runs/eligible');
  });
});
