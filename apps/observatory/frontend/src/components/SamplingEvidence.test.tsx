import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

const chartMocks = vi.hoisted(() => ({
  setOption: vi.fn(),
  resize: vi.fn(),
  dispose: vi.fn(),
}));

vi.mock('echarts/core', () => ({
  use: vi.fn(),
  init: vi.fn(() => chartMocks),
}));

import type { GRPOSamplingEvidence, SummaryMetric } from '../lib/api';
import { SamplingDistribution, SamplingSummary } from './SamplingEvidence';

afterEach(cleanup);

function metric(key: string, label: string, value: number): SummaryMetric {
  return { key, label, metric: `train/rl/${key}`, state: 'available', value, unit: null };
}

const sampling: GRPOSamplingEvidence = {
  strategy: 'olmo3_active',
  algorithm: 'olmo3',
  adaptive_controller: true,
  zero_variance_scope: 'candidate',
  zero_variance: metric('candidate_zero_variance', 'Candidate zero-variance groups', 0.5),
  retained_fraction: metric('retained_fraction', 'Candidate retention', 0.5),
  generation_rounds: metric('generation_rounds', 'Refill rounds', 2),
  generated_groups: metric('generated_groups', 'Candidate task groups', 6),
  retained_groups: metric('retained_groups', 'Retained task groups', 3),
  steps: [{
    step: 1,
    candidate_groups: 6,
    unique_tasks: 6,
    new_tasks: 4,
    discovery_reserved: 2,
    discovery_fulfilled: 2,
    duplicate_fallbacks: 0,
    refill_rounds: 2,
    class_counts: { arithmetic: 3, algebra: 3 },
    retained_groups: 4,
  }],
};

describe('SamplingEvidence', () => {
  it('distinguishes OLMo candidate filtering from the optimizer population', () => {
    render(<SamplingSummary sampling={sampling} />);

    expect(screen.getByRole('region', { name: 'OLMo active sampling evidence' })).toBeInTheDocument();
    expect(screen.getByText('Useful groups')).toBeInTheDocument();
    expect(screen.getByText('Tied groups')).toBeInTheDocument();
    expect(screen.getByText('whole run · all rollouts scored the same, discarded')).toBeInTheDocument();
    expect(screen.getByText('3 / 6')).toBeInTheDocument();
  });

  it('shows complete per-step class allocation and task-selection counts', () => {
    render(<SamplingDistribution sampling={sampling} />);

    expect(screen.getByRole('img', { name: 'Stacked class distribution and total candidates by optimizer step' })).toBeInTheDocument();
    expect(screen.getByText('Step 1: algebra 3, arithmetic 3')).toBeInTheDocument();
    const table = screen.getByRole('table', { name: 'Sampling cost and task mix by optimizer step' });
    expect(table).toBeInTheDocument();
    // 6 sampled, 4 useful kept, so 2 tied groups were discarded.
    const kept = screen.getByRole('rowheader', { name: 'Useful groups kept' });
    expect(kept.closest('tr')).toHaveTextContent('4');
    expect(screen.getByRole('rowheader', { name: 'Tied groups discarded' }).closest('tr')).toHaveTextContent('2');
    expect(kept).toHaveAttribute('title', expect.stringContaining('entered the update'));
    // The quota fixture reserves novelty slots, so that row appears; no repeats, so that row is hidden.
    expect(screen.getByText('2/2')).toBeInTheDocument();
    expect(screen.queryByRole('rowheader', { name: 'Same-step repeats' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Latest 20' })).toBeDisabled();
  });

  it('hides quota-only rows for a yield-first run', () => {
    const yieldFirst = { ...sampling, steps: [{ ...sampling.steps[0], discovery_reserved: 0, discovery_fulfilled: 0 }] };
    render(<SamplingDistribution sampling={yieldFirst} />);

    expect(screen.queryByRole('rowheader', { name: 'Novelty quota filled' })).not.toBeInTheDocument();
    expect(screen.getByRole('rowheader', { name: 'First-time tasks' })).toBeInTheDocument();
  });

  it('defaults to the latest 20 steps and lets the range move without losing the counts', () => {
    const longer = { ...sampling, steps: Array.from({ length: 25 }, (_, index) => ({
      ...sampling.steps[0],
      step: index + 1,
    })) };
    render(<SamplingDistribution sampling={longer} />);

    expect(screen.getByText('Steps 6–25 of 25')).toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: 'Step 5' })).not.toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'Step 6' })).toBeInTheDocument();
    const chartOption = chartMocks.setOption.mock.lastCall?.[0] as { series: Array<{ type: string; barCategoryGap?: string; data: number[] }> };
    expect(chartOption.series[0].barCategoryGap).toBe('2%');
    expect(chartOption.series[0].data).toHaveLength(20);

    fireEvent.change(screen.getByRole('slider', { name: 'First optimizer step' }), { target: { value: '0' } });
    fireEvent.change(screen.getByRole('slider', { name: 'Last optimizer step' }), { target: { value: '10' } });
    expect(screen.getByText('Steps 1–11 of 25')).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'Step 1' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Latest 20' }));
    expect(screen.getByText('Steps 6–25 of 25')).toBeInTheDocument();
  });

  it('keeps a full-run class order and stable colors when the visible range changes', () => {
    const longer = { ...sampling, steps: Array.from({ length: 25 }, (_, index) => ({
      ...sampling.steps[0],
      step: index + 1,
      class_counts: index < 5 ? { alpha: 1, zeta: 20 } : { alpha: 2, zeta: 0 },
    })) };
    render(<SamplingDistribution sampling={longer} />);

    type ChartSeries = { name: string; itemStyle: { color: string }; data: number[] };
    const series = () => (chartMocks.setOption.mock.lastCall?.[0] as { series: ChartSeries[] }).series;
    expect(series().map(({ name }) => name)).toEqual(['zeta', 'alpha']);
    const initialColors = series().map(({ itemStyle }) => itemStyle.color);
    expect(series()[0].data).toEqual(Array(20).fill(0));

    fireEvent.change(screen.getByRole('slider', { name: 'First optimizer step' }), { target: { value: '0' } });
    fireEvent.change(screen.getByRole('slider', { name: 'Last optimizer step' }), { target: { value: '4' } });
    expect(series().map(({ name }) => name)).toEqual(['zeta', 'alpha']);
    expect(series().map(({ itemStyle }) => itemStyle.color)).toEqual(initialColors);
    expect(series()[0].data).toEqual(Array(5).fill(20 / 21 * 100));
  });
});
