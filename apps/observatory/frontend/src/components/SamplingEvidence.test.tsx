import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { GRPOSamplingEvidence, SummaryMetric } from '../lib/api';
import { SamplingDistribution, SamplingSummary } from './SamplingEvidence';

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
  }],
};

describe('SamplingEvidence', () => {
  it('distinguishes OLMo candidate filtering from the optimizer population', () => {
    render(<SamplingSummary sampling={sampling} />);

    expect(screen.getByRole('region', { name: 'OLMo active sampling evidence' })).toBeInTheDocument();
    expect(screen.getByText('Candidate retention')).toBeInTheDocument();
    expect(screen.getByText('candidate population')).toBeInTheDocument();
    expect(screen.getByText('3 / 6')).toBeInTheDocument();
  });

  it('shows complete per-step class allocation and task-selection counts', () => {
    render(<SamplingDistribution sampling={sampling} />);

    expect(screen.getByRole('img', { name: 'Step 1: algebra 3, arithmetic 3' })).toBeInTheDocument();
    expect(screen.getByText('6', { selector: 'strong' })).toBeInTheDocument();
    expect(screen.getByText('4', { selector: 'strong' })).toBeInTheDocument();
    expect(screen.getByText('2/2', { selector: 'strong' })).toBeInTheDocument();
    expect(screen.getByText('0', { selector: 'strong' })).toBeInTheDocument();
  });
});
