import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { TraceEvaluation } from '../lib/api';
import { tracePresentation } from '../lib/trace-presentation';
import { TraceOutcome } from './TraceTable';

const evaluation: TraceEvaluation = {
  state: 'complete',
  metadata: {
    key: 'automationbench',
    label: 'AutomationBench',
    category: 'agentic-tool-use',
    package: 'automationbench-v1',
    dataset: null,
    dataset_revision: null,
    split: null,
    source_revision: null,
    primary_metric: 'partial_credit',
    primary_metric_label: 'Partial credit',
    pass_rate_metric: 'task_completed_correctly',
    pass_rate_basis: 'configured success metric',
    success_definition: null,
    facet_specs: [],
    breakdown_specs: [],
    metrics: [],
  },
  scanned: 0,
  expected: 0,
  included: 0,
  scored: 0,
  mean_reward: null,
  success_rate: 1,
  failures: 0,
  truncated: 0,
  slices: [],
  facets: [],
  breakdowns: [],
  traces: [],
  next_cursor: null,
  live: false,
};

describe('TraceOutcome', () => {
  it('renders compact accessible pass and fail icons', () => {
    const presentation = tracePresentation('eval.general', evaluation);
    const { rerender } = render(<TraceOutcome outcome="pass" presentation={presentation} />);

    expect(screen.getByRole('img', { name: 'Pass' })).toHaveAttribute('title', 'Pass');

    rerender(<TraceOutcome outcome="review" presentation={presentation} />);
    expect(screen.getByRole('img', { name: 'Fail' })).toHaveAttribute('title', 'Fail');
  });
});
