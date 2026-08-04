import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { CapabilityBreakdown } from '../App';
import type { TraceEvaluation } from '../lib/api';

const evaluation: TraceEvaluation = {
  state: 'complete',
  metadata: {
    key: 'math-python',
    label: 'Math Python',
    category: 'math-tool-use',
    package: 'math-python-v1',
    dataset: null,
    dataset_revision: null,
    split: 'test',
    source_revision: null,
    primary_metric: 'math_reward',
    primary_metric_label: 'Math reward',
    pass_rate_metric: 'symbolic_correctness',
    pass_rate_basis: 'metric.symbolic_correctness eq 1',
    success_definition: null,
    facet_specs: [
      { field: 'problem_type', dimension: 'problem_type', label: 'Problem type', transform: 'identity' },
      { field: 'level', dimension: 'difficulty', label: 'Difficulty', transform: 'identity' },
    ],
    breakdown_specs: [],
    metrics: [],
  },
  scanned: 3,
  expected: 3,
  included: 3,
  scored: 2,
  mean_reward: 0.5,
  success_rate: 0.5,
  failures: 1,
  truncated: 0,
  slices: [],
  facets: [
    { key: 'problem_type:Algebra', label: 'Algebra', dimension: 'problem_type', dimension_label: 'Problem type', count: 2, mean_reward: 0.5, success_rate: 0.5 },
    { key: 'problem_type:Geometry', label: 'Geometry', dimension: 'problem_type', dimension_label: 'Problem type', count: 1, mean_reward: null, success_rate: null },
    { key: 'difficulty:Level 1', label: 'Level 1', dimension: 'difficulty', dimension_label: 'Difficulty', count: 2, mean_reward: 1, success_rate: 1 },
    { key: 'difficulty:Level 2', label: 'Level 2', dimension: 'difficulty', dimension_label: 'Difficulty', count: 1, mean_reward: 0, success_rate: 0 },
  ],
  breakdowns: [{
    id: 'problem-type-by-difficulty',
    label: 'Problem type × difficulty',
    dimensions: ['problem_type', 'difficulty'],
    dimension_labels: ['Problem type', 'Difficulty'],
    presentation: 'matrix',
    excluded: 0,
    groups: [
      { key: 'algebra-l1', label: 'Algebra · Level 1', values: [{ dimension: 'problem_type', dimension_label: 'Problem type', value: 'Algebra', label: 'Algebra' }, { dimension: 'difficulty', dimension_label: 'Difficulty', value: 'Level 1', label: 'Level 1' }], count: 1, scored: 1, failures: 0, truncated: 0, mean_reward: 1, success_rate: 1 },
      { key: 'algebra-l2', label: 'Algebra · Level 2', values: [{ dimension: 'problem_type', dimension_label: 'Problem type', value: 'Algebra', label: 'Algebra' }, { dimension: 'difficulty', dimension_label: 'Difficulty', value: 'Level 2', label: 'Level 2' }], count: 1, scored: 1, failures: 0, truncated: 0, mean_reward: 0, success_rate: 0 },
      { key: 'geometry-l1', label: 'Geometry · Level 1', values: [{ dimension: 'problem_type', dimension_label: 'Problem type', value: 'Geometry', label: 'Geometry' }, { dimension: 'difficulty', dimension_label: 'Difficulty', value: 'Level 1', label: 'Level 1' }], count: 1, scored: 0, failures: 1, truncated: 0, mean_reward: null, success_rate: null },
    ],
  }],
  traces: [],
  next_cursor: null,
  live: false,
};

afterEach(cleanup);

describe('CapabilityBreakdown', () => {
  it('defaults to the declared compound matrix and keeps coverage visible', () => {
    render(<CapabilityBreakdown evaluation={evaluation} onTraces={vi.fn()} />);

    expect(screen.getByRole('tab', { name: 'Problem type × difficulty' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('columnheader', { name: 'Level 1' })).toBeInTheDocument();
    expect(screen.getAllByRole('columnheader').map((header) => header.textContent)).toEqual([
      'Problem type',
      'Level 1',
      'Level 2',
    ]);
    expect(screen.getByRole('rowheader', { name: 'Algebra' })).toBeInTheDocument();
    expect(screen.getByText('100.0% pass')).toBeInTheDocument();
    expect(screen.getAllByRole('cell', { name: /1\/1 scored/ })).toHaveLength(2);
    expect(screen.getByText('1 error')).toBeInTheDocument();
  });

  it('switches back to an independently filterable dimension', () => {
    render(<CapabilityBreakdown evaluation={evaluation} onTraces={vi.fn()} />);

    fireEvent.click(screen.getByRole('tab', { name: 'Difficulty' }));

    expect(screen.getByRole('tab', { name: 'Difficulty' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByText('Difficulties')).toBeInTheDocument();
    expect(screen.getByText('Level 2')).toBeInTheDocument();
  });
});
