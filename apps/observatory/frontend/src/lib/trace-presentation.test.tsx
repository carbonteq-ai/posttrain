import { describe, expect, it } from 'vitest';

import type { TraceEvaluation } from './api';
import { tracePresentation, traceSignalColumns, traceSurfaceMode } from './trace-presentation';

function evaluation(overrides: Partial<TraceEvaluation> = {}): TraceEvaluation {
  return {
    state: 'complete',
    metadata: null,
    scanned: 1,
    expected: 1,
    included: 1,
    scored: 1,
    mean_reward: 0.5,
    success_rate: null,
    failures: 0,
    truncated: 0,
    slices: [],
    facets: [],
    breakdowns: [],
    traces: [],
    next_cursor: null,
    live: false,
    ...overrides,
  };
}

describe('tracePresentation', () => {
  it('routes all rollout-producing training jobs to the optimization surface', () => {
    expect(traceSurfaceMode('train.grpo')).toBe('optimization');
    expect(traceSurfaceMode('train.sampo')).toBe('optimization');
    expect(traceSurfaceMode('train.distill')).toBe('optimization');
    expect(traceSurfaceMode('serve.benchmark')).toBe('generic');
  });

  it('uses configured pass semantics for evaluation runs', () => {
    const view = tracePresentation('eval.general', evaluation({
      success_rate: 0.75,
      metadata: {
        key: 'ifeval',
        label: 'IFEval',
        category: null,
        package: null,
        dataset: null,
        dataset_revision: null,
        split: null,
        source_revision: null,
        primary_metric: 'strict_prompt_accuracy',
        primary_metric_label: 'Strict prompt accuracy',
        pass_rate_metric: 'strict_prompt_accuracy',
        pass_rate_basis: 'configured binary metric',
        success_definition: null,
        facet_specs: [],
        breakdown_specs: [],
        metrics: [],
      },
    }));

    expect(view.defaultBreakdownMetric).toBe('success_rate');
    expect(view.outcomeLabel('review')).toBe('Fail');
    expect(view.outcomeHeading).toBe('Verifier outcome');
  });

  it('keeps optimization rollouts reward-led when no pass predicate exists', () => {
    const view = tracePresentation('train.grpo', evaluation());

    expect(view.title).toBe('Rollouts & rewards');
    expect(view.defaultBreakdownMetric).toBe('mean_reward');
    expect(view.outcomeLabel('scored')).toBe('Rewarded');
    expect(view.outcomeLabel('review')).toBe('Needs review');
  });

  it('does not turn reward-only evaluation traces into failed examples', () => {
    const view = tracePresentation('eval.general', evaluation());

    expect(view.passRateConfigured).toBe(false);
    expect(view.defaultBreakdownMetric).toBe('mean_reward');
    expect(view.outcomeLabel('scored')).toBe('Scored');
  });
});

describe('traceSignalColumns', () => {
  it('prioritizes declared reward components and excludes the primary metric', () => {
    const columns = traceSignalColumns(evaluation({
      metadata: {
        key: 'example',
        label: 'Example',
        category: null,
        package: null,
        dataset: null,
        dataset_revision: null,
        split: null,
        source_revision: null,
        primary_metric: 'reward',
        primary_metric_label: 'Reward',
        pass_rate_metric: null,
        pass_rate_basis: null,
        success_definition: null,
        facet_specs: [],
        breakdown_specs: [],
        metrics: [
          { name: 'reward', label: 'Reward', role: 'primary_reward' },
          { name: 'format', label: 'Format', role: 'diagnostic' },
          { name: 'correctness', label: 'Correctness', role: 'reward_component' },
        ],
      },
    }));

    expect(columns).toEqual([
      { name: 'correctness', label: 'Correctness' },
      { name: 'format', label: 'Format' },
    ]);
  });

  it('keeps four declared signals as separate table columns', () => {
    const columns = traceSignalColumns(evaluation({
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
        metrics: [
          { name: 'partial_credit', label: 'Partial credit', role: 'primary_reward' },
          { name: 'task_completed_correctly', label: 'Task completed correctly', role: 'success' },
          { name: 'assertions_excluded', label: 'Assertions excluded', role: 'diagnostic' },
          { name: 'assertions_passed', label: 'Assertions passed', role: 'diagnostic' },
          { name: 'assertions_scored', label: 'Assertions scored', role: 'diagnostic' },
        ],
      },
    }));

    expect(columns.map((column) => column.name)).toEqual([
      'task_completed_correctly',
      'assertions_excluded',
      'assertions_passed',
      'assertions_scored',
    ]);
  });
});
