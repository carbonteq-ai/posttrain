import type { TraceEvaluation, TraceSummary } from './api';

export type TraceSurfaceMode = 'evaluation' | 'optimization' | 'generic';

export type TracePresentation = {
  mode: TraceSurfaceMode;
  eyebrow: string;
  title: string;
  subtitle: string;
  populationLabel: string;
  passRateAvailable: boolean;
  passRateConfigured: boolean;
  defaultBreakdownMetric: 'success_rate' | 'mean_reward';
  outcomeHeading: string;
  outcomeLabel: (outcome: TraceSummary['outcome']) => string;
};

const optimizationJobKinds = new Set(['train.grpo', 'train.sampo', 'train.distill']);

export function traceSurfaceMode(jobKind: string): TraceSurfaceMode {
  return jobKind.startsWith('eval.')
    ? 'evaluation'
    : optimizationJobKinds.has(jobKind) ? 'optimization' : 'generic';
}

export function tracePresentation(jobKind: string, evaluation: TraceEvaluation): TracePresentation {
  const mode = traceSurfaceMode(jobKind);
  const passRateAvailable = evaluation.success_rate != null
    || evaluation.slices.some((slice) => slice.success_rate != null)
    || evaluation.facets.some((facet) => facet.success_rate != null);
  const passRateConfigured = evaluation.metadata?.pass_rate_metric != null
    || evaluation.traces.some((trace) => trace.success != null);
  const copy = mode === 'evaluation'
    ? {
        eyebrow: 'SUPPORTING EVIDENCE',
        title: 'Traces & evaluation',
        subtitle: 'Inspect the examples behind the Overview verdict. Benchmark identity and aggregate quality stay on Overview.',
        populationLabel: 'traces',
      }
    : mode === 'optimization'
      ? {
          eyebrow: 'ROLLOUTS TO LEARNING SIGNAL',
          title: 'Rollouts & rewards',
          subtitle: 'Reward and rollout evidence link each trajectory back to the learning signal consumed by policy updates.',
          populationLabel: 'rollouts',
        }
      : {
          eyebrow: 'TRACE EVIDENCE',
          title: 'Traces',
          subtitle: 'Inspect request-level evidence without assuming an evaluation or optimization-specific success contract.',
          populationLabel: 'traces',
        };

  return {
    mode,
    ...copy,
    passRateAvailable,
    passRateConfigured,
    defaultBreakdownMetric: mode === 'evaluation' && passRateAvailable ? 'success_rate' : 'mean_reward',
    outcomeHeading: passRateConfigured ? 'Verifier outcome' : 'Trace state',
    outcomeLabel: (outcome) => {
      if (outcome === 'pass') return 'Pass';
      if (outcome === 'review') return passRateConfigured ? 'Fail' : 'Needs review';
      if (outcome === 'scored') return mode === 'optimization' ? 'Rewarded' : 'Scored';
      if (outcome === 'error') return 'Error';
      if (outcome === 'truncated') return 'Truncated';
      return 'Unknown';
    },
  };
}

export function traceSignalColumns(evaluation: TraceEvaluation): Array<{ name: string; label: string }> {
  const primary = evaluation.metadata?.primary_metric;
  const roleOrder = { reward_component: 0, success: 1, diagnostic: 2, primary_reward: 3 } as const;
  const declared = [...(evaluation.metadata?.metrics ?? [])]
    .filter((metric) => metric.name !== primary)
    .sort((left, right) => roleOrder[left.role] - roleOrder[right.role])
    .map(({ name, label }) => ({ name, label }));
  if (declared.length) return declared.slice(0, 4);

  const names = [...new Set(evaluation.traces.flatMap((trace) => Object.keys(trace.metrics)))]
    .filter((name) => name !== primary)
    .sort();
  return names.slice(0, 4).map((name) => ({
    name,
    label: name.replaceAll('_', ' ').replace(/\b\w/g, (character) => character.toUpperCase()),
  }));
}
