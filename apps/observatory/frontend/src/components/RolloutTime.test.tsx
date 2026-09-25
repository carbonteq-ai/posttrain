import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import type { RolloutTimeView } from '../lib/api';
import { RolloutTimeSummary } from './RolloutTime';

afterEach(cleanup);

const step = (index: number) => ({
  step: index,
  rollouts: 10,
  timed_rollouts: 10,
  inference_ms: 900_000,
  tools_ms: 60_000,
  setup_ms: 40_000,
  scoring_ms: 0,
  rollout_ms: 1_000_000,
  elapsed_ms: 125_000,
});

describe('RolloutTimeSummary', () => {
  it('leads with elapsed time and averages phases per rollout instead of summing concurrent rollouts', () => {
    const view: RolloutTimeView = { state: 'available', live: false, elapsed_ms: 250_000, steps: [step(1), step(2)] };
    render(<RolloutTimeSummary view={view} />);

    // 2 steps x 10 rollouts x 100s each = 2000s summed, over 250s elapsed.
    expect(screen.getByText('4m 10s')).toBeInTheDocument();
    expect(screen.getByText(/~8\.0 rollouts in flight on average/)).toBeInTheDocument();
    expect(screen.getByText('Per rollout (average 1m 40s):')).toBeInTheDocument();
    expect(screen.getByText('1m 30s')).toBeInTheDocument();
    expect(screen.queryByText('33m 20s')).not.toBeInTheDocument();
  });

  it('omits the elapsed headline when the provider has no wall-clock bounds', () => {
    const view: RolloutTimeView = { state: 'available', live: true, elapsed_ms: null, steps: [{ ...step(1), elapsed_ms: null }] };
    render(<RolloutTimeSummary view={view} />);

    expect(screen.queryByText('generating')).not.toBeInTheDocument();
    expect(screen.queryByText(/in flight/)).not.toBeInTheDocument();
  });
});
