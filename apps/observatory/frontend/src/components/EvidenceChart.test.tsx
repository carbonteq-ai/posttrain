import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const chartMocks = vi.hoisted(() => ({
  setOption: vi.fn(),
  on: vi.fn(),
  dispatchAction: vi.fn(),
  zrenderOn: vi.fn(),
  getZr: vi.fn(() => ({ on: chartMocks.zrenderOn })),
  resize: vi.fn(),
  dispose: vi.fn(),
}));

vi.mock('echarts/core', () => ({
  use: vi.fn(),
  init: vi.fn(() => chartMocks),
}));

import {
  EvidenceChart,
  formatElapsedAxis,
  formatElapsedDuration,
  formatTooltip,
  scaleGroup,
} from './EvidenceChart';

describe('EvidenceChart tooltip', () => {
  it('shows one step header with normalized labels and unit-aware values', () => {
    const html = formatTooltip([
      {
        axisValue: 91,
        color: '#6356c7',
        seriesName: 'train/rl/rollout_tokens_per_second',
        value: [91, 706.3101098],
      },
      {
        axisValue: 91,
        color: '#148c87',
        seriesName: 'train/rl/group_zero_variance_fraction',
        value: [91, 0.125],
      },
    ], {
      'train/rl/rollout_tokens_per_second': 'Rollout throughput',
      'train/rl/group_zero_variance_fraction': 'Zero-variance groups',
    }, {
      'train/rl/rollout_tokens_per_second': 'tokens/s',
      'train/rl/group_zero_variance_fraction': 'ratio',
    });

    expect(html.match(/Step 91/g)).toHaveLength(1);
    expect(html).toContain('Rollout throughput');
    expect(html).toContain('706 tokens/s');
    expect(html).toContain('Zero-variance groups');
    expect(html).toContain('12.5%');
    expect(html).not.toContain('train/rl/');
    expect(html).not.toContain('<strong');
  });

  it('escapes labels supplied by a tracking backend', () => {
    const html = formatTooltip({
      axisValue: 3,
      color: '#6356c7',
      seriesName: 'unsafe',
      value: [3, 1],
    }, { unsafe: '<script>alert(1)</script>' }, {});

    expect(html).toContain('&lt;script&gt;alert(1)&lt;/script&gt;');
    expect(html).not.toContain('<script>');
  });

  it('labels timestamped evidence with wall-clock and elapsed time', () => {
    const observedAt = Date.parse('2026-08-06T12:34:56Z');
    const html = formatTooltip({
      axisValue: 3665,
      color: '#6356c7',
      data: [3665, 77, observedAt],
      seriesName: 'system/gpu_utilization',
      value: [3665, 77, observedAt],
    }, { 'system/gpu_utilization': 'GPU utilization' }, { 'system/gpu_utilization': '%' }, 'elapsed-time');

    expect(html).toContain('Aug');
    expect(html).toContain('+1h 1m');
    expect(html).toContain('GPU utilization');
    expect(html).not.toContain('Step 3,665');
    expect(formatElapsedDuration(90_000)).toBe('1d 1h');
    expect(formatElapsedAxis(3600, 64_000)).toBe('1h');
  });
});

describe('EvidenceChart scale policy', () => {
  beforeEach(() => {
    chartMocks.setOption.mockClear();
    chartMocks.on.mockClear();
    chartMocks.dispatchAction.mockClear();
    chartMocks.zrenderOn.mockClear();
    chartMocks.getZr.mockClear();
  });

  it('publishes pointer steps and clears the shared hover when the pointer leaves', () => {
    const onHoverStep = vi.fn();
    render(<EvidenceChart
      ariaLabel="Shared hover source"
      onHoverStep={onHoverStep}
      series={[{ name: 'train/loss', points: [{ step: 4, value: 1.2 }] }]}
    />);

    const pointerHandler = chartMocks.on.mock.calls.find(([event]) => event === 'updateAxisPointer')?.[1];
    expect(pointerHandler).toBeTypeOf('function');
    fireEvent.mouseEnter(screen.getByRole('img', { name: 'Shared hover source' }));
    pointerHandler({ axesInfo: [{ value: 4 }] });
    expect(onHoverStep).toHaveBeenLastCalledWith(4);

    fireEvent.mouseMove(document.body);
    expect(onHoverStep).toHaveBeenLastCalledWith(null);
  });

  it('opens a sibling tooltip at the nearest recorded logical step', () => {
    render(<EvidenceChart
      ariaLabel="Shared hover target"
      hoveredStep={8}
      series={[{
        name: 'train/loss',
        points: [
          { step: 1, value: 2.1 },
          { step: 7, value: 1.6 },
          { step: 12, value: 1.1 },
        ],
      }]}
    />);

    expect(chartMocks.dispatchAction).toHaveBeenCalledWith({
      type: 'showTip',
      seriesIndex: 0,
      dataIndex: 1,
    });
  });

  it('plots timestamped system evidence by elapsed run time', () => {
    const origin = '2026-08-06T19:35:53Z';
    render(<EvidenceChart
      ariaLabel="System evidence over run time"
      compact
      xDomain="elapsed-time"
      xOrigin={origin}
      xRange={[60_000, 64_200]}
      series={[{
        name: 'system/gpu_utilization',
        points: [
          { step: 0, observed_at: origin, value: 37 },
          { step: 3000, observed_at: '2026-08-07T13:25:53Z', value: 27 },
        ],
      }]}
    />);

    const option = chartMocks.setOption.mock.calls[0][0];
    expect(option.xAxis[0].name).toBe('Elapsed run time');
    expect(option.xAxis[0].scale).toBe(true);
    expect(option.xAxis[0].min).toBe(60_000);
    expect(option.xAxis[0].max).toBe(64_200);
    expect(option.xAxis[0].axisLabel.formatter(3600)).toBe('1h');
    expect(option.series[0].data[0]).toEqual([0, 37, Date.parse(origin)]);
    expect(option.series[0].data[1][0]).toBe(64_200);
    expect(screen.getByText(/plotted by elapsed run time/)).toBeInTheDocument();
  });

  it('groups metrics only when their numeric scales are compatible', () => {
    const units = { 'train/rl/group_zero_variance_fraction': 'ratio' };
    expect(scaleGroup('train/rl/reward_mean', units)).toBe('rl-reward');
    expect(scaleGroup('train/rl/reward_std', units)).toBe('rl-reward');
    expect(scaleGroup('train/rl/group_zero_variance_fraction', units)).toBe('unit:ratio');
  });

  it('keeps two scale families together for direct comparison', () => {
    render(<EvidenceChart
      ariaLabel="Learning signal"
      metricUnits={{ 'train/rl/group_zero_variance_fraction': 'ratio' }}
      series={[
        { name: 'train/rl/reward_mean', points: [{ step: 1, value: -0.2 }] },
        { name: 'train/rl/reward_std', points: [{ step: 1, value: 0.4 }] },
        { name: 'train/rl/group_zero_variance_fraction', points: [{ step: 1, value: 0.1 }] },
      ]}
    />);

    const option = chartMocks.setOption.mock.calls[0][0];
    expect(option.grid).not.toBeInstanceOf(Array);
    expect(option.yAxis).toHaveLength(2);
    expect(option.series.map((item: { yAxisIndex: number }) => item.yAxisIndex)).toEqual([0, 0, 1]);
  });

  it('uses synchronized small multiples instead of a third scale', () => {
    render(<EvidenceChart
      ariaLabel="Policy optimization"
      metricUnits={{ 'train/rl/clip_fraction': 'ratio' }}
      series={[
        { name: 'train/rl/policy_loss', points: [{ step: 1, value: -0.2 }] },
        { name: 'train/rl/entropy', points: [{ step: 1, value: 0.7 }] },
        { name: 'train/rl/clip_fraction', points: [{ step: 1, value: 0.1 }] },
      ]}
    />);

    const option = chartMocks.setOption.mock.calls[0][0];
    expect(option.grid).toHaveLength(3);
    expect(option.xAxis).toHaveLength(3);
    expect(option.yAxis).toHaveLength(3);
    expect(option.yAxis.every((axis: { position: string }) => axis.position === 'left')).toBe(true);
    expect(option.axisPointer.link).toEqual([{ xAxisIndex: 'all' }]);
  });

  it('does not allocate an axis or panel for an unrecorded series', () => {
    render(<EvidenceChart
      ariaLabel="Partially recorded optimization evidence"
      metricUnits={{ 'train/rl/clip_fraction': 'ratio' }}
      series={[
        { name: 'train/rl/policy_loss', points: [{ step: 1, value: -0.2 }] },
        { name: 'train/rl/entropy', points: [] },
        { name: 'train/rl/kl', points: [] },
        { name: 'train/rl/clip_fraction', points: [{ step: 1, value: 0.1 }] },
      ]}
    />);

    const option = chartMocks.setOption.mock.calls[0][0];
    expect(option.grid).not.toBeInstanceOf(Array);
    expect(option.yAxis).toHaveLength(2);
    expect(option.series.map((item: { name: string }) => item.name)).toEqual([
      'train/rl/policy_loss',
      'train/rl/clip_fraction',
    ]);
  });

  it('connects provider points in logical-step order', () => {
    render(<EvidenceChart
      ariaLabel="Out-of-order provider history"
      series={[{
        name: 'train/loss',
        points: [
          { step: 13, value: 2.4 },
          { step: 2, value: 1.8 },
          { step: 14, value: 1.3 },
          { step: 0, value: 4.1 },
        ],
      }]}
    />);

    const option = chartMocks.setOption.mock.calls[0][0];
    expect(option.series[0].data).toEqual([
      [0, 4.1],
      [2, 1.8],
      [13, 2.4],
      [14, 1.3],
    ]);
  });
});
