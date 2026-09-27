import { cleanup, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

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

import type { NoteData, RenderedNote, RenderedView } from '../lib/api';
import { formatTooltip } from './EvidenceChart';
import { NoteMarkdown, noteChartModel } from './NoteViews';

const curve: NoteData = {
  columns: [
    { name: 'update.step', kind: 'dimension', type: 'integer', unit: null, label: 'Update' },
    { name: 'reward', kind: 'metric', type: 'number', unit: null, label: 'Reward' },
    { name: 'entropy', kind: 'metric', type: 'number', unit: null, label: 'Entropy' },
  ],
  rows: [[1, 0.1, 2.5], [2, 0.3, 2.1], [3, 0.42, 1.9]],
  grain: 'update',
  truncated: false,
};

const byRun: NoteData = {
  columns: [
    { name: 'run', kind: 'dimension', type: 'string', unit: null, label: 'Run' },
    { name: 'reward', kind: 'measure', type: 'number', unit: null, label: 'Reward' },
    { name: 'status', kind: 'value', type: 'string', unit: null, label: null },
  ],
  rows: [['run-a', 0.42, 'succeeded'], ['run-b', 0.3, 'failed']],
  grain: 'run',
  truncated: false,
};

function note(markdown: string, views: RenderedView[]): RenderedNote {
  return { markdown, text: markdown, views, data: {}, unresolved: [], template: null };
}

const chartView: RenderedView = {
  index: 0,
  kind: 'chart',
  data: 'curve',
  options: { x: 'update.step', y: ['reward', 'entropy'], title: 'Reward and entropy by update' },
  query: 'measures: [reward, entropy]\nby: update.step',
  result: curve,
};

const valueView: RenderedView = {
  index: 1,
  kind: 'value',
  data: 'runs',
  options: { column: 'reward', label: 'Final reward', where: 'run = run-a', compare: 'run = run-b' },
  query: 'measures: [reward:last]\nby: run',
  result: byRun,
  formatted: '0.42',
  compare_formatted: '0.30',
  difference: '+0.12',
};

const tableView: RenderedView = {
  index: 2,
  kind: 'table',
  data: 'runs',
  options: { columns: ['run', 'reward'], title: 'Runs compared' },
  query: 'measures: [reward:last]\nby: run',
  result: byRun,
};

describe('NoteMarkdown', () => {
  afterEach(cleanup);

  beforeEach(() => {
    chartMocks.setOption.mockClear();
  });

  it('draws note-view fenced blocks with their chart, value and table components', () => {
    const { container } = render(<NoteMarkdown rendered={note(
      'Intro paragraph.\n\n```note-view 0\n```\n\n```note-view 1\n```\n\n```note-view 2\n```\n\n```python\nprint("kept as code")\n```\n',
      [chartView, valueView, tableView],
    )} />);

    expect(screen.getByText('Intro paragraph.')).toBeVisible();

    const chart = screen.getByRole('img', { name: 'Reward and entropy by update' });
    expect(chart).toBeInTheDocument();
    const option = chartMocks.setOption.mock.calls.at(-1)?.[0];
    expect(option.series.map((item: { name: string }) => item.name)).toEqual(['reward', 'entropy']);
    expect(option.series[0].data).toEqual([[1, 0.1], [2, 0.3], [3, 0.42]]);
    expect(option.series[0].type).toBe('line');
    expect(option.xAxis[0].name).toBe('Update');

    const value = screen.getByRole('figure', { name: 'Note value' });
    expect(within(value).getByText('Final reward')).toBeVisible();
    expect(within(value).getByText('0.42')).toBeVisible();
    expect(within(value).getByText('0.30')).toBeVisible();
    expect(within(value).getByText('+0.12')).toBeVisible();

    const table = screen.getByRole('table');
    expect(within(table).getAllByRole('columnheader').map((cell) => cell.textContent)).toEqual(['Run', 'Reward']);
    const rewardCell = within(table).getByText('0.42');
    expect(rewardCell).toHaveClass('text-right');
    expect(within(table).getByText('run-a')).not.toHaveClass('text-right');
    expect(within(table).queryByText('succeeded')).toBeNull();

    // Each view carries a collapsed query disclosure; data blocks are not shown.
    expect(screen.getAllByText('Query')).toHaveLength(3);
    const firstQuery = container.querySelector('figure details');
    expect(firstQuery).not.toHaveAttribute('open');
    expect(firstQuery?.textContent).toContain('by: update.step');

    // Any other fenced block stays a code block.
    const code = container.querySelector('pre code.language-python');
    expect(code?.textContent).toContain('print("kept as code")');
    expect(container.querySelector('pre code.language-note-view')).toBeNull();
  });

  it('shows a view error as a visible unresolved marker instead of an empty chart', () => {
    render(<NoteMarkdown rendered={note('```note-view 0\n```\n\n```note-view 7\n```\n', [{
      index: 0,
      kind: 'chart',
      data: 'curve',
      options: { x: 'step', y: 'loss' },
      query: 'measures: [loss]',
      result: null,
      error: "no column 'step' (columns: update.step, reward)",
    }])} />);

    const marker = screen.getByRole('note', { name: 'Unresolved chart' });
    expect(marker).toHaveTextContent("no column 'step'");
    expect(screen.queryByRole('img')).toBeNull();
    expect(chartMocks.setOption).not.toHaveBeenCalled();
    expect(screen.getByRole('note', { name: 'Unresolved view' })).toHaveTextContent('view 7 is missing');
  });

  it('never renders raw HTML from a note', () => {
    const { container } = render(<NoteMarkdown rendered={note(
      'Safe text <script>alert(1)</script> and <img src="x" onerror="alert(2)">.\n\n<div onclick="alert(3)">block html</div>\n\n<iframe src="https://example.com"></iframe>\n\n[bad link](javascript:alert(4))\n',
      [],
    )} />);

    expect(screen.getByText(/Safe text/)).toBeVisible();
    expect(container.querySelector('script')).toBeNull();
    expect(container.querySelector('img')).toBeNull();
    expect(container.querySelector('iframe')).toBeNull();
    expect(container.querySelector('[onerror]')).toBeNull();
    expect(container.querySelector('[onclick]')).toBeNull();
    // Inline tags are dropped; any text between them stays inert text.
    for (const element of container.querySelectorAll('*')) {
      for (const attribute of element.attributes) expect(attribute.value).not.toContain('alert(');
    }
    expect(screen.queryByText('block html')).toBeNull();
    const link = screen.getByText('bad link');
    expect(link.getAttribute('href') ?? '').not.toContain('javascript');
  });
});

describe('noteChartModel', () => {
  it('draws one line per distinct series value and uses categories for non-numeric x', () => {
    const view: RenderedView = {
      index: 0,
      kind: 'chart',
      options: { x: 'split', y: 'reward', series: 'run', type: 'bar' },
      result: {
        ...byRun,
        columns: [
          { name: 'run', kind: 'dimension', type: 'string', unit: null, label: 'Run' },
          { name: 'split', kind: 'dimension', type: 'string', unit: null, label: 'Split' },
          { name: 'reward', kind: 'measure', type: 'number', unit: 'ratio', label: 'Reward' },
        ],
        rows: [['run-a', 'train', 0.5], ['run-a', 'heldout', 0.4], ['run-b', 'train', 0.3], ['run-b', 'heldout', null]],
      },
    };

    const model = noteChartModel(view);
    expect(model?.xAxis.categories).toEqual(['train', 'heldout']);
    expect(model?.series.map((item) => model.labels[item.name])).toEqual(['run-a', 'run-b']);
    expect(model?.series[0].points).toEqual([{ value: 0.5, step: 0 }, { value: 0.4, step: 1 }]);
    expect(model?.series[1].points).toEqual([{ value: 0.3, step: 0 }]);
    expect(Object.values(model?.units ?? {})).toEqual(['ratio', 'ratio']);

    const tooltip = formatTooltip({ axisValue: 'heldout', seriesName: model!.series[0].name, value: [1, 0.4] }, model!.labels, model!.units, 'logical-step', model!.xAxis);
    expect(tooltip).toContain('Split heldout');
    expect(tooltip).toContain('40.0%');
  });
});
