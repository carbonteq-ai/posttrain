import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('echarts/core', () => ({
  use: vi.fn(),
  init: vi.fn(() => ({
    setOption: vi.fn(),
    on: vi.fn(),
    dispatchAction: vi.fn(),
    getZr: vi.fn(() => ({ on: vi.fn() })),
    resize: vi.fn(),
    dispose: vi.fn(),
  })),
}));

import type { EvaluationIndex, EvaluationRecord } from '../lib/api';
import { baseEvaluations, evaluatedTrainingRuns, evaluationsByParent, runSuites, taskRows } from '../lib/evaluations';
import { EvalComparePage, RunEvaluations } from './EvaluationViews';

const SUITE = 'eval/model/heldout';

function record(runId: string, overrides: Partial<EvaluationRecord> = {}): EvaluationRecord {
  return {
    run_key: `key-${runId}`,
    run_id: runId,
    job_kind: 'eval.general',
    suite: SUITE,
    environment: 'env',
    model: 'models/m',
    parent_run: null,
    parent_run_key: null,
    parent_step: null,
    status: 'succeeded',
    started_at: '2026-09-27T00:00:00Z',
    attempts: 60,
    score: 0.5,
    truncated: 0,
    failed: 0,
    ...overrides,
  };
}

function checkpoint(runId: string, parent: string, step: number, score: number, overrides: Partial<EvaluationRecord> = {}): EvaluationRecord {
  return record(runId, { parent_run: parent, parent_run_key: `key-${parent}`, parent_step: step, score, ...overrides });
}

const index: EvaluationIndex = {
  source_id: 'src',
  score_definition: 'Mean rollout reward.',
  behaviour_definition: 'Per-episode means.',
  records: [
    record('base-1', { score: 0.6, turns: 6.2, turns_completed: 5.9, tool_calls: 9.0, output_tokens: 3400, thinking_tokens: null, endings: { turn_limit: 2, completed: 58 } }),
    record('base-2', { score: 0.58, turns: 6.0, turns_completed: 5.8, tool_calls: 8.6, output_tokens: 3300, thinking_tokens: null }),
    record('base-other-model', { model: 'models/other', score: 0.9 }),
    checkpoint('a-150', 'train-a', 150, 0.57),
    checkpoint('a-100', 'train-a', 100, 0.62, { turns: 5.51, turns_completed: 5.49, tool_calls: 9.1, output_tokens: 3127.89, thinking_tokens: 1840.4, endings: { completed: 60 } }),
    checkpoint('a-100-failed', 'train-a', 100, 0.7, { status: 'failed', started_at: '2026-09-26T00:00:00Z' }),
    checkpoint('b-20', 'train-b', 20, 0.55, { turns: 7.04, tool_calls: 10.2, output_tokens: 4012.2 }),
    checkpoint('c-20', 'train-c', 20, 0.5, { model: 'models/other' }),
  ],
};

function taskScores(keys: string[]) {
  const table: Record<string, Record<string, number>> = {
    'key-base-1': { 'task-a': 0.5, 'task-b': 1 },
    'key-base-2': { 'task-a': 0.7, 'task-b': 1 },
    'key-a-100': { 'task-a': 0.9, 'task-b': 1 },
    'key-a-150': { 'task-a': 0.2, 'task-b': 1 },
    'key-b-20': { 'task-a': 0.4, 'task-b': 0.5 },
  };
  return keys.flatMap((key) => Object.entries(table[key] ?? {}).map(([task, score]) => ({ run_key: key, task, attempts: 3, score, truncated: 0, failed: 0 })));
}

function stubTaskFetch() {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = new URL(String(input), 'http://localhost');
    return new Response(JSON.stringify({ source_id: 'src', score_definition: 'x', scores: taskScores(url.searchParams.getAll('run_key')) }));
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('evaluation grouping', () => {
  it('orders checkpoints by step and finds the base runs of the same model and suite', () => {
    expect(evaluationsByParent(index).get('train-a')?.map((item) => item.run_id)).toEqual(['a-100-failed', 'a-100', 'a-150']);
    expect(baseEvaluations(index, SUITE, 'models/m').map((item) => item.run_id)).toEqual(['base-1', 'base-2']);
    const [suite] = runSuites(index, 'train-a');
    expect(suite.baseScore).toBeCloseTo(0.59);
    expect(evaluatedTrainingRuns(index).map((run) => [run.runId, run.model])).toEqual([
      ['train-a', 'models/m'], ['train-b', 'models/m'], ['train-c', 'models/other'],
    ]);
  });

  it('averages several runs into one task column', () => {
    const rows = taskRows(taskScores(['key-base-1', 'key-base-2', 'key-a-150']), [
      { key: 'base', runKeys: ['key-base-1', 'key-base-2'] },
      { key: 'a', runKeys: ['key-a-150'] },
    ]);
    expect(rows[0]).toEqual({ task: 'task-a', values: { base: 0.6, a: 0.2 } });
  });
});

describe('RunEvaluations', () => {
  it('shows base and each checkpoint with its difference and compares tasks against base', async () => {
    const fetchMock = stubTaskFetch();
    const onOpenRun = vi.fn();
    const { container } = render(<RunEvaluations runId="train-a" index={index} loading={false} onOpenRun={onOpenRun} />);
    const view = within(container);
    const table = view.getByRole('table', { name: `Base and checkpoints on heldout` });
    const rows = within(table).getAllByRole('row').slice(1).map((row) => within(row).getAllByRole('cell').slice(0, 4).map((cell) => cell.textContent));
    expect(rows).toEqual([
      ['Base model', 'base-1', '0.600', ''],
      ['Base model', 'base-2', '0.580', ''],
      ['step 100', 'a-100-failed', '0.700', '—'],
      ['step 100', 'a-100', '0.620', '+0.030'],
      ['step 150', 'a-150', '0.570', '−0.020'],
    ]);
    expect(view.getByText('(not compared)')).toBeInTheDocument();
    // Behaviour sits beside the score: turns, turns on completed episodes, tool calls, output and thinking tokens.
    const behaviour = within(table).getAllByRole('row').slice(1).map((row) => within(row).getAllByRole('cell').slice(4, 9).map((cell) => cell.textContent));
    expect(behaviour).toEqual([
      ['6.2', '5.9', '9.0', '3,400', '—'],
      ['6.0', '5.8', '8.6', '3,300', '—'],
      ['—', '—', '—', '—', '—'],
      ['5.5', '5.5', '9.1', '3,128', '1,840'],
      ['—', '—', '—', '—', '—'],
    ]);
    const baseRow = within(table).getAllByRole('row')[1]!;
    const baseCells = within(baseRow).getAllByRole('cell');
    // A value no episode recorded is shown as missing, never as zero.
    expect(baseCells[8]).toHaveAttribute('title', 'Not recorded for this run');
    // The truncated count names how every episode ended.
    expect(baseCells[10]).toHaveAttribute('title', 'completed 58 · turn limit 2');
    // The failed evaluation is not read for task scores.
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(String(fetchMock.mock.calls[0][0])).not.toContain('a-100-failed');
    const tasks = view.getByRole('region', { name: 'Score by task' });
    await waitFor(() => expect(within(tasks).getByRole('rowheader', { name: 'task-a' })).toBeInTheDocument());
    const taskA = within(tasks).getByRole('rowheader', { name: 'task-a' }).closest('tr')!;
    expect(within(taskA).getAllByRole('cell').map((cell) => cell.textContent)).toEqual(['0.600', '0.900', '0.200', '−0.400']);
    await userEvent.click(view.getByRole('button', { name: /a-150/ }));
    expect(onOpenRun).toHaveBeenCalledWith('key-a-150');
  });

  it('says so when no evaluation records a checkpoint of the run', () => {
    const { container } = render(<RunEvaluations runId="train-z" index={index} loading={false} onOpenRun={vi.fn()} />);
    expect(within(container).getByText('No evaluation run records a checkpoint of this run.')).toBeInTheDocument();
  });
});

describe('EvalComparePage', () => {
  it('pairs two runs of the same base model on a shared suite, overall and per task', async () => {
    stubTaskFetch();
    const { container } = render(<EvalComparePage index={index} loading={false} displayName={(runId) => `name ${runId}`} onOpenRun={vi.fn()} initialRunId="train-a" />);
    const view = within(container);
    expect(view.getByRole('combobox', { name: 'Run A' })).toHaveValue('train-a');
    // Only runs from the same base model are offered as run B.
    const runB = view.getByRole('combobox', { name: 'Run B (same base model)' });
    expect(within(runB).getAllByRole('option').map((option) => option.textContent)).toEqual(['name train-b']);
    expect(view.getByRole('combobox', { name: 'Run A checkpoint' })).toHaveValue('key-a-150');
    const comparison = view.getByRole('region', { name: 'Comparison' });
    expect(within(comparison).getByText('−0.020')).toBeInTheDocument();
    await userEvent.selectOptions(view.getByRole('combobox', { name: 'Run A checkpoint' }), 'key-a-100');
    expect(within(comparison).getByText('−0.070')).toBeInTheDocument();
    const tasks = within(comparison).getByRole('region', { name: 'Score by task' });
    await waitFor(() => expect(within(tasks).getByRole('rowheader', { name: 'task-a' })).toBeInTheDocument());
    const taskA = within(tasks).getByRole('rowheader', { name: 'task-a' }).closest('tr')!;
    expect(within(taskA).getAllByRole('cell').map((cell) => cell.textContent)).toEqual(['0.600', '0.900', '0.400', '+0.300', '−0.200', '−0.500']);
    // Episode behaviour for base (mean of its runs), A and B.
    const episodes = within(comparison).getByRole('table', { name: 'How episodes went' });
    const cells = (label: string) => within(within(episodes).getByRole('rowheader', { name: label }).closest('tr')!).getAllByRole('cell').map((cell) => cell.textContent);
    expect(cells('Turns')).toEqual(['6.1', '5.5', '7.0']);
    expect(cells('Tool calls')).toEqual(['8.8', '9.1', '10.2']);
    expect(cells('Thinking tokens')).toEqual(['—', '1,840', '—']);
    expect(cells('Endings')).toEqual(['2 runs', 'completed 60', '—']);
  });

  it('explains when fewer than two training runs have evaluated checkpoints', () => {
    const single: EvaluationIndex = { ...index, records: index.records.filter((item) => !item.parent_run || item.parent_run === 'train-a') };
    const { container } = render(<EvalComparePage index={single} loading={false} displayName={(runId) => runId} onOpenRun={vi.fn()} />);
    expect(within(container).getByText(/needs two training runs with evaluated checkpoints; this project has 1/)).toBeInTheDocument();
  });
});
