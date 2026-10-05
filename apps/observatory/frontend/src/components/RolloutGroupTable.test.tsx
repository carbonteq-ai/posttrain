import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { PromptGroupRewardView, TraceSummary } from '../lib/api';
import { buildRolloutGroups, RolloutGroupTable } from './RolloutGroupTable';

function trace(id: string, step: number, group: string, task: string, reward: number): TraceSummary {
  return {
    external_id: id,
    trace_type: 'verifiers',
    optimizer_step: step,
    prompt_group_id: group,
    prompt_preview: task,
    task,
    task_label: task,
    task_metadata: null,
    reward,
    success: null,
    outcome: 'scored',
    truncated: false,
    error: null,
    tool_calls: 0,
    model_calls: 1,
    input_tokens: 10,
    completion_tokens: 10,
    latency_ms: 1,
    tokens: 10,
    response_tokens: null,
    response_chars: null,
    thinking_tokens: null,
    thinking_chars: null,
    reward_components: {},
    native_metrics: {},
    metrics: {},
  };
}

describe('RolloutGroupTable', () => {
  const traces = [
    trace('a1', 1, 'step/1/group/1', 'task-a', 0),
    trace('a2', 1, 'step/1/group/1', 'task-a', 1),
    trace('b1', 1, 'step/1/group/2', 'task-b', 0.5),
    trace('b2', 1, 'step/1/group/2', 'task-b', 0.5),
    trace('a3', 2, 'step/2/group/1', 'task-a', 1),
    trace('a4', 2, 'step/2/group/1', 'task-a', 1),
  ];
  const rewards: PromptGroupRewardView = {
    state: 'complete', expected_group_size: 2, fact_rows: 6, recorded_traces: 6, live: true,
    groups: [
      { group_id: 'step/1/group/1', step: 1, task_id: 'task-a', rollouts: 2, reward_coverage: 2, current: { mean: 0.5, std: 0.5 }, prior: null, prior_step: null, prior_rollouts: null },
      { group_id: 'step/1/group/2', step: 1, task_id: 'task-b', rollouts: 2, reward_coverage: 2, current: { mean: 0.5, std: 0 }, prior: null, prior_step: null, prior_rollouts: null },
      { group_id: 'step/2/group/1', step: 2, task_id: 'task-a', rollouts: 2, reward_coverage: 2, current: { mean: 1, std: 0 }, prior: { mean: 0.5, std: 0.5 }, prior_step: 1, prior_rollouts: 2 },
    ],
  };

  it('groups by recorded group identity and shows previous complete reward evidence', () => {
    const groups = buildRolloutGroups(traces, rewards);
    expect(groups).toHaveLength(3);
    expect(groups.map((group) => group.step)).toEqual([2, 1, 1]);
    expect(groups[0].prior).toEqual({ mean: 0.5, std: 0.5 });
    expect(groups[0].current).toEqual({ mean: 1, std: 0 });
    expect(groups[2].current).toEqual({ mean: 0.5, std: 0.5 });
    expect(buildRolloutGroups(traces.slice(0, 1), rewards)[0].current).toEqual({ mean: 0.5, std: 0.5 });
    expect(buildRolloutGroups(traces, null)[0].current).toBeNull();
  });

  it('shows a step as its collection number when collections feed several updates', () => {
    // Collections start at updates 1 and 5: the rollouts recorded at update 5 belong to step 2.
    const collected = traces.map((item) => (item.optimizer_step === 2 ? { ...item, optimizer_step: 5 } : item));
    const shifted = { ...rewards, groups: rewards.groups.map((group) => (group.step === 2 ? { ...group, step: 5 } : group)) };
    const view = render(<RolloutGroupTable traces={collected} allLoadedTraces={collected} total={6} expectedSize={2} rewards={shifted} hasMore={false} loadingMore={false} onLoadMore={vi.fn()} onSelect={vi.fn()} stepLabel={(step) => (step === 5 ? 2 : step)} />);
    const steps = view.getAllByRole('row', { name: /task-a/ }).map((row) => within(row).getAllByRole('cell')[0].textContent);
    expect(steps.sort()).toEqual(['1', '2']);
    view.unmount();
  });

  it('keeps prior reward chronological when loaded traces arrive newest first', () => {
    const groups = buildRolloutGroups([...traces].reverse(), rewards);
    expect(groups[0].id).toBe('step/2/group/1');
    expect(groups[0].prior).toEqual({ mean: 0.5, std: 0.5 });
  });

  it('expands a group into selectable rollouts without inventing update selection', () => {
    const onSelect = vi.fn();
    render(<RolloutGroupTable traces={traces} allLoadedTraces={traces} total={6} expectedSize={2} rewards={rewards} hasMore={false} loadingMore={false} onLoadMore={vi.fn()} onSelect={onSelect} />);
    expect(screen.getByRole('table', { name: 'Rollout prompt groups' })).toBeInTheDocument();
    expect(screen.queryByText('Not recorded')).not.toBeInTheDocument();
    expect(screen.getByText(/Update selection is not recorded/)).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'Std dev' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'Previous' })).toBeInTheDocument();
    expect(screen.getByText('0.500 / 0.500')).toBeInTheDocument();
    fireEvent.click(screen.getAllByRole('button', { name: 'task-a' })[1]);
    const child = screen.getByRole('row', { name: 'Rollout a1' });
    expect(within(child).getAllByRole('cell')).toHaveLength(10);
    expect(within(child).getByText('0.000')).toBeInTheDocument();
    expect(within(child).getByText('10')).toBeInTheDocument();
    fireEvent.click(within(child).getByRole('button', { name: /a1 task-a/ }));
    expect(onSelect).toHaveBeenCalledWith(traces[0]);
  });

  it('keeps reward components, tool calls, and honest token provenance in aligned group and rollout columns', () => {
    const detailed = [
      { ...trace('a1', 1, 'step/1/group/1', 'task-a', 0.25), model_calls: 3, tool_calls: 2, thinking_tokens: 6, response_tokens: 4, completion_tokens: 10, reward_components: { partial_credit: 0.5 } },
      { ...trace('a2', 1, 'step/1/group/1', 'task-a', 0.75), model_calls: 5, tool_calls: 4, thinking_tokens: null, response_tokens: null, completion_tokens: 12, reward_components: { partial_credit: 1 } },
    ];
    const { container } = render(<RolloutGroupTable traces={detailed} allLoadedTraces={detailed} total={2} expectedSize={2} rewards={rewards} metricColumns={[{ name: 'partial_credit', label: 'Partial credit' }]} hasMore={false} loadingMore={false} onLoadMore={vi.fn()} onSelect={vi.fn()} />);
    const table = within(container).getByRole('table', { name: 'Rollout prompt groups' });
    expect(table).toHaveStyle({ minWidth: '1070px' });
    // The prompt column has no fixed width, so it takes the table's spare room.
    expect(table.querySelectorAll('col')[1].className).toBe('');
    expect(within(table).getByRole('columnheader', { name: 'Component' })).toBeInTheDocument();
    expect(within(table).getByRole('columnheader', { name: 'Partial credit' })).toBeInTheDocument();
    expect(within(table).getByRole('columnheader', { name: 'Tool calls' })).toBeInTheDocument();
    expect(within(table).getByRole('columnheader', { name: 'Turns' })).toBeInTheDocument();
    const group = within(table).getByRole('row', { name: 'Prompt group task-a' });
    expect(within(group).getAllByRole('cell').map((cell) => cell.textContent)).toEqual(['1', '2 / 2', '0.500', '0.500', '—', '0.750', '4.0', '3.0', '6', '4', '11']);
    fireEvent.click(within(group).getByRole('button', { name: 'task-a' }));
    const child = within(table).getByRole('row', { name: 'Rollout a2' });
    expect(within(child).getAllByRole('cell').map((cell) => cell.textContent)).toEqual(['', '2 / 2', '0.750', '—', '—', '1.000', '5', '4', '—', '—', '12']);
  });

  it('pages prompt groups, sorts them by a column and asks for older rows past the last loaded page', () => {
    const many = Array.from({ length: 24 }, (_, index) => trace(`t${index}`, index + 1, `step/${index + 1}/group/1`, `task-${String(index).padStart(2, '0')}`, index / 24));
    const onLoadMore = vi.fn();
    const { container, rerender } = render(<RolloutGroupTable traces={many} allLoadedTraces={many} total={100} expectedSize={1} rewards={null} hasMore loadingMore={false} onLoadMore={onLoadMore} onSelect={vi.fn()} />);
    const view = within(container);
    const rows = () => view.getAllByRole('row', { name: /^Prompt group / });
    expect(rows()).toHaveLength(10);
    expect(rows()[0]).toHaveAccessibleName('Prompt group task-23');
    fireEvent.click(view.getByRole('button', { name: 'Page 3' }));
    expect(rows()).toHaveLength(4);
    fireEvent.click(view.getByRole('button', { name: 'Next page' }));
    expect(onLoadMore).toHaveBeenCalledTimes(1);
    // No rows arrived, so the table stays on the last loaded page.
    expect(rows()).toHaveLength(4);
    fireEvent.click(view.getByRole('button', { name: 'Step' }));
    expect(rows()[0]).toHaveAccessibleName('Prompt group task-23');
    fireEvent.click(view.getByRole('button', { name: 'Step' }));
    expect(rows()[0]).toHaveAccessibleName('Prompt group task-00');
    expect(view.getByRole('columnheader', { name: 'Step' })).toHaveAttribute('aria-sort', 'ascending');
  });
});
