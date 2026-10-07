import { useMemo, useState } from 'react';
import { CaretDown, CaretRight } from '@phosphor-icons/react';

import type { PromptGroupRewardView, TraceSummary } from '../lib/api';
import { Pager, SortButton, sortBy, usePaging, type SortState } from './TableControls';

const GROUPS_PER_PAGE = 10;

type RewardStats = { mean: number; std: number };
type MetricColumn = { name: string; label: string };
type RecordedMean = { value: number; count: number };

export type RolloutGroup = {
  id: string;
  step: number | null;
  task: string | null;
  label: string;
  traces: TraceSummary[];
  current: RewardStats | null;
  prior: RewardStats | null;
  priorStep: number | null;
  priorRollouts: number | null;
};

export function buildRolloutGroups(traces: TraceSummary[], rewards: PromptGroupRewardView | null): RolloutGroup[] {
  const rewardById = new Map(rewards?.groups.map((group) => [group.group_id, group]) ?? []);
  const byId = new Map<string, RolloutGroup>();
  for (const trace of traces) {
    // Never merge two occurrences solely because their task or preview matches.
    const id = trace.prompt_group_id ?? `trace:${trace.external_id}`;
    let group = byId.get(id);
    if (!group) {
      group = {
        id,
        step: trace.optimizer_step ?? null,
        task: trace.task,
        label: trace.task_label ?? trace.task ?? trace.prompt_preview ?? 'Unidentified prompt',
        traces: [],
        current: rewardById.get(id)?.current ?? null,
        prior: rewardById.get(id)?.prior ?? null,
        priorStep: rewardById.get(id)?.prior_step ?? null,
        priorRollouts: rewardById.get(id)?.prior_rollouts ?? null,
      };
      byId.set(id, group);
    }
    group.traces.push(trace);
  }
  return [...byId.values()].sort((left, right) =>
    (left.step ?? Number.MAX_SAFE_INTEGER) - (right.step ?? Number.MAX_SAFE_INTEGER)
    || left.id.localeCompare(right.id, undefined, { numeric: true })).reverse();
}

function formatReward(value: number | null | undefined): string {
  return value == null ? '—' : value.toFixed(3);
}

function meanRecorded(traces: TraceSummary[], value: (trace: TraceSummary) => number | null | undefined): RecordedMean | null {
  const recorded = traces.map(value).filter((item): item is number => item != null && Number.isFinite(item));
  return recorded.length ? { value: recorded.reduce((sum, item) => sum + item, 0) / recorded.length, count: recorded.length } : null;
}

function componentValue(trace: TraceSummary, name: string): number | undefined {
  return trace.reward_components[name] ?? trace.native_metrics[name] ?? trace.metrics[name];
}

function groupMeanCell(traces: TraceSummary[], value: (trace: TraceSummary) => number | null | undefined, digits: number, divided = false) {
  const recorded = meanRecorded(traces, value);
  return <td className={`${divided ? 'border-l border-divider ' : ''}px-2 py-2 text-right tabular-nums`} title={recorded ? `Mean of ${recorded.count} of ${traces.length} loaded rollouts` : 'Not recorded on loaded rollouts'}>
    {recorded ? recorded.value.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits }) : '—'}
  </td>;
}

const TRACE_MEANS: Record<string, (trace: TraceSummary) => number | null | undefined> = {
  turns: (trace) => trace.model_calls,
  tools: (trace) => trace.tool_calls,
  thinking: (trace) => trace.thinking_tokens,
  output: (trace) => trace.response_tokens,
  total: (trace) => trace.completion_tokens,
};

function groupSortValue(group: RolloutGroup, key: string): unknown {
  if (key === 'step') return group.step;
  if (key === 'label') return group.label;
  if (key === 'rollouts') return group.traces.length;
  if (key === 'mean') return group.current?.mean;
  if (key === 'std') return group.current?.std;
  if (key === 'prior') return group.prior?.mean;
  if (key.startsWith('metric:')) return meanRecorded(group.traces, (trace) => componentValue(trace, key.slice('metric:'.length)))?.value;
  const value = TRACE_MEANS[key];
  return value ? meanRecorded(group.traces, value)?.value : null;
}

function traceValue(value: number | null | undefined, digits = 0): string {
  return value == null ? '—' : value.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

export function RolloutGroupTable({
  traces,
  allLoadedTraces,
  total,
  filtered = false,
  expectedSize,
  rewards,
  rewardError = '',
  metricColumns = [],
  selectedId = null,
  hasMore,
  loadingMore,
  onLoadMore,
  onSelect,
  stepLabel = (step) => step,
}: {
  /** How a recorded step is shown: its collection number when collections span several updates. */
  stepLabel?: (step: number | null) => number | null;
  traces: TraceSummary[];
  allLoadedTraces: TraceSummary[];
  total: number;
  filtered?: boolean;
  expectedSize: number | null;
  rewards: PromptGroupRewardView | null;
  rewardError?: string;
  metricColumns?: MetricColumn[];
  selectedId?: string | null;
  hasMore: boolean;
  loadingMore: boolean;
  onLoadMore: () => void;
  onSelect: (trace: TraceSummary) => void;
}) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const [sort, setSort] = useState<SortState>(null);
  const allGroups = useMemo(() => buildRolloutGroups(allLoadedTraces, rewards), [allLoadedTraces, rewards]);
  const visibleIds = new Set(traces.map((trace) => trace.external_id));
  const groups = allGroups.filter((group) => group.traces.some((trace) => visibleIds.has(trace.external_id)));
  // A new filter changes which group leads; loading older rows only appends.
  const sorted = sortBy(groups, sort, groupSortValue);
  const paging = usePaging(sorted, GROUPS_PER_PAGE, `${filtered}:${groups[0]?.id ?? ''}:${sort?.key}:${sort?.direction}`, loadingMore);
  const sortable = (key: string, label: string, numeric = true, title?: string) => <SortButton sortKey={key} sort={sort} numeric={numeric} onSort={setSort} label={label} title={title ?? `Sort loaded groups by ${label.toLowerCase()}`} />;
  const ariaSort = (key: string) => sort?.key === key ? (sort.direction === 'asc' ? 'ascending' as const : 'descending' as const) : undefined;
  return <section className="obs-card overflow-hidden bg-white" aria-label="Prompt groups by optimizer step">
    <header className="border-b border-divider px-3 py-2.5">
      <h2 className="text-[12px] font-medium text-ink">Prompt groups ({groups.length})</h2>
      <p className="mt-0.5 text-[10px] text-muted">Newest groups first; column headers sort the loaded groups. Reward mean and std dev come from indexed run-wide facts; prior values use the nearest older step for the same task. Group activity and tokens use loaded rows. — means unavailable, not zero. Update selection is not recorded, so reward spread alone does not show that a group entered an optimizer update.{rewards?.state === 'partial' ? ' Fact coverage is partial.' : ''}</p>
      {rewardError && <p role="alert" className="mt-1 text-[10px] text-rose-700">Run-wide group rewards unavailable: {rewardError}</p>}
      {!rewards && !rewardError && <p className="mt-1 text-[10px] text-muted">Loading indexed group rewards…</p>}
      {rewards?.state === 'unavailable' && <p role="status" className="mt-1 text-[10px] text-amber-700">Indexed group reward facts are not available for this run.</p>}
    </header>
    <div className="overflow-x-auto">
      {/* The prompt column has no width of its own, so it takes the room the fixed columns leave. */}
      <table className="w-full table-fixed text-left text-[11px]" style={{ minWidth: 974 + metricColumns.length * 96 }} aria-label="Rollout prompt groups">
        <colgroup><col className="w-[56px]" /><col /><col className="w-[72px]" /><col className="w-[70px]" /><col className="w-[70px]" /><col className="w-[112px]" />{metricColumns.map((metric) => <col key={metric.name} className="w-[96px]" />)}<col className="w-[64px]" /><col className="w-[76px]" /><col className="w-[80px]" /><col className="w-[70px]" /><col className="w-[64px]" /></colgroup>
        <thead className="bg-white text-[10px] text-muted [&_th]:whitespace-nowrap">
          <tr className="border-b border-divider">
            <th scope="col" rowSpan={2} aria-sort={ariaSort('step')} className="px-2 py-2">{sortable('step', 'Step')}</th><th scope="col" rowSpan={2} aria-sort={ariaSort('label')} className="px-2 py-2">{sortable('label', 'Prompt group / rollout', false, 'Sort loaded groups by task name')}</th><th scope="col" rowSpan={2} aria-sort={ariaSort('rollouts')} className="px-2 py-2">{sortable('rollouts', 'Rollouts', true, 'Loaded rollouts / configured group size')}</th><th scope="colgroup" colSpan={3} className="border-l border-divider px-2 py-1.5 text-center" title="Complete prompt-group rewards from indexed run-wide facts">Reward</th>
            {metricColumns.length > 0 && <th scope="colgroup" colSpan={metricColumns.length} className="border-l border-divider px-2 py-1.5 text-center" title="Reward components">{metricColumns.length === 1 ? 'Component' : 'Reward components'}</th>}
            <th scope="colgroup" colSpan={2} className="border-l border-divider px-2 py-1.5 text-center">Activity</th><th scope="colgroup" colSpan={3} className="border-l border-divider px-2 py-1.5 text-center">Tokens</th>
          </tr>
          <tr className="border-b border-divider">
            <th scope="col" aria-sort={ariaSort('mean')} className="border-l border-divider px-2 py-1.5 text-right">{sortable('mean', 'Mean', true, 'Sort loaded groups by mean reward')}</th><th scope="col" aria-sort={ariaSort('std')} className="px-2 py-1.5 text-right">{sortable('std', 'Std dev', true, 'Population standard deviation of complete prompt-group rewards from indexed facts')}</th><th scope="col" aria-sort={ariaSort('prior')} className="px-2 py-1.5 text-right">{sortable('prior', 'Previous', true, 'Mean / std dev at the nearest older optimizer step with complete groups for the same task')}</th>
            {metricColumns.map((metric) => <th key={metric.name} scope="col" aria-sort={ariaSort(`metric:${metric.name}`)} className="border-l border-divider px-2 py-1.5 text-right">{sortable(`metric:${metric.name}`, metric.label, true, metric.label)}</th>)}
            <th scope="col" aria-sort={ariaSort('turns')} className="border-l border-divider px-2 py-1.5 text-right">{sortable('turns', 'Turns', true, 'Assistant turns (model calls) in the rollout')}</th><th scope="col" aria-sort={ariaSort('tools')} className="px-2 py-1.5 text-right">{sortable('tools', 'Tool calls')}</th>
            <th scope="col" aria-sort={ariaSort('thinking')} className="border-l border-divider px-2 py-1.5 text-right">{sortable('thinking', 'Thinking')}</th><th scope="col" aria-sort={ariaSort('output')} className="px-2 py-1.5 text-right">{sortable('output', 'Output')}</th><th scope="col" aria-sort={ariaSort('total')} className="px-2 py-1.5 text-right">{sortable('total', 'Total', true, 'Total completion tokens, including thinking when recorded')}</th>
          </tr>
        </thead>
        {paging.items.map((group) => <FragmentGroup key={group.id} stepLabel={stepLabel} group={group} expectedSize={expectedSize} metricColumns={metricColumns} selectedId={selectedId} expanded={expanded === group.id} onToggle={() => setExpanded(expanded === group.id ? null : group.id)} onSelect={onSelect} />)}
        {!paging.items.length && <tbody><tr><td colSpan={11 + metricColumns.length} className="px-3 py-8 text-center text-muted">{loadingMore ? 'Loading older rollouts…' : 'No loaded prompt groups match these filters.'}</td></tr></tbody>}
      </table>
    </div>
    <Pager paging={paging} count={groups.length} noun={`groups · ${allLoadedTraces.length.toLocaleString()} of ${total.toLocaleString()} ${filtered ? 'matching ' : ''}rollouts loaded`} hasMore={hasMore} loading={loadingMore} onLoadMore={onLoadMore} />
  </section>;
}

function FragmentGroup({ group, expectedSize, metricColumns, selectedId, expanded, onToggle, onSelect, stepLabel }: {
  stepLabel: (step: number | null) => number | null;
  group: RolloutGroup;
  expectedSize: number | null;
  metricColumns: MetricColumn[];
  selectedId: string | null;
  expanded: boolean;
  onToggle: () => void;
  onSelect: (trace: TraceSummary) => void;
}) {
  return <tbody>
    <tr className="border-b border-divider bg-white hover:bg-subtle/50" aria-label={`Prompt group ${group.label}`}>
      <td className="px-2 py-2 tabular-nums">{stepLabel(group.step) ?? '—'}</td>
      <th scope="row" className="px-2 py-2 font-normal"><button type="button" aria-expanded={expanded} onClick={onToggle} className="flex w-full items-center gap-1.5 text-left text-secondary hover:text-violet-700">{expanded ? <CaretDown size={12} /> : <CaretRight size={12} />}<span className="truncate" title={`${group.label} · ${group.id}`}>{group.label}</span></button></th>
      <td className="px-2 py-2 tabular-nums">{group.traces.length}{expectedSize == null ? '' : ` / ${expectedSize}`}</td>
      <td className="border-l border-divider px-2 py-2 text-right tabular-nums">{formatReward(group.current?.mean)}</td>
      <td className="px-2 py-2 text-right tabular-nums">{formatReward(group.current?.std)}</td>
      <td className="px-2 py-2 text-right tabular-nums" title={group.priorStep == null ? 'No complete earlier task group in the indexed facts' : `Step ${stepLabel(group.priorStep)} · ${group.priorRollouts} rewarded rollouts across complete task groups`}>{group.prior ? `${formatReward(group.prior.mean)} / ${formatReward(group.prior.std)}` : '—'}</td>
      {metricColumns.map((metric) => <td key={metric.name} className="border-l border-divider px-2 py-2 text-right tabular-nums" title={`Mean of recorded ${metric.label} values`}>{formatReward(meanRecorded(group.traces, (trace) => componentValue(trace, metric.name))?.value)}</td>)}
      {groupMeanCell(group.traces, (trace) => trace.model_calls, 1, true)}
      {groupMeanCell(group.traces, (trace) => trace.tool_calls, 1)}
      {groupMeanCell(group.traces, (trace) => trace.thinking_tokens, 0, true)}
      {groupMeanCell(group.traces, (trace) => trace.response_tokens, 0)}
      {groupMeanCell(group.traces, (trace) => trace.completion_tokens, 0)}
    </tr>
    {expanded && group.traces.map((trace, index) => <tr key={trace.external_id} className={`border-b border-divider text-secondary ${selectedId === trace.external_id ? 'bg-violet-100' : 'bg-violet-50/35'}`} aria-label={`Rollout ${trace.external_id}`} aria-selected={selectedId === trace.external_id}>
      <td className="px-2 py-1.5" />
      <th scope="row" className="px-2 py-1.5 font-normal"><button type="button" onClick={() => onSelect(trace)} className="flex w-full min-w-0 items-center gap-2 text-left hover:text-violet-700 focus-visible:outline-2 focus-visible:outline-violet-600"><span className="pl-4 text-violet-500" aria-hidden="true">↳</span><span className="shrink-0 font-mono text-[9px]">{trace.external_id.slice(0, 10)}</span><span className="truncate text-muted" title={trace.prompt_preview ?? undefined}>{trace.prompt_preview ?? group.label}</span></button></th>
      <td className="px-2 py-1.5 tabular-nums">{index + 1} / {group.traces.length}</td>
      <td className="border-l border-divider px-2 py-1.5 text-right tabular-nums">{formatReward(trace.reward)}</td>
      <td className="px-2 py-1.5 text-right text-muted">—</td>
      <td className="px-2 py-1.5 text-right text-muted">—</td>
      {metricColumns.map((metric) => <td key={metric.name} className="border-l border-divider px-2 py-1.5 text-right tabular-nums">{formatReward(componentValue(trace, metric.name))}</td>)}
      <td className="border-l border-divider px-2 py-1.5 text-right tabular-nums">{traceValue(trace.model_calls)}</td>
      <td className="px-2 py-1.5 text-right tabular-nums">{traceValue(trace.tool_calls)}</td>
      <td className="border-l border-divider px-2 py-1.5 text-right tabular-nums">{traceValue(trace.thinking_tokens)}</td>
      <td className="px-2 py-1.5 text-right tabular-nums">{traceValue(trace.response_tokens)}</td>
      <td className="px-2 py-1.5 text-right tabular-nums">{traceValue(trace.completion_tokens)}</td>
    </tr>)}
  </tbody>;
}
