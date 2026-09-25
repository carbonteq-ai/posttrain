import { useMemo, useState } from 'react';
import { CaretDown, CaretRight } from '@phosphor-icons/react';

import type { PromptGroupRewardView, TraceSummary } from '../lib/api';

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

function groupMeanCell(traces: TraceSummary[], value: (trace: TraceSummary) => number | null | undefined, digits: number) {
  const recorded = meanRecorded(traces, value);
  return <td className="px-2 py-2 text-right tabular-nums" title={recorded ? `Mean of ${recorded.count} of ${traces.length} loaded rollouts` : 'Not recorded on loaded rollouts'}>
    {recorded ? recorded.value.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits }) : '—'}
  </td>;
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
}: {
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
  const allGroups = useMemo(() => buildRolloutGroups(allLoadedTraces, rewards), [allLoadedTraces, rewards]);
  const visibleIds = new Set(traces.map((trace) => trace.external_id));
  const groups = allGroups.filter((group) => group.traces.some((trace) => visibleIds.has(trace.external_id)));
  return <section className="obs-card overflow-hidden bg-white" aria-label="Prompt groups by optimizer step">
    <header className="border-b border-divider px-3 py-2.5">
      <h2 className="text-[12px] font-medium text-ink">Prompt groups ({groups.length})</h2>
      <p className="mt-0.5 text-[10px] text-muted">Newest groups first. Reward mean and std dev come from indexed run-wide facts; prior values use the nearest older step for the same task. Group activity and tokens use loaded rows. — means unavailable, not zero.{rewards?.state === 'partial' ? ' Fact coverage is partial.' : ''}</p>
      {rewardError && <p role="alert" className="mt-1 text-[10px] text-rose-700">Run-wide group rewards unavailable: {rewardError}</p>}
      {!rewards && !rewardError && <p className="mt-1 text-[10px] text-muted">Loading indexed group rewards…</p>}
      {rewards?.state === 'unavailable' && <p role="status" className="mt-1 text-[10px] text-amber-700">Indexed group reward facts are not available for this run.</p>}
    </header>
    <div className="max-h-[430px] overflow-auto">
      <table className="w-full table-fixed text-left text-[11px]" style={{ minWidth: 870 + metricColumns.length * 78 }} aria-label="Rollout prompt groups">
        <colgroup><col className="w-[38px]" /><col className="w-[200px]" /><col className="w-[54px]" /><col className="w-[72px]" /><col className="w-[62px]" /><col className="w-[88px]" />{metricColumns.map((metric) => <col key={metric.name} className="w-[78px]" />)}<col className="w-[62px]" /><col className="w-[68px]" /><col className="w-[68px]" /><col className="w-[72px]" /><col className="w-[86px]" /></colgroup>
        <thead className="sticky top-0 z-10 bg-white text-[10px] text-muted">
          <tr className="border-b border-divider">
            <th scope="col" rowSpan={2} className="px-2 py-2">Step</th><th scope="col" rowSpan={2} className="px-2 py-2">Prompt group / rollout</th><th scope="col" rowSpan={2} className="px-2 py-2" title="Loaded rollouts / configured group size">Rollouts</th><th scope="col" rowSpan={2} className="px-2 py-2 text-right">Mean reward</th><th scope="col" rowSpan={2} className="px-2 py-2 text-right" title="Population standard deviation of complete prompt-group rewards from indexed facts">Reward std dev</th><th scope="col" rowSpan={2} className="px-2 py-2 text-right" title="Nearest older optimizer step with complete groups for the same task">Previous mean / std dev</th>
            {metricColumns.length > 0 && <th scope="colgroup" colSpan={metricColumns.length} className="border-l border-divider px-2 py-1.5 text-center">Reward components</th>}
            <th scope="col" rowSpan={2} className="border-l border-divider px-2 py-2 text-right">Tool calls</th><th scope="colgroup" colSpan={3} className="border-l border-divider px-2 py-1.5 text-center">Tokens</th><th scope="col" rowSpan={2} className="border-l border-divider px-2 py-2">Update selection</th>
          </tr>
          <tr className="border-b border-divider">
            {metricColumns.map((metric) => <th key={metric.name} scope="col" className="border-l border-divider px-2 py-1.5 text-right" title={metric.label}><span className="block truncate">{metric.label}</span></th>)}
            <th scope="col" className="border-l border-divider px-2 py-1.5 text-right">Thinking</th><th scope="col" className="px-2 py-1.5 text-right">Output</th><th scope="col" className="px-2 py-1.5 text-right" title="Total completion tokens, including thinking when recorded">Total</th>
          </tr>
        </thead>
        {groups.map((group) => <FragmentGroup key={group.id} group={group} expectedSize={expectedSize} metricColumns={metricColumns} selectedId={selectedId} expanded={expanded === group.id} onToggle={() => setExpanded(expanded === group.id ? null : group.id)} onSelect={onSelect} />)}
        {!groups.length && <tbody><tr><td colSpan={11 + metricColumns.length} className="px-3 py-8 text-center text-muted">No loaded prompt groups match these filters.</td></tr></tbody>}
      </table>
    </div>
    <footer className="flex items-center justify-between gap-3 border-t border-divider bg-subtle/35 px-3 py-2 text-[10px] text-muted">
      <span>Newest {allLoadedTraces.length.toLocaleString()} of {total.toLocaleString()} {filtered ? 'matching ' : ''}rollout summaries loaded</span>
      {hasMore && <button type="button" disabled={loadingMore} onClick={onLoadMore} className="rounded border border-divider bg-white px-2.5 py-1 font-medium text-violet-700 hover:border-violet-300 disabled:cursor-wait">{loadingMore ? 'Loading…' : 'Load 100 more'}</button>}
    </footer>
  </section>;
}

function FragmentGroup({ group, expectedSize, metricColumns, selectedId, expanded, onToggle, onSelect }: {
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
      <td className="px-2 py-2 tabular-nums">{group.step ?? '—'}</td>
      <th scope="row" className="px-2 py-2 font-normal"><button type="button" aria-expanded={expanded} onClick={onToggle} className="flex w-full items-center gap-1.5 text-left text-secondary hover:text-violet-700">{expanded ? <CaretDown size={12} /> : <CaretRight size={12} />}<span className="truncate" title={`${group.label} · ${group.id}`}>{group.label}</span></button></th>
      <td className="px-2 py-2 tabular-nums">{group.traces.length}{expectedSize == null ? '' : ` / ${expectedSize}`}</td>
      <td className="px-2 py-2 text-right tabular-nums">{formatReward(group.current?.mean)}</td>
      <td className="px-2 py-2 text-right tabular-nums">{formatReward(group.current?.std)}</td>
      <td className="px-2 py-2 text-right tabular-nums" title={group.priorStep == null ? 'No complete earlier task group in the indexed facts' : `Optimizer step ${group.priorStep} · ${group.priorRollouts} rewarded rollouts across complete task groups`}>{group.prior ? `${formatReward(group.prior.mean)} / ${formatReward(group.prior.std)}` : '—'}</td>
      {metricColumns.map((metric) => <td key={metric.name} className="border-l border-divider px-2 py-2 text-right tabular-nums" title={`Mean of recorded ${metric.label} values`}>{formatReward(meanRecorded(group.traces, (trace) => componentValue(trace, metric.name))?.value)}</td>)}
      {groupMeanCell(group.traces, (trace) => trace.tool_calls, 1)}
      {groupMeanCell(group.traces, (trace) => trace.thinking_tokens, 0)}
      {groupMeanCell(group.traces, (trace) => trace.response_tokens, 0)}
      {groupMeanCell(group.traces, (trace) => trace.completion_tokens, 0)}
      <td className="px-2 py-2 text-muted" title="This run does not record a per-group update-selection decision. Reward and standard deviation do not prove inclusion in an optimizer update.">Not recorded</td>
    </tr>
    {expanded && group.traces.map((trace, index) => <tr key={trace.external_id} className={`border-b border-divider text-secondary ${selectedId === trace.external_id ? 'bg-violet-100' : 'bg-violet-50/35'}`} aria-label={`Rollout ${trace.external_id}`} aria-selected={selectedId === trace.external_id}>
      <td className="px-2 py-1.5" />
      <th scope="row" className="px-2 py-1.5 font-normal"><button type="button" onClick={() => onSelect(trace)} className="flex w-full min-w-0 items-center gap-2 text-left hover:text-violet-700 focus-visible:outline-2 focus-visible:outline-violet-600"><span className="pl-4 text-violet-500" aria-hidden="true">↳</span><span className="shrink-0 font-mono text-[9px]">{trace.external_id.slice(0, 10)}</span><span className="truncate text-muted" title={trace.prompt_preview ?? undefined}>{trace.prompt_preview ?? group.label}</span></button></th>
      <td className="px-2 py-1.5 tabular-nums">{index + 1} / {group.traces.length}</td>
      <td className="px-2 py-1.5 text-right tabular-nums">{formatReward(trace.reward)}</td>
      <td className="px-2 py-1.5 text-right text-muted">—</td>
      <td className="px-2 py-1.5 text-right text-muted">—</td>
      {metricColumns.map((metric) => <td key={metric.name} className="border-l border-divider px-2 py-1.5 text-right tabular-nums">{formatReward(componentValue(trace, metric.name))}</td>)}
      <td className="px-2 py-1.5 text-right tabular-nums">{traceValue(trace.tool_calls)}</td>
      <td className="px-2 py-1.5 text-right tabular-nums">{traceValue(trace.thinking_tokens)}</td>
      <td className="px-2 py-1.5 text-right tabular-nums">{traceValue(trace.response_tokens)}</td>
      <td className="px-2 py-1.5 text-right tabular-nums">{traceValue(trace.completion_tokens)}</td>
      <td className="px-2 py-1.5 text-muted">Not recorded</td>
    </tr>)}
  </tbody>;
}
