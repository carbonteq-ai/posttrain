import { useEffect, useMemo, useState } from 'react';
import { ArrowSquareOut } from '@phosphor-icons/react';

import { api, type EvaluationIndex, type EvaluationRecord, type EvaluationTaskScore, type MetricSeries } from '../lib/api';
import {
  BEHAVIOUR_COLUMNS,
  NOT_RECORDED,
  behaviourMean,
  defaultCheckpoint,
  deltaClass,
  endingSummary,
  evaluatedTrainingRuns,
  formatDelta,
  formatMean,
  formatScore,
  meanScore,
  runSuites,
  stepLabel,
  suiteLabel,
  taskRows,
  usable,
  type SuiteEvaluations,
  type TaskRow,
} from '../lib/evaluations';
import { EvidenceChart } from './EvidenceChart';
import { FilterInput, Pager, SortButton, sortBy, usePaging, type SortState } from './TableControls';

const TASK_PAGE_ROWS = 15;
const SCORE_UNIT = 'score';

type OpenRun = (runKey: string) => void;

function RunLink({ record, label, onOpenRun }: { record: EvaluationRecord; label?: string; onOpenRun: OpenRun }) {
  return <button type="button" onClick={() => onOpenRun(record.run_key)} title={record.run_id} className="inline-flex max-w-full items-center gap-1 text-left text-violet-700 hover:text-violet-900 hover:underline">
    <span className="truncate">{label ?? record.run_id}</span><ArrowSquareOut size={11} className="shrink-0" aria-hidden="true" />
  </button>;
}

function Stat({ label, value, delta, note }: { label: string; value: string; delta?: number | null; note?: string }) {
  return <div className="min-w-0 rounded-[5px] border border-divider bg-surface px-3 py-2.5">
    <span className="type-label block truncate">{label}</span>
    <strong className="mt-1 block font-serif text-2xl font-normal text-ink">{value}</strong>
    {delta !== undefined && <span className={`text-[11px] tabular-nums ${deltaClass(delta)}`}>{formatDelta(delta)} vs base</span>}
    {note && <small className="block truncate text-[10px] text-muted" title={note}>{note}</small>}
  </div>;
}

/** Score by checkpoint step, with the base model at step 0 and as a flat reference line. */
function scoreSeries(runs: Array<{ name: string; records: EvaluationRecord[] }>, baseScore: number | null): MetricSeries[] {
  const lastStep = Math.max(0, ...runs.flatMap(({ records }) => records.map((record) => record.parent_step ?? 0)));
  const series: MetricSeries[] = runs.map(({ name, records }) => ({
    name,
    points: [
      ...(baseScore == null ? [] : [{ step: 0, value: baseScore }]),
      ...records.filter(usable).map((record) => ({ step: record.parent_step ?? 0, value: record.score as number })),
    ],
  }));
  if (baseScore != null) series.push({ name: 'Base model', points: [{ step: 0, value: baseScore }, { step: lastStep, value: baseScore }] });
  return series;
}

function useTaskScores(runKeys: string[]): { scores: EvaluationTaskScore[]; loading: boolean; error: string } {
  const key = runKeys.join('|');
  const [state, setState] = useState<{ key: string; scores: EvaluationTaskScore[]; error: string } | null>(null);
  useEffect(() => {
    if (!key) return;
    let active = true;
    api.evaluationTasks(key.split('|'))
      .then((value) => { if (active) setState({ key, scores: value.scores, error: '' }); })
      .catch((cause: unknown) => { if (active) setState({ key, scores: [], error: cause instanceof Error ? cause.message : String(cause) }); });
    return () => { active = false; };
  }, [key]);
  if (!key) return { scores: [], loading: false, error: '' };
  return state?.key === key ? { scores: state.scores, loading: false, error: state.error } : { scores: [], loading: true, error: '' };
}

type TaskColumn = { key: string; label: string; title?: string };
type TaskDelta = { key: string; label: string; minus: string; from: string };

/** Tasks against score columns and differences between them; sortable, filterable and paged. */
export function TaskComparisonTable({ rows, columns, deltas, loading, error, title }: {
  rows: TaskRow[];
  columns: TaskColumn[];
  deltas: TaskDelta[];
  loading: boolean;
  error?: string;
  title: string;
}) {
  const [sort, setSort] = useState<SortState>(null);
  const [filter, setFilter] = useState('');
  const needle = filter.trim().toLowerCase();
  const value = (row: TaskRow, key: string): number | string | null => {
    if (key === 'task') return row.task;
    const delta = deltas.find((item) => item.key === key);
    if (!delta) return row.values[key] ?? null;
    const left = row.values[delta.minus];
    const right = row.values[delta.from];
    return left == null || right == null ? null : left - right;
  };
  const matching = needle ? rows.filter((row) => row.task.toLowerCase().includes(needle)) : rows;
  const sorted = sortBy(matching, sort, value);
  const paging = usePaging(sorted, TASK_PAGE_ROWS, `${needle}:${sort?.key}:${sort?.direction}`);
  const ariaSort = (key: string) => sort?.key === key ? (sort.direction === 'asc' ? 'ascending' as const : 'descending' as const) : undefined;
  return <section className="mt-4" aria-label={title}>
    <div className="mb-1.5 flex flex-wrap items-center justify-between gap-2">
      <p className="text-[11px] font-medium text-ink">{title}</p>
      <FilterInput value={filter} onChange={setFilter} label={`Filter ${title} tasks`} placeholder="Filter tasks" />
    </div>
    <div className="overflow-hidden rounded-[4px] border border-divider">
      <div className="overflow-x-auto">
        <table className="min-w-full text-left text-[11px]">
          <thead className="bg-subtle text-[10px] text-muted">
            <tr>
              <th scope="col" aria-sort={ariaSort('task')} className="sticky left-0 z-[1] whitespace-nowrap bg-subtle px-2.5 py-1.5 font-medium shadow-[1px_0_0_var(--obs-divider)]"><SortButton sortKey="task" sort={sort} onSort={setSort} label="Task" /></th>
              {columns.map((column) => <th key={column.key} scope="col" aria-sort={ariaSort(column.key)} className="whitespace-nowrap px-2.5 py-1.5 text-right font-medium"><SortButton sortKey={column.key} sort={sort} numeric onSort={setSort} label={column.label} title={column.title} /></th>)}
              {deltas.map((delta, index) => <th key={delta.key} scope="col" aria-sort={ariaSort(delta.key)} className={`whitespace-nowrap px-2.5 py-1.5 text-right font-medium ${index === 0 ? 'border-l border-divider' : ''}`}><SortButton sortKey={delta.key} sort={sort} numeric onSort={setSort} label={delta.label} /></th>)}
            </tr>
          </thead>
          <tbody className="divide-y divide-divider">
            {paging.items.map((row) => <tr key={row.task}>
              <th scope="row" className="sticky left-0 z-[1] whitespace-nowrap bg-surface px-2.5 py-1.5 font-normal text-secondary shadow-[1px_0_0_var(--obs-divider)]">{row.task}</th>
              {columns.map((column) => <td key={column.key} className="whitespace-nowrap px-2.5 py-1.5 text-right font-mono tabular-nums text-secondary">{formatScore(row.values[column.key])}</td>)}
              {deltas.map((delta, index) => {
                const difference = value(row, delta.key) as number | null;
                return <td key={delta.key} className={`whitespace-nowrap px-2.5 py-1.5 text-right font-mono tabular-nums ${deltaClass(difference)} ${index === 0 ? 'border-l border-divider' : ''}`}>{formatDelta(difference)}</td>;
              })}
            </tr>)}
            {!paging.items.length && <tr><td colSpan={1 + columns.length + deltas.length} className="px-2.5 py-5 text-center text-muted">{loading ? 'Loading task scores…' : error ? `Task scores unavailable: ${error}` : needle ? `No tasks match “${filter.trim()}”.` : 'No task scores were recorded.'}</td></tr>}
          </tbody>
        </table>
      </div>
      {(sorted.length > TASK_PAGE_ROWS || needle) && <Pager paging={paging} count={sorted.length} noun={needle ? `matching tasks · ${rows.length.toLocaleString()} in all` : 'tasks'} />}
    </div>
  </section>;
}

/** How episodes went for a few evaluation columns: one row per behaviour value. */
function BehaviourTable({ columns }: { columns: Array<{ label: string; records: EvaluationRecord[] }> }) {
  return <div className="mt-3 overflow-hidden rounded-[4px] border border-divider">
    <table className="min-w-full text-left text-[11px]" aria-label="How episodes went">
      <thead className="bg-subtle text-[10px] text-muted"><tr>
        <th scope="col" className="whitespace-nowrap px-2.5 py-1.5 font-medium">Per episode</th>
        {columns.map((column) => <th key={column.label} scope="col" className="whitespace-nowrap px-2.5 py-1.5 text-right font-medium">{column.label}</th>)}
      </tr></thead>
      <tbody className="divide-y divide-divider">
        {BEHAVIOUR_COLUMNS.map((row) => <tr key={row.key}>
          <th scope="row" title={row.title} className="whitespace-nowrap px-2.5 py-1.5 font-medium text-ink">{row.label}</th>
          {columns.map((column) => {
            const value = behaviourMean(column.records, row.key);
            return <td key={column.label} title={value == null ? NOT_RECORDED : undefined} className="whitespace-nowrap px-2.5 py-1.5 text-right font-mono tabular-nums">{formatMean(value, row.digits)}</td>;
          })}
        </tr>)}
        <tr>
          <th scope="row" title="How the episodes ended" className="whitespace-nowrap px-2.5 py-1.5 font-medium text-ink">Endings</th>
          {columns.map((column) => <td key={column.label} className="px-2.5 py-1.5 text-right text-[10px]">{column.records.length === 1 ? endingSummary(column.records[0]?.endings) || '—' : column.records.length ? `${column.records.length} runs` : '—'}</td>)}
        </tr>
      </tbody>
    </table>
  </div>;
}

function SuiteSection({ group, onOpenRun }: { group: SuiteEvaluations; onOpenRun: OpenRun }) {
  const usableCheckpoints = group.checkpoints.filter(usable);
  const usableBase = group.base.filter(usable);
  const latest = defaultCheckpoint(group.checkpoints);
  const best = usableCheckpoints.reduce<EvaluationRecord | null>((top, record) => top == null || (record.score as number) > (top.score as number) ? record : top, null);
  const series = useMemo(() => scoreSeries([{ name: 'Checkpoints', records: group.checkpoints }], group.baseScore), [group]);
  const taskKeys = useMemo(() => [...usableBase, ...usableCheckpoints].map((record) => record.run_key), [usableBase, usableCheckpoints]);
  const tasks = useTaskScores(taskKeys);
  const columns = useMemo(() => [
    ...(usableBase.length ? [{ key: 'base', runKeys: usableBase.map((record) => record.run_key) }] : []),
    ...usableCheckpoints.map((record) => ({ key: record.run_key, runKeys: [record.run_key] })),
  ], [usableBase, usableCheckpoints]);
  const rows = useMemo(() => taskRows(tasks.scores, columns), [tasks.scores, columns]);
  const lastUsable = usableCheckpoints[usableCheckpoints.length - 1];
  const tableRows = [...group.base.map((record) => ({ record, base: true })), ...group.checkpoints.map((record) => ({ record, base: false }))];
  return <section className="obs-card mt-4 p-4" aria-label={`Evaluations on ${suiteLabel(group.suite)}`}>
    <header className="flex flex-wrap items-start justify-between gap-2">
      <div className="min-w-0">
        <p className="type-eyebrow">EVALUATION SUITE</p>
        <h2 className="mt-1 font-serif text-xl font-normal">{suiteLabel(group.suite)}</h2>
        <p className="mt-0.5 break-all text-[10px] text-muted">{group.suite ?? 'No work package recorded'} · base model {group.model ?? 'not recorded'}</p>
      </div>
    </header>
    <div className="mt-3 grid gap-2 sm:grid-cols-3">
      <Stat label="Base model" value={formatScore(group.baseScore)} note={usableBase.length ? `mean of ${usableBase.length} ${usableBase.length === 1 ? 'run' : 'runs'}${usableBase.length > 1 ? `: ${usableBase.map((record) => formatScore(record.score)).join(', ')}` : ''}` : 'no base-model evaluation on this suite'} />
      <Stat label={`Latest · ${stepLabel(latest?.parent_step)}`} value={formatScore(latest?.score)} delta={group.baseScore != null && latest?.score != null ? latest.score - group.baseScore : undefined} />
      <Stat label={`Best · ${stepLabel(best?.parent_step)}`} value={formatScore(best?.score)} delta={group.baseScore != null && best?.score != null ? best.score - group.baseScore : undefined} />
    </div>
    {usableCheckpoints.length > 0 && <div className="mt-3">
      <EvidenceChart series={series} height={220} compact ariaLabel={`Score by checkpoint step on ${suiteLabel(group.suite)}`} metricLabels={{ Checkpoints: 'Checkpoint score', 'Base model': 'Base model' }} metricUnits={{ Checkpoints: SCORE_UNIT, 'Base model': SCORE_UNIT }} xAxis={{ name: 'Checkpoint step (0 = base model)' }} />
    </div>}
    <div className="mt-3 overflow-hidden rounded-[4px] border border-divider">
      <div className="overflow-x-auto">
        <table className="min-w-full text-left text-[11px]" aria-label={`Base and checkpoints on ${suiteLabel(group.suite)}`}>
          <thead className="bg-subtle text-[10px] text-muted"><tr>
            {[
              { label: 'Model' }, { label: 'Evaluation run' }, { label: 'Score', numeric: true }, { label: 'Δ vs base', numeric: true },
              ...BEHAVIOUR_COLUMNS.map((column) => ({ label: column.label, title: column.title, numeric: true })),
              { label: 'Attempts', numeric: true }, { label: 'Truncated', numeric: true }, { label: 'Failed', numeric: true }, { label: 'Status' },
            ].map(({ label, title, numeric }: { label: string; title?: string; numeric?: boolean }) => <th key={label} scope="col" title={title} className={`whitespace-nowrap px-2.5 py-1.5 font-medium ${numeric ? 'text-right' : ''}`}>{label}</th>)}
          </tr></thead>
          <tbody className="divide-y divide-divider">
            {tableRows.map(({ record, base }) => {
              const delta = !base && usable(record) && group.baseScore != null ? (record.score as number) - group.baseScore : null;
              return <tr key={record.run_key} className={usable(record) ? '' : 'text-muted'}>
                <td className="whitespace-nowrap px-2.5 py-1.5 font-medium text-ink">{base ? 'Base model' : stepLabel(record.parent_step)}</td>
                <td className="max-w-[28rem] px-2.5 py-1.5"><RunLink record={record} onOpenRun={onOpenRun} /></td>
                <td className="whitespace-nowrap px-2.5 py-1.5 text-right font-mono tabular-nums">{formatScore(record.score)}</td>
                <td className={`whitespace-nowrap px-2.5 py-1.5 text-right font-mono tabular-nums ${deltaClass(delta)}`}>{base ? '' : formatDelta(delta)}</td>
                {BEHAVIOUR_COLUMNS.map((column) => {
                  const value = record[column.key];
                  return <td key={column.key} title={value == null ? NOT_RECORDED : undefined} className="whitespace-nowrap px-2.5 py-1.5 text-right font-mono tabular-nums">{formatMean(value, column.digits)}</td>;
                })}
                <td className="whitespace-nowrap px-2.5 py-1.5 text-right tabular-nums">{record.attempts}</td>
                <td title={endingSummary(record.endings) || undefined} className="whitespace-nowrap px-2.5 py-1.5 text-right tabular-nums">{record.truncated}</td>
                <td className="whitespace-nowrap px-2.5 py-1.5 text-right tabular-nums">{record.failed}</td>
                <td className="whitespace-nowrap px-2.5 py-1.5">{record.status ?? '—'}{!usable(record) && <span className="ml-1 text-[10px]">(not compared)</span>}</td>
              </tr>;
            })}
          </tbody>
        </table>
      </div>
    </div>
    <TaskComparisonTable
      title="Score by task"
      rows={rows}
      loading={tasks.loading}
      error={tasks.error}
      columns={columns.map((column) => column.key === 'base'
        ? { key: 'base', label: 'Base', title: `Mean of ${usableBase.length} base-model ${usableBase.length === 1 ? 'run' : 'runs'}` }
        : { key: column.key, label: stepLabel(usableCheckpoints.find((record) => record.run_key === column.key)?.parent_step) })}
      deltas={usableBase.length && lastUsable ? [{ key: 'delta', label: `Δ ${stepLabel(lastUsable.parent_step)} vs base`, minus: lastUsable.run_key, from: 'base' }] : []}
    />
  </section>;
}

/** A training run's checkpoint evaluations, suite by suite, against the base model. */
export function RunEvaluations({ runId, index, loading, error, onOpenRun }: {
  runId: string;
  index: EvaluationIndex | null;
  loading: boolean;
  error?: string;
  onOpenRun: OpenRun;
}) {
  const suites = useMemo(() => runSuites(index, runId), [index, runId]);
  return <div>
    <p className="type-eyebrow">CHECKPOINT EVALUATIONS</p>
    <h1 className="type-page-title mt-1.5">Evals</h1>
    <p className="type-page-subtitle mt-2">Each evaluation suite this run's checkpoints were scored on, against the base model on the same suite. {index?.score_definition}</p>
    {error && <p role="alert" className="mt-4 text-xs text-rose-700">Evaluations unavailable: {error}</p>}
    {!error && loading && !index && <p className="mt-4 text-xs text-muted">Loading evaluations…</p>}
    {index && !suites.length && <p className="obs-card mt-4 p-5 text-xs text-muted">No evaluation run records a checkpoint of this run.</p>}
    {suites.map((group) => <SuiteSection key={`${group.suite}:${group.model}`} group={group} onOpenRun={onOpenRun} />)}
  </div>;
}

function Select({ label, value, options, onChange }: { label: string; value: string; options: Array<{ value: string; label: string }>; onChange: (value: string) => void }) {
  return <label className="flex min-w-0 flex-col gap-1">
    <span className="type-label">{label}</span>
    <select value={value} onChange={(event) => onChange(event.target.value)} className="obs-control h-8 min-w-0 px-2 text-[11px] text-ink">
      {options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
    </select>
  </label>;
}

/**
 * Two training runs from the same base model on an evaluation suite they both
 * have checkpoints on: base, run A and run B, overall and per task.
 */
export function EvalComparePage({ index, loading, error, displayName, onOpenRun, initialRunId }: {
  index: EvaluationIndex | null;
  loading: boolean;
  error?: string;
  displayName: (runId: string) => string;
  onOpenRun: OpenRun;
  initialRunId?: string | null;
}) {
  const trainingRuns = useMemo(() => evaluatedTrainingRuns(index), [index]);
  const [runA, setRunA] = useState<string>('');
  const [runB, setRunB] = useState<string>('');
  const [suite, setSuite] = useState<string>('');
  const [stepA, setStepA] = useState<string>('');
  const [stepB, setStepB] = useState<string>('');
  const a = trainingRuns.find((run) => run.runId === runA) ?? trainingRuns.find((run) => run.runId === initialRunId) ?? trainingRuns[0];
  const partners = trainingRuns.filter((run) => a && run.runId !== a.runId && run.model === a.model && run.suites.some((item) => a.suites.includes(item)));
  const b = partners.find((run) => run.runId === runB) ?? partners[0];
  const shared = a && b ? a.suites.filter((item) => b.suites.includes(item)) : [];
  const activeSuite = shared.includes(suite) ? suite : shared[0] ?? '';
  const groupA = useMemo(() => a ? runSuites(index, a.runId).find((group) => (group.suite ?? '') === activeSuite && group.model === a.model) : undefined, [a, activeSuite, index]);
  const groupB = useMemo(() => b ? runSuites(index, b.runId).find((group) => (group.suite ?? '') === activeSuite && group.model === b.model) : undefined, [b, activeSuite, index]);
  const checkpointA = groupA?.checkpoints.find((record) => record.run_key === stepA) ?? defaultCheckpoint(groupA?.checkpoints ?? []);
  const checkpointB = groupB?.checkpoints.find((record) => record.run_key === stepB) ?? defaultCheckpoint(groupB?.checkpoints ?? []);
  const base = (groupA?.base ?? []).filter(usable);
  const baseScore = meanScore(base);
  const series = useMemo(() => groupA && groupB ? scoreSeries([
    { name: 'Run A', records: groupA.checkpoints },
    { name: 'Run B', records: groupB.checkpoints },
  ], baseScore) : [], [groupA, groupB, baseScore]);
  const columns = useMemo(() => [
    ...(base.length ? [{ key: 'base', runKeys: base.map((record) => record.run_key) }] : []),
    ...(checkpointA && usable(checkpointA) ? [{ key: 'a', runKeys: [checkpointA.run_key] }] : []),
    ...(checkpointB && usable(checkpointB) ? [{ key: 'b', runKeys: [checkpointB.run_key] }] : []),
  ], [base, checkpointA, checkpointB]);
  const tasks = useTaskScores(columns.flatMap((column) => column.runKeys));
  const rows = useMemo(() => taskRows(tasks.scores, columns), [tasks.scores, columns]);
  const has = (key: string) => columns.some((column) => column.key === key);
  const runOptions = (runs: typeof trainingRuns) => runs.map((run) => ({ value: run.runId, label: displayName(run.runId) }));
  const stepOptions = (group: SuiteEvaluations | undefined) => (group?.checkpoints ?? []).map((record) => ({ value: record.run_key, label: `${stepLabel(record.parent_step)} · ${formatScore(record.score)}${usable(record) ? '' : ' (failed)'}` }));
  const diff = (record: EvaluationRecord | null | undefined) => record && usable(record) && baseScore != null ? (record.score as number) - baseScore : undefined;
  return <div>
    <p className="type-eyebrow">EVALUATION COMPARISON</p>
    <h1 className="type-page-title mt-1.5">Compare evals</h1>
    <p className="type-page-subtitle mt-2">Two training runs from the same base model, on an evaluation suite both have checkpoints on. {index?.score_definition}</p>
    {error && <p role="alert" className="mt-4 text-xs text-rose-700">Evaluations unavailable: {error}</p>}
    {!error && loading && !index && <p className="mt-4 text-xs text-muted">Loading evaluations…</p>}
    {index && trainingRuns.length < 2 && <p className="obs-card mt-4 p-5 text-xs text-muted">Comparison needs two training runs with evaluated checkpoints; this project has {trainingRuns.length}.</p>}
    {a && trainingRuns.length >= 2 && <section className="obs-card mt-4 p-4" aria-label="Choose runs">
      <div className="grid gap-3 md:grid-cols-2">
        <div className="grid gap-2">
          <Select label="Run A" value={a.runId} options={runOptions(trainingRuns)} onChange={(value) => { setRunA(value); setStepA(''); setStepB(''); }} />
          {groupA && <Select label="Run A checkpoint" value={checkpointA?.run_key ?? ''} options={stepOptions(groupA)} onChange={setStepA} />}
        </div>
        <div className="grid gap-2">
          {b ? <Select label="Run B (same base model)" value={b.runId} options={runOptions(partners)} onChange={(value) => { setRunB(value); setStepB(''); }} /> : <p className="text-xs text-muted">No other training run from {a.model ?? 'this base model'} shares an evaluation suite with run A.</p>}
          {groupB && <Select label="Run B checkpoint" value={checkpointB?.run_key ?? ''} options={stepOptions(groupB)} onChange={setStepB} />}
        </div>
      </div>
      {shared.length > 0 && <div className="mt-3 max-w-xl"><Select label={`Evaluation suite (${shared.length} shared)`} value={activeSuite} options={shared.map((item) => ({ value: item, label: suiteLabel(item) }))} onChange={(value) => { setSuite(value); setStepA(''); setStepB(''); }} /></div>}
    </section>}
    {groupA && groupB && <section className="obs-card mt-4 p-4" aria-label="Comparison">
      <p className="break-all text-[10px] text-muted">{activeSuite} · base model {a?.model ?? 'not recorded'}</p>
      <div className="mt-2 grid gap-2 sm:grid-cols-4">
        <Stat label="Base model" value={formatScore(baseScore)} note={base.length ? `mean of ${base.length} ${base.length === 1 ? 'run' : 'runs'}` : 'no base-model evaluation on this suite'} />
        <Stat label={`A · ${stepLabel(checkpointA?.parent_step)}`} value={formatScore(checkpointA?.score)} delta={diff(checkpointA)} note={a ? displayName(a.runId) : undefined} />
        <Stat label={`B · ${stepLabel(checkpointB?.parent_step)}`} value={formatScore(checkpointB?.score)} delta={diff(checkpointB)} note={b ? displayName(b.runId) : undefined} />
        <div className="min-w-0 rounded-[5px] border border-divider bg-surface px-3 py-2.5">
          <span className="type-label block">B − A</span>
          <strong className={`mt-1 block font-serif text-2xl font-normal ${deltaClass(checkpointA?.score != null && checkpointB?.score != null ? checkpointB.score - checkpointA.score : null)}`}>{formatDelta(checkpointA?.score != null && checkpointB?.score != null ? checkpointB.score - checkpointA.score : null)}</strong>
          <small className="block text-[10px] text-muted">score difference</small>
        </div>
      </div>
      <BehaviourTable columns={[
        { label: 'Base', records: base },
        { label: `A · ${stepLabel(checkpointA?.parent_step)}`, records: checkpointA ? [checkpointA] : [] },
        { label: `B · ${stepLabel(checkpointB?.parent_step)}`, records: checkpointB ? [checkpointB] : [] },
      ]} />
      <div className="mt-3">
        <EvidenceChart series={series} height={220} compact ariaLabel="Score by checkpoint step for runs A and B" metricLabels={{ 'Run A': `A · ${a ? displayName(a.runId) : ''}`, 'Run B': `B · ${b ? displayName(b.runId) : ''}`, 'Base model': 'Base model' }} metricUnits={{ 'Run A': SCORE_UNIT, 'Run B': SCORE_UNIT, 'Base model': SCORE_UNIT }} xAxis={{ name: 'Checkpoint step (0 = base model)' }} />
      </div>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px]">
        {checkpointA && <RunLink record={checkpointA} label={`Open A's evaluation (${stepLabel(checkpointA.parent_step)})`} onOpenRun={onOpenRun} />}
        {checkpointB && <RunLink record={checkpointB} label={`Open B's evaluation (${stepLabel(checkpointB.parent_step)})`} onOpenRun={onOpenRun} />}
        {a && <button type="button" onClick={() => onOpenRun(a.runKey)} className="text-violet-700 hover:underline">Open run A</button>}
        {b && <button type="button" onClick={() => onOpenRun(b.runKey)} className="text-violet-700 hover:underline">Open run B</button>}
      </div>
      <TaskComparisonTable
        title="Score by task"
        rows={rows}
        loading={tasks.loading}
        error={tasks.error}
        columns={[
          ...(has('base') ? [{ key: 'base', label: 'Base' }] : []),
          ...(has('a') ? [{ key: 'a', label: `A · ${stepLabel(checkpointA?.parent_step)}` }] : []),
          ...(has('b') ? [{ key: 'b', label: `B · ${stepLabel(checkpointB?.parent_step)}` }] : []),
        ]}
        deltas={[
          ...(has('base') && has('a') ? [{ key: 'da', label: 'A vs base', minus: 'a', from: 'base' }] : []),
          ...(has('base') && has('b') ? [{ key: 'db', label: 'B vs base', minus: 'b', from: 'base' }] : []),
          ...(has('a') && has('b') ? [{ key: 'dba', label: 'B − A', minus: 'b', from: 'a' }] : []),
        ]}
      />
    </section>}
  </div>;
}
