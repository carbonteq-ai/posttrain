import { useMemo, useState, type ReactNode } from 'react';

import type { MetricSeries, NoteData, RenderedNote, RenderedView } from '../lib/api';
import { MarkdownContent, type FencedBlock } from './ContentRenderer';
import { EvidenceChart, type ChartXAxis } from './EvidenceChart';
import { FilterInput, Pager, SortButton, sortBy, usePaging, type SortState } from './TableControls';

/** Fenced blocks the service writes in place of each view: ```note-view <index>. */
export const NOTE_VIEW_FENCE = 'note-view';
const TABLE_PAGE_ROWS = 15;

type ResultColumn = NoteData['columns'][number];

function optionList(value: unknown): string[] {
  if (value == null) return [];
  return (Array.isArray(value) ? value : [value]).map(String);
}

function optionText(value: unknown): string | null {
  return value == null || value === '' ? null : String(value);
}

function columnLabel(column: ResultColumn | undefined, name: string): string {
  return column?.label || name;
}

function isNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

export function formatCell(value: unknown): string {
  if (value == null) return '—';
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return String(value);
    if (value !== 0 && Math.abs(value) < 0.001) return value.toExponential(2);
    return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
  }
  if (typeof value === 'boolean' || typeof value === 'string') return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

/** A visible marker for a view the service could not resolve, never an empty chart. */
export function UnresolvedView({ kind, message }: { kind: string; message: string }) {
  return <div role="note" aria-label={`Unresolved ${kind}`} className="rounded-[4px] border border-amber-300 bg-amber-50 px-3 py-2 text-[11px] leading-5 text-amber-900">
    <span className="font-medium">Unresolved {kind}:</span> {message}
  </div>;
}

function QueryDisclosure({ query }: { query: string | null | undefined }) {
  if (!query) return null;
  return <details className="mt-2 border-t border-divider/70 pt-1.5">
    <summary className="cursor-pointer select-none text-[9px] font-medium uppercase tracking-[.08em] text-muted">Query</summary>
    <pre className="mt-1.5 max-h-60 overflow-auto whitespace-pre-wrap break-words rounded bg-ink/[.035] p-2 font-mono text-[10px] leading-4 text-secondary">{query}</pre>
  </details>;
}

type ChartModel = {
  series: MetricSeries[];
  labels: Record<string, string>;
  units: Record<string, string | null>;
  xAxis: ChartXAxis;
};

/**
 * Turn a chart view's rows into chart series: one series per y column, or,
 * with ``series``, one per distinct series value and y column. Numeric x
 * values are plotted on a value axis; anything else becomes categories.
 */
export function noteChartModel(view: RenderedView): ChartModel | null {
  const result = view.result;
  if (!result) return null;
  const options = view.options ?? {};
  const columns = new Map(result.columns.map((column, index) => [column.name, { column, index }]));
  const xName = optionText(options.x);
  const x = xName == null ? undefined : columns.get(xName);
  const yNames = optionList(options.y);
  const seriesName = optionText(options.series);
  const split = seriesName == null ? undefined : columns.get(seriesName);
  if (!x || yNames.length === 0 || (seriesName != null && !split)) return null;
  const xValues = result.rows.map((row) => row[x.index]);
  const numericX = xValues.every(isNumber);
  const categories = numericX ? undefined : [...new Set(xValues.map(formatCell))];
  const series = new Map<string, MetricSeries>();
  const labels: Record<string, string> = {};
  const units: Record<string, string | null> = {};
  for (const yName of yNames) {
    const y = columns.get(yName);
    if (!y) return null;
    const yLabel = columnLabel(y.column, yName);
    for (const row of result.rows) {
      const value = row[y.index];
      if (!isNumber(value)) continue;
      const group = split ? formatCell(row[split.index]) : null;
      const name = group == null ? yName : `${yName}\u0000${group}`;
      let item = series.get(name);
      if (!item) {
        item = { name, points: [] };
        series.set(name, item);
        labels[name] = group == null ? yLabel : yNames.length > 1 ? `${group} · ${yLabel}` : group;
        // Series of one column share an axis; a real unit also formats values.
        units[name] = y.column.unit || yLabel;
      }
      const xValue = row[x.index];
      item.points.push({ value, step: numericX ? xValue as number : categories!.indexOf(formatCell(xValue)) });
    }
  }
  return { series: [...series.values()], labels, units, xAxis: { name: columnLabel(x.column, x.column.name), categories } };
}

export function NoteChart({ view }: { view: RenderedView }) {
  const model = useMemo(() => noteChartModel(view), [view]);
  const options = view.options ?? {};
  const title = optionText(options.title) ?? optionList(options.y).join(', ');
  if (!model) return <UnresolvedView kind="chart" message="the chart's columns are missing from its data" />;
  if (model.series.length === 0) return <UnresolvedView kind="chart" message="the data has no numeric values to plot" />;
  return <div>
    <p className="mb-1 text-[11px] font-medium text-ink">{title}</p>
    <EvidenceChart
      series={model.series}
      height={240}
      compact
      ariaLabel={title}
      metricLabels={model.labels}
      metricUnits={model.units}
      seriesType={options.type === 'bar' ? 'bar' : 'line'}
      xAxis={model.xAxis}
    />
  </div>;
}

export function NoteValue({ view }: { view: RenderedView }) {
  const options = view.options ?? {};
  const label = optionText(options.label) ?? optionText(options.column) ?? 'Value';
  const where = optionText(options.where);
  const compare = optionText(options.compare);
  if (view.compare_formatted == null) {
    return <div>
      <span className="type-label block">{label}</span>
      <strong className="mt-1 block font-serif text-2xl font-normal text-ink">{view.formatted ?? '—'}</strong>
      {where && <small className="text-[10px] text-muted">{where}</small>}
    </div>;
  }
  return <div>
    <span className="type-label block">{label}</span>
    <div className="mt-1 flex flex-wrap items-end gap-x-5 gap-y-2">
      <div>
        <strong className="block font-serif text-2xl font-normal text-ink">{view.formatted ?? '—'}</strong>
        {where && <small className="text-[10px] text-muted">{where}</small>}
      </div>
      <div>
        <strong className="block font-serif text-2xl font-normal text-secondary">{view.compare_formatted}</strong>
        {compare && <small className="text-[10px] text-muted">{compare}</small>}
      </div>
      {view.difference && <div className="pb-1 text-[11px] text-secondary"><span className="text-muted">Difference</span> <strong className="font-medium text-ink">{view.difference}</strong></div>}
    </div>
  </div>;
}

export function NoteTable({ view }: { view: RenderedView }) {
  const result = view.result;
  const options = view.options ?? {};
  const [sort, setSort] = useState<SortState>(null);
  const [filter, setFilter] = useState('');
  const allRows = result?.rows ?? [];
  const byName = new Map((result?.columns ?? []).map((column, index) => [column.name, { column, index }]));
  const names = optionList(options.columns);
  const selected = (names.length ? names : (result?.columns ?? []).map((column) => column.name))
    .map((name) => ({ name, entry: byName.get(name) }))
    .filter((item): item is { name: string; entry: { column: ResultColumn; index: number } } => item.entry != null);
  const numeric = selected.map(({ entry }) => allRows.some((row) => isNumber(row[entry.index]))
    && allRows.every((row) => row[entry.index] == null || isNumber(row[entry.index])));
  const needle = filter.trim().toLowerCase();
  const matching = needle
    ? allRows.filter((row) => selected.some(({ entry }) => formatCell(row[entry.index]).toLowerCase().includes(needle)))
    : allRows;
  const sorted = sortBy(matching, sort, (row, key) => row[Number(key)]);
  const paging = usePaging(sorted, TABLE_PAGE_ROWS, `${needle}:${sort?.key}:${sort?.direction}`);
  if (!result) return <UnresolvedView kind="table" message="no data" />;
  const title = optionText(options.title);
  const searchable = allRows.length > TABLE_PAGE_ROWS / 3;
  return <div>
    {(title || searchable) && <div className="mb-1.5 flex flex-wrap items-center justify-between gap-2">
      {title ? <p className="text-[11px] font-medium text-ink">{title}</p> : <span />}
      {searchable && <FilterInput value={filter} onChange={setFilter} label={`Filter ${title ?? 'table'} rows`} />}
    </div>}
    <div className="overflow-hidden rounded-[4px] border border-divider">
      <div className="overflow-x-auto">
      <table className="obs-data-table w-full text-left text-[11px]">
        <thead className="bg-subtle text-[10px] text-muted">
          <tr>{selected.map(({ name, entry }, index) => <th key={name} scope="col" aria-sort={sort?.key === String(entry.index) ? (sort.direction === 'asc' ? 'ascending' : 'descending') : undefined} className={`whitespace-nowrap px-2.5 py-1.5 font-medium ${numeric[index] ? 'text-right' : ''}`}>
            <SortButton
              sortKey={String(entry.index)}
              sort={sort}
              numeric={numeric[index]}
              onSort={setSort}
              label={<>{columnLabel(entry.column, name)}{entry.column.unit ? <span className="font-normal"> ({entry.column.unit})</span> : null}</>}
              title={`Sort by ${columnLabel(entry.column, name)}`}
            />
          </th>)}</tr>
        </thead>
        <tbody className="divide-y divide-divider">
          {paging.items.map((row, rowIndex) => <tr key={rowIndex}>{selected.map(({ name, entry }, index) => <td key={name} className={`px-2.5 py-1.5 text-secondary ${numeric[index] ? 'text-right font-mono tabular-nums' : ''}`}>
            {formatCell(row[entry.index])}
          </td>)}</tr>)}
          {!paging.items.length && <tr><td colSpan={selected.length} className="px-2.5 py-4 text-center text-muted">No rows match “{filter.trim()}”.</td></tr>}
        </tbody>
      </table>
      </div>
      {(sorted.length > TABLE_PAGE_ROWS || needle) && <Pager paging={paging} count={sorted.length} noun={needle ? `matching rows · ${allRows.length.toLocaleString()} in all` : 'rows'} />}
    </div>
    {result.truncated && <p className="mt-1 text-[10px] text-muted">The query result was truncated.</p>}
  </div>;
}

/** One rendered view: its component, or an unresolved marker, plus the collapsed query. */
export function NoteView({ view }: { view: RenderedView }) {
  let body: ReactNode;
  if (view.error) body = <UnresolvedView kind={view.kind} message={view.error} />;
  else if (!view.result) body = <UnresolvedView kind={view.kind} message="no data was returned" />;
  else if (view.kind === 'chart') body = <NoteChart view={view} />;
  else if (view.kind === 'value') body = <NoteValue view={view} />;
  else body = <NoteTable view={view} />;
  return <figure className="my-3 rounded-[5px] border border-divider bg-surface px-3 py-2.5" data-note-view={view.index} aria-label={`Note ${view.kind}`}>
    {body}
    <QueryDisclosure query={view.query} />
  </figure>;
}

/**
 * Note Markdown with each ```note-view <index> block drawn from the rendered
 * views. Every other fenced block stays a code block, and raw HTML is dropped.
 */
export function NoteMarkdown({ rendered, compact = false }: { rendered: RenderedNote; compact?: boolean }) {
  const views = useMemo(() => new Map((rendered.views ?? []).map((view) => [view.index, view])), [rendered.views]);
  const fencedBlock = (block: FencedBlock): ReactNode | undefined => {
    if (block.language !== NOTE_VIEW_FENCE) return undefined;
    const index = Number.parseInt(block.meta ?? '', 10);
    const view = Number.isInteger(index) ? views.get(index) : undefined;
    return view
      ? <NoteView key={index} view={view} />
      : <UnresolvedView kind="view" message={`view ${block.meta ?? '?'} is missing from the rendered note`} />;
  };
  return <MarkdownContent compact={compact} hardBreaks={false} fencedBlock={fencedBlock}>{rendered.markdown}</MarkdownContent>;
}
