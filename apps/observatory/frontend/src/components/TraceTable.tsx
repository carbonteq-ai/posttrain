import { useMemo, useRef } from 'react';
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from '@tanstack/react-table';
import { useVirtualizer } from '@tanstack/react-virtual';
import {
  CaretDown,
  CaretUp,
  Check,
  Circle,
  Warning,
  X,
} from '@phosphor-icons/react';

import type { TraceSummary } from '../lib/api';
import type { TracePresentation } from '../lib/trace-presentation';

const column = createColumnHelper<TraceSummary>();

function compactMetricLabel(label: string): string {
  if (/^Task completed correctly$/i.test(label)) return 'Completed';
  return label.replace(/^Task\s+/i, '').replace(/^Assertions\s+/i, '');
}

type TraceTableProps = {
  traces: TraceSummary[];
  selectedId: string | null;
  metricColumns: Array<{ name: string; label: string }>;
  presentation: TracePresentation;
  sorting: SortingState;
  onSortingChange: (sorting: SortingState) => void;
  onSelect: (trace: TraceSummary) => void;
  total: number;
  hasMore: boolean;
  loadingMore: boolean;
  onLoadMore: () => void;
};

export function TraceTable({
  traces,
  selectedId,
  metricColumns,
  presentation,
  sorting,
  onSortingChange,
  onSelect,
  total,
  hasMore,
  loadingMore,
  onLoadMore,
}: TraceTableProps) {
  const hasReward = traces.some((trace) => trace.reward != null);
  const columns = useMemo(
    () => [
      column.accessor('external_id', {
        header: 'Trace ID',
        size: 78,
        cell: (info) => <span className="font-mono text-[9px] text-muted" title={info.getValue()}>{info.getValue().slice(0, 10)}</span>,
      }),
      column.accessor('prompt_preview', {
        header: 'Prompt preview',
        size: 190,
        cell: (info) => (
          <button
            aria-label={`Inspect ${info.row.original.task_label ?? info.row.original.task ?? info.row.original.external_id}`}
            className="block w-full truncate text-left text-[10px] text-secondary hover:text-violet-800 focus-visible:outline-2 focus-visible:outline-violet-600"
            title={info.getValue() ?? info.row.original.task_label ?? info.row.original.task ?? undefined}
            onClick={() => onSelect(info.row.original)}
          >
            {info.getValue() ?? info.row.original.task_label ?? info.row.original.task ?? 'No prompt preview'}
          </button>
        ),
      }),
      column.accessor('task_label', {
        header: presentation.mode === 'generic' ? 'Request' : 'Slice',
        size: 96,
        cell: (info) => <span className="block truncate text-[10px] text-secondary" title={info.getValue() ?? info.row.original.task ?? undefined}>{info.getValue() ?? info.row.original.task ?? (presentation.mode === 'generic' ? 'Request' : 'Unspecified')}</span>,
      }),
      ...(hasReward ? [column.accessor('reward', {
        header: 'Reward',
        size: 60,
        cell: (info) => {
          const value = info.getValue();
          return <span className={value != null && value < 0 ? 'font-medium text-rose-600' : 'font-medium text-ink'}>{value?.toFixed(3) ?? '—'}</span>;
        },
      })] : []),
      ...(metricColumns.length ? [column.group({
        id: 'signals',
        header: 'Reward components',
        columns: metricColumns.map((metric) => {
          const label = compactMetricLabel(metric.label);
          return column.accessor(
            (trace) => trace.reward_components[metric.name]
              ?? trace.native_metrics[metric.name]
              ?? trace.metrics[metric.name],
            {
              id: `signal:${metric.name}`,
              header: () => <span className="block w-full truncate leading-3" title={metric.label}>{label}</span>,
              size: Math.min(84, Math.max(64, label.length * 4 + 22)),
              cell: (info) => <span className="font-medium tabular-nums text-ink">{info.getValue()?.toFixed(2) ?? '—'}</span>,
            },
          );
        }),
      })] : []),
      column.accessor('outcome', {
        header: 'Outcome',
        size: 56,
        cell: (info) => <TraceOutcome outcome={info.row.original.outcome} presentation={presentation} />,
      }),
      column.accessor('tool_calls', { header: 'Tools', size: 44, cell: (info) => info.getValue() ?? '—' }),
      column.accessor('latency_ms', {
        header: 'Latency',
        size: 58,
        cell: (info) => (info.getValue() == null ? '—' : `${(info.getValue()! / 1000).toFixed(1)}s`),
      }),
      column.accessor('tokens', {
        header: 'Tokens',
        size: 60,
        cell: (info) => info.getValue()?.toLocaleString() ?? '—',
      }),
    ],
    [hasReward, metricColumns, onSelect, presentation],
  );
  const table = useReactTable({
    data: traces,
    columns,
    state: { sorting },
    onSortingChange: (updater) =>
      onSortingChange(typeof updater === 'function' ? updater(sorting) : updater),
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });
  const rows = table.getRowModel().rows;
  const leafHeaders = table.getFlatHeaders().filter((header) => header.subHeaders.length === 0);
  const metricHeaders = leafHeaders.filter((header) => header.id.startsWith('signal:'));
  const hasGroupedHeaders = metricHeaders.length > 0;
  const firstMetricColumn = leafHeaders.findIndex((header) => header.id.startsWith('signal:')) + 1;
  const scrollRef = useRef<HTMLDivElement>(null);
  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 46,
    overscan: 8,
  });

  return (
    <div className="obs-card overflow-hidden bg-white">
      <div className="flex items-center justify-between gap-3 border-b border-divider bg-white px-3 py-2.5">
        <div className="flex min-w-0 items-baseline gap-2">
          <h2 className="text-[12px] font-medium text-ink">{presentation.mode === 'optimization' ? 'Rollouts' : 'Traces'} ({traces.length.toLocaleString()})</h2>
          <span className="truncate text-[9px] text-muted">Prompt and task lead; trace ID stays secondary.</span>
        </div>
      </div>
      <div ref={scrollRef} className="max-h-[430px] overflow-auto bg-white">
        <table
          className="grid bg-white text-left text-[11px]"
          style={{ minWidth: Math.max(720, table.getTotalSize()) }}
          aria-label="Trace population"
        >
          <thead
            className="sticky top-0 z-10 grid border-b border-divider bg-white/95 text-[9px] font-medium text-muted backdrop-blur-sm"
            style={{
              gridTemplateColumns: leafHeaders.map((header) => `${header.getSize()}px`).join(' '),
              gridTemplateRows: hasGroupedHeaders ? '26px 26px' : '38px',
            }}
          >
            <tr className="contents">
              {leafHeaders.map((header, index) => {
                if (header.id.startsWith('signal:')) {
                  if (index + 1 !== firstMetricColumn) return null;
                  return <th
                    key="reward-components"
                    colSpan={metricHeaders.length}
                    style={{ gridColumn: `${firstMetricColumn} / span ${metricHeaders.length}`, gridRow: '1' }}
                    className="flex min-w-0 items-center justify-center overflow-hidden px-2 font-medium"
                  >
                    Reward components
                  </th>;
                }
                return <th
                  key={header.id}
                  rowSpan={hasGroupedHeaders ? 2 : 1}
                  style={{ gridColumn: `${index + 1}`, gridRow: `1 / span ${hasGroupedHeaders ? 2 : 1}` }}
                  className="flex min-w-0 items-center overflow-hidden px-2 font-medium"
                >
                  {header.column.getCanSort() ? <button
                    className="flex w-full min-w-0 items-center gap-1 overflow-hidden hover:text-ink"
                    onClick={header.column.getToggleSortingHandler()}
                  >
                    {flexRender(header.column.columnDef.header, header.getContext())}
                    {header.column.getIsSorted() === 'asc' && <CaretUp size={10} />}
                    {header.column.getIsSorted() === 'desc' && <CaretDown size={10} />}
                  </button> : flexRender(header.column.columnDef.header, header.getContext())}
                </th>;
              })}
            </tr>
            {hasGroupedHeaders && <tr className="contents">
              {metricHeaders.map((header) => {
                const columnStart = leafHeaders.findIndex((leaf) => leaf.id === header.id) + 1;
                return <th
                  key={header.id}
                  style={{ gridColumn: `${columnStart}`, gridRow: '2' }}
                  className="flex min-w-0 items-center overflow-hidden border-t border-divider/70 px-2 font-medium"
                >
                  {header.column.getCanSort() ? <button
                    className="flex w-full min-w-0 items-center gap-1 overflow-hidden hover:text-ink"
                    onClick={header.column.getToggleSortingHandler()}
                  >
                    {flexRender(header.column.columnDef.header, header.getContext())}
                    {header.column.getIsSorted() === 'asc' && <CaretUp size={10} />}
                    {header.column.getIsSorted() === 'desc' && <CaretDown size={10} />}
                  </button> : flexRender(header.column.columnDef.header, header.getContext())}
                </th>;
              })}
            </tr>}
          </thead>
          <tbody
            className="relative grid bg-white"
            style={{ height: `${virtualizer.getTotalSize()}px` }}
          >
            {virtualizer.getVirtualItems().map((virtualRow) => {
              const row = rows[virtualRow.index];
              return (
                <tr
                  key={row.id}
                  ref={virtualizer.measureElement}
                  data-index={virtualRow.index}
                  className={`absolute flex w-full border-b border-divider text-secondary transition-colors hover:bg-subtle/70 ${
                    selectedId === row.original.external_id ? 'bg-violet-50/80 ring-1 ring-inset ring-violet-300' : 'bg-white'
                  }`}
                  style={{ transform: `translateY(${virtualRow.start}px)` }}
                >
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id} style={{ width: cell.column.getSize() }} className="shrink-0 truncate px-2.5 py-2.5">
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="border-t border-divider bg-subtle/35 px-3 py-2 text-[10px] text-muted">
        <div className="flex items-center justify-between gap-3">
          <span>{traces.length.toLocaleString()} of {total.toLocaleString()} loaded</span>
          {hasMore && <button
            type="button"
            disabled={loadingMore}
            onClick={onLoadMore}
            className="rounded border border-divider bg-white px-2.5 py-1 text-[10px] font-medium text-violet-700 hover:border-violet-300 disabled:cursor-wait disabled:text-muted"
          >{loadingMore ? 'Loading…' : 'Load 100 more'}</button>}
        </div>
      </div>
    </div>
  );
}

export function TraceOutcome({
  outcome,
  presentation,
}: {
  outcome: TraceSummary['outcome'];
  presentation: TracePresentation;
}) {
  const label = presentation.outcomeLabel(outcome);
  const appearance = outcome === 'pass'
    ? { className: 'bg-emerald-50 text-emerald-700 ring-emerald-200', icon: Check }
    : outcome === 'review' || outcome === 'error'
      ? { className: 'bg-rose-50 text-rose-600 ring-rose-200', icon: X }
      : outcome === 'truncated'
        ? { className: 'bg-amber-50 text-amber-700 ring-amber-200', icon: Warning }
        : outcome === 'scored'
          ? { className: 'bg-violet-50 text-violet-700 ring-violet-200', icon: Check }
          : { className: 'bg-subtle text-muted ring-divider', icon: Circle };
  const Icon = appearance.icon;

  return (
    <span
      role="img"
      aria-label={label}
      title={label}
      className={`inline-flex size-5 items-center justify-center rounded-full ring-1 ring-inset ${appearance.className}`}
    >
      <Icon size={12} weight="bold" aria-hidden="true" />
    </span>
  );
}
