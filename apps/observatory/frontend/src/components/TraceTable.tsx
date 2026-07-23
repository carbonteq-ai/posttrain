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
import { CaretDown, CaretUp } from '@phosphor-icons/react';

import type { TraceSummary } from '../lib/api';

const column = createColumnHelper<TraceSummary>();

type TraceTableProps = {
  traces: TraceSummary[];
  selectedId: string | null;
  sorting: SortingState;
  onSortingChange: (sorting: SortingState) => void;
  onSelect: (trace: TraceSummary) => void;
};

export function TraceTable({
  traces,
  selectedId,
  sorting,
  onSortingChange,
  onSelect,
}: TraceTableProps) {
  const columns = useMemo(
    () => [
      column.accessor('external_id', {
        header: 'Trace',
        size: 142,
        cell: (info) => (
          <button
            className="font-mono text-[11px] text-violet-700 hover:underline focus-visible:outline-2 focus-visible:outline-violet-600"
            onClick={() => onSelect(info.row.original)}
          >
            {info.getValue()}
          </button>
        ),
      }),
      column.accessor('task', { header: 'Slice', size: 100, cell: (info) => info.getValue() ?? '—' }),
      column.accessor('reward', {
        header: 'Reward',
        size: 84,
        cell: (info) => info.getValue()?.toFixed(3) ?? '—',
      }),
      column.accessor('success', {
        header: 'Outcome',
        size: 92,
        cell: (info) => (
          <span className={info.getValue() ? 'text-emerald-700' : 'text-rose-600'}>
            {info.getValue() ? 'Pass' : 'Review'}
          </span>
        ),
      }),
      column.accessor('tool_calls', { header: 'Tools', size: 68, cell: (info) => info.getValue() ?? '—' }),
      column.accessor('latency_ms', {
        header: 'Latency',
        size: 82,
        cell: (info) => (info.getValue() == null ? '—' : `${(info.getValue()! / 1000).toFixed(1)}s`),
      }),
      column.accessor('tokens', {
        header: 'Tokens',
        size: 78,
        cell: (info) => info.getValue()?.toLocaleString() ?? '—',
      }),
    ],
    [onSelect],
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
  const scrollRef = useRef<HTMLDivElement>(null);
  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 46,
    overscan: 8,
  });

  return (
    <div className="obs-card overflow-hidden">
      <div ref={scrollRef} className="max-h-[430px] overflow-auto">
        <table className="grid min-w-[720px] text-left text-[11px]" aria-label="Trace population">
          <thead className="sticky top-0 z-10 grid border-b border-divider bg-surface/95 text-[10px] font-medium text-muted backdrop-blur-sm">
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id} className="flex w-full">
                {headerGroup.headers.map((header) => (
                  <th key={header.id} style={{ width: header.getSize() }} className="shrink-0 px-3 py-2.5 font-medium">
                    <button
                      className="flex items-center gap-1 hover:text-ink"
                      onClick={header.column.getToggleSortingHandler()}
                    >
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      {header.column.getIsSorted() === 'asc' && <CaretUp size={10} />}
                      {header.column.getIsSorted() === 'desc' && <CaretDown size={10} />}
                    </button>
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody
            className="relative grid"
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
                    selectedId === row.original.external_id ? 'bg-violet-50/80 ring-1 ring-inset ring-violet-300' : ''
                  }`}
                  style={{ transform: `translateY(${virtualRow.start}px)` }}
                >
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id} style={{ width: cell.column.getSize() }} className="shrink-0 truncate px-3 py-2.5">
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
        {traces.length.toLocaleString()} visible traces
      </div>
    </div>
  );
}
