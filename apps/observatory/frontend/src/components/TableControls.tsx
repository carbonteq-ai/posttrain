import { useEffect, useState, type ReactNode } from 'react';
import { CaretDown, CaretLeft, CaretRight, CaretUp, CaretUpDown, MagnifyingGlass } from '@phosphor-icons/react';

export type Paging<T> = {
  page: number;
  pageSize: number;
  items: T[];
  /** Pages available without another fetch. */
  loadedPages: number;
  setPage: (page: number) => void;
};

/**
 * One page of ``items``. The page returns to the first when ``resetKey``
 * changes (a new filter or sort) and steps back when the list shrinks under
 * it, except while ``loading`` brings in the rows a forward step asked for.
 */
export function usePaging<T>(items: T[], pageSize: number, resetKey: unknown = null, loading = false): Paging<T> {
  const [page, setPage] = useState(0);
  const [key, setKey] = useState(resetKey);
  if (key !== resetKey) {
    setKey(resetKey);
    setPage(0);
  }
  const loadedPages = Math.max(1, Math.ceil(items.length / pageSize));
  useEffect(() => {
    if (!loading && page >= loadedPages) setPage(loadedPages - 1);
  }, [loading, page, loadedPages]);
  return { page, pageSize, items: items.slice(page * pageSize, (page + 1) * pageSize), loadedPages, setPage };
}

function pageNumbers(current: number, count: number): Array<number | null> {
  if (count <= 7) return Array.from({ length: count }, (_, index) => index);
  const keep = new Set([0, count - 1, current - 1, current, current + 1].filter((page) => page >= 0 && page < count));
  const sorted = [...keep].sort((left, right) => left - right);
  return sorted.flatMap((page, index) => index > 0 && page - sorted[index - 1] > 1 ? [null, page] : [page]);
}

/**
 * Page controls under a table. ``total`` is the full row count when known;
 * with ``hasMore`` the last loaded page's Next asks ``onLoadMore`` for the
 * following rows before moving on.
 */
export function Pager<T>({ paging, count, total, noun = 'rows', hasMore = false, loading = false, onLoadMore, className = '' }: {
  paging: Paging<T>;
  /** Rows available now (loaded). */
  count: number;
  total?: number | null;
  noun?: string;
  hasMore?: boolean;
  loading?: boolean;
  onLoadMore?: () => void;
  className?: string;
}) {
  const { page, pageSize, loadedPages, setPage } = paging;
  const first = count === 0 ? 0 : page * pageSize + 1;
  const last = Math.min(count, (page + 1) * pageSize);
  const known = total != null && total > count ? total : count;
  const canLoad = hasMore && onLoadMore != null;
  const onLastLoaded = page >= loadedPages - 1;
  const next = () => {
    if (onLastLoaded && canLoad) onLoadMore();
    setPage(page + 1);
  };
  if (loadedPages <= 1 && !canLoad) {
    return <footer className={`border-t border-divider bg-subtle/35 px-3 py-2 text-[10px] text-muted ${className}`}>
      {count.toLocaleString()}{total != null && total > count ? ` of ${total.toLocaleString()}` : ''} {noun}
    </footer>;
  }
  return <nav aria-label="Pages" className={`flex flex-wrap items-center justify-between gap-2 border-t border-divider bg-subtle/35 px-3 py-1.5 text-[10px] text-muted ${className}`}>
    <span aria-live="polite">{loading && page >= loadedPages ? 'Loading…' : `${first.toLocaleString()}–${last.toLocaleString()} of ${known.toLocaleString()}${canLoad && total == null ? '+' : ''} ${noun}`}{canLoad && total != null && total > count ? ` · ${count.toLocaleString()} loaded` : ''}</span>
    <div className="flex items-center gap-0.5">
      <button type="button" aria-label="Previous page" disabled={page === 0} onClick={() => setPage(page - 1)} className="grid size-6 place-items-center rounded text-secondary hover:bg-white hover:text-ink disabled:opacity-35 disabled:hover:bg-transparent"><CaretLeft size={12} /></button>
      {pageNumbers(Math.min(page, loadedPages - 1), loadedPages).map((item, index) => item == null
        ? <span key={`gap-${index}`} className="px-1">…</span>
        : <button key={item} type="button" aria-label={`Page ${item + 1}`} aria-current={item === page ? 'page' : undefined} onClick={() => setPage(item)} className={`min-w-6 rounded px-1.5 py-1 tabular-nums ${item === page ? 'bg-white font-medium text-violet-800 shadow-sm ring-1 ring-divider' : 'text-secondary hover:bg-white hover:text-ink'}`}>{item + 1}</button>)}
      {canLoad && <span className="px-1">…</span>}
      <button type="button" aria-label="Next page" disabled={loading || (onLastLoaded && !canLoad)} onClick={next} className="grid size-6 place-items-center rounded text-secondary hover:bg-white hover:text-ink disabled:opacity-35 disabled:hover:bg-transparent"><CaretRight size={12} /></button>
    </div>
  </nav>;
}

export type SortDirection = 'asc' | 'desc';
export type SortState = { key: string; direction: SortDirection } | null;

/** Missing values sort last in either direction; numbers by value, text in natural order. */
export function compareValues(left: unknown, right: unknown, direction: SortDirection): number {
  const missing = (value: unknown) => value == null || value === '' || (typeof value === 'number' && !Number.isFinite(value));
  if (missing(left) || missing(right)) return missing(left) === missing(right) ? 0 : missing(left) ? 1 : -1;
  const order = typeof left === 'number' && typeof right === 'number'
    ? left - right
    : String(left).localeCompare(String(right), undefined, { numeric: true, sensitivity: 'base' });
  return direction === 'asc' ? order : -order;
}

export function sortBy<T>(items: T[], sort: SortState, value: (item: T, key: string) => unknown): T[] {
  if (!sort) return items;
  return items
    .map((item, index) => ({ item, index }))
    .sort((left, right) => compareValues(value(left.item, sort.key), value(right.item, sort.key), sort.direction) || left.index - right.index)
    .map(({ item }) => item);
}

/** A click sorts, a second reverses and a third returns to the default order. Numbers start high. */
export function nextSort(sort: SortState, key: string, numeric: boolean): SortState {
  const first: SortDirection = numeric ? 'desc' : 'asc';
  if (sort?.key !== key) return { key, direction: first };
  return sort.direction === first ? { key, direction: first === 'asc' ? 'desc' : 'asc' } : null;
}

export function SortButton({ label, sortKey, sort, numeric = false, onSort, title }: {
  label: ReactNode;
  sortKey: string;
  sort: SortState;
  numeric?: boolean;
  onSort: (sort: SortState) => void;
  title?: string;
}) {
  const active = sort?.key === sortKey ? sort.direction : null;
  return <button
    type="button"
    title={title}
    onClick={() => onSort(nextSort(sort, sortKey, numeric))}
    className={`inline-flex max-w-full items-center gap-0.5 hover:text-ink ${numeric ? 'flex-row-reverse' : ''} ${active ? 'text-violet-800' : ''}`}
  >
    <span className="truncate">{label}</span>
    {active === 'asc' ? <CaretUp size={9} weight="bold" aria-hidden="true" /> : active === 'desc' ? <CaretDown size={9} weight="bold" aria-hidden="true" /> : <CaretUpDown size={9} className="opacity-30" aria-hidden="true" />}
  </button>;
}

export function FilterInput({ value, onChange, placeholder = 'Filter rows', label = 'Filter rows' }: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  label?: string;
}) {
  return <label className="relative flex w-full max-w-[260px] items-center">
    <MagnifyingGlass size={11} className="pointer-events-none absolute left-2 text-muted" aria-hidden="true" />
    <input
      type="search"
      aria-label={label}
      value={value}
      placeholder={placeholder}
      onChange={(event) => onChange(event.target.value)}
      className="obs-control h-7 w-full pl-6 pr-2 text-[11px] text-ink placeholder:text-muted focus:border-violet-300 focus:outline-none"
    />
  </label>;
}
