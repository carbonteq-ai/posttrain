import { useMemo, useState, type ReactNode } from 'react';
import { CaretRight, CheckCircle, Flag, WarningCircle } from '@phosphor-icons/react';

import type { TraceTimeline, TraceTimelineSegment } from '../lib/api';
import { MessageContent } from './ContentRenderer';
import { formatDuration } from './RolloutTime';
import { TranscriptMessage } from './TranscriptMessage';

type JsonRecord = Record<string, unknown>;

export type ToolExchange = {
  id: string | null;
  name: string;
  args: unknown;
  result: JsonRecord | null;
};

export type RunTurn = {
  index: number;
  node: number;
  message: JsonRecord;
  exchanges: ToolExchange[];
  /** Tool messages the harness returned without a matching call in this turn. */
  unmatched: JsonRecord[];
  /** Non-assistant messages that arrived after this turn (e.g. an environment reply). */
  followUps: JsonRecord[];
  final: boolean;
};

export type ParsedRun = { preamble: JsonRecord[]; turns: RunTurn[] };

function asRecord(value: unknown): JsonRecord | null {
  return value != null && typeof value === 'object' && !Array.isArray(value) ? value as JsonRecord : null;
}

function normalize(message: JsonRecord): JsonRecord {
  const nested = asRecord(message.message);
  return nested == null ? message : { ...nested, ...message };
}

function role(message: JsonRecord): string {
  return String(message.role ?? 'event').toLowerCase();
}

function parseJson(value: unknown): unknown {
  if (typeof value !== 'string') return value;
  const trimmed = value.trim();
  if (!trimmed.startsWith('{') && !trimmed.startsWith('[')) return value;
  try {
    return JSON.parse(trimmed) as unknown;
  } catch {
    return value;
  }
}

function toolCalls(message: JsonRecord): ToolExchange[] {
  const raw = message.tool_calls ?? message.function_call;
  const calls = Array.isArray(raw) ? raw : raw == null ? [] : [raw];
  return calls.map((candidate, index) => {
    const call = asRecord(candidate) ?? {};
    const fn = asRecord(call.function);
    return {
      id: typeof call.id === 'string' ? call.id : null,
      name: String(call.name ?? fn?.name ?? `tool ${index + 1}`),
      args: parseJson(call.arguments ?? fn?.arguments ?? {}),
      result: null,
    };
  });
}

/** Split a transcript into the prompt and one turn per model call, pairing each call with its result. */
export function parseRun(messages: JsonRecord[]): ParsedRun {
  const preamble: JsonRecord[] = [];
  const turns: RunTurn[] = [];
  messages.forEach((raw, node) => {
    const message = normalize(raw);
    const kind = role(message);
    const current = turns.at(-1);
    if (kind === 'assistant') {
      turns.push({ index: turns.length, node, message, exchanges: toolCalls(message), unmatched: [], followUps: [], final: false });
    } else if ((kind === 'tool' || kind === 'function') && current) {
      // Call ids restart every turn, so results pair only within their own turn.
      const id = typeof message.tool_call_id === 'string' ? message.tool_call_id : null;
      const target = current.exchanges.find((exchange) => exchange.result == null && id != null && exchange.id === id)
        ?? current.exchanges.find((exchange) => exchange.result == null && (id == null || exchange.id == null));
      if (target) target.result = message;
      else current.unmatched.push(message);
    } else if (current) {
      current.followUps.push(message);
    } else {
      preamble.push(message);
    }
  });
  const last = turns.at(-1);
  if (last && last.exchanges.length === 0) last.final = true;
  return { preamble, turns };
}

export type ResultSummary = { status: 'ok' | 'error' | 'missing'; text: string; chars: number };

const COUNT_KEYS = ['result_count', 'count', 'total', 'total_count', 'num_results'];

function errorText(record: JsonRecord): string | null {
  const error = record.error ?? record.errors;
  const message = asRecord(error)?.message ?? error ?? record.message;
  if (record.success === false || record.ok === false || record.status === 'error' || (error != null && error !== false && error !== '')) {
    return typeof message === 'string' && message ? message : 'failed';
  }
  return null;
}

/** One line saying whether a tool result succeeded and what it returned. */
export function summarizeResult(result: JsonRecord | null): ResultSummary {
  if (!result) return { status: 'missing', text: 'no result recorded', chars: 0 };
  const content = result.content;
  const raw = typeof content === 'string' ? content : JSON.stringify(content ?? null);
  const parsed = parseJson(content);
  const record = asRecord(parsed);
  if (record) {
    const error = errorText(record);
    if (error) return { status: 'error', text: error, chars: raw.length };
    const counted = COUNT_KEYS.find((key) => typeof record[key] === 'number');
    const array = Object.entries(record).find(([, value]) => Array.isArray(value));
    const parts = [
      counted ? (counted === 'result_count' ? `${String(record[counted])} ${record[counted] === 1 ? 'result' : 'results'}` : `${counted.replace(/_/g, ' ')} ${String(record[counted])}`) : null,
      !counted && array ? `${(array[1] as unknown[]).length} ${array[0].replace(/_/g, ' ')}` : null,
    ].filter(Boolean);
    return { status: 'ok', text: parts.join(' · ') || 'ok', chars: raw.length };
  }
  if (Array.isArray(parsed)) return { status: 'ok', text: `${parsed.length} items`, chars: raw.length };
  const text = String(parsed ?? '').trim();
  if (/^(error|exception|traceback)\b/i.test(text)) return { status: 'error', text: text.split('\n')[0].slice(0, 160), chars: raw.length };
  return { status: 'ok', text: text.split('\n')[0].slice(0, 120) || 'empty', chars: raw.length };
}

function argsPreview(args: unknown): string {
  const record = asRecord(args);
  if (!record) return typeof args === 'string' ? args.slice(0, 80) : '';
  return Object.entries(record).map(([key, value]) => {
    const text = typeof value === 'string' ? JSON.stringify(value) : JSON.stringify(value) ?? 'null';
    return `${key}=${text.length > 32 ? `${text.slice(0, 31)}…` : text}`;
  }).join(', ');
}

function pretty(value: unknown): string {
  const parsed = parseJson(value);
  if (typeof parsed === 'string') return parsed;
  return JSON.stringify(parsed, null, 2) ?? '';
}

const JSON_TOKEN = /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g;

/** Exact JSON text with keys, strings, and literals coloured; never reformats values into tables. */
function Code({ text }: { text: string }) {
  const nodes: ReactNode[] = [];
  let last = 0;
  for (const match of text.matchAll(JSON_TOKEN)) {
    const start = match.index ?? 0;
    if (start > last) nodes.push(text.slice(last, start));
    const [token, string, colon, literal] = match;
    const className = string ? (colon ? 'text-violet-800' : 'text-emerald-800') : literal ? 'text-rose-700' : 'text-sky-800';
    nodes.push(<span key={start} className={className}>{string ?? token}</span>);
    if (colon) nodes.push(colon);
    last = start + token.length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-[4px] bg-ink/[.03] p-2.5 font-mono text-[10px] leading-4 text-secondary">{nodes}</pre>;
}

function ExchangeRow({ exchange, open }: { exchange: ToolExchange; open: boolean }) {
  const summary = summarizeResult(exchange.result);
  const failed = summary.status !== 'ok';
  return <details open={open || undefined} className={`group rounded-[4px] border ${failed ? 'border-rose-200 bg-rose-50/40' : 'border-divider bg-white'}`} aria-label={`Tool ${exchange.name}`}>
    <summary className="flex cursor-pointer list-none items-center gap-2 px-2.5 py-1.5 text-[11px]">
      <CaretRight size={10} className="shrink-0 text-muted transition-transform group-open:rotate-90" />
      {failed ? <WarningCircle size={13} weight="fill" className="shrink-0 text-rose-600" aria-label="Failed" /> : <CheckCircle size={13} weight="fill" className="shrink-0 text-emerald-600" aria-label="Succeeded" />}
      <code className="shrink-0 font-medium text-ink">{exchange.name}</code>
      <code className="min-w-0 flex-1 truncate text-[10px] text-muted" title={argsPreview(exchange.args)}>({argsPreview(exchange.args)})</code>
      <span className={`max-w-[40%] shrink-0 truncate text-[10px] ${failed ? 'font-medium text-rose-700' : 'text-secondary'}`} title={summary.text}>→ {summary.text}</span>
      {summary.chars > 0 && <span className="shrink-0 font-mono text-[9px] tabular-nums text-muted">{summary.chars >= 1000 ? `${(summary.chars / 1000).toFixed(1)}k` : summary.chars} ch</span>}
    </summary>
    <div className="grid gap-2 border-t border-divider px-2.5 py-2 lg:grid-cols-2">
      <div><p className="mb-1 text-[9px] font-medium uppercase tracking-[.08em] text-muted">Arguments</p><Code text={pretty(exchange.args)} /></div>
      <div><p className="mb-1 text-[9px] font-medium uppercase tracking-[.08em] text-muted">Result</p>{exchange.result ? <Code text={pretty(exchange.result.content)} /> : <p className="text-[10px] text-rose-700">The harness recorded no result for this call.</p>}</div>
    </div>
  </details>;
}

function Reasoning({ text }: { text: string }) {
  const [full, setFull] = useState(false);
  const long = text.length > 360 || text.split('\n').length > 4;
  return <div className="rounded-[4px] border-l-2 border-violet-300 bg-violet-50/50 py-1.5 pl-2.5 pr-2">
    <p className="text-[9px] font-medium uppercase tracking-[.08em] text-violet-700">Reasoning</p>
    <p className={`mt-0.5 whitespace-pre-wrap break-words text-[11px] leading-[17px] text-secondary ${long && !full ? 'line-clamp-4' : ''}`}>{text}</p>
    {long && <button type="button" className="mt-0.5 text-[10px] font-medium text-violet-700 hover:underline" onClick={() => setFull(!full)}>{full ? 'Show less' : 'Show full reasoning'}</button>}
  </div>;
}

type TurnTiming = { model: TraceTimelineSegment; toolsMs: number };

function turnTimings(timeline: TraceTimeline | null | undefined, turns: RunTurn[]): Map<number, TurnTiming> {
  const timings = new Map<number, TurnTiming>();
  if (!timeline) return timings;
  const segments = timeline.segments;
  const byNode = new Map(turns.map((turn) => [turn.node, turn.index]));
  segments.forEach((segment, position) => {
    if (segment.kind !== 'inference') return;
    const turn = segment.node != null && byNode.has(segment.node) ? byNode.get(segment.node) : segment.call_index;
    if (turn == null) return;
    let toolsMs = 0;
    for (const next of segments.slice(position + 1)) {
      if (next.kind === 'inference') break;
      if (next.kind === 'tools' || next.kind === 'harness') toolsMs += next.duration_ms;
    }
    timings.set(turn, { model: segment, toolsMs });
  });
  return timings;
}

function TurnHeader({ turn, timing, failures }: { turn: RunTurn; timing: TurnTiming | undefined; failures: number }) {
  const model = timing?.model;
  const truncated = model?.finish_reason === 'length';
  return <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5">
    <h4 className="text-[12px] font-medium text-ink">{turn.final ? 'Final answer' : `Turn ${turn.index + 1}`}</h4>
    {model && <span className="font-mono text-[10px] tabular-nums text-muted" title="Time since the rollout started">+{formatDuration(model.start_ms)}</span>}
    <span className="flex flex-wrap gap-x-2.5 text-[10px] tabular-nums text-secondary">
      {model && <span title="Model call, including queueing in the inference server">model {formatDuration(model.duration_ms)}</span>}
      {timing && timing.toolsMs > 0 && <span title="Harness time running this turn's tool calls">tools {formatDuration(timing.toolsMs)}</span>}
      {model?.completion_tokens != null && <span>{model.completion_tokens.toLocaleString()} out{model.thinking_tokens != null ? ` · ${model.thinking_tokens.toLocaleString()} thinking` : ''}</span>}
      {model?.prompt_tokens != null && <span className="text-muted">{model.prompt_tokens.toLocaleString()} context</span>}
    </span>
    <span className="ml-auto flex gap-1.5 text-[10px]">
      {turn.exchanges.length > 0 && <span className="text-muted">{turn.exchanges.length} tool {turn.exchanges.length === 1 ? 'call' : 'calls'}</span>}
      {failures > 0 && <span className="rounded-full bg-rose-50 px-1.5 font-medium text-rose-700">{failures} failed</span>}
      {model?.finish_reason && <span className={`rounded-full px-1.5 ${truncated ? 'bg-amber-50 font-medium text-amber-800' : 'bg-subtle text-muted'}`} title="Why the model stopped generating">{truncated ? 'cut off (length)' : model.finish_reason}</span>}
    </span>
  </div>;
}

function Turn({ turn, timing, open }: { turn: RunTurn; timing: TurnTiming | undefined; open: boolean }) {
  const failures = turn.exchanges.filter((exchange) => summarizeResult(exchange.result).status !== 'ok').length + turn.unmatched.length;
  const reasoning = typeof turn.message.reasoning_content === 'string' ? turn.message.reasoning_content.trim() : '';
  const content = turn.message.content;
  const hasContent = typeof content === 'string' ? content.trim() !== '' : content != null;
  return <section
    id={`rollout-turn-${turn.index}`}
    data-turn-node={turn.node}
    className={`scroll-mt-28 rounded-md border px-3 py-2.5 ${turn.final ? 'border-violet-300 bg-violet-50/40' : failures ? 'border-rose-200' : 'border-divider'} bg-white`}
    aria-label={turn.final ? 'Final answer' : `Turn ${turn.index + 1}`}
  >
    <TurnHeader turn={turn} timing={timing} failures={failures} />
    <div className="mt-2 space-y-2">
      {reasoning && <Reasoning text={reasoning} />}
      {hasContent && <div className={turn.final ? 'text-[12px]' : ''}><MessageContent value={content} showRaw={false} /></div>}
      {turn.exchanges.length > 0 && <div className="space-y-1">{turn.exchanges.map((exchange, index) => <ExchangeRow key={`${exchange.id ?? index}-${index}-${String(open)}`} exchange={exchange} open={open} />)}</div>}
      {turn.unmatched.map((message, index) => <ExchangeRow key={`unmatched-${index}-${String(open)}`} open={open} exchange={{ id: null, name: String(message.name ?? 'unmatched tool result'), args: {}, result: message }} />)}
    </div>
  </section>;
}

/** A rollout read the way it ran: the prompt, then one row per model call with its tool calls and their results. */
export function RunTranscript({ messages, timeline }: { messages: JsonRecord[]; timeline?: TraceTimeline | null }) {
  const run = useMemo(() => parseRun(messages), [messages]);
  const timings = useMemo(() => turnTimings(timeline, run.turns), [timeline, run.turns]);
  const [open, setOpen] = useState(false);
  const [showSystem, setShowSystem] = useState(false);
  const system = run.preamble.filter((message) => role(message) === 'system');
  const prompt = run.preamble.filter((message) => role(message) !== 'system');
  const failedTurns = new Set(run.turns.filter((turn) => turn.unmatched.length || turn.exchanges.some((exchange) => summarizeResult(exchange.result).status !== 'ok')).map((turn) => turn.index));
  const calls = run.turns.reduce((acc, turn) => acc + turn.exchanges.length, 0);
  const scrollTo = (index: number) => document.getElementById(`rollout-turn-${index}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' });

  return <div className="mt-3 space-y-2" aria-label="transcript">
    {system.length > 0 && <div className="rounded-md border border-divider bg-subtle/60 px-3 py-2">
      <button type="button" className="flex w-full items-center gap-2 text-left text-[10px] font-medium text-secondary" aria-expanded={showSystem} onClick={() => setShowSystem(!showSystem)}>
        <CaretRight size={10} className={`transition-transform ${showSystem ? 'rotate-90' : ''}`} /> System prompt
        <span className="font-normal text-muted">{system.reduce((acc, message) => acc + String(message.content ?? '').length, 0).toLocaleString()} chars</span>
      </button>
      {showSystem && <div className="mt-2">{system.map((message, index) => <MessageContent key={index} value={message.content} />)}</div>}
    </div>}
    {prompt.map((message, index) => <TranscriptMessage key={index} message={message} />)}
    {run.turns.length > 0 && <nav className="sticky top-[66px] z-10 flex flex-wrap items-center gap-1 border-y border-divider bg-surface/95 py-1.5 backdrop-blur" aria-label="Turns">
      <span className="mr-1 text-[10px] text-muted">{run.turns.length} turns · {calls} tool calls{failedTurns.size ? ` · ${failedTurns.size} with failures` : ''}</span>
      {run.turns.map((turn) => <button
        key={turn.index}
        type="button"
        onClick={() => scrollTo(turn.index)}
        className={`inline-flex h-5 min-w-5 items-center justify-center gap-0.5 rounded px-1 font-mono text-[9px] tabular-nums ${turn.final ? 'bg-violet-700 text-white' : failedTurns.has(turn.index) ? 'bg-rose-100 text-rose-800' : 'bg-subtle text-secondary hover:bg-divider'}`}
        aria-label={turn.final ? 'Jump to final answer' : `Jump to turn ${turn.index + 1}`}
      >{turn.final ? <Flag size={10} weight="fill" /> : turn.index + 1}</button>)}
      <button type="button" className="ml-auto text-[10px] font-medium text-violet-700 hover:underline" onClick={() => setOpen(!open)}>{open ? 'Collapse tool calls' : 'Expand tool calls'}</button>
    </nav>}
    {run.turns.map((turn) => <div key={turn.index}>
      <Turn turn={turn} timing={timings.get(turn.index)} open={open} />
      {turn.followUps.map((message, index) => <div key={index} className="mt-2"><TranscriptMessage message={message} /></div>)}
    </div>)}
    {run.turns.length > 0 && !run.turns.at(-1)?.final && <p className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-[11px] text-amber-800">The rollout ended after a tool-calling turn without a final answer.</p>}
  </div>;
}
