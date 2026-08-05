import { BracketsCurly, Wrench } from '@phosphor-icons/react';

type JsonRecord = Record<string, unknown>;

function messageRole(message: JsonRecord): string {
  return String(message.role ?? 'event').toLowerCase();
}

function asRecord(value: unknown): JsonRecord | null {
  return value != null && typeof value === 'object' && !Array.isArray(value)
    ? value as JsonRecord
    : null;
}

function parseStructured(value: unknown): unknown {
  if (typeof value !== 'string') return value;
  const source = value.trim();
  if (!(source.startsWith('{') || source.startsWith('['))) return value;
  try {
    return JSON.parse(source) as unknown;
  } catch {
    return value;
  }
}

function fieldLabel(key: string): string {
  return key.replaceAll('_', ' ').replace(/\b\w/g, (character) => character.toUpperCase());
}

function PrimitiveValue({ value }: { value: unknown }) {
  if (value == null) return <span className="font-mono text-muted">null</span>;
  if (typeof value === 'boolean') return <span className={`font-mono font-medium ${value ? 'text-emerald-700' : 'text-rose-600'}`}>{String(value)}</span>;
  if (typeof value === 'number') return <span className="font-mono tabular-nums text-ink">{value.toLocaleString()}</span>;
  return <span className="break-words text-secondary">{String(value)}</span>;
}

export function StructuredValue({ value, depth = 0 }: { value: unknown; depth?: number }) {
  const parsed = parseStructured(value);
  if (Array.isArray(parsed)) {
    return <div className="space-y-1.5">
      {parsed.map((item, index) => <div key={index} className="grid grid-cols-[20px_minmax(0,1fr)] gap-2">
        <span className="pt-0.5 text-right font-mono text-[9px] text-muted">{index + 1}</span>
        <StructuredValue value={item} depth={depth + 1} />
      </div>)}
    </div>;
  }
  const record = asRecord(parsed);
  if (record) {
    return <dl className={`overflow-hidden rounded-[4px] border border-divider bg-white ${depth ? 'mt-1' : ''}`}>
      {Object.entries(record).map(([key, item]) => {
        const nested = Array.isArray(parseStructured(item)) || asRecord(parseStructured(item)) != null;
        return <div key={key} className={`border-b border-divider px-2.5 py-2 last:border-b-0 ${nested ? 'block' : 'grid grid-cols-[minmax(92px,.8fr)_minmax(0,1.2fr)] gap-3'}`}>
          <dt className="text-[9px] font-medium text-muted">{fieldLabel(key)}</dt>
          <dd className={nested ? 'mt-1.5' : 'min-w-0 text-right text-[10px]'}><StructuredValue value={item} depth={depth + 1} /></dd>
        </div>;
      })}
    </dl>;
  }
  return <PrimitiveValue value={parsed} />;
}

function ToolCalls({ value }: { value: unknown }) {
  const calls = Array.isArray(value) ? value : value == null ? [] : [value];
  return <div className="space-y-2">
    {calls.map((candidate, index) => {
      const call = asRecord(candidate) ?? {};
      const fn = asRecord(call.function);
      const name = String(call.name ?? fn?.name ?? `Tool ${index + 1}`);
      const args = call.arguments ?? fn?.arguments ?? {};
      return <section key={String(call.id ?? index)} className="overflow-hidden rounded-md border border-violet-200 bg-white/90" aria-label={`Tool call ${name}`}>
        <div className="flex items-center gap-2 border-b border-violet-100 bg-violet-50/70 px-2.5 py-2">
          <Wrench size={13} className="text-violet-700" />
          <span className="text-[9px] font-medium uppercase tracking-[.08em] text-violet-700">Tool call</span>
          <code className="ml-auto truncate text-[10px] text-ink" title={name}>{name}</code>
        </div>
        <div className="p-2.5"><StructuredValue value={args} /></div>
      </section>;
    })}
  </div>;
}

function MessageBody({ message, nested = false }: { message: JsonRecord; nested?: boolean }) {
  const role = messageRole(message);
  const isTool = role === 'tool' || role === 'function';
  const content = typeof message.content === 'string'
    ? message.content
    : message.content == null ? null : JSON.stringify(message.content);
  const structuredResult = isTool && content != null ? parseStructured(content) : null;
  const toolCalls = message.tool_calls;
  return <div className={nested ? 'mt-2 border-t border-violet-100 pt-2' : 'mt-2'}>
    {typeof message.reasoning_content === 'string' && message.reasoning_content && <details className="rounded-md border border-violet-200 bg-white/70 px-2.5 py-2"><summary className="cursor-pointer text-[10px] font-medium text-violet-700">Thinking output</summary><p className="mt-2 whitespace-pre-wrap text-[11px] leading-5 text-secondary">{message.reasoning_content}</p></details>}
    {toolCalls != null && <ToolCalls value={toolCalls} />}
    {isTool && structuredResult != null ? <section className="rounded-md border border-emerald-200 bg-emerald-50/60 p-2.5" aria-label="Tool result">
      <div className="mb-1.5 flex items-center gap-1.5 text-[9px] font-medium uppercase tracking-[.08em] text-emerald-700"><BracketsCurly size={12} /> Tool result</div>
      <StructuredValue value={structuredResult} />
    </section> : content && <p className="whitespace-pre-wrap text-xs leading-5 text-secondary">{content}</p>}
  </div>;
}

export function TranscriptMessage({ message, relatedMessages = [] }: { message: JsonRecord; relatedMessages?: JsonRecord[] }) {
  const role = messageRole(message);
  const isUser = role === 'user';
  const isAssistant = role === 'assistant';
  const isTool = role === 'tool' || role === 'function';
  return <article className={`flex ${isAssistant ? 'justify-end' : 'justify-start'}`} aria-label={`${role} message`}>
    <div className={`max-w-[94%] rounded-xl border px-3 py-2.5 ${isUser ? 'rounded-tl-sm border-divider bg-white' : isAssistant ? 'rounded-tr-sm border-violet-200 bg-violet-50/70' : isTool ? 'border-emerald-200 bg-emerald-50/60' : 'border-divider bg-subtle'}`}>
      <div className="flex items-center gap-2"><span className={`flex h-5 w-5 items-center justify-center rounded-full text-[9px] font-semibold uppercase ${isUser ? 'bg-ink text-white' : isAssistant ? 'bg-violet-700 text-white' : isTool ? 'bg-emerald-600 text-white' : 'bg-muted text-white'}`}>{role.slice(0, 1)}</span><span className="text-[10px] font-medium capitalize text-secondary">{role}</span></div>
      <div><MessageBody message={message} />
        {relatedMessages.map((related, index) => <MessageBody key={index} message={related} nested />)}
      </div>
    </div>
  </article>;
}

export function Transcript({ messages }: { messages: JsonRecord[] }) {
  const groups: Array<{ message: JsonRecord; relatedMessages: JsonRecord[] }> = [];
  for (let index = 0; index < messages.length;) {
    const message = messages[index];
    if (messageRole(message) !== 'assistant') {
      groups.push({ message, relatedMessages: [] });
      index += 1;
      continue;
    }
    const relatedMessages: JsonRecord[] = [];
    let cursor = index + 1;
    let hasToolExchange = message.tool_calls != null;
    while (cursor < messages.length) {
      const next = messages[cursor];
      const nextRole = messageRole(next);
      if (nextRole === 'tool' || nextRole === 'function') {
        relatedMessages.push(next);
        hasToolExchange = true;
        cursor += 1;
        continue;
      }
      if (nextRole === 'assistant' && hasToolExchange) {
        relatedMessages.push(next);
        cursor += 1;
        continue;
      }
      break;
    }
    groups.push({ message, relatedMessages });
    index = cursor;
  }
  return <div className="mt-3 space-y-3" aria-label="transcript">
    {groups.map((group, index) => <TranscriptMessage key={index} {...group} />)}
  </div>;
}
