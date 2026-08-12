import { BracketsCurly, Wrench } from '@phosphor-icons/react';

import { MessageContent, StructuredValue } from './ContentRenderer';

export { StructuredValue } from './ContentRenderer';

type JsonRecord = Record<string, unknown>;

function messageRole(message: JsonRecord): string {
  return String(normalizeMessage(message).role ?? 'event').toLowerCase();
}

function asRecord(value: unknown): JsonRecord | null {
  return value != null && typeof value === 'object' && !Array.isArray(value)
    ? value as JsonRecord
    : null;
}

function normalizeMessage(message: JsonRecord): JsonRecord {
  const nested = asRecord(message.message);
  return nested == null ? message : { ...nested, ...message };
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
        <div className="p-2.5"><MessageContent value={args} compact /></div>
      </section>;
    })}
  </div>;
}

function MessageBody({ message, nested = false }: { message: JsonRecord; nested?: boolean }) {
  const normalized = normalizeMessage(message);
  const role = messageRole(normalized);
  const isTool = role === 'tool' || role === 'function';
  const content = normalized.content;
  const toolCalls = normalized.tool_calls ?? normalized.function_call;
  const toolName = typeof normalized.name === 'string' ? normalized.name : null;
  const resultLabel = toolName ? `Tool result ${toolName}` : 'Tool result';
  return <div className={nested ? 'mt-2 border-t border-violet-100 pt-2' : 'mt-2'}>
    {typeof normalized.reasoning_content === 'string' && normalized.reasoning_content && <details className="rounded-md border border-violet-200 bg-white/70 px-2.5 py-2"><summary className="cursor-pointer text-[10px] font-medium text-violet-700">Thinking output</summary><div className="mt-2"><MessageContent value={normalized.reasoning_content} compact /></div></details>}
    {toolCalls != null && <ToolCalls value={toolCalls} />}
    {isTool && content != null ? <section className="rounded-md border border-emerald-200 bg-emerald-50/60 p-2.5" aria-label={resultLabel}>
      <div className="mb-1.5 flex items-center gap-1.5 text-[9px] font-medium uppercase tracking-[.08em] text-emerald-700"><BracketsCurly size={12} /> Tool result{toolName ? ` · ${toolName}` : ''}</div>
      <MessageContent value={content} compact />
    </section> : <MessageContent value={content} />}
  </div>;
}

export function TranscriptMessage({ message, relatedMessages = [] }: { message: JsonRecord; relatedMessages?: JsonRecord[] }) {
  const normalized = normalizeMessage(message);
  const role = messageRole(normalized);
  const isUser = role === 'user';
  const isAssistant = role === 'assistant';
  const isTool = role === 'tool' || role === 'function';
  return <article className={`flex ${isAssistant ? 'justify-end' : 'justify-start'}`} aria-label={`${role} message`}>
    <div className={`max-w-[94%] rounded-xl border px-3 py-2.5 ${isUser ? 'rounded-tl-sm border-divider bg-white' : isAssistant ? 'rounded-tr-sm border-violet-200 bg-violet-50/70' : isTool ? 'border-emerald-200 bg-emerald-50/60' : 'border-divider bg-subtle'}`}>
      <div className="flex items-center gap-2"><span className={`flex h-5 w-5 items-center justify-center rounded-full text-[9px] font-semibold uppercase ${isUser ? 'bg-ink text-white' : isAssistant ? 'bg-violet-700 text-white' : isTool ? 'bg-emerald-600 text-white' : 'bg-muted text-white'}`}>{role.slice(0, 1)}</span><span className="text-[10px] font-medium capitalize text-secondary">{role}</span></div>
      <div><MessageBody message={normalized} />
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
    const normalized = normalizeMessage(message);
    let hasToolExchange = normalized.tool_calls != null || normalized.function_call != null;
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
