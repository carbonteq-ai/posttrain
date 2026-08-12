import { useState, type ComponentPropsWithoutRef, type ReactNode } from 'react';
import ReactMarkdown, { defaultUrlTransform } from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import remarkBreaks from 'remark-breaks';
import remarkGfm from 'remark-gfm';
import { isMap, isSeq, parseDocument } from 'yaml';

type JsonRecord = Record<string, unknown>;

export type ContentKind = 'empty' | 'text' | 'markdown' | 'json' | 'yaml' | 'structured' | 'parts';

export type ClassifiedContent = {
  kind: ContentKind;
  parsed: unknown;
  raw: string;
};

const MAX_STRUCTURED_ENTRIES = 100;
const MAX_STRUCTURED_DEPTH = 8;
const MAX_PARSE_CHARS = 250_000;

function asRecord(value: unknown): JsonRecord | null {
  return value != null && typeof value === 'object' && !Array.isArray(value)
    ? value as JsonRecord
    : null;
}

function isContentParts(value: unknown): value is JsonRecord[] {
  return Array.isArray(value)
    && value.length > 0
    && value.every((item) => {
      const record = asRecord(item);
      return record != null && typeof record.type === 'string';
    });
}

function rawValue(value: unknown): string {
  if (typeof value === 'string') return value;
  if (value == null) return '';
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function parseJsonContainer(source: string): unknown | null {
  const trimmed = source.trim();
  if (trimmed.length > MAX_PARSE_CHARS) return null;
  if (!(trimmed.startsWith('{') || trimmed.startsWith('['))) return null;
  try {
    const parsed: unknown = JSON.parse(trimmed);
    return Array.isArray(parsed) || asRecord(parsed) != null ? parsed : null;
  } catch {
    return null;
  }
}

function looksLikeYamlCollection(source: string): boolean {
  if (source.length > MAX_PARSE_CHARS || !source.includes('\n')) return false;
  const lines = source.split(/\r?\n/);
  const mappingLines = lines.filter((line) => /^\s{0,12}[A-Za-z_][\w.-]*:\s*(?:.*)?$/.test(line)).length;
  const sequenceLines = lines.filter((line) => /^\s+-\s+\S/.test(line)).length;
  const explicitDocument = lines.some((line) => /^\s*---\s*$/.test(line));
  return explicitDocument || mappingLines >= 2 || (mappingLines >= 1 && sequenceLines >= 1);
}

function parseYamlCollection(source: string): unknown | null {
  if (!looksLikeYamlCollection(source)) return null;
  const document = parseDocument(source, {
    logLevel: 'silent',
    strict: true,
    uniqueKeys: true,
  });
  if (document.errors.length > 0 || (!isMap(document.contents) && !isSeq(document.contents))) return null;
  try {
    return document.toJS({ maxAliasCount: 20 });
  } catch {
    return null;
  }
}

export function classifyContent(value: unknown): ClassifiedContent {
  const raw = rawValue(value);
  if (value == null || raw.length === 0) return { kind: 'empty', parsed: null, raw };
  if (isContentParts(value)) return { kind: 'parts', parsed: value, raw };
  if (Array.isArray(value) || asRecord(value) != null) return { kind: 'structured', parsed: value, raw };
  if (typeof value !== 'string') return { kind: 'structured', parsed: value, raw };
  if (value.length > MAX_PARSE_CHARS) return { kind: 'text', parsed: value, raw };

  const json = parseJsonContainer(value);
  if (json != null) return { kind: 'json', parsed: json, raw };
  const yaml = parseYamlCollection(value);
  if (yaml != null) return { kind: 'yaml', parsed: yaml, raw };
  return { kind: 'markdown', parsed: value, raw };
}

function fieldLabel(key: string): string {
  return key.replaceAll('_', ' ').replace(/\b\w/g, (character) => character.toUpperCase());
}

function contentPartLabel(type: string): string {
  if (type === 'image_url') return 'Image URL';
  return fieldLabel(type);
}

function PrimitiveValue({ value }: { value: unknown }) {
  if (value == null) return <span className="font-mono text-muted">null</span>;
  if (typeof value === 'boolean') return <span className={`font-mono font-medium ${value ? 'text-emerald-700' : 'text-rose-600'}`}>{String(value)}</span>;
  if (typeof value === 'number') return <span className="font-mono tabular-nums text-ink">{value.toLocaleString()}</span>;
  return <span className="break-words text-secondary">{String(value)}</span>;
}

function maybeParsedJson(value: unknown): unknown {
  return typeof value === 'string' ? parseJsonContainer(value) ?? value : value;
}

export function StructuredValue({ value, depth = 0 }: { value: unknown; depth?: number }) {
  const parsed = maybeParsedJson(value);
  if (depth >= MAX_STRUCTURED_DEPTH) {
    return <span className="text-[10px] text-muted">Nested value available in Raw</span>;
  }
  if (Array.isArray(parsed)) {
    const visible = parsed.slice(0, MAX_STRUCTURED_ENTRIES);
    return <div className="space-y-1.5">
      {visible.map((item, index) => <div key={index} className="grid grid-cols-[20px_minmax(0,1fr)] gap-2">
        <span className="pt-0.5 text-right font-mono text-[9px] text-muted">{index + 1}</span>
        <StructuredValue value={item} depth={depth + 1} />
      </div>)}
      {parsed.length > visible.length && <p className="text-[10px] text-muted">{(parsed.length - visible.length).toLocaleString()} more items in Raw</p>}
    </div>;
  }
  const record = asRecord(parsed);
  if (record) {
    const entries = Object.entries(record);
    const visible = entries.slice(0, MAX_STRUCTURED_ENTRIES);
    return <dl className={`overflow-hidden rounded-[4px] border border-divider bg-white ${depth ? 'mt-1' : ''}`}>
      {visible.map(([key, item]) => {
        const nestedValue = maybeParsedJson(item);
        const nested = Array.isArray(nestedValue) || asRecord(nestedValue) != null;
        return <div key={key} className={`border-b border-divider px-2.5 py-2 last:border-b-0 ${nested ? 'block' : 'grid grid-cols-[minmax(92px,.8fr)_minmax(0,1.2fr)] gap-3'}`}>
          <dt className="text-[9px] font-medium text-muted">{fieldLabel(key)}</dt>
          <dd className={nested ? 'mt-1.5' : 'min-w-0 text-right text-[10px]'}><StructuredValue value={item} depth={depth + 1} /></dd>
        </div>;
      })}
      {entries.length > visible.length && <div className="px-2.5 py-2 text-[10px] text-muted">{(entries.length - visible.length).toLocaleString()} more fields in Raw</div>}
    </dl>;
  }
  return <PrimitiveValue value={parsed} />;
}

function safeUrlTransform(url: string, key: string): string | undefined {
  if (key === 'src') return undefined;
  const localReference = url.startsWith('#')
    || (url.startsWith('/') && !url.startsWith('//'))
    || url.startsWith('./')
    || url.startsWith('../');
  if (/^(?:https?:|mailto:)/i.test(url) || localReference) {
    return defaultUrlTransform(url);
  }
  return undefined;
}

function MarkdownLink({ href, children, ...props }: ComponentPropsWithoutRef<'a'>) {
  const external = typeof href === 'string' && /^https?:/i.test(href);
  return <a
    {...props}
    href={href}
    className="font-medium text-violet-700 underline decoration-violet-300 underline-offset-2 hover:text-violet-900"
    {...(external ? { target: '_blank', rel: 'noreferrer noopener' } : {})}
  >{children}</a>;
}

function MarkdownContent({ children, compact = false }: { children: string; compact?: boolean }) {
  return <div className={`obs-markdown ${compact ? 'obs-markdown-compact' : ''}`}>
    <ReactMarkdown
      skipHtml
      remarkPlugins={[remarkGfm, remarkBreaks]}
      rehypePlugins={[[rehypeHighlight, { detect: false, plainText: ['text', 'txt'] }]]}
      urlTransform={safeUrlTransform}
      components={{
        a: ({ node: _node, ...props }) => <MarkdownLink {...props} />,
        h1: ({ node: _node, ...props }) => <h4 {...props} />,
        h2: ({ node: _node, ...props }) => <h5 {...props} />,
        h3: ({ node: _node, ...props }) => <h6 {...props} />,
        img: ({ node: _node, alt }) => <span className="rounded bg-subtle px-1.5 py-0.5 text-[10px] text-muted">Image omitted{alt ? `: ${alt}` : ''}</span>,
      }}
    >{children}</ReactMarkdown>
  </div>;
}

function withoutType(record: JsonRecord): JsonRecord {
  const { type: _type, ...rest } = record;
  return rest;
}

function ContentParts({ parts, compact }: { parts: JsonRecord[]; compact: boolean }) {
  return <div className="space-y-2" aria-label="Message content parts">
    {parts.map((part, index) => {
      const type = String(part.type);
      if (type === 'text' || type === 'input_text' || type === 'output_text') {
        const text = typeof part.text === 'string' ? part.text : typeof part.content === 'string' ? part.content : '';
        return <section key={index} aria-label={`${contentPartLabel(type)} content part`}><MarkdownContent compact={compact}>{text}</MarkdownContent></section>;
      }
      if (type === 'image_url' || type === 'input_image' || type === 'output_image') {
        return <section key={index} className="rounded-md border border-divider bg-white/70 p-2.5" aria-label={`${contentPartLabel(type)} content part`}>
          <p className="mb-1.5 text-[9px] font-medium uppercase tracking-[.08em] text-muted">{contentPartLabel(type)}</p>
          <StructuredValue value={withoutType(part)} />
        </section>;
      }
      return <section key={index} className="rounded-md border border-divider bg-white/70 p-2.5" aria-label={`${contentPartLabel(type)} content part`}>
        <p className="mb-1.5 text-[9px] font-medium uppercase tracking-[.08em] text-muted">{contentPartLabel(type)}</p>
        <StructuredValue value={withoutType(part)} />
      </section>;
    })}
  </div>;
}

function RawDisclosure({ raw }: { raw: string }) {
  const [open, setOpen] = useState(false);
  return <details className="mt-2 border-t border-divider/70 pt-1.5" onToggle={(event) => setOpen(event.currentTarget.open)}>
    <summary className="cursor-pointer select-none text-[9px] font-medium uppercase tracking-[.08em] text-muted">Raw</summary>
    {open && <pre className="mt-1.5 max-h-72 overflow-auto whitespace-pre-wrap break-words rounded bg-ink/[.035] p-2 font-mono text-[9px] leading-4 text-secondary">{raw}</pre>}
  </details>;
}

export function MessageContent({ value, compact = false, showRaw = true }: { value: unknown; compact?: boolean; showRaw?: boolean }) {
  const classified = classifyContent(value);
  if (classified.kind === 'empty') return null;

  let rendered: ReactNode;
  if (classified.kind === 'text') {
    rendered = <p className="whitespace-pre-wrap break-words text-xs leading-5 text-secondary">{String(classified.parsed)}</p>;
  } else if (classified.kind === 'markdown') {
    rendered = <MarkdownContent compact={compact}>{String(classified.parsed)}</MarkdownContent>;
  } else if (classified.kind === 'parts') {
    rendered = <ContentParts parts={classified.parsed as JsonRecord[]} compact={compact} />;
  } else {
    rendered = <StructuredValue value={classified.parsed} />;
  }

  return <div>{rendered}{showRaw && <RawDisclosure raw={classified.raw} />}</div>;
}
