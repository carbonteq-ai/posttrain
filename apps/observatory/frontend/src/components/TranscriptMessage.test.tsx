import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { classifyContent } from './ContentRenderer';
import { Transcript, TranscriptMessage } from './TranscriptMessage';

afterEach(cleanup);

describe('TranscriptMessage', () => {
  it('renders tool call arguments as structured fields', () => {
    render(<TranscriptMessage message={{
      role: 'assistant',
      tool_calls: [{
        id: 'call-1',
        name: 'salesforce_opportunity_update',
        arguments: '{"id":"006004","close_date":"2026-03-31"}',
      }],
    }} />);

    expect(screen.getByRole('region', { name: 'Tool call salesforce_opportunity_update' })).toBeInTheDocument();
    expect(screen.getByText('Id')).toBeInTheDocument();
    expect(screen.getByText('006004')).toBeInTheDocument();
    expect(screen.queryByText(/\\"id\\"/)).not.toBeInTheDocument();
  });

  it('keeps a tool exchange and continuation inside one assistant message', () => {
    render(<Transcript messages={[
      {
        role: 'assistant',
        tool_calls: [{ name: 'salesforce_opportunity_update', arguments: '{"id":"006004"}' }],
      },
      {
        role: 'tool',
        content: '{"success":true,"opportunity":{"Name":"Apex Security Suite","Amount":67000}}',
      },
      { role: 'assistant', content: 'Done. The opportunity was updated.' },
    ]} />);

    expect(screen.getByRole('region', { name: 'Tool result' })).toBeInTheDocument();
    expect(screen.getByText('Success')).toBeInTheDocument();
    expect(screen.getByText('Apex Security Suite')).toBeInTheDocument();

    const transcript = screen.getByLabelText('transcript');
    const messages = within(transcript).getAllByRole('article');
    expect(messages).toHaveLength(1);
    expect(messages[0]).toHaveAccessibleName('assistant message');
    expect(messages[0]).toHaveTextContent('Done. The opportunity was updated.');
    expect(within(messages[0]).getByRole('region', { name: 'Tool result' })).toBeInTheDocument();
  });

  it('renders a Gemma-style YAML system prompt as structured evidence with exact raw content', () => {
    const content = [
      'prompt_id: normative-scope-frames.v10',
      'stage: rules',
      'tangents:',
      '  - scope',
      'response:',
      '  format: json',
      '  api_schema_is_authoritative: true',
    ].join('\n');

    render(<TranscriptMessage message={{ role: 'system', content }} />);

    expect(screen.getByText('Prompt Id')).toBeInTheDocument();
    expect(screen.getByText('normative-scope-frames.v10')).toBeInTheDocument();
    expect(screen.getByText('Tangents')).toBeInTheDocument();
    expect(screen.getByText('scope')).toBeInTheDocument();
    expect(screen.getByText('Raw').parentElement?.querySelector('pre')).toBeNull();
    const rawDetails = screen.getByText('Raw').parentElement as HTMLDetailsElement;
    rawDetails.open = true;
    fireEvent(rawDetails, new Event('toggle'));
    const raw = screen.getByText('Raw').parentElement?.querySelector('pre');
    expect(raw).toHaveTextContent(content, { normalizeWhitespace: false });
  });

  it('renders assistant JSON as structured evidence instead of a serialized paragraph', () => {
    const { container } = render(<TranscriptMessage message={{
      role: 'assistant',
      content: '{"resolution":{"status":"resolved"},"rules":[{"rule_id":"r0001"}]}',
    }} />);

    expect(screen.getByText('Resolution')).toBeInTheDocument();
    expect(screen.getByText('Status')).toBeInTheDocument();
    expect(screen.getByText('resolved')).toBeInTheDocument();
    expect(screen.getByText('Rule Id')).toBeInTheDocument();
    expect(container.querySelector('.obs-markdown')).toBeNull();
  });

  it('renders safe GFM Markdown and highlighted fenced code while disabling raw HTML and unsafe links', () => {
    const { container } = render(<TranscriptMessage message={{
      role: 'assistant',
      content: [
        '# Result',
        '',
        '- **Passed**',
        '- `two`',
        '',
        '| Metric | Value |',
        '| --- | ---: |',
        '| reward | 1 |',
        '',
        '[unsafe](javascript:alert(1))',
        '[scheme relative](//example.test/unsafe)',
        '',
        '<script>globalThis.compromised = true</script>',
        '',
        '```js',
        'const answer = 42;',
        '```',
      ].join('\n'),
    }} />);

    expect(screen.getByRole('heading', { name: 'Result' })).toBeInTheDocument();
    expect(screen.getByText('Passed').tagName).toBe('STRONG');
    expect(screen.getByRole('table')).toBeInTheDocument();
    expect(screen.getByText('unsafe').closest('a')).not.toHaveAttribute('href');
    expect(screen.getByText('scheme relative').closest('a')).not.toHaveAttribute('href');
    expect(container.querySelector('script')).toBeNull();
    expect(container.querySelector('code.hljs.language-js')).toHaveTextContent('const answer = 42;');
  });

  it('renders a Qwen reasoning and tool exchange from direct Verifiers fields', () => {
    render(<Transcript messages={[
      {
        role: 'assistant',
        content: null,
        reasoning_content: 'I should **look up** the account first.',
        tool_calls: [{ id: 'call-1', name: 'lookup_crm', arguments: '{"account":"Northwind"}' }],
      },
      {
        role: 'tool',
        name: 'lookup_crm',
        tool_call_id: 'call-1',
        content: '{"results":[{"owner":"Ada"}]}',
      },
      { role: 'assistant', content: 'Done. The owner is **Ada**.' },
    ]} />);

    const exchange = screen.getByRole('article', { name: 'assistant message' });
    expect(within(exchange).getByText('look up').tagName).toBe('STRONG');
    expect(within(exchange).getByRole('region', { name: 'Tool call lookup_crm' })).toBeInTheDocument();
    expect(within(exchange).getByText('Northwind')).toBeInTheDocument();
    expect(within(exchange).getByRole('region', { name: 'Tool result lookup_crm' })).toBeInTheDocument();
    expect(within(exchange).getByText('Owner')).toBeInTheDocument();
    expect(within(exchange).getAllByText('Ada')).toHaveLength(2);
  });

  it('normalizes nested legacy function calls and nested tool metadata', () => {
    render(<Transcript messages={[
      {
        message: {
          role: 'assistant',
          content: null,
          function_call: { name: 'weather', arguments: '{"city":"Lahore"}' },
        },
      },
      {
        message: {
          role: 'function',
          name: 'weather',
          tool_call_id: 'legacy-1',
          content: 'Sunny with a high of **35 C**.',
        },
      },
      { message: { role: 'assistant', content: 'It is **sunny**.' } },
    ]} />);

    const exchange = screen.getByRole('article', { name: 'assistant message' });
    expect(within(exchange).getByRole('region', { name: 'Tool call weather' })).toBeInTheDocument();
    expect(within(exchange).getByText('Lahore')).toBeInTheDocument();
    expect(within(exchange).getByRole('region', { name: 'Tool result weather' })).toBeInTheDocument();
    expect(within(exchange).getByText('35 C').tagName).toBe('STRONG');
    expect(within(exchange).getByText('sunny').tagName).toBe('STRONG');
  });

  it('renders typed content parts individually and preserves unknown parts structurally', () => {
    render(<TranscriptMessage message={{
      role: 'user',
      content: [
        { type: 'text', text: 'Inspect **this** image.' },
        { type: 'image_url', image_url: { url: 'https://example.test/evidence.png', detail: 'low' } },
        { type: 'provider_state', token_count: 12 },
      ],
    }} />);

    expect(screen.getByText('this').tagName).toBe('STRONG');
    expect(screen.getByText('Image URL')).toBeInTheDocument();
    expect(screen.getByText('Provider State')).toBeInTheDocument();
    expect(screen.getByText('Token Count')).toBeInTheDocument();
  });

  it('classifies conservatively and never treats one-line prose as YAML', () => {
    expect(classifyContent('Done: successfully').kind).toBe('markdown');
    expect(classifyContent('{"status":"ok"}').kind).toBe('json');
    expect(classifyContent('key: value\nitems:\n  - one').kind).toBe('yaml');
    expect(classifyContent('{not valid json').kind).toBe('markdown');
    expect(classifyContent([{ type: 'text', text: 'hello' }]).kind).toBe('parts');
  });

  it.each([
    ['empty null', null, 'empty'],
    ['empty string', '', 'empty'],
    ['native object', { status: 'ok' }, 'structured'],
    ['native array', [1, 'two', false], 'structured'],
    ['native number', 42, 'structured'],
    ['JSON container', '[{"status":"ok"}]', 'json'],
    ['JSON scalar', '42', 'markdown'],
    ['ordinary prose with punctuation', 'Status: complete', 'markdown'],
    ['multiline prose', 'Title: Result\nThis remains ordinary prose.', 'markdown'],
    ['YAML scalar document', '---\nplain text', 'markdown'],
    ['malformed JSON', '{"status":', 'markdown'],
    ['malformed YAML-like text', 'first: [unterminated\nsecond: value', 'markdown'],
    ['typed provider parts', [{ type: 'output_text', text: 'done' }], 'parts'],
  ] as const)('classifies %s by retained shape', (_label, value, expected) => {
    expect(classifyContent(value).kind).toBe(expected);
  });

  it('supports nested OpenAI tool calls without model- or task-specific handling', () => {
    render(<Transcript messages={[
      {
        role: 'assistant',
        tool_calls: [{
          id: 'call-2',
          type: 'function',
          function: { name: 'lookup_record', arguments: '{"record_id":"r-17"}' },
        }],
      },
      { role: 'tool', name: 'lookup_record', content: 'No matching record.' },
      { role: 'assistant', content: 'There is **no match**.' },
    ]} />);

    const exchange = screen.getByRole('article', { name: 'assistant message' });
    expect(within(exchange).getByRole('region', { name: 'Tool call lookup_record' })).toBeInTheDocument();
    expect(within(exchange).getByText('r-17')).toBeInTheDocument();
    expect(within(exchange).getByRole('region', { name: 'Tool result lookup_record' })).toBeInTheDocument();
    expect(within(exchange).getByText('no match').tagName).toBe('STRONG');
  });

  it('bounds large structured values while retaining the complete raw value', () => {
    const value = Object.fromEntries(Array.from({ length: 105 }, (_, index) => [`field_${index}`, index]));
    render(<TranscriptMessage message={{ role: 'provider_event', content: value }} />);

    expect(screen.getByRole('article', { name: 'provider_event message' })).toBeInTheDocument();
    expect(screen.getByText('5 more fields in Raw')).toBeInTheDocument();
    expect(screen.getByText('Raw').parentElement?.querySelector('pre')).toBeNull();
    const rawDetails = screen.getByText('Raw').parentElement as HTMLDetailsElement;
    rawDetails.open = true;
    fireEvent(rawDetails, new Event('toggle'));
    expect(screen.getByText('Raw').parentElement?.querySelector('pre')).toHaveTextContent('"field_104": 104');
  });

  it('uses a bounded plain-text path for oversized retained strings', () => {
    const value = `# heading\n${'x'.repeat(250_000)}`;
    const { container } = render(<TranscriptMessage message={{ role: 'assistant', content: value }} />);

    expect(classifyContent(value).kind).toBe('text');
    expect(container.querySelector('.obs-markdown')).toBeNull();
    expect(screen.getByText('Raw').parentElement?.querySelector('pre')).toBeNull();
  });
});
