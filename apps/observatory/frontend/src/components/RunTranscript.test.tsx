import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import type { TraceTimeline } from '../lib/api';
import { parseRun, RunTranscript, summarizeResult } from './RunTranscript';

afterEach(cleanup);

const system = { role: 'system', content: 'You are a workflow automation agent.' };
const user = { role: 'user', content: 'Deduplicate the support conversations.' };

describe('RunTranscript', () => {
  it('shows one row per model call with each tool call beside its result, and pins the final answer', () => {
    render(<RunTranscript messages={[
      system,
      user,
      {
        role: 'assistant',
        content: null,
        reasoning_content: 'I should look up the account first.',
        tool_calls: [{ id: 'call-1', name: 'lookup_crm', arguments: '{"account":"Northwind"}' }],
      },
      { role: 'tool', name: 'lookup_crm', tool_call_id: 'call-1', content: '{"success":true,"result_count":1,"results":[{"owner":"Ada"}]}' },
      { role: 'assistant', content: 'Done. The owner is **Ada**.' },
    ]} />);

    const first = screen.getByRole('region', { name: 'Turn 1' });
    // Reasoning is visible without a click; the tool row is one line until expanded.
    expect(within(first).getByText('I should look up the account first.')).toBeVisible();
    const row = within(first).getByRole('group', { name: 'Tool lookup_crm' });
    expect(row).toHaveTextContent('lookup_crm');
    expect(row).toHaveTextContent('(account="Northwind")');
    expect(row).toHaveTextContent('→ 1 result');
    expect(within(first).getByLabelText('Succeeded')).toBeInTheDocument();

    const final = screen.getByRole('region', { name: 'Final answer' });
    expect(within(final).getByText('Ada').tagName).toBe('STRONG');
    expect(screen.getByRole('button', { name: 'Jump to final answer' })).toBeInTheDocument();
    // The system prompt stays folded; the user task is shown.
    expect(screen.queryByText('You are a workflow automation agent.')).not.toBeInTheDocument();
    expect(screen.getByText('Deduplicate the support conversations.')).toBeInTheDocument();
    expect(screen.queryByText(/without a final answer/)).not.toBeInTheDocument();
  });

  it('pairs results within their own turn because call ids restart every turn', () => {
    const run = parseRun([
      user,
      { role: 'assistant', tool_calls: [{ id: 'call_0', name: 'first' }, { id: 'call_1', name: 'second' }] },
      { role: 'tool', tool_call_id: 'call_1', content: 'B' },
      { role: 'tool', tool_call_id: 'call_0', content: 'A' },
      { role: 'assistant', tool_calls: [{ id: 'call_0', name: 'third' }] },
      { role: 'tool', tool_call_id: 'call_0', content: 'C' },
    ]);

    expect(run.turns).toHaveLength(2);
    expect(run.turns[0].exchanges.map((exchange) => [exchange.name, exchange.result?.content])).toEqual([['first', 'A'], ['second', 'B']]);
    expect(run.turns[1].exchanges[0].result?.content).toBe('C');
    expect(run.turns[1].final).toBe(false);
  });

  it('makes failed tool calls and a missing final answer impossible to miss', () => {
    render(<RunTranscript messages={[
      user,
      { role: 'assistant', tool_calls: [{ id: 'a', name: 'sheets_find', arguments: '{}' }, { id: 'b', name: 'slack_post', arguments: '{}' }] },
      { role: 'tool', tool_call_id: 'a', content: '{"success":false,"error":"Spreadsheet ss_contacts not found"}' },
    ]} />);

    const turn = screen.getByRole('region', { name: 'Turn 1' });
    expect(within(turn).getByText('2 failed')).toBeInTheDocument();
    expect(within(turn).getByRole('group', { name: 'Tool sheets_find' })).toHaveTextContent('→ Spreadsheet ss_contacts not found');
    expect(within(turn).getByRole('group', { name: 'Tool slack_post' })).toHaveTextContent('no result recorded');
    expect(screen.getByRole('button', { name: 'Jump to turn 1' })).toHaveClass('bg-rose-100');
    expect(screen.getByText(/without a final answer/)).toBeInTheDocument();
  });

  it('labels each turn with its model-call timing, tokens, and a cut-off finish', () => {
    const timeline: TraceTimeline = {
      total_ms: 9000, inference_ms: 7000, tools_ms: 1500, setup_ms: 500, scoring_ms: 0, model_calls: 2,
      segments: [
        { kind: 'inference', start_ms: 500, duration_ms: 4000, call_index: 0, node: 1, prompt_tokens: 900, completion_tokens: 300, thinking_tokens: 200, finish_reason: 'tool_calls', tools: [] },
        { kind: 'tools', start_ms: 4500, duration_ms: 1500, call_index: null, node: null, prompt_tokens: null, completion_tokens: null, thinking_tokens: null, finish_reason: null, tools: ['lookup'] },
        { kind: 'inference', start_ms: 6000, duration_ms: 3000, call_index: 1, node: 3, prompt_tokens: 1400, completion_tokens: 512, thinking_tokens: 512, finish_reason: 'length', tools: [] },
      ],
    };
    render(<RunTranscript timeline={timeline} messages={[
      user,
      { role: 'assistant', tool_calls: [{ id: 'x', name: 'lookup', arguments: '{}' }] },
      { role: 'tool', tool_call_id: 'x', content: '{"count":3}' },
      { role: 'assistant', content: 'Partial' },
    ]} />);

    const first = screen.getByRole('region', { name: 'Turn 1' });
    expect(first).toHaveTextContent('model 4.00s');
    expect(first).toHaveTextContent('tools 1.50s');
    expect(first).toHaveTextContent('300 out · 200 thinking');
    expect(first).toHaveTextContent('→ count 3');
    expect(screen.getByRole('region', { name: 'Final answer' })).toHaveTextContent('cut off (length)');
  });

  it('expands every tool call to its exact arguments and result text', () => {
    const { container } = render(<RunTranscript messages={[
      user,
      { role: 'assistant', tool_calls: [{ type: 'function', function: { name: 'lookup_record', arguments: '{"record_id":"r-17"}' } }] },
      { role: 'tool', name: 'lookup_record', content: 'No matching record.' },
      { role: 'assistant', content: 'There is **no match**.' },
    ]} />);

    expect(container.querySelector('details[open]')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Expand tool calls' }));
    const row = screen.getByRole('group', { name: 'Tool lookup_record' });
    expect(row).toHaveAttribute('open');
    expect(row.querySelectorAll('pre')[0]).toHaveTextContent('"record_id": "r-17"');
    expect(row.querySelectorAll('pre')[1]).toHaveTextContent('No matching record.');
    expect(screen.getByText('no match').tagName).toBe('STRONG');
  });

  it('normalizes nested legacy function calls and results without ids', () => {
    const run = parseRun([
      { message: { role: 'assistant', content: null, function_call: { name: 'weather', arguments: '{"city":"Lahore"}' } } },
      { message: { role: 'function', name: 'weather', content: 'Sunny with a high of **35 C**.' } },
      { message: { role: 'assistant', content: 'It is **sunny**.' } },
    ]);

    expect(run.turns[0].exchanges[0]).toMatchObject({ name: 'weather', args: { city: 'Lahore' } });
    expect(run.turns[0].exchanges[0].result?.content).toBe('Sunny with a high of **35 C**.');
    expect(run.turns[1].final).toBe(true);
  });

  it.each([
    ['success false', '{"success":false,"message":"quota exceeded"}', 'error', 'quota exceeded'],
    ['error object', '{"error":{"message":"bad id"}}', 'error', 'bad id'],
    ['plain error text', 'Error: tool timed out\nstack', 'error', 'Error: tool timed out'],
    ['first array', '{"success":true,"messages":[1,2,3]}', 'ok', '3 messages'],
    ['plain text', 'Sent to #support-dedup', 'ok', 'Sent to #support-dedup'],
  ] as const)('summarizes a %s result', (_label, content, status, text) => {
    expect(summarizeResult({ role: 'tool', content })).toMatchObject({ status, text });
  });
});
