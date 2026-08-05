import { render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { Transcript, TranscriptMessage } from './TranscriptMessage';

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
});
