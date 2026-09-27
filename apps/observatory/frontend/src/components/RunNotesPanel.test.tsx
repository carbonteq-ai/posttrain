import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('echarts/core', () => ({
  use: vi.fn(),
  init: vi.fn(() => ({
    setOption: vi.fn(),
    on: vi.fn(),
    dispatchAction: vi.fn(),
    getZr: vi.fn(() => ({ on: vi.fn() })),
    resize: vi.fn(),
    dispose: vi.fn(),
  })),
}));

import type { RenderedNote, RenderedRunNote, RunNote } from '../lib/api';
import { RunNotesPanel } from './RunNotesPanel';

const RUN_KEY = 'WyJsb2NhbCIsInJ1bi1hIl0';

function runNote(overrides: Partial<RunNote> = {}): RunNote {
  return {
    note_id: 'note-1',
    revision: 1,
    scope: 'run',
    run_id: 'run-a',
    kind: 'finding',
    title: 'Memory',
    body_md: 'The first update ran out of memory.',
    source: 'cli',
    created_at: '2026-09-27T10:00:00Z',
    revised_at: '2026-09-27T10:00:00Z',
    deleted: false,
    ...overrides,
  };
}

function rendered(markdown: string, template: string | null = null): RenderedNote {
  return { markdown, text: markdown, views: [], data: {}, unresolved: [], template };
}

function renderedRunNote(note: RunNote): RenderedRunNote {
  return { note, rendered: rendered(note.body_md) };
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
}

type Handler = (path: string, init?: RequestInit) => Response | undefined;

function stubApi({ writes, notes, handler }: { writes: boolean; notes: RenderedRunNote[] | Response; handler?: Handler }) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input);
    const custom = handler?.(path, init);
    if (custom) return custom;
    if (path === '/api/v1/notes/settings') return json({ writes });
    if (path === `/api/v1/runs/${RUN_KEY}/card`) return json(rendered('**grpo** · succeeded · 2h 3m', 'group-policy@1'));
    if (path === `/api/v1/runs/${RUN_KEY}/notes`) return notes instanceof Response ? notes : json(notes);
    throw new Error(`unexpected request ${init?.method ?? 'GET'} ${path}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

describe('RunNotesPanel', () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('shows the run card and notes newest first, without write controls when writes are off', async () => {
    stubApi({
      writes: false,
      notes: [
        renderedRunNote(runNote({ note_id: 'note-2', kind: 'correction', title: 'Served the base model', body_md: 'The evaluation served the base model.', source: 'mcp', revision: 2 })),
        renderedRunNote(runNote()),
      ],
    });
    render(<RunNotesPanel runKey={RUN_KEY} />);

    expect(await screen.findByText('group-policy@1')).toBeVisible();
    expect(screen.getByText('grpo')).toBeVisible();
    const articles = await screen.findAllByRole('article');
    expect(articles.map((article) => article.getAttribute('aria-label'))).toEqual(['Served the base model', 'Memory']);
    expect(within(articles[0]).getByText('correction')).toBeVisible();
    expect(within(articles[0]).getByText(/mcp ·/)).toHaveTextContent('revision 2');
    expect(within(articles[1]).getByText('The first update ran out of memory.')).toBeVisible();

    expect(within(articles[0]).getByRole('button', { name: 'History' })).toBeVisible();
    expect(screen.queryByRole('button', { name: 'Add note' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Edit' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Delete' })).toBeNull();
  });

  it('shows a quiet line when the source cannot store notes', async () => {
    stubApi({ writes: true, notes: json({ code: 'notes_unavailable', message: 'no note store', request_id: 'r' }, 501) });
    render(<RunNotesPanel runKey={RUN_KEY} />);

    expect(await screen.findByText('Notes are not available for this source.')).toBeVisible();
    expect(screen.queryByRole('alert')).toBeNull();
    expect(screen.queryByRole('button', { name: 'Add note' })).toBeNull();
    expect(await screen.findByText('group-policy@1')).toBeVisible();
  });

  it('lists every revision with its raw Markdown in the history view', async () => {
    stubApi({
      writes: false,
      notes: [renderedRunNote(runNote({ revision: 2, source: 'observatory', body_md: 'Second text' }))],
      handler: (path) => path === `/api/v1/runs/${RUN_KEY}/notes/note-1/history`
        ? json([runNote({ body_md: 'First **text**' }), runNote({ revision: 2, source: 'observatory', body_md: 'Second text' })])
        : undefined,
    });
    const user = userEvent.setup();
    render(<RunNotesPanel runKey={RUN_KEY} />);

    await user.click(await screen.findByRole('button', { name: 'History' }));
    const history = await screen.findByRole('list', { name: 'Note history' });
    const revisions = within(history).getAllByRole('listitem');
    expect(revisions).toHaveLength(2);
    expect(within(revisions[0]).getByText('Revision 1')).toBeVisible();
    expect(within(revisions[0]).getByText('First **text**')).toBeVisible();
    expect(within(revisions[1]).getByText('observatory')).toBeVisible();
  });

  it('adds a note with a live preview that lists unresolved references', async () => {
    const added: unknown[] = [];
    stubApi({
      writes: true,
      notes: [],
      handler: (path, init) => {
        if (path === '/api/v1/notes/preview') {
          const body = JSON.parse(String(init?.body)) as { run_key: string; body_md: string };
          expect(body.run_key).toBe(RUN_KEY);
          return json({ ...rendered(`Rendered: ${body.body_md} ⟦unresolved: {{run.nope}} — unknown field⟧`), unresolved: ['⟦unresolved: {{run.nope}} — unknown field⟧'] });
        }
        if (path === `/api/v1/runs/${RUN_KEY}/notes` && init?.method === 'POST') {
          added.push(JSON.parse(String(init.body)));
          return json(runNote({ source: 'observatory' }));
        }
        return undefined;
      },
    });
    const user = userEvent.setup();
    render(<RunNotesPanel runKey={RUN_KEY} />);

    await user.click(await screen.findByRole('button', { name: 'Add note' }));
    const form = screen.getByRole('form', { name: 'Add note' });
    await user.selectOptions(within(form).getByLabelText('Kind'), 'decision');
    await user.type(within(form).getByLabelText('Title (optional)'), 'Keep it');
    await user.type(within(form).getByLabelText('Markdown'), 'Uses {{{{run.nope}}');

    const preview = within(form).getByLabelText('Note preview');
    await waitFor(() => expect(within(preview).getByLabelText('Unresolved references')).toHaveTextContent('1 unresolved reference'));
    expect(within(preview).getByText(/Rendered: Uses/)).toBeVisible();

    await user.click(within(form).getByRole('button', { name: 'Save note' }));
    await waitFor(() => expect(added).toEqual([{ kind: 'decision', body_md: 'Uses {{run.nope}}', title: 'Keep it' }]));
    await waitFor(() => expect(screen.queryByRole('form', { name: 'Add note' })).toBeNull());
  });

  it('keeps the draft and offers to reload when an edit conflicts with a newer revision', async () => {
    const revisions: unknown[] = [];
    stubApi({
      writes: true,
      notes: [renderedRunNote(runNote({ revision: 1 }))],
      handler: (path, init) => {
        if (path === '/api/v1/notes/preview') return json(rendered('preview'));
        if (path === `/api/v1/runs/${RUN_KEY}/notes/note-1` && init?.method === 'PUT') {
          const body = JSON.parse(String(init.body)) as { expected_revision: number };
          revisions.push(body);
          return body.expected_revision === 3
            ? json(runNote({ revision: 4, source: 'observatory' }))
            : json({ code: 'note_conflict', message: 'stale', request_id: 'r', note_id: 'note-1', current_revision: 3 }, 409);
        }
        if (path === `/api/v1/runs/${RUN_KEY}/notes/note-1/history`) {
          return json([runNote(), runNote({ revision: 2 }), runNote({ revision: 3, source: 'mcp', body_md: 'Someone else changed it.' })]);
        }
        return undefined;
      },
    });
    const user = userEvent.setup();
    render(<RunNotesPanel runKey={RUN_KEY} />);

    await user.click(await screen.findByRole('button', { name: 'Edit' }));
    const form = screen.getByRole('form', { name: 'Edit note' });
    const textarea = within(form).getByLabelText('Markdown');
    expect(textarea).toHaveValue('The first update ran out of memory.');
    await user.clear(textarea);
    await user.type(textarea, 'My corrected text');
    await user.click(within(form).getByRole('button', { name: 'Save revision' }));

    const alert = await within(form).findByRole('alert');
    expect(alert).toHaveTextContent('This note changed while you were editing (it is now at revision 3). Your draft is kept.');
    expect(textarea).toHaveValue('My corrected text');
    expect(revisions).toEqual([{ expected_revision: 1, body_md: 'My corrected text', kind: 'finding', title: 'Memory' }]);

    await user.click(within(alert).getByRole('button', { name: 'Reload latest revision' }));
    expect(await within(form).findByText('Editing revision 3')).toBeVisible();
    expect(within(form).getByText('Someone else changed it.')).toBeInTheDocument();
    expect(textarea).toHaveValue('My corrected text');

    await user.click(within(form).getByRole('button', { name: 'Save revision' }));
    await waitFor(() => expect(screen.queryByRole('form', { name: 'Edit note' })).toBeNull());
    expect(revisions.at(-1)).toMatchObject({ expected_revision: 3, body_md: 'My corrected text' });
  });

  it('deletes a note with its expected revision after confirmation', async () => {
    const fetchMock = stubApi({
      writes: true,
      notes: [renderedRunNote(runNote({ revision: 5 }))],
      handler: (path, init) => init?.method === 'DELETE' ? json(runNote({ revision: 6, deleted: true })) : undefined,
    });
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true);
    const user = userEvent.setup();
    render(<RunNotesPanel runKey={RUN_KEY} />);

    await user.click(await screen.findByRole('button', { name: 'Delete' }));
    expect(confirm).toHaveBeenCalled();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      `/api/v1/runs/${RUN_KEY}/notes/note-1?expected_revision=5`,
      { method: 'DELETE' },
    ));
  });
});
