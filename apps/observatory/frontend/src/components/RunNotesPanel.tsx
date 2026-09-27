import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react';
import { CaretRight, ClockCounterClockwise, PencilSimple, Plus, Trash } from '@phosphor-icons/react';

import { api, ApiError, type RenderedNote, type RenderedRunNote, type RunNote } from '../lib/api';
import { NoteMarkdown } from './NoteViews';

export const NOTE_KINDS = ['summary', 'finding', 'correction', 'decision'] as const;
const OTHER_KIND = 'other';
const PREVIEW_DELAY_MS = 400;

type Loadable<T> =
  | { state: 'loading' }
  | { state: 'ready'; value: T }
  | { state: 'unavailable' }
  | { state: 'error'; message: string };

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function isRenderedNote(value: unknown): value is RenderedNote {
  return isRecord(value) && typeof value.markdown === 'string' && (value.views == null || Array.isArray(value.views));
}

function isRunNote(value: unknown): value is RunNote {
  return isRecord(value) && typeof value.note_id === 'string' && typeof value.body_md === 'string' && typeof value.revision === 'number';
}

function isRenderedRunNotes(value: unknown): value is RenderedRunNote[] {
  return Array.isArray(value) && value.every((item) => isRecord(item) && isRunNote(item.note) && isRenderedNote(item.rendered));
}

function errorMessage(cause: unknown): string {
  if (cause instanceof ApiError && cause.code === 'notes_read_only') return 'Writing notes is turned off for this Observatory.';
  return cause instanceof Error ? cause.message : String(cause);
}

function conflictRevision(cause: unknown): number | null {
  if (!(cause instanceof ApiError) || cause.code !== 'note_conflict') return null;
  const revision = Number(cause.body.current_revision);
  return Number.isFinite(revision) ? revision : NaN;
}

export function formatAbsoluteTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date);
}

export function formatRelativeTime(value: string, now = Date.now()): string {
  const time = Date.parse(value);
  if (!Number.isFinite(time)) return value;
  const seconds = Math.round((now - time) / 1000);
  if (seconds < 45) return 'just now';
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.round(hours / 24);
  if (days < 30) return `${days} d ago`;
  return formatAbsoluteTime(value);
}

function NoteTime({ value }: { value: string }) {
  return <time dateTime={value} title={formatAbsoluteTime(value)}>{formatRelativeTime(value)}</time>;
}

function KindBadge({ kind }: { kind: string }) {
  const tone = kind === 'correction'
    ? 'bg-amber-50 text-amber-800 border-amber-200'
    : kind === 'decision'
      ? 'bg-teal-50 text-teal-800 border-teal-200'
      : 'bg-violet-50 text-violet-800 border-violet-200';
  return <span className={`rounded-[3px] border px-1.5 py-px text-[10px] font-medium ${tone}`}>{kind}</span>;
}

const buttonClass = 'obs-control inline-flex items-center gap-1 px-2 py-1 text-[11px] hover:border-violet-300 hover:text-ink disabled:cursor-not-allowed disabled:opacity-50';
const primaryButtonClass = 'inline-flex items-center gap-1 rounded-[5px] bg-violet-700 px-3 py-1.5 text-[11px] font-medium text-white hover:bg-violet-800 disabled:cursor-not-allowed disabled:opacity-50';

function UnresolvedList({ markers }: { markers: string[] }) {
  if (!markers.length) return null;
  return <div className="mb-2 rounded-[4px] border border-amber-300 bg-amber-50 px-3 py-2 text-[11px] text-amber-900" aria-label="Unresolved references">
    <p className="font-medium">{markers.length} unresolved reference{markers.length === 1 ? '' : 's'}</p>
    <ul className="mt-1 list-disc space-y-0.5 pl-4">{markers.map((marker, index) => <li key={index} className="break-words">{marker}</li>)}</ul>
  </div>;
}

type EditorProps = {
  runKey: string;
  note?: RunNote;
  onCancel: () => void;
  onSaved: (note: RunNote) => void;
};

/** Add or revise a note, with a debounced server-rendered preview. */
export function NoteEditor({ runKey, note, onCancel, onSaved }: EditorProps) {
  const knownKind = note == null || (NOTE_KINDS as readonly string[]).includes(note.kind);
  const [kindChoice, setKindChoice] = useState<string>(note == null ? 'finding' : knownKind ? note.kind : OTHER_KIND);
  const [otherKind, setOtherKind] = useState(knownKind ? '' : note?.kind ?? '');
  const [title, setTitle] = useState(note?.title ?? '');
  const [body, setBody] = useState(note?.body_md ?? '');
  const [base, setBase] = useState<RunNote | undefined>(note);
  const [preview, setPreview] = useState<Loadable<RenderedNote> | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [conflict, setConflict] = useState<number | null>(null);
  const [latest, setLatest] = useState<RunNote | null>(null);

  useEffect(() => {
    if (!body.trim()) {
      setPreview(null);
      return;
    }
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      setPreview((current) => current?.state === 'ready' ? current : { state: 'loading' });
      api.previewNote(runKey, body, controller.signal)
        .then((rendered) => {
          if (controller.signal.aborted) return;
          setPreview(isRenderedNote(rendered) ? { state: 'ready', value: rendered } : { state: 'error', message: 'The preview response was not a rendered note.' });
        })
        .catch((cause: unknown) => {
          if (!controller.signal.aborted) setPreview({ state: 'error', message: errorMessage(cause) });
        });
    }, PREVIEW_DELAY_MS);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [body, runKey]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const kind = kindChoice === OTHER_KIND ? otherKind.trim() : kindChoice;
    if (!kind) {
      setError('Name the kind of note.');
      return;
    }
    if (!body.trim()) {
      setError('Write the note before saving it.');
      return;
    }
    setSaving(true);
    setError(null);
    setConflict(null);
    try {
      const saved = base
        ? await api.reviseRunNote(runKey, base.note_id, { expected_revision: base.revision, body_md: body, kind, title: title.trim() || null })
        : await api.addRunNote(runKey, { kind, body_md: body, title: title.trim() || null });
      onSaved(saved);
    } catch (cause) {
      const revision = conflictRevision(cause);
      if (revision != null) setConflict(revision);
      else setError(errorMessage(cause));
    } finally {
      setSaving(false);
    }
  };

  const reloadLatest = async () => {
    if (!base) return;
    setError(null);
    try {
      const history = await api.runNoteHistory(runKey, base.note_id);
      const newest = Array.isArray(history) ? history.at(-1) : undefined;
      if (!newest || !isRunNote(newest)) throw new Error('The note history is empty.');
      setBase(newest);
      setLatest(newest);
      setConflict(null);
      if (newest.deleted) setError('This note has been deleted since you started editing. Saving will restore it with your draft.');
    } catch (cause) {
      setError(errorMessage(cause));
    }
  };

  return <form onSubmit={(event) => void submit(event)} className="border-b border-divider bg-subtle/60 px-4 py-3" aria-label={base ? 'Edit note' : 'Add note'}>
    <div className="flex flex-wrap items-end gap-2">
      <label className="text-[10px] text-muted">
        <span className="type-label block">Kind</span>
        <select aria-label="Kind" value={kindChoice} onChange={(event) => setKindChoice(event.target.value)} className="obs-control mt-1 h-7 px-2">
          {NOTE_KINDS.map((kind) => <option key={kind} value={kind}>{kind}</option>)}
          <option value={OTHER_KIND}>other…</option>
        </select>
      </label>
      {kindChoice === OTHER_KIND && <label className="text-[10px] text-muted">
        <span className="type-label block">Other kind</span>
        <input value={otherKind} onChange={(event) => setOtherKind(event.target.value)} placeholder="for example: incident" className="obs-control mt-1 h-7 w-40 px-2" />
      </label>}
      <label className="min-w-[200px] flex-1 text-[10px] text-muted">
        <span className="type-label block">Title (optional)</span>
        <input value={title} onChange={(event) => setTitle(event.target.value)} className="obs-control mt-1 h-7 w-full px-2" />
      </label>
    </div>
    <div className="mt-3 grid gap-3 lg:grid-cols-2">
      <label className="block text-[10px] text-muted">
        <span className="type-label block">Markdown</span>
        <textarea
          aria-label="Markdown"
          value={body}
          onChange={(event) => setBody(event.target.value)}
          rows={12}
          spellCheck
          placeholder={'What happened in this run, and why it matters.\nReference recorded values with {{run.learning_rate}}.'}
          className="obs-control mt-1 block w-full resize-y px-2.5 py-2 font-mono text-[11px] leading-5 text-ink"
        />
      </label>
      <div className="min-w-0">
        <span className="type-label block">Preview</span>
        <div className="mt-1 min-h-[120px] rounded-[5px] border border-divider bg-surface px-3 py-2.5" aria-label="Note preview" aria-live="polite">
          {preview == null ? <p className="text-[11px] text-muted">The rendered note appears here as you write.</p>
            : preview.state === 'loading' ? <p className="text-[11px] text-muted">Rendering…</p>
            : preview.state === 'error' ? <p className="text-[11px] text-rose-700">Preview failed: {preview.message}</p>
            : preview.state === 'ready' ? <>
              <UnresolvedList markers={preview.value.unresolved ?? []} />
              <NoteMarkdown rendered={preview.value} />
            </> : null}
        </div>
      </div>
    </div>
    {conflict != null && <div role="alert" className="mt-3 flex flex-wrap items-center gap-2 rounded-[4px] border border-amber-300 bg-amber-50 px-3 py-2 text-[11px] text-amber-900">
      <span>This note changed while you were editing{Number.isFinite(conflict) ? ` (it is now at revision ${conflict})` : ''}. Your draft is kept.</span>
      <button type="button" className={buttonClass} onClick={() => void reloadLatest()}>Reload latest revision</button>
    </div>}
    {latest && <details className="mt-3 rounded-[4px] border border-divider bg-surface px-3 py-2 text-[11px]">
      <summary className="cursor-pointer select-none text-secondary">Latest saved text (revision {latest.revision}, {latest.source}); saving now revises it with your draft</summary>
      <pre className="mt-2 max-h-60 overflow-auto whitespace-pre-wrap break-words font-mono text-[10px] leading-4 text-secondary">{latest.body_md}</pre>
    </details>}
    {error && <p role="alert" className="mt-3 text-[11px] text-rose-700">{error}</p>}
    <div className="mt-3 flex items-center gap-2">
      <button type="submit" className={primaryButtonClass} disabled={saving}>{saving ? 'Saving…' : base ? 'Save revision' : 'Save note'}</button>
      <button type="button" className={buttonClass} onClick={onCancel} disabled={saving}>Cancel</button>
      {base && <span className="text-[10px] text-muted">Editing revision {base.revision}</span>}
    </div>
  </form>;
}

/** Every revision of one note, oldest first, with each revision's raw Markdown. */
export function NoteHistory({ runKey, noteId }: { runKey: string; noteId: string }) {
  const [history, setHistory] = useState<Loadable<RunNote[]>>({ state: 'loading' });
  useEffect(() => {
    let active = true;
    api.runNoteHistory(runKey, noteId)
      .then((revisions) => {
        if (active) setHistory(Array.isArray(revisions) ? { state: 'ready', value: revisions.filter(isRunNote) } : { state: 'error', message: 'The history response was not a list.' });
      })
      .catch((cause: unknown) => { if (active) setHistory({ state: 'error', message: errorMessage(cause) }); });
    return () => { active = false; };
  }, [noteId, runKey]);
  if (history.state === 'loading') return <p className="mt-2 text-[11px] text-muted">Loading history…</p>;
  if (history.state !== 'ready') return <p className="mt-2 text-[11px] text-rose-700">History unavailable{history.state === 'error' ? `: ${history.message}` : ''}</p>;
  return <ol className="mt-2 space-y-2 border-l-2 border-divider pl-3" aria-label="Note history">
    {history.value.map((revision) => <li key={revision.revision} className="text-[11px]">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-muted">
        <strong className="font-medium text-ink">Revision {revision.revision}</strong>
        <span>{revision.source}</span>
        <time dateTime={revision.revised_at}>{formatAbsoluteTime(revision.revised_at)}</time>
        <KindBadge kind={revision.kind} />
        {revision.title && <span className="text-secondary">{revision.title}</span>}
        {revision.deleted && <span className="rounded-[3px] bg-rose-50 px-1.5 py-px text-[10px] font-medium text-rose-700">deleted</span>}
      </div>
      {!revision.deleted && <pre className="mt-1 max-h-60 overflow-auto whitespace-pre-wrap break-words rounded bg-ink/[.035] p-2 font-mono text-[10px] leading-4 text-secondary">{revision.body_md}</pre>}
    </li>)}
  </ol>;
}

type NoteItemProps = {
  runKey: string;
  item: RenderedRunNote;
  writes: boolean;
  onEdit: () => void;
  onChanged: () => void;
};

function NoteItem({ runKey, item, writes, onEdit, onChanged }: NoteItemProps) {
  const { note, rendered } = item;
  const [historyOpen, setHistoryOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stale, setStale] = useState(false);

  const remove = async () => {
    const name = note.title ? `“${note.title}”` : `this ${note.kind} note`;
    if (!window.confirm(`Delete ${name}? Its earlier revisions stay in its history.`)) return;
    setDeleting(true);
    setError(null);
    try {
      await api.deleteRunNote(runKey, note.note_id, note.revision);
      onChanged();
    } catch (cause) {
      const revision = conflictRevision(cause);
      if (revision != null) {
        setStale(true);
        setError(`This note changed since it was loaded${Number.isFinite(revision) ? ` (it is now at revision ${revision})` : ''}. Reload the notes before deleting it.`);
      } else {
        setError(errorMessage(cause));
      }
    } finally {
      setDeleting(false);
    }
  };

  return <article className="border-b border-divider px-4 py-3 last:border-b-0" aria-label={note.title ?? `${note.kind} note`}>
    <header className="flex flex-wrap items-center gap-x-2 gap-y-1">
      <KindBadge kind={note.kind} />
      {note.title && <h3 className="text-[12px] font-medium text-ink">{note.title}</h3>}
      <span className="text-[10px] text-muted">
        {note.source} · <NoteTime value={note.revised_at} /> · revision {note.revision}
      </span>
      <span className="ml-auto flex items-center gap-1.5">
        <button type="button" className={buttonClass} aria-expanded={historyOpen} onClick={() => setHistoryOpen((open) => !open)}>
          <ClockCounterClockwise size={12} aria-hidden="true" /> History
        </button>
        {writes && <>
          <button type="button" className={buttonClass} onClick={onEdit}><PencilSimple size={12} aria-hidden="true" /> Edit</button>
          <button type="button" className={buttonClass} onClick={() => void remove()} disabled={deleting}><Trash size={12} aria-hidden="true" /> Delete</button>
        </>}
      </span>
    </header>
    <div className="mt-2"><NoteMarkdown rendered={rendered} /></div>
    {error && <div role="alert" className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-rose-700">
      <span>{error}</span>
      {stale && <button type="button" className={buttonClass} onClick={onChanged}>Reload notes</button>}
    </div>}
    {historyOpen && <NoteHistory runKey={runKey} noteId={note.note_id} />}
  </article>;
}

type EditorState = { mode: 'add' } | { mode: 'edit'; note: RunNote } | null;

/**
 * The run card (the job kind's template rendered for this run) followed by
 * the run's notes, newest first, with add, edit, delete and history.
 */
export function RunNotesPanel({ runKey }: { runKey: string }) {
  const [writes, setWrites] = useState(false);
  const [card, setCard] = useState<Loadable<RenderedNote>>({ state: 'loading' });
  const [cardOpen, setCardOpen] = useState(true);
  const [notes, setNotes] = useState<Loadable<RenderedRunNote[]>>({ state: 'loading' });
  const [editor, setEditor] = useState<EditorState>(null);
  const notesRequest = useRef(0);

  const loadNotes = useCallback(async () => {
    const sequence = ++notesRequest.current;
    try {
      const value = await api.runNotes(runKey);
      if (sequence !== notesRequest.current) return;
      setNotes(isRenderedRunNotes(value)
        ? { state: 'ready', value: value.filter((item) => !item.note.deleted) }
        : { state: 'error', message: 'The notes response was not a list of notes.' });
    } catch (cause) {
      if (sequence !== notesRequest.current) return;
      if (cause instanceof ApiError && cause.code === 'notes_unavailable') setNotes({ state: 'unavailable' });
      else setNotes({ state: 'error', message: errorMessage(cause) });
    }
  }, [runKey]);

  useEffect(() => {
    let active = true;
    setCard({ state: 'loading' });
    setNotes({ state: 'loading' });
    setEditor(null);
    api.noteSettings()
      .then((settings) => { if (active) setWrites(isRecord(settings) && settings.writes === true); })
      .catch(() => { if (active) setWrites(false); });
    api.runCard(runKey)
      .then((value) => {
        if (active) setCard(isRenderedNote(value) ? { state: 'ready', value } : { state: 'error', message: 'The card response was not a rendered note.' });
      })
      .catch((cause: unknown) => { if (active) setCard({ state: 'error', message: errorMessage(cause) }); });
    void loadNotes();
    return () => { active = false; };
  }, [loadNotes, runKey]);

  const canWrite = writes && notes.state === 'ready';
  const count = notes.state === 'ready' ? notes.value.length : null;

  return <section className="obs-card mt-5 overflow-hidden" aria-label="Run notes">
    <header className="flex flex-wrap items-center gap-3 border-b border-divider px-4 py-3">
      <div className="min-w-0">
        <h2 className="text-[13px] font-medium">Card and notes{count ? <span className="ml-1.5 font-normal text-muted">{count}</span> : null}</h2>
        <p className="mt-0.5 text-[11px] text-muted">The run card is rendered from this job kind's template; notes record what people and agents learned from the run.</p>
      </div>
      {canWrite && editor?.mode !== 'add' && <button type="button" className={`${primaryButtonClass} ml-auto`} onClick={() => setEditor({ mode: 'add' })}>
        <Plus size={12} weight="bold" aria-hidden="true" /> Add note
      </button>}
    </header>

    <details open={cardOpen} onToggle={(event) => setCardOpen(event.currentTarget.open)} className="border-b border-divider">
      <summary className="flex cursor-pointer list-none items-center gap-2 px-4 py-2.5 select-none [&::-webkit-details-marker]:hidden">
        <CaretRight size={11} aria-hidden="true" className={`text-muted transition-transform ${cardOpen ? 'rotate-90' : ''}`} />
        <span className="text-[12px] font-medium">Run card</span>
        {card.state === 'ready' && card.value.template && <code className="rounded-[3px] bg-subtle px-1.5 py-px text-[10px] text-muted">{card.value.template}</code>}
      </summary>
      <div className="px-4 pb-4">
        {card.state === 'loading' ? <p className="text-[11px] text-muted">Rendering the run card…</p>
          : card.state === 'ready' ? <NoteMarkdown rendered={card.value} />
          : <p className="text-[11px] text-muted">The run card is unavailable{card.state === 'error' ? `: ${card.message}` : '.'}</p>}
      </div>
    </details>

    {editor && <NoteEditor
      key={editor.mode === 'edit' ? `${editor.note.note_id}:${editor.note.revision}` : 'add'}
      runKey={runKey}
      note={editor.mode === 'edit' ? editor.note : undefined}
      onCancel={() => setEditor(null)}
      onSaved={() => {
        setEditor(null);
        void loadNotes();
      }}
    />}

    {notes.state === 'loading' ? <p className="px-4 py-3 text-[11px] text-muted">Loading notes…</p>
      : notes.state === 'unavailable' ? <p className="px-4 py-3 text-[11px] text-muted">Notes are not available for this source.</p>
      : notes.state === 'error' ? <p className="px-4 py-3 text-[11px] text-rose-700">Notes could not be loaded: {notes.message}</p>
      : notes.value.length === 0 ? <p className="px-4 py-3 text-[11px] text-muted">No notes for this run yet.</p>
      : <div>{notes.value.map((item) => <NoteItem
          key={`${item.note.note_id}:${item.note.revision}`}
          runKey={runKey}
          item={item}
          writes={writes}
          onEdit={() => setEditor({ mode: 'edit', note: item.note })}
          onChanged={() => void loadNotes()}
        />)}</div>}
  </section>;
}

export default RunNotesPanel;
