"""Run notes: Markdown written about runs, stored beside them with revisions.

Notes are not evidence. A note store is separate from `RunDataSource` so that
readers stay write-free; a tracking backend offers one only when it can keep
revisions. Every edit is a new revision guarded by the revision the editor
last saw, and a delete is a tombstone revision that keeps the history.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import Literal, Protocol, runtime_checkable

from pydantic import Field

from .models import TrackingModel

type NoteSource = Literal["cli", "mcp", "observatory"]
type NoteScope = Literal["run", "project"]

NOTE_SOURCES: tuple[NoteSource, ...] = ("cli", "mcp", "observatory")


class RunNote(TrackingModel):
    """One revision of one note."""

    note_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    scope: NoteScope = "run"
    run_id: str | None = None
    kind: str = Field(min_length=1, max_length=128)
    title: str | None = None
    body_md: str
    source: NoteSource
    created_at: datetime
    revised_at: datetime
    deleted: bool = False


class NoteConflict(Exception):
    """A revise or delete was based on a revision that is no longer the latest."""

    def __init__(self, note_id: str, current_revision: int | None) -> None:
        self.note_id = note_id
        self.current_revision = current_revision
        current = "unknown" if current_revision is None else str(current_revision)
        super().__init__(f"note {note_id!r} is at revision {current}; reload it and apply the edit again")


class NotesUnavailable(Exception):
    """The tracking backend cannot store notes (for example an older server)."""


@runtime_checkable
class RunNoteStore(Protocol):
    async def list_notes(
        self, run_id: str | None = None, *, kind: str | None = None, include_deleted: bool = False
    ) -> tuple[RunNote, ...]:
        """Latest revision of each note, newest first; run notes of one run when `run_id` is given."""
        ...

    async def note_history(self, note_id: str) -> tuple[RunNote, ...]:
        """Every revision of one note, oldest first."""
        ...

    async def add_note(
        self,
        *,
        run_id: str | None,
        kind: str,
        body_md: str,
        source: NoteSource,
        title: str | None = None,
        note_id: str | None = None,
    ) -> RunNote: ...

    async def revise_note(
        self,
        note_id: str,
        *,
        expected_revision: int,
        body_md: str,
        source: NoteSource,
        kind: str | None = None,
        title: str | None = None,
    ) -> RunNote: ...

    async def delete_note(self, note_id: str, *, expected_revision: int, source: NoteSource) -> RunNote: ...


class InMemoryRunNoteStore:
    """A complete `RunNoteStore` kept in memory, for tests, fixtures and demos."""

    def __init__(self) -> None:
        self._revisions: dict[str, list[RunNote]] = {}
        self._lock = asyncio.Lock()

    async def list_notes(
        self, run_id: str | None = None, *, kind: str | None = None, include_deleted: bool = False
    ) -> tuple[RunNote, ...]:
        latest = [revisions[-1] for revisions in self._revisions.values()]
        notes = [
            note
            for note in latest
            if (run_id is None or note.run_id == run_id)
            and (kind is None or note.kind == kind)
            and (include_deleted or not note.deleted)
        ]
        return tuple(sorted(notes, key=lambda note: (note.revised_at, note.note_id), reverse=True))

    async def note_history(self, note_id: str) -> tuple[RunNote, ...]:
        if note_id not in self._revisions:
            raise LookupError(f"note {note_id!r} does not exist")
        return tuple(self._revisions[note_id])

    async def add_note(
        self,
        *,
        run_id: str | None,
        kind: str,
        body_md: str,
        source: NoteSource,
        title: str | None = None,
        note_id: str | None = None,
    ) -> RunNote:
        async with self._lock:
            identity = note_id or uuid.uuid4().hex
            existing = self._revisions.get(identity)
            if existing is not None:
                first = existing[0]
                if (first.run_id, first.kind, first.body_md, first.title) == (run_id, kind, body_md, title):
                    return first
                raise NoteConflict(identity, existing[-1].revision)
            now = datetime.now(UTC)
            note = RunNote(
                note_id=identity,
                revision=1,
                scope="run" if run_id is not None else "project",
                run_id=run_id,
                kind=kind,
                title=title,
                body_md=body_md,
                source=source,
                created_at=now,
                revised_at=now,
            )
            self._revisions[identity] = [note]
            return note

    async def revise_note(
        self,
        note_id: str,
        *,
        expected_revision: int,
        body_md: str,
        source: NoteSource,
        kind: str | None = None,
        title: str | None = None,
    ) -> RunNote:
        return await self._append(
            note_id, expected_revision, source=source, body_md=body_md, kind=kind, title=title, deleted=False
        )

    async def delete_note(self, note_id: str, *, expected_revision: int, source: NoteSource) -> RunNote:
        return await self._append(note_id, expected_revision, source=source, deleted=True)

    async def _append(
        self,
        note_id: str,
        expected_revision: int,
        *,
        source: NoteSource,
        deleted: bool,
        body_md: str | None = None,
        kind: str | None = None,
        title: str | None = None,
    ) -> RunNote:
        async with self._lock:
            revisions = self._revisions.get(note_id)
            if revisions is None:
                raise LookupError(f"note {note_id!r} does not exist")
            latest = revisions[-1]
            if latest.revision != expected_revision or latest.deleted:
                raise NoteConflict(note_id, latest.revision)
            note = latest.model_copy(
                update={
                    "revision": latest.revision + 1,
                    "body_md": latest.body_md if body_md is None else body_md,
                    "kind": kind or latest.kind,
                    "title": latest.title if title is None else (title or None),
                    "source": source,
                    "revised_at": datetime.now(UTC),
                    "deleted": deleted,
                }
            )
            revisions.append(note)
            return note


async def verify_note_store(store: RunNoteStore, run_id: str, other_run_id: str) -> None:
    """Exercise the behaviour every note store must have, against two existing runs.

    Backends run this in their tests so that equivalent logical results are
    checked rather than identical storage. Raises `AssertionError` on the first
    difference.
    """

    async def conflict(action: Awaitable[RunNote]) -> NoteConflict:
        try:
            await action
        except NoteConflict as error:
            return error
        raise AssertionError("expected a note conflict")

    first = await store.add_note(run_id=run_id, kind="finding", body_md="First **finding**.", source="cli")
    assert (first.revision, first.run_id, first.scope, first.source) == (1, run_id, "run", "cli"), first
    summary_id = f"{run_id}.summary"
    retried = await store.add_note(run_id=run_id, kind="summary", body_md="Summary.", source="mcp", note_id=summary_id)
    again = await store.add_note(run_id=run_id, kind="summary", body_md="Summary.", source="mcp", note_id=summary_id)
    assert again.note_id == retried.note_id and again.revision == 1, "an identical retried add must be a no-op"
    await conflict(
        store.add_note(run_id=run_id, kind="summary", body_md="Different.", source="mcp", note_id=summary_id)
    )
    await store.add_note(run_id=other_run_id, kind="finding", body_md="Elsewhere.", source="observatory")

    revised = await store.revise_note(
        first.note_id, expected_revision=1, body_md="Corrected finding.", source="observatory", title="Fix"
    )
    assert (revised.revision, revised.body_md, revised.title, revised.source, revised.run_id) == (
        2,
        "Corrected finding.",
        "Fix",
        "observatory",
        run_id,
    ), revised
    assert revised.created_at == first.created_at, "revisions keep the note's creation time"
    stale = await conflict(store.revise_note(first.note_id, expected_revision=1, body_md="Stale edit.", source="cli"))
    assert stale.current_revision == 2, stale

    listed = await store.list_notes(run_id)
    assert {note.note_id: note.revision for note in listed} == {first.note_id: 2, summary_id: 1}, listed
    assert [note.note_id for note in await store.list_notes(run_id, kind="summary")] == [summary_id]

    deleted = await store.delete_note(first.note_id, expected_revision=2, source="cli")
    assert deleted.deleted and deleted.revision == 3, deleted
    assert [note.note_id for note in await store.list_notes(run_id)] == [summary_id]
    assert first.note_id in {note.note_id for note in await store.list_notes(run_id, include_deleted=True)}
    history = await store.note_history(first.note_id)
    assert [(note.revision, note.deleted) for note in history] == [(1, False), (2, False), (3, True)], history
    assert history[0].body_md == "First **finding**."


__all__ = [
    "InMemoryRunNoteStore",
    "NOTE_SOURCES",
    "NoteConflict",
    "NoteScope",
    "NoteSource",
    "NotesUnavailable",
    "RunNote",
    "RunNoteStore",
    "verify_note_store",
]
