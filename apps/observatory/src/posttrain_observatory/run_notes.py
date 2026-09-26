"""Run notes in the Observatory: the one write it offers, and the rendering of notes and cards.

Notes are stored by the tracking backend's `RunNoteStore`, never in run
evidence. The Observatory adds, revises and deletes them only when the
deployment enables note writing, and renders every note through the semantic
layer so that numbers in a note come from recorded data.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from posttrain.tracking import NoteConflict, NoteSource, NotesUnavailable, RunDataSource, RunNote, RunNoteStore
from pydantic import Field

from .models import ObservatoryModel, RunLocator
from .note_render import RenderedNote, render_note
from .run_cards import TemplateSet

if TYPE_CHECKING:
    from .service import ObservatoryService

type NoteStoreFactory = Callable[[str, RunDataSource], RunNoteStore | None]

MAX_NOTE_BYTES = 256 * 1024


class RenderedRunNote(ObservatoryModel):
    note: RunNote
    rendered: RenderedNote


class NoteAddRequest(ObservatoryModel):
    kind: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
    body_md: str = Field(min_length=1, max_length=MAX_NOTE_BYTES)
    title: str | None = Field(default=None, max_length=200)
    # No "/": note ids appear in HTTP paths.
    note_id: str | None = Field(default=None, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


class NoteReviseRequest(ObservatoryModel):
    expected_revision: int = Field(ge=1)
    body_md: str = Field(min_length=1, max_length=MAX_NOTE_BYTES)
    kind: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
    title: str | None = Field(default=None, max_length=200)


class NoteDeleteRequest(ObservatoryModel):
    expected_revision: int = Field(ge=1)


class NotePreviewRequest(ObservatoryModel):
    run_key: str = Field(min_length=1)
    body_md: str = Field(max_length=MAX_NOTE_BYTES)


class NotesDisabled(PermissionError):
    """This Observatory deployment does not write notes."""


class RunNotes:
    """Note stores per source, templates, and the rendering of notes and cards."""

    def __init__(
        self,
        service: ObservatoryService,
        *,
        store_factory: NoteStoreFactory | None,
        templates: TemplateSet,
        writes: bool,
    ) -> None:
        self._service = service
        self._factory = store_factory
        self._stores: dict[str, RunNoteStore | None] = {}
        self.templates = templates
        self.writes = writes

    def store(self, locator: RunLocator) -> RunNoteStore:
        if locator.source_id not in self._stores:
            source = self._service.registry.resolve(locator)
            self._stores[locator.source_id] = (
                self._factory(locator.source_id, source) if self._factory is not None else None
            )
        store = self._stores[locator.source_id]
        if store is None:
            raise NotesUnavailable(f"source {locator.source_id!r} does not store run notes")
        return store

    def writable_store(self, locator: RunLocator) -> RunNoteStore:
        if not self.writes:
            raise NotesDisabled("this Observatory does not write notes; set POSTTRAIN_OBSERVATORY_NOTE_WRITES=1")
        return self.store(locator)

    async def render(self, locator: RunLocator, body_md: str, *, template: str | None = None) -> RenderedNote:
        def link_for(run_id: str) -> str:
            return f"/runs/{RunLocator(source_id=locator.source_id, run_id=run_id).key}"

        return await render_note(
            body_md,
            run_id=locator.run_id,
            query=self._service.query_semantics,
            link_for=link_for,
            template=template,
        )

    async def card(self, locator: RunLocator) -> RenderedNote:
        detail = await self._service.registry.resolve(locator).get_run(locator.run_id)
        template = self.templates.for_job_kind(detail.summary.job_kind)
        return await self.render(locator, template.body, template=template.template_id)

    async def notes(self, locator: RunLocator, *, kind: str | None = None) -> tuple[RenderedRunNote, ...]:
        notes = await self.store(locator).list_notes(locator.run_id, kind=kind)
        return tuple([RenderedRunNote(note=note, rendered=await self.render(locator, note.body_md)) for note in notes])

    async def history(self, locator: RunLocator, note_id: str) -> tuple[RunNote, ...]:
        history = await self.store(locator).note_history(note_id)
        _require_run(history[0], locator)
        return history

    async def add(self, locator: RunLocator, request: NoteAddRequest, source: NoteSource) -> RunNote:
        return await self.writable_store(locator).add_note(
            run_id=locator.run_id,
            kind=request.kind,
            body_md=request.body_md,
            source=source,
            title=request.title,
            note_id=request.note_id,
        )

    async def revise(
        self, locator: RunLocator, note_id: str, request: NoteReviseRequest, source: NoteSource
    ) -> RunNote:
        store = self.writable_store(locator)
        _require_run((await store.note_history(note_id))[0], locator)
        return await store.revise_note(
            note_id,
            expected_revision=request.expected_revision,
            body_md=request.body_md,
            source=source,
            kind=request.kind,
            title=request.title,
        )

    async def delete(
        self, locator: RunLocator, note_id: str, request: NoteDeleteRequest, source: NoteSource
    ) -> RunNote:
        store = self.writable_store(locator)
        _require_run((await store.note_history(note_id))[0], locator)
        return await store.delete_note(note_id, expected_revision=request.expected_revision, source=source)


def _require_run(note: RunNote, locator: RunLocator) -> None:
    if note.run_id != locator.run_id:
        raise LookupError(f"note {note.note_id!r} does not belong to run {locator.run_id!r}")


def conflict_body(error: NoteConflict) -> dict[str, Any]:
    return {"note_id": error.note_id, "current_revision": error.current_revision}


__all__ = [
    "MAX_NOTE_BYTES",
    "NoteAddRequest",
    "NoteDeleteRequest",
    "NotePreviewRequest",
    "NoteReviseRequest",
    "NoteStoreFactory",
    "NotesDisabled",
    "RenderedRunNote",
    "RunNotes",
    "conflict_body",
]
