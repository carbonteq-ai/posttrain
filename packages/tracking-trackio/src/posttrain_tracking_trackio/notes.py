"""Run notes stored by the Trackio fork (`carbonteq-trackio` 0.31.5.post14.dev28 and later).

Trackio keys notes to its own run identity (provider run id and run name), so
this store maps Posttrain run ids to Trackio runs the same way
`TrackioDataSource` does, and records the Posttrain run id in each note's
metadata. Reads need no credential; writes to a remote server need
`TRACKIO_WRITE_TOKEN`.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from threading import Lock
from typing import Any

import trackio
from posttrain.common import ContractError
from posttrain.tracking import NoteConflict, NoteSource, NotesUnavailable, RunNote
from trackio.utils import parse_trackio_server_url

_NOTE_METHODS = ("run_notes", "run_note_history", "add_run_note", "revise_run_note", "delete_run_note")
_POSTTRAIN_RUN = "posttrain_run_id"


class TrackioRunNotes:
    """`RunNoteStore` over one Trackio project."""

    def __init__(
        self,
        project: str,
        *,
        server_url: str | None = None,
        write_token: str | None = None,
        api: Any | None = None,
    ) -> None:
        self.project = project
        base_url, url_token = parse_trackio_server_url(server_url) if server_url else (None, None)
        self._token = write_token or url_token or (os.getenv("TRACKIO_WRITE_TOKEN") if base_url else None)
        self._remote = base_url is not None
        self._api = api if api is not None else _api(base_url, self._token)
        self._lock = Lock()
        self._by_posttrain: dict[str, tuple[str, str]] = {}
        self._by_provider: dict[str, str] = {}

    @property
    def supported(self) -> bool:
        """Whether the installed Trackio client can store notes (the server is checked on first use)."""

        return self._api is not None and all(callable(getattr(self._api, name, None)) for name in _NOTE_METHODS)

    @property
    def writable(self) -> bool:
        return self.supported and (not self._remote or self._token is not None)

    async def list_notes(
        self, run_id: str | None = None, *, kind: str | None = None, include_deleted: bool = False
    ) -> tuple[RunNote, ...]:
        provider = await self._provider(run_id) if run_id is not None else None
        rows = await self._call(
            "run_notes",
            self.project,
            run_id=provider[0] if provider else None,
            scope="run" if run_id is not None else None,
            kind=kind,
            include_deleted=include_deleted,
        )
        return tuple(await self._notes(rows))

    async def note_history(self, note_id: str) -> tuple[RunNote, ...]:
        rows = await self._call("run_note_history", self.project, note_id)
        if not rows:
            raise LookupError(f"note {note_id!r} does not exist in Trackio project {self.project!r}")
        return tuple(await self._notes(rows))

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
        self._require_writable()
        provider = await self._provider(run_id) if run_id is not None else None
        row = await self._call(
            "add_run_note",
            self.project,
            scope="run" if run_id is not None else "project",
            kind=kind,
            body_md=body_md,
            source=source,
            run_id=provider[0] if provider else None,
            run_name=provider[1] if provider else None,
            title=title,
            note_id=note_id,
            metadata={_POSTTRAIN_RUN: run_id} if run_id is not None else None,
        )
        return (await self._notes([row]))[0]

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
        self._require_writable()
        row = await self._call(
            "revise_run_note",
            self.project,
            note_id,
            expected_revision=expected_revision,
            body_md=body_md,
            source=source,
            kind=kind,
            title=title,
        )
        return (await self._notes([row]))[0]

    async def delete_note(self, note_id: str, *, expected_revision: int, source: NoteSource) -> RunNote:
        self._require_writable()
        row = await self._call(
            "delete_run_note", self.project, note_id, expected_revision=expected_revision, source=source
        )
        return (await self._notes([row]))[0]

    def _require_writable(self) -> None:
        if not self.supported:
            raise NotesUnavailable(_UNSUPPORTED)
        if not self.writable:
            raise ContractError("writing run notes to a Trackio server requires TRACKIO_WRITE_TOKEN")

    async def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if not self.supported:
            raise NotesUnavailable(_UNSUPPORTED)
        call: Callable[..., Any] = getattr(self._api, method)
        try:
            return await asyncio.to_thread(call, *args, **kwargs)
        except Exception as error:
            detail = getattr(error, "detail", None)
            if isinstance(detail, Mapping) and detail.get("type") == "run_note":
                current = detail.get("current_revision")
                raise NoteConflict(
                    str(detail.get("note_id") or (args[1] if len(args) > 1 else "")),
                    current if isinstance(current, int) else None,
                ) from error
            if "does not support '/" in str(error):
                raise NotesUnavailable(
                    f"the Trackio server for project {self.project!r} does not store run notes; "
                    "upgrade it to carbonteq-trackio 0.31.5.post14.dev28 or later"
                ) from error
            raise

    async def _provider(self, run_id: str) -> tuple[str, str]:
        with self._lock:
            known = self._by_posttrain.get(run_id)
        if known is None:
            await asyncio.to_thread(self._load_runs)
            with self._lock:
                known = self._by_posttrain.get(run_id)
        if known is None:
            raise LookupError(f"posttrain run {run_id!r} was not found in Trackio project {self.project!r}")
        return known

    def _load_runs(self) -> None:
        mapping: dict[str, tuple[str, str]] = {}
        for run in self._api.runs(self.project):
            posttrain_run_id = (run.config or {}).get("run_id")
            if isinstance(posttrain_run_id, str):
                mapping[posttrain_run_id] = (str(run.id), str(run.name))
        with self._lock:
            self._by_posttrain.update(mapping)
            self._by_provider.update({provider: run_id for run_id, (provider, _) in mapping.items()})

    async def _notes(self, rows: Any) -> list[RunNote]:
        if not isinstance(rows, list):
            raise ContractError("Trackio returned an invalid run-note response")
        needs_mapping = any(
            isinstance(row, Mapping)
            and row.get("run_id") is not None
            and not _metadata_run(row)
            and str(row["run_id"]) not in self._by_provider
            for row in rows
        )
        if needs_mapping:
            await asyncio.to_thread(self._load_runs)
        return [self._note(row) for row in rows]

    def _note(self, row: Any) -> RunNote:
        if not isinstance(row, Mapping):
            raise ContractError("Trackio returned an invalid run note")
        provider_run = row.get("run_id")
        run_id = _metadata_run(row) or (
            self._by_provider.get(str(provider_run), str(provider_run)) if provider_run is not None else None
        )
        return RunNote(
            note_id=str(row["note_id"]),
            revision=int(row["revision"]),
            scope="project" if row.get("scope") == "project" else "run",
            run_id=run_id,
            kind=str(row["kind"]),
            title=row.get("title") or None,
            body_md=str(row["body_md"]),
            source=_source(row.get("source")),
            created_at=_time(row["created_at"]),
            revised_at=_time(row["revised_at"]),
            deleted=bool(row.get("deleted")),
        )


_UNSUPPORTED = (
    "the installed Trackio client cannot store run notes; install carbonteq-trackio 0.31.5.post14.dev28 or later"
)


def _api(server_url: str | None, token: str | None) -> Any:
    api_type: Any = trackio.Api
    try:
        return api_type(server_url=server_url, write_token=token) if token else api_type(server_url=server_url)
    except TypeError:
        # Clients before the run-notes release take no write token; reads of
        # notes are unsupported there anyway.
        return trackio.Api(server_url=server_url)


def _metadata_run(row: Mapping[str, Any]) -> str | None:
    metadata = row.get("metadata")
    value = metadata.get(_POSTTRAIN_RUN) if isinstance(metadata, Mapping) else None
    return value if isinstance(value, str) and value else None


def _source(value: Any) -> NoteSource:
    if value not in ("cli", "mcp", "observatory"):
        raise ContractError(f"Trackio run note has an unknown source {value!r}")
    return value


def _time(value: Any) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


__all__ = ["TrackioRunNotes"]
