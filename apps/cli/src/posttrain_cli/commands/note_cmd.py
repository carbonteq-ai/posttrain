"""`posttrain run card` and `posttrain run note`: run cards and Markdown notes on runs."""

from __future__ import annotations

import asyncio
import importlib
import warnings
from pathlib import Path
from typing import Annotated, Any

import typer
from posttrain.common import ContractError
from posttrain.execution import ExecutionEvidenceSource

from ..context import CliState
from ..execution_provider import evidence_source_for_run
from ..output import emit
from ..run_resolve import resolve_run_id
from ..tracking_config import project_observatory_settings, project_tracking_environment

_RUN = Annotated[str, typer.Argument(help="canonical run id, or an unambiguous prefix")]
_NOTE = Annotated[str, typer.Argument(help="note id")]
_BODY = Annotated[str | None, typer.Option("--body", help="Markdown text")]
_FILE = Annotated[Path | None, typer.Option("--file", help="a Markdown file", exists=True, dir_okay=False)]


def _body(body: str | None, file: Path | None) -> str:
    if (body is None) == (file is None):
        raise ContractError("give exactly one of --body or --file")
    return body if body is not None else file.read_text(encoding="utf-8")  # type: ignore[union-attr]


def _service(state: CliState, run_id: str, *, writes: bool = False) -> tuple[Any, Any, str]:
    layout = state.layout()
    run_id = resolve_run_id(layout, run_id)
    try:
        observatory = importlib.import_module("posttrain_observatory")
    except ImportError as error:
        raise RuntimeError(
            "Observatory is not installed; run `uv add 'posttrain[observatory]'` "
            "or install the posttrain-observatory package"
        ) from error
    source: ExecutionEvidenceSource | None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            source = evidence_source_for_run(layout, run_id)
    except FileNotFoundError:
        # Runs submitted from another machine: use the project's tracking.
        source = None
    settings = project_observatory_settings(layout, observatory.ObservatorySettings, evidence_source=source)
    templates = layout.control_dir / "note_templates"
    settings = settings.model_copy(
        update={
            "note_writes": writes,
            "trackio_write_token": project_tracking_environment(layout).get("TRACKIO_WRITE_TOKEN") if writes else None,
            "note_templates_dir": str(templates) if templates.is_dir() else None,
        }
    )
    locator = observatory.RunLocator(source_id=settings.source_id, run_id=run_id)
    return observatory.create_service(settings), locator, run_id


def _note_line(note: dict[str, Any]) -> str:
    title = f" — {note['title']}" if note.get("title") else ""
    return (
        f"{note['note_id']}  {note['kind']}  revision {note['revision']}  "
        f"({note['source']}, {str(note['revised_at'])[:16].replace('T', ' ')}){title}"
    )


def register(run_app: typer.Typer) -> None:
    note_app = typer.Typer(rich_markup_mode=None, no_args_is_help=True, help="Markdown notes on runs")
    run_app.add_typer(note_app, name="note")

    @run_app.command("card", help="the run's card: its job kind's template rendered from recorded data")
    def card_cmd(ctx: typer.Context, run_id: _RUN) -> None:
        state: CliState = ctx.obj
        service, locator, _ = _service(state, run_id)
        card = asyncio.run(service.run_card(locator)).model_dump(mode="json")
        emit(state, card, card["text"].rstrip() + f"\n\n(template {card['template']})")

    @note_app.command("add", help="attach a Markdown note to a run")
    def add_cmd(
        ctx: typer.Context,
        run_id: _RUN,
        kind: Annotated[str, typer.Option("--kind", help="summary, finding, correction, decision, ...")],
        body: _BODY = None,
        file: _FILE = None,
        title: Annotated[str | None, typer.Option("--title")] = None,
        note_id: Annotated[
            str | None, typer.Option("--note-id", help="a stable id makes a retried add a no-op")
        ] = None,
    ) -> None:
        state: CliState = ctx.obj
        service, locator, _ = _service(state, run_id, writes=True)
        observatory = importlib.import_module("posttrain_observatory.run_notes")
        request = observatory.NoteAddRequest(kind=kind, body_md=_body(body, file), title=title, note_id=note_id)
        preview = asyncio.run(service.render_note(locator, request.body_md))
        note = asyncio.run(service.add_run_note(locator, request, source="cli")).model_dump(mode="json")
        lines = [f"added {_note_line(note)}"]
        lines += [f"warning: {item}" for item in preview.unresolved]
        emit(state, {"note": note, "unresolved": list(preview.unresolved)}, "\n".join(lines))

    @note_app.command("edit", help="save a new revision of a note")
    def edit_cmd(
        ctx: typer.Context,
        run_id: _RUN,
        note_id: _NOTE,
        expected_revision: Annotated[int, typer.Option("--expected-revision", min=1, help="the revision you read")],
        body: _BODY = None,
        file: _FILE = None,
        kind: Annotated[str | None, typer.Option("--kind")] = None,
        title: Annotated[str | None, typer.Option("--title")] = None,
    ) -> None:
        state: CliState = ctx.obj
        service, locator, _ = _service(state, run_id, writes=True)
        notes = importlib.import_module("posttrain_observatory.run_notes")
        request = notes.NoteReviseRequest(
            expected_revision=expected_revision, body_md=_body(body, file), kind=kind, title=title
        )
        note = asyncio.run(service.revise_run_note(locator, note_id, request, source="cli")).model_dump(mode="json")
        emit(state, note, f"revised {_note_line(note)}")

    @note_app.command("delete", help="hide a note behind a tombstone revision (history is kept)")
    def delete_cmd(
        ctx: typer.Context,
        run_id: _RUN,
        note_id: _NOTE,
        expected_revision: Annotated[int, typer.Option("--expected-revision", min=1)],
    ) -> None:
        state: CliState = ctx.obj
        service, locator, _ = _service(state, run_id, writes=True)
        notes = importlib.import_module("posttrain_observatory.run_notes")
        request = notes.NoteDeleteRequest(expected_revision=expected_revision)
        note = asyncio.run(service.delete_run_note(locator, note_id, request, source="cli")).model_dump(mode="json")
        emit(state, note, f"deleted {note['note_id']} (tombstone revision {note['revision']})")

    @note_app.command("show", help="the run's notes, rendered")
    def show_cmd(
        ctx: typer.Context,
        run_id: _RUN,
        raw: Annotated[bool, typer.Option("--raw", help="print the Markdown as written")] = False,
        kind: Annotated[str | None, typer.Option("--kind")] = None,
    ) -> None:
        state: CliState = ctx.obj
        service, locator, resolved = _service(state, run_id)
        notes = [item.model_dump(mode="json") for item in asyncio.run(service.run_notes(locator, kind=kind))]
        blocks = [
            f"## {_note_line(item['note'])}\n\n{(item['note']['body_md'] if raw else item['rendered']['text']).rstrip()}"
            for item in notes
        ]
        emit(state, notes, "\n\n".join(blocks) if blocks else f"no notes on {resolved}")

    @note_app.command("history", help="every revision of one note")
    def history_cmd(ctx: typer.Context, run_id: _RUN, note_id: _NOTE) -> None:
        state: CliState = ctx.obj
        service, locator, _ = _service(state, run_id)
        history = [note.model_dump(mode="json") for note in asyncio.run(service.run_note_history(locator, note_id))]
        blocks = [
            f"## revision {note['revision']}{' (deleted)' if note['deleted'] else ''} — {note['source']}, "
            f"{str(note['revised_at'])[:16].replace('T', ' ')}\n\n{note['body_md'].rstrip()}"
            for note in history
        ]
        emit(state, history, "\n\n".join(blocks))


__all__ = ["register"]
