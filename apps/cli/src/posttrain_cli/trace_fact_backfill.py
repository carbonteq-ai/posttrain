"""Bounded historical projection of retained Verifiers traces."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

import typer
from posttrain.common import ContractError, TraceFactSet
from posttrain.environment import project_verifiers_trace_facts
from posttrain.tracking import TraceQuery

from .context import CliState
from .output import emit, json_value
from .tracking_config import project_tracking_environment

_TRACE_FACT_READ_CHUNK_SIZE = 1000


@dataclass(frozen=True, slots=True)
class TraceFactBackfillPage:
    """Sanitized result of one bounded, replay-safe historical fact page."""

    project: str
    provider_run_id: str
    cursor: str | None
    next_cursor: str | None
    inspected: int
    projected: int
    complete: int
    partial: int
    applied: int
    preview: bool


@dataclass(frozen=True, slots=True)
class TraceFactBackfillWindow:
    """Sanitized result of one bounded sequence of checkpointed pages."""

    project: str
    provider_run_id: str
    cursor: str | None
    next_cursor: str | None
    inspected: int
    projected: int
    complete: int
    partial: int
    applied: int
    preview: bool
    pages: tuple[TraceFactBackfillPage, ...]


def backfill_verifiers_trace_page(
    *,
    project: str,
    server_url: str,
    write_token: str | None,
    provider_run_id: str,
    cursor: str | None,
    page_size: int,
    apply: bool,
) -> TraceFactBackfillPage:
    """Compatibility wrapper for one physical checkpoint page."""

    if page_size < 1 or page_size > _TRACE_FACT_READ_CHUNK_SIZE:
        raise ContractError("trace-fact backfill physical page size must be between 1 and 1000")
    window = backfill_verifiers_trace_window(
        project=project,
        server_url=server_url,
        write_token=write_token,
        provider_run_id=provider_run_id,
        cursor=cursor,
        window_size=page_size,
        apply=apply,
    )
    return window.pages[0]


def backfill_verifiers_trace_window(
    *,
    project: str,
    server_url: str,
    write_token: str | None,
    provider_run_id: str,
    cursor: str | None,
    window_size: int,
    apply: bool,
    checkpoint: Callable[[TraceFactBackfillPage], None] | None = None,
) -> TraceFactBackfillWindow:
    """Process a bounded window while checkpointing every physical page.

    Source and writer construction are deliberately hoisted out of the loop.
    Only one at-most-1,000-trace payload page is resident at a time.  The
    callback runs after that page's write succeeds, so its next cursor is a
    safe resume point even when a later page is interrupted.
    """

    if window_size < 1 or window_size > 5000:
        raise ContractError("trace-fact backfill window size must be between 1 and 5000")
    try:
        adapter = importlib.import_module("posttrain_tracking_trackio")
    except ImportError as error:
        raise RuntimeError("trace-fact backfill requires the posttrain[trackio] extra") from error

    source = adapter.TrackioDataSource(project, server_url=server_url)
    provider_run = source._provider_run_by_id(provider_run_id)
    writer = adapter.TrackioTraceFactWriter(server_url, write_token=write_token) if apply else None
    pages: list[TraceFactBackfillPage] = []
    next_cursor = cursor
    remaining = window_size

    while remaining:
        read_limit = min(remaining, _TRACE_FACT_READ_CHUNK_SIZE)
        raw_page = asyncio.run(
            source.traces_by_provider_run_id(
                provider_run_id,
                TraceQuery(
                    trace_type="verifiers",
                    cursor=next_cursor,
                    limit=read_limit,
                    include_payload=True,
                ),
            )
        )
        complete = 0
        partial = 0
        updates: list[tuple[str, TraceFactSet]] = []
        for trace in raw_page.items:
            facts = project_verifiers_trace_facts(trace.payload, attributes=trace.attributes)
            if facts.state == "complete":
                complete += 1
            else:
                partial += 1
            if writer is not None:
                updates.append((trace.external_id, facts))
        if writer is not None:
            writer.upsert_many(
                project=project,
                run_name=str(provider_run.name),
                provider_run_id=str(provider_run.id),
                trace_type="verifiers",
                updates=updates,
            )

        page = TraceFactBackfillPage(
            project=project,
            provider_run_id=provider_run_id,
            cursor=next_cursor,
            next_cursor=raw_page.next_cursor,
            inspected=len(raw_page.items),
            projected=len(raw_page.items),
            complete=complete,
            partial=partial,
            applied=len(raw_page.items) if apply else 0,
            preview=not apply,
        )
        pages.append(page)
        if checkpoint is not None:
            checkpoint(page)

        remaining -= len(raw_page.items)
        next_cursor = raw_page.next_cursor
        if next_cursor is None or len(raw_page.items) < read_limit:
            break

    return TraceFactBackfillWindow(
        project=project,
        provider_run_id=provider_run_id,
        cursor=cursor,
        next_cursor=next_cursor,
        inspected=sum(page.inspected for page in pages),
        projected=sum(page.projected for page in pages),
        complete=sum(page.complete for page in pages),
        partial=sum(page.partial for page in pages),
        applied=sum(page.applied for page in pages),
        preview=not apply,
        pages=tuple(pages),
    )


def register(app: typer.Typer) -> None:
    trace_facts_app = typer.Typer(
        rich_markup_mode=None,
        no_args_is_help=True,
        help="project retained native traces into generic evidence facts",
    )
    app.add_typer(trace_facts_app, name="trace-facts")

    @trace_facts_app.command("backfill", help="preview or apply one checkpointed Verifiers trace-fact window")
    def trace_facts_backfill_cmd(
        ctx: typer.Context,
        provider_run_id: Annotated[str, typer.Argument(help="exact Trackio provider run id")],
        cursor: Annotated[str | None, typer.Option(help="resume cursor returned by the prior page")] = None,
        window_size: Annotated[
            int,
            typer.Option(
                "--window-size",
                "--page-size",
                min=1,
                max=5000,
                help="bounded orchestration window; checkpoints remain at most 1,000 traces",
            ),
        ] = 200,
        apply: Annotated[
            bool, typer.Option("--apply", help="persist this page; default is a non-mutating preview")
        ] = False,
        trackio_project: Annotated[
            str | None,
            typer.Option(
                "--trackio-project", help="override the configured Trackio project for cross-project maintenance"
            ),
        ] = None,
    ) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        if layout.tracking != "trackio":
            raise ContractError("trace-fact backfill requires a project configured with Trackio")
        environment = project_tracking_environment(layout)
        server_url = environment.get("POSTTRAIN_TRACKIO_SERVER_URL")
        if not server_url:
            raise ContractError("trace-fact backfill requires POSTTRAIN_TRACKIO_SERVER_URL")
        project = trackio_project or environment.get("POSTTRAIN_TRACKIO_PROJECT") or layout.project_id

        def report_checkpoint(page: TraceFactBackfillPage) -> None:
            if state.json_output:
                print(
                    json.dumps({"checkpoint": json_value(page)}, sort_keys=True),
                    file=sys.stderr,
                    flush=True,
                )
                return
            mode = "applied" if apply else "previewed"
            print(
                f"Trace-fact page {mode}: {page.project}/{page.provider_run_id} "
                f"({page.inspected} traces, {page.complete} complete, {page.partial} partial; "
                f"next cursor: {page.next_cursor or 'done'})",
                flush=True,
            )

        receipt = backfill_verifiers_trace_window(
            project=project,
            server_url=server_url,
            write_token=environment.get("TRACKIO_WRITE_TOKEN"),
            provider_run_id=provider_run_id,
            cursor=cursor,
            window_size=window_size,
            apply=apply,
            checkpoint=report_checkpoint,
        )
        mode = "applied" if apply else "previewed"
        emit(
            state,
            receipt,
            f"Trace-fact window {mode}: {receipt.project}/{receipt.provider_run_id} "
            f"({receipt.inspected} traces, {receipt.complete} complete, {receipt.partial} partial; "
            f"next cursor: {receipt.next_cursor or 'done'})",
        )


__all__ = [
    "TraceFactBackfillPage",
    "TraceFactBackfillWindow",
    "backfill_verifiers_trace_page",
    "backfill_verifiers_trace_window",
    "register",
]
