"""Bounded historical projection of retained Verifiers traces."""

from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass
from typing import Annotated

import typer
from posttrain.common import ContractError
from posttrain.environment import project_verifiers_trace_facts
from posttrain.tracking import TraceQuery

from .context import CliState
from .output import emit
from .tracking_config import project_tracking_environment


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
    """Project one retained page and optionally write its generic facts.

    The Trackio adapter supplies bounded raw records and exact-run writes. This
    command owns the Verifiers-aware calculation, so Trackio remains generic.
    """

    if page_size < 1 or page_size > 1000:
        raise ContractError("trace-fact backfill page size must be between 1 and 1000")
    try:
        adapter = importlib.import_module("posttrain_tracking_trackio")
    except ImportError as error:
        raise RuntimeError("trace-fact backfill requires the posttrain[trackio] extra") from error

    source = adapter.TrackioDataSource(project, server_url=server_url)
    provider_run = source._provider_run_by_id(provider_run_id)
    page = asyncio.run(
        source.traces_by_provider_run_id(
            provider_run_id,
            TraceQuery(trace_type="verifiers", cursor=cursor, limit=page_size, include_payload=True),
        )
    )
    writer = adapter.TrackioTraceFactWriter(server_url, write_token=write_token) if apply else None
    complete = 0
    partial = 0
    for trace in page.items:
        facts = project_verifiers_trace_facts(trace.payload, attributes=trace.attributes)
        if facts.state == "complete":
            complete += 1
        else:
            partial += 1
        if writer is not None:
            writer.upsert(
                project=project,
                run_name=str(provider_run.name),
                provider_run_id=str(provider_run.id),
                trace_type=trace.trace_type,
                external_id=trace.external_id,
                facts=facts,
            )
    return TraceFactBackfillPage(
        project=project,
        provider_run_id=provider_run_id,
        cursor=cursor,
        next_cursor=page.next_cursor,
        inspected=len(page.items),
        projected=len(page.items),
        complete=complete,
        partial=partial,
        applied=len(page.items) if apply else 0,
        preview=not apply,
    )


def register(app: typer.Typer) -> None:
    trace_facts_app = typer.Typer(
        rich_markup_mode=None,
        no_args_is_help=True,
        help="project retained native traces into generic evidence facts",
    )
    app.add_typer(trace_facts_app, name="trace-facts")

    @trace_facts_app.command("backfill", help="preview or apply one bounded Verifiers trace-fact page")
    def trace_facts_backfill_cmd(
        ctx: typer.Context,
        provider_run_id: Annotated[str, typer.Argument(help="exact Trackio provider run id")],
        cursor: Annotated[str | None, typer.Option(help="resume cursor returned by the prior page")] = None,
        page_size: Annotated[int, typer.Option("--page-size", min=1, max=1000)] = 200,
        apply: Annotated[bool, typer.Option("--apply", help="persist this page; default is a non-mutating preview")] = False,
        trackio_project: Annotated[
            str | None,
            typer.Option("--trackio-project", help="override the configured Trackio project for cross-project maintenance"),
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
        receipt = backfill_verifiers_trace_page(
            project=project,
            server_url=server_url,
            write_token=environment.get("TRACKIO_WRITE_TOKEN"),
            provider_run_id=provider_run_id,
            cursor=cursor,
            page_size=page_size,
            apply=apply,
        )
        mode = "applied" if apply else "previewed"
        emit(
            state,
            receipt,
            f"Trace-fact page {mode}: {receipt.project}/{receipt.provider_run_id} "
            f"({receipt.inspected} traces, {receipt.complete} complete, {receipt.partial} partial; "
            f"next cursor: {receipt.next_cursor or 'done'})",
        )


__all__ = ["TraceFactBackfillPage", "backfill_verifiers_trace_page", "register"]
