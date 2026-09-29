"""Compact MCP tools backed by the same Observatory service."""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer
from posttrain.tracking import RunQuery

from .models import MetricSeriesQuery, RunLocator, SemanticSummaryRequest, ViewMode
from .run_notes import NoteAddRequest, NoteDeleteRequest, NoteReviseRequest
from .semantic_layer.query import SemanticQuery, SqlQuery
from .service import ObservatoryService


def create_mcp(service: ObservatoryService) -> MCPServer:
    server = MCPServer(
        "Posttrain Observatory",
        instructions="Inspect post-training runs through curated job views and bounded evidence queries.",
    )

    @server.tool()
    async def list_runs(project_id: str | None = None, limit: int = 50) -> list[dict[str, object]]:
        """List compact source-qualified run summaries."""
        return [
            run.model_dump(mode="json") for run in await service.list_runs(RunQuery(project_id=project_id, limit=limit))
        ]

    @server.tool()
    async def get_run_view(source_id: str, run_id: str, mode: ViewMode = "auto") -> dict[str, object]:
        """Get the curated job view or deterministic generic fallback."""
        return (await service.get_run_view_response(RunLocator(source_id=source_id, run_id=run_id), mode)).model_dump(
            mode="json"
        )

    @server.tool()
    async def get_serving_capacity_view(
        work_package_id: str,
        project_id: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, object]:
        """Compare serving contenders and return the strict Pareto frontier when evidence is comparable."""
        return (
            await service.get_serving_capacity_view(
                work_package_id,
                project_id=project_id,
                source_id=source_id,
            )
        ).model_dump(mode="json")

    @server.tool()
    async def list_run_metrics(source_id: str, run_id: str) -> dict[str, object]:
        """List metric namespaces and names without fetching all points."""
        return (await service.list_run_metrics(RunLocator(source_id=source_id, run_id=run_id))).model_dump(mode="json")

    @server.tool()
    async def get_system_metrics(source_id: str, run_id: str) -> dict[str, object]:
        """Get bounded cross-job system and tracking telemetry."""

        return (await service.get_system_metrics(RunLocator(source_id=source_id, run_id=run_id))).model_dump(
            mode="json"
        )

    @server.tool()
    async def get_metric_series(
        source_id: str, run_id: str, names: list[str], max_points: int = 200
    ) -> dict[str, object]:
        """Fetch explicitly named, bounded metric series."""
        query = MetricSeriesQuery(names=tuple(names), max_points=max_points)
        return (await service.get_metric_series(RunLocator(source_id=source_id, run_id=run_id), query)).model_dump(
            mode="json"
        )

    @server.tool()
    async def get_run_alerts(source_id: str, run_id: str) -> list[dict[str, object]]:
        """Return deterministic job-health conditions that fired."""
        return [
            alert.model_dump(mode="json")
            for alert in await service.get_run_alerts(RunLocator(source_id=source_id, run_id=run_id))
        ]

    @server.tool()
    async def get_trace_evaluation_view(source_id: str, run_id: str) -> dict[str, object]:
        """Return bounded Verifiers population aggregates without trace rows."""
        return (
            await service.get_trace_evaluation_view(
                RunLocator(source_id=source_id, run_id=run_id), include_traces=False
            )
        ).model_dump(mode="json")

    @server.tool()
    async def get_trace_summary_page(
        source_id: str,
        run_id: str,
        cursor: str | None = None,
        limit: int = 100,
    ) -> dict[str, object]:
        """Return one bounded page of trace summaries for investigation."""
        return (
            await service.get_trace_summary_page(
                RunLocator(source_id=source_id, run_id=run_id), cursor=cursor, limit=limit
            )
        ).model_dump(mode="json")

    @server.tool()
    async def get_trace_detail(source_id: str, run_id: str, trace_id: str) -> dict[str, object]:
        """Return one redacted trace with transcript and reward components."""
        return (await service.get_trace_detail(RunLocator(source_id=source_id, run_id=run_id), trace_id)).model_dump(
            mode="json"
        )

    @server.tool()
    async def compare_runs(source_id: str, run_ids: list[str]) -> dict[str, object]:
        """Compare runs only when their exact job kind and schema match."""
        locators = tuple(RunLocator(source_id=source_id, run_id=run_id) for run_id in run_ids)
        return (await service.compare_runs(locators)).model_dump(mode="json")

    @server.tool()
    async def summarize_run(source_id: str, run_id: str, scope: str = "run") -> dict[str, object]:
        """Explicitly request a cited, non-authoritative semantic summary."""
        request = SemanticSummaryRequest(scope=scope)  # type: ignore[arg-type]
        return (await service.summarize_run(RunLocator(source_id=source_id, run_id=run_id), request)).model_dump(
            mode="json"
        )

    @server.tool()
    async def get_job_view_schema(job_kind: str) -> dict[str, object]:
        """Return the versioned deterministic telemetry definition for a job kind."""
        return service.get_job_telemetry_schema(job_kind).model_dump(mode="json")

    @server.tool()
    async def get_run_card(source_id: str, run_id: str) -> dict[str, object]:
        """The run's card: its job kind's note template rendered from recorded settings and results."""
        return (await service.run_card(RunLocator(source_id=source_id, run_id=run_id))).model_dump(mode="json")

    @server.tool()
    async def list_run_notes(source_id: str, run_id: str, kind: str | None = None) -> list[dict[str, object]]:
        """Notes on a run (latest revision of each, newest first), each with its rendered text."""
        notes = await service.run_notes(RunLocator(source_id=source_id, run_id=run_id), kind=kind)
        return [note.model_dump(mode="json") for note in notes]

    @server.tool()
    async def get_run_note_history(source_id: str, run_id: str, note_id: str) -> list[dict[str, object]]:
        """Every revision of one note, oldest first."""
        history = await service.run_note_history(RunLocator(source_id=source_id, run_id=run_id), note_id)
        return [note.model_dump(mode="json") for note in history]

    @server.tool()
    async def render_note_preview(source_id: str, run_id: str, body_md: str) -> dict[str, object]:
        """Render note Markdown for a run without saving it; check `unresolved` before adding the note.

        Cite numbers through data blocks instead of typing them: a fenced block ```sql <name>``` holding one
        Doris SQL SELECT over the tables from describe_semantics (an optional first line `-- runs: self, <run id>`
        scopes it; the default is this run), then `{{<name>.<column>}}` inline, or a ```value```, ```table``` or
        ```chart``` block with `data: <name>`.
        `{{run.<dimension>}}` shows a setting of this run and `[[run:<id>]]` links another run."""
        rendered = await service.render_note(RunLocator(source_id=source_id, run_id=run_id), body_md)
        return rendered.model_dump(mode="json")

    if service.notes.writes:

        @server.tool()
        async def add_run_note(
            source_id: str,
            run_id: str,
            kind: str,
            body_md: str,
            title: str | None = None,
            note_id: str | None = None,
        ) -> dict[str, object]:
            """Attach a Markdown note to a run (kind: summary, finding, correction, decision, ...).

            Preview it with render_note_preview first. Passing a note_id makes a retried add a no-op."""
            request = NoteAddRequest(kind=kind, body_md=body_md, title=title, note_id=note_id)
            note = await service.add_run_note(RunLocator(source_id=source_id, run_id=run_id), request, source="mcp")
            return note.model_dump(mode="json")

        @server.tool()
        async def revise_run_note(
            source_id: str,
            run_id: str,
            note_id: str,
            expected_revision: int,
            body_md: str,
            kind: str | None = None,
            title: str | None = None,
        ) -> dict[str, object]:
            """Save a new revision of a note; expected_revision is the revision you read (history is kept)."""
            request = NoteReviseRequest(expected_revision=expected_revision, body_md=body_md, kind=kind, title=title)
            locator = RunLocator(source_id=source_id, run_id=run_id)
            return (await service.revise_run_note(locator, note_id, request, source="mcp")).model_dump(mode="json")

        @server.tool()
        async def delete_run_note(
            source_id: str, run_id: str, note_id: str, expected_revision: int
        ) -> dict[str, object]:
            """Hide a note behind a tombstone revision; its history is kept."""
            request = NoteDeleteRequest(expected_revision=expected_revision)
            locator = RunLocator(source_id=source_id, run_id=run_id)
            return (await service.delete_run_note(locator, note_id, request, source="mcp")).model_dump(mode="json")

    @server.tool()
    async def describe_semantics(job_kinds: list[str] | None = None) -> dict[str, object]:
        """Call this before query_semantics. Lists the SQL tables (runs, updates, rollouts, and Trackio's raw
        tables), their dimensions and measures, and metrics; job_kinds narrows measures to what those provide."""
        return (await service.describe_semantics(job_kinds=tuple(job_kinds or ()))).model_dump(mode="json")

    @server.tool()
    async def query_semantics(
        sql: str | None = None,
        measures: list[str] | None = None,
        by: list[str] | None = None,
        where: dict[str, Any] | None = None,
        runs: list[str] | dict[str, Any] | None = None,
        order_by: list[str] | None = None,
        limit: int = 1000,
        source_id: str | None = None,
    ) -> dict[str, object]:
        """Answer a question about runs, inside the tracking storage. Prefer sql: one read-only Doris SQL SELECT
        over the tables from describe_semantics (every run of the project; runs, as ids or run-dimension filters
        such as {"run.job_kind": "train.sampo"}, narrows them). Or the short form: measures (name or
        name:aggregation), by dimensions, where ({dimension: value, [any of], ">= n" or a "*" wildcard}); it
        compiles to SQL, returned in the result. source_id selects the source (with discovered Trackio projects,
        the project name); omitted, the Observatory's configured default source is read."""
        scope = tuple(runs) if isinstance(runs, list) else runs
        if sql is not None:
            query: SemanticQuery | SqlQuery = SqlQuery(sql=sql, runs=scope)
        else:
            query = SemanticQuery(
                measures=tuple(measures or ()),
                by=tuple(by or ()),
                where=where or {},
                runs=scope,
                order_by=tuple(order_by or ()),
                limit=limit,
            )
        return (await service.query_semantics(query, source_id=source_id)).model_dump(mode="json")

    return server


__all__ = ["create_mcp"]
