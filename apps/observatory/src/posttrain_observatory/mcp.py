"""Compact MCP tools backed by the same Observatory service."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from posttrain.tracking import RunQuery

from .models import MetricSeriesQuery, RunLocator, SemanticSummaryRequest, ViewMode
from .service import ObservatoryService


def create_mcp(service: ObservatoryService) -> FastMCP:
    server = FastMCP(
        "Posttrain Observatory",
        instructions="Inspect post-training runs through curated job views and bounded evidence queries.",
        stateless_http=True,
        json_response=True,
        streamable_http_path="/",
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
        """Return bounded Verifiers population aggregates and trace tips."""
        return (await service.get_trace_evaluation_view(RunLocator(source_id=source_id, run_id=run_id))).model_dump(
            mode="json"
        )

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

    return server


__all__ = ["create_mcp"]
