"""Versioned FastAPI transport over the single Observatory service."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from posttrain.tracking import RunQuery
from pydantic import BeforeValidator, Field

from .mcp import create_mcp
from .models import (
    ErrorResponse,
    ExportRequest,
    MetricSeriesQuery,
    ObservatoryModel,
    RunLocator,
    RunViewResponse,
    SemanticSummaryRequest,
    SourceRefreshStatus,
    ViewMode,
)
from .service import ObservatoryService
from .settings import ObservatorySettings


class CompareRequest(ObservatoryModel):
    run_keys: Annotated[
        tuple[str, ...],
        BeforeValidator(lambda value: tuple(value) if isinstance(value, list) else value),
    ] = Field(min_length=1, max_length=12)


def _locator(run_key: str) -> RunLocator:
    return RunLocator.from_key(run_key)


def create_http_app(
    service: ObservatoryService,
    settings: ObservatorySettings | None = None,
) -> FastAPI:
    settings = settings or ObservatorySettings()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await service.start_source_discovery()
        try:
            yield
        finally:
            await service.stop_source_discovery()

    app = FastAPI(
        title="Posttrain Observatory",
        version="0.1.0",
        description="Provider-neutral, job-aware post-training evidence views.",
        lifespan=lifespan,
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.exception_handler(LookupError)
    async def lookup_error(request: Request, error: LookupError) -> JSONResponse:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        body = ErrorResponse(code="not_found", message=str(error), request_id=request_id)
        return JSONResponse(status_code=404, content=body.model_dump(mode="json"))

    @app.exception_handler(ValueError)
    async def value_error(request: Request, error: ValueError) -> JSONResponse:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        body = ErrorResponse(code="invalid_request", message=str(error), request_id=request_id)
        return JSONResponse(status_code=422, content=body.model_dump(mode="json"))

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready() -> dict[str, object]:
        sources = await service.list_sources()
        return {
            "status": "ready" if any(source.state == "healthy" for source in sources) else "degraded",
            "sources": [source.model_dump(mode="json") for source in sources],
            "source_refresh": service.source_refresh_status().model_dump(mode="json"),
        }

    @app.get("/version")
    async def version() -> dict[str, object]:
        return {
            "product": "posttrain-observatory",
            "version": "0.1.0",
            "api_schema": 1,
            "job_schemas": {definition.job_kind: definition.schema_version for definition in service.list_job_kinds()},
        }

    @app.get("/api/v1/sources")
    async def sources() -> list[dict[str, object]]:
        return [source.model_dump(mode="json") for source in await service.list_sources()]

    @app.post("/api/v1/sources/refresh")
    async def refresh_sources() -> SourceRefreshStatus:
        return await service.refresh_sources()

    @app.get("/api/v1/runs")
    async def runs(
        project_id: str | None = None,
        work_package_id: str | None = None,
        job_kind: tuple[str, ...] = Query(default=()),
        status: tuple[str, ...] = Query(default=()),
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[dict[str, object]]:
        query = RunQuery(
            project_id=project_id,
            work_package_id=work_package_id,
            job_kinds=job_kind,
            statuses=status,  # type: ignore[arg-type]
            limit=limit,
        )
        return [run.model_dump(mode="json") for run in await service.list_runs(query)]

    @app.get("/api/v1/runs/locate")
    async def locate_run(run_id: str = Query(min_length=1)) -> list[dict[str, object]]:
        return [run.model_dump(mode="json") for run in await service.locate_run(run_id)]

    @app.get("/api/v1/runs/{run_key}/view")
    async def run_view(
        run_key: str,
        mode: ViewMode = "auto",
        metric: tuple[str, ...] = Query(default=()),
    ) -> RunViewResponse:
        return await service.get_run_view_response(_locator(run_key), mode, metric)

    @app.get("/api/v1/runs/{run_key}/metrics")
    async def metrics(run_key: str) -> dict[str, object]:
        return (await service.list_run_metrics(_locator(run_key))).model_dump(mode="json")

    @app.get("/api/v1/runs/{run_key}/system-metrics")
    async def system_metrics(run_key: str) -> dict[str, object]:
        return (await service.get_system_metrics(_locator(run_key))).model_dump(mode="json")

    @app.post("/api/v1/runs/{run_key}/metric-series")
    async def metric_series(run_key: str, query: MetricSeriesQuery) -> dict[str, object]:
        return (await service.get_metric_series(_locator(run_key), query)).model_dump(mode="json")

    @app.get("/api/v1/runs/{run_key}/alerts")
    async def alerts(run_key: str) -> list[dict[str, object]]:
        return [alert.model_dump(mode="json") for alert in await service.get_run_alerts(_locator(run_key))]

    @app.get("/api/v1/runs/{run_key}/delta")
    async def delta(run_key: str, cursor: str | None = None) -> dict[str, object]:
        return (await service.get_run_delta(_locator(run_key), cursor)).model_dump(mode="json")

    @app.post("/api/v1/runs/compare")
    async def compare(request: CompareRequest) -> dict[str, object]:
        return (await service.compare_runs(tuple(_locator(key) for key in request.run_keys))).model_dump(mode="json")

    @app.get("/api/v1/runs/{run_key}/traces-evaluation")
    async def traces_evaluation(run_key: str) -> dict[str, object]:
        return (await service.get_trace_evaluation_view(_locator(run_key))).model_dump(mode="json")

    @app.get("/api/v1/runs/{run_key}/traces/{trace_id}")
    async def trace_detail(run_key: str, trace_id: str) -> dict[str, object]:
        return (await service.get_trace_detail(_locator(run_key), trace_id)).model_dump(mode="json")

    @app.post("/api/v1/runs/{run_key}/semantic-summary")
    async def semantic_summary(run_key: str, request: SemanticSummaryRequest) -> dict[str, object]:
        return (await service.summarize_run(_locator(run_key), request)).model_dump(mode="json")

    @app.get("/api/v1/serving-capacity/work-packages/{work_package_id:path}")
    async def serving_capacity_work_package(
        work_package_id: str,
        project_id: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, object]:
        return (
            await service.get_serving_capacity_view(
                work_package_id,
                project_id=project_id,
                source_id=source_id,
            )
        ).model_dump(mode="json")

    @app.get("/api/v1/work-packages/{work_package_id:path}")
    async def work_package(
        work_package_id: str,
        project_id: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, object]:
        return (
            await service.get_work_package_view(
                work_package_id,
                project_id=project_id,
                source_id=source_id,
            )
        ).model_dump(mode="json")

    @app.get("/api/v1/job-kinds")
    async def job_kinds() -> list[dict[str, object]]:
        return [definition.model_dump(mode="json") for definition in service.list_job_kinds()]

    @app.get("/api/v1/job-kinds/{job_kind}")
    async def job_kind(job_kind: str) -> dict[str, object]:
        return service.get_job_telemetry_schema(job_kind).model_dump(mode="json")

    @app.post("/api/v1/exports")
    async def export(request: ExportRequest) -> JSONResponse:
        if request.view == "serving_capacity":
            if request.work_package_id is None:
                return JSONResponse(
                    status_code=422,
                    content={"message": "serving_capacity export requires work_package_id"},
                )
            view = await service.get_serving_capacity_view(
                request.work_package_id,
                project_id=request.project_id,
                source_id=request.source_id,
            )
            return JSONResponse(
                content={
                    "format": request.format,
                    "view": view.model_dump(mode="json"),
                }
            )
        payload = []
        for key in request.run_keys:
            payload.append((await service.get_run_view_response(_locator(key))).model_dump(mode="json"))
        return JSONResponse(content={"format": request.format, "runs": payload})

    app.mount("/mcp", create_mcp(service).streamable_http_app(), name="mcp")

    frontend = Path(settings.frontend_dir) if settings.frontend_dir else Path(__file__).parents[2] / "frontend" / "dist"
    if frontend.exists():
        assets = frontend / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str) -> FileResponse:
            candidate = frontend / path
            return FileResponse(candidate if candidate.is_file() else frontend / "index.html")

    def openapi() -> dict[str, Any]:
        if app.openapi_schema is not None:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        json_value = schema.get("components", {}).get("schemas", {}).get("JsonValue")
        if json_value is not None:
            schema["components"]["schemas"]["JsonValue"] = {
                "title": "JsonValue",
                "description": "Any JSON-compatible value.",
            }
        app.openapi_schema = schema
        return schema

    app.openapi = openapi

    return app


__all__ = ["CompareRequest", "create_http_app"]
