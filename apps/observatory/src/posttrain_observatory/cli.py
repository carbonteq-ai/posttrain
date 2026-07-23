"""Command-line and server entry points for Observatory."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import uvicorn
from posttrain.tracking import RunQuery

from .composition import create_service
from .http import create_http_app
from .mcp import create_mcp
from .models import RunLocator, SemanticSummaryRequest
from .settings import ObservatorySettings
from .telemetry import DEFAULT_TELEMETRY_DEFINITIONS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="posttrain-observatory")
    commands = parser.add_subparsers(dest="command", required=True)
    schema = commands.add_parser("schema", help="print or export product schemas")
    schema.add_argument("job_kind", nargs="?", choices=tuple(sorted(DEFAULT_TELEMETRY_DEFINITIONS)))
    schema.add_argument("--openapi", type=Path)
    schema.add_argument("--mcp", type=Path)
    serve = commands.add_parser("serve", help="serve UI, API, and Streamable HTTP MCP")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    commands.add_parser("mcp", help="serve MCP over stdio")
    runs = commands.add_parser("runs", help="list runs")
    runs.add_argument("--project-id")
    runs.add_argument("--limit", type=int, default=50)
    run = commands.add_parser("run", help="show one run view")
    run.add_argument("source_id")
    run.add_argument("run_id")
    run.add_argument("--mode", choices=("auto", "job", "generic"), default="auto")
    package = commands.add_parser("work-package", help="show a work-package view")
    package.add_argument("work_package_id")
    package.add_argument("--project-id")
    summary = commands.add_parser("summarize", help="generate a grounded semantic summary")
    summary.add_argument("source_id")
    summary.add_argument("run_id")
    summary.add_argument("--scope", choices=("run", "metrics", "evaluation", "trace"), default="run")
    return parser


def _json(value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    print(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            default=lambda item: item.model_dump(mode="json"),
        )
    )


def _mcp_schema(server: Any) -> dict[str, object]:
    return {
        "schema_version": 1,
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.parameters,
                "outputSchema": tool.fn_metadata.output_schema,
            }
            for tool in server._tool_manager.list_tools()
        ],
    }


async def _run_command(args: argparse.Namespace, settings: ObservatorySettings) -> int:
    service = create_service(settings)
    if args.command == "runs":
        _json(await service.list_runs(RunQuery(project_id=args.project_id, limit=args.limit)))
    elif args.command == "run":
        _json(await service.get_run_view_response(RunLocator(source_id=args.source_id, run_id=args.run_id), args.mode))
    elif args.command == "work-package":
        _json(await service.get_work_package_view(args.work_package_id, project_id=args.project_id))
    elif args.command == "summarize":
        _json(
            await service.summarize_run(
                RunLocator(source_id=args.source_id, run_id=args.run_id),
                SemanticSummaryRequest(scope=args.scope),
            )
        )
    else:
        raise AssertionError(f"unhandled async command: {args.command}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = ObservatorySettings.from_env()
    if args.command == "schema":
        if args.job_kind:
            print(DEFAULT_TELEMETRY_DEFINITIONS[args.job_kind].model_dump_json(indent=2))
            return 0
        service = create_service(settings)
        if not args.openapi and not args.mcp:
            raise SystemExit("schema requires JOB_KIND or --openapi/--mcp")
        if args.openapi:
            args.openapi.write_text(
                json.dumps(create_http_app(service, settings).openapi(), indent=2, sort_keys=True) + "\n"
            )
        if args.mcp:
            args.mcp.write_text(json.dumps(_mcp_schema(create_mcp(service)), indent=2, sort_keys=True) + "\n")
        return 0
    if args.command == "serve":
        host = args.host or settings.host
        port = args.port or settings.port
        uvicorn.run(create_http_app(create_service(settings), settings), host=host, port=port)
        return 0
    if args.command == "mcp":
        create_mcp(create_service(settings)).run(transport="stdio")
        return 0
    return asyncio.run(_run_command(args, settings))


if __name__ == "__main__":
    raise SystemExit(main())
