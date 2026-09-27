"""`posttrain query`: ask the Observatory's semantic layer about runs."""

from __future__ import annotations

import asyncio
import csv
import importlib
import io
import json
import re
from pathlib import Path
from typing import Annotated, Any

import click
import typer
import yaml
from posttrain.common import ContractError

from ..context import CliState
from ..output import emit
from ..tracking_config import project_observatory_settings

_FILTER = re.compile(r"^\s*([a-z][a-z0-9_.]*)\s*(>=|<=|!=|>|<|=)\s*(.*?)\s*$")
FORMAT_CHOICE = click.Choice(("table", "csv", "json"))

_LIST_HELP = "comma-separated; repeatable"


def parse_filter(text: str) -> tuple[str, Any]:
    """`name=value`, `name=a,b` (any of), `name>=10`, or `name=prefix-*`."""

    match = _FILTER.match(text)
    if match is None:
        raise ContractError(f"filter {text!r} must look like name=value, name>=number or name=a,b")
    name, operator, value = match.groups()
    if operator != "=":
        return name, f"{operator} {value}"
    if "," in value:
        return name, [item.strip() for item in value.split(",") if item.strip()]
    return name, value


def _split(values: list[str] | None) -> tuple[str, ...]:
    return tuple(item.strip() for value in values or () for item in value.split(",") if item.strip())


def parse_runs(values: list[str] | None) -> tuple[str, ...] | dict[str, Any] | None:
    """Run ids, or run-dimension filters when any entry is a filter."""

    if not values:
        return None
    if any(_FILTER.match(value) for value in values):
        if not all(_FILTER.match(value) for value in values):
            raise ContractError("--runs takes either run ids or run-dimension filters, not both")
        return dict(parse_filter(value) for value in values)
    return _split(values) or None


def build_query(
    *,
    measures: list[str] | None,
    by: list[str] | None,
    where: list[str] | None,
    runs: list[str] | None,
    order_by: list[str] | None,
    limit: int | None,
    sql: str | None,
    file: Path | None,
) -> dict[str, Any]:
    """The JSON form of a semantic or SQL query from command-line options or a query file."""

    if file is not None:
        if measures or by or where or sql:
            raise ContractError("--file cannot be combined with --measures, --by, --where or --sql")
        loaded = yaml.safe_load(file.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ContractError(f"{file} must hold one query mapping")
        query: dict[str, Any] = dict(loaded)
        if runs:
            query["runs"] = parse_runs(runs)
        return query
    scope = parse_runs(runs)
    if sql is not None:
        if measures or by or where:
            raise ContractError("--sql cannot be combined with --measures, --by or --where")
        return {"sql": sql, "runs": scope} if scope is not None else {"sql": sql}
    names = _split(measures)
    if not names:
        raise ContractError("give --measures, --sql or --file")
    query = {"measures": list(names), "by": list(_split(by)), "where": dict(parse_filter(item) for item in where or ())}
    if scope is not None:
        query["runs"] = list(scope) if isinstance(scope, tuple) else scope
    if order_by:
        query["order_by"] = list(_split(order_by))
    if limit is not None:
        query["limit"] = limit
    return query


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_table(result: dict[str, Any]) -> str:
    headers = [column["name"] for column in result["columns"]]
    rows = [[format_value(value) for value in row] for row in result["rows"]]
    widths = [max([len(header), *(len(row[index]) for row in rows)]) for index, header in enumerate(headers)]
    numeric = [
        all(
            isinstance(row[index], int | float) and not isinstance(row[index], bool)
            for row in result["rows"]
            if row[index] is not None
        )
        for index in range(len(headers))
    ]

    def line(cells: list[str]) -> str:
        return "  ".join(
            cell.rjust(width) if right else cell.ljust(width)
            for cell, width, right in zip(cells, widths, numeric, strict=True)
        ).rstrip()

    lines = [line(headers), line(["-" * width for width in widths]), *(line(row) for row in rows)]
    lines += _result_notes(result)
    return "\n".join(lines)


def render_csv(result: dict[str, Any]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(column["name"] for column in result["columns"])
    writer.writerows(["" if value is None else value for value in row] for row in result["rows"])
    return buffer.getvalue().rstrip("\n")


def _result_notes(result: dict[str, Any]) -> list[str]:
    engine = f" on {result['engine']}" if result.get("engine") else ""
    notes = [f"\n{len(result['rows'])} row(s) at {result['grain']} grain{engine}"]
    if result.get("truncated"):
        notes.append("truncated: more rows exist; raise --limit or narrow the query")
    if result.get("grain") != "sql" and result.get("sql"):
        notes += ["", "SQL:", *(f"  {line}" for line in result["sql"].splitlines())]
    return notes


def render_description(description: dict[str, Any]) -> str:
    lines = [f"job kinds: {', '.join(description['job_kinds']) or 'all'}", "", "SQL tables:"]
    for table in description["sql_tables"]:
        lines += [f"  {table['name']}: {table['description']}", f"    {', '.join(table['columns'])}"]
    for entity in description["entities"]:
        name = entity["name"]
        lines += ["", f"{name} dimensions:"]
        lines += [
            f"  {item['name']} ({item['type']}): {item['description']}"
            for item in description["dimensions"]
            if item["entity"] == name
        ]
        measures = [item for item in description["measures"] if item["entity"] == name]
        if measures:
            lines.append(f"{name} measures:")
        for measure in measures:
            unit = f" [{measure['unit']}]" if measure.get("unit") else ""
            lines.append(f"  {measure['name']}{unit} ({measure['aggregation']}): {measure['description']}")
        metrics = [metric for metric in description["metrics"] if metric["entity"] == name]
        lines += [f"  {metric['name']} = {metric['formula']}" for metric in metrics]
    lines += ["", *description["notes"]]
    return "\n".join(lines)


def _service(state: CliState) -> tuple[Any, Any]:
    layout = state.layout()
    if layout.tracking == "none":
        raise ContractError(
            "queries read project tracking; set tracking to 'trackio' or 'wandb' in .posttrain/project.toml"
        )
    try:
        observatory = importlib.import_module("posttrain_observatory")
        semantic = importlib.import_module("posttrain_observatory.semantic_layer")
    except ImportError as error:
        raise RuntimeError(
            "Observatory is not installed; run `uv add 'posttrain[observatory]'` "
            "or install the posttrain-observatory package"
        ) from error
    settings = project_observatory_settings(layout, observatory.ObservatorySettings)
    return observatory.create_service(settings), semantic


def register(app: typer.Typer) -> None:
    query_app = typer.Typer(
        rich_markup_mode=None,
        invoke_without_command=True,
        help="ask the semantic layer about runs: measures by dimensions, or read-only SQL",
    )
    app.add_typer(query_app, name="query")

    @query_app.callback()
    def query_cmd(
        ctx: typer.Context,
        measures: Annotated[
            list[str] | None, typer.Option("--measures", "-m", help=f"measure or metric[:aggregation]; {_LIST_HELP}")
        ] = None,
        by: Annotated[list[str] | None, typer.Option("--by", help=f"dimensions to group by; {_LIST_HELP}")] = None,
        where: Annotated[
            list[str] | None,
            typer.Option("--where", "-w", help="filter: name=value, name=a,b, name>=10, name=prefix-*; repeatable"),
        ] = None,
        runs: Annotated[
            list[str] | None,
            typer.Option(
                "--runs", "-r", help=f"run ids, or run-dimension filters such as run.work_package=x; {_LIST_HELP}"
            ),
        ] = None,
        order_by: Annotated[
            list[str] | None, typer.Option("--order-by", help=f"columns, prefix - for descending; {_LIST_HELP}")
        ] = None,
        limit: Annotated[int | None, typer.Option("--limit", min=1)] = None,
        sql: Annotated[str | None, typer.Option("--sql", help="one read-only SELECT over the semantic tables")] = None,
        file: Annotated[
            Path | None, typer.Option("--file", "-f", help="a YAML or JSON query", exists=True, dir_okay=False)
        ] = None,
        output_format: Annotated[str, typer.Option("--format", click_type=FORMAT_CHOICE)] = "table",
    ) -> None:
        if ctx.invoked_subcommand is not None:
            return
        state: CliState = ctx.obj
        query = build_query(
            measures=measures,
            by=by,
            where=where,
            runs=runs,
            order_by=order_by,
            limit=limit,
            sql=sql,
            file=file,
        )
        service, semantic = _service(state)
        model = semantic.SqlQuery if "sql" in query else semantic.SemanticQuery
        result = asyncio.run(service.query_semantics(model.model_validate(query))).model_dump(mode="json")
        if output_format == "json" or state.json_output:
            emit(state, result, json.dumps(result, indent=2))
        elif output_format == "csv":
            print(render_csv(result), flush=True)
        else:
            print(render_table(result), flush=True)

    @query_app.command("describe", help="list the SQL tables, dimensions, measures and metrics")
    def describe_cmd(
        ctx: typer.Context,
        job_kinds: Annotated[list[str] | None, typer.Option("--job-kind", help=_LIST_HELP)] = None,
    ) -> None:
        state: CliState = ctx.obj
        service, _ = _service(state)
        description = asyncio.run(service.describe_semantics(job_kinds=_split(job_kinds))).model_dump(mode="json")
        emit(state, description, render_description(description))


__all__ = ["build_query", "parse_filter", "parse_runs", "register", "render_csv", "render_table"]
