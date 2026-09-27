"""Answer queries by running SQL inside the tracking backend's storage.

A statement over the semantic tables gets the views it reads prepended
(`views.assemble`) and runs through the source's `ProjectSql`: Doris on the
shared Trackio server, translated SQLite on local Trackio storage. The short
form compiles to SQL first (`compile.compile_query`).
"""

from __future__ import annotations

from typing import Any

from posttrain.tracking import ProjectSql, ProjectSqlError, ProjectSqlUnavailable

from .compile import compile_query, compile_scope
from .model import SemanticModel
from .query import QueryError, ResultColumn, RunScope, SemanticQuery, SemanticResult, SqlQuery
from .views import assemble

DEFAULT_TIMEOUT_SECONDS = 10.0


async def run_sql(
    model: SemanticModel,
    source: Any,
    sql: str,
    runs: RunScope | None,
    *,
    max_rows: int,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    columns: tuple[ResultColumn, ...] | None = None,
    grain: str = "sql",
) -> SemanticResult:
    """Run one read-only statement over the semantic tables of the runs in scope."""

    if not isinstance(source, ProjectSql):
        raise ProjectSqlUnavailable("this source cannot run SQL in its storage; the query layer needs Trackio")
    final = assemble(model, sql, compile_scope(model, runs))
    try:
        result = await source.project_sql(final, max_rows=max_rows, timeout_seconds=timeout_seconds)
    except ProjectSqlError as error:
        raise QueryError(f"SQL error: {error}") from error
    if columns is None:
        columns = tuple(ResultColumn(name=name, kind="value") for name in result.columns)
    booleans = {index for index, column in enumerate(columns) if column.type == "boolean"}
    rows = tuple(
        tuple(bool(value) if index in booleans and value is not None else value for index, value in enumerate(row))
        for row in result.rows
    )
    return SemanticResult(
        columns=columns,
        rows=rows,
        grain=grain,
        sql=sql,
        engine=result.engine,
        truncated=result.truncated,
    )


async def run_semantic_query(
    model: SemanticModel, source: Any, query: SemanticQuery, *, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
) -> SemanticResult:
    """Answer the short form by compiling it to SQL (with one extra row to detect truncation)."""

    compiled = compile_query(model, query)
    result = await run_sql(
        model,
        source,
        compiled.sql,
        query.runs,
        max_rows=query.limit + 1,
        timeout_seconds=timeout_seconds,
        columns=compiled.columns,
        grain=compiled.grain,
    )
    return result.model_copy(
        update={"rows": result.rows[: query.limit], "truncated": result.truncated or len(result.rows) > query.limit}
    )


async def run_sql_query(
    model: SemanticModel, source: Any, query: SqlQuery, *, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
) -> SemanticResult:
    return await run_sql(model, source, query.sql, query.runs, max_rows=query.max_rows, timeout_seconds=timeout_seconds)


__all__ = ["DEFAULT_TIMEOUT_SECONDS", "run_semantic_query", "run_sql", "run_sql_query"]
