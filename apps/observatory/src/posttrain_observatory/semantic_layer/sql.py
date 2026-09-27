"""Read-only SQL over the semantic tables of the runs in scope.

Each query gets a fresh in-memory SQLite database holding only the scoped data:
``runs``, ``updates``, ``rollouts``, ``eval_tasks``, ``load_levels`` and, when
asked for by name, ``raw_metrics``. While the user's statement runs, SQLite's
authorizer allows only reading and functions, and a progress handler stops it
after the time limit.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Mapping, Sequence
from typing import Any

from .execute import (
    MAX_SERIES_POINTS,
    SERIES_BATCH,
    SemanticReader,
    _Context,
    applies,
    load_rollouts,
    load_run_metrics,
    load_updates,
    load_view_rows,
    per_run,
    resolve_runs,
    run_row,
)
from .model import SemanticModel
from .query import QueryError, ResultColumn, SemanticResult, SqlQuery

TABLES = ("runs", "updates", "rollouts", "eval_tasks", "load_levels", "raw_metrics")
DEFAULT_TIMEOUT_SECONDS = 5.0
_ALLOWED_ACTIONS = frozenset(
    {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, 33}
)  # 33: WITH RECURSIVE


def column_name(semantic_name: str) -> str:
    """`run.learning_rate` -> `learning_rate`; `update.step` -> `step`."""
    return semantic_name.split(".", 1)[1] if "." in semantic_name else semantic_name


def tables_named(sql: str) -> tuple[str, ...]:
    return tuple(table for table in TABLES if re.search(rf"\b{table}\b", sql, flags=re.IGNORECASE))


async def build_database(
    model: SemanticModel,
    reader: SemanticReader,
    query: SqlQuery,
    *,
    max_runs: int,
) -> tuple[sqlite3.Connection, _Context, list[str]]:
    context = _Context(model=model, reader=reader)
    runs = await resolve_runs(context, query.runs, {}, max_runs=max_runs)
    wanted = dict(query.load) if query.load is not None else {table: () for table in tables_named(query.sql)}
    wanted.setdefault("runs", ())
    unknown = sorted(set(wanted) - set(TABLES))
    if unknown:
        raise QueryError(f"unknown tables {', '.join(unknown)}; available: {', '.join(TABLES)}")
    connection = sqlite3.connect(":memory:")
    run_dimensions = [dimension for dimension in model.dimensions if dimension.entity == "run"]
    run_measures = [
        measure
        for measure in model.measures
        if measure.entity == "run"
        and measure.source.kind == "metric_series"
        and any(applies(measure, summary.job_kind) for summary in runs)
    ]

    async def run_record(summary: Any) -> dict[str, Any]:
        return {
            **await run_row(context, summary, run_dimensions),
            **await load_run_metrics(context, summary, run_measures),
        }

    run_rows = await per_run(runs, run_record)
    run_columns = [
        "id",
        *(column_name(d.name) for d in run_dimensions if d.name != "run.id"),
        "duration_seconds",
        *(m.name for m in run_measures),
    ]
    _create(
        connection,
        "runs",
        run_columns,
        [{"id": row["run.id"], **{column_name(k): v for k, v in row.items() if k != "run.id"}} for row in run_rows],
    )

    def measures_for(entity: str) -> list[Any]:
        names = set(wanted.get(_table_of(entity), ()))
        return [
            measure
            for measure in model.measures
            if measure.entity == entity
            and (not names or measure.name in names)
            and any(applies(measure, summary.job_kind) for summary in runs)
        ]

    if "updates" in wanted:
        measures = measures_for("update")
        rows = [
            row
            for loaded in await per_run(runs, lambda summary: load_updates(context, summary, measures))
            for row in loaded
        ]
        _create(
            connection,
            "updates",
            ["run_id", "step", "time", *(m.name for m in measures)],
            [
                {
                    "run_id": r["run.id"],
                    "step": r["update.step"],
                    "time": r["update.time"],
                    **{m.name: r.get(m.name) for m in measures},
                }
                for r in rows
            ],
        )
    if "rollouts" in wanted:
        measures = measures_for("rollout")
        dimensions = [d for d in model.dimensions if d.entity == "rollout"]
        rows = [
            row
            for loaded in await per_run(runs, lambda summary: load_rollouts(context, summary, measures, dimensions))
            for row in loaded
        ]
        value_columns: list[str] = []
        for measure in measures:
            value_columns += (
                [measure.name]
                if measure.source.name == "trace_count"
                else [measure.name, f"{measure.name}_sum", f"{measure.name}_count"]
            )
        table_rows = []
        for row in rows:
            record: dict[str, Any] = {
                "run_id": row["run.id"],
                **{column_name(d.name): row.get(d.name) for d in dimensions},
            }
            for measure in measures:
                if measure.source.name == "trace_count":
                    record[measure.name] = row.get("__traces")
                    continue
                total, count = row.get(f"{measure.name}__sum"), row.get(f"{measure.name}__count")
                record[f"{measure.name}_sum"], record[f"{measure.name}_count"] = total, count
                record[measure.name] = (total / count) if total is not None and count else None
            table_rows.append(record)
        _create(
            connection, "rollouts", ["run_id", *(column_name(d.name) for d in dimensions), *value_columns], table_rows
        )
    for entity, table in (("eval_task", "eval_tasks"), ("load_level", "load_levels")):
        if table not in wanted:
            continue
        measures = measures_for(entity)
        dimensions = [d for d in model.dimensions if d.entity == entity]
        providers = [summary for summary in runs if any(applies(measure, summary.job_kind) for measure in measures)]
        rows = [
            row
            for loaded in await per_run(
                providers,
                lambda summary, entity=entity, measures=measures, dimensions=dimensions: load_view_rows(
                    context, summary, entity, measures, dimensions
                ),
            )
            for row in loaded
        ]
        _create(
            connection,
            table,
            ["run_id", *(column_name(d.name) for d in dimensions), *(m.name for m in measures)],
            [
                {
                    "run_id": r["run.id"],
                    **{column_name(d.name): r.get(d.name) for d in dimensions},
                    **{m.name: r.get(m.name) for m in measures},
                }
                for r in rows
            ],
        )
    if "raw_metrics" in wanted:
        names = tuple(wanted["raw_metrics"])
        if not names:
            raise QueryError(
                "raw_metrics needs the metric names to load, for example load: {raw_metrics: [train/rl/entropy]}"
            )
        records = []
        for summary in runs:
            for start in range(0, len(names), SERIES_BATCH):
                series, downsampled = await reader.metric_series(
                    summary.run_id, names[start : start + SERIES_BATCH], max_points=MAX_SERIES_POINTS
                )
                context.downsampled |= downsampled
                for item in series:
                    records.extend(
                        {"run_id": summary.run_id, "metric": item.name, "step": p.step, "value": p.value}
                        for p in item.points
                    )
        _create(connection, "raw_metrics", ["run_id", "metric", "step", "value"], records)
    return connection, context, [summary.run_id for summary in runs]


def _table_of(entity: str) -> str:
    return {"update": "updates", "rollout": "rollouts", "eval_task": "eval_tasks", "load_level": "load_levels"}[entity]


def _create(
    connection: sqlite3.Connection, table: str, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    quoted = ", ".join(f'"{column}"' for column in columns)
    connection.execute(f'CREATE TABLE "{table}" ({quoted})')
    placeholders = ", ".join("?" for _ in columns)
    connection.executemany(
        f'INSERT INTO "{table}" VALUES ({placeholders})',
        [tuple(_sqlite_value(row.get(column)) for column in columns) for row in rows],
    )


def _sqlite_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)) or value is None:
        return value
    return str(value)


def run_readonly(
    connection: sqlite3.Connection,
    sql: str,
    *,
    max_rows: int,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[list[str], list[tuple[Any, ...]], bool]:
    """Run one statement that may only read; return columns, rows and whether rows were cut off."""

    def authorize(action: int, *_: Any) -> int:
        return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY

    deadline = time.monotonic() + timeout_seconds
    connection.set_authorizer(authorize)
    connection.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cursor = connection.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
    except sqlite3.OperationalError as error:
        if "interrupted" in str(error):
            raise QueryError(f"the SQL query ran longer than {timeout_seconds:g} seconds") from error
        raise QueryError(f"SQL error: {error}") from error
    except (sqlite3.DatabaseError, sqlite3.Warning, sqlite3.ProgrammingError) as error:
        raise QueryError(f"SQL error: {error}") from error
    finally:
        connection.set_authorizer(None)
        connection.set_progress_handler(None, 0)
    columns = [description[0] for description in cursor.description or ()]
    return columns, rows[:max_rows], len(rows) > max_rows


async def run_sql_query(
    model: SemanticModel,
    reader: SemanticReader,
    query: SqlQuery,
    *,
    max_runs: int,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> SemanticResult:
    connection, context, run_ids = await build_database(model, reader, query, max_runs=max_runs)
    try:
        columns, rows, truncated = run_readonly(
            connection, query.sql, max_rows=query.max_rows, timeout_seconds=timeout_seconds
        )
    finally:
        connection.close()
    return SemanticResult(
        columns=tuple(ResultColumn(name=name, kind="value") for name in columns),
        rows=tuple(tuple(row) for row in rows),
        grain="sql",
        runs=tuple(run_ids),
        sources={name: tuple(sorted(values)) for name, values in context.sources.items()},
        downsampled=context.downsampled,
        unavailable=tuple(dict.fromkeys(context.unavailable)),
        truncated=truncated,
    )


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "TABLES",
    "build_database",
    "column_name",
    "run_readonly",
    "run_sql_query",
    "tables_named",
]
