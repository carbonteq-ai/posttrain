"""The short query form (measures by dimensions) and run scopes, compiled to Doris SQL.

SQL over the semantic tables is the one query language; the short form is a
convenience for the command line and agents, and every result carries the SQL
it compiled to.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .model import Dimension, Measure, Metric, SemanticModel
from .query import (
    Condition,
    QueryError,
    ResultColumn,
    RunScope,
    SemanticQuery,
    parse_condition,
    split_measure,
    suggest,
)
from .views import SEMANTIC_TABLES, column_name

_SQL_AGGREGATE = {"mean": "AVG", "sum": "SUM", "min": "MIN", "max": "MAX", "count": "COUNT", "stddev": "STDDEV_SAMP"}
_PERCENTILES = {"p50": 0.5, "p90": 0.9, "p95": 0.95, "p99": 0.99}


@dataclass(frozen=True, slots=True)
class CompiledQuery:
    sql: str
    columns: tuple[ResultColumn, ...]
    grain: str


@dataclass(frozen=True, slots=True)
class CompiledScope:
    """A filter on the project's runs, over columns of the `runs` view."""

    predicate: str
    columns: frozenset[str]


def compile_scope(model: SemanticModel, runs: RunScope | None) -> CompiledScope | None:
    """Run ids, or run-dimension filters such as ``{"run.job_kind": "train.sampo"}``."""

    if runs is None:
        return None
    if isinstance(runs, Mapping):
        predicates, columns = [], set()
        for name, value in runs.items():
            # Bare names are run dimensions ("job_kind"); prefixed names keep their entity.
            key = name if "." in name else f"run.{name}"
            dimension = _dimension(model, key)
            if dimension.entity != "run":
                raise QueryError(f"{key!r} is not a run dimension")
            predicates.append(predicate(dimension, parse_condition(value), qualify=False))
            columns.add(column_name(dimension.name))
        return CompiledScope(" AND ".join(predicates) or "1 = 1", frozenset(columns))
    ids = tuple(runs)
    if not ids:
        raise QueryError("give at least one run id")
    return CompiledScope(f"id IN ({', '.join(_text(item) for item in ids)})", frozenset({"id"}))


def compile_query(model: SemanticModel, query: SemanticQuery) -> CompiledQuery:
    requested = [_resolve(model, value) for value in query.measures]
    conditions = {name: parse_condition(value) for name, value in query.where.items()}
    dimensions = {name: _dimension(model, name) for name in dict.fromkeys([*query.by, *conditions])}
    entities = {item.entity for _, item, _ in requested} | {dimension.entity for dimension in dimensions.values()}
    detailed = entities - {"run"}
    if len(detailed) > 1:
        raise QueryError(f"a query can use one entity besides run; this one uses {', '.join(sorted(detailed))}")
    grain = next(iter(detailed), "run")
    if grain != "run" and any(isinstance(item, Measure) and item.entity == "run" for _, item, _ in requested):
        raise QueryError("run measures can only be queried at run grain")

    selects: list[str] = []
    columns: list[ResultColumn] = []
    for name in query.by:
        dimension = dimensions[name]
        selects.append(f"{_column(dimension)} AS {_alias(name)}")
        columns.append(ResultColumn(name=name, kind="dimension", type=dimension.type, label=dimension.description))
    for column, item, aggregation in requested:
        if isinstance(item, Metric):
            selects.append(f"({item.formula}) AS {_alias(column)}")
            columns.append(ResultColumn(name=column, kind="metric", type="number", unit=item.unit, label=item.label))
        else:
            selects.append(f"{_aggregate(model, item, aggregation)} AS {_alias(column)}")
            columns.append(ResultColumn(name=column, kind="measure", type="number", unit=item.unit, label=item.label))

    table = SEMANTIC_TABLES[grain]
    if grain == "run":
        source = "runs AS r"
    elif query.by and all(dimensions[name].entity == "run" for name in query.by):
        # Keep runs without rows visible, with empty measures.
        source = f"runs AS r LEFT JOIN {table} AS t ON t.run_id = r.id"
    else:
        source = f"{table} AS t JOIN runs AS r ON r.id = t.run_id"
    clauses = [f"SELECT {', '.join(selects)}", f"FROM {source}"]
    predicates = [predicate(dimensions[name], condition) for name, condition in conditions.items()]
    if predicates:
        clauses.append("WHERE " + " AND ".join(predicates))
    if query.by:
        clauses.append("GROUP BY " + ", ".join(_column(dimensions[name]) for name in query.by))
    names = [column.name for column in columns]
    ordering = []
    for item in query.order_by or ((query.by[0],) if query.by else ()):
        name = item.lstrip("-")
        if name not in names:
            raise QueryError(f"cannot order by {name!r}; it is not a result column")
        # Doris sorts NULLs first ascending and last descending; say it explicitly.
        ordering.append(f"{_alias(name)} {'DESC NULLS LAST' if item.startswith('-') else 'ASC NULLS LAST'}")
    if ordering:
        clauses.append("ORDER BY " + ", ".join(ordering))
    clauses.append(f"LIMIT {query.limit + 1}")
    return CompiledQuery(sql="\n".join(clauses), columns=tuple(columns), grain=grain)


def _resolve(model: SemanticModel, value: str) -> tuple[str, Measure | Metric, str]:
    name, aggregation = split_measure(value)
    column = name if aggregation is None else f"{name}_{aggregation}"
    try:
        measure = model.measure(name)
    except KeyError:
        try:
            metric = model.metric(name)
        except KeyError:
            known = [*(item.name for item in model.measures), *(item.name for item in model.metrics)]
            raise QueryError(f"unknown measure or metric {name!r}{suggest(name, known)}") from None
        if aggregation is not None:
            raise QueryError(f"metric {name!r} takes no aggregation") from None
        return column, metric, "formula"
    chosen = aggregation or measure.aggregation
    if chosen not in measure.allowed:
        raise QueryError(f"{name!r} cannot be aggregated with {chosen!r}; allowed: {', '.join(measure.allowed)}")
    return column, measure, chosen


def _dimension(model: SemanticModel, name: str) -> Dimension:
    try:
        return model.dimension(name)
    except KeyError:
        if any(measure.name == name for measure in model.measures):
            raise QueryError(
                f"{name!r} is a measure; by and where take dimensions. Filter on a measure in SQL (GROUP BY ... HAVING)"
            ) from None
        raise QueryError(f"unknown dimension {name!r}{suggest(name, [d.name for d in model.dimensions])}") from None


def _column(item: Dimension | Measure, *, qualify: bool = True) -> str:
    name = column_name(item.name) if isinstance(item, Dimension) else item.name
    quoted = "`" + name.replace("`", "``") + "`"
    if not qualify:
        return quoted
    return f"{'r' if item.entity == 'run' else 't'}.{quoted}"


def _alias(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def _aggregate(model: SemanticModel, measure: Measure, aggregation: str) -> str:
    column = _column(measure)
    if aggregation in {"first", "last"}:
        order = model.entity(measure.entity).order_dimension
        if order is None:
            order_column = "r.`started_at`" if measure.entity == "run" else None
        else:
            order_column = _column(model.dimension(order))
        if order_column is None:
            raise QueryError(f"{measure.name!r} has no order, so first and last are undefined")
        # Skip rows where the value is missing, so "last" is the last recorded value.
        guarded = f"CASE WHEN {column} IS NOT NULL THEN {order_column} END"
        return f"{'min_by' if aggregation == 'first' else 'max_by'}({column}, {guarded})"
    if aggregation in _PERCENTILES:
        return f"percentile({column}, {_PERCENTILES[aggregation]})"
    return f"{_SQL_AGGREGATE[aggregation]}({column})"


def predicate(dimension: Dimension, condition: Condition, *, qualify: bool = True) -> str:
    column = _column(dimension, qualify=qualify)
    if condition.operator == "in":
        return f"{column} IN ({', '.join(_value(dimension, item) for item in condition.value)})"
    if condition.operator == "glob":
        # `*` matches any run of characters (as LIKE `%`); `_` also matches any one character.
        return f"{column} LIKE {_text(str(condition.value).replace('*', '%'))}"
    if condition.value is None:
        return f"{column} IS {'NOT ' if condition.operator == '!=' else ''}NULL"
    value = _value(dimension, condition.value)
    if condition.operator == "!=":
        return f"({column} IS NULL OR {column} != {value})"
    return f"{column} {condition.operator} {value}"


def _value(dimension: Dimension, value: Any) -> str:
    """A SQL literal for a filter value, converted to the dimension's type."""

    if dimension.type in {"integer", "number", "boolean"} and isinstance(value, str):
        text = value.strip().lower()
        if dimension.type == "boolean" and text in {"true", "false"}:
            value = text == "true"
        else:
            try:
                value = int(text) if dimension.type == "integer" else float(text)
            except ValueError:
                pass
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int | float):
        return repr(value)
    return _text(str(value))


def _text(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "''") + "'"


__all__ = ["CompiledQuery", "CompiledScope", "compile_query", "compile_scope", "predicate"]
