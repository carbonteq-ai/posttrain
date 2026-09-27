"""Describe what can be asked, so queries use real names."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ..models import ObservatoryModel, StringTuple
from .model import Dimension, Entity, Measure, Metric, SemanticModel
from .views import SEMANTIC_TABLES, TRACKIO_TABLES, table_columns


class DescribedTable(ObservatoryModel):
    name: str
    columns: tuple[str, ...]
    description: str


class SemanticDescribeRequest(ObservatoryModel):
    """Job kinds to describe; none describes everything."""

    job_kinds: StringTuple = ()
    runs: StringTuple | dict[str, Any] | None = None


class SemanticDescription(ObservatoryModel):
    job_kinds: tuple[str, ...]
    entities: tuple[Entity, ...]
    dimensions: tuple[Dimension, ...]
    measures: tuple[Measure, ...]
    metrics: tuple[Metric, ...]
    sql_tables: tuple[DescribedTable, ...]
    notes: tuple[str, ...] = Field(
        default=(
            "SQL is Doris SQL: one read-only SELECT (optionally WITH) over the sql_tables. Only the tables and "
            "columns a statement reads are computed, for every run of the project unless runs narrows them.",
            "Useful functions: max_by(value, step) and min_by(value, step) for last and first, percentile(x, 0.9), "
            "stddev_samp(x). On local SQLite storage only these and standard SQL are available.",
            "Short form: measures (name or name:aggregation, including p50/p90/p95/p99), by (dimensions), where "
            "(dimension: value, [any of], '>= n', or a '*' wildcard), runs (ids or run-dimension filters), "
            "order_by (prefix - for descending), limit. It compiles to SQL, returned with the result.",
            "Evaluation tasks are rollouts grouped by task: filter out truncated and failed rollouts to read "
            "valid rewards.",
        )
    )


def describe_semantics(model: SemanticModel, *, job_kinds: tuple[str, ...] = ()) -> SemanticDescription:
    def provided(measure: Measure) -> bool:
        return not job_kinds or "*" in measure.job_kinds or any(kind in measure.job_kinds for kind in job_kinds)

    measures = tuple(measure for measure in model.measures if provided(measure))
    names = {measure.name for measure in measures}
    tables = []
    for entity, table in SEMANTIC_TABLES.items():
        columns = tuple(
            column
            for column, owner in table_columns(model)[table].items()
            if not isinstance(owner, Measure) or owner.name in names
        )
        description = next(item.description for item in model.entities if item.name == entity)
        tables.append(DescribedTable(name=table, columns=columns, description=description))
    tables += [
        DescribedTable(name=table, columns=columns, description=f"Trackio's {table} for this project (raw).")
        for table, columns in TRACKIO_TABLES.items()
    ]
    return SemanticDescription(
        job_kinds=tuple(job_kinds),
        entities=model.entities,
        dimensions=model.dimensions,
        measures=measures,
        metrics=model.metrics,
        sql_tables=tuple(tables),
    )


__all__ = ["DescribedTable", "SemanticDescribeRequest", "SemanticDescription", "describe_semantics"]
