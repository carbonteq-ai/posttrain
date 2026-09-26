"""Describe what can be asked about a set of runs, so queries use real names."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field

from ..models import ObservatoryModel, StringTuple
from .execute import SemanticReader, _Context, applies, resolve_runs
from .model import Dimension, Entity, Measure, Metric, SemanticModel
from .query import DEFAULT_MAX_RUNS
from .sql import column_name


class DescribedMeasure(ObservatoryModel):
    measure: Measure
    runs: tuple[str, ...] = ()


class DescribedTable(ObservatoryModel):
    name: str
    columns: tuple[str, ...]


class SemanticDescribeRequest(ObservatoryModel):
    """Runs (ids or run-dimension filters) or job kinds to describe; neither describes everything."""

    runs: StringTuple | dict[str, Any] | None = None
    job_kinds: StringTuple = ()


class SemanticDescription(ObservatoryModel):
    job_kinds: tuple[str, ...]
    runs: tuple[str, ...] = ()
    entities: tuple[Entity, ...]
    dimensions: tuple[Dimension, ...]
    measures: tuple[DescribedMeasure, ...]
    metrics: tuple[Metric, ...]
    sql_tables: tuple[DescribedTable, ...]
    notes: tuple[str, ...] = Field(
        default=(
            "Query form: measures (name or name:aggregation), by (dimensions), where (dimension: value, [any of], "
            "'>= n', or a '*' wildcard), runs (ids or run-dimension filters), order_by (prefix - for descending), limit.",
            "A query may use one entity besides run; run dimensions apply to every entity.",
            "SQL mode: one SELECT over the sql_tables; raw_metrics(run_id, metric, step, value) loads any recorded "
            "series by name when listed in load.",
        )
    )


async def describe_semantics(
    model: SemanticModel,
    reader: SemanticReader,
    *,
    runs: tuple[str, ...] | Mapping[str, Any] | None = None,
    job_kinds: tuple[str, ...] = (),
    max_runs: int = DEFAULT_MAX_RUNS,
) -> SemanticDescription:
    selected: list[Any] = []
    if runs is not None:
        selected = await resolve_runs(_Context(model=model, reader=reader), runs, {}, max_runs=max_runs)
    kinds = tuple(sorted({run.job_kind for run in selected} | set(job_kinds)))
    if not kinds:
        described = [DescribedMeasure(measure=measure) for measure in model.measures]
    else:
        described = [
            DescribedMeasure(
                measure=measure,
                runs=tuple(run.run_id for run in selected if applies(measure, run.job_kind)),
            )
            for measure in model.measures
            if any(applies(measure, kind) for kind in kinds)
        ]
    entities = {"run"} | {item.measure.entity for item in described}
    measure_names = {item.measure.name for item in described}
    metrics = tuple(
        metric
        for metric in model.metrics
        if metric.entity in entities and all(name in measure_names for name in _formula_measures(metric))
    )
    tables = [
        DescribedTable(
            name="runs",
            columns=(
                "id",
                *(column_name(d.name) for d in model.dimensions if d.entity == "run" and d.name != "run.id"),
                "duration_seconds",
                *(
                    item.measure.name
                    for item in described
                    if item.measure.entity == "run" and item.measure.source.kind != "derived"
                ),
            ),
        )
    ]
    for entity, table, leading in (
        ("update", "updates", ("run_id", "step", "time")),
        ("rollout", "rollouts", ("run_id", *(column_name(d.name) for d in model.dimensions if d.entity == "rollout"))),
        (
            "eval_task",
            "eval_tasks",
            ("run_id", *(column_name(d.name) for d in model.dimensions if d.entity == "eval_task")),
        ),
        (
            "load_level",
            "load_levels",
            ("run_id", *(column_name(d.name) for d in model.dimensions if d.entity == "load_level")),
        ),
    ):
        if entity not in entities:
            continue
        values: list[str] = []
        for item in described:
            if item.measure.entity != entity:
                continue
            name = item.measure.name
            values += (
                [name]
                if entity != "rollout" or item.measure.source.name == "trace_count"
                else [name, f"{name}_sum", f"{name}_count"]
            )
        tables.append(DescribedTable(name=table, columns=(*leading, *values)))
    tables.append(DescribedTable(name="raw_metrics", columns=("run_id", "metric", "step", "value")))
    return SemanticDescription(
        job_kinds=kinds,
        runs=tuple(run.run_id for run in selected),
        entities=tuple(entity for entity in model.entities if entity.name in entities),
        dimensions=tuple(dimension for dimension in model.dimensions if dimension.entity in entities),
        measures=tuple(described),
        metrics=metrics,
        sql_tables=tuple(tables),
    )


def _formula_measures(metric: Metric) -> tuple[str, ...]:
    from .formula import parse_formula

    return tuple(name for _, name in parse_formula(metric.formula).references())


__all__ = [
    "DescribedMeasure",
    "DescribedTable",
    "SemanticDescribeRequest",
    "SemanticDescription",
    "describe_semantics",
]
