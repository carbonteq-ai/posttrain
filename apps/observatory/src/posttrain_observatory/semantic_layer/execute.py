"""Answer semantic queries: resolve runs, read only what is needed, group and aggregate.

Rows are loaded at the query's grain (the one non-run entity it names, or run),
filtered, grouped by the ``by`` dimensions and aggregated per measure. Rollout
rows arrive already aggregated by the tracking backend as sums, counts and sums
of squares, so their mean and standard deviation are combined exactly.
"""

from __future__ import annotations

import difflib
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from posttrain.tracking import MetricSeries, RunDetail, RunQuery, RunSummary, TraceAggregateResult, TraceFactsQuery
from posttrain.tracking.models import TraceFactAggregate

from .formula import parse_formula
from .model import Dimension, Measure, Metric, SemanticModel
from .query import (
    DEFAULT_MAX_RUNS,
    Condition,
    QueryError,
    ResultColumn,
    SemanticQuery,
    SemanticResult,
    parse_condition,
    split_measure,
)

MAX_SERIES_POINTS = 2000
SERIES_BATCH = 12
RUN_SCAN_LIMIT = 1000
_PUSHDOWN = {
    "run.work_package": "work_package_id",
    "run.job_kind": "job_kinds",
    "run.status": "statuses",
    "run.project": "project_id",
}


class SemanticReader(Protocol):
    """The reads the semantic layer needs; the Observatory service implements it."""

    async def list_runs(self, query: RunQuery) -> tuple[RunSummary, ...]: ...

    async def run_detail(self, run_id: str) -> RunDetail: ...

    async def metric_series(
        self, run_id: str, names: tuple[str, ...], *, max_points: int
    ) -> tuple[tuple[MetricSeries, ...], bool]: ...

    def trace_facts_available(self, run_id: str) -> bool: ...

    async def trace_facts(self, run_id: str, query: TraceFactsQuery) -> TraceAggregateResult: ...

    async def eval_tasks(self, run_id: str) -> tuple[Mapping[str, Any], ...]: ...

    async def load_levels(self, run_id: str) -> tuple[Mapping[str, Any], ...]: ...


@dataclass
class _Context:
    model: SemanticModel
    reader: SemanticReader
    details: dict[str, RunDetail] = field(default_factory=dict)
    unavailable: list[str] = field(default_factory=list)
    sources: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    downsampled: bool = False

    async def detail(self, run_id: str) -> RunDetail:
        if run_id not in self.details:
            self.details[run_id] = await self.reader.run_detail(run_id)
        return self.details[run_id]


@dataclass(frozen=True, slots=True)
class _Requested:
    column: str
    name: str
    aggregation: str
    measure: Measure | None
    metric: Metric | None


# ---------------------------------------------------------------- runs


def setting_value(resolved_inputs: Mapping[str, Any], path: str) -> Any:
    """Read ``a.b.c`` from resolved inputs; ``x|y`` tries each path in order."""

    for alternative in path.split("|"):
        value: Any = resolved_inputs
        for part in alternative.split("."):
            value = value.get(part) if isinstance(value, Mapping) else None
            if value is None:
                break
        if value is not None:
            return value
    return None


def run_field(summary: RunSummary, name: str) -> Any:
    if name == "error":
        return summary.error.type if summary.error is not None else None
    value = getattr(summary, name, None)
    return value.isoformat() if isinstance(value, datetime) else value


def duration_seconds(summary: RunSummary) -> float:
    end = summary.finished_at or datetime.now(UTC)
    return max((end - summary.started_at).total_seconds(), 0.0)


def applies(measure: Measure, job_kind: str) -> bool:
    return "*" in measure.job_kinds or job_kind in measure.job_kinds


async def run_row(context: _Context, summary: RunSummary, dimensions: Iterable[Dimension]) -> dict[str, Any]:
    row: dict[str, Any] = {"run.id": summary.run_id, "runs": 1, "duration_seconds": duration_seconds(summary)}
    for dimension in dimensions:
        if dimension.entity != "run":
            continue
        if dimension.source.kind == "run_field":
            row[dimension.name] = run_field(summary, dimension.source.name)
        elif dimension.source.kind == "setting":
            detail = await context.detail(summary.run_id)
            row[dimension.name] = setting_value(detail.resolved_inputs, dimension.source.name)
    return row


async def resolve_runs(
    context: _Context,
    scope: tuple[str, ...] | Mapping[str, Any] | None,
    run_filters: Mapping[str, Condition],
    *,
    max_runs: int = DEFAULT_MAX_RUNS,
) -> list[RunSummary]:
    """Select runs by explicit ids and/or run-dimension filters."""

    filters = dict(run_filters)
    if isinstance(scope, Mapping):
        for name, value in scope.items():
            key = name if name.startswith("run.") else f"run.{name}"
            filters[key] = parse_condition(value)
    explicit = tuple(scope) if isinstance(scope, (tuple, list)) else ()
    if "run.id" in filters:
        condition = filters["run.id"]
        values = condition.pushdown_values()
        if values is not None and not explicit:
            explicit = values
    for name in filters:
        try:
            dimension = context.model.dimension(name)
        except KeyError:
            raise QueryError(
                f"unknown dimension {name!r}{_suggest(name, [d.name for d in context.model.dimensions])}"
            ) from None
        if dimension.entity != "run":
            raise QueryError(f"{name!r} is not a run dimension")
    if explicit:
        summaries = [(await context.detail(run_id)).summary for run_id in explicit]
    else:
        pushed: dict[str, Any] = {}
        for name, key in _PUSHDOWN.items():
            values = filters[name].pushdown_values() if name in filters else None
            if values is None:
                continue
            if key in {"job_kinds", "statuses"}:
                pushed[key] = values
            elif len(values) == 1:
                pushed[key] = values[0]
        summaries = list(await context.reader.list_runs(RunQuery(limit=RUN_SCAN_LIMIT, **pushed)))
    dimensions = [context.model.dimension(name) for name in filters]
    selected: list[RunSummary] = []
    for summary in summaries:
        row = await run_row(context, summary, dimensions) if dimensions else {}
        if all(condition.matches(row.get(name)) for name, condition in filters.items()):
            selected.append(summary)
    if len(selected) > max_runs:
        raise QueryError(f"the query selects {len(selected)} runs; narrow it to at most {max_runs}")
    if not selected:
        raise QueryError("no runs match the query")
    return selected


# ---------------------------------------------------------------- loaders


def _within_step(values: Sequence[float], rule: str) -> float:
    if rule == "sum":
        return math.fsum(values)
    if rule == "mean":
        return math.fsum(values) / len(values)
    return values[-1]


async def load_updates(context: _Context, summary: RunSummary, measures: Sequence[Measure]) -> list[dict[str, Any]]:
    wanted = [measure for measure in measures if applies(measure, summary.job_kind)]
    if not wanted:
        return []
    names = tuple(dict.fromkeys(measure.source.name for measure in wanted))
    by_step: dict[int, dict[str, Any]] = {}
    repeated: dict[tuple[int, str], list[float]] = {}
    for start in range(0, len(names), SERIES_BATCH):
        batch = names[start : start + SERIES_BATCH]
        series, downsampled = await context.reader.metric_series(summary.run_id, batch, max_points=MAX_SERIES_POINTS)
        context.downsampled |= downsampled
        values = {item.name: item for item in series}
        for measure in wanted:
            if measure.source.name not in batch:
                continue
            context.sources[measure.name].add(measure.source.name)
            found = values.get(measure.source.name)
            if found is None:
                continue
            for point in found.points:
                if point.step is None:
                    continue
                row = by_step.setdefault(
                    point.step, {"run.id": summary.run_id, "update.step": point.step, "update.time": None}
                )
                step_values = repeated.setdefault((point.step, measure.name), [])
                step_values.append(point.value)
                value = _within_step(step_values, measure.source.within_step)
                row[measure.name] = 1.0 - value if measure.source.transform == "one_minus" else value
                if point.observed_at is not None and row["update.time"] is None:
                    row["update.time"] = max((point.observed_at - summary.started_at).total_seconds(), 0.0)
    return [by_step[step] for step in sorted(by_step)]


async def load_run_metrics(context: _Context, summary: RunSummary, measures: Sequence[Measure]) -> dict[str, Any]:
    """Run measures logged as metric series: each run's value is its last logged point."""

    wanted = [
        measure
        for measure in measures
        if measure.entity == "run" and measure.source.kind == "metric_series" and applies(measure, summary.job_kind)
    ]
    values: dict[str, Any] = {}
    names = tuple(dict.fromkeys(measure.source.name for measure in wanted))
    for start in range(0, len(names), SERIES_BATCH):
        series, _ = await context.reader.metric_series(
            summary.run_id, names[start : start + SERIES_BATCH], max_points=MAX_SERIES_POINTS
        )
        last = {
            item.name: max(item.points, key=lambda point: (point.step is not None, point.step or 0)).value
            for item in series
            if item.points
        }
        for measure in wanted:
            context.sources[measure.name].add(measure.source.name)
            if measure.source.name in last:
                value = last[measure.source.name]
                values[measure.name] = 1.0 - value if measure.source.transform == "one_minus" else value
    return values


async def load_rollouts(
    context: _Context, summary: RunSummary, measures: Sequence[Measure], dimensions: Sequence[Dimension]
) -> list[dict[str, Any]]:
    if not context.reader.trace_facts_available(summary.run_id):
        context.unavailable.append(f"rollouts of {summary.run_id}: the source has no trace facts")
        return []
    facts = [measure.source.name for measure in measures if measure.source.name != "trace_count"]
    aggregates = tuple(
        TraceFactAggregate(measure=fact, operation=operation)  # type: ignore[arg-type]
        for fact in dict.fromkeys(facts)
        for operation in ("sum", "count", "sum_squares")
    ) or (TraceFactAggregate(measure="model_calls", operation="count"),)
    group_by = tuple(dict.fromkeys(dimension.source.name for dimension in dimensions))
    result = await context.reader.trace_facts(summary.run_id, TraceFactsQuery(group_by=group_by, aggregates=aggregates))  # type: ignore[arg-type]
    if result.state != "available":
        context.unavailable.append(f"rollouts of {summary.run_id}: trace facts are {result.state}")
        return []
    by_fact = {dimension.source.name: dimension.name for dimension in dimensions}
    rows = []
    for bucket in result.buckets:
        row: dict[str, Any] = {"run.id": summary.run_id, "__traces": bucket.trace_count}
        for fact, value in bucket.dimensions.items():
            if fact in by_fact:
                row[by_fact[fact]] = value
        for measure in measures:
            context.sources[measure.name].add(f"trace_fact:{measure.source.name}")
            if measure.source.name == "trace_count":
                continue
            row[f"{measure.name}__sum"] = bucket.values.get(f"sum_{measure.source.name}")
            row[f"{measure.name}__count"] = bucket.values.get(f"count_{measure.source.name}")
            row[f"{measure.name}__sumsq"] = bucket.values.get(f"sum_squares_{measure.source.name}")
        rows.append(row)
    return rows


async def load_view_rows(
    context: _Context, summary: RunSummary, entity: str, measures: Sequence[Measure], dimensions: Sequence[Dimension]
) -> list[dict[str, Any]]:
    raw = await (context.reader.eval_tasks if entity == "eval_task" else context.reader.load_levels)(summary.run_id)
    if not raw:
        context.unavailable.append(f"{entity} rows of {summary.run_id}: the run has none")
    rows = []
    for item in raw:
        row: dict[str, Any] = {"run.id": summary.run_id}
        for dimension in dimensions:
            row[dimension.name] = item.get(dimension.source.name)
        for measure in measures:
            if applies(measure, summary.job_kind):
                row[measure.name] = item.get(measure.source.name)
                context.sources[measure.name].add(f"{entity}:{measure.source.name}")
        rows.append(row)
    return rows


# ---------------------------------------------------------------- aggregation


def aggregate_values(aggregation: str, values: Sequence[tuple[Any, Any]]) -> float | None:
    """Aggregate (order key, value) pairs; missing values are ignored."""

    present = [
        (key, value)
        for key, value in values
        if value is not None and not (isinstance(value, float) and math.isnan(value))
    ]
    if aggregation == "count":
        return float(len(present))
    if not present:
        return None
    numbers = [float(value) for _, value in present]
    if aggregation == "last":
        return float(max(present, key=lambda item: (item[0] is not None, item[0]))[1])
    if aggregation == "first":
        return float(min(present, key=lambda item: (item[0] is None, item[0]))[1])
    if aggregation == "min":
        return min(numbers)
    if aggregation == "max":
        return max(numbers)
    if aggregation == "sum":
        return math.fsum(numbers)
    mean = math.fsum(numbers) / len(numbers)
    if aggregation == "mean":
        return mean
    if len(numbers) < 2:
        return 0.0
    return math.sqrt(math.fsum((value - mean) ** 2 for value in numbers) / (len(numbers) - 1))


def aggregate_rollouts(aggregation: str, measure: Measure, rows: Sequence[Mapping[str, Any]]) -> float | None:
    if measure.source.name == "trace_count":
        total = float(sum(row.get("__traces") or 0 for row in rows))
        return total
    total = math.fsum(row.get(f"{measure.name}__sum") or 0.0 for row in rows)
    count = sum(row.get(f"{measure.name}__count") or 0 for row in rows)
    squares = math.fsum(row.get(f"{measure.name}__sumsq") or 0.0 for row in rows)
    if aggregation == "count":
        return float(count)
    if aggregation == "sum":
        return total if count else None
    if count == 0:
        return None
    mean = total / count
    if aggregation == "mean":
        return mean
    return math.sqrt(max(squares / count - mean * mean, 0.0))


# ---------------------------------------------------------------- query


def _requested(model: SemanticModel, query: SemanticQuery) -> list[_Requested]:
    requested = []
    for value in query.measures:
        name, aggregation = split_measure(value)
        column = name if aggregation is None else f"{name}_{aggregation}"
        try:
            measure = model.measure(name)
        except KeyError:
            try:
                metric = model.metric(name)
            except KeyError:
                raise QueryError(
                    f"unknown measure or metric {name!r}"
                    f"{_suggest(name, [*(m.name for m in model.measures), *(m.name for m in model.metrics)])}"
                ) from None
            if aggregation is not None:
                raise QueryError(f"metric {name!r} takes no aggregation") from None
            requested.append(_Requested(column, name, "formula", None, metric))
            continue
        chosen = aggregation or measure.aggregation
        if chosen not in measure.allowed:
            raise QueryError(f"{name!r} cannot be aggregated with {chosen!r}; allowed: {', '.join(measure.allowed)}")
        requested.append(_Requested(column, name, chosen, measure, None))
    return requested


def _summary_filter(model: SemanticModel, name: str) -> bool:
    key = name if name.startswith("run.") else f"run.{name}"
    try:
        return model.dimension(key).source.kind == "run_field"
    except KeyError:
        return False


def _suggest(name: str, known: Sequence[str]) -> str:
    """A hint naming the closest known names, so a mistyped query can be fixed without describe."""

    close = difflib.get_close_matches(name, known, n=3, cutoff=0.5)
    close += [item for item in known if name.split(".")[-1] in item and item not in close][: 3 - len(close)]
    return f"; did you mean {', '.join(repr(item) for item in close)}?" if close else "; call describe for the names"


def _grain(model: SemanticModel, requested: Sequence[_Requested], dimensions: Iterable[str]) -> str:
    entities = {item.measure.entity if item.measure else item.metric.entity for item in requested}  # type: ignore[union-attr]
    for name in dimensions:
        try:
            entities.add(model.dimension(name).entity)
        except KeyError:
            raise QueryError(
                f"unknown dimension {name!r}{_suggest(name, [d.name for d in model.dimensions])}"
            ) from None
    detailed = entities - {"run"}
    if len(detailed) > 1:
        raise QueryError(f"a query can use one entity besides run; this one uses {', '.join(sorted(detailed))}")
    grain = next(iter(detailed), "run")
    if grain != "run" and any(item.measure is not None and item.measure.entity == "run" for item in requested):
        raise QueryError("run measures can only be queried at run grain")
    return grain


async def run_semantic_query(
    model: SemanticModel,
    reader: SemanticReader,
    query: SemanticQuery,
    *,
    max_runs: int = DEFAULT_MAX_RUNS,
) -> SemanticResult:
    context = _Context(model=model, reader=reader)
    requested = _requested(model, query)
    conditions = {name: parse_condition(value) for name, value in query.where.items()}
    grain = _grain(model, requested, [*query.by, *conditions])
    run_filters = {name: condition for name, condition in conditions.items() if name.startswith("run.")}
    entity_filters = {name: condition for name, condition in conditions.items() if not name.startswith("run.")}
    measures: list[Measure] = []
    for item in requested:
        if item.measure is not None:
            measures.append(item.measure)
        else:
            for _, name in parse_formula(item.metric.formula).references():  # type: ignore[union-attr]
                measures.append(model.measure(name))
    measures = list({measure.name: measure for measure in measures}.values())
    # Questions answered from run summaries alone (how many runs, of which kind,
    # how long) read nothing per run, so they may cover the whole project.
    summary_only = (
        grain == "run"
        and all(measure.source.kind == "derived" for measure in measures)
        and all(
            model.dimension(name).source.kind == "run_field"
            for name in [*query.by, *run_filters]
            if name.startswith("run.")
        )
        and (not isinstance(query.runs, Mapping) or all(_summary_filter(model, name) for name in query.runs))
    )
    runs = await resolve_runs(context, query.runs, run_filters, max_runs=RUN_SCAN_LIMIT if summary_only else max_runs)
    for measure in measures:
        providers = [run for run in runs if applies(measure, run.job_kind)]
        if not providers:
            kinds = ", ".join(sorted({run.job_kind for run in runs}))
            raise QueryError(f"no selected run provides {measure.name!r} (selected job kinds: {kinds})")
        for run in runs:
            if run not in providers:
                context.unavailable.append(f"{measure.name}: not provided by {run.job_kind} run {run.run_id}")

    run_dimensions = [
        model.dimension(name) for name in dict.fromkeys([*query.by, *run_filters]) if name.startswith("run.")
    ]
    entity_dimensions = [
        model.dimension(name) for name in dict.fromkeys([*query.by, *entity_filters]) if not name.startswith("run.")
    ]
    rows: list[dict[str, Any]] = []
    for summary in runs:
        base = await run_row(context, summary, run_dimensions)
        if grain == "run":
            loaded = [{**base, **await load_run_metrics(context, summary, measures)}]
        elif grain == "update":
            loaded = [{**base, **row} for row in await load_updates(context, summary, measures)]
        elif grain == "rollout":
            loaded = [{**base, **row} for row in await load_rollouts(context, summary, measures, entity_dimensions)]
        else:
            loaded = [
                {**base, **row} for row in await load_view_rows(context, summary, grain, measures, entity_dimensions)
            ]
        if not loaded and all(name.startswith("run.") for name in query.by):
            # Keep the run visible with empty measures; `unavailable` says why.
            loaded = [dict(base)]
        rows.extend(
            row for row in loaded if all(condition.matches(row.get(name)) for name, condition in entity_filters.items())
        )

    order_key = model.entity(grain).order_dimension if grain != "run" else None
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(name) for name in query.by)].append(row)
    if not query.by and not rows:
        groups[()] = []

    def value_of(aggregation: str, measure: Measure, members: Sequence[Mapping[str, Any]]) -> float | None:
        if grain == "rollout":
            return aggregate_rollouts(aggregation, measure, members)
        return aggregate_values(
            aggregation, [(row.get(order_key) if order_key else None, row.get(measure.name)) for row in members]
        )

    result_rows: list[tuple[Any, ...]] = []
    for key, members in groups.items():
        values: list[Any] = list(key)
        for item in requested:
            if item.measure is not None:
                values.append(value_of(item.aggregation, item.measure, members))
            else:
                formula = parse_formula(item.metric.formula)  # type: ignore[union-attr]
                values.append(
                    formula.evaluate(
                        lambda aggregation, name, group=members: value_of(aggregation, model.measure(name), group)
                    )
                )
        result_rows.append(tuple(values))

    columns = [
        ResultColumn(
            name=name, kind="dimension", type=model.dimension(name).type, label=model.dimension(name).description
        )
        for name in query.by
    ] + [
        ResultColumn(
            name=item.column,
            kind="measure" if item.measure else "metric",
            unit=(item.measure or item.metric).unit,  # type: ignore[union-attr]
            label=(item.measure or item.metric).label,  # type: ignore[union-attr]
            type="number",
        )
        for item in requested
    ]
    result_rows = _ordered(result_rows, [column.name for column in columns], query.order_by)
    truncated = len(result_rows) > query.limit
    return SemanticResult(
        columns=tuple(columns),
        rows=tuple(result_rows[: query.limit]),
        grain=grain,
        runs=tuple(run.run_id for run in runs),
        sources={name: tuple(sorted(values)) for name, values in context.sources.items()},
        downsampled=context.downsampled,
        unavailable=tuple(dict.fromkeys(context.unavailable)),
        truncated=truncated,
    )


def _ordered(rows: list[tuple[Any, ...]], names: Sequence[str], order_by: Sequence[str]) -> list[tuple[Any, ...]]:
    if not order_by:
        return sorted(
            rows,
            key=lambda row: tuple(
                (value is None, str(value) if isinstance(value, str) else value) for value in row[:1]
            ),
        )
    for item in reversed(order_by):
        descending = item.startswith("-")
        name = item.lstrip("-")
        if name not in names:
            raise QueryError(f"cannot order by {name!r}; it is not a result column")
        index = names.index(name)
        present = [row for row in rows if row[index] is not None]
        missing = [row for row in rows if row[index] is None]
        rows = sorted(present, key=lambda row: row[index], reverse=descending) + missing
    return rows


__all__ = [
    "SemanticReader",
    "aggregate_rollouts",
    "aggregate_values",
    "applies",
    "duration_seconds",
    "load_rollouts",
    "load_run_metrics",
    "load_updates",
    "load_view_rows",
    "resolve_runs",
    "run_field",
    "run_row",
    "run_semantic_query",
    "setting_value",
]
