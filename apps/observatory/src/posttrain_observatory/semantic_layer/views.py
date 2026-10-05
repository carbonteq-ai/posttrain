"""The semantic tables as SQL views over Trackio's project tables, in Doris SQL.

Trackio's project SQL (`ProjectSql`) exposes ``metric_rows`` (one row per
logged batch, values in a JSON object), ``run_configs`` (each run's config as
JSON), ``traces`` (trace facts as columns) and ``run_notes``. This module
defines ``runs``, ``updates`` and ``rollouts`` over them and generates, for one
statement, only the views and columns that statement reads.

``updates`` applies the same two rules as `posttrain.tracking.logical_series`:
a metric replayed from traces (``observation_source = 'verifiers'`` with a
``source_step``) belongs to that update and replaces live points of the same
metric there; and runs recorded before rollout metrics were written once per
update have several points per update, which are summed (counts, seconds,
tokens) or averaged (rates). Otherwise the last point written wins.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import sqlglot
from posttrain.tracking.logical import LEGACY_ROLLOUT_BATCH_METRICS
from sqlglot import exp
from sqlglot.errors import OptimizeError, SqlglotError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope

from .model import Dimension, Measure, SemanticModel
from .query import QueryError

if TYPE_CHECKING:
    from .compile import CompiledScope

SEMANTIC_TABLES = {"run": "runs", "collection": "collections", "update": "updates", "rollout": "rollouts"}
TRACKIO_TABLES: dict[str, tuple[str, ...]] = {
    "metric_rows": ("run_id", "run_name", "step", "timestamp", "metrics"),
    "run_configs": ("run_id", "run_name", "config", "created_at"),
    "traces": (
        "run_id",
        "run_name",
        "step",
        "timestamp",
        "trace_type",
        "external_id",
        "metadata",
        "fact_state",
        "fact_calculator_version",
        "fact_model",
        "fact_task_type",
        "fact_task_id",
        "fact_prompt_group_id",
        "fact_rollout_step",
        "fact_is_truncated",
        "fact_has_error",
        "fact_model_input_tokens",
        "fact_model_output_tokens",
        "fact_thinking_tokens",
        "fact_tool_calls",
        "fact_model_calls",
        "fact_trace_latency_ms",
        "fact_task_reward",
        "fact_algorithm_reward",
    ),
    "run_notes": (
        "note_id",
        "revision",
        "scope",
        "run_id",
        "run_name",
        "kind",
        "title",
        "body_md",
        "source",
        "created_at",
        "revised_at",
        "deleted",
    ),
}
_RESERVED_PREFIX = "_pt_"
_CONFIG_FIELDS = {
    "run_id": "run_id",
    "project_id": "project_id",
    "work_package_id": "work_package_id",
    "job_kind": "job_kind",
    "stage": "stage",
}
_LIFECYCLE_FIELDS = ("status", "started_at", "finished_at", "error", "error_message")
_FACT_COLUMN = {"trace_count": None}


def column_name(semantic_name: str) -> str:
    """`run.learning_rate` -> `learning_rate`; `update.step` -> `step`."""
    return semantic_name.split(".", 1)[1] if "." in semantic_name else semantic_name


def table_columns(model: SemanticModel) -> dict[str, dict[str, Dimension | Measure | None]]:
    """Each semantic table's columns and the dimension or measure behind each (None for keys)."""

    tables: dict[str, dict[str, Dimension | Measure | None]] = {}
    for entity, table in SEMANTIC_TABLES.items():
        columns: dict[str, Dimension | Measure | None] = {"id" if entity == "run" else "run_id": None}
        for dimension in model.dimensions:
            if dimension.entity == entity and dimension.name != "run.id":
                columns[column_name(dimension.name)] = dimension
        for measure in model.measures:
            if measure.entity == entity:
                columns[measure.name] = measure
        tables[table] = columns
    return tables


# ---------------------------------------------------------------- planning


def plan(model: SemanticModel, statement: exp.Query) -> dict[str, frozenset[str]]:
    """The semantic tables a statement reads and the columns it reads from each."""

    semantic = table_columns(model)
    schema: dict[str, object] = {
        **{table: {column: "TEXT" for column in columns} for table, columns in semantic.items()},
        **{table: {column: "TEXT" for column in columns} for table, columns in TRACKIO_TABLES.items()},
    }
    try:
        qualified = qualify(statement.copy(), schema=schema, dialect="doris", validate_qualify_columns=True)
    except (OptimizeError, SqlglotError) as error:
        raise QueryError(_explain(str(error), semantic)) from error
    read: dict[str, set[str]] = {}
    for scope in traverse_scope(qualified):
        for source in scope.sources.values():
            if isinstance(source, exp.Table) and source.name in semantic:
                read.setdefault(source.name, set())
        for column in scope.columns:
            source = scope.sources.get(column.table)
            if isinstance(source, exp.Table) and source.name in semantic:
                read.setdefault(source.name, set()).add(column.name)
    return {table: frozenset(columns) for table, columns in read.items()}


def _explain(message: str, semantic: Mapping[str, Mapping[str, Any]]) -> str:
    import difflib

    for token in ("Column '", "column '"):
        if token in message:
            name = message.split(token, 1)[1].split("'", 1)[0]
            known = sorted({column for columns in semantic.values() for column in columns})
            close = difflib.get_close_matches(name.split(".")[-1], known, n=3, cutoff=0.5)
            hint = f"; did you mean {', '.join(close)}?" if close else "; call describe for the columns"
            return f"SQL error: unknown column {name!r}{hint}"
    return f"SQL error: {message}"


# ---------------------------------------------------------------- SQL helpers


def literal(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "''") + "'"


def json_path(*keys: str) -> str:
    """A Doris JSON path with every key quoted, for example `$."train/rl/entropy"`."""
    return literal("$" + "".join('."' + key.replace("\\", "\\\\").replace('"', '\\"') + '"' for key in keys))


def _extract(column: str, kind: str, *keys: str) -> str:
    function = {
        "string": "json_extract_string",
        "time": "json_extract_string",
        "integer": "json_extract_bigint",
        "number": "json_extract_double",
        "boolean": "json_extract_bool",
    }[kind]
    return f"{function}({column}, {json_path(*keys)})"


def _quote(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


# ---------------------------------------------------------------- views


def build_views(
    model: SemanticModel, planned: Mapping[str, frozenset[str]], scope: CompiledScope | None
) -> list[tuple[str, str]]:
    """The CTEs (name, SQL) a statement needs, in dependency order."""

    if not planned:
        return []
    semantic = table_columns(model)
    run_columns = set(planned.get("runs", ())) | {"id"} | set(scope.columns if scope else ())
    # Entity views read each run's start (update time) through the scope.
    needs_start = "time" in planned.get("updates", ()) or "time" in planned.get("collections", ())
    if needs_start:
        run_columns.add("started_at")
    views: list[tuple[str, str]] = []
    owners = semantic["runs"]
    dimensions = [owner for name in sorted(run_columns) if isinstance(owner := owners.get(name), Dimension)]
    measures = [owner for name in sorted(run_columns) if isinstance(owner := owners.get(name), Measure)]
    lifecycle = any(
        isinstance(item, Dimension) and item.source.kind == "run_field" and item.source.name in _LIFECYCLE_FIELDS
        for item in dimensions
    ) or any(item.name == "duration_seconds" for item in measures)
    if lifecycle:
        views.append(("_pt_lifecycle", _lifecycle_sql()))
    events = sorted({item.source.name for item in dimensions if item.source.kind == "event"})
    for index, event in enumerate(events):
        views.append((f"_pt_event_{index}", _event_sql(event, dimensions)))
    series = [item for item in measures if item.source.kind == "metric_series"]
    for index, measure in enumerate(series):
        views.append((f"_pt_run_metric_{index}", _run_metric_sql(measure)))
    views.append(("_pt_runs_all", _runs_all_sql(dimensions, measures, lifecycle, events, series)))
    scope_columns = ["provider_id", "id", *(["started_at"] if needs_start else [])]
    where = f" WHERE {scope.predicate}" if scope else ""
    views.append(("_pt_scope", f"SELECT {', '.join(scope_columns)} FROM _pt_runs_all{where}"))
    if "runs" in planned:
        visible = [name for name in owners if name in run_columns]
        views.append(
            (
                "runs",
                f"SELECT {', '.join(_quote(name) for name in visible)} FROM _pt_runs_all"
                f" WHERE provider_id IN (SELECT provider_id FROM _pt_scope)",
            )
        )
    collection_steps = "collections" in planned or "collection_step" in planned.get("updates", ())
    if collection_steps:
        read = planned.get("collections", frozenset())
        markers = sorted(
            {
                owner.source.name
                for name, owner in semantic["collections"].items()
                if name in read and isinstance(owner, Measure) and owner.source.kind == "metric_series"
            }
        )
        views.append(("_pt_collection_starts", _collection_starts_sql(markers)))
        views.append(("_pt_update_collection", _update_collection_sql()))
    if "collections" in planned:
        views.extend(_collections_sql(semantic["collections"], planned["collections"]))
    if "updates" in planned:
        views.extend(_updates_sql(model, semantic["updates"], planned["updates"]))
    if "rollouts" in planned:
        views.append(("rollouts", _rollouts_sql(semantic["rollouts"], planned["rollouts"])))
    return views


def _lifecycle_sql() -> str:
    status = _extract("metrics", "string", "run/status")
    fields = ", ".join(
        f"max_by({_extract('metrics', 'string', key)}, `timestamp`) AS {name}"
        for name, key in (
            ("status", "run/status"),
            ("started_at", "run/started_at"),
            ("finished_at", "run/finished_at"),
            ("error_type", "run/error_type"),
            ("error_message", "run/error_message"),
        )
    )
    return f"SELECT run_id, {fields} FROM metric_rows WHERE {status} IS NOT NULL GROUP BY run_id"


def _event_sql(event: str, dimensions: Sequence[Dimension]) -> str:
    name, _, _ = event.partition(":")
    values = ", ".join(
        f"min_by({_extract('metrics', item.type, 'event/attributes', item.source.name.partition(':')[2])}, step)"
        f" AS {_quote(column_name(item.name))}"
        for item in dimensions
        if item.source.kind == "event" and item.source.name == event
    )
    return (
        f"SELECT run_id, {values} FROM metric_rows WHERE {_extract('metrics', 'string', 'event/name')} = "
        f"{literal(name)} GROUP BY run_id"
    )


def _run_metric_sql(measure: Measure) -> str:
    value = _extract("metrics", "number", measure.source.name)
    last = "max_by(v, step)"
    if measure.source.transform == "one_minus":
        last = f"1 - {last}"
    return (
        f"SELECT run_id, {last} AS v FROM (SELECT run_id, step, {value} AS v FROM metric_rows) AS p"
        " WHERE v IS NOT NULL GROUP BY run_id"
    )


def _runs_all_sql(
    dimensions: Sequence[Dimension],
    measures: Sequence[Measure],
    lifecycle: bool,
    events: Sequence[str],
    series: Sequence[Measure],
) -> str:
    started = f"COALESCE(l.started_at, {_extract('c.config', 'string', 'started_at')})"
    columns = ["c.run_id AS provider_id", f"{_extract('c.config', 'string', 'run_id')} AS id"]
    for item in dimensions:
        name = column_name(item.name)
        if name == "id":
            continue
        source = item.source
        if source.kind == "run_field":
            if source.name == "display_name":
                expression = "c.run_name"
            elif source.name in _CONFIG_FIELDS:
                expression = _extract("c.config", "string", _CONFIG_FIELDS[source.name])
            elif source.name == "status":
                expression = "COALESCE(l.status, 'running')"
            elif source.name == "started_at":
                expression = started
            elif source.name == "finished_at":
                expression = "l.finished_at"
            elif source.name == "error":
                expression = "CASE WHEN l.status = 'failed' THEN COALESCE(l.error_type, 'RunFailed') END"
            elif source.name == "error_message":
                expression = "CASE WHEN l.status = 'failed' THEN COALESCE(l.error_message, 'run failed') END"
            else:  # pragma: no cover - the framework declares no other run fields
                raise QueryError(f"run field {source.name!r} has no SQL definition")
        elif source.kind == "setting":
            alternatives = [
                _extract("c.config", item.type, "resolved_selections", *path.split("."))
                for path in source.name.split("|")
            ]
            expression = alternatives[0] if len(alternatives) == 1 else f"COALESCE({', '.join(alternatives)})"
        elif source.kind == "event":
            expression = f"e{events.index(source.name)}.{_quote(name)}"
        else:  # pragma: no cover
            raise QueryError(f"dimension {item.name!r} has no SQL definition")
        columns.append(f"{expression} AS {_quote(name)}")
    for measure in measures:
        if measure.name == "runs":
            expression = "1"
        elif measure.name == "duration_seconds":
            expression = f"COALESCE(unix_timestamp(l.finished_at), unix_timestamp()) - unix_timestamp({started})"
        else:
            expression = f"m{series.index(measure)}.v"
        columns.append(f"{expression} AS {_quote(measure.name)}")
    joins = ["run_configs AS c"]
    if lifecycle:
        joins.append("LEFT JOIN _pt_lifecycle AS l ON l.run_id = c.run_id")
    joins += [f"LEFT JOIN _pt_event_{index} AS e{index} ON e{index}.run_id = c.run_id" for index in range(len(events))]
    joins += [
        f"LEFT JOIN _pt_run_metric_{index} AS m{index} ON m{index}.run_id = c.run_id" for index in range(len(series))
    ]
    return (
        f"SELECT {', '.join(columns)} FROM {' '.join(joins)}"
        f" WHERE {_extract('c.config', 'integer', 'schema_version')} = 4"
    )


def _updates_sql(
    model: SemanticModel, owners: Mapping[str, Dimension | Measure | None], columns: frozenset[str]
) -> list[tuple[str, str]]:
    measures = [owner for name, owner in owners.items() if name in columns and isinstance(owner, Measure)]
    # An update is a step with an update time, plus any step a read metric has a point at, so counting
    # updates does not depend on which columns a statement reads.
    backbone = model.measure("update_seconds")
    present = [*measures, *([] if backbone in measures else [backbone])]
    # One pass for every metric read: stack their points, keep replayed points where a metric has
    # them for an update (replay authority), then one row per update with one aggregate per metric.
    # (One sub-query per metric joined together takes Doris seconds to plan.)
    points = " UNION ALL ".join(_update_points_sql(measure, index) for index, measure in enumerate(present))
    views = [
        (
            "_pt_update_points",
            f"SELECT p.*, MAX(replay) OVER (PARTITION BY run_id, m, step) AS any_replay FROM ({points}) AS p",
        )
    ]
    selected = ["s.id AS run_id", "u.step AS step"]
    if "time" in columns:
        firsts = ", ".join(f"MIN(CASE WHEN m = {index} THEN ts END)" for index in range(len(present)))
        seen = firsts if len(present) == 1 else f"COALESCE({firsts})"
        selected.append(f"unix_timestamp({seen}) - unix_timestamp(MIN(s.started_at)) AS `time`")
    for measure in measures:
        selected.append(f"{_update_aggregate(measure, present.index(measure))} AS {_quote(measure.name)}")
    collection = ""
    if "collection_step" in columns:
        selected.append("MAX(c.collection_step) AS `collection_step`")
        # One row per run and step, so the join never repeats a point.
        collection = " LEFT JOIN _pt_update_collection AS c ON c.run_id = u.run_id AND c.step = u.step"
    views.append(
        (
            "updates",
            f"SELECT {', '.join(selected)} FROM _pt_update_points AS u"
            f" JOIN _pt_scope AS s ON s.provider_id = u.run_id{collection}"
            " WHERE u.replay = u.any_replay GROUP BY s.id, u.step",
        )
    )
    return views


# Metrics a trainer writes once per collection (sampled population). A step carrying any of them
# starts a collection; every update of a run that trained on each population once starts its own.
_COLLECTION_MARKERS = ("train/rl/collection_updates", "train/rl/reward_mean", "train/rl/rollouts_attempted")


def _replayed_step(column: str = "step") -> str:
    """A point's logical step: the replayed `source_step` for points recomputed from traces."""
    observation = _extract("metrics", "string", "metric/attributes", "observation_source")
    source_step = _extract("metrics", "integer", "metric/attributes", "source_step")
    return f"CASE WHEN {observation} = 'verifiers' AND {source_step} >= 0 THEN {source_step} ELSE {column} END"


def _collection_starts_sql(read: Sequence[str] = ()) -> str:
    """Steps that start a collection: a marker's step, or a step where a collection metric the statement
    reads has a point (so legacy runs keep every population row whatever columns are read)."""

    markers = dict.fromkeys((*_COLLECTION_MARKERS, *read))
    present = " OR ".join(f"{_extract('metrics', 'number', marker)} IS NOT NULL" for marker in markers)
    return (
        f"SELECT run_id, {_replayed_step()} AS step, MIN(`timestamp`) AS start_ts FROM metric_rows"
        f" WHERE ({present}) AND run_id IN (SELECT provider_id FROM _pt_scope)"
        f" GROUP BY run_id, {_replayed_step()}"
    )


def _update_collection_sql() -> str:
    """Each update's collection: the trainer's `collection_step` tag, else the latest collection start at
    or before it (runs recorded before the tag). Also the update's step time, for collection totals."""

    tagged = _extract("metrics", "integer", "metric/attributes", "collection_step")
    seconds = _extract("metrics", "number", "train/step_time_seconds")
    return (
        "SELECT u.run_id, u.step, COALESCE(MAX(u.tagged), MAX(c.step)) AS collection_step,"
        " MAX(u.seconds) AS seconds"
        f" FROM (SELECT run_id, step, {tagged} AS tagged, {seconds} AS seconds FROM metric_rows"
        f" WHERE ({tagged} IS NOT NULL OR {seconds} IS NOT NULL)"
        " AND run_id IN (SELECT provider_id FROM _pt_scope)) AS u"
        " LEFT JOIN _pt_collection_starts AS c ON c.run_id = u.run_id AND c.step <= u.step"
        " GROUP BY u.run_id, u.step"
    )


def _collections_sql(
    owners: Mapping[str, Dimension | Measure | None], columns: frozenset[str]
) -> list[tuple[str, str]]:
    """One row per run and collection: its collection metrics, and its updates' count and step time."""

    measures = [
        owner
        for name, owner in owners.items()
        if name in columns and isinstance(owner, Measure) and owner.source.kind == "metric_series"
    ]
    views: list[tuple[str, str]] = []
    joins = [
        " LEFT JOIN (SELECT run_id, collection_step, COUNT(seconds) AS updates, SUM(seconds) AS seconds"
        " FROM _pt_update_collection GROUP BY run_id, collection_step) AS uc"
        " ON uc.run_id = c.run_id AND uc.collection_step = c.step"
    ]
    if measures:
        points = " UNION ALL ".join(_update_points_sql(measure, index) for index, measure in enumerate(measures))
        views.append(
            (
                "_pt_collection_points",
                f"SELECT p.*, MAX(replay) OVER (PARTITION BY run_id, m, step) AS any_replay FROM ({points}) AS p",
            )
        )
        joins.append(
            " LEFT JOIN _pt_collection_points AS u"
            " ON u.run_id = c.run_id AND u.step = c.step AND u.replay = u.any_replay"
        )
    selected = ["s.id AS run_id", "c.step AS step"]
    if "time" in columns:
        selected.append("unix_timestamp(MIN(c.start_ts)) - unix_timestamp(MIN(s.started_at)) AS `time`")
    if "updates" in columns:
        selected.append("COALESCE(MAX(uc.updates), 0) AS `updates`")
    if "collection_seconds" in columns:
        selected.append("MAX(uc.seconds) AS `collection_seconds`")
    for index, measure in enumerate(measures):
        selected.append(f"{_update_aggregate(measure, index)} AS {_quote(measure.name)}")
    views.append(
        (
            "collections",
            f"SELECT {', '.join(selected)} FROM _pt_collection_starts AS c"
            f" JOIN _pt_scope AS s ON s.provider_id = c.run_id{''.join(joins)}"
            " GROUP BY s.id, c.step",
        )
    )
    return views


def _update_points_sql(measure: Measure, index: int) -> str:
    metric = measure.source.name
    value = _extract("metrics", "number", metric)
    observation = (
        f"COALESCE({_extract('metrics', 'string', metric + '/attributes', 'observation_source')}, "
        f"{_extract('metrics', 'string', 'metric/attributes', 'observation_source')})"
    )
    source_step = (
        f"COALESCE({_extract('metrics', 'integer', metric + '/attributes', 'source_step')}, "
        f"{_extract('metrics', 'integer', 'metric/attributes', 'source_step')})"
    )
    replayed = f"{observation} = 'verifiers' AND {source_step} >= 0"
    # Every row is one write (Trackio keys rows by log id): rollout batches that repeat a
    # value, and even an ordinal, are separate points.
    return (
        f"SELECT run_id, {index} AS m, `timestamp` AS ts, {value} AS v, "
        f"CASE WHEN {replayed} THEN 1 ELSE 0 END AS replay, "
        f"CASE WHEN {replayed} THEN {source_step} ELSE step END AS step "
        f"FROM metric_rows WHERE {value} IS NOT NULL AND run_id IN (SELECT provider_id FROM _pt_scope)"
    )


def _update_aggregate(measure: Measure, index: int) -> str:
    value = f"CASE WHEN m = {index} THEN v END"
    combine = LEGACY_ROLLOUT_BATCH_METRICS.get(measure.source.name)
    if combine == "sum":
        aggregate = f"SUM({value})"
    elif combine == "mean":
        aggregate = f"AVG({value})"
    else:
        aggregate = f"max_by({value}, CASE WHEN m = {index} THEN ts END)"  # last written
    return f"1 - {aggregate}" if measure.source.transform == "one_minus" else aggregate


def _rollouts_sql(owners: Mapping[str, Dimension | Measure | None], columns: frozenset[str]) -> str:
    selected = ["s.id AS run_id"]
    for name, owner in owners.items():
        if name not in columns or not isinstance(owner, Dimension | Measure):
            continue
        fact = owner.source.name
        if owner.source.kind == "trace_attribute":
            expression = _extract("t.metadata", "string", fact)
        elif owner.source.fallback_attribute is not None:
            fallback = _extract("t.metadata", "string", owner.source.fallback_attribute)
            expression = f"COALESCE(t.fact_{fact}, {fallback})"
        else:
            expression = "1" if fact == "trace_count" else f"t.fact_{fact}"
        selected.append(f"{expression} AS {_quote(name)}")
    return (
        f"SELECT {', '.join(selected)} FROM traces AS t JOIN _pt_scope AS s ON s.provider_id = t.run_id"
        " WHERE t.fact_state IS NOT NULL"
    )


# ---------------------------------------------------------------- assembly


def parse_statement(sql: str) -> exp.Query:
    try:
        statements = [item for item in sqlglot.parse(sql, read="doris") if item is not None]
    except SqlglotError as error:
        raise QueryError(f"SQL error: cannot parse the query: {error}") from error
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        raise QueryError("SQL error: give exactly one read-only SELECT (optionally with WITH)")
    statement = statements[0]
    own = {cte.alias_or_name for cte in statement.find_all(exp.CTE)}
    reserved = sorted(name for name in own if name in SEMANTIC_TABLES.values() or name.startswith(_RESERVED_PREFIX))
    if reserved:
        raise QueryError(f"SQL error: a CTE cannot use the name {reserved[0]!r}")
    return statement


def assemble(model: SemanticModel, sql: str, scope: CompiledScope | None) -> str:
    """The statement with the semantic views it reads prepended as CTEs, in Doris SQL."""

    statement = parse_statement(sql)
    views = build_views(model, plan(model, statement), scope)
    if not views:
        return statement.sql(dialect="doris")
    ctes = [
        exp.CTE(this=sqlglot.parse_one(body, read="doris"), alias=exp.TableAlias(this=exp.to_identifier(name)))
        for name, body in views
    ]
    with_ = statement.args.get("with_")
    if with_ is None:
        statement.set("with_", exp.With(expressions=ctes))
    else:
        with_.set("expressions", [*ctes, *with_.expressions])
    return statement.sql(dialect="doris")


__all__ = [
    "SEMANTIC_TABLES",
    "TRACKIO_TABLES",
    "assemble",
    "build_views",
    "column_name",
    "json_path",
    "literal",
    "parse_statement",
    "plan",
    "table_columns",
]
