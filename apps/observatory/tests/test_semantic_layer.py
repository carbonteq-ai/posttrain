"""The semantic model, the short-form compiler and view planning (no storage needed).

Execution inside Trackio's storage is tested in `test_semantic_sql.py`.
"""

from __future__ import annotations

import pytest
from posttrain_observatory.metric_catalog import CATALOG_BY_METRIC, METRIC_CATALOG
from posttrain_observatory.semantic_layer import (
    FRAMEWORK_MODEL,
    QueryError,
    SemanticModel,
    SemanticQuery,
    assemble,
    compile_query,
    compile_scope,
    describe_semantics,
    plan,
)
from posttrain_observatory.semantic_layer.model import Dimension, Entity, Measure, Source
from posttrain_observatory.semantic_layer.views import SEMANTIC_TABLES, parse_statement, table_columns
from posttrain_observatory.telemetry import DEFAULT_TELEMETRY_DEFINITIONS


def test_one_catalog_describes_every_metric_job_views_show() -> None:
    shown = {metric for definition in DEFAULT_TELEMETRY_DEFINITIONS.values() for metric in definition.metric_names}
    assert shown - set(CATALOG_BY_METRIC) == set(), "every metric a job view shows is described in the catalog"
    measured = {measure.source.name for measure in FRAMEWORK_MODEL.measures if measure.source.kind == "metric_series"}
    assert measured == {entry.metric for entry in METRIC_CATALOG if entry.entity is not None}
    summaries = {
        field.metric for definition in DEFAULT_TELEMETRY_DEFINITIONS.values() for field in definition.summary_fields
    }
    assert summaries - measured == set(), "every summary field can be queried"


def test_model_rejects_duplicate_names_and_unprefixed_dimensions() -> None:
    entity = Entity(name="update", description="u")
    measure = Measure(
        name="reward",
        entity="update",
        label="r",
        description="r",
        source=Source(kind="metric_series", name="x"),
        job_kinds=("train.grpo",),
    )
    with pytest.raises(ValueError, match="unique"):
        SemanticModel(entities=(entity,), dimensions=(), measures=(measure, measure))
    with pytest.raises(ValueError, match="prefixed"):
        SemanticModel(
            entities=(entity,),
            dimensions=(
                Dimension(
                    name="step",
                    entity="update",
                    type="integer",
                    description="s",
                    source=Source(kind="derived", name="step"),
                ),
            ),
            measures=(),
        )


def test_metric_formulas_read_only_measures_of_their_own_table() -> None:
    for metric in FRAMEWORK_MODEL.metrics:
        table = SEMANTIC_TABLES[metric.entity]
        planned = plan(FRAMEWORK_MODEL, parse_statement(f"SELECT {metric.formula} FROM {table}"))
        assert set(planned) == {table}, metric.name
        owners = table_columns(FRAMEWORK_MODEL)[table]
        assert all(isinstance(owners[column], Measure) for column in planned[table]), metric.name


def test_short_form_compiles_to_readable_doris_sql() -> None:
    compiled = compile_query(
        FRAMEWORK_MODEL,
        SemanticQuery(
            measures=("rollout_share", "entropy:last", "update_seconds:p90"),
            by=("run.id",),
            where={"update.step": ">= 10", "run.work_package": "train/lfm*"},
            order_by=("-rollout_share",),
            limit=5,
        ),
    )
    assert compiled.grain == "update"
    assert compiled.sql.splitlines() == [
        "SELECT r.`id` AS `run.id`, (sum(rollout_seconds) / sum(update_seconds)) AS `rollout_share`, "
        "max_by(t.`entropy`, CASE WHEN t.`entropy` IS NOT NULL THEN t.`step` END) AS `entropy_last`, "
        "percentile(t.`update_seconds`, 0.9) AS `update_seconds_p90`",
        "FROM runs AS r LEFT JOIN updates AS t ON t.run_id = r.id",
        "WHERE t.`step` >= 10 AND r.`work_package` LIKE 'train/lfm%'",
        "GROUP BY r.`id`",
        "ORDER BY `rollout_share` DESC NULLS LAST",
        "LIMIT 6",
    ]
    rollouts = compile_query(FRAMEWORK_MODEL, SemanticQuery(measures=("rollouts",), by=("rollout.task",)))
    assert "FROM rollouts AS t JOIN runs AS r ON r.id = t.run_id" in rollouts.sql


def test_short_form_mistakes_are_explained() -> None:
    with pytest.raises(QueryError, match="did you mean 'reward'"):
        compile_query(FRAMEWORK_MODEL, SemanticQuery(measures=("rewrd",)))
    with pytest.raises(QueryError, match="one entity besides run"):
        compile_query(FRAMEWORK_MODEL, SemanticQuery(measures=("entropy", "rollouts")))
    with pytest.raises(QueryError, match="is a measure"):
        compile_query(FRAMEWORK_MODEL, SemanticQuery(measures=("entropy",), by=("reward",)))
    with pytest.raises(QueryError, match="run measures can only be queried at run grain"):
        compile_query(FRAMEWORK_MODEL, SemanticQuery(measures=("duration_seconds",), by=("update.step",)))


def test_run_scopes_become_predicates_on_run_columns() -> None:
    by_id = compile_scope(FRAMEWORK_MODEL, ("a", "b'c"))
    assert by_id is not None and by_id.predicate == "id IN ('a', 'b''c')"
    by_filters = compile_scope(FRAMEWORK_MODEL, {"run.job_kind": "train.sampo", "learning_rate": "<= 1e-4"})
    assert by_filters is not None
    assert by_filters.predicate == "`job_kind` = 'train.sampo' AND `learning_rate` <= 0.0001"
    assert by_filters.columns == {"job_kind", "learning_rate"}
    with pytest.raises(QueryError, match="not a run dimension"):
        compile_scope(FRAMEWORK_MODEL, {"update.step": 3})


def test_planning_follows_columns_through_ctes_subqueries_and_stars() -> None:
    statement = parse_statement(
        "with u as (select run_id, avg(entropy) as e from updates group by run_id) "
        "select r.*, u.e, (select count(*) from rollouts where rollouts.run_id = r.id and truncated) as t "
        "from runs r join u on u.run_id = r.id"
    )
    planned = plan(FRAMEWORK_MODEL, statement)
    assert planned["updates"] == {"run_id", "entropy"}
    assert planned["rollouts"] == {"run_id", "truncated"}
    assert planned["runs"] == set(table_columns(FRAMEWORK_MODEL)["runs"])
    with pytest.raises(QueryError, match="did you mean entropy"):
        plan(FRAMEWORK_MODEL, parse_statement("select entrophy from updates"))


def test_views_contain_only_what_a_statement_reads() -> None:
    sql = assemble(FRAMEWORK_MODEL, "select run_id, avg(entropy) from updates group by run_id", None)
    assert '"train/rl/entropy"' in sql and '"train/step_time_seconds"' in sql
    assert '"train/rl/kl"' not in sql and "traces" not in sql and "_pt_lifecycle" not in sql
    raw = assemble(FRAMEWORK_MODEL, "select count(*) from metric_rows", None)
    assert raw == "SELECT COUNT(*) FROM metric_rows"


def test_describe_lists_the_tables_and_narrows_by_job_kind() -> None:
    everything = describe_semantics(FRAMEWORK_MODEL)
    assert [table.name for table in everything.sql_tables][:3] == ["runs", "updates", "rollouts"]
    sampo = describe_semantics(FRAMEWORK_MODEL, job_kinds=("train.sampo",))
    names = {measure.name for measure in sampo.measures}
    assert "step_reward_share" in names and "preference_accuracy" not in names


def test_episode_endings_are_a_rollout_dimension_and_update_rates() -> None:
    from posttrain.common import EPISODE_ENDINGS

    ending = FRAMEWORK_MODEL.dimension("rollout.ending")
    assert ending.source == Source(kind="trace_attribute", name="episode_ending")
    for name in EPISODE_ENDINGS:
        assert name in ending.description
        measure = FRAMEWORK_MODEL.measure(f"ending_{name}_rate")
        assert measure.entity == "update" and measure.source.name == f"train/rl/ending_{name}_rate"
    sql = assemble(FRAMEWORK_MODEL, "select ending, count(*) from rollouts group by ending", None)
    assert """JSON_EXTRACT_STRING(t.metadata, '$."episode_ending"') AS `ending`""" in sql
