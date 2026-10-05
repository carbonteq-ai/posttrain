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
            measures=("actor_share", "entropy:last", "update_seconds:p90"),
            by=("run.id",),
            where={"update.step": ">= 10", "run.work_package": "train/lfm*"},
            order_by=("-actor_share",),
            limit=5,
        ),
    )
    assert compiled.grain == "update"
    assert compiled.sql.splitlines() == [
        "SELECT r.`id` AS `run.id`, (sum(actor_seconds) / sum(update_seconds)) AS `actor_share`, "
        "max_by(t.`entropy`, CASE WHEN t.`entropy` IS NOT NULL THEN t.`step` END) AS `entropy_last`, "
        "percentile(t.`update_seconds`, 0.9) AS `update_seconds_p90`",
        "FROM runs AS r LEFT JOIN updates AS t ON t.run_id = r.id",
        "WHERE t.`step` >= 10 AND r.`work_package` LIKE 'train/lfm%'",
        "GROUP BY r.`id`",
        "ORDER BY `actor_share` DESC NULLS LAST",
        "LIMIT 6",
    ]
    # Population values are per collection, ordered by the collection's first update.
    collections = compile_query(
        FRAMEWORK_MODEL,
        SemanticQuery(measures=("rollout_share", "reward:last"), by=("run.id",), where={"collection.step": ">= 10"}),
    )
    assert collections.grain == "collection"
    assert "FROM runs AS r LEFT JOIN collections AS t ON t.run_id = r.id" in collections.sql
    assert "max_by(t.`reward`, CASE WHEN t.`reward` IS NOT NULL THEN t.`step` END)" in collections.sql
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
    assert [table.name for table in everything.sql_tables][:4] == ["runs", "collections", "updates", "rollouts"]
    sampo = describe_semantics(FRAMEWORK_MODEL, job_kinds=("train.sampo",))
    names = {measure.name for measure in sampo.measures}
    assert "step_reward_share" in names and "preference_accuracy" not in names


def test_episode_endings_are_a_rollout_dimension_and_collection_rates() -> None:
    from posttrain.common import EPISODE_ENDINGS

    ending = FRAMEWORK_MODEL.dimension("rollout.ending")
    assert ending.source == Source(kind="trace_fact", name="episode_ending", fallback_attribute="episode_ending")
    for name in EPISODE_ENDINGS:
        assert name in ending.description
        measure = FRAMEWORK_MODEL.measure(f"ending_{name}_rate")
        assert measure.entity == "collection" and measure.source.name == f"train/rl/ending_{name}_rate"
    sql = assemble(FRAMEWORK_MODEL, "select ending, count(*) from rollouts group by ending", None)
    assert (
        """COALESCE(t.fact_episode_ending, JSON_EXTRACT_STRING(t.metadata, '$."episode_ending"')) AS `ending`""" in sql
    )
    with pytest.raises(ValueError, match="only a trace_fact source"):
        Source(kind="metric_series", name="train/rl/reward_mean", fallback_attribute="episode_ending")


def test_catalog_describes_trl_per_round_active_sampling_metrics() -> None:
    from posttrain_observatory.metric_catalog import ACTIVE_SAMPLING_CATALOG_ROUNDS

    # The lab's largest max_candidate_batches is 10; every round a run may record is described.
    assert ACTIVE_SAMPLING_CATALOG_ROUNDS >= 10
    for round_index in range(1, ACTIVE_SAMPLING_CATALOG_ROUNDS + 1):
        for kind in ("requested", "generated", "retained"):
            entry = CATALOG_BY_METRIC[f"train/rl/active_sampling_round_{round_index}_{kind}_groups"]
            assert entry.entity is None  # described, not a semantic measure: the round is in the name
            assert entry.help().label == f"Active-sampling round {round_index} {kind} groups"


def test_catalog_describes_resolved_update_counters_with_their_aggregation() -> None:
    from posttrain_observatory.metric_catalog import RESOLVED_UPDATE_KINDS

    measures = {
        measure.source.name: measure for measure in FRAMEWORK_MODEL.measures if measure.source.kind == "metric_series"
    }
    for metric in ("train/rl/applied_optimizer_updates", "train/rl/optimizer_attempts"):
        entry = CATALOG_BY_METRIC[metric]
        # Cumulative counters: a run total is their maximum, never a sum of points.
        assert entry.aggregation == "max" and entry.allowed is not None and "sum" not in entry.allowed
        assert measures[metric].aggregation == "max"
    for metric in ("train/rl/selected_policy_actions", "train/rl/selected_kl_actions"):
        assert CATALOG_BY_METRIC[metric].aggregation == "sum"
    nonzero = CATALOG_BY_METRIC["train/rl/advantage_nonzero_fraction"]
    assert nonzero.unit == "ratio" and nonzero.allowed is not None and "sum" not in nonzero.allowed
    for kind in ("reserved", "generated", "retained", "unused"):
        entry = CATALOG_BY_METRIC[f"train/rl/active_sampling_candidate_groups_{kind}"]
        assert entry.aggregation == "sum" and "train.sampo" in entry.job_kinds
    for metric in (
        "train/rl/loss",
        "train/rl/kl_loss",
        "train/rl/advantage_mean",
        "train/rl/advantage_abs_mean",
        "train/rl/advantage_std",
        "train/loss_scale",
        "train/optimizer_step_skipped",
        "train/optimizer_steps_skipped",
    ):
        assert CATALOG_BY_METRIC[metric].job_kinds == RESOLVED_UPDATE_KINDS, metric
    skipped = next(metric for metric in FRAMEWORK_MODEL.metrics if metric.name == "skipped_optimizer_attempts")
    assert skipped.formula == "max(optimizer_attempts) - max(applied_updates)"
