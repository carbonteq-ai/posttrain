"""Semantic layer: model validation, queries, aggregation, SQL mode and describe."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from posttrain.tracking import (
    MetricPoint,
    MetricSeries,
    RunDetail,
    RunQuery,
    RunSummary,
    TraceAggregateResult,
    TraceFactsQuery,
)
from posttrain.tracking.models import TraceAggregateBucket
from posttrain_observatory.semantic_layer import (
    FRAMEWORK_MODEL,
    QueryError,
    SemanticModel,
    SemanticQuery,
    SqlQuery,
    describe_semantics,
    parse_formula,
    run_semantic_query,
    run_sql_query,
)
from posttrain_observatory.semantic_layer.formula import FormulaError
from posttrain_observatory.semantic_layer.model import Dimension, Entity, Measure, Metric, Source
from posttrain_observatory.telemetry import DEFAULT_TELEMETRY_DEFINITIONS

START = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)


def _summary(run_id: str, job_kind: str, work_package: str, status: str = "succeeded") -> RunSummary:
    return RunSummary(
        provider="fake",
        run_id=run_id,
        display_name=run_id,
        project_id="posttrain-lab",
        work_package_id=work_package,
        stage="train" if job_kind.startswith("train.") else "qualify",
        job_kind=job_kind,
        job_definition_version="fake@1",
        status=status,  # type: ignore[arg-type]
        started_at=START,
        finished_at=START + timedelta(seconds=600),
    )


def _settings(**resolved: Any) -> dict[str, Any]:
    return {"settings": {"selection_id": "settings-x", "resolved": resolved}}


class FakeReader:
    def __init__(self) -> None:
        self.summaries = {
            "sampo-a": _summary("sampo-a", "train.sampo", "train/sampo-8gb"),
            "sampo-b": _summary("sampo-b", "train.sampo", "train/sampo-8gb"),
            "grpo-kl": _summary("grpo-kl", "train.grpo", "train/vortex-kl"),
            "sft-1": _summary("sft-1", "train.sft", "train/sft"),
            "eval-1": _summary("eval-1", "eval.general", "eval/heldout"),
        }
        self.inputs = {
            "sampo-a": _settings(learning_rate=5e-5, beta=0.005),
            "sampo-b": _settings(learning_rate=5e-5, beta=0.005),
            "grpo-kl": _settings(learning_rate=5e-5, beta=0.005, algorithm="olmo3"),
            "sft-1": _settings(learning_rate=1e-5),
            "eval-1": {"model_source": {"source_run_id": "grpo-kl", "checkpoint_step": 20}},
        }
        self.series = {
            ("sampo-a", "train/step_time_seconds"): [(1, 261.939), (2, 137.999), (3, 186.663)],
            ("sampo-a", "train/rl/time/rollout_seconds"): [(1, 240.0), (2, 120.0), (3, 170.0)],
            ("sampo-a", "train/rl/entropy"): [(1, 0.5), (2, 0.4), (3, 0.45)],
            ("sampo-a", "train/rl/sparse_reward_projection_fraction"): [(1, 0.0), (2, 0.25), (3, 0.0)],
            ("sampo-b", "train/step_time_seconds"): [(1, 278.601), (2, 167.321), (3, 320.32)],
            ("sampo-b", "train/rl/entropy"): [(1, 0.53), (2, 0.43), (3, 0.48)],
            ("grpo-kl", "train/rl/entropy"): [(1, 0.18), (2, 0.19), (40, 0.25)],
            ("grpo-kl", "train/rl/kl"): [(1, 0.0004), (40, 0.02)],
            # Readers return one value per update (posttrain.tracking.logical_series).
            ("grpo-kl", "train/step_time_seconds"): [(1, 400.0), (2, 100.0)],
            ("grpo-kl", "train/rl/time/rollout_seconds"): [(1, 360.0), (2, 80.0)],
            ("grpo-kl", "train/rl/rollouts_attempted"): [(1, 96.0), (2, 64.0)],
            ("grpo-kl", "train/rl/tool_call_frequency"): [(1, 0.75), (2, 0.75)],
            ("eval-1", "eval/run/rollouts_failed"): [(0, 44.0)],
        }
        self.calls: list[tuple[str, Any]] = []

    async def list_runs(self, query: RunQuery) -> tuple[RunSummary, ...]:
        self.calls.append(("list_runs", query))
        runs = list(self.summaries.values())
        if query.work_package_id:
            runs = [run for run in runs if run.work_package_id == query.work_package_id]
        if query.job_kinds:
            runs = [run for run in runs if run.job_kind in query.job_kinds]
        return tuple(runs)

    async def run_detail(self, run_id: str) -> RunDetail:
        self.calls.append(("run_detail", run_id))
        return RunDetail(summary=self.summaries[run_id], resolved_inputs=self.inputs.get(run_id, {}))

    async def metric_series(
        self, run_id: str, names: tuple[str, ...], *, max_points: int
    ) -> tuple[tuple[MetricSeries, ...], bool]:
        self.calls.append(("metric_series", (run_id, names)))
        result = []
        for name in names:
            points = self.series.get((run_id, name))
            if points is not None:
                result.append(
                    MetricSeries(
                        name=name,
                        points=tuple(
                            MetricPoint(step=step, value=value, observed_at=START + timedelta(seconds=step * 100))
                            for step, value in points
                        ),
                    )
                )
        return tuple(result), False

    def trace_facts_available(self, run_id: str) -> bool:
        return True

    async def trace_facts(self, run_id: str, query: TraceFactsQuery) -> TraceAggregateResult:
        self.calls.append(("trace_facts", (run_id, query.group_by)))
        # Two truncated rollouts with reward 0 and 0.5, and three finished ones with 1, 1 and 0.5.
        groups = {True: [0.0, 0.5], False: [1.0, 1.0, 0.5]}
        buckets = []
        for truncated, rewards in groups.items():
            values: dict[str, float | None] = {}
            for aggregate in query.aggregates:
                if aggregate.measure != "task_reward":
                    continue
                values[f"{aggregate.operation}_task_reward"] = {
                    "sum": sum(rewards),
                    "count": float(len(rewards)),
                    "sum_squares": sum(value * value for value in rewards),
                }[aggregate.operation]
            dimensions: dict[str, Any] = {"is_truncated": truncated} if "is_truncated" in query.group_by else {}
            buckets.append(TraceAggregateBucket(dimensions=dimensions, trace_count=len(rewards), values=values))
        if "is_truncated" not in query.group_by:
            merged: dict[str, float | None] = {}
            for bucket in buckets:
                for key, value in bucket.values.items():
                    merged[key] = (merged.get(key) or 0.0) + (value or 0.0)
            buckets = [TraceAggregateBucket(dimensions={}, trace_count=5, values=merged)]
        return TraceAggregateResult(state="available", buckets=tuple(buckets))

    async def eval_tasks(self, run_id: str) -> tuple[Mapping[str, Any], ...]:
        return (
            {
                "key": "simple.gmail",
                "label": "Simple.gmail",
                "mean_reward": 1.0,
                "success_frequency": 1.0,
                "valid_repetitions": 3,
            },
            {
                "key": "finance.vendor",
                "label": "Finance.vendor",
                "mean_reward": 0.2,
                "success_frequency": 0.0,
                "valid_repetitions": 3,
            },
        )

    async def load_levels(self, run_id: str) -> tuple[Mapping[str, Any], ...]:
        return ()


def _query(reader: FakeReader, **kwargs: Any) -> Any:
    return asyncio.run(run_semantic_query(FRAMEWORK_MODEL, reader, SemanticQuery(**kwargs)))


def _rows(result: Any) -> list[dict[str, Any]]:
    names = [column.name for column in result.columns]
    return [dict(zip(names, row, strict=True)) for row in result.rows]


def test_framework_model_covers_every_telemetry_summary_metric() -> None:
    covered = {measure.source.name for measure in FRAMEWORK_MODEL.measures if measure.source.kind == "metric_series"}
    missing = {
        kind: sorted({field.metric for field in definition.summary_fields} - covered)
        for kind, definition in DEFAULT_TELEMETRY_DEFINITIONS.items()
    }
    assert {kind: names for kind, names in missing.items() if names} == {}


def test_model_rejects_duplicate_names_and_cross_entity_formulas() -> None:
    entity = Entity(name="update", description="u")
    run = Entity(name="run", description="r")
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
    run_measure = measure.model_copy(update={"name": "duration", "entity": "run"})
    with pytest.raises(ValueError, match="mixes entities"):
        SemanticModel(
            entities=(entity, run),
            dimensions=(),
            measures=(measure, run_measure),
            metrics=(
                Metric(name="bad", entity="update", label="b", description="b", formula="sum(reward) / sum(duration)"),
            ),
        )
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


def test_formulas_are_arithmetic_only() -> None:
    formula = parse_formula("sum(rollout_seconds) / sum(update_seconds) * 100 - -1")
    values = {("sum", "rollout_seconds"): 30.0, ("sum", "update_seconds"): 60.0}
    assert formula.evaluate(lambda aggregation, name: values[(aggregation, name)]) == 51.0
    assert parse_formula("sum(a) / sum(b)").evaluate(lambda aggregation, name: 0.0) is None
    for text in ('__import__("os")', "sum(a) + open(b)", "sum(a) ; 1"):
        with pytest.raises(FormulaError):
            parse_formula(text)


def test_update_query_reads_only_the_named_series_and_aggregates_per_run() -> None:
    reader = FakeReader()
    result = _query(reader, measures=("update_seconds",), by=("run.id",), where={"run.id": ["sampo-a", "sampo-b"]})
    assert result.grain == "update"
    assert {row["run.id"]: round(row["update_seconds"], 1) for row in _rows(result)} == {
        "sampo-a": 195.5,
        "sampo-b": 255.4,
    }
    series_calls = [call for call in reader.calls if call[0] == "metric_series"]
    assert {names for _, (_, names) in series_calls} == {("train/step_time_seconds",)}
    assert not any(call[0] == "list_runs" for call in reader.calls)


def test_last_and_first_follow_the_step_order() -> None:
    reader = FakeReader()
    rows = _rows(
        _query(reader, measures=("entropy:first", "entropy:last", "kl:max"), by=("run.id",), runs=("grpo-kl",))
    )
    assert rows == [{"run.id": "grpo-kl", "entropy_first": 0.18, "entropy_last": 0.25, "kl_max": 0.02}]


def test_by_step_returns_one_row_per_update_and_filters_steps() -> None:
    reader = FakeReader()
    rows = _rows(
        _query(
            reader,
            measures=("entropy",),
            by=("run.id", "update.step"),
            runs=("sampo-a",),
            where={"update.step": ">= 2"},
        )
    )
    assert rows == [
        {"run.id": "sampo-a", "update.step": 2, "entropy": 0.4},
        {"run.id": "sampo-a", "update.step": 3, "entropy": 0.45},
    ]


def test_metric_formula_and_transform() -> None:
    reader = FakeReader()
    rows = _rows(
        _query(reader, measures=("rollout_share", "step_reward_share:mean"), by=("run.id",), runs=("sampo-a",))
    )
    assert round(rows[0]["rollout_share"], 4) == round(530.0 / 586.601, 4)
    assert round(rows[0]["step_reward_share_mean"], 4) == round((1.0 + 0.75 + 1.0) / 3, 4)


def test_update_measures_and_rollout_share_by_update() -> None:
    reader = FakeReader()
    rows = _rows(
        _query(
            reader,
            measures=("rollout_seconds:sum", "rollouts_attempted:sum", "tool_call_rate:mean"),
            by=("update.step",),
            runs=("grpo-kl",),
            order_by=("update.step",),
        )
    )
    assert [row["rollout_seconds_sum"] for row in rows] == [360.0, 80.0]
    assert [row["rollouts_attempted_sum"] for row in rows] == [96.0, 64.0]
    assert [row["tool_call_rate_mean"] for row in rows] == [0.75, 0.75]
    share = _rows(_query(reader, measures=("rollout_share",), by=("run.id",), runs=("grpo-kl",)))
    assert share[0]["rollout_share"] == pytest.approx(440.0 / 500.0)


def test_run_dimensions_from_settings_and_wildcard_pushdown() -> None:
    reader = FakeReader()
    rows = _rows(
        _query(
            reader,
            measures=("entropy:last",),
            by=("run.id", "run.learning_rate", "run.kl_beta"),
            where={"run.work_package": "train/sampo-*"},
        )
    )
    assert {row["run.id"] for row in rows} == {"sampo-a", "sampo-b"}
    assert all(row["run.learning_rate"] == 5e-5 and row["run.kl_beta"] == 0.005 for row in rows)
    exact = FakeReader()
    _query(exact, measures=("entropy",), where={"run.work_package": "train/sampo-8gb"})
    assert next(call for call in exact.calls if call[0] == "list_runs")[1].work_package_id == "train/sampo-8gb"


def test_missing_measures_are_reported_not_invented() -> None:
    reader = FakeReader()
    result = _query(reader, measures=("entropy:last",), by=("run.id",), runs=("sampo-a", "sft-1"))
    assert {row["run.id"]: row["entropy_last"] for row in _rows(result)} == {"sampo-a": 0.45, "sft-1": None}
    assert any("sft-1" in note for note in result.unavailable)
    with pytest.raises(QueryError, match="no selected run provides 'entropy'"):
        _query(FakeReader(), measures=("entropy",), runs=("sft-1",))


def test_query_errors_name_the_problem() -> None:
    with pytest.raises(QueryError, match="unknown measure"):
        _query(FakeReader(), measures=("rewrd",), runs=("sampo-a",))
    with pytest.raises(QueryError, match="one entity besides run"):
        _query(FakeReader(), measures=("entropy", "rollouts"), runs=("sampo-a",))
    with pytest.raises(QueryError, match="cannot be aggregated"):
        _query(FakeReader(), measures=("rollout_reward:last",), runs=("sampo-a",))
    with pytest.raises(QueryError, match="narrow it"):
        asyncio.run(run_semantic_query(FRAMEWORK_MODEL, FakeReader(), SemanticQuery(measures=("entropy",)), max_runs=2))
    with pytest.raises(QueryError, match="did you mean 'reward'"):
        _query(FakeReader(), measures=("rewrd",), runs=("sampo-a",))


def test_questions_about_run_summaries_are_not_capped() -> None:
    reader = FakeReader()
    result = asyncio.run(
        run_semantic_query(
            FRAMEWORK_MODEL,
            reader,
            SemanticQuery(measures=("runs", "duration_seconds"), by=("run.job_kind",)),
            max_runs=2,
        )
    )
    counts = {row["run.job_kind"]: row["runs"] for row in _rows(result)}
    assert counts == {"train.sampo": 2, "train.grpo": 1, "train.sft": 1, "eval.general": 1}
    assert not any(call[0] == "run_detail" for call in reader.calls)


def test_rollouts_are_aggregated_by_the_backend_and_combined_exactly() -> None:
    reader = FakeReader()
    rows = _rows(
        _query(
            reader,
            measures=("rollouts", "rollout_reward", "rollout_reward:stddev"),
            by=("rollout.truncated",),
            runs=("sampo-a",),
        )
    )
    by_truncated = {row["rollout.truncated"]: row for row in rows}
    assert by_truncated[True]["rollouts"] == 2 and by_truncated[True]["rollout_reward"] == 0.25
    assert by_truncated[False]["rollouts"] == 3 and round(by_truncated[False]["rollout_reward"], 4) == round(2.5 / 3, 4)
    overall = _rows(_query(FakeReader(), measures=("rollouts", "rollout_reward"), runs=("sampo-a",)))
    assert overall == [{"rollouts": 5.0, "rollout_reward": 0.6}]
    assert ("trace_facts", ("sampo-a", ("is_truncated",))) in reader.calls


def test_eval_tasks_and_run_metrics() -> None:
    rows = _rows(
        _query(
            FakeReader(),
            measures=("task_reward", "success_rate"),
            by=("eval_task.task",),
            runs=("eval-1",),
            order_by=("-task_reward",),
        )
    )
    assert rows == [
        {"eval_task.task": "simple.gmail", "task_reward": 1.0, "success_rate": 1.0},
        {"eval_task.task": "finance.vendor", "task_reward": 0.2, "success_rate": 0.0},
    ]
    runs = _rows(
        _query(
            FakeReader(),
            measures=("eval_rollouts_failed",),
            by=("run.id", "run.parent_run", "run.parent_step"),
            runs=("eval-1",),
        )
    )
    assert runs == [
        {"run.id": "eval-1", "run.parent_run": "grpo-kl", "run.parent_step": 20, "eval_rollouts_failed": 44.0}
    ]


def test_sql_mode_reads_semantic_tables() -> None:
    reader = FakeReader()
    query = SqlQuery(
        sql=(
            "select r.id, r.learning_rate, u.step, u.entropy, "
            "avg(u.entropy) over (partition by u.run_id order by u.step rows 1 preceding) as smoothed "
            "from updates u join runs r on r.id = u.run_id order by r.id, u.step"
        ),
        runs=("sampo-a", "sampo-b"),
        load={"updates": ("entropy",)},
    )
    result = asyncio.run(run_sql_query(FRAMEWORK_MODEL, reader, query, max_runs=50))
    rows = _rows(result)
    assert len(rows) == 6 and rows[1]["smoothed"] == pytest.approx(0.45)
    assert {names for kind, (_, names) in ((c[0], c[1]) for c in reader.calls if c[0] == "metric_series")} >= {
        ("train/rl/entropy",)
    }


@pytest.mark.parametrize(
    "sql",
    [
        "attach database ':memory:' as other",
        "pragma table_info(runs)",
        "insert into runs(id) values ('x')",
        "create temp table t as select 1",
        "select 1; select 2",
    ],
)
def test_sql_mode_refuses_anything_but_reading(sql: str) -> None:
    with pytest.raises(QueryError, match="SQL error"):
        asyncio.run(run_sql_query(FRAMEWORK_MODEL, FakeReader(), SqlQuery(sql=sql, runs=("sampo-a",)), max_runs=50))


def test_sql_mode_stops_runaway_queries() -> None:
    sql = "with recursive n(i) as (select 1 union all select i + 1 from n) select count(*) from n"
    with pytest.raises(QueryError, match="longer than"):
        asyncio.run(
            run_sql_query(
                FRAMEWORK_MODEL, FakeReader(), SqlQuery(sql=sql, runs=("sampo-a",)), max_runs=50, timeout_seconds=0.2
            )
        )


def test_describe_lists_what_the_runs_provide() -> None:
    described = asyncio.run(describe_semantics(FRAMEWORK_MODEL, FakeReader(), runs=("sampo-a", "sft-1")))
    names = {item.measure.name: item.runs for item in described.measures}
    assert names["anchor_group_size"] == ("sampo-a",)
    assert names["token_accuracy"] == ("sft-1",)
    assert "throughput" not in names
    tables = {table.name: table.columns for table in described.sql_tables}
    assert "entropy" in tables["updates"] and "learning_rate" in tables["runs"]
    assert "rollout_share" in {metric.name for metric in described.metrics}
