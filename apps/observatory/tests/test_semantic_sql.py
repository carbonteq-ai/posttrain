"""The semantic layer running inside Trackio's storage, on a local project written by Posttrain itself.

These tests need a Trackio client with project SQL (carbonteq-trackio 0.31.5.post14.dev29 or later).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from posttrain_observatory.semantic_layer import (
    FRAMEWORK_MODEL,
    QueryError,
    SemanticQuery,
    SqlQuery,
    run_semantic_query,
    run_sql_query,
)
from posttrain_tracking_trackio import TrackioDataSource


def _rows(result: Any) -> list[dict[str, Any]]:
    names = [column.name for column in result.columns]
    return [dict(zip(names, row, strict=True)) for row in result.rows]


def _query(source: TrackioDataSource, **kwargs: Any) -> Any:
    return asyncio.run(run_semantic_query(FRAMEWORK_MODEL, source, SemanticQuery(**kwargs)))


def test_runs_come_from_configs_and_lifecycle(trackio_project: TrackioDataSource) -> None:
    rows = _rows(
        _query(
            trackio_project,
            measures=("duration_seconds",),
            by=(
                "run.id",
                "run.job_kind",
                "run.status",
                "run.error",
                "run.failed_phase",
                "run.failed_step",
                "run.learning_rate",
                "run.model",
            ),
        )
    )
    assert rows == [
        {
            "run.id": "grpo-a",
            "run.job_kind": "train.grpo",
            "run.status": "succeeded",
            "run.error": None,
            "run.failed_phase": None,
            "run.failed_step": None,
            "run.learning_rate": 5e-5,
            "run.model": "models/lfm@bf16",
            "duration_seconds": 620.0,
        },
        {
            "run.id": "sampo-b",
            "run.job_kind": "train.sampo",
            "run.status": "failed",
            "run.error": "OutOfMemoryError",
            "run.failed_phase": "actor_update",
            "run.failed_step": 2,
            "run.learning_rate": 1e-4,
            "run.model": "models/lfm@bf16",
            "duration_seconds": 300.0,
        },
    ]


def test_updates_apply_the_same_rules_as_the_python_readers(
    trackio_project: TrackioDataSource, trackio_engine: str
) -> None:
    result = _query(
        trackio_project,
        measures=("update_seconds:sum", "rollout_seconds:sum", "rollout_share", "entropy:last", "tool_call_rate"),
        by=("run.id",),
        runs=("grpo-a",),
    )
    (row,) = _rows(result)
    assert row["update_seconds_sum"] == 620.0
    assert row["rollout_seconds_sum"] == 530.0  # 360 + 80 + 90: the three batches of update 1 add up
    assert row["rollout_share"] == pytest.approx(530.0 / 620.0)
    assert row["entropy_last"] == 0.25
    assert row["tool_call_rate"] == 0.75
    assert result.sql is not None and "updates AS t" in result.sql and result.engine == trackio_engine

    # Equivalence: the SQL view and posttrain.tracking.logical_series (through the reader) agree per update.
    for metric, measure in (
        ("train/rl/time/rollout_seconds", "rollout_seconds"),
        ("train/rl/rollouts_attempted", "rollouts_attempted"),
        ("train/rl/tool_call_frequency", "tool_call_rate"),
        ("train/step_time_seconds", "update_seconds"),
    ):
        (series,) = asyncio.run(trackio_project.metric_series("grpo-a", (metric,)))
        python = {point.step: point.value for point in series.points}
        sql = {
            row["update.step"]: row[measure]
            for row in _rows(_query(trackio_project, measures=(measure,), by=("update.step",), runs=("grpo-a",)))
            if row[measure] is not None
        }
        assert sql == python, metric
        if measure == "rollouts_attempted":
            assert sql[1] == 136.0  # 48 + 48 + 40: batches with equal values are still separate batches


def test_resolved_update_counters_aggregate_by_their_meaning(trackio_project: TrackioDataSource) -> None:
    # Applied and attempt counters are cumulative (max), selected actions are per
    # update (sum), and skipped attempts are computed rather than recorded.
    (row,) = _rows(
        _query(
            trackio_project,
            measures=(
                "applied_updates",
                "optimizer_attempts",
                "skipped_optimizer_attempts",
                "selected_policy_actions",
                "selected_kl_actions",
                "advantage_nonzero_share",
            ),
            by=("run.id",),
            runs=("grpo-a",),
        )
    )
    assert row["applied_updates"] == 3.0
    assert row["optimizer_attempts"] == 4.0
    assert row["skipped_optimizer_attempts"] == 1.0
    assert row["selected_policy_actions"] == 600.0
    assert row["selected_kl_actions"] == 300.0
    assert row["advantage_nonzero_share"] == pytest.approx(0.5)
    with pytest.raises(QueryError, match="sum"):
        _query(trackio_project, measures=("applied_updates:sum",), by=("run.id",))


def test_rollouts_are_rows_so_any_aggregation_works(trackio_project: TrackioDataSource) -> None:
    rows = _rows(
        _query(
            trackio_project,
            measures=("rollouts", "rollout_reward", "output_tokens:p50"),
            by=("rollout.truncated",),
            runs={"run.job_kind": "train.grpo"},
        )
    )
    by_truncated = {row["rollout.truncated"]: row for row in rows}
    assert by_truncated[True]["rollouts"] == 1 and by_truncated[True]["rollout_reward"] == 0.0
    assert by_truncated[False]["rollouts"] == 3
    assert by_truncated[False]["rollout_reward"] == pytest.approx(1.75 / 3)
    assert by_truncated[False]["output_tokens_p50"] == 300.0


def test_rollouts_group_by_the_recorded_episode_ending(trackio_project: TrackioDataSource) -> None:
    rows = _rows(
        _query(
            trackio_project,
            measures=("rollouts", "rollout_reward"),
            by=("rollout.ending", "rollout.truncated"),
            runs={"run.job_kind": "train.grpo"},
        )
    )
    by_ending = {row["rollout.ending"]: row for row in rows}
    assert set(by_ending) == {"completed", "turn_limit", None}
    assert by_ending["turn_limit"]["rollouts"] == 1 and by_ending["turn_limit"]["rollout.truncated"] is True
    assert by_ending["completed"]["rollouts"] == 2 and by_ending["completed"]["rollout.truncated"] is False
    # rollout-1's ending is only a fact and rollout-3's only a trace attribute; both count.
    # A trace recorded before the label has no ending, and keeps its truncation fact.
    assert by_ending[None]["rollouts"] == 1 and by_ending[None]["rollout_reward"] == 0.5

    sql = SqlQuery(
        sql="select ending, count(*) as n from rollouts where ending is not null group by ending order by ending"
    )
    result = asyncio.run(run_sql_query(FRAMEWORK_MODEL, trackio_project, sql))
    assert [tuple(row) for row in result.rows] == [("completed", 2), ("turn_limit", 1)]


def test_sql_reads_every_run_of_the_project_or_the_scoped_ones(trackio_project: TrackioDataSource) -> None:
    query = SqlQuery(
        sql=(
            "select r.id, count(u.step) as updates, max_by(u.entropy, u.step) as last_entropy "
            "from runs r left join updates u on u.run_id = r.id group by r.id order by r.id"
        )
    )
    result = asyncio.run(run_sql_query(FRAMEWORK_MODEL, trackio_project, query))
    assert [list(row) for row in result.rows] == [["grpo-a", 3, 0.25], ["sampo-b", 1, None]]
    scoped = asyncio.run(
        run_sql_query(
            FRAMEWORK_MODEL,
            trackio_project,
            SqlQuery(sql="select id from runs", runs={"run.status": "failed"}),
        )
    )
    assert [list(row) for row in scoped.rows] == [["sampo-b"]]
    task_rewards = asyncio.run(
        run_sql_query(
            FRAMEWORK_MODEL,
            trackio_project,
            SqlQuery(
                sql=(
                    "select task, avg(rollout_reward) as reward, count(*) as valid from rollouts "
                    "where not truncated and not failed group by task order by task"
                ),
                runs=("grpo-a",),
            ),
        )
    )
    assert [list(row) for row in task_rewards.rows] == [["t1", 0.75, 2], ["t2", 0.25, 1]]


def test_mistakes_are_explained(trackio_project: TrackioDataSource) -> None:
    with pytest.raises(QueryError, match="did you mean entropy"):
        asyncio.run(run_sql_query(FRAMEWORK_MODEL, trackio_project, SqlQuery(sql="select entrophy from updates")))
    with pytest.raises(QueryError, match="read-only SELECT"):
        asyncio.run(run_sql_query(FRAMEWORK_MODEL, trackio_project, SqlQuery(sql="delete from runs")))
    with pytest.raises(QueryError, match="cannot use the name 'runs'"):
        asyncio.run(
            run_sql_query(FRAMEWORK_MODEL, trackio_project, SqlQuery(sql="with runs as (select 1) select * from runs"))
        )
    with pytest.raises(QueryError, match="unknown measure"):
        _query(trackio_project, measures=("rewrd",))


def test_statements_that_read_no_semantic_table_pass_through(
    trackio_project: TrackioDataSource, tmp_path: Path
) -> None:
    result = asyncio.run(
        run_sql_query(FRAMEWORK_MODEL, trackio_project, SqlQuery(sql="select count(*) as notes from run_notes"))
    )
    assert result.rows == ((0,),)
