from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from typing import Any

from posttrain_observatory.evaluations import evaluation_index, evaluation_tasks
from posttrain_observatory.models import RunLocator
from posttrain_observatory.semantic_layer.query import SemanticQuery, SemanticResult, SqlQuery


def _result(names: list[str], rows: list[list[Any]]) -> SemanticResult:
    return SemanticResult.model_validate_json(
        json.dumps({"columns": [{"name": name, "kind": "value"} for name in names], "rows": rows, "grain": "sql"})
    )


class FakeQuery:
    def __init__(self, result: SemanticResult) -> None:
        self.result = result
        self.queries: list[SemanticQuery | SqlQuery] = []

    async def __call__(self, query: SemanticQuery | SqlQuery) -> SemanticResult:
        self.queries.append(query)
        return self.result


INDEX_COLUMNS = [
    "run_id",
    "job_kind",
    "suite",
    "environment",
    "model",
    "parent_run",
    "parent_step",
    "status",
    "started_at",
    "attempts",
    "score",
    "truncated",
    "failed",
]


def test_index_reads_every_eval_run_with_its_checkpoint_lineage() -> None:
    fake = FakeQuery(
        _result(
            INDEX_COLUMNS,
            [
                [
                    "eval-base",
                    "eval.general",
                    "eval/suite-a",
                    "env",
                    "models/m",
                    None,
                    None,
                    "succeeded",
                    "2026-09-27T05:17:07Z",
                    60,
                    0.589,
                    0,
                    0,
                ],
                [
                    "eval-step100",
                    "eval.general",
                    "eval/suite-a",
                    "env",
                    "models/m",
                    "train-a",
                    100,
                    "succeeded",
                    "2026-09-27T04:45:11Z",
                    60,
                    0.595,
                    1,
                    0,
                ],
                [
                    "eval-empty",
                    "eval.general",
                    "eval/suite-a",
                    "env",
                    "models/m",
                    "",
                    "",
                    "failed",
                    None,
                    0,
                    None,
                    None,
                    None,
                ],
            ],
        )
    )
    index = asyncio.run(evaluation_index("src", fake))

    query = fake.queries[0]
    assert isinstance(query, SqlQuery)
    assert query.runs == {"run.job_kind": "eval.*"}
    base, step, empty = index.records
    assert base.parent_run is None and base.parent_step is None
    assert base.run_key == RunLocator(source_id="src", run_id="eval-base").key
    assert step.parent_run == "train-a" and step.parent_step == 100
    assert step.parent_run_key == RunLocator(source_id="src", run_id="train-a").key
    assert step.score == 0.595 and step.truncated == 1
    assert empty.parent_run is None and empty.score is None and empty.attempts == 0 and empty.status == "failed"


def test_task_scores_are_read_only_for_the_requested_runs() -> None:
    fake = FakeQuery(
        _result(
            ["run_id", "task", "attempts", "score", "truncated", "failed"],
            [
                ["eval-base", "hr.onboarding", 3, 0.5, 0, 0],
                ["eval-step100", "hr.onboarding", 3, 0.75, 1, 0],
                ["someone-else", "hr.onboarding", 3, 1.0, 0, 0],
            ],
        )
    )
    scores = asyncio.run(evaluation_tasks("src", ["eval-base", "eval-step100", "eval-base"], fake))

    query = fake.queries[0]
    assert isinstance(query, SqlQuery)
    assert query.runs == ("eval-base", "eval-step100")
    assert [(item.run_key, item.score) for item in scores.scores] == [
        (RunLocator(source_id="src", run_id="eval-base").key, 0.5),
        (RunLocator(source_id="src", run_id="eval-step100").key, 0.75),
    ]
    assert asyncio.run(evaluation_tasks("src", [], fake)).scores == ()


def test_decimal_counts_from_sql_become_integers() -> None:
    row = [
        "eval-a",
        "eval.general",
        "s",
        None,
        "m",
        "train-a",
        Decimal("20"),
        "succeeded",
        None,
        Decimal("60"),
        Decimal("0.5"),
        Decimal("2"),
        Decimal("0"),
    ]
    fake = FakeQuery(SemanticResult(columns=_result(INDEX_COLUMNS, []).columns, rows=(tuple(row),), grain="sql"))
    (record,) = asyncio.run(evaluation_index("src", fake)).records
    assert (record.parent_step, record.attempts, record.score, record.truncated) == (20, 60, 0.5, 2)


class RoutingFakeQuery:
    """Answers the index query and the endings query with separate results."""

    def __init__(self, index: SemanticResult, endings: SemanticResult) -> None:
        self.index = index
        self.endings = endings
        self.queries: list[SemanticQuery | SqlQuery] = []

    async def __call__(self, query: SemanticQuery | SqlQuery) -> SemanticResult:
        self.queries.append(query)
        assert isinstance(query, SqlQuery)
        return self.endings if "o.ending as ending" in query.sql else self.index


BEHAVIOUR_COLUMNS = ["turns", "turns_completed", "tool_calls", "output_tokens", "thinking_tokens"]


def test_index_reports_episode_behaviour_and_endings_beside_the_score() -> None:
    base = ["eval-base", "eval.general", "eval/suite-a", "env", "models/m", None, None, "succeeded", None, 100, 0.607]
    step = [
        "eval-step50",
        "eval.general",
        "eval/suite-a",
        "env",
        "models/m",
        "train-a",
        50,
        "succeeded",
        None,
        100,
        0.636,
    ]
    fake = RoutingFakeQuery(
        _result(
            INDEX_COLUMNS + BEHAVIOUR_COLUMNS,
            [
                [*base, 3, 0, 6.2, 5.9, 9.0, 3400.0, None],
                [*step, 0, 0, 5.51, 5.51, 9.1, 3127.89, 1840.0],
            ],
        ),
        _result(
            ["run_id", "ending", "episodes"],
            [
                ["eval-base", "completed", 97],
                ["eval-base", "turn_limit", 3],
                ["eval-step50", "completed", 100],
                ["eval-step50", None, 4],
                ["someone-else", "completed", 0],
            ],
        ),
    )
    index = asyncio.run(evaluation_index("src", fake))

    assert len(fake.queries) == 2 and all(isinstance(query, SqlQuery) for query in fake.queries)
    assert all(query.runs == {"run.job_kind": "eval.*"} for query in fake.queries if isinstance(query, SqlQuery))
    base_record, step_record = index.records
    assert (base_record.turns, base_record.turns_completed, base_record.tool_calls) == (6.2, 5.9, 9.0)
    assert base_record.output_tokens == 3400.0
    # A fact no attempt recorded stays missing; it is never reported as zero.
    assert base_record.thinking_tokens is None
    assert base_record.endings == {"completed": 97, "turn_limit": 3}
    assert (step_record.turns, step_record.tool_calls, step_record.thinking_tokens) == (5.51, 9.1, 1840.0)
    assert step_record.endings == {"completed": 100}
    assert "turns" in index.behaviour_definition


def test_task_scores_carry_the_same_behaviour_means() -> None:
    fake = FakeQuery(
        _result(
            ["run_id", "task", "attempts", "score", "truncated", "failed", *BEHAVIOUR_COLUMNS],
            [["eval-base", "hr.onboarding", 3, 0.5, 0, 0, 7.0, None, 11.0, 4100.0, None]],
        )
    )
    (score,) = asyncio.run(evaluation_tasks("src", ["eval-base"], fake)).scores
    assert (score.turns, score.turns_completed, score.tool_calls, score.output_tokens) == (7.0, None, 11.0, 4100.0)
    assert score.thinking_tokens is None
