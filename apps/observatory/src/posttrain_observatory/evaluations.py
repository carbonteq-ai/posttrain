"""Evaluation runs joined to the model they scored.

An evaluation run (job kinds ``eval.*``) scores one model on one evaluation
suite, its work package. When the model is a training checkpoint the run
records the training run and step (``run.parent_run``, ``run.parent_step``);
a base-model evaluation leaves both empty. These reads group evaluations by
that lineage so a training run can be compared with its base model, and two
training runs with each other, on the suites they share.

Score: mean rollout reward over attempts that did not fail. A truncated attempt
counts at its recorded reward (0 when none was recorded); failed attempts are
execution errors, not model outcomes, and are counted separately.

Behaviour: per-episode means over the same attempts, computed from the recorded
trace facts: turns (model calls), turns on episodes that ended on their own,
tool calls, output tokens and thinking tokens, plus how many episodes ended each
way. A mean is missing (``None``), never zero, when no attempt recorded the fact.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from pydantic import Field

from .models import ObservatoryModel, RunLocator
from .semantic_layer.query import SemanticQuery, SemanticResult, SqlQuery

type Query = Callable[[SemanticQuery | SqlQuery], Awaitable[SemanticResult]]

SCORE_DEFINITION = (
    "Mean rollout reward over attempts that did not fail; truncated attempts count at their recorded reward "
    "(0 when none). Failed attempts are execution errors and are counted separately."
)

_SCORE = "avg(case when o.failed then null else coalesce(o.rollout_reward, 0) end)"
_TRUNCATED = "sum(case when o.truncated then 1 else 0 end)"
_FAILED = "sum(case when o.failed then 1 else 0 end)"


def _mean(column: str, *, completed_only: bool = False) -> str:
    excluded = "o.failed or o.ending <> 'completed'" if completed_only else "o.failed"
    return f"avg(case when {excluded} then null else o.{column} end)"


_BEHAVIOUR = (
    f"{_mean('model_calls')} as turns, {_mean('model_calls', completed_only=True)} as turns_completed, "
    f"{_mean('tool_calls')} as tool_calls, {_mean('output_tokens')} as output_tokens, "
    f"{_mean('thinking_tokens')} as thinking_tokens"
)

INDEX_SQL = f"""
select r.id as run_id, r.job_kind as job_kind, r.work_package as suite, r.environment as environment,
       r.model as model, r.parent_run as parent_run, r.parent_step as parent_step, r.status as status,
       r.started_at as started_at, count(o.task) as attempts, {_SCORE} as score,
       {_TRUNCATED} as truncated, {_FAILED} as failed, {_BEHAVIOUR}
from runs r left join rollouts o on o.run_id = r.id
group by r.id, r.job_kind, r.work_package, r.environment, r.model, r.parent_run, r.parent_step, r.status,
         r.started_at
order by r.started_at
""".strip()

TASKS_SQL = f"""
select o.run_id as run_id, o.task as task, count(*) as attempts, {_SCORE} as score,
       {_TRUNCATED} as truncated, {_FAILED} as failed, {_BEHAVIOUR}
from rollouts o
group by o.run_id, o.task
order by o.task, o.run_id
""".strip()

ENDINGS_SQL = """
select o.run_id as run_id, o.ending as ending, count(*) as episodes
from rollouts o
group by o.run_id, o.ending
""".strip()

BEHAVIOUR_DEFINITION = (
    "Per-episode means over attempts that did not fail: turns are model calls; turns (completed) counts only "
    "episodes that ended on their own; tool calls, output and thinking tokens come from the recorded trace facts. "
    "A missing value means no attempt recorded it."
)


class EvaluationRecord(ObservatoryModel):
    """One evaluation run: the model it scored, its suite and its score."""

    run_key: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    job_kind: str = Field(min_length=1)
    suite: str | None = None
    environment: str | None = None
    model: str | None = None
    parent_run: str | None = None
    parent_run_key: str | None = None
    parent_step: int | None = None
    status: str | None = None
    started_at: str | None = None
    attempts: int = 0
    score: float | None = None
    truncated: int = 0
    failed: int = 0
    turns: float | None = None
    turns_completed: float | None = None
    tool_calls: float | None = None
    output_tokens: float | None = None
    thinking_tokens: float | None = None
    endings: dict[str, int] = Field(default_factory=dict)


class EvaluationIndex(ObservatoryModel):
    source_id: str = Field(min_length=1)
    score_definition: str = SCORE_DEFINITION
    behaviour_definition: str = BEHAVIOUR_DEFINITION
    records: tuple[EvaluationRecord, ...] = ()


class EvaluationTaskScore(ObservatoryModel):
    run_key: str = Field(min_length=1)
    task: str
    attempts: int = 0
    score: float | None = None
    truncated: int = 0
    failed: int = 0
    turns: float | None = None
    turns_completed: float | None = None
    tool_calls: float | None = None
    output_tokens: float | None = None
    thinking_tokens: float | None = None


class EvaluationTaskScores(ObservatoryModel):
    source_id: str = Field(min_length=1)
    score_definition: str = SCORE_DEFINITION
    scores: tuple[EvaluationTaskScore, ...] = ()


def _rows(result: SemanticResult) -> list[dict[str, Any]]:
    names = [column.name for column in result.columns]
    return [dict(zip(names, row, strict=True)) for row in result.rows]


def _text(value: Any) -> str | None:
    return None if value is None or value == "" else str(value)


def _int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _behaviour(row: dict[str, Any]) -> dict[str, float | None]:
    return {
        name: _float(row.get(name))
        for name in ("turns", "turns_completed", "tool_calls", "output_tokens", "thinking_tokens")
    }


async def evaluation_index(source_id: str, query: Query) -> EvaluationIndex:
    """Every evaluation run in the source with its lineage and score."""
    result = await query(SqlQuery(sql=INDEX_SQL, runs={"run.job_kind": "eval.*"}))
    endings: dict[str, dict[str, int]] = {}
    for row in _rows(await query(SqlQuery(sql=ENDINGS_SQL, runs={"run.job_kind": "eval.*"}))):
        run_id, ending, count = _text(row.get("run_id")), _text(row.get("ending")), _int(row.get("episodes"))
        if run_id is not None and ending is not None and count:
            endings.setdefault(run_id, {})[ending] = count
    records = []
    for row in _rows(result):
        run_id = _text(row.get("run_id"))
        if run_id is None:
            continue
        parent = _text(row.get("parent_run"))
        records.append(
            EvaluationRecord(
                run_key=RunLocator(source_id=source_id, run_id=run_id).key,
                run_id=run_id,
                job_kind=_text(row.get("job_kind")) or "eval",
                suite=_text(row.get("suite")),
                environment=_text(row.get("environment")),
                model=_text(row.get("model")),
                parent_run=parent,
                parent_run_key=RunLocator(source_id=source_id, run_id=parent).key if parent else None,
                parent_step=_int(row.get("parent_step")) if parent else None,
                status=_text(row.get("status")),
                started_at=_text(row.get("started_at")),
                attempts=_int(row.get("attempts")) or 0,
                score=_float(row.get("score")),
                truncated=_int(row.get("truncated")) or 0,
                failed=_int(row.get("failed")) or 0,
                **_behaviour(row),
                endings=endings.get(run_id, {}),
            )
        )
    return EvaluationIndex(source_id=source_id, records=tuple(records))


async def evaluation_tasks(source_id: str, run_ids: Sequence[str], query: Query) -> EvaluationTaskScores:
    """Per-task scores of the given evaluation runs."""
    ids = tuple(dict.fromkeys(run_id for run_id in run_ids if run_id))
    if not ids:
        return EvaluationTaskScores(source_id=source_id)
    result = await query(SqlQuery(sql=TASKS_SQL, runs=ids))
    scores = []
    for row in _rows(result):
        run_id = _text(row.get("run_id"))
        if run_id is None or run_id not in ids:
            continue
        scores.append(
            EvaluationTaskScore(
                run_key=RunLocator(source_id=source_id, run_id=run_id).key,
                task=_text(row.get("task")) or "(no task)",
                attempts=_int(row.get("attempts")) or 0,
                score=_float(row.get("score")),
                truncated=_int(row.get("truncated")) or 0,
                failed=_int(row.get("failed")) or 0,
                **_behaviour(row),
            )
        )
    return EvaluationTaskScores(source_id=source_id, scores=tuple(scores))
