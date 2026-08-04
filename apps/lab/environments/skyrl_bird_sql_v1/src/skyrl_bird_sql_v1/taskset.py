"""Native Verifiers v1 taskset for SkyRL-SQL over ReViSQL/BIRD."""

from __future__ import annotations

from typing import Any, Literal

import verifiers.v1 as vf

from .assets import database_path, load_rows
from .protocol import ProtocolError, corrective_observation, parse_turn
from .scoring import parse_grading_method, results_match
from .sqlite import SQLExecutionError, execute_readonly_async, format_observation, render_schema

SYSTEM_PROMPT = """You are a database analyst. Solve the user's question against the supplied SQLite schema.

Every assistant turn must contain a non-empty <think>...</think> block followed by exactly one action:
- use <sql>...</sql> to run one exploratory read-only SQL query and receive an <observation>;
- use <solution>...</solution> to submit the final SQL query and end the task.

Do not write or modify data. Do not invent <observation> messages. Return SQL inside the protocol tags, not Markdown fences."""


def _field(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    raise KeyError(f"ReViSQL row is missing all expected fields: {', '.join(names)}")


class SkyRLBirdSQLData(vf.TaskData):
    question_id: str
    db_id: str
    question: str
    evidence: str
    gold_sql: str
    grading_method: str


class SkyRLBirdSQLState(vf.State):
    user_finished: bool = False
    terminal_reason: str = "running"
    turn_count: int = 0
    exploratory_query_count: int = 0
    format_valid: bool = True
    final_query: str | None = None
    prediction_rows: tuple[tuple[Any, ...], ...] = ()
    prediction_columns: tuple[str, ...] = ()
    gold_rows: tuple[tuple[Any, ...], ...] = ()
    gold_columns: tuple[str, ...] = ()
    prediction_executed: bool = False
    gold_executed: bool = False
    execution_correct: bool = False
    error_category: str | None = None
    observation_truncated: bool = False


class SkyRLBirdSQLUserConfig(vf.UserConfig):
    max_turns: int = 5
    query_timeout_seconds: float = 10.0
    maximum_result_rows: int = 100_000
    observation_rows: int = 50
    observation_cell_characters: int = 200


class SkyRLBirdSQLTaskConfig(vf.TaskConfig):
    user: SkyRLBirdSQLUserConfig = SkyRLBirdSQLUserConfig()


class SkyRLBirdSQLUser(vf.User[SkyRLBirdSQLUserConfig, SkyRLBirdSQLState]):
    async def setup_task(self, task: Any) -> None:
        self.database = database_path(str(task.db_id))
        self.gold_sql = str(task.gold_sql)

    async def respond(self, message: str) -> vf.Messages:
        self.state.turn_count += 1
        try:
            turn = parse_turn(message)
        except ProtocolError as error:
            self.state.format_valid = False
            self.state.error_category = "protocol"
            if self.state.turn_count >= self.config.max_turns:
                self.state.user_finished = True
                self.state.terminal_reason = "malformed"
                return []
            return [vf.UserMessage(content=corrective_observation(str(error)))]

        if turn.kind == "solution":
            self.state.final_query = turn.query
            self.state.user_finished = True
            await self._execute_final(turn.query)
            return []

        self.state.exploratory_query_count += 1
        try:
            result = await execute_readonly_async(
                self.database,
                turn.query,
                timeout_seconds=self.config.query_timeout_seconds,
                maximum_rows=self.config.maximum_result_rows,
            )
            observation, truncated = format_observation(
                result,
                maximum_rows=self.config.observation_rows,
                maximum_cell_characters=self.config.observation_cell_characters,
            )
            self.state.observation_truncated |= truncated
        except SQLExecutionError as error:
            self.state.error_category = "exploratory_sql"
            observation = f"SQL error: {error}"

        if self.state.turn_count >= self.config.max_turns:
            self.state.user_finished = True
            self.state.terminal_reason = "missing_final"
            return []
        return [vf.UserMessage(content=f"<observation>{observation}</observation>")]

    async def _execute_final(self, prediction: str) -> None:
        try:
            predicted = await execute_readonly_async(
                self.database,
                prediction,
                timeout_seconds=self.config.query_timeout_seconds,
                maximum_rows=self.config.maximum_result_rows,
            )
            self.state.prediction_rows = predicted.rows
            self.state.prediction_columns = predicted.columns
            self.state.prediction_executed = True
        except SQLExecutionError:
            self.state.error_category = "prediction_sql"
            self.state.terminal_reason = "execution_failed"
            return
        try:
            gold = await execute_readonly_async(
                self.database,
                self.gold_sql,
                timeout_seconds=self.config.query_timeout_seconds,
                maximum_rows=self.config.maximum_result_rows,
            )
            self.state.gold_rows = gold.rows
            self.state.gold_columns = gold.columns
            self.state.gold_executed = True
        except SQLExecutionError:
            self.state.error_category = "gold_sql"
            self.state.terminal_reason = "gold_execution_failed"
            return
        self.state.terminal_reason = "submitted"


class SkyRLBirdSQLTask(vf.Task[SkyRLBirdSQLData, SkyRLBirdSQLState, SkyRLBirdSQLTaskConfig]):
    user = SkyRLBirdSQLUser

    @vf.stop
    async def user_finished(self, trace: vf.Trace) -> bool:
        return trace.state.user_finished

    def _correct(self, trace: vf.Trace) -> bool:
        state = trace.state
        if not state.prediction_executed or not state.gold_executed:
            return False
        from .sqlite import QueryResult

        return results_match(
            QueryResult(state.prediction_columns, state.prediction_rows),
            QueryResult(state.gold_columns, state.gold_rows),
            self.data.grading_method,
        )

    @vf.reward(weight=1.0)
    async def reward(self, trace: vf.Trace) -> float:
        if not trace.state.format_valid or trace.state.final_query is None:
            return -1.0
        correct = self._correct(trace)
        trace.state.execution_correct = correct
        return 1.0 if correct else 0.0

    @vf.metric
    async def outcome_metrics(self, trace: vf.Trace) -> dict[str, float]:
        correct = self._correct(trace)
        trace.state.execution_correct = correct
        return {
            "format_valid": float(trace.state.format_valid),
            "prediction_executed": float(trace.state.prediction_executed),
            "gold_executed": float(trace.state.gold_executed),
            "execution_correct": float(correct),
            "turn_count": float(trace.state.turn_count),
            "exploratory_query_count": float(trace.state.exploratory_query_count),
            "observation_truncated": float(trace.state.observation_truncated),
        }

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        del runtime
        correct = self._correct(trace)
        trace.state.execution_correct = correct
        trace.info["skyrl_bird_sql"] = {
            "question_id": self.data.question_id,
            "db_id": self.data.db_id,
            "grading_method": self.data.grading_method,
            "turn_count": trace.state.turn_count,
            "exploratory_query_count": trace.state.exploratory_query_count,
            "format_valid": trace.state.format_valid,
            "prediction_executed": trace.state.prediction_executed,
            "gold_executed": trace.state.gold_executed,
            "execution_correct": correct,
            "terminal_reason": trace.state.terminal_reason,
            "error_category": trace.state.error_category,
            "observation_truncated": trace.state.observation_truncated,
        }

    async def validate(self, runtime: vf.Runtime) -> bool:
        del runtime
        parse_grading_method(self.data.grading_method)
        database = database_path(self.data.db_id)
        await execute_readonly_async(database, self.data.gold_sql)
        return True


class SkyRLBirdSQLConfig(vf.TasksetConfig):
    split: Literal["train", "validation"] = "train"
    selection: Literal["all", "canary"] = "all"
    task: SkyRLBirdSQLTaskConfig = SkyRLBirdSQLTaskConfig()


def _canary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for family in ("set", "list", "multiset", "subset"):
        candidates = [
            row
            for row in rows
            if parse_grading_method(str(_field(row, "grading_method"))).family == family
        ]
        if not candidates:
            raise RuntimeError(f"ReViSQL train split contains no {family} grading task")
        selected.append(min(candidates, key=lambda row: str(_field(row, "question_id"))))
    return selected


def _prompt(question: str, evidence: str, schema: str) -> str:
    evidence_block = evidence.strip() or "No additional evidence was provided."
    return f"Question:\n{question.strip()}\n\nEvidence:\n{evidence_block}\n\nDatabase schema:\n{schema}"


class SkyRLBirdSQLTaskset(vf.Taskset[SkyRLBirdSQLTask, SkyRLBirdSQLConfig]):
    def load(self) -> list[SkyRLBirdSQLTask]:
        rows = load_rows(self.config.split)
        if self.config.selection == "canary":
            if self.config.split != "train":
                raise ValueError("the deterministic canary is defined only for the train split")
            rows = _canary_rows(rows)
        rows = sorted(rows, key=lambda row: str(_field(row, "question_id")))
        tasks: list[SkyRLBirdSQLTask] = []
        for index, row in enumerate(rows):
            question_id = str(_field(row, "question_id"))
            db_id = str(_field(row, "db_id"))
            question = str(_field(row, "question"))
            evidence = str(row.get("evidence") or "")
            gold_sql = str(_field(row, "SQL", "sql"))
            grading_method = str(_field(row, "grading_method")).strip()
            tasks.append(
                SkyRLBirdSQLTask(
                    SkyRLBirdSQLData(
                        idx=index,
                        name=f"revisql/{self.config.split}/{question_id}",
                        prompt=_prompt(question, evidence, render_schema(database_path(db_id))),
                        system_prompt=SYSTEM_PROMPT,
                        question_id=question_id,
                        db_id=db_id,
                        question=question,
                        evidence=evidence,
                        gold_sql=gold_sql,
                        grading_method=grading_method,
                    ),
                    self.config.task,
                )
            )
        return tasks


__all__ = [
    "SkyRLBirdSQLConfig",
    "SkyRLBirdSQLData",
    "SkyRLBirdSQLState",
    "SkyRLBirdSQLTask",
    "SkyRLBirdSQLTaskset",
    "SkyRLBirdSQLUser",
]


if __name__ == "__main__":
    SkyRLBirdSQLUser.run()
