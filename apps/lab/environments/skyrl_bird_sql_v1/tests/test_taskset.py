from __future__ import annotations

from pathlib import Path

import verifiers.v1 as vf
from skyrl_bird_sql_v1.taskset import SkyRLBirdSQLConfig, SkyRLBirdSQLState, SkyRLBirdSQLTaskset


def test_canary_selects_one_deterministic_task_per_grading_family(bird_cache: Path) -> None:
    taskset = SkyRLBirdSQLTaskset(SkyRLBirdSQLConfig(split="train", selection="canary"))
    tasks = taskset.load()

    assert [task.data.question_id for task in tasks] == ["list-1", "multiset-1", "set-1", "subset-1"]
    assert {task.data.grading_method for task in tasks} == {"list", "multiset", "set", "subset,=,1"}
    assert all(task.data.name == f"revisql/train/{task.data.question_id}" for task in tasks)
    assert all("Database schema:\nTABLE people" in task.data.prompt_text for task in tasks)
    assert all(task.data.system_prompt and "<solution>" in task.data.system_prompt for task in tasks)


async def test_user_executes_exploration_and_correct_solution(bird_cache: Path) -> None:
    task = SkyRLBirdSQLTaskset(SkyRLBirdSQLConfig(split="train")).load()[0]
    user = task.user_server()
    assert user is not None
    user._inert_state = SkyRLBirdSQLState()
    await user.setup_task(task.data)

    observation = await user.respond("<think>inspect</think><sql>SELECT name FROM people LIMIT 1</sql>")
    assert observation[0].content.startswith("<observation>name\nAda")
    assert user.state.exploratory_query_count == 1

    finished = await user.respond("<think>answer</think><solution>SELECT name FROM people</solution>")
    assert finished == []
    assert user.state.user_finished is True
    assert user.state.prediction_executed is True
    assert user.state.gold_executed is True


async def test_malformed_turn_is_corrected_then_terminally_invalid(bird_cache: Path) -> None:
    task = SkyRLBirdSQLTaskset(SkyRLBirdSQLConfig(split="train")).load()[0]
    user = task.user_server()
    assert user is not None
    user._inert_state = SkyRLBirdSQLState()
    await user.setup_task(task.data)

    response = await user.respond("SELECT name FROM people")
    assert response[0].content.startswith("<observation>Protocol error:")
    assert user.state.format_valid is False


async def test_terminal_reward_is_negative_invalid_zero_wrong_and_positive_correct(bird_cache: Path) -> None:
    task = next(
        task
        for task in SkyRLBirdSQLTaskset(SkyRLBirdSQLConfig(split="train")).load()
        if task.data.grading_method == "set"
    )

    def trace(state: SkyRLBirdSQLState) -> vf.Trace:
        return vf.Trace(
            task=vf.TraceTask(type=type(task).__name__, data=task.data),
            state=state,
        )

    invalid = trace(SkyRLBirdSQLState(format_valid=False, user_finished=True, terminal_reason="malformed"))
    wrong = trace(
        SkyRLBirdSQLState(
            user_finished=True,
            final_query="SELECT name FROM people WHERE id = 1",
            prediction_executed=True,
            gold_executed=True,
            prediction_columns=("name",),
            gold_columns=("name",),
            prediction_rows=(("Ada",),),
            gold_rows=(("Ada",), ("Grace",), ("Linus",)),
        )
    )
    correct = trace(
        SkyRLBirdSQLState(
            user_finished=True,
            final_query="SELECT name FROM people",
            prediction_executed=True,
            gold_executed=True,
            prediction_columns=("name",),
            gold_columns=("name",),
            prediction_rows=(("Linus",), ("Ada",), ("Grace",)),
            gold_rows=(("Ada",), ("Grace",), ("Linus",)),
        )
    )

    assert await task.reward(invalid) == -1.0
    assert await task.reward(wrong) == 0.0
    assert await task.reward(correct) == 1.0
