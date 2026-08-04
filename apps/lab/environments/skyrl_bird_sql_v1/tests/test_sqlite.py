from __future__ import annotations

from pathlib import Path

import pytest
from skyrl_bird_sql_v1.assets import database_path
from skyrl_bird_sql_v1.sqlite import (
    SQLExecutionError,
    execute_readonly,
    format_observation,
    render_schema,
)


def test_readonly_execution_and_deterministic_schema(bird_cache: Path) -> None:
    database = database_path("demo", bird_cache)
    result = execute_readonly(database, "SELECT name, city FROM people ORDER BY id")
    schema = render_schema(database)

    assert result.columns == ("name", "city")
    assert result.rows[0] == ("Ada", "London")
    assert "TABLE people" in schema
    assert "id: INTEGER [PK1]" in schema
    assert "FK manager_id -> people.id" in schema
    assert schema == render_schema(database)


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM people",
        "UPDATE people SET city = 'x'",
        "CREATE TABLE unsafe(value TEXT)",
        "ATTACH DATABASE '/tmp/unsafe.sqlite' AS unsafe",
        "PRAGMA user_version",
        "SELECT 1; SELECT 2",
    ],
)
def test_executor_rejects_mutation_and_multiple_statements(bird_cache: Path, query: str) -> None:
    with pytest.raises(SQLExecutionError):
        execute_readonly(database_path("demo", bird_cache), query)


def test_observation_is_bounded(bird_cache: Path) -> None:
    result = execute_readonly(database_path("demo", bird_cache), "SELECT name, city FROM people ORDER BY id")
    observation, truncated = format_observation(result, maximum_rows=1, maximum_cell_characters=3)

    assert "Ada | Lon…" in observation
    assert "2 additional row(s) omitted" in observation
    assert truncated is True


def test_result_ceiling_is_enforced(bird_cache: Path) -> None:
    with pytest.raises(SQLExecutionError, match="more than 1 rows"):
        execute_readonly(database_path("demo", bird_cache), "SELECT * FROM people", maximum_rows=1)


def test_query_timeout_is_enforced(bird_cache: Path) -> None:
    expensive = (
        "WITH RECURSIVE counter(value) AS ("
        "VALUES(0) UNION ALL SELECT value + 1 FROM counter WHERE value < 100000000"
        ") SELECT sum(value) FROM counter"
    )

    with pytest.raises(SQLExecutionError, match="interrupted"):
        execute_readonly(database_path("demo", bird_cache), expensive, timeout_seconds=0.001)
