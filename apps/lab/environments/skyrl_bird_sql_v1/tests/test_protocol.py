from __future__ import annotations

import pytest
from skyrl_bird_sql_v1.protocol import ProtocolError, parse_turn


def test_parse_exploration_and_solution_turns() -> None:
    exploration = parse_turn("<think>inspect the rows</think><sql>SELECT * FROM people</sql>")
    solution = parse_turn("\n<think>done</think>\n<solution>SELECT name FROM people</solution>\n")

    assert exploration.kind == "sql"
    assert exploration.query == "SELECT * FROM people"
    assert solution.kind == "solution"


@pytest.mark.parametrize(
    "message",
    [
        "<sql>SELECT 1</sql>",
        "<think></think><sql>SELECT 1</sql>",
        "<think>x</think><sql></sql>",
        "<think>x</think><sql>SELECT 1</sql><solution>SELECT 1</solution>",
        "<think>x</think><observation>made up</observation><sql>SELECT 1</sql>",
        "```sql\nSELECT 1\n```",
    ],
)
def test_parse_rejects_malformed_or_hallucinated_protocol(message: str) -> None:
    with pytest.raises(ProtocolError):
        parse_turn(message)
