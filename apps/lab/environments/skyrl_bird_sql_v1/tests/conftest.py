from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def bird_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "bird-cache"
    database_dir = root / "databases" / "demo"
    database_dir.mkdir(parents=True)
    database = database_dir / "demo.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE people (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                city TEXT,
                manager_id INTEGER REFERENCES people(id)
            );
            INSERT INTO people(id, name, city, manager_id) VALUES
                (1, 'Ada', 'London', NULL),
                (2, 'Grace', 'New York', 1),
                (3, 'Linus', 'Helsinki', 1);
            """
        )
    rows = [
        {
            "question_id": "set-1",
            "db_id": "demo",
            "question": "Which names are present?",
            "evidence": "Return names only.",
            "SQL": "SELECT name FROM people",
            "grading_method": "set",
        },
        {
            "question_id": "list-1",
            "db_id": "demo",
            "question": "List names by id.",
            "evidence": "Order by id.",
            "SQL": "SELECT name FROM people ORDER BY id",
            "grading_method": "list",
        },
        {
            "question_id": "multiset-1",
            "db_id": "demo",
            "question": "List manager ids.",
            "evidence": "Duplicates matter.",
            "SQL": "SELECT manager_id FROM people WHERE manager_id IS NOT NULL",
            "grading_method": " multiset",
        },
        {
            "question_id": "subset-1",
            "db_id": "demo",
            "question": "Return one city.",
            "evidence": "Any one is accepted.",
            "SQL": "SELECT city FROM people",
            "grading_method": "subset,=,1",
        },
    ]
    revisql = root / "revisql"
    revisql.mkdir()
    (revisql / "bird-verified-train.json").write_text(json.dumps(rows), encoding="utf-8")
    (revisql / "bird-verified-val.json").write_text(json.dumps(rows[:1]), encoding="utf-8")
    monkeypatch.setenv("POSTTRAIN_SKYRL_BIRD_CACHE", str(root))
    return root
