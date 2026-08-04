from __future__ import annotations

import os

import pytest
from skyrl_bird_sql_v1.assets import database_path, load_rows, validate
from skyrl_bird_sql_v1.sqlite import execute_readonly


@pytest.mark.network
def test_pinned_assets_and_all_gold_queries_execute() -> None:
    if os.environ.get("POSTTRAIN_SKYRL_BIRD_INTEGRATION") != "1":
        pytest.skip("set POSTTRAIN_SKYRL_BIRD_INTEGRATION=1 after preparing the 20 GB pinned asset cache")

    report = validate()
    assert report["train_rows"] == 2_064
    assert report["validation_rows"] == 398
    assert report["required_database_count"] == 69

    for split in ("train", "validation"):
        for row in load_rows(split):
            execute_readonly(database_path(str(row["db_id"])), str(row.get("SQL", row.get("sql"))))
