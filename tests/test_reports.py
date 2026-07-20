from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reports.query import list_runs, query_database


class RawQueryTest(unittest.TestCase):
    def test_queries_sqlite_in_read_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "trackio.db"
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE metrics (run TEXT, value REAL)")
                connection.execute("INSERT INTO metrics VALUES ('run-one', 12.5)")

            rows = query_database(database, "SELECT run, value FROM metrics")

            self.assertEqual(rows, [{"run": "run-one", "value": 12.5}])
            with self.assertRaises(sqlite3.OperationalError):
                query_database(database, "DELETE FROM metrics")

    def test_decodes_json_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "trackio.db"
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE configs (config BLOB)")
                connection.execute("INSERT INTO configs VALUES (?)", (b'{\"run_kind\":\"general-eval\"}',))

            rows = query_database(database, "SELECT config FROM configs")

            self.assertEqual(rows, [{"config": {"run_kind": "general-eval"}}])

    def test_lists_stable_run_summaries(self) -> None:
        configs = [
            {
                "run_id": "run-1",
                "run_name": "smoke",
                "created_at": "now",
                "config": {
                    "schema_version": 1,
                    "run_kind": "serving-benchmark",
                    "model_profile_id": "model-one",
                    "model_artifact": "hf://org/model@revision",
                },
            }
        ]
        metrics = [{"run_id": "run-1", "metrics": {"run/status": "complete"}}]
        with mock.patch("reports.query.query_project", side_effect=[configs, metrics]):
            rows = list_runs("lab", run_kind="serving-benchmark")

        self.assertEqual(rows[0]["status"], "complete")
        self.assertEqual(rows[0]["model_profile_id"], "model-one")


if __name__ == "__main__":
    unittest.main()
