"""Read-only raw SQL access to a local Trackio project."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from trackio.sqlite_storage import SQLiteStorage


def _decode_cell(value: Any) -> Any:
    """Decode Trackio's JSON blobs while leaving normal SQLite values intact."""

    if not isinstance(value, bytes):
        return value
    text = value.decode("utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _read_only_connection(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Trackio database does not exist: {resolved}")
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def query_database(
    database: Path,
    sql: str,
    parameters: Sequence[Any] = (),
) -> list[dict[str, Any]]:
    """Execute one read-only statement and return JSON-compatible rows."""

    with _read_only_connection(database) as connection:
        cursor = connection.execute(sql, tuple(parameters))
        return [{key: _decode_cell(value) for key, value in dict(row).items()} for row in cursor.fetchall()]


def query_project(
    project: str,
    sql: str,
    parameters: Sequence[Any] = (),
) -> list[dict[str, Any]]:
    database = SQLiteStorage.get_project_db_path(project)
    return query_database(database, sql, parameters)


def list_runs(project: str, *, run_kind: str | None = None) -> list[dict[str, Any]]:
    """Return a stable run summary over Trackio's current physical schema."""

    configs = query_project(
        project,
        "SELECT run_id, run_name, created_at, config FROM configs ORDER BY id DESC",
    )
    metric_rows = query_project(
        project,
        "SELECT run_id, metrics FROM metrics ORDER BY id",
    )
    statuses: dict[str, str] = {}
    for row in metric_rows:
        metrics = row["metrics"]
        if isinstance(metrics, dict) and "run/status" in metrics:
            statuses[row["run_id"]] = str(metrics["run/status"])

    runs: list[dict[str, Any]] = []
    for row in configs:
        config = row["config"] if isinstance(row["config"], dict) else {}
        current_kind = config.get("run_kind")
        if run_kind is not None and current_kind != run_kind:
            continue
        runs.append(
            {
                "run_id": row["run_id"],
                "run_name": row["run_name"],
                "created_at": row["created_at"],
                "run_kind": current_kind,
                "status": statuses.get(row["run_id"], "unknown"),
                "schema_version": config.get("schema_version"),
                "model_profile_id": config.get("model_profile_id"),
                "model_artifact": config.get("model_artifact"),
            }
        )
    return runs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a read-only SQL query against a local Trackio project.")
    parser.add_argument("project", help="Trackio project name")
    parser.add_argument("sql", help="SQL SELECT/PRAGMA statement")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(query_project(args.project, args.sql), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
