"""Read-only, bounded SQLite execution and deterministic schema rendering."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

_DENIED_ACTIONS = frozenset(
    action
    for action in (
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_INDEX,
        sqlite3.SQLITE_CREATE_TEMP_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
        sqlite3.SQLITE_CREATE_TEMP_VIEW,
        sqlite3.SQLITE_CREATE_TRIGGER,
        sqlite3.SQLITE_CREATE_VIEW,
        sqlite3.SQLITE_DROP_INDEX,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_INDEX,
        sqlite3.SQLITE_DROP_TEMP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_TRIGGER,
        sqlite3.SQLITE_DROP_TEMP_VIEW,
        sqlite3.SQLITE_DROP_TRIGGER,
        sqlite3.SQLITE_DROP_VIEW,
        sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_REINDEX,
        sqlite3.SQLITE_ANALYZE,
        sqlite3.SQLITE_ATTACH,
        sqlite3.SQLITE_DETACH,
        sqlite3.SQLITE_PRAGMA,
    )
    if isinstance(action, int)
)


class SQLExecutionError(RuntimeError):
    """A query was unsafe, timed out, failed, or exceeded the result ceiling."""


@dataclass(frozen=True, slots=True)
class QueryResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]


def _uri(path: Path) -> str:
    return f"file:{quote(str(path.resolve()))}?mode=ro&immutable=1"


def _authorizer(action: int, _arg1: str | None, _arg2: str | None, _db: str | None, _trigger: str | None) -> int:
    return sqlite3.SQLITE_DENY if action in _DENIED_ACTIONS else sqlite3.SQLITE_OK


def execute_readonly(
    database: Path,
    query: str,
    *,
    timeout_seconds: float = 10.0,
    maximum_rows: int = 100_000,
) -> QueryResult:
    """Execute exactly one read-only statement on a fresh immutable connection."""

    if timeout_seconds <= 0 or maximum_rows < 1:
        raise ValueError("query timeout and result ceiling must be positive")
    started = time.monotonic()
    try:
        with sqlite3.connect(_uri(database), uri=True, timeout=timeout_seconds) as connection:
            connection.execute("PRAGMA query_only = ON")
            connection.set_authorizer(_authorizer)
            connection.set_progress_handler(
                lambda: 1 if time.monotonic() - started > timeout_seconds else 0,
                1_000,
            )
            cursor = connection.cursor()
            cursor.execute(query)
            if cursor.description is None:
                raise SQLExecutionError("the statement did not return a result set")
            rows = cursor.fetchmany(maximum_rows + 1)
            if len(rows) > maximum_rows:
                raise SQLExecutionError(f"query returned more than {maximum_rows} rows")
            return QueryResult(
                tuple(str(item[0]) for item in cursor.description),
                tuple(tuple(cell for cell in row) for row in rows),
            )
    except SQLExecutionError:
        raise
    except sqlite3.Error as error:
        raise SQLExecutionError(str(error)) from error


async def execute_readonly_async(
    database: Path,
    query: str,
    *,
    timeout_seconds: float = 10.0,
    maximum_rows: int = 100_000,
) -> QueryResult:
    result: QueryResult | None = None
    error: BaseException | None = None
    completed = threading.Event()

    def run() -> None:
        nonlocal result, error
        try:
            result = execute_readonly(
                database,
                query,
                timeout_seconds=timeout_seconds,
                maximum_rows=maximum_rows,
            )
        except BaseException as caught:
            error = caught
        finally:
            completed.set()

    # The pinned Verifiers runtime imports networking libraries that can prevent
    # asyncio's selector from waking when a default-executor callback is its only
    # event. Polling a private bounded worker avoids that integration deadlock
    # while keeping SQLite work off the rollout event loop.
    worker = threading.Thread(target=run, name="skyrl-bird-sqlite", daemon=True)
    worker.start()
    while not completed.is_set():
        await asyncio.sleep(0.001)
    worker.join()
    if error is not None:
        raise error
    if result is None:  # pragma: no cover - the worker always sets a result or error
        raise RuntimeError("SQLite worker completed without a result")
    return result


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def render_schema(database: Path) -> str:
    """Render sorted tables, columns, primary keys, and foreign keys."""

    sections: list[str] = []
    with sqlite3.connect(_uri(database), uri=True) as connection:
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name COLLATE NOCASE, name"
            )
        ]
        for table in tables:
            quoted = _quote_identifier(table)
            columns = list(connection.execute(f"PRAGMA table_info({quoted})"))
            foreign_keys = list(connection.execute(f"PRAGMA foreign_key_list({quoted})"))
            rendered_columns = []
            for _cid, name, kind, not_null, default, primary_key in columns:
                flags = []
                if primary_key:
                    flags.append(f"PK{primary_key}")
                if not_null:
                    flags.append("NOT NULL")
                if default is not None:
                    flags.append(f"DEFAULT {default}")
                suffix = f" [{' '.join(flags)}]" if flags else ""
                rendered_columns.append(f"  - {name}: {kind or 'ANY'}{suffix}")
            rendered_foreign_keys = [
                f"  - FK {source} -> {target_table}.{target_column}"
                for _id, _seq, target_table, source, target_column, *_rest in sorted(
                    foreign_keys,
                    key=lambda row: (str(row[3]).casefold(), str(row[2]).casefold(), str(row[4]).casefold()),
                )
            ]
            sections.append("\n".join([f"TABLE {table}", *rendered_columns, *rendered_foreign_keys]))
    return "\n\n".join(sections)


def format_observation(
    result: QueryResult,
    *,
    maximum_rows: int = 50,
    maximum_cell_characters: int = 200,
) -> tuple[str, bool]:
    """Produce a bounded deterministic plain-text observation."""

    def cell(value: object) -> str:
        text = "NULL" if value is None else str(value)
        return text if len(text) <= maximum_cell_characters else text[:maximum_cell_characters] + "…"

    rows = result.rows[:maximum_rows]
    lines = [" | ".join(result.columns)]
    lines.extend(" | ".join(cell(value) for value in row) for row in rows)
    truncated = len(result.rows) > maximum_rows or any(
        len("NULL" if value is None else str(value)) > maximum_cell_characters
        for row in rows
        for value in row
    )
    if len(result.rows) > maximum_rows:
        lines.append(f"… {len(result.rows) - maximum_rows} additional row(s) omitted")
    return "\n".join(lines), truncated


def result_from_rows(rows: Sequence[Sequence[object]], columns: Sequence[str] = ()) -> QueryResult:
    """Small fixture helper kept public for scorer tests."""

    return QueryResult(tuple(columns), tuple(tuple(row) for row in rows))


__all__ = [
    "QueryResult",
    "SQLExecutionError",
    "execute_readonly",
    "execute_readonly_async",
    "format_observation",
    "render_schema",
    "result_from_rows",
]
