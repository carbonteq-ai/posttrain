"""Read-only SQL inside the tracking backend's storage.

A backend that can run SQL where its data lives offers `ProjectSql`: one
read-only statement, written in Doris SQL, over the project's logical tables
(`metric_rows`, `run_configs`, `traces`, `run_notes`). Trackio implements it on
Doris storage and, translated, on SQLite storage. It is separate from
`RunDataSource` because not every backend can run SQL.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from posttrain.common import JsonValue
from pydantic import Field

from .models import TrackingModel


class ProjectSqlResult(TrackingModel):
    engine: str = Field(min_length=1)
    columns: tuple[str, ...]
    rows: tuple[tuple[JsonValue, ...], ...]
    truncated: bool = False


class ProjectSqlUnavailable(Exception):
    """The source cannot run SQL in its storage (for example W&B, or an older Trackio)."""


class ProjectSqlError(ValueError):
    """The storage refused or failed the statement; the message says why."""


@runtime_checkable
class ProjectSql(Protocol):
    async def project_sql(
        self, sql: str, *, max_rows: int = 10_000, timeout_seconds: float = 10.0
    ) -> ProjectSqlResult: ...


__all__ = ["ProjectSql", "ProjectSqlError", "ProjectSqlResult", "ProjectSqlUnavailable"]
