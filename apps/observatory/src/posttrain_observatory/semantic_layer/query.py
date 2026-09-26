"""Semantic and SQL queries, their results, and filter conditions."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field, field_validator

from ..models import ObservatoryModel
from .model import AGGREGATIONS

type JsonScalar = str | int | float | bool | None
type RunScope = tuple[str, ...] | dict[str, Any]

DEFAULT_MAX_RUNS = 50
_COMPARISON = re.compile(r"^\s*(>=|<=|!=|>|<|=)\s*(.+?)\s*$")


class SemanticQuery(ObservatoryModel):
    """Measures (or metrics), grouped by dimensions, filtered by conditions.

    ``measures`` entries are names, optionally with an aggregation
    (``entropy:last``). ``runs`` is shorthand for run filters: explicit run ids,
    or a mapping of run dimensions to conditions.
    """

    measures: tuple[str, ...] = Field(min_length=1)
    by: tuple[str, ...] = ()
    where: dict[str, Any] = Field(default_factory=dict)
    runs: tuple[str, ...] | dict[str, Any] | None = None
    order_by: tuple[str, ...] = ()
    limit: int = Field(default=1000, ge=1, le=100_000)

    @field_validator("measures")
    @classmethod
    def _valid_measures(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            name, _, aggregation = value.partition(":")
            if not name or (aggregation and aggregation not in AGGREGATIONS):
                raise ValueError(f"measure {value!r} must be a name, optionally followed by :<aggregation>")
        return values


class SqlQuery(ObservatoryModel):
    """One read-only SELECT over the semantic tables of the runs in scope."""

    sql: str = Field(min_length=1)
    runs: tuple[str, ...] | dict[str, Any]
    load: dict[str, tuple[str, ...]] | None = None
    max_rows: int = Field(default=10_000, ge=1, le=100_000)


class ResultColumn(ObservatoryModel):
    name: str = Field(min_length=1)
    kind: Literal["dimension", "measure", "metric", "value"]
    type: str | None = None
    unit: str | None = None
    label: str | None = None


class SemanticResult(ObservatoryModel):
    columns: tuple[ResultColumn, ...]
    rows: tuple[tuple[Any, ...], ...]
    grain: str
    runs: tuple[str, ...] = ()
    sources: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    downsampled: bool = False
    unavailable: tuple[str, ...] = ()
    truncated: bool = False


class QueryError(ValueError):
    """A query that cannot be answered, with a reason a person can act on."""


@dataclass(frozen=True, slots=True)
class Condition:
    """One filter on one dimension."""

    operator: Literal["=", "!=", ">", ">=", "<", "<=", "in", "glob"]
    value: Any

    def matches(self, actual: Any) -> bool:
        if self.operator == "in":
            return any(Condition("=", item).matches(actual) for item in self.value)
        if actual is None:
            return self.operator == "!=" and self.value is not None
        if self.operator == "glob":
            return fnmatch.fnmatchcase(str(actual), str(self.value))
        expected = _coerce_like(self.value, actual)
        if self.operator == "=":
            return actual == expected
        if self.operator == "!=":
            return actual != expected
        try:
            if self.operator == ">":
                return actual > expected
            if self.operator == ">=":
                return actual >= expected
            if self.operator == "<":
                return actual < expected
            return actual <= expected
        except TypeError:
            return False

    def pushdown_values(self) -> tuple[str, ...] | None:
        """Exact string values a backend can filter on, or None."""
        if self.operator == "=" and isinstance(self.value, str):
            return (self.value,)
        if self.operator == "in" and all(isinstance(item, str) and "*" not in item for item in self.value):
            return tuple(self.value)
        return None


def parse_condition(value: Any) -> Condition:
    if isinstance(value, (list, tuple)):
        if not value:
            raise QueryError("an 'any of' filter needs at least one value")
        if any(isinstance(item, str) and "*" in item for item in value):
            raise QueryError("wildcards are not allowed inside an 'any of' list")
        return Condition("in", tuple(value))
    if isinstance(value, str):
        # The pattern only matches a value that starts with an operator.
        if (match := _COMPARISON.match(value)) is not None:
            return Condition(match.group(1), _literal(match.group(2)))  # type: ignore[arg-type]
        if "*" in value:
            return Condition("glob", value)
        return Condition("=", value)
    return Condition("=", value)


def _literal(text: str) -> JsonScalar:
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text.strip("'\"")


def _coerce_like(value: Any, actual: Any) -> Any:
    if isinstance(actual, bool) and isinstance(value, str):
        return value.lower() == "true"
    if isinstance(actual, (int, float)) and not isinstance(actual, bool) and isinstance(value, str):
        literal = _literal(value)
        return literal if isinstance(literal, (int, float)) else value
    if isinstance(actual, str) and not isinstance(value, str):
        return str(value)
    return value


def split_measure(value: str) -> tuple[str, str | None]:
    name, _, aggregation = value.partition(":")
    return name, aggregation or None


__all__ = [
    "Condition",
    "DEFAULT_MAX_RUNS",
    "QueryError",
    "ResultColumn",
    "RunScope",
    "SemanticQuery",
    "SemanticResult",
    "SqlQuery",
    "parse_condition",
    "split_measure",
]
