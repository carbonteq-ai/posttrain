"""Pinned ReViSQL result-comparison semantics."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from .sqlite import QueryResult

type GradingFamily = Literal["set", "multiset", "list", "subset"]


@dataclass(frozen=True, slots=True)
class GradingMethod:
    family: GradingFamily
    comparison: Literal["=", ">="] | None = None
    count: int | None = None


def parse_grading_method(value: str) -> GradingMethod:
    parts = [part.strip().casefold() for part in value.split(",")]
    if parts[0] in {"set", "multiset", "list"} and len(parts) == 1:
        return GradingMethod(parts[0])  # type: ignore[arg-type]
    if len(parts) == 3 and parts[0] == "subset" and parts[1] in {"=", ">="}:
        try:
            count = int(parts[2])
        except ValueError as error:
            raise ValueError(f"invalid subset count in grading method {value!r}") from error
        if count < 0:
            raise ValueError("subset count cannot be negative")
        return GradingMethod("subset", parts[1], count)  # type: ignore[arg-type]
    raise ValueError(f"unsupported grading method {value!r}")


def _blank(value: object) -> bool:
    return value is None or not str(value).strip()


def _clean(result: QueryResult) -> tuple[tuple[object, ...], ...]:
    return tuple(row for row in result.rows if not all(_blank(value) for value in row))


def _numeric_scalar(rows: tuple[tuple[object, ...], ...]) -> float | None:
    if len(rows) != 1 or len(rows[0]) != 1:
        return None
    scalar = rows[0][0]
    if not isinstance(scalar, (str, bytes, int, float)):
        return None
    try:
        value = float(scalar)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _canonical(rows: tuple[tuple[object, ...], ...]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(sorted("" if value is None else str(value) for value in row)) for row in rows)


def results_match(predicted: QueryResult, expected: QueryResult, method: str | GradingMethod) -> bool:
    grading = parse_grading_method(method) if isinstance(method, str) else method
    prediction = _clean(predicted)
    gold = _clean(expected)
    predicted_width = len(prediction[0]) if prediction else len(predicted.columns)
    gold_width = len(gold[0]) if gold else len(expected.columns)
    if predicted_width != gold_width:
        return False

    predicted_scalar = _numeric_scalar(prediction)
    gold_scalar = _numeric_scalar(gold)
    if predicted_scalar is not None and gold_scalar is not None:
        tolerance = 0.01 * max(abs(gold_scalar), 1e-12)
        return abs(predicted_scalar - gold_scalar) <= tolerance

    predicted_rows = _canonical(prediction)
    gold_rows = _canonical(gold)
    if grading.family == "set":
        return set(predicted_rows) == set(gold_rows)
    if grading.family == "multiset":
        return Counter(predicted_rows) == Counter(gold_rows)
    if grading.family == "list":
        return predicted_rows == gold_rows

    assert grading.count is not None and grading.comparison is not None
    row_count_ok = (
        len(predicted_rows) == grading.count
        if grading.comparison == "="
        else len(predicted_rows) >= grading.count
    )
    predicted_counts = Counter(predicted_rows)
    gold_counts = Counter(gold_rows)
    return row_count_ok and all(count <= gold_counts[row] for row, count in predicted_counts.items())


__all__ = ["GradingMethod", "parse_grading_method", "results_match"]
