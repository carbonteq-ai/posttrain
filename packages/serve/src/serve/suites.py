"""Versioned inference workload suites and deterministic matrix expansion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


class SuiteError(ValueError):
    """Raised when an inference benchmark suite is invalid."""


@dataclass(frozen=True, slots=True)
class WorkloadShape:
    id: str
    output_tokens: int
    input_tokens: int | None = None
    input_fraction: float | None = None

    def input_for(self, context_window: int) -> int:
        if self.input_tokens is not None:
            return self.input_tokens
        assert self.input_fraction is not None
        return int(context_window * self.input_fraction)


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    suite_id: str
    shape_id: str
    context_window: int
    concurrency: int
    input_tokens: int
    output_tokens: int
    warmup_iterations: int
    iterations: int
    prompt_source: str = "controlled_tokens"
    reasoning_mode: str = "not_applicable"
    serve_variant: str | None = None

    @property
    def id(self) -> str:
        return f"{self.shape_id}-ctx{self.context_window}-c{self.concurrency}"

    def as_config(self) -> dict[str, Any]:
        return {"case_id": self.id, **asdict(self)}


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    id: str
    description: str
    contexts: tuple[int, ...]
    concurrencies: tuple[int, ...]
    shapes: tuple[WorkloadShape, ...]
    warmup_iterations: int
    iterations: int
    context_variants: dict[int, str]
    schema_version: int = 1

    def cases(self) -> tuple[BenchmarkCase, ...]:
        cases: list[BenchmarkCase] = []
        for context in self.contexts:
            for shape in self.shapes:
                input_tokens = shape.input_for(context)
                if input_tokens + shape.output_tokens > context:
                    continue
                for concurrency in self.concurrencies:
                    cases.append(
                        BenchmarkCase(
                            suite_id=self.id,
                            shape_id=shape.id,
                            context_window=context,
                            concurrency=concurrency,
                            input_tokens=input_tokens,
                            output_tokens=shape.output_tokens,
                            warmup_iterations=self.warmup_iterations,
                            iterations=self.iterations,
                            serve_variant=self.context_variants.get(context),
                        )
                    )
        return tuple(cases)


def _positive_ints(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise SuiteError(f"{field} must be a non-empty list")
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in value):
        raise SuiteError(f"{field} must contain positive integers")
    return tuple(value)


def load_suite(path: Path) -> BenchmarkSuite:
    """Load and validate one checked-in inference benchmark suite."""

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SuiteError(f"benchmark suite does not exist: {path}") from error
    except yaml.YAMLError as error:
        raise SuiteError(f"invalid benchmark suite YAML {path}: {error}") from error
    if not isinstance(data, dict):
        raise SuiteError(f"benchmark suite must be a mapping: {path}")
    if data.get("schema_version") != 1:
        raise SuiteError("benchmark suite schema_version must be 1")
    suite_id = data.get("id")
    if not isinstance(suite_id, str) or not suite_id.strip():
        raise SuiteError("benchmark suite requires a non-empty id")

    raw_shapes = data.get("shapes")
    if not isinstance(raw_shapes, list) or not raw_shapes:
        raise SuiteError("shapes must be a non-empty list")
    shapes: list[WorkloadShape] = []
    seen: set[str] = set()
    for raw in raw_shapes:
        if not isinstance(raw, dict):
            raise SuiteError("each shape must be a mapping")
        shape_id = raw.get("id")
        if not isinstance(shape_id, str) or not shape_id or shape_id in seen:
            raise SuiteError(f"shape id must be non-empty and unique: {shape_id!r}")
        seen.add(shape_id)
        output_tokens = raw.get("output_tokens")
        if not isinstance(output_tokens, int) or isinstance(output_tokens, bool) or output_tokens < 1:
            raise SuiteError(f"shape {shape_id!r} requires positive output_tokens")
        input_tokens = raw.get("input_tokens")
        input_fraction = raw.get("input_fraction")
        if (input_tokens is None) == (input_fraction is None):
            raise SuiteError(
                f"shape {shape_id!r} requires exactly one of input_tokens or input_fraction"
            )
        if input_tokens is not None and (
            not isinstance(input_tokens, int)
            or isinstance(input_tokens, bool)
            or input_tokens < 1
        ):
            raise SuiteError(f"shape {shape_id!r} input_tokens must be positive")
        if input_fraction is not None and (
            not isinstance(input_fraction, (int, float))
            or isinstance(input_fraction, bool)
            or not 0 < float(input_fraction) < 1
        ):
            raise SuiteError(f"shape {shape_id!r} input_fraction must be between 0 and 1")
        shapes.append(
            WorkloadShape(
                id=shape_id,
                input_tokens=input_tokens,
                input_fraction=float(input_fraction) if input_fraction is not None else None,
                output_tokens=output_tokens,
            )
        )

    warmup = data.get("warmup_iterations", 1)
    iterations = data.get("iterations", 3)
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise SuiteError("warmup_iterations must be a non-negative integer")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
        raise SuiteError("iterations must be a positive integer")
    raw_variants = data.get("context_variants", {})
    if not isinstance(raw_variants, dict):
        raise SuiteError("context_variants must be a mapping")
    context_variants: dict[int, str] = {}
    for context, variant in raw_variants.items():
        if not isinstance(context, int) or context not in data["contexts"]:
            raise SuiteError(f"context variant references unknown context: {context!r}")
        if not isinstance(variant, str) or not variant:
            raise SuiteError(f"context variant for {context} must be a non-empty string")
        context_variants[context] = variant

    return BenchmarkSuite(
        id=suite_id,
        description=str(data.get("description", "")),
        contexts=_positive_ints(data.get("contexts"), "contexts"),
        concurrencies=_positive_ints(data.get("concurrencies"), "concurrencies"),
        shapes=tuple(shapes),
        warmup_iterations=warmup,
        iterations=iterations,
        context_variants=context_variants,
    )


__all__ = [
    "BenchmarkCase",
    "BenchmarkSuite",
    "SuiteError",
    "WorkloadShape",
    "load_suite",
]
