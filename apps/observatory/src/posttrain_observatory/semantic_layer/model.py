"""The semantic model: what can be asked about runs, and where each answer comes from.

An entity is a kind of row: a run, one training update, or one rollout. A dimension is a field of an entity to
group or filter by; run dimensions apply to every entity because every row
belongs to a run. A measure is a number of an entity with a default aggregation
and a source. A metric is a named formula over aggregated measures of one
entity.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, model_validator

from ..models import ObservatoryModel

type Aggregation = Literal["last", "first", "min", "max", "mean", "sum", "count", "stddev", "p50", "p90", "p95", "p99"]
type EntityName = Literal["run", "update", "rollout"]
type SourceKind = Literal["run_field", "setting", "event", "metric_series", "trace_fact", "trace_attribute", "derived"]
type DimensionType = Literal["string", "integer", "number", "time", "boolean"]

AGGREGATIONS: tuple[Aggregation, ...] = (
    "last",
    "first",
    "min",
    "max",
    "mean",
    "sum",
    "count",
    "stddev",
    "p50",
    "p90",
    "p95",
    "p99",
)
_NAME = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")


class Source(ObservatoryModel):
    """Where values come from: a field, a settings path, a metric name, a trace fact.

    A ``trace_attribute`` is a key the producer recorded in the trace's metadata
    (its attributes); it serves labels Trackio has no fact column for yet.
    """

    kind: SourceKind
    name: str = Field(min_length=1)
    # `one_minus` turns a fraction into its complement (for example the share of
    # rollouts that did not fall back to sparse rewards).
    transform: Literal["identity", "one_minus"] = "identity"


class Entity(ObservatoryModel):
    name: EntityName
    description: str = Field(min_length=1)
    order_dimension: str | None = None


class Dimension(ObservatoryModel):
    name: str = Field(min_length=1)
    entity: EntityName
    type: DimensionType
    description: str = Field(min_length=1)
    source: Source


class Measure(ObservatoryModel):
    name: str = Field(min_length=1)
    entity: EntityName
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    unit: str | None = None
    source: Source
    aggregation: Aggregation = "mean"
    allowed: tuple[Aggregation, ...] = AGGREGATIONS
    job_kinds: tuple[str, ...] = Field(min_length=1)


class Metric(ObservatoryModel):
    """A SQL expression over aggregated measures of one entity, for example
    `sum(rollout_seconds) / sum(update_seconds)`; the tests plan every formula against its table."""

    name: str = Field(min_length=1)
    entity: EntityName
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    unit: str | None = None
    formula: str = Field(min_length=1)


class SemanticModel(ObservatoryModel):
    entities: tuple[Entity, ...]
    dimensions: tuple[Dimension, ...]
    measures: tuple[Measure, ...]
    metrics: tuple[Metric, ...] = ()

    @model_validator(mode="after")
    def validate_model(self) -> SemanticModel:
        entities = {entity.name for entity in self.entities}
        names: list[str] = [*(d.name for d in self.dimensions), *(m.name for m in self.measures)]
        names += [metric.name for metric in self.metrics]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"semantic names must be unique: {', '.join(duplicates)}")
        for name in names:
            if not _NAME.fullmatch(name):
                raise ValueError(f"semantic name {name!r} must be lower-case words joined by dots or underscores")
        for item in (*self.dimensions, *self.measures, *self.metrics):
            if item.entity not in entities:
                raise ValueError(f"{item.name!r} belongs to undeclared entity {item.entity!r}")
        for dimension in self.dimensions:
            if not dimension.name.startswith(f"{dimension.entity}."):
                raise ValueError(f"dimension {dimension.name!r} must be prefixed with its entity")
        for measure in self.measures:
            if measure.aggregation not in measure.allowed:
                raise ValueError(f"measure {measure.name!r} default aggregation is not allowed")
        return self

    def dimension(self, name: str) -> Dimension:
        for dimension in self.dimensions:
            if dimension.name == name:
                return dimension
        raise KeyError(name)

    def measure(self, name: str) -> Measure:
        for measure in self.measures:
            if measure.name == name:
                return measure
        raise KeyError(name)

    def metric(self, name: str) -> Metric:
        for metric in self.metrics:
            if metric.name == name:
                return metric
        raise KeyError(name)

    def entity(self, name: str) -> Entity:
        for entity in self.entities:
            if entity.name == name:
                return entity
        raise KeyError(name)


__all__ = [
    "AGGREGATIONS",
    "Aggregation",
    "Dimension",
    "DimensionType",
    "Entity",
    "EntityName",
    "Measure",
    "Metric",
    "SemanticModel",
    "Source",
    "SourceKind",
]
