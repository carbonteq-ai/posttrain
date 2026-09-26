"""The semantic model: what can be asked about runs, and where each answer comes from.

An entity is a kind of row (a run, one training update, one rollout, one
evaluation task, one serving load level). A dimension is a field of an entity to
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

type Aggregation = Literal["last", "first", "min", "max", "mean", "sum", "count", "stddev"]
type EntityName = Literal["run", "update", "rollout", "eval_task", "load_level"]
type SourceKind = Literal["run_field", "setting", "metric_series", "trace_fact", "eval_task", "load_level", "derived"]
type DimensionType = Literal["string", "integer", "number", "time", "boolean"]

AGGREGATIONS: tuple[Aggregation, ...] = ("last", "first", "min", "max", "mean", "sum", "count", "stddev")
# Rollouts are aggregated by the tracking backend, which returns count, sum and
# sum of squares; order-dependent or extreme aggregations are not available.
ROLLOUT_AGGREGATIONS: tuple[Aggregation, ...] = ("mean", "sum", "count", "stddev")
_NAME = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")


class Source(ObservatoryModel):
    """Where values come from: a field, a settings path, a metric name, a trace fact."""

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
    """A formula over aggregated measures, for example `sum(rollout_seconds) / sum(update_seconds)`."""

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
            if measure.entity == "rollout" and not set(measure.allowed) <= set(ROLLOUT_AGGREGATIONS):
                raise ValueError(f"rollout measure {measure.name!r} allows aggregations trace facts cannot compute")
        measures = {measure.name: measure for measure in self.measures}
        from .formula import parse_formula  # local import: formula depends on this module's types

        for metric in self.metrics:
            for aggregation, measure_name in parse_formula(metric.formula).references():
                measure = measures.get(measure_name)
                if measure is None:
                    raise ValueError(f"metric {metric.name!r} references unknown measure {measure_name!r}")
                if measure.entity != metric.entity:
                    raise ValueError(f"metric {metric.name!r} mixes entities {metric.entity!r} and {measure.entity!r}")
                if aggregation not in measure.allowed:
                    raise ValueError(
                        f"metric {metric.name!r} uses {aggregation}() which {measure_name!r} does not allow"
                    )
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
    "ROLLOUT_AGGREGATIONS",
    "SemanticModel",
    "Source",
    "SourceKind",
]
