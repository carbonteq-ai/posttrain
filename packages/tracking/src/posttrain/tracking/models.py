"""Strict normalized read and job-view models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from posttrain.common import JsonValue
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import RunStatus, Stage


class TrackingModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TrackingCapabilities(TrackingModel):
    provider: str = Field(min_length=1)
    live_metrics: bool = True
    live_traces: bool = False
    artifacts: bool = True
    artifact_lineage: bool = True


class RunQuery(TrackingModel):
    project_id: str | None = None
    work_package_id: str | None = None
    job_kinds: tuple[str, ...] = ()
    statuses: tuple[RunStatus, ...] = ()
    limit: int = Field(default=100, ge=1, le=1000)


class SafeRunError(TrackingModel):
    type: str = Field(min_length=1)
    message: str = Field(min_length=1)


class RunSummary(TrackingModel):
    provider: str = Field(min_length=1)
    provider_run_id: str | None = None
    run_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    work_package_id: str = Field(min_length=1)
    stage: Stage
    job_kind: str = Field(min_length=1)
    job_definition_version: str = Field(min_length=1)
    status: RunStatus
    started_at: datetime
    finished_at: datetime | None = None
    error: SafeRunError | None = None

    @model_validator(mode="after")
    def validate_lifecycle(self) -> RunSummary:
        if self.started_at.tzinfo is None or (self.finished_at is not None and self.finished_at.tzinfo is None):
            raise ValueError("run timestamps must be timezone-aware")
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("run cannot finish before it starts")
        if self.status == "failed" and self.error is None:
            raise ValueError("failed runs require safe error information")
        return self

    @property
    def duration_seconds(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()


class EventRecord(TrackingModel):
    name: str = Field(min_length=1)
    occurred_at: datetime
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class RunDetail(TrackingModel):
    summary: RunSummary
    resolved_inputs: dict[str, JsonValue] = Field(default_factory=dict)
    source_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    metric_names: tuple[str, ...] = ()
    events: tuple[EventRecord, ...] = ()
    trace_count: int = Field(default=0, ge=0)


class MetricPoint(TrackingModel):
    value: float
    step: int | None = Field(default=None, ge=0)
    observed_at: datetime | None = None
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class MetricSeries(TrackingModel):
    name: str = Field(min_length=1)
    points: tuple[MetricPoint, ...] = ()


class TraceQuery(TrackingModel):
    trace_type: str | None = None
    cursor: str | None = None
    limit: int = Field(default=100, ge=1, le=1000)


class TraceRecord(TrackingModel):
    trace_type: str = Field(min_length=1)
    external_id: str = Field(min_length=1)
    payload: dict[str, JsonValue]
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class TracePage(TrackingModel):
    items: tuple[TraceRecord, ...] = ()
    next_cursor: str | None = None
    live: bool = False


class StoredArtifact(TrackingModel):
    provider: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    digest: str | None = None
    provider_metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ArtifactLink(TrackingModel):
    direction: Literal["input", "output"]
    logical_name: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    artifact: StoredArtifact


class ArtifactSet(TrackingModel):
    items: tuple[ArtifactLink, ...] = ()

    @property
    def inputs(self) -> tuple[ArtifactLink, ...]:
        return tuple(item for item in self.items if item.direction == "input")

    @property
    def outputs(self) -> tuple[ArtifactLink, ...]:
        return tuple(item for item in self.items if item.direction == "output")


__all__ = [
    "ArtifactLink",
    "ArtifactSet",
    "EventRecord",
    "MetricPoint",
    "MetricSeries",
    "RunDetail",
    "RunQuery",
    "RunSummary",
    "SafeRunError",
    "StoredArtifact",
    "TracePage",
    "TraceQuery",
    "TraceRecord",
    "TrackingCapabilities",
    "TrackingModel",
]
