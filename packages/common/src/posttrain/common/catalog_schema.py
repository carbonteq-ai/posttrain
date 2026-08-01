"""Pydantic schemas for untrusted catalog file boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .artifacts import JsonValue


class CatalogSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HubArtifactSchema(CatalogSchema):
    kind: Literal["hub"]
    repo_id: str
    revision: str


class LocalArtifactSchema(CatalogSchema):
    kind: Literal["local"]
    path: Path
    digest: str


class TrackioArtifactSchema(CatalogSchema):
    kind: Literal["trackio"]
    project: str
    name: str
    version: str
    alias: str | None = None


type ArtifactSchema = Annotated[
    HubArtifactSchema | LocalArtifactSchema | TrackioArtifactSchema,
    Field(discriminator="kind"),
]


class ModelCapabilitiesSchema(CatalogSchema):
    modalities: tuple[str, ...]
    native_context_window: int = Field(gt=0)
    mtp: bool = False


class ModelVariantSchema(CatalogSchema):
    id: str
    artifact: ArtifactSchema
    base: HubArtifactSchema | None = None
    form: Literal["foundation", "adapter", "peft-adapter", "merged", "full-finetuned", "weight-quantized"]
    weight_precision: str = "bf16"
    family: str
    parameters: int = Field(gt=0)
    instruction_tuned: bool
    renderer_contract: str
    capabilities: ModelCapabilitiesSchema
    revision: str | None = None
    digest: str | None = None
    tokenizer_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    quantization: dict[str, JsonValue] = Field(default_factory=dict)
    parent: str | None = None
    provenance: dict[str, JsonValue] = Field(default_factory=dict)


class ExecutionTargetSchema(CatalogSchema):
    id: str
    revision: str
    device_class: str
    memory_gb: float | None = Field(default=None, gt=0)
    placement: dict[str, JsonValue] = Field(default_factory=dict)
    host_constraints: dict[str, JsonValue] = Field(default_factory=dict)


class WorkloadSchema(CatalogSchema):
    id: str
    revision: str
    requests: dict[str, JsonValue]
    concurrency: tuple[int, ...] = (1,)
    warmup_repetitions: int = Field(default=0, ge=0)
    measured_repetitions: int = Field(default=1, gt=0)
    required_measures: tuple[str, ...] = ()
    plateau_improvement_ratio: float = Field(default=0.05, gt=0, lt=1)
    plateau_intervals: int = Field(default=2, gt=0)
    max_consecutive_point_failures: int = Field(default=1, gt=0)


class CatalogLinkSchema(CatalogSchema):
    id: str


class InferenceBindingSchema(CatalogSchema):
    id: str
    revision: str
    model: str | CatalogLinkSchema
    backend: str
    renderer: str
    engine: dict[str, JsonValue]
    sampling: dict[str, JsonValue]
    target: str | CatalogLinkSchema
    purpose: tuple[Literal["screen", "eval", "rollout", "teacher-score", "smoke", "handoff"], ...]
    startup_timeout_seconds: float = Field(default=180.0, gt=0)


__all__ = [
    "ExecutionTargetSchema",
    "InferenceBindingSchema",
    "ModelVariantSchema",
    "WorkloadSchema",
]
