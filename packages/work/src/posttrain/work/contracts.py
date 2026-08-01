"""Typed recipe and work-package contracts for first-party composition."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal

import yaml
from posttrain.common import (
    CatalogRef,
    ContractError,
    JsonValue,
    PublishedArtifact,
    RunContext,
)
from posttrain.common.selections import (
    Selection,
    SelectionFamily,
    validate_revision,
    validate_selection_id,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

type Stage = Literal["screen", "train", "qualify"]
type JobKind = Literal[
    "data.prepare",
    "serve.benchmark",
    "serve.smoke",
    "eval.general",
    "eval.domain",
    "train.sft",
    "train.dpo",
    "train.grpo",
    "train.sampo",
    "train.distill",
    "model.transform",
]
type JobStatus = Literal["succeeded", "not_run"]
type SeatBinding = CatalogRef | Selection
type ResolvedSeats = Mapping[str, Selection]
type JobOperation = Callable[[RunContext, ResolvedSeats], object]
type StaticSeatValidator = Callable[[ResolvedSeats], None]

_STAGES = frozenset({"screen", "train", "qualify"})
_JOB_KINDS = frozenset(
    {
        "data.prepare",
        "serve.benchmark",
        "serve.smoke",
        "eval.general",
        "eval.domain",
        "train.sft",
        "train.dpo",
        "train.grpo",
        "train.sampo",
        "train.distill",
        "model.transform",
    }
)
_SELECTION_FAMILIES = frozenset(
    {
        "model",
        "dataset",
        "environment",
        "inference",
        "training",
        "quantization",
        "evaluation",
        "remote-evaluation",
        "workload",
        "target",
        "recipe",
    }
)


@dataclass(frozen=True, slots=True)
class RecipeJob:
    id: str
    kind: JobKind
    definition: str
    optional: bool = False

    def __post_init__(self) -> None:
        validate_selection_id(self.id, "recipe job id")
        validate_selection_id(self.definition, "job definition id")
        if self.kind not in _JOB_KINDS:
            raise ContractError(f"unsupported recipe job kind: {self.kind!r}")


@dataclass(frozen=True, slots=True)
class Recipe:
    id: str
    revision: str
    stage: Stage
    seats: Mapping[str, SelectionFamily]
    jobs: tuple[RecipeJob, ...]
    expected_artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_selection_id(self.id, "recipe id")
        validate_revision(self.revision, "recipe revision")
        if self.stage not in _STAGES:
            raise ContractError(f"unsupported recipe stage: {self.stage!r}")
        if not self.seats or any(not name.strip() for name in self.seats):
            raise ContractError("recipes require named seats")
        if any(family not in _SELECTION_FAMILIES for family in self.seats.values()):
            raise ContractError("recipe seats require known selection families")
        if not self.jobs or len({job.id for job in self.jobs}) != len(self.jobs):
            raise ContractError("recipes require unique jobs")
        if any(not artifact.strip() for artifact in self.expected_artifacts):
            raise ContractError("expected artifact names cannot be empty")
        object.__setattr__(self, "seats", MappingProxyType(dict(self.seats)))


@dataclass(frozen=True, slots=True)
class WorkPackage:
    project_id: str
    work_package_id: str
    stage: Stage
    recipe: CatalogRef | Recipe
    bindings: Mapping[str, SeatBinding]
    enabled_optional_jobs: tuple[str, ...] = ()
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    description: str | None = None

    def __post_init__(self) -> None:
        validate_selection_id(self.project_id, "project id")
        validate_selection_id(self.work_package_id, "work package id")
        if self.stage not in _STAGES:
            raise ContractError(f"unsupported work-package stage: {self.stage!r}")
        if isinstance(self.recipe, CatalogRef) and self.recipe.family != "recipe":
            raise ContractError("work-package recipe refs must use the recipe family")
        if len(set(self.enabled_optional_jobs)) != len(self.enabled_optional_jobs):
            raise ContractError("enabled optional job ids must be unique")
        for job_id in self.enabled_optional_jobs:
            validate_selection_id(job_id, "enabled optional job id")
        if any(not name.strip() for name in self.bindings):
            raise ContractError("work-package binding names cannot be empty")
        if any(
            not isinstance(binding, CatalogRef) and not isinstance(binding, Selection)
            for binding in self.bindings.values()
        ):
            raise ContractError("work-package bindings must be catalog refs or selections")
        forbidden_metadata = {"decision", "outcome", "status", "accept", "reject"}.intersection(self.metadata)
        if forbidden_metadata:
            raise ContractError("work-package metadata cannot contain job outcomes or decisions")
        if self.description is not None:
            if not self.description.strip():
                raise ContractError("work-package description cannot be empty")
            object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(self, "bindings", MappingProxyType(dict(self.bindings)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class JobDefinition:
    id: str
    kind: JobKind
    seats: Mapping[str, type[object]]
    operation: JobOperation
    description: str | None = None
    required_artifact_roles: tuple[str, ...] = ()
    selection_seats: Mapping[str, type[object]] = field(default_factory=dict)
    static_validator: StaticSeatValidator | None = None

    def __post_init__(self) -> None:
        validate_selection_id(self.id, "job definition id")
        if self.kind not in _JOB_KINDS:
            raise ContractError(f"unsupported job-definition kind: {self.kind!r}")
        if not self.seats or any(not name.strip() for name in self.seats):
            raise ContractError("job definitions require typed seats")
        if not callable(self.operation):
            raise ContractError("job definition operation must be callable")
        if any(not role.strip() for role in self.required_artifact_roles):
            raise ContractError("required artifact roles cannot be empty")
        if len(set(self.required_artifact_roles)) != len(self.required_artifact_roles):
            raise ContractError("required artifact roles must be unique")
        if unknown := sorted(set(self.selection_seats) - set(self.seats)):
            raise ContractError("job-definition selection seats are not runtime seats: " + ", ".join(unknown))
        if self.static_validator is not None and not callable(self.static_validator):
            raise ContractError("job-definition static validator must be callable")
        if self.description is not None:
            if not self.description.strip():
                raise ContractError("job-definition description cannot be empty")
        object.__setattr__(self, "seats", MappingProxyType(dict(self.seats)))
        object.__setattr__(
            self,
            "selection_seats",
            MappingProxyType(dict(self.selection_seats)),
        )
        if self.description is not None:
            object.__setattr__(self, "description", self.description.strip())


@dataclass(frozen=True, slots=True)
class WorkPackageJobResult:
    job_id: str
    kind: JobKind
    definition: str
    status: JobStatus
    run_id: str | None = None
    value: object | None = None
    published_artifacts: tuple[PublishedArtifact, ...] = ()


@dataclass(frozen=True, slots=True)
class WorkPackageResult:
    project_id: str
    work_package_id: str
    jobs: tuple[WorkPackageJobResult, ...]


class _Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CatalogRefSchema(_Schema):
    type: Literal["ref"]
    family: SelectionFamily
    id: str


class RecipeJobSchema(_Schema):
    id: str
    kind: JobKind
    definition: str
    optional: bool = False


class InlineRecipeSchema(_Schema):
    type: Literal["inline"]
    id: str
    revision: str
    stage: Stage
    seats: dict[str, SelectionFamily]
    jobs: tuple[RecipeJobSchema, ...]
    expected_artifacts: tuple[str, ...] = ()


type RecipeSelectionSchema = Annotated[
    CatalogRefSchema | InlineRecipeSchema,
    Field(discriminator="type"),
]


class WorkPackageSchema(_Schema):
    project_id: str
    work_package_id: str
    stage: Stage
    recipe: RecipeSelectionSchema
    bindings: dict[str, CatalogRefSchema]
    enabled_optional_jobs: tuple[str, ...] = ()
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    description: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def recipe_stage_matches(self) -> WorkPackageSchema:
        if isinstance(self.recipe, InlineRecipeSchema) and self.recipe.stage != self.stage:
            raise ValueError("inline recipe stage must match the work-package stage")
        return self


def load_work_package(path: Path) -> WorkPackage:
    """Load one strict project-owned YAML work package."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        schema = WorkPackageSchema.model_validate(raw)
    except (yaml.YAMLError, ValidationError) as error:
        raise ContractError(f"invalid work-package config {path}: {error}") from error
    recipe: Recipe | CatalogRef
    if isinstance(schema.recipe, CatalogRefSchema):
        recipe = CatalogRef(schema.recipe.family, schema.recipe.id)
    else:
        recipe = Recipe(
            id=schema.recipe.id,
            revision=schema.recipe.revision,
            stage=schema.recipe.stage,
            seats=schema.recipe.seats,
            jobs=tuple(RecipeJob(**job.model_dump()) for job in schema.recipe.jobs),
            expected_artifacts=schema.recipe.expected_artifacts,
        )
    return WorkPackage(
        project_id=schema.project_id,
        work_package_id=schema.work_package_id,
        stage=schema.stage,
        recipe=recipe,
        bindings={name: CatalogRef(binding.family, binding.id) for name, binding in schema.bindings.items()},
        enabled_optional_jobs=schema.enabled_optional_jobs,
        metadata=schema.metadata,
        description=schema.description,
    )


__all__ = [
    "JobDefinition",
    "JobKind",
    "JobOperation",
    "JobStatus",
    "Recipe",
    "RecipeJob",
    "ResolvedSeats",
    "SeatBinding",
    "Stage",
    "WorkPackage",
    "WorkPackageJobResult",
    "WorkPackageResult",
    "WorkPackageSchema",
    "load_work_package",
]
