"""Strict catalog schema and decoder for portable environment bindings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal, cast

from posttrain.common import CatalogRef, ContractError, JsonValue
from posttrain.common.catalog import SelectionDecoder
from posttrain.common.selections import Selection, SelectionFamily
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .requests import (
    EnvironmentActivation,
    EnvironmentBinding,
    EnvironmentFactory,
    EnvironmentPackageSource,
    EnvironmentSource,
    ProjectPathActivationResource,
    ProjectPathEnvironmentSource,
    PythonFactoryActivation,
    SamplingPolicy,
    VerifiersV1ConfigActivation,
)


class EnvironmentCatalogSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EnvironmentSourceSchema(EnvironmentCatalogSchema):
    kind: Literal["git"] = "git"
    package: str
    repository: str
    revision: str
    subdirectory: str | None = None


class ProjectPathEnvironmentSourceSchema(EnvironmentCatalogSchema):
    kind: Literal["project-path"]
    package: str
    path: str


EnvironmentPackageSourceSchema = EnvironmentSourceSchema | ProjectPathEnvironmentSourceSchema


class SamplingPolicySchema(EnvironmentCatalogSchema):
    max_tokens: int = Field(gt=0)
    temperature: float = Field(default=0.0, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    reasoning_effort: str | None = None


class ProjectPathActivationResourceSchema(EnvironmentCatalogSchema):
    kind: Literal["project-path"]
    path: str


ActivationResourceSourceSchema = Annotated[
    ProjectPathActivationResourceSchema,
    Field(discriminator="kind"),
]


class ActivationResourceSchema(EnvironmentCatalogSchema):
    source: ActivationResourceSourceSchema


class VerifiersV1ConfigActivationSchema(EnvironmentCatalogSchema):
    kind: Literal["verifiers-config"]
    config: dict[str, JsonValue]
    resources: dict[str, ActivationResourceSchema] = Field(default_factory=dict)


class PythonFactoryActivationSchema(EnvironmentCatalogSchema):
    kind: Literal["python-factory"]
    reference: str


EnvironmentActivationSchema = Annotated[
    VerifiersV1ConfigActivationSchema | PythonFactoryActivationSchema,
    Field(discriminator="kind"),
]


class EnvironmentBindingSchema(EnvironmentCatalogSchema):
    id: str
    category: str
    source: EnvironmentPackageSourceSchema
    activation: EnvironmentActivationSchema | None = None
    factory: str | None = None
    sampling: SamplingPolicySchema
    num_tasks: int = Field(gt=0)
    num_rollouts: int = Field(default=1, gt=0)
    max_concurrent: int = Field(default=4, gt=0)
    qualification: Literal["required", "deferred"] = "required"
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    reward_components: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_one_activation(self) -> EnvironmentBindingSchema:
        if (self.activation is None) == (self.factory is None):
            raise ValueError("environment binding requires exactly one of activation or legacy factory")
        return self


def environment_catalog_decoders(
    factories: Mapping[str, EnvironmentActivation | EnvironmentFactory] | None = None,
) -> Mapping[SelectionFamily, SelectionDecoder]:
    """Build detached environment decoders without importing implementations."""

    aliases = {name: _normalize_activation(value) for name, value in (factories or {}).items()}

    def decode_environment(
        ref: CatalogRef,
        data: Mapping[str, object],
        known: Mapping[CatalogRef, Selection],
    ) -> Selection:
        del ref, known
        payload = EnvironmentBindingSchema.model_validate(data)
        activation = (
            _activation_from_schema(payload.activation)
            if payload.activation is not None
            else _legacy_activation(payload.factory, aliases)
        )
        return EnvironmentBinding(
            id=payload.id,
            category=payload.category,
            source=_source_from_schema(payload.source),
            activation=activation,
            sampling=SamplingPolicy(**payload.sampling.model_dump()),
            num_tasks=payload.num_tasks,
            num_rollouts=payload.num_rollouts,
            max_concurrent=payload.max_concurrent,
            qualification=payload.qualification,
            parameters=payload.parameters,
            reward_components=payload.reward_components,
        )

    return cast(Mapping[SelectionFamily, SelectionDecoder], {"environment": decode_environment})


def _activation_from_schema(
    payload: VerifiersV1ConfigActivationSchema | PythonFactoryActivationSchema,
) -> EnvironmentActivation:
    if isinstance(payload, VerifiersV1ConfigActivationSchema):
        return VerifiersV1ConfigActivation(
            payload.config,
            {name: ProjectPathActivationResource(resource.source.path) for name, resource in payload.resources.items()},
        )
    return PythonFactoryActivation(payload.reference)


def _source_from_schema(payload: EnvironmentPackageSourceSchema) -> EnvironmentPackageSource:
    if isinstance(payload, ProjectPathEnvironmentSourceSchema):
        return ProjectPathEnvironmentSource(package=payload.package, path=payload.path)
    return EnvironmentSource(
        package=payload.package,
        repository=payload.repository,
        revision=payload.revision,
        subdirectory=payload.subdirectory,
    )


def _legacy_activation(
    name: str | None,
    aliases: Mapping[str, EnvironmentActivation],
) -> EnvironmentActivation:
    if name is None:
        raise AssertionError("validated environment binding has no activation")
    if ":" in name:
        return PythonFactoryActivation(name)
    try:
        return aliases[name]
    except KeyError as error:
        raise ContractError(f"legacy environment factory alias is not registered: {name}") from error


def _normalize_activation(value: EnvironmentActivation | EnvironmentFactory) -> EnvironmentActivation:
    if isinstance(value, (VerifiersV1ConfigActivation, PythonFactoryActivation)):
        return value
    return PythonFactoryActivation.from_callable(value)


__all__ = [
    "ActivationResourceSchema",
    "EnvironmentActivationSchema",
    "EnvironmentBindingSchema",
    "EnvironmentPackageSourceSchema",
    "EnvironmentSourceSchema",
    "PythonFactoryActivationSchema",
    "ProjectPathEnvironmentSourceSchema",
    "ProjectPathActivationResourceSchema",
    "SamplingPolicySchema",
    "VerifiersV1ConfigActivationSchema",
    "environment_catalog_decoders",
]
