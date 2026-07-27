"""Strict catalog schemas for evaluation plans and environment bindings."""

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
    EnvironmentSource,
    EvaluationPlan,
    PythonFactoryActivation,
    SamplingPolicy,
    VerifiersV1ConfigActivation,
)


class EvalCatalogSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EnvironmentSourceSchema(EvalCatalogSchema):
    package: str
    repository: str
    revision: str
    subdirectory: str | None = None


class SamplingPolicySchema(EvalCatalogSchema):
    max_tokens: int = Field(gt=0)
    temperature: float = Field(default=0.0, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    reasoning_effort: str | None = None


class VerifiersV1ConfigActivationSchema(EvalCatalogSchema):
    kind: Literal["verifiers-config"]
    config: dict[str, JsonValue]


class PythonFactoryActivationSchema(EvalCatalogSchema):
    kind: Literal["python-factory"]
    reference: str


EnvironmentActivationSchema = Annotated[
    VerifiersV1ConfigActivationSchema | PythonFactoryActivationSchema,
    Field(discriminator="kind"),
]


class EnvironmentBindingSchema(EvalCatalogSchema):
    id: str
    category: str
    source: EnvironmentSourceSchema
    activation: EnvironmentActivationSchema | None = None
    factory: str | None = None
    sampling: SamplingPolicySchema
    num_tasks: int = Field(gt=0)
    num_rollouts: int = Field(default=1, gt=0)
    max_concurrent: int = Field(default=4, gt=0)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    reward_components: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_one_activation(self) -> EnvironmentBindingSchema:
        if (self.activation is None) == (self.factory is None):
            raise ValueError("environment binding requires exactly one of activation or legacy factory")
        return self


class EvaluationPlanSchema(EvalCatalogSchema):
    id: str
    kind: Literal["general", "domain"]
    environments: tuple[str, ...]
    revision: str = "1"
    inference_requirements: dict[str, JsonValue] = Field(default_factory=dict)
    metrics_and_slices: tuple[str, ...] = ()
    aggregation: dict[str, JsonValue] = Field(default_factory=dict)
    comparison: dict[str, JsonValue] = Field(default_factory=dict)


def evaluation_catalog_decoders(
    factories: Mapping[str, EnvironmentActivation | EnvironmentFactory] | None = None,
) -> Mapping[SelectionFamily, SelectionDecoder]:
    """Build detached decoders without importing environment implementations."""

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
            source=EnvironmentSource(**payload.source.model_dump()),
            activation=activation,
            sampling=SamplingPolicy(**payload.sampling.model_dump()),
            num_tasks=payload.num_tasks,
            num_rollouts=payload.num_rollouts,
            max_concurrent=payload.max_concurrent,
            parameters=payload.parameters,
            reward_components=payload.reward_components,
        )

    def decode_evaluation(
        ref: CatalogRef,
        data: Mapping[str, object],
        known: Mapping[CatalogRef, Selection],
    ) -> Selection:
        del ref
        payload = EvaluationPlanSchema.model_validate(data)
        environments: list[EnvironmentBinding] = []
        for environment_id in payload.environments:
            value = known.get(CatalogRef("environment", environment_id))
            if not isinstance(value, EnvironmentBinding):
                raise ContractError(f"unresolved catalog link: environment/{environment_id}")
            environments.append(value)
        values = payload.model_dump(exclude={"environments"})
        return EvaluationPlan(environments=tuple(environments), **values)

    return cast(
        Mapping[SelectionFamily, SelectionDecoder],
        {
            "environment": decode_environment,
            "evaluation": decode_evaluation,
        },
    )


def _activation_from_schema(
    payload: VerifiersV1ConfigActivationSchema | PythonFactoryActivationSchema,
) -> EnvironmentActivation:
    if isinstance(payload, VerifiersV1ConfigActivationSchema):
        return VerifiersV1ConfigActivation(payload.config)
    return PythonFactoryActivation(payload.reference)


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


def _normalize_activation(
    value: EnvironmentActivation | EnvironmentFactory,
) -> EnvironmentActivation:
    if isinstance(value, (VerifiersV1ConfigActivation, PythonFactoryActivation)):
        return value
    return PythonFactoryActivation.from_callable(value)


__all__ = [
    "EnvironmentActivationSchema",
    "EnvironmentBindingSchema",
    "EvaluationPlanSchema",
    "PythonFactoryActivationSchema",
    "VerifiersV1ConfigActivationSchema",
    "evaluation_catalog_decoders",
]
