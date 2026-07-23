"""Strict catalog schemas for evaluation plans and environment bindings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

from posttrain.common import CatalogRef, ContractError, JsonValue
from posttrain.common.catalog import SelectionDecoder
from posttrain.common.selections import Selection, SelectionFamily
from pydantic import BaseModel, ConfigDict, Field

from .requests import (
    EnvironmentBinding,
    EnvironmentFactory,
    EnvironmentSource,
    EvaluationPlan,
    SamplingPolicy,
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


class EnvironmentBindingSchema(EvalCatalogSchema):
    id: str
    category: str
    source: EnvironmentSourceSchema
    factory: str
    sampling: SamplingPolicySchema
    num_tasks: int = Field(gt=0)
    num_rollouts: int = Field(default=1, gt=0)
    max_concurrent: int = Field(default=4, gt=0)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    reward_components: tuple[str, ...] = ()


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
    factories: Mapping[str, EnvironmentFactory],
) -> Mapping[SelectionFamily, SelectionDecoder]:
    """Build host-bound decoders without serializing Python callables into YAML."""

    def decode_environment(
        ref: CatalogRef,
        data: Mapping[str, object],
        known: Mapping[CatalogRef, Selection],
    ) -> Selection:
        del ref, known
        payload = EnvironmentBindingSchema.model_validate(data)
        try:
            factory = factories[payload.factory]
        except KeyError as error:
            raise ContractError(f"environment factory is not registered: {payload.factory}") from error
        return EnvironmentBinding(
            id=payload.id,
            category=payload.category,
            source=EnvironmentSource(**payload.source.model_dump()),
            factory=factory,
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


__all__ = [
    "EnvironmentBindingSchema",
    "EvaluationPlanSchema",
    "evaluation_catalog_decoders",
]
