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
    EvaluationBreakdownDefinition,
    EvaluationNumericPredicate,
    EvaluationPlan,
    EvaluationSignalRef,
    EvaluationSuccessDefinition,
    ExternalInferenceService,
    PythonFactoryActivation,
    RemoteEvaluationBinding,
    RemotePolicy,
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


class EvaluationSignalRefSchema(EvalCatalogSchema):
    namespace: Literal["reward", "metric"]
    name: str


class EvaluationNumericPredicateSchema(EvalCatalogSchema):
    operator: Literal["eq", "gt", "gte", "lt", "lte", "between"]
    value: float
    upper: float | None = None
    tolerance: float = Field(default=0.0, ge=0)


class EvaluationSuccessDefinitionSchema(EvalCatalogSchema):
    id: str
    label: str
    source: EvaluationSignalRefSchema
    predicate: EvaluationNumericPredicateSchema
    missing: Literal["error", "exclude"] = "error"


class EvaluationBreakdownDefinitionSchema(EvalCatalogSchema):
    id: str
    label: str
    dimensions: tuple[str, ...] = Field(min_length=2, max_length=2)
    presentation: Literal["matrix"] = "matrix"
    multi_value: Literal["reject", "cross"] = "reject"
    missing: Literal["exclude", "bucket"] = "exclude"


class EvaluationPlanSchema(EvalCatalogSchema):
    id: str
    kind: Literal["general", "domain"]
    environments: tuple[str, ...]
    revision: str = "1"
    inference_requirements: dict[str, JsonValue] = Field(default_factory=dict)
    metrics_and_slices: tuple[str, ...] = ()
    success: dict[str, EvaluationSuccessDefinitionSchema] = Field(default_factory=dict)
    breakdowns: dict[str, tuple[EvaluationBreakdownDefinitionSchema, ...]] = Field(default_factory=dict)
    aggregation: dict[str, JsonValue] = Field(default_factory=dict)
    comparison: dict[str, JsonValue] = Field(default_factory=dict)


class RemotePolicySchema(EvalCatalogSchema):
    id: str
    revision: str
    model: str
    context_window: int = Field(gt=0)
    capabilities: dict[str, JsonValue] = Field(default_factory=dict)


class ExternalInferenceServiceSchema(EvalCatalogSchema):
    id: str
    revision: str
    base_url: str
    api_key_var: str
    headers: dict[str, str] = Field(default_factory=dict)
    request_defaults: dict[str, JsonValue] = Field(default_factory=dict)
    protocol: Literal["openai-chat@1"] = "openai-chat@1"


class RemoteEvaluationBindingSchema(EvalCatalogSchema):
    id: str
    revision: str
    policy: RemotePolicySchema
    service: ExternalInferenceServiceSchema
    purpose: tuple[Literal["screen", "eval"], ...]


def evaluation_catalog_decoders(
    factories: Mapping[str, EnvironmentActivation | EnvironmentFactory] | None = None,
) -> Mapping[SelectionFamily, SelectionDecoder]:
    """Build evaluation decoders; environment decoding moved to posttrain.environment."""

    # Retain the argument for one release so callers using the former combined
    # decoder factory do not fail at import time. Environment aliases are now
    # interpreted by posttrain.environment.environment_catalog_decoders().
    del factories

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
        values["success"] = {
            environment_id: EvaluationSuccessDefinition(
                id=definition.id,
                label=definition.label,
                source=EvaluationSignalRef(**definition.source.model_dump()),
                predicate=EvaluationNumericPredicate(**definition.predicate.model_dump()),
                missing=definition.missing,
            )
            for environment_id, definition in payload.success.items()
        }
        values["breakdowns"] = {
            environment_id: tuple(
                EvaluationBreakdownDefinition(**definition.model_dump()) for definition in definitions
            )
            for environment_id, definitions in payload.breakdowns.items()
        }
        return EvaluationPlan(environments=tuple(environments), **values)

    def decode_remote_evaluation(
        ref: CatalogRef,
        data: Mapping[str, object],
        known: Mapping[CatalogRef, Selection],
    ) -> Selection:
        del ref, known
        payload = RemoteEvaluationBindingSchema.model_validate(data)
        return RemoteEvaluationBinding(
            id=payload.id,
            revision=payload.revision,
            policy=RemotePolicy(**payload.policy.model_dump()),
            service=ExternalInferenceService(**payload.service.model_dump()),
            purpose=payload.purpose,
        )

    return cast(
        Mapping[SelectionFamily, SelectionDecoder],
        {
            "evaluation": decode_evaluation,
            "remote-evaluation": decode_remote_evaluation,
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
    "RemoteEvaluationBindingSchema",
    "RemotePolicySchema",
    "VerifiersV1ConfigActivationSchema",
    "evaluation_catalog_decoders",
]
