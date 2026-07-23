"""Reference qualification package over the shared composition contracts."""

from __future__ import annotations

from dataclasses import dataclass

from posttrain.common import Catalog, CatalogRef, ExecutionTarget, InferenceBinding, JsonValue, ModelVariant
from posttrain.eval import (
    EnvironmentBinding,
    EvaluateRequest,
    EvaluationBudget,
    EvaluationEndpoint,
    EvaluationPlan,
)

from .contracts import Recipe, RecipeJob, WorkPackage
from .runner import resolve_work_package


@dataclass(frozen=True, slots=True)
class QualificationPackage:
    definition: WorkPackage
    context_window: int
    budget: EvaluationBudget

    @property
    def project_id(self) -> str:
        return self.definition.project_id

    @property
    def work_package_id(self) -> str:
        return self.definition.work_package_id

    def binding(self, name: str) -> CatalogRef:
        value = self.definition.bindings[name]
        if not isinstance(value, CatalogRef):
            raise TypeError(f"qualification binding {name!r} is inline")
        return value


@dataclass(frozen=True, slots=True)
class ResolvedQualificationPackage:
    definition: QualificationPackage
    model: ModelVariant
    inference: InferenceBinding
    target: ExecutionTarget
    plan: EvaluationPlan
    environment: EnvironmentBinding
    snapshot: dict[str, JsonValue]

    def request(self, endpoint: EvaluationEndpoint) -> EvaluateRequest:
        return EvaluateRequest(
            model=self.model,
            plan=self.plan,
            inference=self.inference,
            target=self.target,
            endpoint=endpoint,
            environment_id=self.environment.id,
            context_window=self.definition.context_window,
            budget=self.definition.budget,
        )


QUALIFICATION_RECIPE = Recipe(
    id="recipes/general-qualification@1",
    revision="1",
    stage="qualify",
    seats={
        "model": "model",
        "evaluation_inference": "inference",
        "target": "target",
        "evaluation_plan": "evaluation",
        "environment": "environment",
    },
    jobs=(
        RecipeJob(
            id="general-eval",
            kind="eval.general",
            definition="eval/verifiers-general@1",
        ),
    ),
    expected_artifacts=("native evaluation result",),
)

GSM8K_QUALIFICATION = QualificationPackage(
    definition=WorkPackage(
        project_id="foundation-models",
        work_package_id="qualify/qwen3.5-2b-gsm8k",
        stage="qualify",
        recipe=QUALIFICATION_RECIPE,
        bindings={
            "model": CatalogRef("model", "models/qwen3.5-2b@bf16"),
            "evaluation_inference": CatalogRef("inference", "inference/qwen3.5-2b-vllm-eval@1"),
            "target": CatalogRef("target", "targets/local-cuda-8gb"),
            "evaluation_plan": CatalogRef("evaluation", "general-smoke-v1"),
            "environment": CatalogRef("environment", "math-gsm8k"),
        },
        description="Qualify the Qwen 3.5 2B candidate against the bounded GSM8K evaluation cell before handoff.",
    ),
    context_window=8_192,
    budget=EvaluationBudget(num_tasks=1, max_concurrent=1),
)


def resolve_qualification_package(
    catalog: Catalog,
    package: QualificationPackage,
) -> ResolvedQualificationPackage:
    resolved = resolve_work_package(catalog, package.definition)
    model = resolved.seat("model", ModelVariant)
    inference = resolved.seat("evaluation_inference", InferenceBinding)
    target = resolved.seat("target", ExecutionTarget)
    plan = resolved.seat("evaluation_plan", EvaluationPlan)
    environment = resolved.seat("environment", EnvironmentBinding)
    if inference.model != model:
        raise ValueError("qualification model conflicts with its inference binding")
    if inference.target != target:
        raise ValueError("qualification target conflicts with its inference binding")
    if plan.environment(environment.id) != environment:
        raise ValueError("qualification environment is not the resolved plan cell")
    return ResolvedQualificationPackage(
        package,
        model,
        inference,
        target,
        plan,
        environment,
        dict(resolved.snapshot),
    )


__all__ = [
    "GSM8K_QUALIFICATION",
    "QUALIFICATION_RECIPE",
    "QualificationPackage",
    "ResolvedQualificationPackage",
    "resolve_qualification_package",
]
