"""Reference screen work package over the shared composition contracts."""

from __future__ import annotations

from dataclasses import dataclass

from posttrain.common import Catalog, CatalogRef, ExecutionTarget, InferenceBinding, JsonValue, ModelVariant, Workload
from posttrain.serve import ServeBenchmarkRequest

from .contracts import Recipe, RecipeJob, WorkPackage
from .runner import resolve_work_package

ScreenPackage = WorkPackage


@dataclass(frozen=True, slots=True)
class ResolvedScreenPackage:
    definition: WorkPackage
    request: ServeBenchmarkRequest
    snapshot: dict[str, JsonValue]


FOUNDATION_SCREEN_RECIPE = Recipe(
    id="recipes/foundation-screen@1",
    revision="1",
    stage="screen",
    seats={
        "model": "model",
        "screen_inference": "inference",
        "evaluation_inference": "inference",
        "workload": "workload",
        "target": "target",
        "evaluation_plan": "evaluation",
        "environment": "environment",
    },
    jobs=(
        RecipeJob(
            id="benchmark",
            kind="serve.benchmark",
            definition="serve/vllm-benchmark@1",
        ),
        RecipeJob(
            id="general-eval",
            kind="eval.general",
            definition="eval/verifiers-general@1",
            optional=True,
        ),
    ),
    expected_artifacts=("serve benchmark result", "optional native evaluation result"),
)

QWEN_FOUNDATION_SCREEN = WorkPackage(
    project_id="foundation-models",
    work_package_id="screen/qwen3.5-2b-smoke",
    stage="screen",
    recipe=FOUNDATION_SCREEN_RECIPE,
    bindings={
        "model": CatalogRef("model", "models/qwen3.5-2b@bf16"),
        "screen_inference": CatalogRef("inference", "inference/qwen3.5-2b-vllm-screen@1"),
        "evaluation_inference": CatalogRef("inference", "inference/qwen3.5-2b-vllm-eval@1"),
        "workload": CatalogRef("workload", "workloads/foundation-smoke-v1@1"),
        "target": CatalogRef("target", "targets/local-cuda-8gb"),
        "evaluation_plan": CatalogRef("evaluation", "general-smoke-v1"),
        "environment": CatalogRef("environment", "math-gsm8k"),
    },
    description="Establish whether the Qwen 3.5 2B foundation model meets the bounded serving screen before project-specific training begins.",
)


def resolve_screen_package(catalog: Catalog, package: WorkPackage) -> ResolvedScreenPackage:
    resolved = resolve_work_package(catalog, package)
    model = resolved.seat("model", ModelVariant)
    inference = resolved.seat("screen_inference", InferenceBinding)
    workload = resolved.seat("workload", Workload)
    target = resolved.seat("target", ExecutionTarget)
    if inference.model != model:
        raise ValueError("screen package model conflicts with its inference binding")
    request = ServeBenchmarkRequest(inference, workload, target)
    return ResolvedScreenPackage(package, request, dict(resolved.snapshot))


__all__ = [
    "FOUNDATION_SCREEN_RECIPE",
    "QWEN_FOUNDATION_SCREEN",
    "ResolvedScreenPackage",
    "ScreenPackage",
    "resolve_screen_package",
]
