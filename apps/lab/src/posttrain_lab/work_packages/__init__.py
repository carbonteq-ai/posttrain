"""Reference work-package compositions."""

from .contracts import (
    JobDefinition,
    Recipe,
    RecipeJob,
    WorkPackage,
    WorkPackageJobResult,
    WorkPackageResult,
    WorkPackageSchema,
    load_work_package,
)
from .definitions import (
    distillation_definition,
    dpo_definition,
    general_evaluation_definition,
    grpo_definition,
    managed_evaluation_definition,
    model_transform_definition,
    serve_benchmark_definition,
    serve_smoke_definition,
    sft_definition,
)
from .foundation_screen import (
    FOUNDATION_SCREEN_RECIPE,
    QWEN_FOUNDATION_SCREEN,
    ResolvedScreenPackage,
    ScreenPackage,
    resolve_screen_package,
)
from .qualification import (
    GSM8K_QUALIFICATION,
    QualificationPackage,
    ResolvedQualificationPackage,
    resolve_qualification_package,
)
from .runner import (
    ResolvedSeat,
    ResolvedWorkPackage,
    WorkPackageContext,
    resolve_work_package,
    run_work_package,
    validate_work_package,
)

__all__ = [
    "FOUNDATION_SCREEN_RECIPE",
    "GSM8K_QUALIFICATION",
    "JobDefinition",
    "QWEN_FOUNDATION_SCREEN",
    "QualificationPackage",
    "Recipe",
    "RecipeJob",
    "ResolvedQualificationPackage",
    "ResolvedScreenPackage",
    "ScreenPackage",
    "ResolvedSeat",
    "ResolvedWorkPackage",
    "WorkPackage",
    "WorkPackageContext",
    "WorkPackageJobResult",
    "WorkPackageResult",
    "WorkPackageSchema",
    "general_evaluation_definition",
    "dpo_definition",
    "distillation_definition",
    "grpo_definition",
    "load_work_package",
    "managed_evaluation_definition",
    "model_transform_definition",
    "resolve_qualification_package",
    "resolve_work_package",
    "resolve_screen_package",
    "run_work_package",
    "validate_work_package",
    "serve_benchmark_definition",
    "serve_smoke_definition",
    "sft_definition",
]
