"""Qualification wrappers plus re-exports of technique-stable standard jobs.

SFT/DPO/GRPO/distill/serve definitions come from ``posttrain.jobs``. The
evaluation and transform helpers below only inject lab-specific qualification
knobs (endpoint, context window, output id). They are not a second definition
registry for product projects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Literal

from posttrain.common import (
    ExecutionTarget,
    InferenceBinding,
    ModelVariant,
    RunContext,
    StoredArtifactRef,
    TrackioArtifactRef,
)
from posttrain.eval import (
    EnvironmentBinding,
    EvaluateRequest,
    EvaluationBudget,
    EvaluationEndpoint,
    EvaluationPlan,
    EvaluationResult,
)
from posttrain.jobs import (
    distillation_definition,
    dpo_definition,
    grpo_definition,
    serve_benchmark_definition,
    serve_smoke_definition,
    sft_definition,
)
from posttrain.serve import ServeLaunchRequest
from posttrain.train import QuantizationPlan, TransformRequest

from .contracts import JobDefinition, ResolvedSeats

_DEFAULT_EVALUATION_BUDGET = EvaluationBudget()


def general_evaluation_definition(
    endpoint: EvaluationEndpoint,
    operation: Callable[[RunContext, EvaluateRequest], object],
    *,
    context_window: int,
    budget: EvaluationBudget = _DEFAULT_EVALUATION_BUDGET,
    definition_id: str = "eval/verifiers-general@1",
    description: str = "Run the selected general evaluation cell through Verifiers.",
) -> JobDefinition:
    """Lab qualification wrapper: bind an explicit endpoint and context window."""

    def run(context: RunContext, seats: ResolvedSeats) -> object:
        model = _seat(seats, "model", ModelVariant)
        plan = _seat(seats, "evaluation_plan", EvaluationPlan)
        environment = _seat(seats, "environment", EnvironmentBinding)
        inference = _seat(seats, "evaluation_inference", InferenceBinding)
        target = _seat(seats, "target", ExecutionTarget)
        if inference.model != model:
            raise ValueError("evaluation model conflicts with its inference binding")
        if plan.environment(environment.id) != environment:
            raise ValueError("evaluation environment is not a cell in the selected plan")
        return operation(
            context,
            EvaluateRequest(
                model=model,
                plan=plan,
                inference=inference,
                target=target,
                endpoint=endpoint,
                environment_id=environment.id,
                context_window=context_window,
                budget=budget,
            ),
        )

    return JobDefinition(
        definition_id,
        "eval.general",
        {
            "model": ModelVariant,
            "evaluation_plan": EvaluationPlan,
            "environment": EnvironmentBinding,
            "evaluation_inference": InferenceBinding,
            "target": ExecutionTarget,
        },
        run,
        description,
    )


def model_transform_definition(
    operation: Callable[[RunContext, TransformRequest], object],
    *,
    output_id: str,
    definition_id: str = "model/llm-compressor@2",
    description: str = "Transform the selected model through the lab qualification adapter.",
) -> JobDefinition:
    """Lab qualification wrapper: pin the expected transform output id."""

    def run(context: RunContext, seats: ResolvedSeats) -> object:
        return operation(
            context,
            TransformRequest(
                model=_seat(seats, "model", ModelVariant),
                plan=_seat(seats, "quantization", QuantizationPlan),
                target=_seat(seats, "target", ExecutionTarget),
                output_id=output_id,
            ),
        )

    return JobDefinition(
        definition_id,
        "model.transform",
        {
            "model": ModelVariant,
            "quantization": QuantizationPlan,
            "target": ExecutionTarget,
        },
        run,
        description,
    )


def managed_evaluation_definition(
    operation: Callable[[RunContext, ServeLaunchRequest, EvaluateRequest], EvaluationResult],
    *,
    context_window: int,
    budget: EvaluationBudget = _DEFAULT_EVALUATION_BUDGET,
    kind: Literal["eval.general", "eval.domain"] = "eval.domain",
    definition_id: str = "eval/verifiers-managed@1",
    description: str = "Launch the selected model and run the lab's managed evaluation cell.",
) -> JobDefinition:
    """Lab qualification wrapper: materialize artifacts then run managed eval."""

    def run(context: RunContext, seats: ResolvedSeats) -> EvaluationResult:
        model = _seat(seats, "model", ModelVariant)
        inference = _seat(seats, "evaluation_inference", InferenceBinding)
        target = _seat(seats, "target", ExecutionTarget)
        plan = _seat(seats, "evaluation_plan", EvaluationPlan)
        environment = _seat(seats, "environment", EnvironmentBinding)
        if isinstance(model.artifact, (StoredArtifactRef, TrackioArtifactRef)):
            input_name = "model_adapter" if model.form in {"adapter", "peft-adapter"} else "model_weights"
            local = context.input_artifact(input_name)
            model = replace(model, artifact=local, revision=None, digest=local.digest)
            inference = replace(inference, model=model)
        launch_request = ServeLaunchRequest(inference)
        request = EvaluateRequest(
            model,
            plan,
            inference,
            target,
            EvaluationEndpoint(launch_request.endpoint.base_url, launch_request.endpoint.model),
            environment.id,
            context_window,
            budget=budget,
        )
        return operation(context, launch_request, request)

    return JobDefinition(
        definition_id,
        kind,
        {
            "model": ModelVariant,
            "evaluation_inference": InferenceBinding,
            "target": ExecutionTarget,
            "evaluation_plan": EvaluationPlan,
            "environment": EnvironmentBinding,
        },
        run,
        description,
    )


def _seat[SelectionT: object](
    seats: ResolvedSeats,
    name: str,
    expected: type[SelectionT],
) -> SelectionT:
    value = seats[name]
    if not isinstance(value, expected):
        raise TypeError(f"resolved seat {name!r} has the wrong type")
    return value


__all__ = [
    "distillation_definition",
    "dpo_definition",
    "general_evaluation_definition",
    "grpo_definition",
    "managed_evaluation_definition",
    "model_transform_definition",
    "serve_benchmark_definition",
    "serve_smoke_definition",
    "sft_definition",
]
