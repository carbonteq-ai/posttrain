"""First-party job definitions that translate resolved seats into requests."""

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
    Workload,
)
from posttrain.data import PreferenceDataSource, SupervisedDataSource
from posttrain.eval import (
    EnvironmentBinding,
    EvaluateRequest,
    EvaluationBudget,
    EvaluationEndpoint,
    EvaluationPlan,
    EvaluationResult,
)
from posttrain.serve import GenerationResult, ServeBenchmarkRequest, ServeLaunchRequest
from posttrain.train import (
    DPORequest,
    DPOSettings,
    GRPOSettings,
    OnPolicyDistillationSettings,
    QuantizationPlan,
    SFTRequest,
    SFTSettings,
    TrainingBinding,
    TransformRequest,
)

from ..jobs.environment_grpo import VerifiersGRPOJobRequest
from ..jobs.gsm8k_posttraining import GSM8KDistillationJobRequest
from .contracts import JobDefinition, ResolvedSeats

_DEFAULT_EVALUATION_BUDGET = EvaluationBudget()


def serve_benchmark_definition(
    operation: Callable[[RunContext, ServeBenchmarkRequest], object],
    *,
    definition_id: str = "serve/vllm-benchmark@1",
    description: str = "Launch the selected model and measure a bounded serving workload on the chosen execution target.",
) -> JobDefinition:
    def run(context: RunContext, seats: ResolvedSeats) -> object:
        model = _seat(seats, "model", ModelVariant)
        inference = _seat(seats, "screen_inference", InferenceBinding)
        workload = _seat(seats, "workload", Workload)
        target = _seat(seats, "target", ExecutionTarget)
        if inference.model != model:
            raise ValueError("serve benchmark model conflicts with its inference binding")
        return operation(context, ServeBenchmarkRequest(inference, workload, target))

    return JobDefinition(
        definition_id,
        "serve.benchmark",
        {
            "model": ModelVariant,
            "screen_inference": InferenceBinding,
            "workload": Workload,
            "target": ExecutionTarget,
        },
        run,
        description,
    )


def general_evaluation_definition(
    endpoint: EvaluationEndpoint,
    operation: Callable[[RunContext, EvaluateRequest], object],
    *,
    context_window: int,
    budget: EvaluationBudget = _DEFAULT_EVALUATION_BUDGET,
    definition_id: str = "eval/verifiers-general@1",
    description: str = "Run the selected general evaluation cell through Verifiers with a bounded task and concurrency budget.",
) -> JobDefinition:
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
        request = EvaluateRequest(
            model=model,
            plan=plan,
            inference=inference,
            target=target,
            endpoint=endpoint,
            environment_id=environment.id,
            context_window=context_window,
            budget=budget,
        )
        return operation(context, request)

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


def sft_definition(
    operation: Callable[[RunContext, SFTRequest], object],
    *,
    definition_id: str = "train/trl-sft@1",
    with_validation: bool = False,
    description: str = "Render supervised examples and update the selected model with the configured SFT and training bindings.",
) -> JobDefinition:
    def run(context: RunContext, seats: ResolvedSeats) -> object:
        request = SFTRequest(
            model=_seat(seats, "model", ModelVariant),
            data=_seat(seats, "dataset", SupervisedDataSource),
            settings=_seat(seats, "settings", SFTSettings),
            training=_seat(seats, "training", TrainingBinding),
            validation_data=(_seat(seats, "validation_dataset", SupervisedDataSource) if with_validation else None),
        )
        return operation(context, request)

    seat_types: dict[str, type[object]] = {
        "model": ModelVariant,
        "dataset": SupervisedDataSource,
        "settings": SFTSettings,
        "training": TrainingBinding,
    }
    if with_validation:
        seat_types["validation_dataset"] = SupervisedDataSource
    return JobDefinition(
        definition_id,
        "train.sft",
        seat_types,
        run,
        description,
    )


def dpo_definition(
    operation: Callable[[RunContext, DPORequest], object],
    *,
    definition_id: str = "train/trl-dpo@1",
    description: str = "Optimize the selected policy from preference pairs using the configured DPO objective and training binding.",
) -> JobDefinition:
    def run(context: RunContext, seats: ResolvedSeats) -> object:
        request = DPORequest(
            model=_seat(seats, "model", ModelVariant),
            data=_seat(seats, "dataset", PreferenceDataSource),
            settings=_seat(seats, "settings", DPOSettings),
            training=_seat(seats, "training", TrainingBinding),
        )
        return operation(context, request)

    return JobDefinition(
        definition_id,
        "train.dpo",
        {
            "model": ModelVariant,
            "dataset": PreferenceDataSource,
            "settings": DPOSettings,
            "training": TrainingBinding,
        },
        run,
        description,
    )


def grpo_definition(
    operation: Callable[[RunContext, VerifiersGRPOJobRequest], object],
    *,
    definition_id: str = "train/trl-grpo@1",
    description: str = "Generate grouped rollouts and update the selected policy with the configured GRPO objective.",
) -> JobDefinition:
    def run(context: RunContext, seats: ResolvedSeats) -> object:
        request = VerifiersGRPOJobRequest(
            model=_seat(seats, "model", ModelVariant),
            environment=_seat(seats, "environment", EnvironmentBinding),
            settings=_seat(seats, "settings", GRPOSettings),
            training=_seat(seats, "training", TrainingBinding),
            inference=_seat(seats, "rollout_inference", InferenceBinding),
        )
        return operation(context, request)

    return JobDefinition(
        definition_id,
        "train.grpo",
        {
            "model": ModelVariant,
            "environment": EnvironmentBinding,
            "settings": GRPOSettings,
            "training": TrainingBinding,
            "rollout_inference": InferenceBinding,
        },
        run,
        description,
    )


def distillation_definition(
    operation: Callable[[RunContext, GSM8KDistillationJobRequest], object],
    *,
    definition_id: str = "train/trl-distill@1",
    description: str = "Generate fresh student rollouts, score them with the frozen teacher, and apply the configured distillation update.",
) -> JobDefinition:
    def run(context: RunContext, seats: ResolvedSeats) -> object:
        request = GSM8KDistillationJobRequest(
            student=_seat(seats, "student", ModelVariant),
            teacher=_seat(seats, "teacher", ModelVariant),
            environment=_seat(seats, "environment", EnvironmentBinding),
            settings=_seat(seats, "settings", OnPolicyDistillationSettings),
            training=_seat(seats, "training", TrainingBinding),
            rollout_inference=_seat(seats, "rollout_inference", InferenceBinding),
            teacher_inference=_seat(seats, "teacher_inference", InferenceBinding),
        )
        return operation(context, request)

    return JobDefinition(
        definition_id,
        "train.distill",
        {
            "student": ModelVariant,
            "teacher": ModelVariant,
            "environment": EnvironmentBinding,
            "settings": OnPolicyDistillationSettings,
            "training": TrainingBinding,
            "rollout_inference": InferenceBinding,
            "teacher_inference": InferenceBinding,
        },
        run,
        description,
    )


def model_transform_definition(
    operation: Callable[[RunContext, TransformRequest], object],
    *,
    output_id: str,
    definition_id: str = "model/llm-compressor@2",
    description: str = "Transform the selected model with the resolved quantization plan and publish an immutable descendant artifact.",
) -> JobDefinition:
    def run(context: RunContext, seats: ResolvedSeats) -> object:
        request = TransformRequest(
            model=_seat(seats, "model", ModelVariant),
            plan=_seat(seats, "quantization", QuantizationPlan),
            target=_seat(seats, "target", ExecutionTarget),
            output_id=output_id,
        )
        return operation(context, request)

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


def serve_smoke_definition(
    operation: Callable[[RunContext, ServeLaunchRequest], GenerationResult],
    *,
    definition_id: str = "serve/vllm-smoke@1",
    description: str = "Launch the selected inference binding and execute a lightweight serving compatibility probe.",
) -> JobDefinition:
    def run(context: RunContext, seats: ResolvedSeats) -> GenerationResult:
        inference = _seat(seats, "inference", InferenceBinding)
        return operation(context, ServeLaunchRequest(inference))

    return JobDefinition(definition_id, "serve.smoke", {"inference": InferenceBinding}, run, description)


def managed_evaluation_definition(
    operation: Callable[[RunContext, ServeLaunchRequest, EvaluateRequest], EvaluationResult],
    *,
    context_window: int,
    budget: EvaluationBudget = _DEFAULT_EVALUATION_BUDGET,
    kind: Literal["eval.general", "eval.domain"] = "eval.domain",
    definition_id: str = "eval/verifiers-managed@1",
    description: str = "Launch the selected model and run the managed Verifiers evaluation cell against its recorded environment.",
) -> JobDefinition:
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
        endpoint = EvaluationEndpoint(launch_request.endpoint.base_url, launch_request.endpoint.model)
        request = EvaluateRequest(
            model,
            plan,
            inference,
            target,
            endpoint,
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
    "dpo_definition",
    "distillation_definition",
    "general_evaluation_definition",
    "grpo_definition",
    "managed_evaluation_definition",
    "model_transform_definition",
    "serve_benchmark_definition",
    "serve_smoke_definition",
    "sft_definition",
]
