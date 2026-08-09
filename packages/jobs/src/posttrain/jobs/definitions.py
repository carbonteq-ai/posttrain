"""Technique-stable standard job definitions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any, Literal

from posttrain.common import (
    ContractError,
    ExecutionTarget,
    InferenceBinding,
    LocalArtifactRef,
    ModelVariant,
    RunContext,
    StoredArtifactRef,
    TrackioArtifactRef,
    Workload,
)
from posttrain.data import (
    DatasetLoadPlan,
    DatasetPrepareRequest,
    PreferenceDataSource,
    SupervisedDataSource,
    prepare,
)
from posttrain.environment import EnvironmentBinding
from posttrain.eval import (
    EvaluateRequest,
    EvaluationBudget,
    EvaluationEndpoint,
    EvaluationPlan,
    EvaluationResult,
    RemoteEvaluationBinding,
    domain,
    general,
)
from posttrain.serve import (
    GenerationRequest,
    GenerationResult,
    ProbeResult,
    ServeBenchmarkRequest,
    ServeLaunchRequest,
    benchmark,
    generate,
    launch,
    probe,
)
from posttrain.train import (
    DPORequest,
    DPOSettings,
    GRPORequest,
    GRPOSettings,
    OnPolicyDistillationRequest,
    OnPolicyDistillationSettings,
    QuantizationPlan,
    SAMPORequest,
    SAMPOSettings,
    SFTRequest,
    SFTSettings,
    TrainingBinding,
    TransformRequest,
    TransformResult,
    build_verifiers_distillation_request,
    build_verifiers_grpo_request,
    build_verifiers_sampo_request,
    distill,
    dpo,
    grpo,
    run_llm_compressor,
    sampo,
    sft,
    transform,
)
from posttrain.work import JobDefinition, ResolvedSeats

_DEFAULT_EVALUATION_BUDGET = EvaluationBudget()


def supervised_data_prepare_definition(
    operation: Callable[[RunContext, DatasetPrepareRequest], object] = prepare,
    *,
    definition_id: str = "data/canonicalize-supervised@1",
) -> JobDefinition:
    """Build the standard supervised dataset canonicalization job."""

    def run(context: RunContext, seats: ResolvedSeats) -> object:
        _seat(seats, "target", ExecutionTarget)
        data = _seat(seats, "dataset", SupervisedDataSource)
        if data.descriptor.kind != "supervised":
            raise TypeError("supervised data.prepare requires supervised data")
        return operation(context, DatasetPrepareRequest(data))

    return JobDefinition(
        definition_id,
        "data.prepare",
        {"dataset": SupervisedDataSource, "target": ExecutionTarget},
        run,
        "Validate and retain one canonical supervised dataset snapshot.",
        required_artifact_roles=("dataset",),
        selection_seats={"dataset": DatasetLoadPlan},
        static_validator=_validate_supervised_prepare_seats,
    )


def preference_data_prepare_definition(
    operation: Callable[[RunContext, DatasetPrepareRequest], object] = prepare,
    *,
    definition_id: str = "data/canonicalize-preference@1",
) -> JobDefinition:
    """Build the standard preference dataset canonicalization job."""

    def run(context: RunContext, seats: ResolvedSeats) -> object:
        _seat(seats, "target", ExecutionTarget)
        data = _seat(seats, "dataset", PreferenceDataSource)
        if data.descriptor.kind != "preference":
            raise TypeError("preference data.prepare requires preference data")
        return operation(context, DatasetPrepareRequest(data))

    return JobDefinition(
        definition_id,
        "data.prepare",
        {"dataset": PreferenceDataSource, "target": ExecutionTarget},
        run,
        "Validate and retain one canonical preference dataset snapshot.",
        required_artifact_roles=("dataset",),
        selection_seats={"dataset": DatasetLoadPlan},
        static_validator=_validate_preference_prepare_seats,
    )


def sft_definition(
    operation: Callable[[RunContext, SFTRequest], object] = sft,
    *,
    definition_id: str = "train/trl-sft@1",
    with_validation: bool = False,
) -> JobDefinition:
    def run(context: RunContext, seats: ResolvedSeats) -> object:
        model = _materialize_selected_model_variant(context, _seat(seats, "model", ModelVariant))
        return operation(
            context,
            SFTRequest(
                model=model,
                data=_seat(seats, "dataset", SupervisedDataSource),
                settings=_seat(seats, "settings", SFTSettings),
                training=_seat(seats, "training", TrainingBinding),
                resume_from=_recovery_checkpoint(context),
                validation_data=(_seat(seats, "validation_dataset", SupervisedDataSource) if with_validation else None),
            ),
        )

    seat_types: dict[str, type[object]] = {
        "model": ModelVariant,
        "dataset": SupervisedDataSource,
        "settings": SFTSettings,
        "training": TrainingBinding,
    }
    if with_validation:
        seat_types["validation_dataset"] = SupervisedDataSource
    selection_seats = {"dataset": DatasetLoadPlan}
    if with_validation:
        selection_seats["validation_dataset"] = DatasetLoadPlan
    return JobDefinition(
        definition_id,
        "train.sft",
        seat_types,
        run,
        "Render supervised examples and update the selected model with the configured SFT bindings.",
        required_artifact_roles=("model", "summary"),
        selection_seats=selection_seats,
        static_validator=_validate_supervised_dataset_seats,
    )


def dpo_definition(
    operation: Callable[[RunContext, DPORequest], object] = dpo,
    *,
    definition_id: str = "train/trl-dpo@1",
) -> JobDefinition:
    def run(context: RunContext, seats: ResolvedSeats) -> object:
        model = _materialize_selected_model_variant(context, _seat(seats, "model", ModelVariant))
        return operation(
            context,
            DPORequest(
                model=model,
                data=_seat(seats, "dataset", PreferenceDataSource),
                settings=_seat(seats, "settings", DPOSettings),
                training=_seat(seats, "training", TrainingBinding),
                resume_from=_recovery_checkpoint(context),
            ),
        )

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
        "Optimize the selected policy from preference pairs using the configured DPO objective.",
        required_artifact_roles=("model", "summary"),
        selection_seats={"dataset": DatasetLoadPlan},
        static_validator=_validate_preference_dataset_seats,
    )


def grpo_definition(
    operation: Callable[[RunContext, GRPORequest], object] = grpo,
    *,
    tasks: Mapping[int, Any] | None = None,
    definition_id: str = "train/trl-grpo@1",
) -> JobDefinition:
    def run(context: RunContext, seats: ResolvedSeats) -> object:
        policy, inference = _materialize_grpo_policy(
            context,
            _seat(seats, "model", ModelVariant),
            _seat(seats, "rollout_inference", InferenceBinding),
        )
        request = build_verifiers_grpo_request(
            policy=policy,
            environment=_seat(seats, "environment", EnvironmentBinding),
            settings=_seat(seats, "settings", GRPOSettings),
            training=_seat(seats, "training", TrainingBinding),
            inference=inference,
            trace_path=context.workspace / "training" / "grpo" / "verifiers-traces.jsonl",
            run_id=context.run_id,
            tasks=tasks,
        )
        return operation(context, replace(request, resume_from=_recovery_checkpoint(context)))

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
        "Generate grouped Verifiers rollouts and update the selected policy with the selected GRPO-family objective.",
        required_artifact_roles=("model", "summary"),
        static_validator=_validate_online_rl_batch_seats,
    )


def _materialize_grpo_policy(
    context: RunContext,
    policy: ModelVariant,
    inference: InferenceBinding,
) -> tuple[ModelVariant, InferenceBinding]:
    """Resolve a stored adapter input before constructing an online-RL request.

    A catalog adapter reference identifies an artifact owned by the tracking
    provider; it is not a path that exists inside the worker. The execution
    runner declares it as ``model_adapter`` and materializes it before invoking
    the job definition. Keep both policy and inference bound to that local
    artifact so TRL and vLLM consume the same adapter weights.
    """

    if "model_adapter" in context.input_artifacts or "model_weights" in context.input_artifacts:
        policy = _materialize_selected_model_variant(context, policy)
        return policy, replace(inference, model=policy)
    if policy.form not in {"adapter", "peft-adapter"} or not isinstance(
        policy.artifact, (StoredArtifactRef, TrackioArtifactRef)
    ):
        return policy, inference
    policy = _materialize_selected_model_variant(context, policy)
    return policy, replace(inference, model=policy)


def distillation_definition(
    operation: Callable[[RunContext, OnPolicyDistillationRequest], object] = distill,
    *,
    tasks: Mapping[int, Any] | None = None,
    definition_id: str = "train/trl-distill@1",
) -> JobDefinition:
    def run(context: RunContext, seats: ResolvedSeats) -> object:
        student, rollout_inference = _materialize_selected_model(
            context,
            _seat(seats, "student", ModelVariant),
            _seat(seats, "rollout_inference", InferenceBinding),
        )
        request = build_verifiers_distillation_request(
            student=student,
            teacher=_seat(seats, "teacher", ModelVariant),
            environment=_seat(seats, "environment", EnvironmentBinding),
            settings=_seat(seats, "settings", OnPolicyDistillationSettings),
            training=_seat(seats, "training", TrainingBinding),
            rollout_inference=rollout_inference,
            teacher_inference=_seat(seats, "teacher_inference", InferenceBinding),
            trace_path=context.workspace / "training" / "distill" / "verifiers-traces.jsonl",
            run_id=context.run_id,
            tasks=tasks,
        )
        return operation(context, replace(request, resume_from=_recovery_checkpoint(context)))

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
        "Generate fresh student rollouts, score with the teacher, and apply distillation.",
        required_artifact_roles=("model", "summary"),
    )


def sampo_definition(
    operation: Callable[[RunContext, SAMPORequest], object] = sampo,
    *,
    tasks: Mapping[int, Any] | None = None,
    definition_id: str = "train/trl-sampo@1",
) -> JobDefinition:
    def run(context: RunContext, seats: ResolvedSeats) -> object:
        request = build_verifiers_sampo_request(
            policy=_seat(seats, "model", ModelVariant),
            environment=_seat(seats, "environment", EnvironmentBinding),
            settings=_seat(seats, "settings", SAMPOSettings),
            training=_seat(seats, "training", TrainingBinding),
            inference=_seat(seats, "rollout_inference", InferenceBinding),
            trace_path=context.workspace / "training" / "sampo" / "verifiers-traces.jsonl",
            run_id=context.run_id,
            tasks=tasks,
        )
        return operation(context, replace(request, resume_from=_recovery_checkpoint(context)))

    return JobDefinition(
        definition_id,
        "train.sampo",
        {
            "model": ModelVariant,
            "environment": EnvironmentBinding,
            "settings": SAMPOSettings,
            "training": TrainingBinding,
            "rollout_inference": InferenceBinding,
        },
        run,
        "Train a multi-turn tool policy with sequence clipping and hierarchical episode/turn advantages.",
        required_artifact_roles=("model", "summary"),
    )


def serve_benchmark_definition(
    operation: Callable[[RunContext, ServeBenchmarkRequest], object] = benchmark,
    *,
    definition_id: str = "serve/vllm-benchmark@1",
) -> JobDefinition:
    def run(context: RunContext, seats: ResolvedSeats) -> object:
        model = _seat(seats, "model", ModelVariant)
        inference = _seat(seats, "screen_inference", InferenceBinding)
        model, inference = _materialize_selected_model(context, model, inference)
        if inference.model != model:
            raise ValueError("serve benchmark model conflicts with its inference binding")
        return operation(
            context,
            ServeBenchmarkRequest(
                inference,
                _seat(seats, "workload", Workload),
                _seat(seats, "target", ExecutionTarget),
            ),
        )

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
        "Measure a bounded serving workload on the selected execution target.",
        required_artifact_roles=("benchmark",),
    )


def _serve_smoke(context: RunContext, request: ServeLaunchRequest) -> ProbeResult:
    with launch(context, request) as endpoint:
        return probe(context, endpoint)


def _materialize_selected_model(
    context: RunContext,
    model: ModelVariant,
    inference: InferenceBinding,
) -> tuple[ModelVariant, InferenceBinding]:
    """Use a run-selected model view while retaining the catalog interface facts."""

    model = _materialize_selected_model_variant(context, model)
    return model, replace(inference, model=model)


def _materialize_selected_model_variant(context: RunContext, model: ModelVariant) -> ModelVariant:
    input_name = (
        "model_adapter"
        if "model_adapter" in context.input_artifacts
        else ("model_weights" if "model_weights" in context.input_artifacts else None)
    )
    if input_name is None:
        if model.form not in {"adapter", "peft-adapter"} or not isinstance(
            model.artifact, (StoredArtifactRef, TrackioArtifactRef)
        ):
            return model
        input_name = "model_adapter"
    local = context.input_artifact(input_name)
    form = "adapter" if input_name == "model_adapter" else "full-finetuned"
    return replace(
        model,
        artifact=local,
        form=form,
        revision=None,
        digest=local.digest,
        parent=(model.parent or model.id) if form == "adapter" else model.parent,
    )


def serve_smoke_definition(
    operation: Callable[[RunContext, ServeLaunchRequest], object] = _serve_smoke,
    *,
    definition_id: str = "serve/vllm-smoke@1",
) -> JobDefinition:
    def run(context: RunContext, seats: ResolvedSeats) -> object:
        model, inference = _materialize_selected_model(
            context,
            _seat(seats, "inference", InferenceBinding).model,
            _seat(seats, "inference", InferenceBinding),
        )
        del model
        return operation(context, ServeLaunchRequest(inference))

    return JobDefinition(
        definition_id,
        "serve.smoke",
        {"inference": InferenceBinding},
        run,
        "Launch the selected inference binding and execute a health probe.",
    )


def _serve_generation_smoke(context: RunContext, request: ServeLaunchRequest) -> GenerationResult:
    model = request.inference.model
    max_tokens = request.inference.sampling.get("max_tokens", 128)
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
        raise ValueError("generation smoke inference max_tokens must be an integer")
    with launch(context, request) as endpoint:
        health = probe(context, endpoint)
        if not health.model_available:
            raise RuntimeError(f"launched endpoint does not expose {endpoint.model!r}")
        result = generate(
            context,
            GenerationRequest(
                endpoint=endpoint,
                messages=({"role": "user", "content": "What is 2 + 2? Answer concisely."},),
                max_tokens=max_tokens,
            ),
            model,
        )
        if not result.content.strip():
            raise RuntimeError(f"generation smoke produced no final answer (finish_reason={result.finish_reason!r})")
        return result


def serve_generation_smoke_definition(
    operation: Callable[[RunContext, ServeLaunchRequest], object] = _serve_generation_smoke,
    *,
    definition_id: str = "serve/vllm-generation-smoke@1",
) -> JobDefinition:
    """Launch an inference binding and require one nonempty text completion."""

    def run(context: RunContext, seats: ResolvedSeats) -> object:
        inference = _seat(seats, "inference", InferenceBinding)
        _model, inference = _materialize_selected_model(context, inference.model, inference)
        return operation(context, ServeLaunchRequest(inference))

    return JobDefinition(
        definition_id,
        "serve.smoke",
        {"inference": InferenceBinding},
        run,
        "Launch the selected inference binding and require one nonempty text completion.",
    )


def general_evaluation_definition(
    operation: Callable[[RunContext, EvaluateRequest], object] = general,
    *,
    budget: EvaluationBudget = _DEFAULT_EVALUATION_BUDGET,
    definition_id: str = "eval/verifiers-general@1",
) -> JobDefinition:
    def run(context: RunContext, seats: ResolvedSeats) -> object:
        request = _evaluation_request(seats, budget=budget)
        return operation(context, request)

    return _evaluation_job(
        definition_id,
        "eval.general",
        run,
        "Run one general Verifiers evaluation cell against its declared endpoint.",
    )


def remote_evaluation_definition(
    operation: Callable[[RunContext, EvaluateRequest], object] | None = None,
    *,
    budget: EvaluationBudget = _DEFAULT_EVALUATION_BUDGET,
    kind: Literal["eval.general", "eval.domain"] = "eval.general",
    definition_id: str = "eval/verifiers-remote-general@1",
) -> JobDefinition:
    """Run one Verifiers cell through an externally owned remote-policy binding."""

    def run(context: RunContext, seats: ResolvedSeats) -> object:
        binding = _seat(seats, "remote_evaluation", RemoteEvaluationBinding)
        plan = _seat(seats, "evaluation_plan", EvaluationPlan)
        environment = _seat(seats, "environment", EnvironmentBinding)
        if plan.environment(environment.id) != environment:
            raise ValueError("evaluation environment is not a cell in the selected plan")
        request = EvaluateRequest(
            model=binding.policy,
            plan=plan,
            inference=binding,
            target=_seat(seats, "target", ExecutionTarget),
            endpoint=None,
            environment_id=environment.id,
            context_window=binding.policy.context_window,
            budget=budget,
        )
        active = operation or (general if kind == "eval.general" else domain)
        return active(context, request)

    return JobDefinition(
        definition_id,
        kind,
        {
            "remote_evaluation": RemoteEvaluationBinding,
            "target": ExecutionTarget,
            "evaluation_plan": EvaluationPlan,
            "environment": EnvironmentBinding,
        },
        run,
        "Run one Verifiers evaluation cell through a declared external OpenAI-compatible service.",
        required_artifact_roles=("evaluation",),
    )


def _managed_evaluation(
    context: RunContext,
    launch_request: ServeLaunchRequest,
    request: EvaluateRequest,
) -> EvaluationResult:
    with launch(context, launch_request) as endpoint:
        live = replace(
            request,
            endpoint=EvaluationEndpoint(endpoint.base_url, endpoint.model),
        )
        return general(context, live) if live.plan.kind == "general" else domain(context, live)


def managed_evaluation_definition(
    operation: Callable[[RunContext, ServeLaunchRequest, EvaluateRequest], object] = _managed_evaluation,
    *,
    budget: EvaluationBudget = _DEFAULT_EVALUATION_BUDGET,
    kind: Literal["eval.general", "eval.domain"] = "eval.domain",
    definition_id: str = "eval/verifiers-managed@1",
) -> JobDefinition:
    def run(context: RunContext, seats: ResolvedSeats) -> object:
        request = _evaluation_request(seats, budget=budget, materialize_model=context)
        if not isinstance(request.inference, InferenceBinding):
            raise AssertionError("managed evaluation requires a local inference binding")
        return operation(context, ServeLaunchRequest(request.inference), request)

    return _evaluation_job(
        definition_id,
        kind,
        run,
        "Launch the selected model and run its managed Verifiers evaluation cell.",
    )


def managed_general_evaluation_definition(
    operation: Callable[[RunContext, ServeLaunchRequest, EvaluateRequest], object] = _managed_evaluation,
    *,
    budget: EvaluationBudget = _DEFAULT_EVALUATION_BUDGET,
    definition_id: str = "eval/verifiers-managed-general@1",
) -> JobDefinition:
    """Build a self-contained general-evaluation cell with a managed endpoint."""

    return managed_evaluation_definition(
        operation,
        budget=budget,
        kind="eval.general",
        definition_id=definition_id,
    )


def model_transform_definition(
    operation: Callable[[RunContext, TransformRequest], TransformResult] | None = None,
    *,
    definition_id: str = "model/llm-compressor@2",
) -> JobDefinition:
    def default_operation(context: RunContext, request: TransformRequest) -> TransformResult:
        return transform(context, request, runner=run_llm_compressor)

    selected_operation = operation or default_operation

    def run(context: RunContext, seats: ResolvedSeats) -> TransformResult:
        model = _seat(seats, "model", ModelVariant)
        plan = _seat(seats, "quantization", QuantizationPlan)
        output_suffix = plan.id.rsplit("/", maxsplit=1)[-1].replace("@", "-")
        return selected_operation(
            context,
            TransformRequest(
                model=model,
                plan=plan,
                target=_seat(seats, "target", ExecutionTarget),
                output_id=f"{model.id}/quantized-{output_suffix}",
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
        "Transform the selected foundation model into an immutable derived variant.",
        required_artifact_roles=("model",),
    )


def standard_definitions() -> dict[str, JobDefinition]:
    """Return the immutable standard definition registry."""

    definitions = (
        supervised_data_prepare_definition(),
        preference_data_prepare_definition(),
        sft_definition(),
        dpo_definition(),
        grpo_definition(),
        sampo_definition(),
        distillation_definition(),
        serve_benchmark_definition(),
        serve_smoke_definition(),
        serve_generation_smoke_definition(),
        general_evaluation_definition(),
        remote_evaluation_definition(),
        managed_evaluation_definition(),
        managed_general_evaluation_definition(),
        model_transform_definition(),
    )
    registry = {definition.id: definition for definition in definitions}
    if len(registry) != len(definitions):
        raise AssertionError("standard job definition ids must be unique")
    return registry


def _evaluation_job(
    definition_id: str,
    kind: Literal["eval.general", "eval.domain"],
    operation: Callable[[RunContext, ResolvedSeats], object],
    description: str,
) -> JobDefinition:
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
        operation,
        description,
        required_artifact_roles=("evaluation",),
    )


def _evaluation_request(
    seats: ResolvedSeats,
    *,
    budget: EvaluationBudget,
    materialize_model: RunContext | None = None,
) -> EvaluateRequest:
    model = _seat(seats, "model", ModelVariant)
    inference = _seat(seats, "evaluation_inference", InferenceBinding)
    if materialize_model is not None and (
        "model_adapter" in materialize_model.input_artifacts
        or "model_weights" in materialize_model.input_artifacts
        or isinstance(model.artifact, (StoredArtifactRef, TrackioArtifactRef))
    ):
        input_name = (
            "model_adapter"
            if "model_adapter" in materialize_model.input_artifacts or model.form in {"adapter", "peft-adapter"}
            else "model_weights"
        )
        local = materialize_model.input_artifact(input_name)
        form = "adapter" if input_name == "model_adapter" else "full-finetuned"
        model = replace(model, artifact=local, form=form, revision=None, digest=local.digest)
        inference = replace(inference, model=model)
    plan = _seat(seats, "evaluation_plan", EvaluationPlan)
    environment = _seat(seats, "environment", EnvironmentBinding)
    if plan.environment(environment.id) != environment:
        raise ValueError("evaluation environment is not a cell in the selected plan")
    launch_request = ServeLaunchRequest(inference)
    context_window = inference.engine.get("max_model_len")
    if not isinstance(context_window, int):
        context_window = model.capabilities.native_context_window
    return EvaluateRequest(
        model=model,
        plan=plan,
        inference=inference,
        target=_seat(seats, "target", ExecutionTarget),
        endpoint=EvaluationEndpoint(launch_request.endpoint.base_url, launch_request.endpoint.model),
        environment_id=environment.id,
        context_window=context_window,
        budget=budget,
    )


def _validate_supervised_dataset_seats(seats: ResolvedSeats) -> None:
    for name in ("dataset", "validation_dataset"):
        plan = seats.get(name)
        if isinstance(plan, DatasetLoadPlan) and plan.kind != "supervised":
            raise ContractError(f"SFT seat {name!r} requires a supervised dataset plan")


def _validate_supervised_prepare_seats(seats: ResolvedSeats) -> None:
    plan = seats.get("dataset")
    if isinstance(plan, DatasetLoadPlan) and plan.kind != "supervised":
        raise ContractError("supervised data.prepare seat 'dataset' requires a supervised dataset plan")


def _validate_preference_dataset_seats(seats: ResolvedSeats) -> None:
    plan = seats.get("dataset")
    if isinstance(plan, DatasetLoadPlan) and plan.kind != "preference":
        raise ContractError("DPO seat 'dataset' requires a preference dataset plan")


def _validate_preference_prepare_seats(seats: ResolvedSeats) -> None:
    plan = seats.get("dataset")
    if isinstance(plan, DatasetLoadPlan) and plan.kind != "preference":
        raise ContractError("preference data.prepare seat 'dataset' requires a preference dataset plan")


def _validate_online_rl_batch_seats(seats: ResolvedSeats) -> None:
    settings = _seat(seats, "settings", GRPOSettings)
    training = _seat(seats, "training", TrainingBinding)
    expected_batch = settings.num_prompts_per_step * settings.num_generations
    global_batch = training.runtime.global_batch_size
    if isinstance(global_batch, int) and global_batch != expected_batch:
        raise ContractError("training global batch must equal prompt groups times generations")


def _recovery_checkpoint(context: RunContext) -> LocalArtifactRef | None:
    """Return explicitly materialized trainer state without treating it as a model selection."""

    return context.input_artifacts.get("recovery_checkpoint")


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
    "sampo_definition",
    "managed_evaluation_definition",
    "managed_general_evaluation_definition",
    "model_transform_definition",
    "preference_data_prepare_definition",
    "serve_benchmark_definition",
    "serve_smoke_definition",
    "serve_generation_smoke_definition",
    "sft_definition",
    "standard_definitions",
    "supervised_data_prepare_definition",
]
