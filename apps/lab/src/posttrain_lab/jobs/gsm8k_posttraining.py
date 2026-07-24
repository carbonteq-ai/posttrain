"""Code-defined GSM8K post-training job composition."""

from __future__ import annotations

from typing import Any

from posttrain.common import InferenceBinding, ModelVariant, RunContext
from posttrain.eval import EnvironmentBinding, EnvironmentSource, EvaluationPlan, SamplingPolicy
from posttrain.train import (
    DPORequest,
    GRPORequest,
    LoRAUpdate,
    OnPolicyDistillationRequest,
    QLoRAUpdate,
    SFTRequest,
    TrainingResult,
    distill,
    dpo,
    parameter_update_digest,
    sft,
)

from ..environments import VERIFIERS_REVISION

JOB_ID = "posttraining/gsm8k"


GSM8KDistillationJobRequest = OnPolicyDistillationRequest


def _training_environment() -> object:
    try:
        from verifiers.v1.env import EnvConfig
    except ImportError as error:
        raise RuntimeError("install posttrain-lab with the gpu-posttrain extra") from error
    config: dict[str, Any] = {
        "taskset": {"id": "gsm8k-v1", "split": "train"},
        "harness": {"id": "null", "runtime": {"type": "subprocess"}},
        "timeout": {"setup": 120, "rollout": 180, "finalize": 60, "scoring": 120},
    }
    return EnvConfig.model_validate(config)


GSM8K_TRAINING_ROLLOUTS = EvaluationPlan(
    id="gsm8k-training-rollouts-v1",
    kind="domain",
    environments=(
        EnvironmentBinding(
            id="gsm8k-train-candidates",
            category="math-reasoning",
            source=EnvironmentSource(
                package="gsm8k-v1",
                repository="https://github.com/PrimeIntellect-ai/verifiers",
                revision=VERIFIERS_REVISION,
                subdirectory="environments/gsm8k_v1",
            ),
            factory=_training_environment,
            sampling=SamplingPolicy(max_tokens=1_024, temperature=0.8, top_p=0.95),
            num_tasks=4,
            num_rollouts=2,
            max_concurrent=1,
        ),
    ),
)

GSM8K_LFM_TRAINING_ROLLOUTS = EvaluationPlan(
    id="gsm8k-lfm-training-rollouts-v1",
    kind="domain",
    environments=(
        EnvironmentBinding(
            id="gsm8k-train-candidates",
            category="math-reasoning",
            source=GSM8K_TRAINING_ROLLOUTS.environments[0].source,
            factory=_training_environment,
            sampling=SamplingPolicy(max_tokens=4_096, temperature=0.8, top_p=0.95),
            num_tasks=4,
            num_rollouts=2,
            max_concurrent=1,
        ),
    ),
)


def run_sft(context: RunContext, request: SFTRequest) -> TrainingResult:
    return sft(context, request)


def run_dpo(context: RunContext, request: DPORequest) -> TrainingResult:
    return dpo(context, request)


def run_dpo_materialized(
    context: RunContext,
    request: DPORequest,
    *,
    input_name: str = "model_adapter",
) -> TrainingResult:
    local_model = ModelVariant(
        id=request.model.id,
        artifact=context.input_artifact(input_name),
        form="peft-adapter",
        weight_precision=request.model.weight_precision,
        family=request.model.family,
        parameters=request.model.parameters,
        instruction_tuned=request.model.instruction_tuned,
        renderer=request.model.renderer,
        capabilities=request.model.capabilities,
        base=request.model.base,
        tokenizer_fingerprint=request.model.tokenizer_fingerprint,
        parent=request.model.parent,
        provenance=request.model.provenance,
    )
    return dpo(
        context,
        DPORequest(
            model=local_model,
            data=request.data,
            settings=request.settings,
            training=request.training,
            resume_from=request.resume_from,
        ),
    )


def run_distillation(context: RunContext, request: OnPolicyDistillationRequest) -> TrainingResult:
    return distill(context, request)


def training_inputs(request: SFTRequest | DPORequest | GRPORequest) -> dict[str, str | int | float | bool]:
    """Stable run config; large examples and traces remain datasets/artifacts, not config."""

    if isinstance(request, GRPORequest):
        model = request.policy
        dataset = None
        dataset_examples = None
    else:
        model = request.model
        dataset = request.data.descriptor
        dataset_examples = dataset.num_examples
    values: dict[str, str | int | float | bool] = {
        "model_variant_id": model.id,
        "input_model_form": model.form,
        "training_settings_id": request.settings.id,
        "training_binding_id": request.training.id,
        "training_target_id": request.training.target.id,
        "training_renderer_id": request.training.renderer.id,
        "reasoning_mode": request.training.renderer.reasoning_mode,
        "parameter_update_kind": request.training.update.kind,
        "parameter_update_digest": parameter_update_digest(request.training.update),
        "max_steps": request.settings.loop.max_steps,
        "max_length": request.settings.loop.max_length,
        "learning_rate": request.settings.loop.learning_rate,
    }
    if dataset is not None:
        values["dataset_id"] = dataset.id
        values["dataset_revision"] = dataset.revision
    if isinstance(request.training.update, LoRAUpdate | QLoRAUpdate):
        values["peft_rank"] = request.training.update.rank
        values["peft_alpha"] = request.training.update.alpha
        values["peft_target_modules"] = request.training.update.target_modules
    if isinstance(request.training.update, QLoRAUpdate):
        values["qlora_quant_type"] = request.training.update.quant_type
        values["qlora_compute_dtype"] = request.training.update.compute_dtype
        values["qlora_double_quant"] = request.training.update.double_quant
    if dataset_examples is not None:
        values["dataset_examples"] = dataset_examples
    if isinstance(request, DPORequest):
        values["dpo_beta"] = request.settings.beta
        values["dpo_loss_kernel"] = request.settings.loss_kernel
    if isinstance(request, GRPORequest):
        values["environment_id"] = request.environment.id
        values["environment_revision"] = request.environment.revision
        values["online_rl_algorithm"] = request.settings.algorithm
        values["clip_epsilon_low"] = request.settings.clip_epsilon_low
        values["clip_epsilon_high"] = request.settings.resolved_clip_epsilon_high
        values["mask_truncated_completions"] = request.settings.mask_truncated_completions
        values["overlong_penalty_factor"] = request.settings.overlong_penalty_factor
        if request.settings.overlong_buffer_tokens is not None:
            values["overlong_buffer_tokens"] = request.settings.overlong_buffer_tokens
        values["grpo_beta"] = request.settings.beta
        values["grpo_num_generations"] = request.settings.num_generations
        values["grpo_max_prompt_length"] = request.settings.max_prompt_length
        values["grpo_max_completion_length"] = request.settings.max_completion_length
        values["grpo_temperature"] = _float_setting(request.inference, "temperature", 1.0)
        values["grpo_top_p"] = _float_setting(request.inference, "top_p", 1.0)
        values["rollout_binding_id"] = request.inference.id
        values["rollout_target_id"] = request.inference.target.id
        if request.quantization is not None:
            values["quantization_plan_id"] = request.quantization.id
            values["quantization_recipe_digest"] = request.quantization.recipe_digest
    else:
        values["dataset_schema_version"] = request.data.descriptor.schema_version
    if isinstance(request, SFTRequest) and request.validation_data is not None:
        validation = request.validation_data.descriptor
        schedule = request.settings.validation
        assert schedule is not None
        values["validation_dataset_id"] = validation.id
        values["validation_dataset_revision"] = validation.revision
        if validation.num_examples is not None:
            values["validation_dataset_examples"] = validation.num_examples
        values["validation_steps"] = schedule.steps
        values["validation_on_start"] = schedule.on_start
        values["validation_at_end"] = schedule.at_end
    values["base_model_revision"] = model.base.revision
    return values


def _number(value: object, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def _float_setting(binding: InferenceBinding, key: str, default: float) -> float:
    return _number(binding.sampling.get(key), default)
