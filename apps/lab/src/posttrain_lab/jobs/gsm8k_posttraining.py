"""Code-defined GSM8K post-training job composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from posttrain.common import ExecutionContext, Job, JobAction, ModelVariant
from posttrain.eval import EnvironmentProgram, EnvironmentSource, EvaluationProgram, SamplingPolicy
from posttrain.train import (
    DPORequest,
    GRPOProfile,
    GRPORequest,
    SFTRequest,
    TrainingResult,
    dpo,
    grpo,
    sft,
)

from ..environments import VERIFIERS_REVISION, GSM8KRewardBridge, load_gsm8k_rollout_dataset

JOB_ID = "posttraining/gsm8k"


@dataclass(frozen=True, slots=True)
class GSM8KGRPOJobRequest:
    model: ModelVariant
    profile: GRPOProfile
    task_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.model.profile.family != self.profile.model_family:
            raise ValueError("GRPO profile is incompatible with the model family")
        if not self.task_indices:
            raise ValueError("GRPO requires at least one task")


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


GSM8K_TRAINING_ROLLOUTS = EvaluationProgram(
    id="gsm8k-training-rollouts-v1",
    kind="domain",
    environments=(
        EnvironmentProgram(
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

GSM8K_LFM_TRAINING_ROLLOUTS = EvaluationProgram(
    id="gsm8k-lfm-training-rollouts-v1",
    kind="domain",
    environments=(
        EnvironmentProgram(
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


def gsm8k_posttraining_job(version: str) -> Job:
    return Job(id=JOB_ID, version=version, name="GSM8K post-training")


def sft_action(request: SFTRequest) -> JobAction:
    return JobAction(
        job_id=JOB_ID,
        id=f"train/sft/{request.model.profile.id}/{request.profile.id}",
        kind="supervised-finetuning",
    )


def dpo_action(request: DPORequest) -> JobAction:
    return JobAction(
        job_id=JOB_ID,
        id=f"train/dpo/{request.model.profile.id}/{request.profile.id}",
        kind="preference-optimization",
    )


def grpo_action(request: GSM8KGRPOJobRequest) -> JobAction:
    return JobAction(
        job_id=JOB_ID,
        id=f"train/grpo/{request.model.profile.id}/{request.profile.id}",
        kind="reinforcement-learning",
    )


def rollout_collection_action(model_profile_id: str) -> JobAction:
    return JobAction(
        job_id=JOB_ID,
        id=f"eval/domain/{model_profile_id}/gsm8k-train-candidates",
        kind="preference-rollout-collection",
    )


def run_sft(context: ExecutionContext, request: SFTRequest) -> TrainingResult:
    return sft(context, request)


def run_dpo(context: ExecutionContext, request: DPORequest) -> TrainingResult:
    return dpo(context, request)


def run_dpo_materialized(
    context: ExecutionContext,
    request: DPORequest,
    *,
    input_name: str = "model_adapter",
) -> TrainingResult:
    local_model = ModelVariant(
        profile=request.model.profile,
        artifact=context.input_artifact(input_name),
        format="peft-adapter",
    )
    return dpo(
        context,
        DPORequest(
            model=local_model,
            dataset=request.dataset,
            profile=request.profile,
            resume_from=request.resume_from,
        ),
    )


def run_grpo_materialized(
    context: ExecutionContext,
    request: GSM8KGRPOJobRequest,
    *,
    input_name: str = "model_adapter",
) -> TrainingResult:
    local_model = ModelVariant(
        profile=request.model.profile,
        artifact=context.input_artifact(input_name),
        format="peft-adapter",
    )
    dataset, tasks = load_gsm8k_rollout_dataset(request.task_indices)
    bridge = GSM8KRewardBridge(
        context=context,
        tasks=tasks,
        trace_path=context.workspace / "training" / "grpo" / "verifiers-traces.jsonl",
        model_profile_id=local_model.profile.id,
        training_profile_id=request.profile.id,
    )
    try:
        return grpo(
            context,
            GRPORequest(
                model=local_model,
                dataset=dataset,
                profile=request.profile,
                reward=bridge.reward_function(),
            ),
        )
    finally:
        bridge.publish_native_artifact()


def training_inputs(request: SFTRequest | DPORequest | GRPORequest) -> dict[str, str | int | float | bool]:
    """Stable run config; large examples and traces remain datasets/artifacts, not config."""

    values: dict[str, str | int | float | bool] = {
        "model_profile_id": request.model.profile.id,
        "input_model_format": request.model.format,
        "training_profile_id": request.profile.id,
        "renderer_profile_id": request.profile.renderer.id,
        "reasoning_mode": request.profile.renderer.reasoning_mode,
        "dataset_id": request.dataset.id,
        "dataset_revision": request.dataset.revision,
        "dataset_examples": len(request.dataset.examples),
        "max_steps": request.profile.loop.max_steps,
        "max_length": request.profile.loop.max_length,
        "learning_rate": request.profile.loop.learning_rate,
        "qlora_rank": request.profile.qlora.lora_rank,
        "qlora_quant_type": request.profile.qlora.quant_type,
    }
    if isinstance(request, DPORequest):
        values["dpo_beta"] = request.profile.beta
        values["dpo_loss_kernel"] = request.profile.loss_kernel
    if isinstance(request, GRPORequest):
        values["grpo_beta"] = request.profile.beta
        values["grpo_num_generations"] = request.profile.num_generations
        values["grpo_max_prompt_length"] = request.profile.max_prompt_length
        values["grpo_max_completion_length"] = request.profile.max_completion_length
    if isinstance(request.model, ModelVariant):
        values["base_model_revision"] = request.model.base_artifact.revision
    return values


def grpo_job_inputs(request: GSM8KGRPOJobRequest) -> dict[str, str | int | float | bool]:
    return {
        "model_profile_id": request.model.profile.id,
        "input_model_format": request.model.format,
        "base_model_revision": request.model.base_artifact.revision,
        "training_profile_id": request.profile.id,
        "renderer_profile_id": request.profile.renderer.id,
        "reasoning_mode": request.profile.renderer.reasoning_mode,
        "environment_id": "gsm8k-v1",
        "environment_revision": VERIFIERS_REVISION,
        "task_indices": ",".join(str(value) for value in request.task_indices),
        "max_steps": request.profile.loop.max_steps,
        "learning_rate": request.profile.loop.learning_rate,
        "qlora_rank": request.profile.qlora.lora_rank,
        "qlora_quant_type": request.profile.qlora.quant_type,
        "grpo_beta": request.profile.beta,
        "grpo_num_generations": request.profile.num_generations,
        "grpo_max_prompt_length": request.profile.max_prompt_length,
        "grpo_max_completion_length": request.profile.max_completion_length,
    }
