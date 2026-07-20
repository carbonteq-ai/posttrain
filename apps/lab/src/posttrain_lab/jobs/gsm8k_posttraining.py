"""Code-defined GSM8K post-training job composition."""

from __future__ import annotations

from typing import Any

from posttrain.common import ExecutionContext, Job, JobAction, ModelVariant
from posttrain.eval import EnvironmentProgram, EnvironmentSource, EvaluationProgram, SamplingPolicy
from posttrain.train import DPORequest, SFTRequest, TrainingResult, dpo, sft

JOB_ID = "posttraining/gsm8k"
VERIFIERS_REVISION = "284a868d6a9022109b749710672a0460e8a996d4"


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


def training_inputs(request: SFTRequest | DPORequest) -> dict[str, str | int | float | bool]:
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
    if isinstance(request.model, ModelVariant):
        values["base_model_revision"] = request.model.base_artifact.revision
    return values
