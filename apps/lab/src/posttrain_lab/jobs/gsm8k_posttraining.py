"""Code-defined GSM8K post-training job composition."""

from __future__ import annotations

from posttrain.common import ExecutionContext, Job, JobAction, ModelVariant
from posttrain.train import DPORequest, SFTRequest, TrainingResult, dpo, sft

JOB_ID = "posttraining/gsm8k"


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


def run_sft(context: ExecutionContext, request: SFTRequest) -> TrainingResult:
    return sft(context, request)


def run_dpo(context: ExecutionContext, request: DPORequest) -> TrainingResult:
    return dpo(context, request)


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
    if isinstance(request.model, ModelVariant):
        values["base_model_revision"] = request.model.base_artifact.revision
    return values
