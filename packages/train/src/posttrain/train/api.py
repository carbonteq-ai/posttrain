"""Public observer-neutral training operations."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

from posttrain.common import ExecutionContext, LocalArtifactRef, ModelVariant, ProducedArtifact

from .backends.trl import run_dpo, run_grpo, run_sft
from .backends.trl.common import BackendTrainingResult
from .requests import DPORequest, GRPORequest, SFTRequest
from .results import TrainingResult


class TrainingRequest(Protocol):
    model: ModelVariant


type SFTBackend = Callable[[ExecutionContext, SFTRequest, Path], BackendTrainingResult]
type DPOBackend = Callable[[ExecutionContext, DPORequest, Path], BackendTrainingResult]
type GRPOBackend = Callable[[ExecutionContext, GRPORequest, Path], BackendTrainingResult]


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with child.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _finish(
    context: ExecutionContext,
    request: SFTRequest | DPORequest | GRPORequest,
    technique: Literal["sft", "dpo", "grpo"],
    backend: BackendTrainingResult,
) -> TrainingResult:
    attributes = {
        "technique": technique,
        "model_profile_id": request.model.profile.id,
        "source_model_format": request.model.format,
        "training_profile_id": request.profile.id,
        "dataset_id": request.dataset.id,
        "dataset_revision": request.dataset.revision,
    }
    adapter_ref = LocalArtifactRef(backend.adapter_dir.resolve(), _digest(backend.adapter_dir))
    model_artifact = ProducedArtifact(
        name=f"training/{request.model.profile.id}/{technique}/adapter",
        kind="model-adapter",
        reference=adapter_ref,
        metadata=attributes,
    )
    recovery_artifact = None
    if backend.recovery_checkpoint is not None:
        recovery_ref = LocalArtifactRef(
            backend.recovery_checkpoint.resolve(),
            _digest(backend.recovery_checkpoint),
        )
        recovery_artifact = ProducedArtifact(
            name=f"training/{request.model.profile.id}/{technique}/recovery-checkpoint",
            kind="training-checkpoint",
            reference=recovery_ref,
            metadata={**attributes, "global_step": backend.summary.global_step},
        )
    summary_ref = LocalArtifactRef(backend.summary_file.resolve(), _digest(backend.summary_file))
    native_artifact = ProducedArtifact(
        name=f"training/{request.model.profile.id}/{technique}/summary",
        kind="training-summary",
        reference=summary_ref,
        metadata=attributes,
    )
    context.artifact(model_artifact)
    if recovery_artifact is not None:
        context.artifact(recovery_artifact)
    context.artifact(native_artifact)
    context.metrics(
        {
            "train/global_step": backend.summary.global_step,
            "train/final_loss": backend.summary.train_loss,
            "train/runtime_seconds": backend.summary.runtime_seconds,
            "train/samples_per_second": backend.summary.samples_per_second,
            "train/steps_per_second": backend.summary.steps_per_second,
        },
        attributes=attributes,
    )
    context.event("training_completed", attributes)
    output_model = ModelVariant(request.model.profile, adapter_ref, "peft-adapter")
    return TrainingResult(
        technique,
        request.model,
        output_model,
        backend.summary,
        model_artifact,
        recovery_artifact,
        native_artifact,
    )


def sft(
    context: ExecutionContext,
    request: SFTRequest,
    *,
    runner: SFTBackend = run_sft,
) -> TrainingResult:
    attributes = {
        "technique": "sft",
        "model_profile_id": request.model.profile.id,
        "training_profile_id": request.profile.id,
        "dataset_id": request.dataset.id,
    }
    context.event("training_started", attributes)
    output_dir = context.workspace / "training" / "sft" / "trainer"
    output_dir.mkdir(parents=True, exist_ok=False)
    return _finish(context, request, "sft", runner(context, request, output_dir))


def dpo(
    context: ExecutionContext,
    request: DPORequest,
    *,
    runner: DPOBackend = run_dpo,
) -> TrainingResult:
    attributes = {
        "technique": "dpo",
        "model_profile_id": request.model.profile.id,
        "training_profile_id": request.profile.id,
        "dataset_id": request.dataset.id,
    }
    context.event("training_started", attributes)
    output_dir = context.workspace / "training" / "dpo" / "trainer"
    output_dir.mkdir(parents=True, exist_ok=False)
    return _finish(context, request, "dpo", runner(context, request, output_dir))


def grpo(
    context: ExecutionContext,
    request: GRPORequest,
    *,
    runner: GRPOBackend = run_grpo,
) -> TrainingResult:
    attributes = {
        "technique": "grpo",
        "model_profile_id": request.model.profile.id,
        "training_profile_id": request.profile.id,
        "dataset_id": request.dataset.id,
    }
    context.event("training_started", attributes)
    output_dir = context.workspace / "training" / "grpo" / "trainer"
    output_dir.mkdir(parents=True, exist_ok=False)
    return _finish(context, request, "grpo", runner(context, request, output_dir))


__all__ = ["DPOBackend", "GRPOBackend", "SFTBackend", "dpo", "grpo", "sft"]
