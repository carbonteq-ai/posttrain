"""Public request builders for environment-backed training."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from posttrain.common import InferenceBinding, LocalArtifactRef, ModelVariant

from .bindings import QuantizationPlan, TrainingBinding
from .integrations.verifiers import (
    VerifiersEnvironmentSelection,
    create_verifiers_training_bridge,
)
from .profiles import GRPOSettings, OnPolicyDistillationSettings, SAMPOSettings
from .requests import GRPORequest, OnPolicyDistillationRequest, SAMPORequest


def build_verifiers_grpo_request(
    *,
    policy: ModelVariant,
    environment: VerifiersEnvironmentSelection,
    settings: GRPOSettings,
    training: TrainingBinding,
    inference: InferenceBinding,
    trace_path: Path,
    run_id: str,
    tasks: Mapping[int, Any] | None = None,
    quantization: QuantizationPlan | None = None,
    reference: ModelVariant | None = None,
) -> GRPORequest:
    """Build the public GRPO request directly from an environment binding."""

    bridge = create_verifiers_training_bridge(
        environment,
        trace_path,
        run_id,
        max_tokens=settings.max_completion_length,
        temperature=_sampling_number(inference, "temperature", 1.0),
        top_p=_sampling_number(inference, "top_p", 1.0),
        purpose=settings.algorithm,
        tasks=tasks,
    )
    return GRPORequest(
        policy=policy,
        bridge=bridge,
        settings=settings,
        environment=environment,
        training=training,
        inference=inference,
        quantization=quantization,
        reference=reference,
    )


def build_verifiers_distillation_request(
    *,
    student: ModelVariant,
    teacher: ModelVariant,
    environment: VerifiersEnvironmentSelection,
    settings: OnPolicyDistillationSettings,
    training: TrainingBinding,
    rollout_inference: InferenceBinding,
    teacher_inference: InferenceBinding,
    trace_path: Path,
    run_id: str,
    tasks: Mapping[int, Any] | None = None,
    quantization: QuantizationPlan | None = None,
    resume_from: LocalArtifactRef | None = None,
) -> OnPolicyDistillationRequest:
    """Build the public distillation request directly from an environment binding."""

    bridge = create_verifiers_training_bridge(
        environment,
        trace_path,
        run_id,
        max_tokens=settings.max_completion_length,
        temperature=_sampling_number(rollout_inference, "temperature", settings.temperature),
        top_p=_sampling_number(rollout_inference, "top_p", 1.0),
        purpose="distill",
        tasks=tasks,
    )
    return OnPolicyDistillationRequest(
        student=student,
        teacher=teacher,
        bridge=bridge,
        settings=settings,
        environment=environment,
        training=training,
        rollout_inference=rollout_inference,
        teacher_inference=teacher_inference,
        quantization=quantization,
        resume_from=resume_from,
    )


def build_verifiers_sampo_request(
    *,
    policy: ModelVariant,
    environment: VerifiersEnvironmentSelection,
    settings: SAMPOSettings,
    training: TrainingBinding,
    inference: InferenceBinding,
    trace_path: Path,
    run_id: str,
    tasks: Mapping[int, Any] | None = None,
    quantization: QuantizationPlan | None = None,
    reference: ModelVariant | None = None,
) -> SAMPORequest:
    """Build a SAMPO request from a multi-turn Verifiers environment."""

    bridge = create_verifiers_training_bridge(
        environment,
        trace_path,
        run_id,
        max_tokens=settings.max_completion_length,
        temperature=_sampling_number(inference, "temperature", 1.0),
        top_p=_sampling_number(inference, "top_p", 1.0),
        purpose="sampo",
        tasks=tasks,
    )
    return SAMPORequest(
        policy=policy,
        bridge=bridge,
        settings=settings,
        environment=environment,
        training=training,
        inference=inference,
        quantization=quantization,
        reference=reference,
    )


def _sampling_number(binding: InferenceBinding, key: str, default: float) -> float:
    value = binding.sampling.get(key)
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else default


__all__ = ["build_verifiers_distillation_request", "build_verifiers_grpo_request", "build_verifiers_sampo_request"]
