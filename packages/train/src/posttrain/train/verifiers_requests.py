"""Public request builders for environment-backed training."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from posttrain.common import InferenceBinding, ModelVariant

from .bindings import QuantizationPlan, TrainingBinding
from .integrations.verifiers import (
    VerifiersEnvironmentSelection,
    create_verifiers_training_bridge,
)
from .online_rl import PolicySampling, policy_sampling_from_binding, policy_sampling_from_environment
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

    sampling = validate_verifiers_policy_sampling(environment, inference, settings.max_completion_length)
    bridge = create_verifiers_training_bridge(
        environment,
        trace_path,
        run_id,
        sampling=sampling,
        purpose=settings.algorithm,
        tasks=tasks,
        model_identity=policy.trace_identity(),
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
) -> OnPolicyDistillationRequest:
    """Build the public distillation request directly from an environment binding."""

    sampling = validate_verifiers_policy_sampling(
        environment,
        rollout_inference,
        settings.max_completion_length,
        default_temperature=settings.temperature,
    )
    bridge = create_verifiers_training_bridge(
        environment,
        trace_path,
        run_id,
        sampling=sampling,
        purpose="distill",
        tasks=tasks,
        model_identity=student.trace_identity(),
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

    sampling = validate_verifiers_policy_sampling(environment, inference, settings.max_completion_length)
    bridge = create_verifiers_training_bridge(
        environment,
        trace_path,
        run_id,
        sampling=sampling,
        purpose="sampo",
        tasks=tasks,
        model_identity=policy.trace_identity(),
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


def validate_verifiers_policy_sampling(
    environment: VerifiersEnvironmentSelection,
    inference: InferenceBinding,
    max_tokens: int,
    *,
    default_temperature: float = 1.0,
) -> PolicySampling:
    environment_sampling = policy_sampling_from_environment(environment.sampling)
    inference_sampling = policy_sampling_from_binding(
        inference,
        max_tokens,
        default_temperature=default_temperature,
    )
    if environment_sampling != inference_sampling:
        raise ValueError(
            "environment and rollout inference must declare the same complete sampling policy: "
            f"environment={environment_sampling!r}, inference={inference_sampling!r}"
        )
    return inference_sampling


__all__ = [
    "build_verifiers_distillation_request",
    "build_verifiers_grpo_request",
    "build_verifiers_sampo_request",
    "validate_verifiers_policy_sampling",
]
