"""Pure training-model resolution used before a backend imports ML libraries."""

from __future__ import annotations

from dataclasses import dataclass

from posttrain.common import InferenceBinding, ModelVariant, SettingOrigin

from .bindings import TrainingBinding, validate_parameter_update


@dataclass(frozen=True, slots=True)
class ResolvedTrainingModelConfiguration:
    """The model-facing subset of a training binding with explainable origins."""

    model: ModelVariant
    training: TrainingBinding
    reasoning_mode: str
    origins: tuple[SettingOrigin, ...]


def resolve_training_model_configuration(
    model: ModelVariant,
    training: TrainingBinding,
    *,
    inference: InferenceBinding | None = None,
    role: str = "policy",
) -> ResolvedTrainingModelConfiguration:
    """Validate training/model intent without selecting a backend implementation.

    A rollout inference binding is optional because SFT/DPO can be offline. If
    supplied, its reasoning mode must match the exact-token training renderer.
    """

    if not role.strip():
        raise ValueError("training model role cannot be empty")
    if training.renderer.model_family != model.family:
        raise ValueError(
            f"training renderer family {training.renderer.model_family!r} does not match model family {model.family!r}"
        )
    model.conversation.reasoning_mode(training.renderer.reasoning_mode)
    validate_parameter_update(model, training.update)
    if inference is not None:
        if inference.model != model:
            raise ValueError("training rollout inference must use the selected training model variant")
        if inference.resolved_reasoning_mode != training.renderer.reasoning_mode:
            raise ValueError(
                "training renderer reasoning mode must match rollout inference for exact-token policy training"
            )
    origins = [
        SettingOrigin(f"{role}.model", model.id, "explicit", model.id, model.revision),
        SettingOrigin(f"{role}.renderer", training.renderer.id, "explicit", training.id, training.revision),
        SettingOrigin(
            f"{role}.reasoning_mode",
            training.renderer.reasoning_mode,
            "explicit",
            training.id,
            training.revision,
        ),
        SettingOrigin(f"{role}.update.kind", training.update.kind, "explicit", training.id, training.revision),
    ]
    rank = getattr(training.update, "rank", None)
    if isinstance(rank, int):
        origins.append(SettingOrigin(f"{role}.update.rank", rank, "explicit", training.id, training.revision))
    return ResolvedTrainingModelConfiguration(model, training, training.renderer.reasoning_mode, tuple(origins))


__all__ = ["ResolvedTrainingModelConfiguration", "resolve_training_model_configuration"]
