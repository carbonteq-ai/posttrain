"""Typed public training operation requests."""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol

from posttrain.common import LocalArtifactRef, ModelVariant, ProducedArtifact

from .data import CompletedRollout, PreferenceDataset, RolloutDataset, RolloutScore, SupervisedDataset
from .profiles import DPOProfile, GRPOProfile, SFTProfile


class OnlineRLEnvironment(Protocol):
    """Backend-neutral prompts, scoring, traces, and native evidence for online RL."""

    @property
    def dataset(self) -> RolloutDataset: ...

    def score(self, rollout: CompletedRollout) -> Awaitable[RolloutScore]: ...

    def finalize(self) -> tuple[ProducedArtifact, ...]: ...


@dataclass(frozen=True, slots=True)
class SFTRequest:
    model: ModelVariant
    dataset: SupervisedDataset
    profile: SFTProfile
    resume_from: LocalArtifactRef | None = None

    def __post_init__(self) -> None:
        if self.model.profile.family != self.profile.model_family:
            raise ValueError("SFT profile is incompatible with the model family")


@dataclass(frozen=True, slots=True)
class DPORequest:
    model: ModelVariant
    dataset: PreferenceDataset
    profile: DPOProfile
    resume_from: LocalArtifactRef | None = None

    def __post_init__(self) -> None:
        if self.model.profile.family != self.profile.model_family:
            raise ValueError("DPO profile is incompatible with the model family")


@dataclass(frozen=True, slots=True)
class GRPORequest:
    model: ModelVariant
    environment: OnlineRLEnvironment
    profile: GRPOProfile
    resume_from: LocalArtifactRef | None = None

    def __post_init__(self) -> None:
        if self.model.profile.family != self.profile.model_family:
            raise ValueError("GRPO profile is incompatible with the model family")
        if not self.environment.dataset.examples:
            raise ValueError("GRPO requires at least one rollout example")


__all__ = ["DPORequest", "GRPORequest", "OnlineRLEnvironment", "SFTRequest"]
