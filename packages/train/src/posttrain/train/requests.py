"""Typed public training operation requests."""

from __future__ import annotations

from dataclasses import dataclass

from posttrain.common import LocalArtifactRef, ModelVariant

from .data import PreferenceDataset, SupervisedDataset
from .online_rl import OnlineRLBridge
from .profiles import DPOProfile, GRPOProfile, SFTProfile


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
    bridge: OnlineRLBridge
    profile: GRPOProfile
    resume_from: LocalArtifactRef | None = None

    def __post_init__(self) -> None:
        if self.model.profile.family != self.profile.model_family:
            raise ValueError("GRPO profile is incompatible with the model family")
        if not self.bridge.dataset.examples:
            raise ValueError("GRPO requires at least one rollout example")


__all__ = ["DPORequest", "GRPORequest", "SFTRequest"]
