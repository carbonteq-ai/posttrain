"""Typed public training operation requests."""

from __future__ import annotations

from dataclasses import dataclass

from posttrain.common import LocalArtifactRef, ModelVariant

from .data import PreferenceDataset, SupervisedDataset
from .profiles import DPOProfile, SFTProfile


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


__all__ = ["DPORequest", "SFTRequest"]
