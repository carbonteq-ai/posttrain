"""Typed public training operation requests."""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Protocol

from posttrain.common import LocalArtifactRef, ModelVariant

from .data import PreferenceDataset, RolloutDataset, SupervisedDataset
from .profiles import DPOProfile, GRPOProfile, SFTProfile


class RewardFunction(Protocol):
    __name__: str

    def __call__(self, **kwargs: Any) -> list[float | None] | Awaitable[list[float | None]]: ...


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
    dataset: RolloutDataset
    profile: GRPOProfile
    reward: RewardFunction
    resume_from: LocalArtifactRef | None = None

    def __post_init__(self) -> None:
        if self.model.profile.family != self.profile.model_family:
            raise ValueError("GRPO profile is incompatible with the model family")
        if not callable(self.reward):
            raise ValueError("GRPO requires a callable reward")


__all__ = ["DPORequest", "GRPORequest", "RewardFunction", "SFTRequest"]
