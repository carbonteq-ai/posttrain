"""Typed training outcomes and direct observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from posttrain.common import ModelVariant, ProducedArtifact


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    global_step: int
    train_loss: float
    runtime_seconds: float
    samples_per_second: float
    steps_per_second: float


@dataclass(frozen=True, slots=True)
class TeacherScoringSummary:
    teacher: ModelVariant
    mode: Literal["external-exact-token"]
    temperature: float
    top_k: int


@dataclass(frozen=True, slots=True)
class TrainingResult:
    technique: Literal["sft", "dpo", "grpo", "distill"]
    source_model: ModelVariant
    model: ModelVariant
    summary: TrainingSummary
    model_artifact: ProducedArtifact
    recovery_artifact: ProducedArtifact | None
    native_artifact: ProducedArtifact
    teacher_scoring: TeacherScoringSummary | None = None


__all__ = ["TeacherScoringSummary", "TrainingResult", "TrainingSummary"]
