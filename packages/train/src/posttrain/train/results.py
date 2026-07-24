"""Typed training outcomes and direct observations."""

from __future__ import annotations

import math
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

    def __post_init__(self) -> None:
        if self.global_step < 1:
            raise ValueError("completed training summaries require a positive global step")
        values = {
            "train_loss": self.train_loss,
            "runtime_seconds": self.runtime_seconds,
            "samples_per_second": self.samples_per_second,
            "steps_per_second": self.steps_per_second,
        }
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("completed training summary values must be finite")
        if any(values[name] < 0 for name in ("runtime_seconds", "samples_per_second", "steps_per_second")):
            raise ValueError("completed training runtime and rates cannot be negative")


@dataclass(frozen=True, slots=True)
class TeacherScoringSummary:
    teacher: ModelVariant
    mode: Literal["exact-token"]
    temperature: float
    top_k: int
    inference_binding_id: str
    inference_binding_revision: str
    backend: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0 or self.top_k < 1:
            raise ValueError("teacher scoring requires a positive finite temperature and top-k")
        if not self.inference_binding_id.strip() or not self.inference_binding_revision.strip():
            raise ValueError("teacher scoring requires an inference binding identity")
        if "@" not in self.backend:
            raise ValueError("teacher scoring backend must include a product and version")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    technique: Literal["sft", "dpo", "grpo", "dapo", "sampo", "distill"]
    source_model: ModelVariant
    model: ModelVariant
    summary: TrainingSummary
    model_artifact: ProducedArtifact
    recovery_artifact: ProducedArtifact | None
    native_artifact: ProducedArtifact
    teacher_scoring: TeacherScoringSummary | None = None


__all__ = ["TeacherScoringSummary", "TrainingResult", "TrainingSummary"]
