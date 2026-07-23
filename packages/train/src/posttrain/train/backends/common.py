"""Trainer-adapter result values shared by private backend implementations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..results import TrainingSummary


@dataclass(frozen=True, slots=True)
class BackendTrainingResult:
    summary: TrainingSummary
    model_dir: Path
    recovery_checkpoint: Path | None
    summary_file: Path


__all__ = ["BackendTrainingResult"]
