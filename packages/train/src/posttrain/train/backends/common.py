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

    def validate(self, workspace: Path) -> None:
        """Require completed backend outputs to exist inside the run workspace."""

        if not workspace.is_absolute():
            raise ValueError("training workspace must be absolute")
        root = workspace.resolve()
        paths = {
            "model directory": self.model_dir,
            "summary file": self.summary_file,
        }
        if self.recovery_checkpoint is not None:
            paths["recovery checkpoint"] = self.recovery_checkpoint
        for label, path in paths.items():
            if not path.is_absolute():
                raise ValueError(f"training backend {label} must be absolute")
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"training backend {label} must remain inside the run workspace")
        if not self.model_dir.is_dir():
            raise FileNotFoundError(self.model_dir)
        if not self.summary_file.is_file():
            raise FileNotFoundError(self.summary_file)
        if self.recovery_checkpoint is not None and not self.recovery_checkpoint.exists():
            raise FileNotFoundError(self.recovery_checkpoint)


__all__ = ["BackendTrainingResult"]
