"""Typed requests accepted by serving operations."""

from __future__ import annotations

from dataclasses import dataclass

from posttrain.common import ModelProfile

from .benchmarks import BenchmarkCell
from .profiles import VllmServeProfile


@dataclass(frozen=True, slots=True)
class BenchmarkRequest:
    model: ModelProfile
    profile: VllmServeProfile
    cell: BenchmarkCell
    reasoning_mode: str | None = None

    def __post_init__(self) -> None:
        self.profile.validate_model(self.model)
        if self.cell.context_window > self.model.capabilities.native_context_window:
            raise ValueError("benchmark context exceeds the model's native context window")
        if self.cell.input_tokens + self.cell.output_tokens > self.cell.context_window:
            raise ValueError("input and output tokens exceed the benchmark context")
        if self.cell.required_variant is not None and self.profile.variant != self.cell.required_variant:
            raise ValueError(f"benchmark cell requires {self.cell.required_variant!r}, got {self.profile.variant!r}")
        if self.cell.concurrency > 4 and self.profile.engine.max_num_seqs == 4:
            raise ValueError("this serve profile supports at most four concurrent sequences")

    @property
    def resolved_reasoning_mode(self) -> str:
        return self.reasoning_mode or self.model.default_reasoning_mode
