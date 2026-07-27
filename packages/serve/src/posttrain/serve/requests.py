"""Typed requests accepted by serving operations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from posttrain.common import ExecutionTarget, InferenceBinding, Workload


@dataclass(frozen=True, slots=True)
class ServeBenchmarkRequest:
    """Canonical benchmark seats resolved from the composed catalog."""

    inference: InferenceBinding
    workload: Workload
    target: ExecutionTarget | None = None

    def __post_init__(self) -> None:
        if "screen" not in self.inference.purpose:
            raise ValueError("serve.benchmark requires an inference binding with screen purpose")
        if self.target is not None and self.target != self.inference.target:
            raise ValueError("benchmark target conflicts with the inference binding target")
        required = {"suite_id", "shape_id", "context_window", "output_tokens"}
        missing = required.difference(self.workload.requests)
        if missing:
            raise ValueError(f"benchmark workload is missing request fields: {', '.join(sorted(missing))}")
        context_window = self._request_int("context_window")
        output_tokens = self._request_int("output_tokens")
        if context_window > self.inference.model.capabilities.native_context_window:
            raise ValueError("benchmark context exceeds the model's native context window")
        cohort = self.workload.requests.get("cohort", "controlled")
        if cohort == "controlled":
            input_tokens = self._request_int("input_tokens")
            if input_tokens + output_tokens > context_window:
                raise ValueError("input and output tokens exceed the benchmark context")
        elif cohort == "representative":
            corpus = self.workload.requests.get("corpus")
            if not isinstance(corpus, Mapping):
                raise ValueError("representative benchmark workload requires a corpus selection")
            for field in ("id", "revision", "digest"):
                value = corpus.get(field)
                if not isinstance(value, str) or not value:
                    raise ValueError(f"representative benchmark corpus {field} must be a non-empty string")
            seed = self.workload.requests.get("selection_seed")
            if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                raise ValueError("representative benchmark selection_seed must be a non-negative integer")
            if output_tokens >= context_window:
                raise ValueError("benchmark output tokens must leave room for representative prompts")
        else:
            raise ValueError(f"unsupported benchmark cohort: {cohort!r}")

    @property
    def resolved_target(self) -> ExecutionTarget:
        return self.target or self.inference.target

    def _request_int(self, name: str) -> int:
        value = self.workload.requests[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"benchmark workload {name} must be a positive integer")
        return value
