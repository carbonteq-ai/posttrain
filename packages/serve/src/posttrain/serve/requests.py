"""Typed requests accepted by serving operations."""

from __future__ import annotations

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
        if len(self.workload.concurrency) != 1:
            raise ValueError("one serve.benchmark run requires exactly one workload concurrency")
        required = {"suite_id", "shape_id", "context_window", "input_tokens", "output_tokens"}
        missing = required.difference(self.workload.requests)
        if missing:
            raise ValueError(f"benchmark workload is missing request fields: {', '.join(sorted(missing))}")
        context_window = self._request_int("context_window")
        input_tokens = self._request_int("input_tokens")
        output_tokens = self._request_int("output_tokens")
        if context_window > self.inference.model.capabilities.native_context_window:
            raise ValueError("benchmark context exceeds the model's native context window")
        if input_tokens + output_tokens > context_window:
            raise ValueError("input and output tokens exceed the benchmark context")

    @property
    def resolved_target(self) -> ExecutionTarget:
        return self.target or self.inference.target

    def _request_int(self, name: str) -> int:
        value = self.workload.requests[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"benchmark workload {name} must be a positive integer")
        return value
