"""Code-defined, backend-neutral serving workload matrix."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WorkloadShape:
    id: str
    output_tokens: int
    input_tokens: int | None = None
    input_fraction: float | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("workload shape id cannot be empty")
        if self.output_tokens < 1:
            raise ValueError("output_tokens must be positive")
        if (self.input_tokens is None) == (self.input_fraction is None):
            raise ValueError("exactly one input sizing strategy is required")
        if self.input_tokens is not None and self.input_tokens < 1:
            raise ValueError("input_tokens must be positive")
        if self.input_fraction is not None and not 0 < self.input_fraction < 1:
            raise ValueError("input_fraction must be between zero and one")

    def input_for(self, context_window: int) -> int:
        if self.input_tokens is not None:
            return self.input_tokens
        assert self.input_fraction is not None
        return int(context_window * self.input_fraction)


@dataclass(frozen=True, slots=True)
class BenchmarkCell:
    suite_id: str
    shape_id: str
    context_window: int
    concurrency: int
    input_tokens: int
    output_tokens: int
    warmup_iterations: int
    iterations: int
    required_variant: str | None = None

    @property
    def id(self) -> str:
        return f"{self.shape_id}-ctx{self.context_window}-c{self.concurrency}"


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    id: str
    contexts: tuple[int, ...]
    concurrencies: tuple[int, ...]
    shapes: tuple[WorkloadShape, ...]
    warmup_iterations: int = 1
    iterations: int = 3
    turboquant_context_threshold: int = 32_768

    def cells(self, *, max_concurrency: int | None = None) -> tuple[BenchmarkCell, ...]:
        cells: list[BenchmarkCell] = []
        for context in self.contexts:
            for shape in self.shapes:
                input_tokens = shape.input_for(context)
                if input_tokens + shape.output_tokens > context:
                    continue
                for concurrency in self.concurrencies:
                    if max_concurrency is not None and concurrency > max_concurrency:
                        continue
                    cells.append(
                        BenchmarkCell(
                            suite_id=self.id,
                            shape_id=shape.id,
                            context_window=context,
                            concurrency=concurrency,
                            input_tokens=input_tokens,
                            output_tokens=shape.output_tokens,
                            warmup_iterations=self.warmup_iterations,
                            iterations=self.iterations,
                            required_variant=("turboquant" if context >= self.turboquant_context_threshold else None),
                        )
                    )
        return tuple(cells)


CORE_INFERENCE_V1 = BenchmarkSuite(
    id="core-inference-v1",
    contexts=(1_024, 2_048, 4_096, 8_192, 16_384, 32_768),
    concurrencies=(1, 2, 4, 8),
    shapes=(
        WorkloadShape("short-interactive", input_tokens=128, output_tokens=128),
        WorkloadShape("decode-heavy", input_tokens=128, output_tokens=512),
        WorkloadShape("balanced", input_fraction=0.45, output_tokens=512),
        WorkloadShape("prefill-heavy", input_fraction=0.75, output_tokens=128),
    ),
)
