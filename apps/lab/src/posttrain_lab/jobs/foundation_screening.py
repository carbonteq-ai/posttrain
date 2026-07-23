"""Reusable foundation-model screening operations."""

from __future__ import annotations

from posttrain.common import RunContext
from posttrain.eval import EvaluateRequest, EvaluationResult, evaluate
from posttrain.serve import (
    GenerationRequest,
    GenerationResult,
    ServeBenchmarkRequest,
    ServeLaunchRequest,
    benchmark,
    generate,
    launch,
    probe,
)
from posttrain.serve.results import BenchmarkResult


def run_screen_benchmark(context: RunContext, request: ServeBenchmarkRequest) -> BenchmarkResult:
    return benchmark(context, request)


def run_online_smoke(context: RunContext, request: ServeLaunchRequest) -> GenerationResult:
    model = request.inference.model
    max_tokens = request.inference.sampling.get("max_tokens", 128)
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
        raise ValueError("smoke inference max_tokens must be an integer")
    with launch(context, request) as endpoint:
        health = probe(context, endpoint)
        if not health.model_available:
            raise RuntimeError(f"launched endpoint does not expose {endpoint.model!r}")
        result = generate(
            context,
            GenerationRequest(
                endpoint=endpoint,
                messages=({"role": "user", "content": "What is 2 + 2? Answer concisely."},),
                max_tokens=max_tokens,
            ),
            model,
        )
        if not result.content.strip():
            raise RuntimeError(f"online smoke produced no final answer (finish_reason={result.finish_reason!r})")
        return result


def run_managed_evaluation(
    context: RunContext,
    launch_request: ServeLaunchRequest,
    request: EvaluateRequest,
) -> EvaluationResult:
    """Compose serving and evaluation without either capability package importing the other."""

    with launch(context, launch_request) as endpoint:
        health = probe(context, endpoint)
        if not health.model_available:
            raise RuntimeError(f"launched endpoint does not expose {endpoint.model!r}")
        return evaluate(context, request)


__all__ = ["run_managed_evaluation", "run_online_smoke", "run_screen_benchmark"]
