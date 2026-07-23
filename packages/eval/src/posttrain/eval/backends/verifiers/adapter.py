"""Compose the pinned Verifiers v1 environment, client, runner, and trace schema."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from posttrain.common import JsonValue, RunContext, TraceObservation

from ...requests import EvaluateRequest
from .synchronization import TraceSyncStats, VerifiersTraceSynchronizer

type EvaluationContext = RunContext


@dataclass(frozen=True, slots=True)
class VerifiersRunResult:
    trace_ids: tuple[str, ...]
    synchronization: TraceSyncStats


class _NativeEnvConfig(Protocol):
    max_total_tokens: int | None

    def model_dump(self, *, mode: str) -> dict[str, Any]: ...


def _imports() -> tuple[type[Any], type[Any], Any]:
    try:
        from verifiers.v1.cli.eval.runner import run_eval  # pyright: ignore[reportMissingImports]
        from verifiers.v1.configs.eval import EvalConfig  # pyright: ignore[reportMissingImports]
        from verifiers.v1.env import EnvConfig, Environment  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("install posttrain-eval with the verifiers extra") from error
    return EvalConfig, Environment, (EnvConfig, run_eval)


def _native_sampling(request: EvaluateRequest) -> dict[str, JsonValue]:
    policy = request.environment.sampling
    values: dict[str, JsonValue] = {
        "temperature": policy.temperature,
        "max_tokens": policy.max_tokens,
    }
    if policy.top_p is not None:
        values["top_p"] = policy.top_p
    if policy.reasoning_effort is not None:
        values["reasoning_effort"] = policy.reasoning_effort
    template_kwargs = request.model.conversation.reasoning_mode(request.resolved_reasoning_mode).kwargs()
    if template_kwargs:
        values["chat_template_kwargs"] = cast(dict[str, JsonValue], template_kwargs)
    return values


def _build_native(request: EvaluateRequest, output_dir: Path) -> tuple[Any, Any, Any]:
    EvalConfig, Environment, (EnvConfig, run_eval) = _imports()
    base = request.environment.factory()
    if not isinstance(base, EnvConfig):
        raise TypeError("environment factories must return verifiers.v1.EnvConfig")
    base = cast(_NativeEnvConfig, base)
    num_tasks, num_rollouts, max_concurrent = request.resolved_budget
    raw = base.model_dump(mode="python")
    raw.update(
        {
            "model": request.endpoint.served_model,
            "client": {
                "type": "eval",
                "base_url": request.endpoint.base_url,
                "api_key_var": request.endpoint.api_key_var,
            },
            "sampling": _native_sampling(request),
            "num_tasks": num_tasks,
            "num_rollouts": num_rollouts,
            "max_concurrent": max_concurrent,
            "shuffle": request.shuffle,
            "max_total_tokens": min(
                request.context_window,
                base.max_total_tokens or request.context_window,
            ),
            "output_dir": output_dir,
            "push": False,
            "rich": False,
            "server": False,
        }
    )
    config = EvalConfig.model_validate(raw)
    return Environment(config), config, run_eval


def _emit_batch(context: EvaluationContext, request: EvaluateRequest, records: list[dict[str, Any]]) -> None:
    attributes = {
        "model_variant_id": request.model.id,
        "evaluation_plan_id": request.plan.id,
        "evaluation_plan_kind": request.plan.kind,
        "environment_id": request.environment.id,
        "environment_category": request.environment.category,
        "inference_binding_id": request.inference.id,
        "execution_target_id": request.target.id,
    }
    for record in records:
        context.trace(
            TraceObservation(
                trace_type="verifiers",
                external_id=str(record["id"]),
                payload=record,
                attributes=attributes,
            )
        )


async def _run(context: EvaluationContext, request: EvaluateRequest, output_dir: Path) -> VerifiersRunResult:
    environment, config, native_run = _build_native(request, output_dir)
    trace_path = output_dir / "traces.jsonl"
    sync = VerifiersTraceSynchronizer(
        trace_path,
        lambda records: _emit_batch(context, request, records),
    )
    task = asyncio.create_task(native_run(environment, config))
    try:
        while not task.done():
            context.cancellation.raise_if_cancelled()
            sync.drain()
            await asyncio.sleep(0.1)
        traces = await task
    except BaseException:
        task.cancel()
        try:
            await task
        except BaseException:
            pass
        raise
    finally:
        stats = sync.finalize()
    return VerifiersRunResult(tuple(trace.id for trace in traces), stats)


def run_verifiers(
    context: EvaluationContext,
    request: EvaluateRequest,
    output_dir: Path,
) -> VerifiersRunResult:
    """Run one native Verifiers environment and stream completed trace records."""

    with context.phase("evaluation", {"backend": "verifiers"}):
        return asyncio.run(_run(context, request, output_dir))


__all__ = ["VerifiersRunResult", "run_verifiers"]
