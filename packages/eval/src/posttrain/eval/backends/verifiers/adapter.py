"""Compose the pinned Verifiers v1 environment, client, runner, and trace schema."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from posttrain.common import InferenceBinding, JsonValue, RunContext, TraceObservation
from posttrain.environment import project_verifiers_trace_facts, verifiers_trace_attributes

from ...requests import EvaluateRequest, RemotePolicy
from ...results import EvaluationPopulation
from .synchronization import TraceSyncStats, VerifiersTraceSynchronizer

type EvaluationContext = RunContext


@dataclass(frozen=True, slots=True)
class VerifiersRunResult:
    trace_ids: tuple[str, ...]
    synchronization: TraceSyncStats
    population: EvaluationPopulation


class _NativeEnvConfig(Protocol):
    max_total_tokens: int | None

    def model_dump(self, *, mode: str) -> dict[str, Any]: ...


def _imports() -> tuple[type[Any], type[Any], Any]:
    try:
        from .runtime import configure_preinstalled_runtime

        configure_preinstalled_runtime()
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
    values["top_k"] = policy.top_k
    if policy.min_p is not None:
        values["min_p"] = policy.min_p
    values["repetition_penalty"] = policy.repetition_penalty
    values["presence_penalty"] = policy.presence_penalty
    if policy.reasoning_effort is not None:
        values["reasoning_effort"] = policy.reasoning_effort
    if not isinstance(request.model, RemotePolicy):
        if not isinstance(request.inference, InferenceBinding):
            raise TypeError("local model evaluation requires a local inference binding")
        inference = cast(InferenceBinding, request.inference)
        values.update(inference.sampling)
        template_kwargs = request.model.conversation.reasoning_mode(request.resolved_reasoning_mode).kwargs()
        declared_kwargs = values.get("chat_template_kwargs")
        if declared_kwargs is not None and not isinstance(declared_kwargs, dict):
            raise ValueError("inference chat_template_kwargs must be an object")
        merged_kwargs = dict(cast(dict[str, JsonValue], declared_kwargs or {}))
        for key, value in template_kwargs.items():
            merged_kwargs.setdefault(key, value)
        if merged_kwargs:
            values["chat_template_kwargs"] = merged_kwargs
    service = request.remote_service
    if service is not None:
        values.update(service.request_defaults)
    return values


def _build_native(request: EvaluateRequest, output_dir: Path) -> tuple[Any, Any, Any]:
    EvalConfig, Environment, (EnvConfig, run_eval) = _imports()
    base = request.environment.activate()
    if not isinstance(base, EnvConfig):
        raise TypeError("environment factories must return verifiers.v1.EnvConfig")
    base = cast(_NativeEnvConfig, base)
    num_tasks, num_rollouts, max_concurrent = request.resolved_budget
    endpoint = request.resolved_endpoint
    service = request.remote_service
    client: dict[str, JsonValue] = {
        "type": "eval",
        "base_url": endpoint.base_url,
        "api_key_var": endpoint.api_key_var,
    }
    if service is not None and service.headers:
        client["headers"] = dict(service.headers)
    raw = base.model_dump(mode="python")
    raw.update(
        {
            "model": endpoint.served_model,
            "client": client,
            "sampling": _native_sampling(request),
            "num_tasks": num_tasks,
            "num_rollouts": num_rollouts,
            "max_concurrent": max_concurrent,
            "shuffle": request.resolved_shuffle,
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
    num_tasks, _, _ = request.resolved_budget
    attributes = {
        "evaluation_subject_id": request.model.id,
        "evaluation_plan_id": request.plan.id,
        "evaluation_plan_kind": request.plan.kind,
        "environment_id": request.environment.id,
        "environment_category": request.environment.category,
        "inference_binding_id": request.inference.id,
        "execution_target_id": request.target.id,
        "num_tasks": num_tasks,
        "task_selection": "verifiers-fixed-shuffle" if request.resolved_shuffle else "head",
    }
    if isinstance(request.model, RemotePolicy):
        assert request.remote_service is not None
        attributes.update(
            {
                "evaluation_subject_kind": "remote-policy",
                "remote_policy_revision": request.model.revision,
                "remote_service_id": request.remote_service.id,
                "remote_service_revision": request.remote_service.revision,
                "remote_service_protocol": request.remote_service.protocol,
                "remote_service_origin": request.remote_service.origin,
            }
        )
    else:
        attributes.update(
            {
                "evaluation_subject_kind": "model-variant",
                "model_variant_id": request.model.id,
                **request.model.trace_identity(),
            }
        )
    reward_component_sources = request.plan.environment(request.environment_id).reward_component_sources
    for record in records:
        observed = _observed_model(record)
        trace_attributes = {**attributes, **verifiers_trace_attributes(record)}
        if observed is not None:
            trace_attributes["observed_model"] = observed
        facts = project_verifiers_trace_facts(
            record,
            attributes=trace_attributes,
            reward_component_sources=reward_component_sources,
        )
        context.trace(
            TraceObservation(
                trace_type="verifiers",
                external_id=str(record["id"]),
                payload=record,
                attributes=trace_attributes,
                facts=(facts,),
            )
        )


def _observed_model(record: dict[str, Any]) -> str | None:
    agent = record.get("agent")
    model = agent.get("model") if isinstance(agent, dict) else None
    return str(model) if isinstance(model, str) and model else None


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
    expected = request.resolved_budget[0] * request.resolved_budget[1]
    attempted = len(traces)
    population = EvaluationPopulation(
        attempted=attempted,
        complete=sum(bool(trace.is_completed) for trace in traces),
        failed=sum(bool(trace.has_error) for trace in traces),
        truncated=sum(bool(trace.is_truncated) for trace in traces),
        coverage_missing=max(expected - attempted, 0),
    )
    return VerifiersRunResult(
        tuple(trace.id for trace in traces),
        stats,
        population,
    )


def run_verifiers(
    context: EvaluationContext,
    request: EvaluateRequest,
    output_dir: Path,
) -> VerifiersRunResult:
    """Run one native Verifiers environment and stream completed trace records."""

    with context.phase("evaluation", {"backend": "verifiers"}):
        return asyncio.run(_run(context, request, output_dir))


__all__ = ["VerifiersRunResult", "run_verifiers"]
