"""Public serving operations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from time import time_ns
from typing import Protocol

import httpx
from posttrain.common import LocalArtifactRef, ModelVariant, ProducedArtifact, RunContext, TraceObservation

from .backends.vllm import VllmServer, run_offline_benchmark
from .backends.vllm.bindings import VllmBenchmarkConfig
from .backends.vllm.bindings import benchmark_config as vllm_benchmark_config
from .backends.vllm.offline import BenchmarkMetric, BenchmarkPhase
from .online import (
    Endpoint,
    GenerationRequest,
    GenerationResult,
    ProbeResult,
    ServeLaunchRequest,
)
from .online import (
    generate as run_generation,
)
from .online import (
    probe as run_probe,
)
from .requests import ServeBenchmarkRequest
from .results import BenchmarkSweepResult


class BenchmarkRunner(Protocol):
    def __call__(
        self,
        request: VllmBenchmarkConfig,
        *,
        phase: BenchmarkPhase,
        metric: BenchmarkMetric,
    ) -> BenchmarkSweepResult: ...


class ManagedServer(Protocol):
    def start(self) -> Endpoint: ...

    def close(self) -> None: ...


type ServerFactory = Callable[[ServeLaunchRequest, Path, Path | None], ManagedServer]


def benchmark(
    context: RunContext,
    request: ServeBenchmarkRequest,
    *,
    runner: BenchmarkRunner = run_offline_benchmark,
) -> BenchmarkSweepResult:
    """Run one ordered concurrency sweep and emit only direct observations."""

    adapter_request = vllm_benchmark_config(request)
    attributes = _benchmark_attributes(request, adapter_request)
    context.event(
        "serve_benchmark_started",
        attributes,
    )

    def observe_backend_metrics(
        values: dict[str, float],
        runtime_attributes: dict[str, str | int],
    ) -> None:
        context.metrics(values, attributes={**attributes, **runtime_attributes})

    sweep = runner(adapter_request, phase=context.phase, metric=observe_backend_metrics)
    for sweep_index, result in enumerate(sweep.points):
        metric_attributes = {
            **attributes,
            "cell_id": result.cell_id,
            "sweep_index": sweep_index,
            "concurrency": result.concurrency,
        }
        context.metrics(result.metrics(), step=sweep_index, attributes=metric_attributes)
        for index, request_result in enumerate(result.request_results):
            context.trace(
                TraceObservation(
                    trace_type="inference",
                    external_id=f"{context.run_id}:{result.cell_id}:request:{index}",
                    payload={
                        "model": result.model,
                        "revision": result.revision,
                        "backend_request_id": request_result.request_id,
                        "record_id": request_result.record_id,
                        "cohort": result.cohort,
                        "sweep_index": sweep_index,
                        "concurrency": result.concurrency,
                        "warmup": False,
                        "input_tokens": request_result.input_tokens,
                        "output_tokens": request_result.output_tokens,
                        "queue_seconds": request_result.queue_seconds,
                        "prefill_seconds": request_result.prefill_seconds,
                        "decode_seconds": request_result.decode_seconds,
                        "engine_e2e_seconds": request_result.engine_e2e_seconds,
                        "ttft_seconds": request_result.ttft_seconds,
                        "tpot_seconds": request_result.tpot_seconds,
                        "truncated": False,
                        "error_class": None,
                    },
                    attributes=metric_attributes,
                )
            )
    for failure in sweep.point_failures:
        failure_attributes = {
            **attributes,
            "sweep_index": failure.sweep_index,
            "concurrency": failure.concurrency,
            "point_status": failure.status,
        }
        context.metrics(
            {
                "serve/run/concurrency": failure.concurrency,
                "serve/run/requests_attempted": 0,
                "serve/run/requests_measured": 0,
                "serve/run/requests_failed": int(failure.status == "failed"),
                "serve/run/requests_unsupported": int(failure.status == "unsupported"),
                "serve/run/point_resource_exhausted": int(failure.status == "resource_exhausted"),
                "serve/run/point_unsupported": int(failure.status == "unsupported"),
                "serve/run/point_failed": int(failure.status == "failed"),
            },
            step=failure.sweep_index,
            attributes=failure_attributes,
        )
        context.event(
            "serve_benchmark_point_failed",
            {
                **failure_attributes,
                "error_class": failure.error_class,
                "message": failure.message,
                "recoverable": failure.recoverable,
            },
        )
    with context.phase("artifact_export", {"backend": sweep.points[0].backend}):
        artifact_payload = sweep.as_json(
            runtime_configuration={
                "engine": asdict(adapter_request.engine),
                "sampling": asdict(adapter_request.sampling),
                "cohort": adapter_request.cohort,
            }
        )
        artifact_payload.update(
            {
                "model_variant_id": request.inference.model.id,
                "inference_binding_id": request.inference.id,
                "workload_id": request.workload.id,
                "execution_target_id": request.resolved_target.id,
            }
        )
        encoded = (json.dumps(artifact_payload, indent=2, sort_keys=True) + "\n").encode()
        output = context.workspace / "serving-result.json"
        output.write_bytes(encoded)
        context.artifact(
            ProducedArtifact(
                name=f"serving/{request.inference.model.id}/{adapter_request.cells[0].suite_id}",
                kind="serving-result",
                reference=LocalArtifactRef(output, hashlib.sha256(encoded).hexdigest()),
                metadata={**attributes, "schema_version": sweep.schema_version},
                role="benchmark",
            )
        )
    context.event(
        "serve_benchmark_completed",
        {
            "measured_points": len(sweep.points),
            "failed_points": len(sweep.point_failures),
            "termination_reason": sweep.termination_reason,
        },
    )
    return sweep


def _benchmark_attributes(
    request: ServeBenchmarkRequest,
    adapter: VllmBenchmarkConfig,
) -> dict[str, str]:
    attributes = {
        "model_variant_id": request.inference.model.id,
        "inference_binding_id": request.inference.id,
        "workload_id": request.workload.id,
        "execution_target_id": request.resolved_target.id,
        "suite_id": adapter.cells[0].suite_id,
        "cohort": adapter.cohort,
    }
    if adapter.corpus is not None:
        attributes.update(
            {
                "corpus_id": adapter.corpus.manifest.id,
                "corpus_revision": adapter.corpus.manifest.revision,
                "corpus_digest": adapter.corpus.manifest.digest,
            }
        )
    return attributes


@contextmanager
def launch(
    context: RunContext,
    request: ServeLaunchRequest,
    *,
    server_factory: ServerFactory = VllmServer,
) -> Iterator[Endpoint]:
    """Launch one managed endpoint and retain its server log as evidence."""

    log_path = context.workspace / "vllm-server.log"
    model = request.inference.model
    template = model.conversation.chat_template.text()
    template_path: Path | None = None
    if template is not None:
        template_path = context.workspace / "chat-template.jinja"
        template_path.write_text(template, encoding="utf-8")
    server = server_factory(request, log_path, template_path)
    context.event(
        "serve_launch_started",
        {"model_variant_id": model.id, "inference_binding_id": request.inference.id},
    )
    try:
        endpoint = server.start()
        context.event("serve_endpoint_ready", {"base_url": endpoint.base_url, "model": endpoint.model})
        yield endpoint
    finally:
        server.close()
        if log_path.is_file():
            encoded = log_path.read_bytes()
            context.artifact(
                ProducedArtifact(
                    name=f"serving/{model.id}/server-log",
                    kind="serving-log",
                    reference=LocalArtifactRef(log_path, hashlib.sha256(encoded).hexdigest()),
                    metadata={"inference_binding_id": request.inference.id},
                )
            )
        context.event("serve_endpoint_stopped", {"model_variant_id": model.id})


def probe(
    context: RunContext,
    endpoint: Endpoint,
    *,
    client: httpx.Client | None = None,
) -> ProbeResult:
    result = run_probe(endpoint, client=client)
    context.metrics(
        {
            "serve/probe_latency_seconds": result.latency_seconds,
            "serve/probe_healthy": int(result.healthy),
            "serve/probe_model_available": int(result.model_available),
        },
        attributes={"endpoint_model": endpoint.model},
    )
    return result


def generate(
    context: RunContext,
    request: GenerationRequest,
    model: ModelVariant,
    *,
    client: httpx.Client | None = None,
) -> GenerationResult:
    result = run_generation(request, model, client=client)
    observation_suffix = str(time_ns())
    external_id = f"{context.run_id}:generation:{observation_suffix}"
    metrics: dict[str, int | float] = {"serve/request_latency_seconds": result.latency_seconds}
    if result.ttft_seconds is not None:
        metrics["serve/request_ttft_seconds"] = result.ttft_seconds
    if result.input_tokens is not None:
        metrics["serve/request_input_tokens"] = result.input_tokens
    if result.output_tokens is not None:
        metrics["serve/request_output_tokens"] = result.output_tokens
    context.metrics(metrics, attributes={"model_variant_id": model.id, "endpoint_model": request.endpoint.model})
    context.trace(
        TraceObservation(
            trace_type="inference",
            external_id=external_id,
            payload={
                "messages": [
                    *(dict(message) for message in request.messages),
                    {"role": "assistant", "content": result.content},
                ],
                "reasoning": result.reasoning,
                "tool_call_deltas": list(result.tool_call_deltas),
                "finish_reason": result.finish_reason,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "latency_seconds": result.latency_seconds,
                "ttft_seconds": result.ttft_seconds,
            },
            attributes={"model_variant_id": model.id, "endpoint_model": request.endpoint.model},
        )
    )
    native = {
        "request": {
            "endpoint_model": request.endpoint.model,
            "messages": [dict(message) for message in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "reasoning_mode": request.reasoning_mode or model.default_reasoning_mode,
            "tools": [dict(tool) for tool in request.tools],
        },
        "response": result.as_json(),
    }
    encoded = (json.dumps(native, indent=2, sort_keys=True) + "\n").encode()
    output = context.workspace / f"generation-{observation_suffix}.json"
    output.write_bytes(encoded)
    context.artifact(
        ProducedArtifact(
            name=f"serving/{model.id}/generation-{observation_suffix}",
            kind="inference-output",
            reference=LocalArtifactRef(output, hashlib.sha256(encoded).hexdigest()),
            metadata={"external_id": external_id, "endpoint_model": request.endpoint.model},
        )
    )
    return result
