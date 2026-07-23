"""Public serving operations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from time import time_ns
from typing import Protocol

import httpx
from posttrain.common import LocalArtifactRef, ModelVariant, ProducedArtifact, RunContext, TraceObservation

from .backends.vllm import VllmServer, run_offline_benchmark
from .backends.vllm.bindings import VllmBenchmarkConfig
from .backends.vllm.bindings import benchmark_config as vllm_benchmark_config
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
from .results import BenchmarkResult

type BenchmarkRunner = Callable[[VllmBenchmarkConfig], BenchmarkResult]


class ManagedServer(Protocol):
    def start(self) -> Endpoint: ...

    def close(self) -> None: ...


type ServerFactory = Callable[[ServeLaunchRequest, Path, Path | None], ManagedServer]


def benchmark(
    context: RunContext,
    request: ServeBenchmarkRequest,
    *,
    runner: BenchmarkRunner = run_offline_benchmark,
) -> BenchmarkResult:
    """Run one benchmark cell and emit only direct observations."""

    adapter_request = vllm_benchmark_config(request)
    attributes = _benchmark_attributes(request, adapter_request)
    context.event(
        "serve_benchmark_started",
        attributes,
    )
    result = runner(adapter_request)
    metric_attributes = attributes
    context.metrics(result.metrics(), attributes=metric_attributes)
    for index, sample in enumerate(result.samples):
        context.trace(
            TraceObservation(
                trace_type="inference",
                external_id=f"{context.run_id}:{adapter_request.cell.id}:{index}",
                payload={
                    "messages": [{"role": "assistant", "content": sample}],
                    "model": result.model,
                    "revision": result.revision,
                    "target_input_tokens": result.target_input_tokens,
                    "target_output_tokens": result.target_output_tokens,
                },
                attributes=metric_attributes,
            )
        )
    encoded = (json.dumps(result.as_json(), indent=2, sort_keys=True) + "\n").encode()
    output = context.workspace / "benchmark.json"
    output.write_bytes(encoded)
    context.artifact(
        ProducedArtifact(
            name=f"serving/{request.inference.model.id}/{adapter_request.cell.id}",
            kind="serving-benchmark",
            reference=LocalArtifactRef(output, hashlib.sha256(encoded).hexdigest()),
            metadata=metric_attributes,
        )
    )
    context.event("serve_benchmark_completed", {"cell_id": adapter_request.cell.id})
    return result


def _benchmark_attributes(
    request: ServeBenchmarkRequest,
    adapter: VllmBenchmarkConfig,
) -> dict[str, str]:
    return {
        "model_variant_id": request.inference.model.id,
        "inference_binding_id": request.inference.id,
        "workload_id": request.workload.id,
        "execution_target_id": request.resolved_target.id,
        "suite_id": adapter.cell.suite_id,
        "cell_id": adapter.cell.id,
    }


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
