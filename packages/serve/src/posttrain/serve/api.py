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
from posttrain.common import ExecutionContext, LocalArtifactRef, ModelProfile, ProducedArtifact, TraceObservation

from .backends.vllm import VllmServer, run_offline_benchmark
from .online import (
    Endpoint,
    GenerationRequest,
    GenerationResult,
    LaunchRequest,
    ProbeResult,
)
from .online import (
    generate as run_generation,
)
from .online import (
    probe as run_probe,
)
from .requests import BenchmarkRequest
from .results import BenchmarkResult

type BenchmarkRunner = Callable[[BenchmarkRequest], BenchmarkResult]


class ManagedServer(Protocol):
    def start(self) -> Endpoint: ...

    def close(self) -> None: ...


type ServerFactory = Callable[[LaunchRequest, Path, Path | None], ManagedServer]


def benchmark(
    context: ExecutionContext,
    request: BenchmarkRequest,
    *,
    runner: BenchmarkRunner = run_offline_benchmark,
) -> BenchmarkResult:
    """Run one benchmark cell and emit only direct observations."""

    context.event(
        "serve_benchmark_started",
        {
            "model_profile_id": request.model.id,
            "serve_profile_id": request.profile.id,
            "suite_id": request.cell.suite_id,
            "cell_id": request.cell.id,
        },
    )
    result = runner(request)
    metric_attributes = {
        "model_profile_id": request.model.id,
        "serve_profile_id": request.profile.id,
        "suite_id": request.cell.suite_id,
        "cell_id": request.cell.id,
    }
    context.metrics(result.metrics(), attributes=metric_attributes)
    for index, sample in enumerate(result.samples):
        context.trace(
            TraceObservation(
                trace_type="inference",
                external_id=f"{context.attempt.id}:{request.cell.id}:{index}",
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
            name=f"serving/{request.model.id}/{request.cell.id}",
            kind="serving-benchmark",
            reference=LocalArtifactRef(output, hashlib.sha256(encoded).hexdigest()),
            metadata=metric_attributes,
        )
    )
    context.event("serve_benchmark_completed", {"cell_id": request.cell.id})
    return result


@contextmanager
def launch(
    context: ExecutionContext,
    request: LaunchRequest,
    *,
    server_factory: ServerFactory = VllmServer,
) -> Iterator[Endpoint]:
    """Launch one managed endpoint and retain its server log as evidence."""

    log_path = context.workspace / "vllm-server.log"
    template = request.model.conversation.chat_template.text()
    template_path: Path | None = None
    if template is not None:
        template_path = context.workspace / "chat-template.jinja"
        template_path.write_text(template, encoding="utf-8")
    server = server_factory(request, log_path, template_path)
    context.event(
        "serve_launch_started",
        {"model_profile_id": request.model.id, "serve_profile_id": request.profile.id},
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
                    name=f"serving/{request.model.id}/server-log",
                    kind="serving-log",
                    reference=LocalArtifactRef(log_path, hashlib.sha256(encoded).hexdigest()),
                    metadata={"serve_profile_id": request.profile.id},
                )
            )
        context.event("serve_endpoint_stopped", {"model_profile_id": request.model.id})


def probe(
    context: ExecutionContext,
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
    context: ExecutionContext,
    request: GenerationRequest,
    model: ModelProfile,
    *,
    client: httpx.Client | None = None,
) -> GenerationResult:
    result = run_generation(request, model, client=client)
    metrics: dict[str, int | float] = {"serve/request_latency_seconds": result.latency_seconds}
    if result.ttft_seconds is not None:
        metrics["serve/request_ttft_seconds"] = result.ttft_seconds
    if result.input_tokens is not None:
        metrics["serve/request_input_tokens"] = result.input_tokens
    if result.output_tokens is not None:
        metrics["serve/request_output_tokens"] = result.output_tokens
    context.metrics(metrics, attributes={"model_profile_id": model.id, "endpoint_model": request.endpoint.model})
    context.trace(
        TraceObservation(
            trace_type="inference",
            external_id=f"{context.attempt.id}:generation:{len(result.events)}:{time_ns()}",
            payload={
                "messages": [
                    *(dict(message) for message in request.messages),
                    {"role": "assistant", "content": result.content},
                ],
                "reasoning": result.reasoning,
                "tool_call_deltas": list(result.tool_call_deltas),
                "finish_reason": result.finish_reason,
                "events": list(result.events),
            },
            attributes={"model_profile_id": model.id, "endpoint_model": request.endpoint.model},
        )
    )
    return result
