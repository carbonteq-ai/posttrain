"""Tests for the serving package API."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import httpx
from posttrain.common import (
    EventObservation,
    InferenceBinding,
    LocalArtifactRef,
    MetricBatchObservation,
    MetricObservation,
    ProducedArtifact,
    RunContext,
    TraceObservation,
    Workload,
)
from posttrain.serve import (
    Endpoint,
    GenerationRequest,
    ServeBenchmarkRequest,
    ServeLaunchRequest,
    benchmark,
    generate,
    launch,
)
from posttrain.serve.backends.vllm.bindings import VllmBenchmarkConfig
from posttrain.serve.results import BenchmarkResult


@dataclass
class RecordingObserver:
    events: list[EventObservation] = field(default_factory=list)
    metrics_seen: list[MetricBatchObservation] = field(default_factory=list)
    artifacts: list[ProducedArtifact] = field(default_factory=list)
    traces: list[TraceObservation] = field(default_factory=list)

    def event(self, observation: EventObservation) -> None:
        self.events.append(observation)

    def metric(self, observation: MetricObservation) -> None:
        self.metrics_seen.append(MetricBatchObservation({observation.name: observation.value}, observation.step))

    def metrics(self, observation: MetricBatchObservation) -> None:
        self.metrics_seen.append(observation)

    def trace(self, observation: TraceObservation) -> None:
        self.traces.append(observation)

    def artifact(self, artifact: ProducedArtifact) -> None:
        self.artifacts.append(artifact)


def _context(tmp_path: Path, observer: RecordingObserver) -> RunContext:
    return RunContext(
        project_id="foundation-models",
        work_package_id="screen/qwen-smoke",
        run_id="00000000-0000-0000-0000-000000000001",
        job_kind="serve.benchmark",
        job_definition_version="1",
        workspace=tmp_path.resolve(),
        observer=observer,
    )


def _result(request: VllmBenchmarkConfig) -> BenchmarkResult:
    return BenchmarkResult(
        "vllm",
        request.model.base.repo_id,
        request.model.base.revision,
        request.inference_binding_id,
        request.cell.suite_id,
        request.cell.id,
        request.cell.shape_id,
        request.cell.context_window,
        request.cell.concurrency,
        request.cell.input_tokens,
        request.cell.output_tokens,
        1,
        1,
        request.cell.input_tokens,
        request.cell.output_tokens,
        0.1,
        0.5,
        2.0,
        256.0,
        64.0,
        320.0,
        0.5,
        0.01,
        0.01,
        0.02,
        0.005,
        0.005,
        0.006,
        1.0,
        2.0,
        1.0,
        ("answer",),
    )


def test_benchmark_consumes_catalog_seats_and_emits_run_identity(
    tmp_path: Path,
    qwen_screen_binding: InferenceBinding,
    foundation_smoke_workload: Workload,
) -> None:
    observer = RecordingObserver()
    inference = qwen_screen_binding
    request = ServeBenchmarkRequest(inference, foundation_smoke_workload)
    result = benchmark(_context(tmp_path, observer), request, runner=_result)

    assert result.output_token_throughput == 64.0
    assert observer.traces[0].external_id.startswith("00000000-0000-0000-0000-000000000001")
    attributes = observer.metrics_seen[0].attributes
    assert attributes["work_package_id"] == "screen/qwen-smoke"
    assert attributes["model_variant_id"] == inference.model.id
    assert attributes["inference_binding_id"] == inference.id
    assert isinstance(observer.artifacts[0].reference, LocalArtifactRef)


def test_generate_emits_variant_scoped_trace_and_native_artifact(
    tmp_path: Path,
    qwen_screen_binding: InferenceBinding,
) -> None:
    observer = RecordingObserver()
    context = _context(tmp_path, observer)
    model = qwen_screen_binding.model
    endpoint = Endpoint("http://model.test/v1", model.base.repo_id)

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read())
        assert payload["enable_thinking"] is False
        return httpx.Response(
            200,
            text="\n".join(
                [
                    'data: {"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}',
                    'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":1}}',
                    "data: [DONE]",
                ]
            ),
        )

    request = GenerationRequest(endpoint, ({"role": "user", "content": "Question"},), max_tokens=8)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = generate(context, request, model, client=client)

    assert result.content == "answer"
    assert observer.traces[0].attributes["model_variant_id"] == model.id
    native = cast(LocalArtifactRef, observer.artifacts[0].reference)
    assert len(json.loads(native.path.read_text())["response"]["events"]) == 2


def test_launch_manages_server_template_log_and_lifecycle(
    tmp_path: Path,
    qwen_screen_binding: InferenceBinding,
) -> None:
    observer = RecordingObserver()
    context = _context(tmp_path, observer)
    request = ServeLaunchRequest(qwen_screen_binding)
    closed: list[bool] = []

    class FakeServer:
        def __init__(self, request: ServeLaunchRequest, log_path: Path, template_path: Path | None) -> None:
            self.endpoint = request.endpoint
            self.log_path = log_path
            assert template_path is None

        def start(self) -> Endpoint:
            self.log_path.write_text("ready\n", encoding="utf-8")
            return self.endpoint

        def close(self) -> None:
            closed.append(True)

    with launch(context, request, server_factory=FakeServer) as endpoint:
        assert endpoint.model == request.inference.model.base.repo_id

    assert closed == [True]
    assert observer.artifacts[0].metadata["inference_binding_id"] == request.inference.id
