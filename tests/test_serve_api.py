from __future__ import annotations

import uuid
from pathlib import Path

from posttrain.common import (
    EventObservation,
    ExecutionContext,
    Invocation,
    Job,
    JobAction,
    LocalArtifactRef,
    MetricBatchObservation,
    MetricObservation,
    ProducedArtifact,
    RunAttempt,
    TraceObservation,
)
from posttrain.common.profiles import QWEN_35_2B
from posttrain.serve import QWEN35_VLLM_TEXT, BenchmarkCell, BenchmarkRequest, BenchmarkResult, benchmark


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[EventObservation] = []
        self.metric_observations: list[MetricObservation] = []
        self.traces: list[TraceObservation] = []
        self.artifacts: list[ProducedArtifact] = []

    def event(self, observation: EventObservation) -> None:
        self.events.append(observation)

    def metric(self, observation: MetricObservation) -> None:
        self.metric_observations.append(observation)

    def metrics(self, observation: MetricBatchObservation) -> None:
        for name, value in observation.values.items():
            self.metric_observations.append(MetricObservation(name, value, observation.step, observation.attributes))

    def trace(self, observation: TraceObservation) -> None:
        self.traces.append(observation)

    def artifact(self, artifact: ProducedArtifact) -> None:
        self.artifacts.append(artifact)


def _request() -> BenchmarkRequest:
    return BenchmarkRequest(
        model=QWEN_35_2B,
        profile=QWEN35_VLLM_TEXT,
        cell=BenchmarkCell(
            suite_id="test-suite",
            shape_id="short",
            context_window=1_024,
            concurrency=1,
            input_tokens=128,
            output_tokens=32,
            warmup_iterations=1,
            iterations=1,
        ),
    )


def _result(request: BenchmarkRequest) -> BenchmarkResult:
    return BenchmarkResult(
        backend="vllm",
        model=request.model.artifact.repo_id,
        revision=request.model.artifact.revision,
        profile_id=request.profile.id,
        suite_id=request.cell.suite_id,
        cell_id=request.cell.id,
        shape_id=request.cell.shape_id,
        context_window=request.cell.context_window,
        concurrency=1,
        target_input_tokens=128,
        target_output_tokens=32,
        requests=1,
        iterations=1,
        input_tokens=128,
        output_tokens=32,
        engine_start_seconds=10.0,
        elapsed_seconds=0.5,
        request_throughput=2.0,
        input_token_throughput=256.0,
        output_token_throughput=64.0,
        total_token_throughput=320.0,
        mean_batch_latency=0.5,
        mean_ttft=0.04,
        p50_ttft=0.04,
        p95_ttft=0.04,
        mean_tpot=0.014,
        p50_tpot=0.014,
        p95_tpot=0.014,
        baseline_gpu_memory_gib=1.2,
        peak_gpu_memory_gib=6.9,
        peak_gpu_memory_delta_gib=5.7,
        samples=("answer",),
    )


def test_benchmark_emits_direct_metrics_trace_and_native_artifact(tmp_path: Path) -> None:
    observer = RecordingObserver()
    context = ExecutionContext(
        job=Job("tests/serve", "a" * 40, "Serve test"),
        action=JobAction("tests/serve", "benchmark", "serving-benchmark"),
        invocation=Invocation(str(uuid.uuid4())),
        attempt=RunAttempt.new(),
        workspace=tmp_path.resolve(),
        observer=observer,
    )
    request = _request()
    result = benchmark(context, request, runner=lambda value: _result(value))

    assert result.output_token_throughput == 64.0
    assert {metric.name for metric in observer.metric_observations} >= {
        "serve/output_token_throughput",
        "serve/engine_start_seconds",
        "serve/peak_gpu_memory_gib",
    }
    assert observer.traces[0].trace_type == "inference"
    assert observer.traces[0].payload["messages"] == [{"role": "assistant", "content": "answer"}]
    artifact_ref = observer.artifacts[0].reference
    assert isinstance(artifact_ref, LocalArtifactRef)
    assert artifact_ref.path.read_text().startswith("{\n")
    assert [event.name for event in observer.events] == ["serve_benchmark_started", "serve_benchmark_completed"]
