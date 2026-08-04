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
from posttrain.serve.backends.vllm.offline import BenchmarkMetric, BenchmarkPhase
from posttrain.serve.results import (
    BenchmarkPointFailure,
    BenchmarkResult,
    BenchmarkSweepResult,
    InferenceRequestResult,
)


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


def _result(
    request: VllmBenchmarkConfig,
    *,
    phase: BenchmarkPhase,
    metric: BenchmarkMetric,
) -> BenchmarkSweepResult:
    with phase("model_loading", {"backend": "vllm"}):
        pass
    points = []
    for sweep_index, cell in enumerate(request.cells):
        with phase("benchmark_point_setup", {"backend": "vllm", "sweep_index": sweep_index}):
            pass
        with phase("benchmark_warmup", {"backend": "vllm", "iterations": 1, "sweep_index": sweep_index}):
            pass
        with phase("benchmark_measurement", {"backend": "vllm", "iterations": 1, "sweep_index": sweep_index}):
            pass
        metric(
            {"serve/backend/kv_cache_usage_ratio": 0.5},
            {"backend": "vllm", "phase": "measurement", "sweep_index": sweep_index},
        )
        points.append(
            BenchmarkResult(
                "vllm",
                request.model.base.repo_id,
                request.model.base.revision,
                request.inference_binding_id,
                cell.suite_id,
                cell.id,
                cell.shape_id,
                cell.context_window,
                cell.concurrency,
                cell.input_tokens or 0,
                cell.output_tokens,
                1,
                1,
                cell.input_tokens or 0,
                cell.output_tokens,
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
                request_results=(
                    InferenceRequestResult(
                        request_id="request-1",
                        record_id=None,
                        input_tokens=cell.input_tokens or 0,
                        output_tokens=cell.output_tokens,
                        queue_seconds=0.001,
                        prefill_seconds=0.01,
                        decode_seconds=0.2,
                        engine_e2e_seconds=0.211,
                        ttft_seconds=0.011,
                        tpot_seconds=0.005,
                    ),
                ),
                kv_cache_capacity_tokens=8192,
                kv_cache_peak_usage_ratio=0.5,
            )
        )
    with phase("runtime_cleanup", {"backend": "vllm"}):
        pass
    return BenchmarkSweepResult(
        points=tuple(points),
        configured_concurrencies=tuple(cell.concurrency for cell in request.cells),
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

    assert result.points[0].output_token_throughput == 64.0
    assert observer.traces[0].external_id.startswith("00000000-0000-0000-0000-000000000001")
    assert observer.traces[0].payload["prefill_seconds"] == 0.01
    assert observer.traces[0].payload["concurrency"] == request.workload.concurrency[0]
    assert observer.traces[0].payload["sweep_index"] == 0
    assert "messages" not in observer.traces[0].payload
    assert any(batch.values.get("serve/backend/kv_cache_usage_ratio") == 0.5 for batch in observer.metrics_seen)
    attributes = observer.metrics_seen[0].attributes
    assert attributes["work_package_id"] == "screen/qwen-smoke"
    assert attributes["model_variant_id"] == inference.model.id
    assert attributes["inference_binding_id"] == inference.id
    assert isinstance(observer.artifacts[0].reference, LocalArtifactRef)
    phase_events = [
        (event.name, event.attributes["phase"]) for event in observer.events if event.name.startswith("runtime_phase_")
    ]
    assert phase_events == [
        ("runtime_phase_started", "model_loading"),
        ("runtime_phase_completed", "model_loading"),
        ("runtime_phase_started", "benchmark_point_setup"),
        ("runtime_phase_completed", "benchmark_point_setup"),
        ("runtime_phase_started", "benchmark_warmup"),
        ("runtime_phase_completed", "benchmark_warmup"),
        ("runtime_phase_started", "benchmark_measurement"),
        ("runtime_phase_completed", "benchmark_measurement"),
        ("runtime_phase_started", "runtime_cleanup"),
        ("runtime_phase_completed", "runtime_cleanup"),
        ("runtime_phase_started", "artifact_export"),
        ("runtime_phase_completed", "artifact_export"),
    ]


def test_benchmark_emits_ordered_points_from_one_run(
    tmp_path: Path,
    qwen_screen_binding: InferenceBinding,
    foundation_smoke_workload: Workload,
) -> None:
    observer = RecordingObserver()
    workload = Workload(
        id=foundation_smoke_workload.id,
        revision=foundation_smoke_workload.revision,
        requests=foundation_smoke_workload.requests,
        concurrency=(1, 2, 4),
        warmup_repetitions=1,
        measured_repetitions=1,
    )

    result = benchmark(
        _context(tmp_path, observer),
        ServeBenchmarkRequest(qwen_screen_binding, workload),
        runner=_result,
    )

    assert result.completed_concurrencies == (1, 2, 4)
    assert [trace.payload["sweep_index"] for trace in observer.traces] == [0, 1, 2]
    point_metrics = [batch for batch in observer.metrics_seen if "serve/run/output_tokens_measured" in batch.values]
    assert [batch.step for batch in point_metrics] == [0, 1, 2]
    assert all("serve/output_token_throughput" not in batch.values for batch in point_metrics)
    assert observer.artifacts[0].kind == "serving-result"
    artifact = json.loads(cast(LocalArtifactRef, observer.artifacts[0].reference).path.read_text())
    assert artifact["schema_version"] == 2
    assert artifact["methodology"] == "single_run_concurrency_sweep"
    assert artifact["model_variant_id"] == qwen_screen_binding.model.id
    assert artifact["inference_binding_id"] == qwen_screen_binding.id
    assert artifact["workload_id"] == workload.id
    assert artifact["execution_target_id"] == qwen_screen_binding.target.id
    assert [point["concurrency"] for point in artifact["points"]] == [1, 2, 4]


def test_benchmark_retains_safe_terminal_point_failure(
    tmp_path: Path,
    qwen_screen_binding: InferenceBinding,
    foundation_smoke_workload: Workload,
) -> None:
    observer = RecordingObserver()
    workload = Workload(
        id=foundation_smoke_workload.id,
        revision=foundation_smoke_workload.revision,
        requests=foundation_smoke_workload.requests,
        concurrency=(1, 2),
        warmup_repetitions=1,
        measured_repetitions=1,
    )

    def partial_runner(
        request: VllmBenchmarkConfig,
        *,
        phase: BenchmarkPhase,
        metric: BenchmarkMetric,
    ) -> BenchmarkSweepResult:
        measured = _result(request, phase=phase, metric=metric)
        return BenchmarkSweepResult(
            points=measured.points[:1],
            configured_concurrencies=(1, 2),
            point_failures=(
                BenchmarkPointFailure(
                    sweep_index=1,
                    concurrency=2,
                    status="resource_exhausted",
                    error_class="OutOfMemoryError",
                    message="CUDA out of memory",
                    recoverable=True,
                ),
            ),
            termination_reason="resource_exhausted_boundary",
        )

    result = benchmark(
        _context(tmp_path, observer),
        ServeBenchmarkRequest(qwen_screen_binding, workload),
        runner=partial_runner,
    )

    assert result.completed_concurrencies == (1,)
    assert result.point_failures[0].status == "resource_exhausted"
    failure_event = next(event for event in observer.events if event.name == "serve_benchmark_point_failed")
    assert failure_event.attributes["concurrency"] == 2
    artifact = json.loads(cast(LocalArtifactRef, observer.artifacts[0].reference).path.read_text())
    assert artifact["point_failures"][0]["error_class"] == "OutOfMemoryError"
    failure_metrics = next(batch for batch in observer.metrics_seen if batch.step == 1)
    assert failure_metrics.values["serve/run/point_resource_exhausted"] == 1


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
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
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
