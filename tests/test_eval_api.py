from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from posttrain.common import (
    EventObservation,
    ExecutionContext,
    Invocation,
    Job,
    JobAction,
    MetricBatchObservation,
    MetricObservation,
    ProducedArtifact,
    RunAttempt,
    TraceObservation,
)
from posttrain.common.profiles import QWEN_35_2B
from posttrain.eval import (
    EnvironmentProgram,
    EnvironmentSource,
    EvaluationBudget,
    EvaluationProgram,
    EvaluationRequest,
    EvaluationTarget,
    SamplingPolicy,
    evaluate,
)
from posttrain.eval.backends.verifiers import VerifiersRunResult
from posttrain.eval.backends.verifiers.synchronization import TraceSyncStats
from posttrain.eval.programs import AGENTIC_SMOKE, AUTOMATIONBENCH_PUBLIC, GENERAL_SMOKE

REVISION = "a" * 40


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[EventObservation] = []
        self.metrics_log: list[MetricBatchObservation] = []
        self.traces: list[TraceObservation] = []
        self.artifacts: list[ProducedArtifact] = []

    def event(self, observation: EventObservation) -> None:
        self.events.append(observation)

    def metric(self, observation: MetricObservation) -> None:
        del observation

    def metrics(self, observation: MetricBatchObservation) -> None:
        self.metrics_log.append(observation)

    def trace(self, observation: TraceObservation) -> None:
        self.traces.append(observation)

    def artifact(self, artifact: ProducedArtifact) -> None:
        self.artifacts.append(artifact)


def context(tmp_path: Path, observer: RecordingObserver) -> ExecutionContext:
    return ExecutionContext(
        job=Job("tests/eval", REVISION, "Evaluation test"),
        action=JobAction("tests/eval", "general/math", "general-evaluation"),
        invocation=Invocation("00000000-0000-4000-8000-000000000001"),
        attempt=RunAttempt("00000000-0000-4000-8000-000000000002", 1),
        workspace=tmp_path.resolve(),
        observer=observer,
    )


def request(*, context_window: int = 8_192) -> EvaluationRequest:
    source = EnvironmentSource("fake-env", "https://example.test/environments", REVISION)
    program = EvaluationProgram(
        "general-test-v1",
        "general",
        (
            EnvironmentProgram(
                "math",
                "math-reasoning",
                source,
                lambda: object(),
                SamplingPolicy(max_tokens=512),
                num_tasks=2,
            ),
        ),
    )
    return EvaluationRequest(
        model=QWEN_35_2B,
        target=EvaluationTarget("http://127.0.0.1:8000/v1", QWEN_35_2B.artifact.repo_id),
        program=program,
        environment_id="math",
        context_window=context_window,
    )


def test_general_smoke_is_code_defined_and_category_selectable() -> None:
    assert GENERAL_SMOKE.kind == "general"
    assert {item.category for item in GENERAL_SMOKE.environments} == {
        "math-reasoning",
        "instruction-following",
        "code-generation",
        "multi-turn-state",
    }
    assert all(len(item.source.revision) == 40 for item in GENERAL_SMOKE.environments)
    assert GENERAL_SMOKE.select("math-gsm8k")[0].source.package == "gsm8k-v1"


def test_agentic_and_domain_programs_share_the_native_port() -> None:
    assert AGENTIC_SMOKE.kind == "general"
    assert AGENTIC_SMOKE.environments[0].source.package == "automationbench-v1"
    assert AGENTIC_SMOKE.environments[0].max_concurrent == 1
    assert AUTOMATIONBENCH_PUBLIC.kind == "domain"
    assert {item.category for item in AUTOMATIONBENCH_PUBLIC.environments} == {
        "business-sales",
        "business-marketing",
        "business-operations",
        "business-support",
        "business-finance",
        "business-hr",
    }


def test_evaluation_request_rejects_response_budget_at_context_limit() -> None:
    with pytest.raises(ValueError, match="response budget"):
        request(context_window=512)


def test_invocation_budget_can_select_a_small_subset_without_mutating_program() -> None:
    base = request()
    smaller = EvaluationRequest(
        model=base.model,
        target=base.target,
        program=base.program,
        environment_id=base.environment_id,
        context_window=base.context_window,
        budget=EvaluationBudget(num_tasks=1, max_concurrent=1),
    )
    assert smaller.resolved_budget == (1, 1, 1)
    assert base.environment.num_tasks == 2


def test_evaluate_emits_direct_sync_metrics_and_native_artifact(tmp_path: Path) -> None:
    observer = RecordingObserver()

    def fake_runner(
        execution: ExecutionContext,
        evaluation: EvaluationRequest,
        output: Path,
    ) -> VerifiersRunResult:
        del execution, evaluation
        (output / "config.toml").write_text("model = 'fake'\n", encoding="utf-8")
        (output / "traces.jsonl").write_text('{"id":"trace-1"}\n', encoding="utf-8")
        return VerifiersRunResult(
            ("trace-1",),
            TraceSyncStats(observed_records=1, emitted_records=1),
        )

    result = evaluate(context(tmp_path, observer), request(), runner=fake_runner)

    assert result.trace_ids == ("trace-1",)
    assert result.synchronization.complete
    assert result.native_artifact.reference.path.name == "math"  # type: ignore[union-attr]
    assert observer.artifacts == [result.native_artifact]
    values = observer.metrics_log[0].values
    assert values["eval/traces_observed"] == 1
    assert "eval/mean_reward" not in values
    assert observer.events[-1].name == "evaluation_completed"


def test_program_rejects_unknown_environment() -> None:
    with pytest.raises(ValueError, match="unknown environment"):
        request().program.environment("missing")


def test_general_program_factories_return_native_configs_when_extra_is_installed() -> None:
    pytest.importorskip("verifiers.v1")
    config: Any = GENERAL_SMOKE.environment("math-gsm8k").factory()
    assert config.taskset.id == "gsm8k-v1"
    assert config.harness.id == "null"
    assert config.timeout.rollout == 180
