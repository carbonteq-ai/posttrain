"""Tests for the evaluation package API."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from posttrain.common import (
    Catalog,
    CatalogRef,
    EventObservation,
    ExecutionTarget,
    InferenceBinding,
    MetricBatchObservation,
    MetricObservation,
    ProducedArtifact,
    RunContext,
    TraceObservation,
)
from posttrain.common.variants import QWEN_35_2B
from posttrain.eval import (
    EnvironmentBinding,
    EnvironmentSource,
    EvaluateRequest,
    EvaluationBudget,
    EvaluationEndpoint,
    EvaluationPlan,
    EvaluationPopulation,
    ExternalInferenceService,
    PythonFactoryActivation,
    RemoteEvaluationBinding,
    RemotePolicy,
    SamplingPolicy,
    VerifiersV1ConfigActivation,
    domain,
    evaluate,
    evaluation_catalog_decoders,
    general,
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


def context(tmp_path: Path, observer: RecordingObserver) -> RunContext:
    return RunContext(
        project_id="tests",
        work_package_id="qualify/eval",
        run_id="00000000-0000-4000-8000-000000000002",
        job_kind="eval.general",
        job_definition_version="1",
        workspace=tmp_path.resolve(),
        observer=observer,
    )


def request(*, context_window: int = 8_192) -> EvaluateRequest:
    source = EnvironmentSource("fake-env", "https://example.test/environments", REVISION)
    plan = EvaluationPlan(
        "general-test-v1",
        "general",
        (
            EnvironmentBinding(
                "math",
                "math-reasoning",
                source,
                PythonFactoryActivation("builtins:object"),
                SamplingPolicy(max_tokens=512),
                num_tasks=2,
            ),
        ),
    )
    target = ExecutionTarget("targets/local-cuda-8gb", "1", "nvidia-cuda", 8)
    inference = InferenceBinding(
        "inference/qwen-eval@1",
        "1",
        QWEN_35_2B,
        "vllm@0.25.1",
        QWEN_35_2B.renderer_contract,
        {"max_model_len": 8192, "gpu_memory_utilization": 0.75},
        {"max_tokens": 512},
        target,
        ("eval",),
    )
    return EvaluateRequest(
        model=QWEN_35_2B,
        plan=plan,
        inference=inference,
        target=target,
        endpoint=EvaluationEndpoint("http://127.0.0.1:8000/v1", QWEN_35_2B.base.repo_id),
        environment_id="math",
        context_window=context_window,
    )


@pytest.mark.parametrize(
    ("repository", "subdirectory"),
    [
        ("ssh://git@github.com/org/env", "environment"),
        ("https://user:secret@github.com/org/env", "environment"),
        ("https://github.com/org/env?token=secret", "environment"),
        ("https://GitHub.com/org/env", "environment"),
        ("https://github.com/org/env", "../environment"),
        ("https://github.com/org/env", "environment/./nested"),
    ],
)
def test_environment_source_rejects_nonportable_git_identity(
    repository: str,
    subdirectory: str,
) -> None:
    with pytest.raises(ValueError):
        EnvironmentSource(
            "fake-env",
            repository,
            REVISION,
            subdirectory,
        )


def canonical_request() -> EvaluateRequest:
    return request()


def remote_request(*, request_defaults: dict[str, Any] | None = None) -> EvaluateRequest:
    source = EnvironmentSource("fake-env", "https://example.test/environments", REVISION)
    plan = EvaluationPlan(
        "remote-screen-test-v1",
        "general",
        (
            EnvironmentBinding(
                "tool-loop",
                "tool-use",
                source,
                PythonFactoryActivation("builtins:object"),
                SamplingPolicy(max_tokens=512),
                num_tasks=2,
            ),
        ),
    )
    target = ExecutionTarget("targets/external-screen", "1", "network-client")
    policy = RemotePolicy(
        "policies/qwen-via-openrouter@1",
        "2026-07-31",
        "qwen/qwen3.5-2b",
        32_768,
        {"tools": True},
    )
    service = ExternalInferenceService(
        "services/openrouter@1",
        "1",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        {"HTTP-Referer": "https://posttrain.example"},
        request_defaults or {"provider": {"allow_fallbacks": False}},
    )
    binding = RemoteEvaluationBinding(
        "inference/qwen-via-openrouter-eval@1",
        "1",
        policy,
        service,
        ("screen", "eval"),
    )
    return EvaluateRequest(
        model=policy,
        plan=plan,
        inference=binding,
        target=target,
        endpoint=None,
        environment_id="tool-loop",
        context_window=8_192,
    )


def test_general_smoke_is_code_defined_and_category_selectable() -> None:
    assert isinstance(GENERAL_SMOKE, EvaluationPlan)
    assert isinstance(GENERAL_SMOKE.environments[0], EnvironmentBinding)
    assert not hasattr(GENERAL_SMOKE, "model")
    assert GENERAL_SMOKE.kind == "general"
    assert {item.category for item in GENERAL_SMOKE.environments} == {
        "math-reasoning",
        "instruction-following",
        "code-generation",
        "multi-turn-state",
    }
    assert all(len(item.source.revision) == 40 for item in GENERAL_SMOKE.environments)
    assert GENERAL_SMOKE.select("math-gsm8k")[0].source.package == "gsm8k-v1"
    assert isinstance(
        GENERAL_SMOKE.select("math-gsm8k")[0].activation,
        VerifiersV1ConfigActivation,
    )


def test_python_factory_activation_is_lazy_until_runtime() -> None:
    activation = PythonFactoryActivation("package_that_is_not_installed_for_detached_planning:create_environment")

    with pytest.raises(RuntimeError, match="module is not installed"):
        activation.activate()


def test_verifiers_activation_digest_is_canonical() -> None:
    first = VerifiersV1ConfigActivation({"taskset": {"id": "example", "split": "train"}, "max_turns": 4})
    second = VerifiersV1ConfigActivation({"max_turns": 4, "taskset": {"split": "train", "id": "example"}})

    assert first.digest == second.digest
    assert first.to_payload()["kind"] == "verifiers-config"


def test_agentic_and_domain_programs_share_the_native_port() -> None:
    assert AGENTIC_SMOKE.kind == "general"
    assert AGENTIC_SMOKE.environments[0].source.package == "automationbench-v1"
    source = AGENTIC_SMOKE.environments[0].source
    assert isinstance(source, EnvironmentSource)
    assert source.repository == ("https://github.com/carbonteq-ai/posttrain")
    assert source.revision == ("02848b756727d86a55564557e79e7f613fc8762c")
    assert source.subdirectory == "environments/automationbench_v1"
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


def test_remote_evaluation_keeps_policy_service_and_endpoint_separate() -> None:
    evaluation = remote_request()

    assert evaluation.endpoint is None
    assert evaluation.resolved_endpoint.base_url == "https://openrouter.ai/api/v1"
    assert evaluation.resolved_endpoint.served_model == "qwen/qwen3.5-2b"
    assert evaluation.remote_service is evaluation.inference.service  # type: ignore[union-attr]
    assert evaluation.resolved_reasoning_mode == "provider-default"


@pytest.mark.parametrize(
    ("headers", "defaults", "match"),
    [
        ({"Authorization": "Bearer secret"}, {}, "must not carry credentials"),
        ({}, {"model": "different"}, "cannot override evaluation-owned fields"),
        ({}, {"tools": []}, "cannot override evaluation-owned fields"),
    ],
)
def test_external_service_rejects_secrets_and_evaluation_owned_request_fields(
    headers: dict[str, str], defaults: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ExternalInferenceService(
            "services/invalid@1",
            "1",
            "https://openrouter.ai/api/v1",
            "OPENROUTER_API_KEY",
            headers,
            defaults,
        )


def test_remote_evaluation_rejects_local_endpoint_and_reasoning_override() -> None:
    evaluation = remote_request()
    with pytest.raises(ValueError, match="resolves its endpoint"):
        replace(
            evaluation,
            endpoint=EvaluationEndpoint("http://127.0.0.1:8000/v1", "wrong"),
        )
    with pytest.raises(ValueError, match="reasoning belongs"):
        replace(evaluation, reasoning_mode="thinking")


def test_remote_binding_maps_to_the_native_verifiers_client_without_a_custom_loop(tmp_path: Path) -> None:
    pytest.importorskip("verifiers.v1")
    from posttrain.eval.backends.verifiers.adapter import _build_native

    evaluation = remote_request()
    environment = GENERAL_SMOKE.environment("math-gsm8k")
    evaluation = replace(
        evaluation,
        plan=replace(evaluation.plan, environments=(environment,)),
        environment_id=environment.id,
    )

    _native_environment, config, _runner = _build_native(evaluation, tmp_path)

    assert config.model == "qwen/qwen3.5-2b"
    assert config.client.type == "eval"
    assert config.client.base_url == "https://openrouter.ai/api/v1"
    assert config.client.api_key_var == "OPENROUTER_API_KEY"
    assert config.client.headers == {"HTTP-Referer": "https://posttrain.example"}
    assert config.sampling.provider == {"allow_fallbacks": False}


def test_remote_evaluation_binding_decodes_from_a_catalog_family() -> None:
    catalog = Catalog.open(
        {
            "layer_id": "remote-eval-test",
            "remote-evaluation": {
                "inference/qwen-via-openrouter-eval@1": {
                    "revision": "1",
                    "policy": {
                        "id": "policies/qwen-via-openrouter@1",
                        "revision": "2026-07-31",
                        "model": "qwen/qwen3.5-2b",
                        "context_window": 32768,
                        "capabilities": {"tools": True},
                    },
                    "service": {
                        "id": "services/openrouter@1",
                        "revision": "1",
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_key_var": "OPENROUTER_API_KEY",
                        "request_defaults": {"provider": {"allow_fallbacks": False}},
                    },
                    "purpose": ["screen", "eval"],
                }
            },
        },
        decoders=evaluation_catalog_decoders(),
    )

    binding = catalog.resolve(CatalogRef("remote-evaluation", "inference/qwen-via-openrouter-eval@1")).value

    assert isinstance(binding, RemoteEvaluationBinding)
    assert binding.service.origin == "https://openrouter.ai"
    assert binding.policy.model == "qwen/qwen3.5-2b"


def test_invocation_budget_can_select_a_small_subset_without_mutating_program() -> None:
    base = request()
    smaller = replace(
        base,
        budget=EvaluationBudget(num_tasks=1, max_concurrent=1, shuffle=True),
    )
    assert smaller.resolved_budget == (1, 1, 1)
    assert smaller.resolved_shuffle
    assert base.environment.num_tasks == 2
    assert not base.resolved_shuffle


def test_request_shuffle_remains_a_compatibility_default_for_budget() -> None:
    base = request()
    assert replace(base, shuffle=True).resolved_shuffle
    assert not replace(base, budget=EvaluationBudget(shuffle=False), shuffle=True).resolved_shuffle
    with pytest.raises(TypeError, match="shuffle override"):
        EvaluationBudget(shuffle="yes")  # type: ignore[arg-type]


def test_evaluate_emits_direct_sync_metrics_and_native_artifact(tmp_path: Path) -> None:
    observer = RecordingObserver()

    def fake_runner(
        execution: RunContext,
        evaluation: EvaluateRequest,
        output: Path,
    ) -> VerifiersRunResult:
        del execution, evaluation
        (output / "config.toml").write_text("model = 'fake'\n", encoding="utf-8")
        (output / "traces.jsonl").write_text('{"id":"trace-1"}\n', encoding="utf-8")
        return VerifiersRunResult(
            ("trace-1",),
            TraceSyncStats(observed_records=1, emitted_records=1),
            EvaluationPopulation(
                attempted=1,
                complete=1,
                failed=0,
                truncated=0,
                coverage_missing=1,
            ),
        )

    result = evaluate(context(tmp_path, observer), request(), runner=fake_runner)

    assert result.trace_ids == ("trace-1",)
    assert result.synchronization.complete
    assert result.status == "partial"
    assert result.native_artifact.reference.path.name == "math"  # type: ignore[union-attr]
    assert observer.artifacts[0].name == result.native_artifact.name
    assert observer.artifacts[0].reference == result.native_artifact.reference
    assert observer.artifacts[0].metadata["work_package_id"] == "qualify/eval"
    values = observer.metrics_log[0].values
    assert values["eval/run/rollouts_attempted"] == 1
    assert values["eval/run/rollouts_complete"] == 1
    assert values["eval/run/rollouts_failed"] == 0
    assert values["eval/run/rollouts_truncated"] == 0
    assert values["eval/run/coverage_missing"] == 1
    assert values["eval/traces_observed"] == 1
    assert "eval/mean_reward" not in values
    assert observer.events[0].attributes["task_selection"] == "head"
    assert observer.events[-1].name == "evaluation_completed"


def test_evaluate_records_shuffled_subset_policy(tmp_path: Path) -> None:
    observer = RecordingObserver()
    evaluation = replace(request(), budget=EvaluationBudget(num_tasks=1, shuffle=True))

    def fake_runner(
        execution: RunContext,
        request_value: EvaluateRequest,
        output: Path,
    ) -> VerifiersRunResult:
        del execution, request_value
        (output / "traces.jsonl").write_text('{"id":"trace-1"}\n', encoding="utf-8")
        return VerifiersRunResult(
            ("trace-1",),
            TraceSyncStats(observed_records=1, emitted_records=1),
            EvaluationPopulation(
                attempted=1,
                complete=1,
                failed=0,
                truncated=0,
                coverage_missing=0,
            ),
        )

    evaluate(context(tmp_path, observer), evaluation, runner=fake_runner)

    assert observer.events[0].attributes["num_tasks"] == 1
    assert observer.events[0].attributes["task_selection"] == "verifiers-fixed-shuffle"
    assert observer.metrics_log[0].attributes["task_selection"] == "verifiers-fixed-shuffle"


def test_general_uses_canonical_seats_and_marks_partial_trace_sync(tmp_path: Path) -> None:
    observer = RecordingObserver()
    canonical = canonical_request()
    run_context = RunContext(
        project_id="foundation-models",
        work_package_id="qualify/qwen-gsm8k",
        run_id="00000000-0000-0000-0000-000000000001",
        job_kind="eval.general",
        job_definition_version="posttrain-eval@0.1.0",
        workspace=tmp_path.resolve(),
        observer=observer,
    )

    def fake_runner(
        execution: RunContext,
        evaluation: EvaluateRequest,
        output: Path,
    ) -> VerifiersRunResult:
        del execution, evaluation
        (output / "traces.jsonl").write_text('{"id":"trace-1"}\n', encoding="utf-8")
        return VerifiersRunResult(
            ("trace-1",),
            TraceSyncStats(observed_records=1, unsynchronized_records=1),
            EvaluationPopulation(
                attempted=1,
                complete=0,
                failed=1,
                truncated=1,
                coverage_missing=1,
            ),
        )

    result = general(run_context, canonical, runner=fake_runner)

    assert result.plan_id == canonical.plan.id
    assert result.model_id == canonical.model.id
    assert result.status == "partial"
    assert result.native_artifact.kind == "verifiers-evaluation"
    values = observer.metrics_log[0].values
    assert values["eval/run/rollouts_attempted"] == 1
    assert values["eval/run/rollouts_complete"] == 0
    assert values["eval/run/rollouts_failed"] == 1
    assert values["eval/run/rollouts_truncated"] == 1
    assert values["eval/run/coverage_missing"] == 1
    assert values["eval/trace_sync_complete"] == 0
    assert "eval/mean_reward" not in values
    attributes = observer.metrics_log[0].attributes
    assert attributes["project_id"] == "foundation-models"
    assert attributes["work_package_id"] == "qualify/qwen-gsm8k"
    assert attributes["evaluation_plan_id"] == canonical.plan.id
    assert attributes["model_variant_id"] == canonical.model.id
    assert observer.events[-1].attributes["evaluation_status"] == "partial"


def test_domain_rejects_a_general_plan(tmp_path: Path) -> None:
    canonical = canonical_request()
    run_context = RunContext(
        project_id="foundation-models",
        work_package_id="qualify/qwen-gsm8k",
        run_id="00000000-0000-0000-0000-000000000001",
        job_kind="eval.domain",
        job_definition_version="posttrain-eval@0.1.0",
        workspace=tmp_path.resolve(),
    )
    with pytest.raises(ValueError, match="domain evaluation plan"):
        domain(run_context, canonical)


def test_program_rejects_unknown_environment() -> None:
    with pytest.raises(ValueError, match="unknown environment"):
        request().plan.environment("missing")


def test_general_program_factories_return_native_configs_when_extra_is_installed() -> None:
    pytest.importorskip("verifiers.v1")
    config: Any = GENERAL_SMOKE.environment("math-gsm8k").activate()
    assert config.taskset.id == "gsm8k-v1"
    assert config.harness.id == "null"
    assert config.timeout.rollout == 180
