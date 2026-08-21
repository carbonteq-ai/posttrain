"""Tests for reusable work-package composition."""

from __future__ import annotations

from pathlib import Path

import pytest
from posttrain.common import (
    Catalog,
    CatalogLayer,
    CatalogRef,
    ContractError,
    ExecutionTarget,
    InferenceBinding,
    PublishedArtifact,
    StoredArtifactRef,
)
from posttrain.common.variants import QWEN_35_2B
from posttrain.environment import (
    EnvironmentBinding,
    EnvironmentSource,
    EvaluationFacetField,
    EvaluationObservation,
    SamplingPolicy,
    VerifiersV1ConfigActivation,
)
from posttrain.eval import (
    EvaluationBreakdownDefinition,
    EvaluationNumericPredicate,
    EvaluationPlan,
    EvaluationSignalRef,
    EvaluationSuccessDefinition,
)
from posttrain.train import ActiveGroupSampling, GRPOSettings, SFTSettings, TrainingLoop
from posttrain.work import (
    FinalizedRunResult,
    JobDefinition,
    ProjectBrief,
    RecipeJob,
    RunSpec,
    WorkPackageContext,
    load_work_package,
    prepare_work_package_job,
    run_work_package,
    run_work_package_job,
    validate_work_package,
)
from posttrain.work.runner import _evaluation_contract_snapshot, _preflight_job, _selection_details


def _fixture(path: Path) -> None:
    path.write_text(
        """
project_id: example
work_package_id: screen/cpu-check
stage: screen
description: Confirm that the selected CPU target resolves and executes.
recipe:
  type: inline
  id: recipes/cpu-check@1
  revision: "1"
  stage: screen
  seats:
    target: target
  jobs:
    - id: check
      kind: data.prepare
      definition: data/cpu-check@1
bindings:
  target:
    type: ref
    family: target
    id: targets/cpu
enabled_optional_jobs: []
metadata:
  question: Does reusable composition work outside the lab?
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_preflight_and_execution_are_available_without_lab(tmp_path: Path) -> None:
    path = tmp_path / "cpu-check.yaml"
    _fixture(path)
    target = ExecutionTarget("targets/cpu", "1", "cpu")
    catalog = Catalog(CatalogLayer("framework-v1", {CatalogRef("target", target.id): target}), (), "example")
    seen: list[str] = []
    definition = JobDefinition(
        "data/cpu-check@1",
        "data.prepare",
        {"target": ExecutionTarget},
        lambda context, seats: seen.append(seats["target"].id) or "checked",
        "Check one resolved CPU target.",
    )
    context = WorkPackageContext(catalog, {definition.id: definition})
    package = load_work_package(path)

    resolved = validate_work_package(context, package)
    assert seen == []
    assert resolved.seat("target", ExecutionTarget) is target

    result = run_work_package(context, package)
    assert seen == ["targets/cpu"]
    assert result.jobs[0].status == "succeeded"
    assert result.jobs[0].value == "checked"


def test_selected_job_preserves_preallocated_run_id(tmp_path: Path) -> None:
    path = tmp_path / "cpu-check.yaml"
    _fixture(path)
    target = ExecutionTarget("targets/cpu", "1", "cpu")
    catalog = Catalog(
        CatalogLayer(
            "framework-v1",
            {CatalogRef("target", target.id): target},
        ),
        (),
        "example",
    )
    definition = JobDefinition(
        "data/cpu-check@1",
        "data.prepare",
        {"target": ExecutionTarget},
        lambda context, seats: context.run_id,
    )
    specs: list[RunSpec] = []
    context = WorkPackageContext(
        catalog,
        {definition.id: definition},
        executor=lambda spec, operation: specs.append(spec) or operation,
    )

    result = run_work_package_job(
        context,
        load_work_package(path),
        "check",
        run_id="run-preallocated-1",
    )

    assert specs[0].run_id == "run-preallocated-1"
    assert result.jobs[0].run_id == "run-preallocated-1"


def test_selected_job_can_be_prepared_without_execution(tmp_path: Path) -> None:
    path = tmp_path / "cpu-check.yaml"
    _fixture(path)
    target = ExecutionTarget("targets/cpu", "1", "cpu")
    catalog = Catalog(
        CatalogLayer(
            "framework-v1",
            {CatalogRef("target", target.id): target},
        ),
        (),
        "example",
    )
    seen: list[str] = []
    definition = JobDefinition(
        "data/cpu-check@1",
        "data.prepare",
        {"target": ExecutionTarget},
        lambda context, seats: seen.append(context.run_id),
    )
    context = WorkPackageContext(catalog, {definition.id: definition})

    prepared = prepare_work_package_job(
        context,
        load_work_package(path),
        "check",
        run_id="run-planned-1",
    )

    assert seen == []
    assert prepared.recipe_job.id == "check"
    assert prepared.definition.id == "data/cpu-check@1"
    assert prepared.spec.run_id == "run-planned-1"
    assert prepared.seats["target"] is target


def test_run_snapshot_includes_project_brief_and_digest(tmp_path: Path) -> None:
    path = tmp_path / "cpu-check.yaml"
    _fixture(path)
    target = ExecutionTarget("targets/cpu", "1", "cpu")
    catalog = Catalog(CatalogLayer("framework-v1", {CatalogRef("target", target.id): target}), (), "example")
    definition = JobDefinition(
        "data/cpu-check@1",
        "data.prepare",
        {"target": ExecutionTarget},
        lambda context, seats: "checked",
    )
    specs: list[RunSpec] = []
    brief = ProjectBrief(objective="Confirm project-brief snapshot propagation.")
    context = WorkPackageContext(
        catalog,
        {definition.id: definition},
        project_brief=brief,
        executor=lambda spec, operation: specs.append(spec) or "captured",
    )

    run_work_package(context, load_work_package(path))

    assert len(specs) == 1
    snapshot = specs[0].resolved_inputs["project_brief"]
    assert snapshot["objective"] == brief.objective  # type: ignore[index]
    assert isinstance(snapshot["digest"], str)  # type: ignore[index]


def test_inference_snapshot_includes_startup_timeout() -> None:
    target = ExecutionTarget("targets/local-cuda-8gb", "1", "nvidia-cuda", memory_gb=8)
    binding = InferenceBinding(
        id="inference/qwen3.5-2b-vllm-eval@2",
        revision="2",
        model=QWEN_35_2B,
        backend="vllm@0.25.1",
        renderer=QWEN_35_2B.renderer.id,
        engine={},
        sampling={},
        target=target,
        purpose=("eval",),
        startup_timeout_seconds=600,
    )

    assert _selection_details(binding)["startup_timeout_seconds"] == 600


def test_grpo_snapshot_retains_algorithm_and_active_sampling_contract() -> None:
    settings = GRPOSettings(
        "training/olmo3@1",
        TrainingLoop(max_steps=10, per_device_batch_size=2, gradient_accumulation_steps=64),
        num_prompts_per_step=32,
        num_generations=4,
        algorithm="olmo3",
        beta=0,
        advantage_scaling="none",
        importance_sampling_mode="token_truncate",
        importance_sampling_clip_min=None,
        importance_sampling_clip_max=2,
        active_sampling=ActiveGroupSampling(max_candidate_batches=10),
    )

    snapshot = _selection_details(settings)

    assert snapshot["algorithm"] == "olmo3"
    assert snapshot["num_prompts_per_step"] == 32
    assert snapshot["num_generations"] == 4
    assert snapshot["active_sampling"] == {"max_candidate_batches": 10}
    assert snapshot["advantage_scaling"] == "none"
    assert snapshot["clip_epsilon_high"] == pytest.approx(0.272)


def test_training_snapshot_retains_optimizer_contract() -> None:
    settings = SFTSettings(
        "training/sft-optimizer-contract@1",
        TrainingLoop(
            max_steps=10,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            max_grad_norm=0.75,
        ),
    )

    snapshot = _selection_details(settings)

    assert snapshot["warmup_ratio"] == 0.03
    assert snapshot["lr_scheduler_type"] == "cosine"
    assert snapshot["weight_decay"] == 0.01
    assert snapshot["max_grad_norm"] == 0.75


def test_detached_preflight_rejects_tool_environment_with_plain_inference() -> None:
    target = ExecutionTarget("targets/local-cuda-8gb", "1", "nvidia-cuda", memory_gb=8)
    inference = InferenceBinding(
        id="inference/qwen3.5-2b-vllm-eval@2",
        revision="2",
        model=QWEN_35_2B,
        backend="vllm@0.25.1",
        renderer=QWEN_35_2B.renderer.id,
        engine={},
        sampling={},
        target=target,
        purpose=("eval",),
    )
    environment = EnvironmentBinding(
        id="env/tool-loop",
        category="tool-use",
        source=EnvironmentSource(
            package="test-env",
            repository="https://example.com/test-env",
            revision="a" * 40,
        ),
        activation=VerifiersV1ConfigActivation({"taskset": {"id": "test-v1"}}),
        sampling=SamplingPolicy(max_tokens=128),
        num_tasks=1,
        required_inference_capabilities=("tool-calling",),
    )
    job = RecipeJob("evaluate", "eval.general", "eval/verifiers-managed@1")

    with pytest.raises(ContractError, match="missing environment capabilities: tool-calling"):
        _preflight_job(None, job, {"environment": environment, "evaluation_inference": inference})


def test_evaluation_contract_snapshot_is_versioned_and_complete() -> None:
    environment = EnvironmentBinding(
        id="env/test",
        category="math",
        source=EnvironmentSource(
            package="test-env",
            repository="https://example.com/test-env",
            revision="a" * 40,
            subdirectory="environments/test",
        ),
        activation=VerifiersV1ConfigActivation({"taskset": {"id": "test-v1"}}),
        sampling=SamplingPolicy(max_tokens=128),
        num_tasks=2,
        reward_components=("correct",),
        observation=EvaluationObservation(
            facets=(
                EvaluationFacetField("topic", "topic", "Topic"),
                EvaluationFacetField("difficulty", "difficulty", "Difficulty"),
            )
        ),
    )
    plan = EvaluationPlan(
        id="eval/test-v1",
        kind="general",
        environments=(environment,),
        success={
            "env/test": EvaluationSuccessDefinition(
                "correct",
                "Correct",
                EvaluationSignalRef("reward", "correct"),
                EvaluationNumericPredicate("eq", 1.0),
            )
        },
        breakdowns={
            "env/test": (
                EvaluationBreakdownDefinition(
                    "topic-by-difficulty",
                    "Topic × difficulty",
                    ("topic", "difficulty"),
                ),
            )
        },
        aggregation={"slice_weighting": "micro"},
        comparison={"population": "same"},
    )

    snapshot = _evaluation_contract_snapshot(plan, environment)

    assert snapshot["contract"] == {"id": "posttrain.eval.verifiers-observation", "schema_version": 3}
    assert snapshot["plan"]["success"]["source"] == {"namespace": "reward", "name": "correct"}  # type: ignore[index]
    assert snapshot["plan"]["aggregation"] == {"slice_weighting": "micro"}  # type: ignore[index]
    assert snapshot["plan"]["breakdowns"] == [  # type: ignore[index]
        {
            "id": "topic-by-difficulty",
            "label": "Topic × difficulty",
            "dimensions": ["topic", "difficulty"],
            "presentation": "matrix",
            "multi_value": "reject",
            "missing": "exclude",
        }
    ]
    assert snapshot["signal_manifest"]["reward_components"] == ["correct"]  # type: ignore[index]
    assert snapshot["native_evidence"] == {"schema_id": "verifiers.trace", "schema_version": "v1"}
    assert snapshot["population"]["taskset"] == {"id": "test-v1"}  # type: ignore[index]
    assert snapshot["population"]["dataset"] == {"id": None, "revision": None, "split": None}  # type: ignore[index]


def test_work_package_exposes_published_artifacts_without_wrapping_value(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cpu-check.yaml"
    _fixture(path)
    target = ExecutionTarget("targets/cpu", "1", "cpu")
    catalog = Catalog(
        CatalogLayer(
            "framework-v1",
            {CatalogRef("target", target.id): target},
        ),
        (),
        "example",
    )
    definition = JobDefinition(
        "data/cpu-check@1",
        "data.prepare",
        {"target": ExecutionTarget},
        lambda context, seats: "checked",
    )
    published = PublishedArtifact(
        "model/final",
        "model",
        StoredArtifactRef(
            "trackio",
            "example",
            "model-final",
            "v0",
            "a" * 64,
        ),
    )
    context = WorkPackageContext(
        catalog,
        {definition.id: definition},
        executor=lambda spec, operation: FinalizedRunResult(
            "checked",
            (published,),
        ),
    )

    result = run_work_package(context, load_work_package(path))

    assert result.jobs[0].value == "checked"
    assert result.jobs[0].published_artifacts == (published,)
