"""Tests for standard definitions and default runtime composition."""

from dataclasses import replace
from pathlib import Path

import pytest
from posttrain.catalog import open_catalog
from posttrain.common import CatalogRef, ContractError, ModelVariant, NullObserver, RunContext
from posttrain.data import DatasetLoadPlan, SupervisedDataset
from posttrain.jobs import (
    build_job_runtime,
    sft_definition,
    standard_definitions,
)
from posttrain.train import SFTRequest, SFTSettings, TrainingBinding
from posttrain.work import (
    JobDefinition,
    ProjectBrief,
    ProjectExecutionRequest,
    Recipe,
    RecipeJob,
    ResolvedSeat,
    ServingRequirements,
    WorkPackage,
    validate_work_package,
)


def _selection(catalog, family, selection_id):
    return catalog.resolve(CatalogRef(family, selection_id)).value


def _request(tmp_path: Path) -> ProjectExecutionRequest:
    catalog = open_catalog(scope="jobs-test")
    work_package_path = tmp_path / ".posttrain" / "work_packages" / "sft.yaml"
    work_package_path.parent.mkdir(parents=True)
    work_package_path.write_text("placeholder\n", encoding="utf-8")
    state_dir = tmp_path / ".posttrain" / "state"
    state_dir.mkdir(parents=True)
    return ProjectExecutionRequest(
        project_id="jobs-test",
        project_root=tmp_path.resolve(),
        state_dir=state_dir.resolve(),
        work_package_path=work_package_path.resolve(),
        catalog=catalog,
    )


def test_standard_definition_registry_covers_every_technique() -> None:
    definitions = standard_definitions()

    assert {
        "train/trl-sft@1",
        "train/trl-dpo@1",
        "train/trl-grpo@1",
        "train/trl-sampo@1",
        "train/trl-distill@1",
        "serve/vllm-benchmark@1",
        "serve/vllm-smoke@1",
        "eval/verifiers-general@1",
        "eval/verifiers-managed@1",
        "model/llm-compressor@2",
    } == set(definitions)


def test_runtime_materializes_global_dataset_for_standard_sft_definition(tmp_path: Path) -> None:
    request = _request(tmp_path)
    runtime = build_job_runtime(request, tracking="none")
    plan = _selection(request.catalog, "dataset", "datasets/posttrain-sft-smoke@1")
    assert isinstance(plan, DatasetLoadPlan)
    assert runtime.seat_resolver is not None
    dataset = runtime.seat_resolver(
        ResolvedSeat(
            "dataset",
            plan,
            CatalogRef("dataset", plan.id),
            "base",
        )
    )
    assert isinstance(dataset, SupervisedDataset)

    model = _selection(request.catalog, "model", "models/qwen3.5-2b@bf16")
    settings = _selection(request.catalog, "training", "qwen3.5-2b/sft-smoke-v2")
    training = _selection(request.catalog, "training", "training/qwen3.5-trl-lora@1")
    assert isinstance(model, ModelVariant)
    assert isinstance(settings, SFTSettings)
    assert isinstance(training, TrainingBinding)
    definition = sft_definition(lambda context, value: value)
    context = RunContext(
        project_id="jobs-test",
        work_package_id="train/sft",
        run_id="run-1",
        job_kind="train.sft",
        job_definition_version=definition.id,
        workspace=tmp_path / "workspace",
        observer=NullObserver(),
    )
    result = definition.operation(
        context,
        {
            "model": model,
            "dataset": dataset,
            "settings": settings,
            "training": training,
        },
    )

    assert isinstance(result, SFTRequest)
    assert result.data.descriptor.num_examples == 2


def test_runtime_rejects_shadowing_standard_definition(tmp_path: Path) -> None:
    request = _request(tmp_path)
    standard = standard_definitions()["train/trl-sft@1"]
    shadow = JobDefinition(standard.id, standard.kind, standard.seats, standard.operation)

    with pytest.raises(ValueError, match="cannot shadow standard ids"):
        build_job_runtime(
            request,
            tracking="none",
            extra_definitions={shadow.id: shadow},
        )


def _serving_package() -> WorkPackage:
    return WorkPackage(
        project_id="jobs-test",
        work_package_id="screen/capacity",
        stage="screen",
        recipe=Recipe(
            id="recipes/capacity@1",
            revision="1",
            stage="screen",
            seats={
                "model": "model",
                "screen_inference": "inference",
                "workload": "workload",
                "target": "target",
            },
            jobs=(
                RecipeJob(
                    "benchmark",
                    "serve.benchmark",
                    "serve/vllm-benchmark@1",
                ),
            ),
        ),
        bindings={
            "model": CatalogRef("model", "models/qwen3.5-2b@bf16"),
            "screen_inference": CatalogRef("inference", "inference/qwen3.5-2b-vllm-screen@1"),
            "workload": CatalogRef("workload", "workloads/foundation-smoke-v1@1"),
            "target": CatalogRef("target", "targets/local-cuda-8gb"),
        },
    )


def _brief(required_context_tokens: int) -> ProjectBrief:
    return ProjectBrief(
        objective="Select a serving candidate.",
        serving=ServingRequirements(
            required_context_tokens=required_context_tokens,
            min_sustained_output_tokens_per_second=50,
            max_p95_ttft_ms=1000,
            max_p95_tpot_ms=30,
            max_failure_rate=0.01,
        ),
    )


def test_runtime_preflight_accepts_serving_workload_that_meets_project_context(tmp_path: Path) -> None:
    request = replace(_request(tmp_path), project_brief=_brief(1024))
    runtime = build_job_runtime(request, tracking="none")

    resolved = validate_work_package(runtime, _serving_package())

    assert resolved.definition.work_package_id == "screen/capacity"


def test_runtime_preflight_rejects_serving_workload_below_project_context(tmp_path: Path) -> None:
    request = replace(_request(tmp_path), project_brief=_brief(32_768))
    runtime = build_job_runtime(request, tracking="none")

    with pytest.raises(ContractError, match="below the project serving requirement"):
        validate_work_package(runtime, _serving_package())
