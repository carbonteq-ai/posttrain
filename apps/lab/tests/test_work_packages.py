"""Tests for lab work-package resolution and execution."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from posttrain.common import Catalog, CatalogRef, ContractError
from posttrain.common.variants import QWEN_35_2B
from posttrain.eval import EnvironmentBinding, EvaluateRequest, EvaluationBudget, EvaluationEndpoint
from posttrain.serve import ServeBenchmarkRequest
from posttrain.train import (
    QWEN35_SFT_SMOKE,
    GRPOSettings,
    SFTRequest,
    SFTSettings,
    SFTValidationSettings,
    TrainingLoop,
)
from posttrain_lab.catalog import QWEN35_TRL_QLORA, open_catalog
from posttrain_lab.data import GSM8KSupervisedSource
from posttrain_lab.execution import RunSpec
from posttrain_lab.jobs import GSM8KDistillationJobRequest, VerifiersGRPOJobRequest
from posttrain_lab.work_packages import (
    FOUNDATION_SCREEN_RECIPE,
    Recipe,
    RecipeJob,
    WorkPackage,
    WorkPackageContext,
    distillation_definition,
    general_evaluation_definition,
    grpo_definition,
    load_work_package,
    resolve_work_package,
    run_work_package,
    serve_benchmark_definition,
    sft_definition,
)

WORKSPACE = Path(__file__).resolve().parents[3]
WORK_PACKAGES = WORKSPACE / "apps" / "lab" / ".posttrain" / "work_packages"


def test_reference_yaml_runs_screen_and_skips_optional_eval() -> None:
    package = load_work_package(WORK_PACKAGES / "foundation_screen.yaml")
    resolved_package = resolve_work_package(open_catalog(scope=package.project_id), package)
    seen: list[object] = []
    benchmark = serve_benchmark_definition(lambda context, request: seen.append(request) or "screened")

    result = run_work_package(
        WorkPackageContext(
            open_catalog(scope=package.project_id),
            {benchmark.id: benchmark},
        ),
        package,
    )

    assert [job.status for job in result.jobs] == ["succeeded", "not_run"]
    assert [job.value for job in result.jobs] == ["screened", None]
    assert len(seen) == 1
    assert isinstance(seen[0], ServeBenchmarkRequest)
    assert package.description is not None
    assert "foundation model" in package.description
    assert resolved_package.snapshot["work_package"]["description"] == package.description  # type: ignore[index]
    inference_snapshot = resolved_package.snapshot["screen_inference"]
    assert isinstance(inference_snapshot, dict)
    assert inference_snapshot["resolved"]["engine"]["max_model_len"] == 32_768  # type: ignore[index]


def test_distillation_yaml_resolves_every_seat_through_the_catalog() -> None:
    pytest.importorskip("verifiers")
    package = load_work_package(WORK_PACKAGES / "gsm8k_distillation.yaml")
    catalog = open_catalog(scope=package.project_id)
    resolved = resolve_work_package(catalog, package)
    definition = distillation_definition(
        lambda context, request: request,
        tasks={0: SimpleNamespace(data=SimpleNamespace(prompt="2 + 2"))},
    )

    result = run_work_package(
        WorkPackageContext(catalog, {definition.id: definition}),
        package,
    )

    request = result.jobs[0].value
    assert isinstance(request, GSM8KDistillationJobRequest)
    assert request.environment.id == "gsm8k-distill-train"
    assert request.student.tokenizer_fingerprint == request.teacher.tokenizer_fingerprint
    training = resolved.snapshot["training"]
    assert isinstance(training, dict)
    assert training["resolved"]["parameter_update"] == {  # type: ignore[index]
        "rank": 8,
        "alpha": 16,
        "dropout": 0.0,
        "target_modules": "all-linear",
        "kind": "lora",
    }
    assert training["resolved"]["backend_options"] == {  # type: ignore[index]
        "dependency_lock": "trl-fork@1.9.2.post9",
        "source_revision": "aa82dea49be38838571c388bd7bb530c26c65319",
        "dependency_lock_sha256": hashlib.sha256((WORKSPACE / "uv.lock").read_bytes()).hexdigest(),
        "bf16": False,
        "model_dtype": "float32",
    }
    execution_targets = resolved.snapshot["execution_targets"]
    assert isinstance(execution_targets, dict)
    raw_targets = execution_targets["targets"]
    assert isinstance(raw_targets, list)
    assert all(isinstance(target, dict) for target in raw_targets)
    targets = cast(list[dict[str, object]], raw_targets)
    assert {cast(str, target["selection_id"]) for target in targets} == {"targets/carbonteq-cuda-24gb-plus"}
    assert {role for target in targets for role in cast(list[str], target["roles"])} == {
        "rollout_inference",
        "teacher_inference",
        "training",
    }
    assert all(target["memory_gb"] == 24 for target in targets)
    teacher_inference = resolved.snapshot["teacher_inference"]
    assert isinstance(teacher_inference, dict)
    assert teacher_inference["resolved"]["engine"] == {  # type: ignore[index]
        "base_url": "http://127.0.0.1:8000",
        "gpu_memory_utilization": 0.35,
        "max_model_len": 640,
        "tensor_parallel_size": 1,
        "enforce_eager": True,
    }
    assert set(resolved.snapshot) == {
        "catalog",
        "execution_targets",
        "work_package",
        "recipe",
        "student",
        "teacher",
        "environment",
        "settings",
        "training",
        "rollout_inference",
        "teacher_inference",
    }


def test_verl_sampling_contract_canary_resolves_the_complete_policy() -> None:
    package = load_work_package(WORK_PACKAGES / "qwen08b_verl_sampling_contract_canary.yaml")
    catalog = open_catalog(scope=package.project_id, overlays=(WORKSPACE / "apps/lab/.posttrain/catalog",))
    resolved = resolve_work_package(catalog, package)

    settings = resolved.snapshot["settings"]
    assert isinstance(settings, dict)
    assert settings["resolved"]["max_steps"] == 1  # type: ignore[index]
    assert settings["resolved"]["num_prompts_per_step"] == 1  # type: ignore[index]
    assert settings["resolved"]["num_generations"] == 2  # type: ignore[index]

    training = resolved.snapshot["training"]
    assert isinstance(training, dict)
    assert training["resolved"]["runtime"]["global_batch_size"] == 2  # type: ignore[index]

    environment = resolved.snapshot["environment"]
    assert isinstance(environment, dict)
    assert environment["resolved"]["sampling"] == {  # type: ignore[index]
        "max_tokens": 384,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.01,
        "repetition_penalty": 1.1,
        "presence_penalty": 1.5,
        "reasoning_effort": None,
    }

    rollout = resolved.snapshot["rollout_inference"]
    assert isinstance(rollout, dict)
    assert rollout["resolved"]["sampling"] == {  # type: ignore[index]
        "max_tokens": 384,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.01,
        "repetition_penalty": 1.1,
        "presence_penalty": 1.5,
    }


def test_grpo_definition_accepts_only_an_environment_population_seat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "posttrain.train.verifiers_requests.create_verifiers_training_bridge",
        lambda *_args, **_kwargs: SimpleNamespace(dataset=SimpleNamespace(examples=(object(),))),
    )
    catalog = open_catalog(scope="posttrain-lab")
    definition = grpo_definition(
        lambda context, request: request,
        tasks={
            0: SimpleNamespace(data=SimpleNamespace(prompt="task zero")),
            1: SimpleNamespace(data=SimpleNamespace(prompt="task one")),
        },
    )
    environment_ref = CatalogRef("environment", "automationbench-zapier-simple-grpo")
    environment = catalog.resolve(environment_ref).value
    settings = catalog.resolve(CatalogRef("training", "automationbench/qwen3.5-0.8b/grpo-mtp-smoke-v1")).value
    assert isinstance(environment, EnvironmentBinding)
    assert isinstance(settings, GRPOSettings)
    assert set(definition.seats) == {
        "model",
        "environment",
        "settings",
        "training",
        "rollout_inference",
    }

    package = load_work_package(WORK_PACKAGES / "automationbench_zapier_grpo.yaml")
    resolved = resolve_work_package(catalog, package)
    result = run_work_package(
        WorkPackageContext(catalog, {definition.id: definition}),
        package,
    )

    request = result.jobs[0].value
    assert isinstance(request, VerifiersGRPOJobRequest)
    assert request.environment is environment
    assert "dataset" not in resolved.snapshot
    snapshot = resolved.snapshot["environment"]
    assert isinstance(snapshot, dict)
    assert snapshot["resolved"]["parameters"] == {  # type: ignore[index]
        "domains": ["simple"],
        "sampling_seed": 17,
        "toolset": "zapier",
        "search_top_k": 20,
        "max_turns": 50,
        "max_total_tokens": 8192,
        "rollout_timeout_seconds": 1800,
    }


def test_same_work_package_config_can_enable_the_optional_eval_cell() -> None:
    package = replace(
        load_work_package(WORK_PACKAGES / "foundation_screen.yaml"),
        enabled_optional_jobs=("general-eval",),
    )
    seen: list[object] = []
    benchmark = serve_benchmark_definition(lambda context, request: seen.append(request))
    evaluation = general_evaluation_definition(
        EvaluationEndpoint("http://127.0.0.1:8000/v1", QWEN_35_2B.base.repo_id),
        lambda context, request: seen.append(request),
        context_window=8_192,
        budget=EvaluationBudget(num_tasks=1, max_concurrent=1),
    )

    result = run_work_package(
        WorkPackageContext(
            open_catalog(scope=package.project_id),
            {benchmark.id: benchmark, evaluation.id: evaluation},
        ),
        package,
    )

    assert [job.status for job in result.jobs] == ["succeeded", "succeeded"]
    assert isinstance(seen[0], ServeBenchmarkRequest)
    assert isinstance(seen[1], EvaluateRequest)
    for run in result.jobs:
        assert run.run_id is not None


@pytest.mark.parametrize(
    ("filename", "expected_id"),
    [
        ("gemma4_31b_serve_smoke_qualification.yaml", "inference/gemma4-31b-it-vllm-screen@1"),
        ("gemma4_31b_sft_qualification.yaml", "models/gemma4-31b-it@bf16"),
    ],
)
def test_gemma4_31b_qualification_work_packages_resolve(filename: str, expected_id: str) -> None:
    package = load_work_package(WORK_PACKAGES / filename)
    catalog = open_catalog(scope=package.project_id, overlays=(WORKSPACE / "apps/lab/.posttrain/catalog",))
    resolved = resolve_work_package(catalog, package)

    assert package.work_package_id.endswith("qualification")
    assert expected_id in str(resolved.snapshot)


def test_run_snapshot_carries_work_package_and_job_definition_descriptions() -> None:
    package = load_work_package(WORK_PACKAGES / "foundation_screen.yaml")
    definition = serve_benchmark_definition(lambda context, request: request)
    specs: list[RunSpec] = []

    run_work_package(
        WorkPackageContext(
            open_catalog(scope=package.project_id),
            {definition.id: definition},
            executor=lambda spec, operation: specs.append(spec),
        ),
        package,
    )

    assert len(specs) == 1
    assert specs[0].resolved_inputs["work_package"]["description"] == package.description  # type: ignore[index]
    assert specs[0].resolved_inputs["job_definition"] == {
        "id": definition.id,
        "kind": definition.kind,
        "description": definition.description,
    }


def test_runner_validates_all_enabled_job_seats_before_side_effects() -> None:
    package = load_work_package(WORK_PACKAGES / "foundation_screen.yaml")
    package = replace(
        package,
        bindings={name: binding for name, binding in package.bindings.items() if name != "evaluation_inference"},
        enabled_optional_jobs=("general-eval",),
    )
    calls: list[str] = []
    benchmark = serve_benchmark_definition(lambda context, request: None)
    evaluation = general_evaluation_definition(
        EvaluationEndpoint("http://127.0.0.1:8000/v1", QWEN_35_2B.base.repo_id),
        lambda context, request: None,
        context_window=8_192,
    )

    def executor(spec, operation):
        calls.append(spec.job_kind)
        raise AssertionError("preflight must finish before execution")

    with pytest.raises(ContractError, match="requires unbound seat"):
        run_work_package(
            WorkPackageContext(
                open_catalog(scope=package.project_id),
                {benchmark.id: benchmark, evaluation.id: evaluation},
                executor=executor,
            ),
            package,
        )
    assert calls == []


def test_python_work_package_can_reference_a_catalog_recipe() -> None:
    catalog = Catalog.open(
        {
            "layer_id": "recipe-test",
            "recipe": {FOUNDATION_SCREEN_RECIPE.id: FOUNDATION_SCREEN_RECIPE},
        },
        scope="tests",
    )
    package = WorkPackage(
        project_id="tests",
        work_package_id="screen/catalog-recipe",
        stage="screen",
        recipe=CatalogRef("recipe", FOUNDATION_SCREEN_RECIPE.id),
        bindings={},
    )

    with pytest.raises(ContractError, match="requires unbound seat"):
        benchmark = serve_benchmark_definition(lambda context, request: None)
        run_work_package(
            WorkPackageContext(catalog, {benchmark.id: benchmark}),
            package,
        )


def test_work_package_yaml_rejects_decision_fields(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        """
project_id: tests
work_package_id: screen/invalid
stage: screen
recipe:
  type: inline
  id: recipes/invalid@1
  revision: '1'
  stage: screen
  seats: {model: model}
  jobs:
    - {id: benchmark, kind: serve.benchmark, definition: serve/test@1}
bindings:
  model: {type: ref, family: model, id: models/test}
decision: accept
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="extra_forbidden"):
        load_work_package(path)


def test_sft_job_is_composed_from_explicit_work_package_seats() -> None:
    source = GSM8KSupervisedSource(count=2)
    definition = sft_definition(lambda context, request: request)
    package = WorkPackage(
        project_id="tests",
        work_package_id="train/qwen-sft-smoke",
        stage="train",
        recipe=Recipe(
            id="recipes/sft-smoke@1",
            revision="1",
            stage="train",
            seats={"model": "model", "dataset": "dataset", "settings": "training", "training": "training"},
            jobs=(RecipeJob("sft", "train.sft", definition.id),),
        ),
        bindings={
            "model": QWEN_35_2B,
            "dataset": source,
            "settings": QWEN35_SFT_SMOKE,
            "training": QWEN35_TRL_QLORA,
        },
    )

    result = run_work_package(
        WorkPackageContext(Catalog.open({}, scope="tests"), {definition.id: definition}),
        package,
    )

    request = result.jobs[0].value
    assert isinstance(request, SFTRequest)
    assert request.model == QWEN_35_2B
    assert request.data is source
    assert request.settings == QWEN35_SFT_SMOKE


def test_validated_sft_job_requires_and_forwards_validation_seat() -> None:
    train = GSM8KSupervisedSource(count=2)
    validation = GSM8KSupervisedSource(count=1, offset=2)
    settings = SFTSettings(
        "qwen3.5-2b/sft-validation-test",
        TrainingLoop(max_steps=2),
        validation=SFTValidationSettings(steps=1),
    )
    definition = sft_definition(
        lambda context, request: request,
        definition_id="train/trl-sft-validated-test@1",
        with_validation=True,
    )
    package = WorkPackage(
        project_id="tests",
        work_package_id="train/qwen-sft-validation",
        stage="train",
        recipe=Recipe(
            id="recipes/sft-validation@1",
            revision="1",
            stage="train",
            seats={
                "model": "model",
                "dataset": "dataset",
                "validation_dataset": "dataset",
                "settings": "training",
                "training": "training",
            },
            jobs=(RecipeJob("sft", "train.sft", definition.id),),
        ),
        bindings={
            "model": QWEN_35_2B,
            "dataset": train,
            "validation_dataset": validation,
            "settings": settings,
            "training": QWEN35_TRL_QLORA,
        },
    )

    result = run_work_package(
        WorkPackageContext(Catalog.open({}, scope="tests"), {definition.id: definition}),
        package,
    )

    request = result.jobs[0].value
    assert isinstance(request, SFTRequest)
    assert request.data is train
    assert request.validation_data is validation
