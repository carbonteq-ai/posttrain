"""Tests for standard definitions and default runtime composition."""

from dataclasses import replace
from pathlib import Path

import pytest
from posttrain.catalog import open_catalog
from posttrain.common import (
    CatalogRef,
    ContractError,
    ExecutionTarget,
    ModelVariant,
    NullObserver,
    RunContext,
)
from posttrain.data import (
    DatasetLoadPlan,
    DatasetPrepareRequest,
    PreferenceDataset,
    PreferenceExample,
    SupervisedDataset,
    SupervisedExample,
)
from posttrain.eval import (
    EnvironmentBinding,
    EnvironmentSource,
    EvaluationPlan,
    ExternalInferenceService,
    PythonFactoryActivation,
    RemoteEvaluationBinding,
    RemotePolicy,
    SamplingPolicy,
)
from posttrain.jobs import (
    build_job_runtime,
    grpo_definition,
    preference_data_prepare_definition,
    remote_evaluation_definition,
    sft_definition,
    standard_definitions,
    supervised_data_prepare_definition,
)
from posttrain.train import (
    GRPOSettings,
    SFTRequest,
    SFTSettings,
    TrainingBinding,
    TrainingLoop,
)
from posttrain.work import (
    JobDefinition,
    ProjectBrief,
    ProjectExecutionRequest,
    Recipe,
    RecipeJob,
    ResolvedSeat,
    ServingRequirements,
    WorkPackage,
    prepare_work_package_job,
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
        "data/canonicalize-supervised@1",
        "data/canonicalize-preference@1",
        "train/trl-sft@1",
        "train/trl-dpo@1",
        "train/trl-grpo@1",
        "train/trl-sampo@1",
        "train/trl-distill@1",
        "serve/vllm-benchmark@1",
        "serve/vllm-generation-smoke@1",
        "serve/vllm-smoke@1",
        "eval/verifiers-general@1",
        "eval/verifiers-remote-general@1",
        "eval/verifiers-managed@1",
        "eval/verifiers-managed-general@1",
        "model/llm-compressor@2",
    } == set(definitions)
    assert definitions["eval/verifiers-general@1"].kind == "eval.general"
    assert definitions["eval/verifiers-managed@1"].kind == "eval.domain"
    assert definitions["eval/verifiers-managed-general@1"].kind == "eval.general"
    assert definitions["data/canonicalize-supervised@1"].kind == "data.prepare"
    assert definitions["data/canonicalize-preference@1"].kind == "data.prepare"


def test_remote_evaluation_definition_does_not_construct_a_local_vllm_endpoint(tmp_path: Path) -> None:
    captured = []
    definition = remote_evaluation_definition(lambda context, request: captured.append((context, request)))
    source = EnvironmentSource("fake-env", "https://example.test/environments", "a" * 40)
    environment = EnvironmentBinding(
        "tool-loop",
        "tool-use",
        source,
        PythonFactoryActivation("builtins:object"),
        SamplingPolicy(max_tokens=128),
        num_tasks=1,
    )
    policy = RemotePolicy("policies/example@1", "2026-07-31", "example/model", 8192)
    binding = RemoteEvaluationBinding(
        "inference/example-remote@1",
        "1",
        policy,
        ExternalInferenceService(
            "services/example@1",
            "1",
            "https://api.example.test/v1",
            "EXAMPLE_API_KEY",
        ),
        ("screen", "eval"),
    )
    context = RunContext(
        project_id="jobs-test",
        work_package_id="screen/remote",
        run_id="run-remote",
        job_kind="eval.general",
        job_definition_version=definition.id,
        workspace=(tmp_path / "workspace").resolve(),
        observer=NullObserver(),
    )
    definition.operation(
        context,
        {
            "remote_evaluation": binding,
            "target": ExecutionTarget("targets/external", "1", "network-client"),
            "evaluation_plan": EvaluationPlan("remote-general-v1", "general", (environment,)),
            "environment": environment,
        },
    )

    assert len(captured) == 1
    request = captured[0][1]
    assert request.model is policy
    assert request.inference is binding
    assert request.endpoint is None
    assert request.resolved_endpoint.base_url == "https://api.example.test/v1"


def test_standard_data_prepare_definitions_bind_their_dataset_kinds(
    tmp_path: Path,
) -> None:
    captured: list[DatasetPrepareRequest] = []

    def capture(context, request):
        del context
        captured.append(request)
        return request

    supervised_definition = supervised_data_prepare_definition(capture)
    preference_definition = preference_data_prepare_definition(capture)
    context = RunContext(
        project_id="jobs-test",
        work_package_id="train/prepare",
        run_id="run-prepare",
        job_kind="data.prepare",
        job_definition_version=supervised_definition.id,
        workspace=(tmp_path / "workspace").resolve(),
        observer=NullObserver(),
    )
    supervised = SupervisedDataset(
        "datasets/supervised",
        "1",
        (
            SupervisedExample(
                "example-1",
                (
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ),
                (1,),
            ),
        ),
    )
    preference = PreferenceDataset(
        "datasets/preference",
        "1",
        (
            PreferenceExample(
                "pair-1",
                ({"role": "user", "content": "Question"},),
                ({"role": "assistant", "content": "Better"},),
                ({"role": "assistant", "content": "Worse"},),
            ),
        ),
    )
    target = ExecutionTarget("targets/cpu", "1", "cpu")

    supervised_result = supervised_definition.operation(
        context,
        {"dataset": supervised, "target": target},
    )
    preference_result = preference_definition.operation(
        context,
        {"dataset": preference, "target": target},
    )
    assert isinstance(supervised_result, DatasetPrepareRequest)
    assert isinstance(preference_result, DatasetPrepareRequest)
    assert supervised_result.data is supervised
    assert preference_result.data is preference
    assert [request.data for request in captured] == [supervised, preference]
    for definition in (supervised_definition, preference_definition):
        assert definition.required_artifact_roles == ("dataset",)
        assert definition.selection_seats == {"dataset": DatasetLoadPlan}
        assert definition.seats["target"] is ExecutionTarget


def test_data_prepare_static_validation_rejects_wrong_dataset_kind() -> None:
    supervised = DatasetLoadPlan(
        id="datasets/supervised@1",
        revision="1",
        kind="supervised",
        source={"kind": "fixture", "resource": "supervised.jsonl"},
        format="messages",
    )
    preference = DatasetLoadPlan(
        id="datasets/preference@1",
        revision="1",
        kind="preference",
        source={"kind": "fixture", "resource": "preference.jsonl"},
        format="trl",
    )
    supervised_validator = supervised_data_prepare_definition().static_validator
    preference_validator = preference_data_prepare_definition().static_validator
    assert supervised_validator is not None
    assert preference_validator is not None

    with pytest.raises(ContractError, match="requires a supervised dataset plan"):
        supervised_validator({"dataset": preference})
    with pytest.raises(ContractError, match="requires a preference dataset plan"):
        preference_validator({"dataset": supervised})


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


def test_static_sft_preparation_retains_dataset_plan_without_materializing_it(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    runtime = replace(
        build_job_runtime(request, tracking="none"),
        seat_resolver=None,
    )
    plan = _selection(request.catalog, "dataset", "datasets/posttrain-sft-smoke@1")
    package = WorkPackage(
        project_id="jobs-test",
        work_package_id="train/static-sft",
        stage="train",
        recipe=Recipe(
            id="recipes/static-sft@1",
            revision="1",
            stage="train",
            seats={
                "model": "model",
                "dataset": "dataset",
                "settings": "training",
                "training": "training",
            },
            jobs=(
                RecipeJob(
                    "train",
                    "train.sft",
                    "train/trl-sft@1",
                ),
            ),
        ),
        bindings={
            "model": CatalogRef("model", "models/qwen3.5-2b@bf16"),
            "dataset": CatalogRef("dataset", "datasets/posttrain-sft-smoke@1"),
            "settings": CatalogRef("training", "qwen3.5-2b/sft-smoke-v2"),
            "training": CatalogRef("training", "training/qwen3.5-trl-lora@1"),
        },
    )

    prepared = prepare_work_package_job(runtime, package, "train")

    assert isinstance(plan, DatasetLoadPlan)
    assert prepared.seats["dataset"] is plan
    assert not (request.state_dir / "data").exists()


def test_static_grpo_preparation_rejects_training_batch_mismatch() -> None:
    catalog = open_catalog(scope="jobs-test")
    settings = GRPOSettings(
        id="grpo-static-mismatch",
        loop=TrainingLoop(
            max_steps=1,
            per_device_batch_size=1,
            gradient_accumulation_steps=8,
        ),
        num_prompts_per_step=2,
        num_generations=4,
    )
    training = _selection(
        catalog,
        "training",
        "training/qwen3.5-0.8b-trl-distill-lora@1",
    )
    assert isinstance(settings, GRPOSettings)
    assert isinstance(training, TrainingBinding)
    definition = grpo_definition()
    assert definition.static_validator is not None

    with pytest.raises(
        ContractError,
        match="global batch must equal prompt groups times generations",
    ):
        definition.static_validator(
            {
                "settings": settings,
                "training": training,
            }
        )


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


def test_runtime_validation_does_not_materialize_remote_environment(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    environment = EnvironmentBinding(
        id="environments/remote-only",
        category="qualification",
        source=EnvironmentSource(
            package="remote-only",
            repository="https://example.com/remote-only.git",
            revision="a" * 40,
        ),
        activation=PythonFactoryActivation("remote_environment_that_is_not_installed:create_environment"),
        sampling=SamplingPolicy(max_tokens=64),
        num_tasks=1,
    )
    definition = JobDefinition(
        "eval/remote-only@1",
        "eval.general",
        {"environment": EnvironmentBinding},
        lambda context, seats: None,
    )
    runtime = build_job_runtime(
        request,
        tracking="none",
        extra_definitions={definition.id: definition},
    )
    package = WorkPackage(
        project_id="jobs-test",
        work_package_id="qualify/remote-only",
        stage="qualify",
        recipe=Recipe(
            id="recipes/remote-only@1",
            revision="1",
            stage="qualify",
            seats={"environment": "environment"},
            jobs=(
                RecipeJob(
                    "evaluate",
                    "eval.general",
                    definition.id,
                ),
            ),
        ),
        bindings={"environment": environment},
    )

    resolved = validate_work_package(runtime, package)

    assert resolved.seat("environment", EnvironmentBinding) is environment


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
