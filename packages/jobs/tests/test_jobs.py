"""Tests for standard definitions and default runtime composition."""

import re
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from posttrain.catalog import open_catalog
from posttrain.common import (
    CatalogRef,
    ContractError,
    ExecutionTarget,
    InferenceBinding,
    JsonValue,
    LocalArtifactRef,
    ModelVariant,
    NullObserver,
    RunContext,
    StoredArtifactRef,
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
    EvaluationBudget,
    EvaluationNumericPredicate,
    EvaluationPlan,
    EvaluationSignalRef,
    EvaluationSuccessDefinition,
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
from posttrain.jobs.definitions import (
    _judge_service_bindings,
    _materialize_grpo_policy,
    _materialize_selected_model_variant,
    _validate_task_supply,
    _validate_verl_training_loop_seats,
    distillation_definition,
    sampo_definition,
    structured_rl_definition,
)
from posttrain.train import (
    AdaptiveCurriculum,
    GRPOSettings,
    SAMPOSettings,
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
        "train/grpo-family-judged@1",
        "train/trl-sampo@1",
        "train/sampo-turns@1",
        "train/gdpo@1",
        "train/capo@1",
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
    assert definitions["train/grpo-family-judged@1"].kind == "train.grpo"
    assert "judge_inference" in definitions["train/grpo-family-judged@1"].seats


def test_remote_evaluation_definition_does_not_construct_a_local_vllm_endpoint(tmp_path: Path) -> None:
    captured = []
    definition = remote_evaluation_definition(
        lambda context, request: captured.append((context, request)),
        budget=EvaluationBudget(num_tasks=1, shuffle=True),
    )
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
            "evaluation_plan": EvaluationPlan(
                "remote-general-v1",
                "general",
                (environment,),
                success={
                    "tool-loop": EvaluationSuccessDefinition(
                        "task-success",
                        "Task success",
                        EvaluationSignalRef("reward", "reward"),
                        EvaluationNumericPredicate("eq", 1.0),
                    )
                },
            ),
            "environment": environment,
        },
    )

    assert len(captured) == 1
    request = captured[0][1]
    assert request.model is policy
    assert request.inference is binding
    assert request.endpoint is None
    assert request.resolved_endpoint.base_url == "https://api.example.test/v1"
    assert request.resolved_budget == (1, 1, 4)
    assert request.resolved_shuffle


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


def test_standard_training_definition_forwards_materialized_recovery_checkpoint(tmp_path: Path) -> None:
    request = _request(tmp_path)
    runtime = build_job_runtime(request, tracking="none")
    assert runtime.seat_resolver is not None
    plan = _selection(request.catalog, "dataset", "datasets/posttrain-sft-smoke@1")
    dataset = runtime.seat_resolver(ResolvedSeat("dataset", plan, CatalogRef("dataset", plan.id), "base"))
    model = _selection(request.catalog, "model", "models/qwen3.5-2b@bf16")
    settings = _selection(request.catalog, "training", "qwen3.5-2b/sft-smoke-v2")
    training = _selection(request.catalog, "training", "training/qwen3.5-trl-lora@1")
    recovery = (tmp_path / "checkpoint-1").resolve()
    recovery.mkdir()
    reference = LocalArtifactRef(recovery, "a" * 64)
    definition = sft_definition(lambda context, value: value)
    context = RunContext(
        project_id="jobs-test",
        work_package_id="train/sft-resume",
        run_id="run-resume",
        job_kind="train.sft",
        job_definition_version=definition.id,
        workspace=(tmp_path / "workspace-resume").resolve(),
        observer=NullObserver(),
        input_artifacts={"recovery_checkpoint": reference},
    )

    result = definition.operation(
        context,
        {"model": model, "dataset": dataset, "settings": settings, "training": training},
    )

    assert isinstance(result, SFTRequest)
    assert result.resume_from == reference


def test_training_definition_consumes_a_checkpoint_model_view_for_a_fresh_branch(tmp_path: Path) -> None:
    request = _request(tmp_path)
    runtime = build_job_runtime(request, tracking="none")
    assert runtime.seat_resolver is not None
    dataset_plan = _selection(request.catalog, "dataset", "datasets/posttrain-sft-smoke@1")
    dataset = runtime.seat_resolver(
        ResolvedSeat("dataset", dataset_plan, CatalogRef("dataset", dataset_plan.id), "base")
    )
    model = cast(ModelVariant, _selection(request.catalog, "model", "models/qwen3.5-2b@bf16"))
    settings = cast(SFTSettings, _selection(request.catalog, "training", "qwen3.5-2b/sft-smoke-v2"))
    training = cast(TrainingBinding, _selection(request.catalog, "training", "training/qwen3.5-trl-lora@1"))
    adapter_path = (tmp_path / "model-adapter").resolve()
    adapter_path.mkdir()
    adapter = LocalArtifactRef(adapter_path, "c" * 64)
    definition = sft_definition(lambda context, value: value)
    context = RunContext(
        project_id="jobs-test",
        work_package_id="train/sft-branch",
        run_id="run-branch",
        job_kind="train.sft",
        job_definition_version=definition.id,
        workspace=(tmp_path / "workspace-branch").resolve(),
        observer=NullObserver(),
        input_artifacts={"model_adapter": adapter},
    )

    result = definition.operation(
        context,
        {"model": model, "dataset": dataset, "settings": settings, "training": training},
    )

    assert isinstance(result, SFTRequest)
    assert result.resume_from is None
    assert result.model.artifact is adapter
    assert result.model.form == "adapter"


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


def test_static_sampo_preparation_rejects_training_batch_mismatch() -> None:
    catalog = open_catalog(scope="jobs-test")
    settings = SAMPOSettings(
        id="sampo-static-mismatch",
        loop=TrainingLoop(max_steps=1, max_length=384, per_device_batch_size=1, gradient_accumulation_steps=8),
        num_prompts_per_step=2,
        num_generations=4,
    )
    training = _selection(catalog, "training", "training/qwen3.5-0.8b-trl-distill-lora@1")
    assert isinstance(training, TrainingBinding)
    definition = sampo_definition()
    assert definition.static_validator is not None

    with pytest.raises(ContractError, match="global batch must equal prompt groups times generations"):
        definition.static_validator({"settings": settings, "training": training})


def test_static_preparation_rejects_curriculum_field_the_environment_does_not_declare() -> None:
    from types import SimpleNamespace

    from posttrain.train.profiles import ActiveGroupSampling, AdaptiveCurriculum

    settings = SAMPOSettings(
        id="sampo-curriculum-facet",
        loop=TrainingLoop(max_steps=1, max_length=384, per_device_batch_size=1, gradient_accumulation_steps=8),
        num_prompts_per_step=2,
        num_generations=4,
        active_sampling=ActiveGroupSampling(3),
        adaptive_curriculum=AdaptiveCurriculum(class_field="domain", policy="yield_first", seed=1),
    )
    definition = sampo_definition()
    assert definition.static_validator is not None
    undeclared = SimpleNamespace(observation=SimpleNamespace(facets=()))

    with pytest.raises(ContractError, match="class field 'domain' must be an observation facet"):
        _validate_task_supply(settings, undeclared, None)


def test_static_preparation_rejects_a_candidate_pool_larger_than_the_environment() -> None:
    from types import SimpleNamespace

    from posttrain.train.profiles import ActiveGroupSampling

    settings = SAMPOSettings(
        id="sampo-candidate-pool",
        loop=TrainingLoop(max_steps=1, max_length=384, per_device_batch_size=1, gradient_accumulation_steps=8),
        num_prompts_per_step=2,
        num_generations=4,
        active_sampling=ActiveGroupSampling(10),
    )
    definition = sampo_definition()
    assert definition.static_validator is not None
    catalog = open_catalog(scope="jobs-test")
    training = _selection(catalog, "training", "training/qwen3.5-0.8b-trl-distill-lora@1")
    assert isinstance(training, TrainingBinding) and training.backend.startswith("trl@")
    training = replace(training, runtime=replace(training.runtime, global_batch_size=8))
    small = SimpleNamespace(num_tasks=15, observation=None)

    with pytest.raises(ContractError, match="15 tasks but each update reserves 20 candidate prompts"):
        _validate_task_supply(settings, small, training)


def _oversampled_sampo_seats(
    *,
    oversample: int = 1,
    oversample_refill: int = 0,
    max_num_seqs: int | None = 12,
    max_concurrent: int = 12,
    worker_slots: tuple[int, int] | None = (3, 4),
) -> dict[str, object]:
    """Two prompts x four generations; one oversampled group makes a 12-episode first round."""

    from posttrain.train.profiles import ActiveGroupSampling

    catalog = open_catalog(scope="jobs-test")
    model = cast(ModelVariant, _selection(catalog, "model", "models/qwen3.5-2b@bf16"))
    settings = SAMPOSettings(
        id="sampo-oversample",
        loop=TrainingLoop(max_steps=1, max_length=384, per_device_batch_size=1, gradient_accumulation_steps=8),
        num_prompts_per_step=2,
        num_generations=4,
        active_sampling=ActiveGroupSampling(3, oversample=oversample, oversample_refill=oversample_refill),
    )
    base_training = _selection(catalog, "training", "training/qwen3.5-0.8b-trl-distill-lora@1")
    assert isinstance(base_training, TrainingBinding) and base_training.backend.startswith("trl@")
    options = dict(base_training.backend_options)
    if worker_slots is not None:
        options["rollout_execution"] = {
            "env_workers": worker_slots[0],
            "episodes_per_worker": worker_slots[1],
            "worker_native_threads": 1,
        }
    training = replace(
        base_training,
        runtime=replace(base_training.runtime, global_batch_size=8),
        backend_options=options,
    )
    environment = EnvironmentBinding(
        "environments/static-oversample",
        "tool-use",
        EnvironmentSource("static", "https://example.test/static", "b" * 40),
        PythonFactoryActivation("builtins:object"),
        SamplingPolicy(max_tokens=128, temperature=0.7),
        num_tasks=6,
        max_concurrent=max_concurrent,
    )
    engine: dict[str, JsonValue] = {"max_model_len": 4096}
    if max_num_seqs is not None:
        engine["max_num_seqs"] = max_num_seqs
    inference = InferenceBinding(
        "inference/static-oversample@1",
        "1",
        model,
        "vllm@0.25.1",
        model.renderer_contract,
        engine,
        {"max_tokens": 128, "temperature": 0.7},
        ExecutionTarget("targets/static", "1", "nvidia-cuda"),
        ("rollout",),
    )
    return {"settings": settings, "training": training, "environment": environment, "rollout_inference": inference}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"max_num_seqs": 8}, "rollout inference engine max_num_seqs is 8"),
        # Without a declared limit TRL admits one generation batch (2 x 4) per process.
        ({"max_num_seqs": None}, "rollout inference engine max_num_seqs is 8"),
        ({"max_concurrent": 8, "worker_slots": (2, 4)}, "environment max_concurrent is 8"),
        ({"worker_slots": (2, 4)}, "env_workers x episodes_per_worker is 2 x 4 = 8"),
    ],
)
def test_static_preparation_rejects_an_oversampled_round_beyond_rollout_concurrency(changes, message) -> None:
    validator = sampo_definition().static_validator
    assert validator is not None
    validator(_oversampled_sampo_seats())  # type: ignore[arg-type]

    with pytest.raises(ContractError, match=f"oversample 1 needs 12 concurrent episodes.*{message}"):
        validator(_oversampled_sampo_seats(**changes))  # type: ignore[arg-type]


def test_static_preparation_bounds_concurrency_by_first_round_oversampling_only() -> None:
    validator = sampo_definition().static_validator
    assert validator is not None
    exact = {"max_num_seqs": 8, "max_concurrent": 8, "worker_slots": (2, 4)}
    validator(_oversampled_sampo_seats(oversample=0, **exact))  # type: ignore[arg-type]
    # Refill rounds never exceed the first round, so refill oversampling needs no extra concurrency.
    validator(_oversampled_sampo_seats(oversample=0, oversample_refill=5, **exact))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("form", "kl_reference", "message"),
    [
        ("full-finetuned", "base", "veRL uses the starting checkpoint as the KL reference"),
        ("adapter", "start", "cannot hold a frozen copy of the starting adapter"),
    ],
)
def test_static_preparation_rejects_a_kl_reference_verl_cannot_provide(form, kl_reference, message) -> None:
    validator = sampo_definition().static_validator
    assert validator is not None
    seats = _oversampled_sampo_seats(oversample=0)
    settings = cast(SAMPOSettings, seats["settings"])
    training = cast(TrainingBinding, seats["training"])
    inference = cast(InferenceBinding, seats["rollout_inference"])
    # A constant learning rate keeps the loop representable on veRL, so the KL check is reached.
    seats["settings"] = replace(
        settings, beta=0.01, kl_reference=kl_reference, loop=replace(settings.loop, lr_scheduler_type="constant")
    )
    seats["model"] = inference.model
    validator(seats)  # type: ignore[arg-type]
    seats["model"] = replace(inference.model, form=form)
    validator(seats)  # type: ignore[arg-type]  # TRL provides either reference

    seats["training"] = replace(training, backend="verl@candidate")
    with pytest.raises(ContractError, match=message):
        validator(seats)  # type: ignore[arg-type]


def test_static_grpo_preparation_rejects_sampling_policy_mismatch() -> None:
    catalog = open_catalog(scope="jobs-test")
    model = cast(ModelVariant, _selection(catalog, "model", "models/qwen3.5-2b@bf16"))
    settings = GRPOSettings(
        id="grpo-static-sampling-mismatch",
        loop=TrainingLoop(max_steps=1, per_device_batch_size=1, gradient_accumulation_steps=8),
        num_prompts_per_step=2,
        num_generations=4,
        max_completion_length=128,
    )
    base_training = _selection(catalog, "training", "training/qwen3.5-0.8b-trl-distill-lora@1")
    assert isinstance(base_training, TrainingBinding)
    training = replace(base_training, runtime=replace(base_training.runtime, global_batch_size=8))
    environment = EnvironmentBinding(
        "environments/static-sampling",
        "tool-use",
        EnvironmentSource("static", "https://example.test/static", "b" * 40),
        PythonFactoryActivation("builtins:object"),
        SamplingPolicy(max_tokens=128, temperature=1.0),
        num_tasks=1,
    )
    inference = InferenceBinding(
        "inference/static-sampling@1",
        "1",
        model,
        "vllm@0.25.1",
        model.renderer_contract,
        {"max_model_len": 4096},
        {"max_tokens": 128, "temperature": 0.7, "top_p": 1.0},
        ExecutionTarget("targets/static", "1", "nvidia-cuda"),
        ("rollout",),
    )

    with pytest.raises(ContractError, match="online-RL sampling policy is inconsistent"):
        grpo_definition().static_validator(  # type: ignore[misc]
            {
                "settings": settings,
                "training": training,
                "environment": environment,
                "rollout_inference": inference,
            }
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {
                "algorithm": "dapo",
                "clip_epsilon_high": 0.28,
                "adaptive_curriculum": AdaptiveCurriculum(class_field="category"),
            },
            "adaptive_curriculum with DAPO is currently supported by the TRL backend only",
        ),
        ({"advantage_scaling": "batch"}, "advantage_scaling='batch' is currently supported by the TRL backend only"),
        ({"importance_sampling_clip_max": 2.0}, "importance_sampling_clip_max=2.0"),
        ({"max_admission_attempts": 1}, "max_admission_attempts=1"),
    ],
)
def test_static_grpo_preparation_rejects_settings_verl_would_ignore(changes: dict[str, object], message: str) -> None:
    catalog = open_catalog(scope="jobs-test")
    model = cast(ModelVariant, _selection(catalog, "model", "models/qwen3.5-2b@bf16"))
    settings = GRPOSettings(
        id="grpo-static-verl-unsupported",
        loop=TrainingLoop(max_steps=1, per_device_batch_size=1, gradient_accumulation_steps=8),
        num_prompts_per_step=2,
        num_generations=4,
        max_completion_length=128,
    )
    base_training = _selection(catalog, "training", "training/qwen3.5-0.8b-trl-distill-lora@1")
    assert isinstance(base_training, TrainingBinding)
    training = replace(base_training, runtime=replace(base_training.runtime, global_batch_size=8))
    environment = EnvironmentBinding(
        "environments/static-verl-unsupported",
        "tool-use",
        EnvironmentSource("static", "https://example.test/static", "b" * 40),
        PythonFactoryActivation("builtins:object"),
        SamplingPolicy(max_tokens=128, temperature=1.0),
        num_tasks=1,
    )
    inference = InferenceBinding(
        "inference/static-verl-unsupported@1",
        "1",
        model,
        "vllm@0.25.1",
        model.renderer_contract,
        {"max_model_len": 4096},
        {"max_tokens": 128, "temperature": 1.0, "top_p": 1.0},
        ExecutionTarget("targets/static", "1", "nvidia-cuda"),
        ("rollout",),
    )
    seats = {
        "settings": replace(settings, **changes),  # type: ignore[arg-type]
        "training": replace(training, backend="verl@candidate"),
        "environment": environment,
        "rollout_inference": inference,
    }

    with pytest.raises(ContractError, match=re.escape(message)):
        grpo_definition().static_validator(seats)  # type: ignore[misc,arg-type]


def test_static_preparation_rejects_a_training_loop_verl_cannot_run() -> None:
    validator = sampo_definition().static_validator
    assert validator is not None
    seats = _oversampled_sampo_seats(oversample=0)
    settings = cast(SAMPOSettings, seats["settings"])
    training = cast(TrainingBinding, seats["training"])
    seats["settings"] = replace(settings, loop=replace(settings.loop, lr_scheduler_type="linear"))
    validator(seats)  # type: ignore[arg-type]  # TRL runs a linear schedule

    seats["training"] = replace(training, backend="verl@candidate")
    with pytest.raises(ContractError, match="lr_scheduler_type 'linear' is not available on the veRL backend"):
        validator(seats)  # type: ignore[arg-type]
    seats["settings"] = replace(settings, loop=replace(settings.loop, lr_scheduler_type="constant", logging_steps=5))
    with pytest.raises(ContractError, match="logging_steps 5 is not available on the veRL backend"):
        validator(seats)  # type: ignore[arg-type]


def test_structured_and_distillation_preparation_check_the_verl_training_loop() -> None:
    seats = _oversampled_sampo_seats(oversample=0)
    settings = cast(SAMPOSettings, seats["settings"])
    training = cast(TrainingBinding, seats["training"])
    for definition in (distillation_definition(), structured_rl_definition("gdpo"), structured_rl_definition("capo")):
        assert definition.static_validator is _validate_verl_training_loop_seats
    linear = {"settings": replace(settings, loop=replace(settings.loop, lr_scheduler_type="linear"))}
    _validate_verl_training_loop_seats({**linear, "training": training})  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="not available on the veRL backend"):
        _validate_verl_training_loop_seats(
            {**linear, "training": replace(training, backend="verl@candidate")}  # type: ignore[arg-type]
        )


def test_static_grpo_preparation_rejects_inference_completion_length_mismatch() -> None:
    catalog = open_catalog(scope="jobs-test")
    model = cast(ModelVariant, _selection(catalog, "model", "models/qwen3.5-2b@bf16"))
    settings = GRPOSettings(
        id="grpo-static-inference-length-mismatch",
        loop=TrainingLoop(max_steps=1, per_device_batch_size=1, gradient_accumulation_steps=8),
        num_prompts_per_step=2,
        num_generations=4,
        max_completion_length=128,
    )
    base_training = _selection(catalog, "training", "training/qwen3.5-0.8b-trl-distill-lora@1")
    assert isinstance(base_training, TrainingBinding)
    training = replace(base_training, runtime=replace(base_training.runtime, global_batch_size=8))
    environment = EnvironmentBinding(
        "environments/static-inference-length",
        "tool-use",
        EnvironmentSource("static", "https://example.test/static", "b" * 40),
        PythonFactoryActivation("builtins:object"),
        SamplingPolicy(max_tokens=128, temperature=1.0),
        num_tasks=1,
    )
    inference = InferenceBinding(
        "inference/static-inference-length@1",
        "1",
        model,
        "vllm@0.25.1",
        model.renderer_contract,
        {"max_model_len": 4096},
        {"max_tokens": 256, "temperature": 1.0, "top_p": 1.0},
        ExecutionTarget("targets/static", "1", "nvidia-cuda"),
        ("rollout",),
    )

    with pytest.raises(ContractError, match="sampling max_tokens must equal"):
        grpo_definition().static_validator(  # type: ignore[misc]
            {
                "settings": settings,
                "training": training,
                "environment": environment,
                "rollout_inference": inference,
            }
        )


def test_static_grpo_preparation_rejects_context_overcommit() -> None:
    catalog = open_catalog(scope="jobs-test")
    model = cast(ModelVariant, _selection(catalog, "model", "models/qwen3.5-2b@bf16"))
    settings = GRPOSettings(
        id="grpo-static-context-overcommit",
        loop=TrainingLoop(max_steps=1, per_device_batch_size=1, gradient_accumulation_steps=8),
        num_prompts_per_step=2,
        num_generations=4,
        max_prompt_length=4000,
        max_completion_length=128,
    )
    base_training = _selection(catalog, "training", "training/qwen3.5-0.8b-trl-distill-lora@1")
    assert isinstance(base_training, TrainingBinding)
    training = replace(base_training, runtime=replace(base_training.runtime, global_batch_size=8))
    environment = EnvironmentBinding(
        "environments/static-context-overcommit",
        "tool-use",
        EnvironmentSource("static", "https://example.test/static", "b" * 40),
        PythonFactoryActivation("builtins:object"),
        SamplingPolicy(max_tokens=128, temperature=1.0),
        num_tasks=1,
    )
    inference = InferenceBinding(
        "inference/static-context-overcommit@1",
        "1",
        model,
        "vllm@0.25.1",
        model.renderer_contract,
        {"max_model_len": 4096},
        {"max_tokens": 128, "temperature": 1.0, "top_p": 1.0},
        ExecutionTarget("targets/static", "1", "nvidia-cuda"),
        ("rollout",),
    )

    with pytest.raises(ContractError, match="prompt and completion budgets exceed"):
        grpo_definition().static_validator(  # type: ignore[misc]
            {
                "settings": settings,
                "training": training,
                "environment": environment,
                "rollout_inference": inference,
            }
        )


def test_grpo_materializes_stored_adapter_for_policy_and_inference(tmp_path: Path) -> None:
    catalog = open_catalog(scope="jobs-test")
    foundation = cast(ModelVariant, _selection(catalog, "model", "models/qwen3.5-2b@bf16"))
    adapter = replace(
        foundation,
        id="models/qwen3.5-2b-sft-test@v0",
        artifact=StoredArtifactRef("trackio", "ambient-agent", "sft-adapter", "v0"),
        form="peft-adapter",
        parent=foundation.id,
    )
    inference = cast(InferenceBinding, _selection(catalog, "inference", "inference/qwen3.5-2b-vllm-eval@1"))
    materialized_path = (tmp_path / "model-adapter").resolve()
    materialized_path.mkdir()
    materialized = LocalArtifactRef(materialized_path, "a" * 64)
    context = RunContext(
        project_id="jobs-test",
        work_package_id="train/grpo-materialize",
        run_id="run-materialize",
        job_kind="train.grpo",
        job_definition_version="train/trl-grpo@1",
        workspace=(tmp_path / "workspace").resolve(),
        observer=NullObserver(),
        input_artifacts={"model_adapter": materialized},
    )

    policy, rollout = _materialize_grpo_policy(context, adapter, inference)

    assert policy.artifact is materialized
    assert policy.digest == materialized.digest
    assert rollout.model is policy


def test_model_roles_cannot_consume_another_roles_materialized_artifact(tmp_path: Path) -> None:
    catalog = open_catalog(scope="jobs-test")
    model = cast(ModelVariant, _selection(catalog, "model", "models/qwen3.5-2b@bf16"))
    policy_artifact = LocalArtifactRef((tmp_path / "policy").resolve(), "a" * 64)
    judge_artifact = LocalArtifactRef((tmp_path / "judge").resolve(), "b" * 64)
    context = RunContext(
        project_id="jobs-test",
        work_package_id="train/role-isolation",
        run_id="run-role-isolation",
        job_kind="train.gdpo",
        job_definition_version="train/gdpo-judged@1",
        workspace=(tmp_path / "workspace").resolve(),
        observer=NullObserver(),
        input_artifacts={
            "model_adapter": policy_artifact,
            "judge_inference_weights": judge_artifact,
        },
    )

    policy = _materialize_selected_model_variant(context, model)
    judge = _materialize_selected_model_variant(context, model, role="judge_inference")
    untouched_draft = _materialize_selected_model_variant(context, model, role="draft_inference")

    assert policy.artifact is policy_artifact
    assert policy.form == "adapter"
    assert judge.artifact is judge_artifact
    assert judge.form == "full-finetuned"
    assert untouched_draft is model


def test_judge_plugins_bind_to_services_independently_of_service_seats() -> None:
    services = {"judge/shared": ("judge_inference", 8123)}

    assert _judge_service_bindings(
        services,
        {"quality": "judge/shared", "efficiency": "judge/shared"},
    ) == {"quality": "judge/shared", "efficiency": "judge/shared"}

    with pytest.raises(ValueError, match="every selected service"):
        _judge_service_bindings(services, {"quality": "judge/missing"})


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


def test_static_preparation_checks_sampo_oversampling_capacity_on_verl() -> None:
    validator = sampo_definition().static_validator
    assert validator is not None

    def on_verl(**changes: object) -> dict[str, object]:
        seats = _oversampled_sampo_seats(**changes)  # type: ignore[arg-type]
        training = cast(TrainingBinding, seats["training"])
        settings = cast(SAMPOSettings, seats["settings"])
        seats["settings"] = replace(settings, loop=replace(settings.loop, lr_scheduler_type="constant"))
        seats["training"] = replace(training, backend="verl@d344b545")
        return seats

    validator(on_verl())  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="oversample 1 needs 12 concurrent episodes"):
        validator(on_verl(max_num_seqs=8))  # type: ignore[arg-type]
