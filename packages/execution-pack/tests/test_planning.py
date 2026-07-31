from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from posttrain.data import DatasetLoadPlan
from posttrain.eval import (
    EnvironmentBinding,
    EnvironmentSource,
    EvaluationPlan,
    PythonFactoryActivation,
    SamplingPolicy,
    VerifiersV1ConfigActivation,
)
from posttrain.execution import RuntimeImageRef
from posttrain.execution_pack import (
    ImagePublicationSpec,
    environment_bindings,
    plan_job_pack,
)
from posttrain.tracking import RunSpec
from posttrain.work import PreparedWorkPackageJob, ResolvedSeats

COMMIT = "1" * 40
OTHER_COMMIT = "2" * 40
DIGEST = "a" * 64
BASE = RuntimeImageRef(f"registry.lan/posttrain/base@sha256:{'b' * 64}")
KIND = RuntimeImageRef(f"registry.lan/posttrain/online-rl@sha256:{'c' * 64}")
PUBLICATION = ImagePublicationSpec("registry.lan/posttrain/jobs")


def _environment(
    environment_id: str,
    package: str,
    subdirectory: str,
    *,
    revision: str = COMMIT,
    activation: VerifiersV1ConfigActivation | PythonFactoryActivation | None = None,
) -> EnvironmentBinding:
    return EnvironmentBinding(
        id=environment_id,
        category="reasoning",
        source=EnvironmentSource(
            package=package,
            repository="https://github.com/CarbonTeq/envs",
            revision=revision,
            subdirectory=subdirectory,
        ),
        activation=activation or VerifiersV1ConfigActivation({"taskset": {"id": environment_id, "split": "train"}}),
        sampling=SamplingPolicy(max_tokens=64),
        num_tasks=4,
    )


def _prepared(
    seats: ResolvedSeats,
    *,
    run_id: str = "run-a",
    source_metadata: dict[str, str] | None = None,
) -> PreparedWorkPackageJob:
    spec = RunSpec(
        project_id="project",
        work_package_id="train/grpo",
        stage="train",
        job_kind="train.grpo",
        job_definition_version="train/trl-grpo@1",
        run_id=run_id,
        resolved_inputs={"selection": {"id": "stable", "revision": "1"}},
        source_metadata=source_metadata or {},
    )
    value = SimpleNamespace(
        seats=seats,
        spec=spec,
        recipe_job=SimpleNamespace(id="grpo", kind="train.grpo"),
        definition=SimpleNamespace(
            id="train/trl-grpo@1",
            required_artifact_roles=("summary", "model"),
        ),
    )
    return cast(PreparedWorkPackageJob, value)


def test_extracts_direct_and_nested_environments_without_duplicates() -> None:
    gsm8k = _environment("math-gsm8k", "gsm8k-v1", "environments/gsm8k_v1")
    reverse = _environment(
        "text-reverse",
        "reverse-text-v1",
        "environments/reverse_text_v1",
    )
    plan = EvaluationPlan(
        id="evals/general",
        kind="general",
        environments=(reverse, gsm8k),
    )

    selected = environment_bindings(cast(ResolvedSeats, {"direct": gsm8k, "plan": plan}))

    assert [binding.id for binding in selected] == [
        "math-gsm8k",
        "text-reverse",
    ]


def test_plan_deduplicates_one_repository_into_two_package_roots() -> None:
    gsm8k = _environment("math-gsm8k", "gsm8k-v1", "environments/gsm8k_v1")
    reverse = _environment(
        "text-reverse",
        "reverse-text-v1",
        "environments/reverse_text_v1",
        activation=PythonFactoryActivation("reverse_text:load_environment"),
    )

    plan = plan_job_pack(
        _prepared(cast(ResolvedSeats, {"train": gsm8k, "test": reverse})),
        framework_source_digest=DIGEST,
        project_source_digest="d" * 64,
        universal_image=BASE,
        kind_image=KIND,
        publication=PUBLICATION,
        runtime_variant="online-rl-trl-py312",
    )

    assert plan.spec.kind_profile == "online-rl"
    assert plan.spec.runtime_variant == "online-rl-trl-py312"
    assert len(plan.spec.git_sources) == 1
    assert plan.spec.git_sources[0].subdirectories == (
        "environments/gsm8k_v1",
        "environments/reverse_text_v1",
    )
    assert [wheel.package for wheel in plan.spec.environment_wheels] == [
        "gsm8k-v1",
        "reverse-text-v1",
    ]
    assert [
        (activation.environment_id, activation.kind, activation.reference)
        for activation in plan.spec.environment_activations
    ] == [
        ("math-gsm8k", "verifiers-config", None),
        (
            "text-reverse",
            "python-factory",
            "reverse_text:load_environment",
        ),
    ]
    assert plan.spec.expected_artifact_roles == ("model", "summary")


def test_plan_retains_dataset_seats_without_materializing_them() -> None:
    dataset = DatasetLoadPlan(
        id="datasets/sft-smoke@1",
        revision="1",
        kind="supervised",
        source={"kind": "fixture", "resource": "example:data.jsonl"},
        format="messages",
    )

    plan = plan_job_pack(
        _prepared(cast(ResolvedSeats, {"dataset": dataset})),
        framework_source_digest=DIGEST,
        project_source_digest="d" * 64,
        universal_image=BASE,
        kind_image=KIND,
        publication=PUBLICATION,
    )

    assert len(plan.spec.datasets) == 1
    assert plan.spec.datasets[0].seat_name == "dataset"
    assert plan.spec.datasets[0].selection is dataset


def test_plan_key_ignores_run_provider_paths_and_publication() -> None:
    environment = _environment(
        "math-gsm8k",
        "gsm8k-v1",
        "environments/gsm8k_v1",
    )
    first = plan_job_pack(
        _prepared(
            cast(ResolvedSeats, {"environment": environment}),
            run_id="run-a",
            source_metadata={
                "provider": "local",
                "project_root": "/private/developer/path",
            },
        ),
        framework_source_digest=DIGEST,
        project_source_digest="d" * 64,
        universal_image=BASE,
        kind_image=KIND,
        publication=PUBLICATION,
    )
    second = plan_job_pack(
        _prepared(
            cast(ResolvedSeats, {"environment": environment}),
            run_id="run-b",
            source_metadata={
                "provider": "dstack",
                "project_root": "/different/absolute/path",
            },
        ),
        framework_source_digest=DIGEST,
        project_source_digest="d" * 64,
        universal_image=BASE,
        kind_image=KIND,
        publication=ImagePublicationSpec("registry.example/posttrain/jobs"),
    )

    assert first.plan_key == second.plan_key
    assert first.publication_plan_key != second.publication_plan_key
    serialized = first.spec.to_payload()
    assert "run-a" not in str(serialized)
    assert "/private/developer/path" not in str(serialized)


def test_plan_key_binds_the_complete_catalog_family_registry_lock() -> None:
    environment = _environment("math-gsm8k", "gsm8k-v1", "environments/gsm8k_v1")
    common = dict(
        framework_source_digest=DIGEST,
        project_source_digest="d" * 64,
        universal_image=BASE,
        kind_image=KIND,
        publication=PUBLICATION,
    )
    first = plan_job_pack(
        _prepared(cast(ResolvedSeats, {"environment": environment})),
        **common,
        family_registry_lock={"entries": [{"name": "environment"}], "digest": "e" * 64},
    )
    second = plan_job_pack(
        _prepared(cast(ResolvedSeats, {"environment": environment})),
        **common,
        family_registry_lock={
            "entries": [{"name": "environment"}, {"name": "unrelated-provider"}],
            "digest": "f" * 64,
        },
    )

    assert first.plan_key != second.plan_key


def test_input_or_build_semantics_change_plan_key() -> None:
    environment = _environment(
        "math-gsm8k",
        "gsm8k-v1",
        "environments/gsm8k_v1",
    )
    base = plan_job_pack(
        _prepared(cast(ResolvedSeats, {"environment": environment})),
        framework_source_digest=DIGEST,
        project_source_digest="d" * 64,
        universal_image=BASE,
        kind_image=KIND,
        publication=PUBLICATION,
    )
    changed_source = plan_job_pack(
        _prepared(cast(ResolvedSeats, {"environment": environment})),
        framework_source_digest=DIGEST,
        project_source_digest="e" * 64,
        universal_image=BASE,
        kind_image=KIND,
        publication=PUBLICATION,
    )
    changed_runtime = plan_job_pack(
        _prepared(cast(ResolvedSeats, {"environment": environment})),
        framework_source_digest=DIGEST,
        project_source_digest="d" * 64,
        universal_image=BASE,
        kind_image=RuntimeImageRef(f"registry.lan/posttrain/online-rl-verl@sha256:{'e' * 64}"),
        publication=PUBLICATION,
        runtime_variant="online-rl-verl-two-venv",
    )

    assert base.plan_key != changed_source.plan_key
    assert base.plan_key != changed_runtime.plan_key


def test_runtime_variant_must_refine_logical_kind_profile() -> None:
    with pytest.raises(ValueError, match="refine"):
        plan_job_pack(
            _prepared(cast(ResolvedSeats, {})),
            framework_source_digest=DIGEST,
            project_source_digest="d" * 64,
            universal_image=BASE,
            kind_image=KIND,
            publication=PUBLICATION,
            runtime_variant="supervised-trl-py312",
        )


def test_rejects_conflicting_revisions_of_one_repository() -> None:
    first = _environment("math-a", "math-a", "environments/a")
    second = _environment(
        "math-b",
        "math-b",
        "environments/b",
        revision=OTHER_COMMIT,
    )

    with pytest.raises(
        ValueError,
        match="multiple revisions of the same environment repository",
    ):
        plan_job_pack(
            _prepared(cast(ResolvedSeats, {"first": first, "second": second})),
            framework_source_digest=DIGEST,
            project_source_digest="d" * 64,
            universal_image=BASE,
            kind_image=KIND,
            publication=PUBLICATION,
        )


@pytest.mark.parametrize(
    "repository",
    [
        "https://user:secret@registry.example/posttrain/jobs",
        "https://registry.example/posttrain/jobs",
        "registry.example/posttrain/jobs@sha256:" + "a" * 64,
    ],
)
def test_publication_repository_rejects_urls_credentials_and_digests(
    repository: str,
) -> None:
    with pytest.raises(ValueError, match="credential-free OCI repository"):
        ImagePublicationSpec(repository)
