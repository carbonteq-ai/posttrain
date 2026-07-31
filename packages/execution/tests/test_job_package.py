from __future__ import annotations

from dataclasses import replace

import pytest
from posttrain.common import ContractError
from posttrain.execution import (
    BackendRuntimeLock,
    DatasetPackageLock,
    EnvironmentActivationLock,
    EnvironmentPackageLock,
    JobPackageManifest,
    RuntimeDependencyLock,
    RuntimeImageRef,
)


def _manifest() -> JobPackageManifest:
    return JobPackageManifest(
        project_id="foundation-models",
        work_package_id="train/qwen-grpo",
        job_id="grpo",
        job_definition_id="train/trl-grpo@1",
        job_kind="train.grpo",
        resolved_inputs_digest="a" * 64,
        framework_source_digest="b" * 64,
        project_source_digest="6" * 64,
        runtime_dependencies_digest="3" * 64,
        code_requirements_digest="4" * 64,
        resolved_config_digest="5" * 64,
        project_config_digest="7" * 64,
        universal_image=RuntimeImageRef(f"registry.lan/posttrain-base@sha256:{'9' * 64}"),
        kind_image=RuntimeImageRef(f"registry.lan/posttrain-kind-online-rl@sha256:{'c' * 64}"),
        runtime_variant="online-rl-trl-py312",
        environment_packages=(
            EnvironmentPackageLock(
                package="gsm8k-v1",
                repository="https://github.com/PrimeIntellect-ai/verifiers",
                revision="d" * 40,
                subdirectory="environments/gsm8k_v1",
                tree_digest="e" * 64,
                wheel_filename="gsm8k_v1-1.0.0-py3-none-any.whl",
                wheel_digest="f" * 64,
                wheel_size_bytes=2048,
            ),
        ),
        environment_activations=(
            EnvironmentActivationLock(
                environment_id="gsm8k-train",
                package="gsm8k-v1",
                kind="verifiers-config",
                digest="65bc8fb086f81b6d5bbe66ad373f2adac5e334f2ed9c4a11ff37444b80b3166d",
                config={"taskset": {"id": "gsm8k", "split": "train"}},
            ),
        ),
        datasets=(
            DatasetPackageLock(
                seat_name="dataset",
                selection_id="datasets/gsm8k-train",
                selection_revision="1",
                dataset_revision="main",
                kind="supervised",
                schema_version=1,
                digest="1" * 64,
                package_path="datasets/gsm8k-train/data.jsonl",
                manifest_path="datasets/gsm8k-train/manifest.json",
                size_bytes=1024,
                num_records=16,
            ),
        ),
        expected_artifact_roles=("model", "summary"),
    )


def test_job_package_round_trips_and_excludes_execution_identity() -> None:
    manifest = _manifest()

    loaded = JobPackageManifest.from_bytes(manifest.to_bytes())
    payload = loaded.to_payload()

    assert loaded == manifest
    assert loaded.package_key == manifest.package_key
    assert set(payload).isdisjoint(
        {
            "run_id",
            "attempt",
            "provider",
            "target",
            "environment_names",
            "mounts",
            "runtime_image",
        }
    )


def test_job_package_key_changes_with_meaning_but_not_a_run() -> None:
    manifest = _manifest()

    changed = replace(
        manifest,
        datasets=(replace(manifest.datasets[0], digest="2" * 64),),
    )

    assert changed.package_key != manifest.package_key
    assert (
        replace(
            manifest,
            runtime_variant="online-rl-trl-alternate",
        ).package_key
        != manifest.package_key
    )


def test_job_package_runtime_variant_must_refine_logical_profile() -> None:
    with pytest.raises(ContractError, match="refine"):
        replace(_manifest(), runtime_variant="supervised-trl-py312")


def test_verl_runtime_identity_is_interpreter_specific_and_package_bound() -> None:
    locks = (
        RuntimeDependencyLock(
            role="backend",
            python_version="3.13.12",
            python_executable="/opt/posttrain-verl/bin/python",
            requirements_path="locks/runtime.backend.requirements.txt",
            requirements_digest="1" * 64,
            resolution_digest="2" * 64,
        ),
        RuntimeDependencyLock(
            role="control",
            python_version="3.13.12",
            python_executable="/opt/posttrain/venv/bin/python",
            requirements_path="locks/runtime.control.requirements.txt",
            requirements_digest="3" * 64,
            resolution_digest="4" * 64,
        ),
    )
    backend = BackendRuntimeLock(
        backend="verl",
        source_repository="https://github.com/carbonteq-ai/verl.git",
        source_revision="5" * 40,
        dependency_lock_path="/opt/posttrain-verl/release/uv.lock",
        dependency_lock_digest="6" * 64,
        working_directory="/opt/posttrain-verl/workdir",
        projection_path="/opt/posttrain-verl/projection",
        projection_digest="7" * 64,
        worker_module="posttrain.train.backends.verl.worker",
    )
    manifest = replace(
        _manifest(),
        runtime_variant="online-rl-verl-py313",
        runtime_dependency_locks=locks,
        backend_runtime=backend,
    )

    assert JobPackageManifest.from_bytes(manifest.to_bytes()) == manifest
    assert (
        replace(
            manifest,
            runtime_dependency_locks=(
                replace(locks[0], requirements_digest="8" * 64),
                locks[1],
            ),
        ).package_key
        != manifest.package_key
    )

    with pytest.raises(ContractError, match="capsule"):
        replace(backend, working_directory="/home/user/verl")
    with pytest.raises(ContractError, match="backend and control"):
        replace(manifest, runtime_dependency_locks=(locks[1],))


def test_job_package_rejects_floating_environment_revision() -> None:
    with pytest.raises(ContractError, match="full commit"):
        replace(_manifest().environment_packages[0], revision="main")


def test_job_package_rejects_duplicate_environment_packages() -> None:
    manifest = _manifest()

    with pytest.raises(ContractError, match="environment names"):
        replace(
            manifest,
            environment_packages=(manifest.environment_packages + manifest.environment_packages),
        )


def test_job_package_rejects_host_coupled_dataset_paths() -> None:
    manifest = _manifest()

    with pytest.raises(ContractError, match="normalized relative path"):
        replace(
            manifest,
            datasets=(
                replace(
                    manifest.datasets[0],
                    package_path="/opt/posttrain/data/gsm8k.jsonl",
                ),
            ),
        )


def test_job_package_requires_an_importable_python_factory_ref() -> None:
    manifest = _manifest()

    with pytest.raises(ContractError, match="import reference"):
        replace(
            manifest,
            environment_activations=(
                replace(
                    manifest.environment_activations[0],
                    kind="python-factory",
                    reference="gsm8k-factory",
                ),
            ),
        )


def test_job_package_declarative_activation_cannot_name_a_factory() -> None:
    manifest = _manifest()

    with pytest.raises(ContractError, match="cannot name a Python factory"):
        replace(
            manifest,
            environment_activations=(
                replace(
                    manifest.environment_activations[0],
                    reference="vf_gsm8k:load_environment",
                ),
            ),
        )


def test_job_package_allows_two_activations_from_one_environment_wheel() -> None:
    manifest = _manifest()

    expanded = replace(
        manifest,
        environment_activations=(
            *manifest.environment_activations,
            replace(
                manifest.environment_activations[0],
                environment_id="gsm8k-test",
            ),
        ),
    )

    assert len(expanded.environment_packages) == 1
    assert len(expanded.environment_activations) == 2


def test_environment_activation_config_is_deeply_immutable() -> None:
    activation = _manifest().environment_activations[0]
    taskset = activation.config["taskset"] if activation.config is not None else None

    with pytest.raises(TypeError):
        taskset["split"] = "test"  # type: ignore[index]


def test_environment_activation_qualification_policy_round_trips() -> None:
    manifest = _manifest()
    deferred = replace(
        manifest,
        environment_activations=(replace(manifest.environment_activations[0], qualification="deferred"),),
    )

    restored = JobPackageManifest.from_bytes(deferred.to_bytes())

    assert restored.environment_activations[0].qualification == "deferred"


def test_job_package_rejects_activation_without_its_environment_wheel() -> None:
    manifest = _manifest()

    with pytest.raises(ContractError, match="missing packages"):
        replace(
            manifest,
            environment_activations=(
                replace(
                    manifest.environment_activations[0],
                    package="another-env",
                ),
            ),
        )
