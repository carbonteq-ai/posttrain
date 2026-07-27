from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from posttrain.common import ContractError, JsonValue
from posttrain.data import DatasetLoadPlan, DatasetMaterialization
from posttrain.execution import (
    EnvironmentActivationLock,
    EnvironmentPackageLock,
    RuntimeImageRef,
)
from posttrain.execution_pack import (
    DatasetPackRequest,
    EnvironmentWheelRequest,
    GitSourceRequest,
    ImagePublicationSpec,
    ImmutableDatasetPackager,
    JobImagePublicationRequest,
    JobPackInputs,
    JobPackPlan,
    JobPackService,
    JobPackSpec,
    MaterializedDatasetPackages,
    MaterializedEnvironmentPackage,
    MaterializedEnvironments,
    ProjectConfigBundle,
    SourcePackage,
    digest_source_package,
)
from posttrain.execution_pack.service import _backend_runtime_lock

COMMIT = "1" * 40
BASE = RuntimeImageRef(f"registry.lan/posttrain/base@sha256:{'b' * 64}")
KIND = RuntimeImageRef(f"registry.lan/posttrain/supervised@sha256:{'c' * 64}")


def _digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _source(root: Path, package: str) -> SourcePackage:
    package_root = root / package
    (package_root / "src" / package.replace("-", "_")).mkdir(parents=True)
    (package_root / "pyproject.toml").write_text(
        "\n".join(
            (
                "[project]",
                f'name = "{package}"',
                'version = "0.1.0"',
                "",
                "[build-system]",
                'requires = ["hatchling"]',
                'build-backend = "hatchling.build"',
                "",
            )
        ),
        encoding="utf-8",
    )
    (package_root / "src" / package.replace("-", "_") / "__init__.py").write_text(
        '"""fixture."""\n',
        encoding="utf-8",
    )
    return SourcePackage(root=root.resolve(), install_roots=(package,))


def _project_config(*, overlay: bytes = b"models: []\n") -> ProjectConfigBundle:
    return ProjectConfigBundle(
        files={
            ".posttrain/catalog/models.yaml": overlay,
            ".posttrain/project.toml": (
                b"schema_version = 2\n"
                b'project_id = "project"\n'
                b'catalog_overlays = ["catalog"]\n'
                b'work_packages = "work_packages"\n'
                b'state = "state"\n'
                b'tracking = "trackio"\n'
                b'entry = "project_pkg.entry:configure"\n'
                b'project_brief = "project.yaml"\n'
            ),
            ".posttrain/project.yaml": b"objective: qualify\n",
            ".posttrain/work_packages/train.yaml": (b"id: train/sft\nproject_id: project\nstage: train\n"),
        },
        selected_work_package=".posttrain/work_packages/train.yaml",
    )


def _inputs(tmp_path: Path, *, overlay: bytes = b"models: []\n") -> JobPackInputs:
    framework = _source(tmp_path / "framework", "posttrain-runtime-fixture")
    project = _source(tmp_path / "project", "project-pkg")
    return JobPackInputs(
        framework_source=framework,
        project_source=project,
        resolved_inputs={"dataset": {"id": "dataset/sft@1", "revision": "1"}},
        project_config=_project_config(overlay=overlay),
    )


def _activation(environment_id: str, package: str) -> EnvironmentActivationLock:
    payload = {
        "kind": "verifiers-config",
        "config": {"taskset": {"id": environment_id, "split": "train"}},
    }
    return EnvironmentActivationLock(
        environment_id=environment_id,
        package=package,
        kind="verifiers-config",
        digest=_digest_json(payload),
        config=payload["config"],
    )


def _plan(
    inputs: JobPackInputs,
    *,
    environments: bool = False,
    dataset: DatasetLoadPlan | None = None,
    repository: str = "registry.lan/posttrain/jobs",
) -> JobPackPlan:
    git_sources: tuple[GitSourceRequest, ...] = ()
    wheel_requests: tuple[EnvironmentWheelRequest, ...] = ()
    activations: tuple[EnvironmentActivationLock, ...] = ()
    if environments:
        git_sources = (
            GitSourceRequest(
                repository="https://github.com/CarbonTeq/envs",
                revision=COMMIT,
                subdirectories=("environments/math", "environments/text"),
            ),
        )
        wheel_requests = (
            EnvironmentWheelRequest(
                "math-env",
                git_sources[0].repository,
                COMMIT,
                "environments/math",
            ),
            EnvironmentWheelRequest(
                "text-env",
                git_sources[0].repository,
                COMMIT,
                "environments/text",
            ),
        )
        activations = (
            _activation("math-train", "math-env"),
            _activation("text-train", "text-env"),
        )
    resolved = dict(inputs.resolved_inputs)
    return JobPackPlan(
        spec=JobPackSpec(
            project_id="project",
            work_package_id="train/sft",
            job_id="sft",
            job_definition_id="train/trl-sft@1",
            job_kind="train.sft",
            kind_profile="supervised",
            runtime_variant="supervised",
            resolved_inputs_digest=_digest_json(resolved),
            framework_source_digest=digest_source_package(inputs.framework_source),
            project_source_digest=digest_source_package(inputs.project_source),
            universal_image=BASE,
            kind_image=KIND,
            datasets=((DatasetPackRequest("dataset", dataset),) if dataset is not None else ()),
            git_sources=git_sources,
            environment_wheels=wheel_requests,
            environment_activations=activations,
            expected_artifact_roles=("model", "summary"),
        ),
        publication=ImagePublicationSpec(repository),
    )


def _dataset_packager(tmp_path: Path) -> ImmutableDatasetPackager:
    return ImmutableDatasetPackager(
        state_dir=(tmp_path / "dataset-state").resolve(),
        project_root=tmp_path.resolve(),
    )


def _materializing_dataset_packager(
    tmp_path: Path,
) -> ImmutableDatasetPackager:
    def materialize(
        plan: DatasetLoadPlan,
        state_dir: Path,
        project_root: Path,
    ) -> DatasetMaterialization:
        assert state_dir.is_absolute()
        assert project_root.is_absolute()
        root = state_dir / "materialized"
        root.mkdir(parents=True, exist_ok=True)
        contents = b'{"messages":[{"role":"user","content":"hello"}]}\n'
        digest = hashlib.sha256(contents).hexdigest()
        data = root / "data.jsonl"
        data.write_bytes(contents)
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "selection_id": plan.id,
                    "selection_revision": plan.revision,
                    "dataset_revision": plan.dataset_revision,
                    "source_kind": plan.source_kind,
                    "content_sha256": digest,
                    "examples": 1,
                    "data": "data.jsonl",
                }
            ),
            encoding="utf-8",
        )
        return DatasetMaterialization(
            selection_id=plan.id,
            selection_revision=plan.revision,
            source_kind=plan.source_kind,
            path=data,
            manifest_path=manifest,
            content_sha256=digest,
            examples=1,
            created=True,
        )

    return ImmutableDatasetPackager(
        state_dir=(tmp_path / "dataset-state").resolve(),
        project_root=tmp_path.resolve(),
        materializer=materialize,
    )


@dataclass
class _FakeEnvironmentPackager:
    salt: bytes = b""
    calls: int = 0

    def package(
        self,
        *,
        git_sources: tuple[GitSourceRequest, ...],
        wheel_requests: tuple[EnvironmentWheelRequest, ...],
        kind_profile: str,
        output_root: Path,
    ) -> MaterializedEnvironments:
        self.calls += 1
        assert len(git_sources) == 1
        assert kind_profile == "supervised"
        packages: list[MaterializedEnvironmentPackage] = []
        requirements: list[str] = []
        for index, request in enumerate(wheel_requests):
            contents = f"{request.package}-wheel".encode()
            digest = hashlib.sha256(contents).hexdigest()
            filename = f"{request.package.replace('-', '_')}-1.0.0-py3-none-any.whl"
            path = output_root / filename
            path.write_bytes(contents)
            packages.append(
                MaterializedEnvironmentPackage(
                    path=path.resolve(),
                    lock=EnvironmentPackageLock(
                        package=request.package,
                        repository=request.repository,
                        revision=request.revision,
                        subdirectory=request.subdirectory,
                        tree_digest=f"{index + 1}" * 64,
                        wheel_filename=filename,
                        wheel_digest=digest,
                        wheel_size_bytes=len(contents),
                    ),
                )
            )
            requirements.extend(
                (
                    f"./wheels/environments/{filename} \\",
                    f"    --hash=sha256:{digest}",
                )
            )
        requirements.append(
            f"typing-extensions==4.15.0 \\\n    --hash=sha256:{hashlib.sha256(self.salt or b'typing').hexdigest()}"
        )
        lock_contents = ("\n".join(requirements) + "\n").encode()
        lock_path = output_root / "runtime.requirements.txt"
        lock_path.write_bytes(lock_contents)
        return MaterializedEnvironments(
            packages=tuple(packages),
            runtime_requirements=lock_path.resolve(),
            runtime_dependencies_digest=hashlib.sha256(lock_contents).hexdigest(),
        )


def test_packs_and_reuses_one_deterministic_provider_neutral_context(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path / "inputs")
    plan = _plan(inputs)
    service = JobPackService(
        output_root=(tmp_path / "packages").resolve(),
        dataset_packager=_dataset_packager(tmp_path),
    )

    first = service.pack(plan, inputs)
    second = service.pack(plan, inputs)

    assert first == second
    assert first.root.name == first.manifest.package_key
    assert {path.name for path in first.root.iterdir()} == {
        "config",
        "datasets",
        "locks",
        "package.json",
        "sources",
        "wheels",
    }
    assert (first.root / "config/project/.posttrain/project.toml").is_file()
    resolved = json.loads((first.root / "config/resolved.json").read_text(encoding="utf-8"))
    assert resolved["schema"] == "posttrain.resolved-job.v1"
    assert resolved["runtime_variant"] == "supervised"
    assert resolved["selected_work_package"] == "project/.posttrain/work_packages/train.yaml"
    assert first.manifest.job_kind == "train.sft"
    assert first.manifest.runtime_dependencies_digest == hashlib.sha256(b"").hexdigest()
    publication = JobImagePublicationRequest(
        manifest=first.manifest,
        staged_context=first.root,
        publication=plan.publication,
    )
    assert first.publication_key == publication.publication_key


def test_packs_multiple_environment_wheels_and_activation_configs(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path / "inputs")
    plan = _plan(inputs, environments=True)
    environment_packager = _FakeEnvironmentPackager()
    result = JobPackService(
        output_root=(tmp_path / "packages").resolve(),
        dataset_packager=_dataset_packager(tmp_path),
        environment_packager=environment_packager,
    ).pack(plan, inputs)

    assert environment_packager.calls == 1
    assert [lock.package for lock in result.manifest.environment_packages] == [
        "math-env",
        "text-env",
    ]
    activation_ids: list[object] = []
    for lock in result.manifest.environment_activations:
        taskset = dict(lock.config or {})["taskset"]
        assert isinstance(taskset, Mapping)
        activation_ids.append(taskset["id"])
    assert activation_ids == ["math-train", "text-train"]
    for lock in result.manifest.environment_packages:
        assert (result.root / "wheels/environments" / lock.wheel_filename).is_file()


def test_verl_backend_identity_rejects_host_paths_and_digests_projection(
    tmp_path: Path,
) -> None:
    framework_root = tmp_path / "framework"
    for package in ("common", "data", "train"):
        path = framework_root / "packages" / package / "src" / "posttrain" / package
        path.mkdir(parents=True)
        (path / "__init__.py").write_text(f'"""{package}."""\n', encoding="utf-8")
    source = SourcePackage(
        root=framework_root.resolve(),
        install_roots=("packages/common", "packages/data", "packages/train"),
    )
    options = {
        "python_executable": "/opt/posttrain-verl/bin/python",
        "working_directory": "/opt/posttrain-verl/workdir",
        "source_revision": "a" * 40,
        "dependency_lock_sha256": "b" * 64,
        "source_dirty": False,
    }
    resolved = {
        "training": {
            "resolved": {
                "backend": "verl@candidate",
                "backend_options": options,
            }
        }
    }

    lock = _backend_runtime_lock(
        runtime_variant="online-rl-verl-py313",
        resolved_inputs=cast(Mapping[str, JsonValue], resolved),
        framework_source=source,
        work=tmp_path / "work",
    )

    assert lock is not None
    assert lock.projection_digest
    assert lock.working_directory == "/opt/posttrain-verl/workdir"

    resolved["training"]["resolved"]["backend_options"] = {  # type: ignore[index]
        **options,
        "working_directory": "/home/developer/verl",
    }
    with pytest.raises(ContractError, match="capsule-owned"):
        _backend_runtime_lock(
            runtime_variant="online-rl-verl-py313",
            resolved_inputs=cast(Mapping[str, JsonValue], resolved),
            framework_source=source,
            work=tmp_path / "other-work",
        )


def test_packs_materialized_dataset_into_the_same_final_manifest(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path / "inputs")
    dataset = DatasetLoadPlan(
        id="datasets/sft@1",
        revision="1",
        kind="supervised",
        source={"kind": "fixture", "resource": "tests:data.jsonl"},
        format="messages",
    )
    result = JobPackService(
        output_root=(tmp_path / "packages").resolve(),
        dataset_packager=_materializing_dataset_packager(tmp_path),
    ).pack(_plan(inputs, dataset=dataset), inputs)

    assert len(result.manifest.datasets) == 1
    lock = result.manifest.datasets[0]
    assert lock.seat_name == "dataset"
    assert lock.num_records == 1
    assert (result.root / lock.package_path).is_file()
    assert (result.root / lock.manifest_path).is_file()


def test_materialized_dependency_or_project_config_changes_final_package_key(
    tmp_path: Path,
) -> None:
    first_inputs = _inputs(tmp_path / "first-inputs")
    second_inputs = _inputs(
        tmp_path / "second-inputs",
        overlay=b"models:\n  - id: changed\n",
    )
    first_plan = _plan(first_inputs, environments=True)
    second_plan = _plan(second_inputs, environments=True)

    first = JobPackService(
        output_root=(tmp_path / "first-packages").resolve(),
        dataset_packager=_dataset_packager(tmp_path / "first"),
        environment_packager=_FakeEnvironmentPackager(salt=b"first"),
    ).pack(first_plan, first_inputs)
    dependency_changed = JobPackService(
        output_root=(tmp_path / "dependency-packages").resolve(),
        dataset_packager=_dataset_packager(tmp_path / "dependency"),
        environment_packager=_FakeEnvironmentPackager(salt=b"second"),
    ).pack(first_plan, first_inputs)
    config_changed = JobPackService(
        output_root=(tmp_path / "config-packages").resolve(),
        dataset_packager=_dataset_packager(tmp_path / "config"),
        environment_packager=_FakeEnvironmentPackager(salt=b"first"),
    ).pack(second_plan, second_inputs)

    assert first.manifest.package_key != dependency_changed.manifest.package_key
    assert first.manifest.package_key != config_changed.manifest.package_key


def test_rejects_plan_source_or_resolved_input_drift_before_materialization(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path / "inputs")
    plan = _plan(inputs, environments=True)
    environment_packager = _FakeEnvironmentPackager()
    service = JobPackService(
        output_root=(tmp_path / "packages").resolve(),
        dataset_packager=_dataset_packager(tmp_path),
        environment_packager=environment_packager,
    )
    (
        inputs.framework_source.root / "posttrain-runtime-fixture" / "src" / "posttrain_runtime_fixture" / "__init__.py"
    ).write_text("# drift\n", encoding="utf-8")

    with pytest.raises(ContractError, match="framework source tree differs"):
        service.pack(plan, inputs)

    assert environment_packager.calls == 0


def test_project_config_bundle_requires_closed_declared_inputs() -> None:
    files = dict(_project_config().files)
    del files[".posttrain/catalog/models.yaml"]

    with pytest.raises(ContractError, match="omits declared catalog overlay"):
        ProjectConfigBundle(
            files=files,
            selected_work_package=".posttrain/work_packages/train.yaml",
        )


def test_project_config_bundle_rejects_state_unselected_jobs_and_secrets() -> None:
    files = dict(_project_config().files)
    files[".posttrain/work_packages/other.yaml"] = b"id: other\n"
    with pytest.raises(ContractError, match="only the selected"):
        ProjectConfigBundle(
            files=files,
            selected_work_package=".posttrain/work_packages/train.yaml",
        )

    files = dict(_project_config().files)
    files[".posttrain/state/runtime.json"] = b"{}\n"
    with pytest.raises(ContractError, match="state cannot be packed"):
        ProjectConfigBundle(
            files=files,
            selected_work_package=".posttrain/work_packages/train.yaml",
        )

    files = dict(_project_config().files)
    files[".posttrain/catalog/models.yaml"] = b"api_key: leaked\n"
    with pytest.raises(ContractError, match="secret field"):
        ProjectConfigBundle(
            files=files,
            selected_work_package=".posttrain/work_packages/train.yaml",
        )


def test_job_pack_inputs_snapshot_nested_json_and_reject_non_finite_values(
    tmp_path: Path,
) -> None:
    values = [1, 2]
    selected = {
        "nested": cast(JsonValue, {"values": values}),
    }
    inputs = JobPackInputs(
        framework_source=_source(tmp_path / "framework", "framework"),
        project_source=_source(tmp_path / "project", "project"),
        resolved_inputs=selected,
        project_config=_project_config(),
    )
    values.append(3)

    assert dict(inputs.resolved_inputs) == {"nested": {"values": [1, 2]}}
    nested = inputs.resolved_inputs["nested"]
    assert isinstance(nested, dict)
    with pytest.raises(TypeError, match="immutable"):
        nested["changed"] = True

    with pytest.raises(ContractError, match="only JSON values"):
        JobPackInputs(
            framework_source=inputs.framework_source,
            project_source=inputs.project_source,
            resolved_inputs={"loss": float("nan")},
            project_config=inputs.project_config,
        )


@dataclass
class _RogueDatasetPackager:
    def package(
        self,
        requests: Sequence[DatasetPackRequest],
        *,
        output_root: Path,
    ) -> MaterializedDatasetPackages:
        del requests
        (output_root / "datasets/rogue.txt").write_text("not locked")
        return MaterializedDatasetPackages(output_root, ())


def test_rejects_dataset_packager_files_without_locks(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "inputs")
    with pytest.raises(ContractError, match="outside its declared locks"):
        JobPackService(
            output_root=(tmp_path / "packages").resolve(),
            dataset_packager=_RogueDatasetPackager(),
        ).pack(_plan(inputs), inputs)


def test_rejects_dirty_retained_context(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "inputs")
    plan = _plan(inputs)
    service = JobPackService(
        output_root=(tmp_path / "packages").resolve(),
        dataset_packager=_dataset_packager(tmp_path),
    )
    result = service.pack(plan, inputs)
    (result.root / "config/resolved.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ContractError, match="dirty filesystem drift"):
        service.pack(plan, inputs)
