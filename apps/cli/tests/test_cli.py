"""Tests for the primary posttrain command-line interface."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from posttrain.catalog import ProjectLayout
from posttrain.common import ContractError, ExecutionTarget, RunContext
from posttrain.execution import (
    AdmissionEntry,
    AdmissionResult,
    ExecutionCleanupReceipt,
    ExecutionEvidenceSource,
    ExecutionHandle,
    ExecutionPlan,
    ExecutionPolicy,
    ExecutionReconciliation,
    ExecutionRecord,
    ExecutionRequest,
    ExecutionSubmission,
    ExecutionSubmissionStore,
    JobPackageManifest,
    LogCursor,
    LogPage,
    RuntimeImageRef,
    TrackingCancellationRecovery,
)
from posttrain.execution_pack import LocalPublishedJobImage, PackedJobContext, PublishedJobImage
from posttrain.jobs import build_job_runtime
from posttrain.runtime_images.manifest import load_manifest
from posttrain.serve import WorkloadMaterialization
from posttrain.tracking import RunSpec
from posttrain.work import (
    JobDefinition,
    ResolvedSeats,
    WorkPackageContext,
    WorkPackageHostRequest,
)
from posttrain_cli.cli import main
from posttrain_cli.commands.controller import controller_sweep


def test_job_help_exposes_product_path_not_compatibility_flags(capsys) -> None:
    for command in ("plan", "pack", "run"):
        assert main(["job", command, "--help"]) == 0
        help_text = capsys.readouterr().out
        assert "--entry" in help_text
        assert "--project-package" in help_text
        assert "--source-include" in help_text
        assert "--host" not in help_text
        assert "--in-process" not in help_text
        assert "work-package file or project-relative work-package path" in help_text
        assert "selected execution-target id" in help_text
        assert "selected job-kind runtime" in help_text
        if command != "pack":
            assert "[local|dstack]" in help_text
            assert "[tool.posttrain.execution] provider" in help_text
            assert "durable run identity" in help_text
            assert "idempotency namespace" in help_text
        if command == "run":
            assert "--resume-from-run" in help_text
            assert "checkpoint" in help_text


def test_recovery_checkpoint_rebinds_a_new_training_run() -> None:
    from posttrain.tracking import ArtifactLink, StoredArtifact
    from posttrain_cli.execution_config import ResolvedExecutionSettings
    from posttrain_cli.execution_planning import (
        PlannedJobExecution,
        PlannedJobLaunch,
        PlannedJobPackage,
        with_recovery_checkpoint,
    )

    spec = RunSpec(
        project_id="example",
        work_package_id="train/example",
        stage="train",
        job_kind="train.grpo",
        job_definition_version="train/trl-grpo@1",
        run_id="new-run",
    )
    planned = PlannedJobExecution(
        package=cast(PlannedJobPackage, SimpleNamespace()),
        launch=PlannedJobLaunch(spec, cast(ResolvedExecutionSettings, SimpleNamespace()), ()),
    )
    artifact = ArtifactLink(
        direction="output",
        logical_name="training/model/grpo/recovery-checkpoint",
        kind="training-checkpoint",
        artifact=StoredArtifact(
            provider="trackio",
            namespace="example",
            name="training-model-grpo-recovery-checkpoint",
            version="v3",
            digest="a" * 64,
            provider_metadata={"global_step": 25},
        ),
    )

    rebound = with_recovery_checkpoint(planned, source_run_id="old-run", artifact=artifact)

    selected = rebound.launch.run_spec.artifacts["recovery_checkpoint"]
    assert selected.kind == "training-checkpoint"
    assert selected.reference.version == "v3"
    assert rebound.launch.run_spec.resolved_inputs["recovery_checkpoint"] == {
        "source_run_id": "old-run",
        "logical_name": "training/model/grpo/recovery-checkpoint",
        "provider": "trackio",
        "namespace": "example",
        "name": "training-model-grpo-recovery-checkpoint",
        "version": "v3",
        "digest": "a" * 64,
    }


def test_recovery_checkpoint_requires_a_new_run_identity() -> None:
    from posttrain.tracking import ArtifactLink, StoredArtifact
    from posttrain_cli.execution_config import ResolvedExecutionSettings
    from posttrain_cli.execution_planning import (
        PlannedJobExecution,
        PlannedJobLaunch,
        PlannedJobPackage,
        with_recovery_checkpoint,
    )

    spec = RunSpec(
        project_id="example",
        work_package_id="train/example",
        stage="train",
        job_kind="train.sft",
        job_definition_version="train/trl-sft@1",
        run_id="same-run",
    )
    planned = PlannedJobExecution(
        package=cast(PlannedJobPackage, SimpleNamespace()),
        launch=PlannedJobLaunch(spec, cast(ResolvedExecutionSettings, SimpleNamespace()), ()),
    )
    artifact = ArtifactLink(
        direction="output",
        logical_name="training/model/sft/recovery-checkpoint",
        kind="training-checkpoint",
        artifact=StoredArtifact(
            provider="trackio",
            namespace="example",
            name="checkpoint",
            version="v0",
        ),
    )

    with pytest.raises(ContractError, match="new run identity"):
        with_recovery_checkpoint(planned, source_run_id="same-run", artifact=artifact)


def test_controller_once_renders_one_bounded_sweep(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()

    async def sweep(_layout):
        return [{"run_id": "run-1", "action": "reconcile", "state": "consistent"}]

    monkeypatch.setattr("posttrain_cli.commands.controller.controller_sweep", sweep)

    assert main(["--json", "--project-root", str(project), "controller", "run", "--once"]) == 0
    assert json.loads(capsys.readouterr().out) == [{"run_id": "run-1", "action": "reconcile", "state": "consistent"}]


def test_controller_status_reads_health_without_running_a_sweep(tmp_path: Path, capsys) -> None:
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    health = tmp_path / "controller-health"
    health.write_text(f"{datetime.now(UTC).timestamp()}\n", encoding="utf-8")

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "controller",
                "status",
                "--health-file",
                str(health),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["healthy"] is True
    assert payload["health_file"] == str(health)


def test_purge_preview_is_plan_only_and_blocked_until_inventory_is_complete(
    tmp_path: Path,
    capsys,
) -> None:
    project = tmp_path / "purge-project"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()

    assert main(["--json", "--project-root", str(project), "run", "purge", "missing-run"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["mode"] == "run"
    assert preview["blockers"] == ["run 'missing-run' was not found"]
    assert preview["purge_id"].startswith("purge-")

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "purge",
                "show",
                preview["purge_id"],
            ]
        )
        == 0
    )
    shown = json.loads(capsys.readouterr().out)
    assert shown["digest"] == preview["digest"]

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "purge",
                "apply",
                preview["purge_id"],
                "--expect-digest",
                preview["digest"],
                "--yes",
            ]
        )
        == 1
    )
    assert "blocked" in capsys.readouterr().err


def test_controller_loop_does_not_emit_idle_sweeps(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()

    async def sweep(_layout):
        return []

    monkeypatch.setattr("posttrain_cli.commands.controller.controller_sweep", sweep)

    def stop_after_first_sweep(_seconds: float) -> None:
        raise RuntimeError("stop test loop")

    monkeypatch.setattr("posttrain_cli.commands.controller.time.sleep", stop_after_first_sweep)

    assert main(["--project-root", str(project), "controller", "run"]) == 1
    assert capsys.readouterr().out == ""


def test_controller_sweep_ignores_legacy_entries_without_project_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Admission:
        def pump_available(self):
            return None

        def list(self):
            return [SimpleNamespace(run_id="legacy-run", state="submitted", control_locator=None)]

        def service(self, _run_id: str):
            raise AssertionError("legacy run must not reconstruct a provider")

    monkeypatch.setattr(
        "posttrain_cli.commands.controller.execution_admission_service",
        lambda _layout: Admission(),
    )

    assert asyncio.run(controller_sweep(cast(ProjectLayout, object()))) == []


def _record_submission(
    project: Path,
    *,
    run_id: str,
    evidence_source: ExecutionEvidenceSource | None,
) -> None:
    ExecutionSubmissionStore((project / ".posttrain" / "state").resolve()).save(
        ExecutionSubmission(
            run_id=run_id,
            provider="local-docker",
            provider_id=f"container-{run_id}",
            idempotency_key=f"{run_id}-attempt-1",
            job_image=f"registry.lan/posttrain@sha256:{'a' * 64}",
            submitted_at=datetime.now(UTC),
            evidence_source=evidence_source,
        )
    )


def _cli_execution_plan(run_id: str) -> ExecutionPlan:
    return ExecutionPlan(
        "local-docker",
        ExecutionRequest(
            run_spec=RunSpec(
                project_id="example",
                work_package_id="work/example",
                stage="train",
                run_id=run_id,
                job_kind="train.sft",
                job_definition_version="train/sft@1",
            ),
            job_definition_id="train/sft@1",
            image=RuntimeImageRef(f"registry.lan/posttrain@sha256:{'b' * 64}"),
            target=ExecutionTarget("targets/local", "1", "cuda", 24),
            command=(
                "posttrain-runtime",
                "execute",
                "--manifest",
                "/opt/posttrain/job/package.json",
            ),
            idempotency_key=f"{run_id}-attempt-1",
            policy=ExecutionPolicy(300),
        ),
    )


def _write_exact_execution_config(
    project: Path,
    *,
    variants: tuple[str, ...] = (
        "supervised",
        "online-rl-trl-py312",
        "online-rl-verl-py313",
        "eval",
        "serve",
        "transform",
    ),
) -> None:
    if not (project / "pyproject.toml").exists():
        package_name = project.name.replace("-", "_")
        source = project / "src" / package_name
        source.mkdir(parents=True)
        (source / "__init__.py").write_text('"""test project."""\n', encoding="utf-8")
        (project / "pyproject.toml").write_text(
            "\n".join(
                (
                    "[project]",
                    f'name = "{project.name}"',
                    'version = "0.1.0"',
                    "",
                    "[build-system]",
                    'requires = ["hatchling"]',
                    'build-backend = "hatchling.build"',
                    "",
                    "[tool.posttrain.pack]",
                    'project_packages = ["."]',
                    'source_includes = ["pyproject.toml", "src"]',
                    "",
                )
            ),
            encoding="utf-8",
        )
    state = project / ".posttrain" / "state"
    state.mkdir(parents=True, exist_ok=True)
    constraints = state / "constraints.txt"
    constraints.write_text("pydantic==2.12.5\n", encoding="utf-8")
    digest = hashlib.sha256(constraints.read_bytes()).hexdigest()
    image = f"registry.lan/carbonteq/posttrain@sha256:{'a' * 64}"
    repository = Path(__file__).resolve().parents[3]
    lines = [
        "schema_version = 1",
        "",
        "[registry]",
        'repository = "registry.lan/carbonteq/posttrain-job"',
        f'universal_image = "{image}"',
        f'framework_source_root = "{repository}"',
        "",
        "[registry.kind_images]",
        *(f'{profile} = "{image}"' for profile in variants),
        "",
    ]
    for profile in variants:
        lines.extend(
            (
                f"[registry.constraint_profiles.{profile}]",
                'path = "constraints.txt"',
                f'sha256 = "{digest}"',
                "",
            )
        )
    config = state / "execution.toml"
    config.write_text("\n".join(lines), encoding="utf-8")
    config.chmod(0o600)


def _write_target_overlay(
    project: Path,
    *,
    target_id: str = "targets/remote-rtx4090-24gb",
    memory_gb: int = 24,
) -> None:
    catalog = project / ".posttrain" / "catalog"
    (catalog / "targets.yaml").write_text(
        "\n".join(
            (
                "target:",
                f"  {target_id}:",
                '    revision: "1"',
                "    device_class: nvidia-cuda",
                f"    memory_gb: {memory_gb}",
                "    placement:",
                "      world_size: 1",
                "      instances:",
                "        - remote.lan",
                "",
            )
        ),
        encoding="utf-8",
    )
    (catalog / "layer.yaml").write_text(
        "\n".join(
            (
                "schema_version: 1",
                "layer_id: test-targets-v1",
                "files:",
                "  - targets.yaml",
                "",
            )
        ),
        encoding="utf-8",
    )


def _check_target(context: RunContext, seats: ResolvedSeats) -> dict[str, object]:
    target = seats["target"]
    if not isinstance(target, ExecutionTarget):
        raise TypeError("test host requires an execution target")
    world_size = target.placement["world_size"]
    if not isinstance(world_size, int):
        raise TypeError("test target world size must be an integer")
    context.metric("target/world_size", world_size)
    return {"device_class": target.device_class, "target_id": target.id}


def create_test_host(request: WorkPackageHostRequest) -> WorkPackageContext:
    return WorkPackageContext(
        catalog=request.catalog,
        definitions={
            "data/cpu-check@1": JobDefinition(
                "data/cpu-check@1",
                "data.prepare",
                {"target": ExecutionTarget},
                _check_target,
            )
        },
        source_metadata={"host": "cli-test"},
    )


def create_test_entry(request: WorkPackageHostRequest) -> WorkPackageContext:
    return build_job_runtime(
        request,
        tracking="none",
        extra_definitions=create_test_host(request).definitions,
    )


def test_init_creates_portable_project_and_valid_empty_overlay(
    tmp_path: Path,
    capsys,
) -> None:
    project = tmp_path / "Example Project"

    assert main(["init", str(project)]) == 0
    initialized = capsys.readouterr()

    assert f"Initialized post-training project example-project at {project.resolve()}" in initialized.out
    assert (project / ".posttrain" / "project.toml").is_file()
    assert "schema_version = 2" in (project / ".posttrain" / "project.toml").read_text(encoding="utf-8")
    assert 'project_brief = "project.yaml"' in (project / ".posttrain" / "project.toml").read_text(encoding="utf-8")
    assert (project / ".posttrain" / "project.yaml").is_file()
    assert (project / ".posttrain" / "catalog" / "layer.yaml").read_text(encoding="utf-8").endswith("files: []\n")
    assert (project / ".posttrain" / ".gitignore").read_text(encoding="utf-8") == "state/\n"
    runtime_environment = project / "posttrain.env"
    assert runtime_environment.read_text(encoding="utf-8") == ""
    assert runtime_environment.stat().st_mode & 0o077 == 0
    assert "posttrain.env" in (project / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "POSTTRAIN_REGISTRY" in (project / "posttrain.env.example").read_text(encoding="utf-8")
    assert main(["--project-root", str(project), "catalog", "validate"]) == 0
    validated = capsys.readouterr()
    assert "Catalog valid: framework-v1" in validated.out
    assert "0 project entries" in validated.out


def test_init_refuses_to_overwrite_existing_project(tmp_path: Path, capsys) -> None:
    project = tmp_path / "example"

    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    assert main(["init", str(project)]) == 1
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "refusing to overwrite existing project files" in captured.err


def test_machine_init_creates_shared_defaults_and_scoped_credentials(
    tmp_path: Path,
    capsys,
) -> None:
    from posttrain.catalog import load_project_layout
    from posttrain_cli.execution_config import load_execution_environment, load_local_execution_config

    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()

    assert (
        main(
            [
                "machine",
                "init",
                "--project",
                str(project),
                "--machine-name",
                "rtx-pro-96gb.lan",
                "--trackio-endpoint",
                "https://trackio.lan",
                "--python-index-url",
                "https://pypi.lan/simple/",
                "--job-registry",
                "registry.lan/posttrain",
                "--dstack-project",
                "main",
            ]
        )
        == 0
    )
    initialized = capsys.readouterr()
    assert "Initialized machine configuration" in initialized.out

    machine_root = Path(os.environ["XDG_CONFIG_HOME"]) / "posttrain"
    config = machine_root / "config.toml"
    credentials = machine_root / "credentials"
    assert config.stat().st_mode & 0o077 == 0o044
    assert not (project / ".posttrain" / "state" / "execution.toml").exists()
    assert "TRACKIO_WRITE_TOKEN" not in config.read_text(encoding="utf-8")
    for filename in ("trackio.env", "huggingface.env", "python-index.env", "dstack.env"):
        assert (credentials / filename).stat().st_mode & 0o077 == 0

    assert main(["--json", "machine", "show"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["name"] == "rtx-pro-96gb.lan"
    assert shown["credentials"]["dstack-default"].endswith("/credentials/dstack.env")
    assert "dstack-secret" not in json.dumps(shown)

    (credentials / "trackio.env").write_text("TRACKIO_WRITE_TOKEN=trackio-secret\n", encoding="utf-8")
    (credentials / "huggingface.env").write_text("HF_TOKEN=hf-secret\n", encoding="utf-8")
    (credentials / "python-index.env").write_text(
        "UV_INDEX_USERNAME=reader\nUV_INDEX_PASSWORD=index-secret\n",
        encoding="utf-8",
    )
    (credentials / "dstack.env").write_text("DSTACK_TOKEN=dstack-secret\n", encoding="utf-8")

    loaded = load_local_execution_config(load_project_layout(project))
    environment = load_execution_environment(loaded)
    assert environment["POSTTRAIN_TRACKIO_SERVER_URL"] == "https://trackio.lan"
    assert environment["TRACKIO_WRITE_TOKEN"] == "trackio-secret"
    assert environment["HF_TOKEN"] == "hf-secret"
    assert environment["UV_INDEX_USERNAME"] == "reader"
    assert environment["UV_INDEX_PASSWORD"] == "index-secret"
    assert environment["POSTTRAIN_REGISTRY"] == "registry.lan/posttrain"
    assert "DSTACK_TOKEN" not in environment
    assert loaded.dstack is not None
    assert loaded.dstack.environment_file == credentials / "dstack.env"

    second_project = tmp_path / "second"
    assert main(["init", str(second_project)]) == 0
    capsys.readouterr()
    assert main(["machine", "project", "add", str(second_project)]) == 0
    assert "Registered project" in capsys.readouterr().out
    assert main(["machine", "project", "add", str(second_project)]) == 0
    assert "already registered" in capsys.readouterr().out
    loaded_again = load_local_execution_config(load_project_layout(project))
    assert loaded_again.machine is not None
    assert loaded_again.machine.projects == (project.resolve(), second_project.resolve())

    assert main(["machine", "init"]) == 1
    assert "refusing to overwrite existing machine configuration" in capsys.readouterr().err


def test_machine_init_omits_redundant_hostname_by_default(tmp_path: Path, capsys) -> None:
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()

    assert main(["machine", "init", "--project", str(project)]) == 0
    capsys.readouterr()

    config = Path(os.environ["XDG_CONFIG_HOME"]) / "posttrain" / "config.toml"
    assert "machine_name" not in config.read_text(encoding="utf-8")


def test_environment_new_scaffolds_a_project_local_verifiers_package(tmp_path: Path, capsys) -> None:
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()

    assert main(["--json", "--project-root", str(project), "environment", "new", "kg-extract"]) == 0
    payload = json.loads(capsys.readouterr().out)
    root = project / "environments" / "kg-extract"
    assert payload["package"] == "kg-extract-env"
    assert (root / "pyproject.toml").is_file()
    assert "verifiers" in (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "create_environment" in (root / "src" / "kg_extract_env" / "taskset.py").read_text(encoding="utf-8")
    assert main(["--project-root", str(project), "environment", "new", "kg-extract"]) == 1


def test_mutating_run_selection_requires_a_complete_canonical_id(tmp_path: Path) -> None:
    from posttrain.catalog import load_project_layout
    from posttrain_cli.run_resolve import resolve_run_id

    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    layout = load_project_layout(project)
    known = ("01234567-89ab-cdef-0123-456789abcdef",)
    assert resolve_run_id(layout, known[0], known=known, exact_only=True) == known[0]
    with pytest.raises(ContractError, match="complete canonical"):
        resolve_run_id(layout, "01234567", known=known, exact_only=True)
    with pytest.raises(ContractError, match="--last is read-only"):
        resolve_run_id(layout, None, last=True, known=known, exact_only=True)


def test_global_env_file_option_reaches_the_command_state(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from posttrain_cli.commands import project_cmd

    project = tmp_path / "example"
    override = tmp_path / "alternate.env"
    override.write_text("POSTTRAIN_REGISTRY=registry.example/project\n", encoding="utf-8")
    override.chmod(0o600)
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    seen: list[Path | None] = []
    monkeypatch.setattr(project_cmd, "emit", lambda state, *_args: seen.append(state.env_file))

    assert main(["--project-root", str(project), "--env-file", str(override), "project", "show"]) == 0

    assert seen == [override]


def test_init_sft_template_writes_installable_project_and_valid_standard_job(
    tmp_path: Path,
    capsys,
) -> None:
    project = tmp_path / "sft-demo"

    assert (
        main(
            [
                "init",
                str(project),
                "--template",
                "sft",
                "--project-id",
                "sft-demo",
                "--no-install",
            ]
        )
        == 0
    )
    capsys.readouterr()

    pyproject = (project / "pyproject.toml").read_text(encoding="utf-8")
    settings = (project / ".posttrain" / "catalog" / "settings.yaml").read_text(encoding="utf-8")
    work_package = (project / ".posttrain" / "work_packages" / "sft.yaml").read_text(encoding="utf-8")
    assert '"posttrain[observatory,trackio,trl]' in pyproject
    assert "carbonteq-ai/trackio.git" not in pyproject
    assert "carbonteq-trackio" not in pyproject
    assert "carbonteq-ai/trl.git" not in pyproject
    assert "selection_type: sft-settings" in settings
    assert "datasets/posttrain-sft-smoke@1" in work_package
    assert "train/trl-sft@1" in work_package
    assert "posttrain_lab" not in pyproject + settings + work_package
    assert "posttrain.env" in (project / ".gitignore").read_text(encoding="utf-8")

    assert (
        main(
            [
                "--project-root",
                str(project),
                "work-package",
                "validate",
                "sft.yaml",
            ]
        )
        == 0
    )
    validated = capsys.readouterr()
    assert "Static composition validation: complete" in validated.out


def test_init_grpo_template_declares_environment_and_selected_extras(
    tmp_path: Path,
    capsys,
) -> None:
    project = tmp_path / "grpo-demo"

    assert (
        main(
            [
                "init",
                str(project),
                "--template",
                "grpo",
                "--project-id",
                "grpo-demo",
                "--no-install",
            ]
        )
        == 0
    )
    capsys.readouterr()

    pyproject = (project / "pyproject.toml").read_text(encoding="utf-8")
    work_package = (project / ".posttrain" / "work_packages" / "grpo.yaml").read_text(encoding="utf-8")
    assert '"posttrain[observatory,trackio,trl,verifiers]' in pyproject
    assert "PrimeIntellect-ai/verifiers.git@284a868d" in pyproject
    environment = (project / ".posttrain" / "catalog" / "environments.yaml").read_text(encoding="utf-8")
    assert "starter-gsm8k-train" in work_package
    assert "kind: project-path" in environment
    assert (project / "environments" / "starter-gsm8k" / "pyproject.toml").is_file()
    from posttrain.catalog import load_project_layout
    from posttrain.project import load_project_pack_config

    assert load_project_pack_config(load_project_layout(project)).environment_candidates == (
        "environments/starter-gsm8k",
    )
    assert "train/trl-grpo@1" in work_package
    assert "posttrain_lab" not in pyproject + work_package


def test_init_template_installs_with_uv(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    project = tmp_path / "sft-demo"
    calls: list[tuple[list[str], Path]] = []

    monkeypatch.setattr("posttrain_cli.cli.shutil.which", lambda command: f"/usr/bin/{command}")

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        check: bool,
        stdout=None,
        stderr=None,
    ) -> None:
        assert check is True
        assert stdout is None
        assert stderr is None
        calls.append((command, cwd))

    monkeypatch.setattr("posttrain_cli.cli.subprocess.run", fake_run)

    assert main(["init", str(project), "--template", "sft"]) == 0
    captured = capsys.readouterr()

    assert calls == [(["/usr/bin/uv", "sync", "--python", "3.13"], project.resolve())]
    assert captured.out.index("Initialized post-training project") < captured.out.index("Installing dependencies...")
    assert f"Environment ready: {project.resolve() / '.venv'}" in captured.out


def test_observatory_up_uses_discovered_project_tracking(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    import posttrain_observatory

    project = tmp_path / "observed"
    assert main(["init", str(project), "--project-id", "observed-project"]) == 0
    capsys.readouterr()
    served = []
    monkeypatch.setattr(posttrain_observatory, "serve", served.append)

    assert (
        main(
            [
                "--project-root",
                str(project),
                "observatory",
                "up",
                "--host",
                "127.0.0.1",
                "--port",
                "8787",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()

    assert captured.out == "Observatory listening at http://127.0.0.1:8787\n"
    assert len(served) == 1
    settings = served[0]
    assert settings.source == "trackio"
    assert settings.trackio_project == "observed-project"
    assert settings.source_id == "trackio-observed-project"


def test_observatory_up_rejects_project_without_tracking(
    tmp_path: Path,
    capsys,
) -> None:
    project = tmp_path / "untracked"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    manifest = project / ".posttrain" / "project.toml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace('tracking = "trackio"', 'tracking = "none"'),
        encoding="utf-8",
    )

    assert main(["--project-root", str(project), "observatory", "up"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Observatory requires project tracking" in captured.err


def test_project_show_discovers_from_nested_directory(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    project = tmp_path / "example"
    assert main(["init", str(project), "--project-id", "support-agent"]) == 0
    capsys.readouterr()
    nested = project / "src" / "jobs"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    assert main(["--json", "project", "show"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["project_id"] == "support-agent"
    assert Path(payload["root"]) == project.resolve()
    assert payload["catalog_overlays"] == [str((project / ".posttrain" / "catalog").resolve())]
    assert payload["serving_requirements"] == "not_configured"
    assert isinstance(payload["project_brief_digest"], str)


def test_catalog_list_and_show_include_resolution_provenance(
    tmp_path: Path,
    capsys,
) -> None:
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "catalog",
                "list",
                "--family",
                "target",
            ]
        )
        == 0
    )
    entries = json.loads(capsys.readouterr().out)
    assert entries
    assert {entry["source_layer"] for entry in entries} == {"base"}

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "catalog",
                "show",
                "target",
                "targets/local-cuda-8gb",
            ]
        )
        == 0
    )
    selection = json.loads(capsys.readouterr().out)
    assert selection["source_layer"] == "base"
    assert selection["selection"]["device_class"] == "nvidia-cuda"


def test_empty_overlay_lists_global_assets_and_dataset_validate_is_idempotent(
    tmp_path: Path,
    capsys,
) -> None:
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()

    for family in ("model", "dataset", "environment"):
        assert (
            main(
                [
                    "--json",
                    "--project-root",
                    str(project),
                    "catalog",
                    "list",
                    "--family",
                    family,
                ]
            )
            == 0
        )
        entries = json.loads(capsys.readouterr().out)
        assert entries
        assert {entry["source_layer"] for entry in entries} == {"base"}

    command = [
        "--json",
        "--project-root",
        str(project),
        "dataset",
        "validate",
        "datasets/posttrain-sft-smoke@1",
    ]
    assert main(command) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["materialized"] is True
    assert first["created"] is True
    assert first["examples"] == 2
    assert Path(first["path"]).is_file()

    assert main(command) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["materialized"] is False
    assert second["created"] is False
    assert second["path"] == first["path"]
    assert second["content_sha256"] == first["content_sha256"]


def test_dataset_materialize_verify_and_validate_alias(tmp_path: Path, capsys) -> None:
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()

    materialize = [
        "--json",
        "--project-root",
        str(project),
        "dataset",
        "materialize",
        "datasets/posttrain-sft-smoke@1",
    ]
    assert main(materialize) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["materialized"] is True
    assert isinstance(first["build_key"], str)
    manifest = Path(first["manifest"])
    before = manifest.stat().st_mtime_ns

    assert main(materialize) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["materialized"] is False
    assert second["content_sha256"] == first["content_sha256"]

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "dataset",
                "verify",
                "datasets/posttrain-sft-smoke@1",
            ]
        )
        == 0
    )
    verified = json.loads(capsys.readouterr().out)
    assert verified["verified"] is True
    assert verified["baseline_content_sha256"] == first["content_sha256"]
    assert Path(verified["path"]) == Path(first["path"])
    assert manifest.stat().st_mtime_ns == before

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "dataset",
                "validate",
                "datasets/posttrain-sft-smoke@1",
            ]
        )
        == 0
    )
    alias = json.loads(capsys.readouterr().out)
    assert alias["deprecated"] is True
    assert alias["replacement"].startswith("posttrain dataset materialize ")


def test_doctor_reports_readiness_and_missing_project(
    tmp_path: Path,
    capsys,
) -> None:
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()

    assert main(["--json", "--project-root", str(project), "doctor"]) == 0
    ready = json.loads(capsys.readouterr().out)
    assert ready["ok"] is True
    statuses = {check["name"]: check["status"] for check in ready["checks"]}

    # A fresh project has nowhere to publish job images yet. That blocks
    # submission but leaves validation entirely usable, so it is reportable
    # without being fatal.
    assert statuses["registry"] == "warn"
    assert {name: status for name, status in statuses.items() if name != "registry"} == {
        "python": "ok",
        "project": "ok",
        "catalog": "ok",
        "work_packages": "ok",
        "runtime_images": "warn",
        # No internal authority is installed on a machine running the tests, so
        # jobs trust public authorities only. That is a complete answer, not a
        # missing one.
        "trust": "ok",
        "catalog_overlays": "ok",
    }

    assert main(["--json", "--project-root", str(tmp_path / "missing"), "doctor"]) == 1
    missing = json.loads(capsys.readouterr().out)
    assert missing["ok"] is False
    assert any(check["name"] == "project" and check["status"] == "error" for check in missing["checks"])


def test_workers_names_the_holder_and_who_waits(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    from posttrain.execution import Placement

    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    now = datetime.now(UTC)

    class FakeAdmission:
        def placements(self):
            return (
                Placement(
                    key="host:pop-os.lan",
                    provider="local-docker",
                    holder="active-run",
                    holder_state="submitted",
                    holder_since=now,
                    waiting=("waiting-run",),
                ),
            )

    monkeypatch.setattr(
        "posttrain_cli.commands.workers.execution_admission_service",
        lambda layout: FakeAdmission(),
    )
    monkeypatch.setattr(
        "posttrain_cli.commands.workers.resolve_admission_state_root",
        lambda: tmp_path / "machine-admission",
    )

    assert main(["--json", "--project-root", str(project), "workers"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["admission_root"] == str(tmp_path / "machine-admission")
    assert payload["placements"] == [
        {
            "key": "host:pop-os.lan",
            "provider": "local-docker",
            "holder": "active-run",
            "holder_state": "submitted",
            "holder_since": now.isoformat(),
            "holder_message": None,
            "waiting": ["waiting-run"],
        }
    ]
    assert payload["orphaned_project_ledger"] is None

    assert main(["--project-root", str(project), "workers"]) == 0
    human = capsys.readouterr().out
    assert "host:pop-os.lan" in human
    assert "holder=active-run" in human
    assert "waiting-run" in human


def test_expected_errors_do_not_print_tracebacks(tmp_path: Path, capsys) -> None:
    assert (
        main(
            [
                "--project-root",
                str(tmp_path),
                "catalog",
                "show",
                "target",
                "targets/missing",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "Traceback" not in captured.err


def test_work_package_validate_resolves_project_catalog_seats(
    tmp_path: Path,
    capsys,
) -> None:
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    package = project / ".posttrain" / "work_packages" / "cpu-check.yaml"
    package.write_text(
        """
project_id: example
work_package_id: screen/cpu-check
stage: screen
recipe:
  type: inline
  id: recipes/cpu-check@1
  revision: "1"
  stage: screen
  seats: {model: model, dataset: dataset, settings: training, training: training}
  jobs:
    - {id: validate, kind: train.sft, definition: train/trl-sft@1}
bindings:
  model: {type: ref, family: model, id: models/qwen3.5-2b@bf16}
  dataset: {type: ref, family: dataset, id: datasets/posttrain-sft-smoke@1}
  settings: {type: ref, family: training, id: qwen3.5-2b/sft-smoke-v2}
  training: {type: ref, family: training, id: training/qwen3.5-trl-lora@1}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "work-package",
                "validate",
                "cpu-check.yaml",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["work_package_id"] == "screen/cpu-check"
    assert payload["resolved_seats"] == ["dataset", "model", "settings", "training"]
    assert payload["composition_validation"] == "complete"
    assert payload["validation_level"] == "project"


def test_work_package_validate_and_run_through_explicit_host(
    tmp_path: Path,
    capsys,
) -> None:
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    package = project / ".posttrain" / "work_packages" / "cpu-check.yaml"
    package.write_text(
        """
project_id: example
work_package_id: screen/cpu-check
stage: screen
recipe:
  type: inline
  id: recipes/cpu-check@1
  revision: "1"
  stage: screen
  seats: {target: target}
  jobs:
    - {id: validate, kind: data.prepare, definition: data/cpu-check@1}
bindings:
  target: {type: ref, family: target, id: targets/local-cuda-8gb}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    host = f"{__name__}:create_test_host"

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "work-package",
                "validate",
                "cpu-check.yaml",
                "--host",
                host,
            ]
        )
        == 0
    )
    validated = json.loads(capsys.readouterr().out)
    assert validated["validation_level"] == "host"
    assert validated["composition_validation"] == "complete"

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "work-package",
                "run",
                "cpu-check.yaml",
                "--job",
                "validate",
                "--in-process",
                "--host",
                host,
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "succeeded"
    assert result["project_id"] == "example"
    assert result["work_package_id"] == "screen/cpu-check"
    assert result["jobs"][0]["status"] == "succeeded"
    assert result["jobs"][0]["run_id"]
    assert result["jobs"][0]["value"] == {
        "device_class": "nvidia-cuda",
        "target_id": "targets/local-cuda-8gb",
    }


def test_work_package_run_uses_project_entry_without_host_flag(
    tmp_path: Path,
    capsys,
) -> None:
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    manifest = project / ".posttrain" / "project.toml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + f'entry = "{__name__}:create_test_entry"\n',
        encoding="utf-8",
    )
    package = project / ".posttrain" / "work_packages" / "cpu-check.yaml"
    package.write_text(
        """
project_id: example
work_package_id: screen/cpu-check
stage: screen
recipe:
  type: inline
  id: recipes/cpu-check@1
  revision: "1"
  stage: screen
  seats: {target: target}
  jobs:
    - {id: validate, kind: data.prepare, definition: data/cpu-check@1}
bindings:
  target: {type: ref, family: target, id: targets/local-cuda-8gb}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "work-package",
                "run",
                "cpu-check.yaml",
                "--job",
                "validate",
                "--in-process",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["host"] is None
    assert result["entry"] == f"{__name__}:create_test_entry"
    assert result["jobs"][0]["value"]["target_id"] == "targets/local-cuda-8gb"


def test_work_package_plan_is_read_only_and_reports_provenance(
    tmp_path: Path,
    capsys,
) -> None:
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    package = project / ".posttrain" / "work_packages" / "cpu-check.yaml"
    package.write_text(
        """
project_id: example
work_package_id: screen/cpu-check
stage: screen
recipe:
  type: inline
  id: recipes/cpu-check@1
  revision: "1"
  stage: screen
  seats: {target: target}
  jobs:
    - {id: validate, kind: data.prepare, definition: data/cpu-check@1}
bindings:
  target: {type: ref, family: target, id: targets/local-cuda-8gb}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    host = f"{__name__}:create_test_host"

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "work-package",
                "plan",
                "cpu-check.yaml",
                "--job",
                "validate",
                "--host",
                host,
            ]
        )
        == 0
    )
    planned = json.loads(capsys.readouterr().out)

    assert planned["project_id"] == "example"
    assert planned["job_id"] == "validate"
    assert planned["job_kind"] == "data.prepare"
    assert "provider" not in planned
    assert "pack" not in planned
    assert not (project / ".posttrain" / "state" / "pack").exists()


def test_grpo_plan_is_static_and_selects_online_rl_runtime(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    from posttrain.eval import PythonFactoryActivation
    from posttrain.eval.programs.general_smoke import GENERAL_ENVIRONMENT_FACTORIES

    project = tmp_path / "grpo-plan"
    assert (
        main(
            [
                "init",
                str(project),
                "--template",
                "grpo",
                "--project-id",
                "grpo-plan",
                "--no-install",
            ]
        )
        == 0
    )
    capsys.readouterr()
    monkeypatch.setitem(
        GENERAL_ENVIRONMENT_FACTORIES,
        "math-gsm8k-train",
        PythonFactoryActivation("environment_not_installed_during_planning:create_environment"),
    )
    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "work-package",
                "plan",
                "grpo.yaml",
                "--job",
                "train",
            ]
        )
        == 0
    )
    planned = json.loads(capsys.readouterr().out)

    assert planned["job_kind"] == "train.grpo"
    assert planned["job_definition_id"] == "train/trl-grpo@1"
    assert "pack" not in planned
    assert not (project / ".posttrain" / "state" / "pack").exists()

    assert (
        main(
            [
                "--project-root",
                str(project),
                "job",
                "plan",
                "grpo.yaml",
                "--job",
                "train",
                "--runtime-profile",
                "framework/online-rl-verl-py313@1",
            ]
        )
        == 1
    )
    assert "job plan resolves project job meaning only" in capsys.readouterr().err

    # Naming one variant no longer removes the rest. Every published job-kind
    # image is pinned by the release, so a partial machine binding overrides
    # only what it names and the remainder resolves from the installed
    # manifest. This is what removes the second hand transcription.
    _write_exact_execution_config(project, variants=("supervised",))
    from posttrain_cli.context import CliState
    from posttrain_cli.execution_planning import plan_job_package

    derived = plan_job_package(
        CliState(project_root=project),
        Path("grpo.yaml"),
        job="train",
    )
    manifest = load_manifest()
    assert derived.pack_plan.spec.kind_image.value == manifest.reference("online-rl-trl-py312")


def test_work_package_run_rejects_invalid_host(tmp_path: Path, capsys) -> None:
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    package = project / ".posttrain" / "work_packages" / "cpu-check.yaml"
    package.write_text(
        """
project_id: example
work_package_id: screen/cpu-check
stage: screen
recipe:
  type: inline
  id: recipes/cpu-check@1
  revision: "1"
  stage: screen
  seats: {target: target}
  jobs:
    - {id: validate, kind: data.prepare, definition: data/cpu-check@1}
bindings:
  target: {type: ref, family: target, id: targets/local-cuda-8gb}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--project-root",
                str(project),
                "work-package",
                "run",
                "cpu-check.yaml",
                "--job",
                "validate",
                "--host",
                "missing-host",
                "--in-process",
            ]
        )
        == 1
    )
    assert "MODULE:FACTORY" in capsys.readouterr().err


def test_work_package_validate_rejects_project_mismatch(tmp_path: Path, capsys) -> None:
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    package = project / ".posttrain" / "work_packages" / "mismatch.yaml"
    package.write_text(
        """
project_id: another-project
work_package_id: screen/mismatch
stage: screen
recipe:
  type: inline
  id: recipes/mismatch@1
  revision: "1"
  stage: screen
  seats: {target: target}
  jobs:
    - {id: validate, kind: data.prepare, definition: data/check@1}
bindings:
  target: {type: ref, family: target, id: targets/local-cuda-8gb}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--project-root",
                str(project),
                "work-package",
                "validate",
                "mismatch.yaml",
            ]
        )
        == 1
    )
    assert "does not match project manifest" in capsys.readouterr().err


def test_dataset_add_jsonl_and_catalog_materialize(tmp_path: Path, capsys) -> None:
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    data = project / "data"
    data.mkdir()
    (data / "train.jsonl").write_text(
        '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}\n',
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "dataset",
                "add",
                "jsonl",
                "--id",
                "datasets/local-sft@1",
                "--path",
                "data/train.jsonl",
                "--format",
                "messages",
            ]
        )
        == 0
    )
    added = json.loads(capsys.readouterr().out)
    assert added["id"] == "datasets/local-sft@1"
    assert added["source_kind"] == "jsonl"
    assert (project / ".posttrain" / "catalog" / "datasets.yaml").is_file()

    work_package = project / ".posttrain" / "work_packages" / "local_sft.yaml"
    work_package.write_text(
        "\n".join(
            (
                "project_id: example",
                "work_package_id: train/local-sft",
                "stage: train",
                "recipe:",
                "  type: inline",
                "  id: recipes/local-sft@1",
                '  revision: "1"',
                "  stage: train",
                "  seats:",
                "    dataset: dataset",
                "  jobs:",
                "    - id: train",
                "      kind: train.sft",
                "      definition: train/trl-sft@1",
                "bindings:",
                "  dataset:",
                "    type: ref",
                "    family: dataset",
                "    id: datasets/local-sft@1",
                "",
            )
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "catalog",
                "materialize",
                "--work-package",
                "local_sft.yaml",
            ]
        )
        == 0
    )
    materialized = json.loads(capsys.readouterr().out)
    assert materialized["items"]
    assert materialized["items"][0]["family"] == "dataset"
    assert materialized["items"][0]["id"] == "datasets/local-sft@1"
    assert materialized["items"][0]["status"] == "materialized"


def test_workload_commands_delegate_to_serve_owned_operations(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    seen: dict[str, object] = {}

    def result(*, materialized: bool, path: str) -> WorkloadMaterialization:
        return WorkloadMaterialization(
            workload_id="workloads/general-serving-32k-sweep@1",
            workload_revision="1",
            corpus_id="general-serving-v1",
            corpus_revision="1",
            record_count=128,
            content_sha256="9a9467fd8a5e744968d09a4d8fd6f4d92a089c50a84e1e6e7e5c5520a9f4e50e",
            path=path,
            manifest=path.replace(".jsonl", ".manifest.json"),
            materialized=materialized,
        )

    def fake_materialize(workload, *, output: Path) -> WorkloadMaterialization:
        seen["materialize_workload"] = workload.id
        seen["output"] = output
        return result(materialized=True, path=str(output / "general-serving-v1.jsonl"))

    def fake_verify(workload) -> WorkloadMaterialization:
        seen["verify_workload"] = workload.id
        return result(materialized=False, path="packaged/general-serving-v1.jsonl")

    monkeypatch.setattr("posttrain_cli.commands.workload.materialize_workload", fake_materialize)
    monkeypatch.setattr("posttrain_cli.commands.workload.verify_workload", fake_verify)

    command = [
        "--json",
        "--project-root",
        str(project),
        "workload",
        "materialize",
        "workloads/general-serving-32k-sweep@1",
    ]
    assert main(command) == 0
    materialized = json.loads(capsys.readouterr().out)
    assert materialized["materialized"] is True
    assert seen["output"] == project / ".posttrain/state/workloads/general-serving-32k-sweep-1"

    command[4] = "verify"
    assert main(command) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["verified"] is True
    assert seen["materialize_workload"] == seen["verify_workload"]


def test_environment_add_local_writes_overlay(tmp_path: Path, capsys) -> None:
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "environment",
                "add",
                "local",
                "--id",
                "custom-gsm8k",
                "--package",
                "gsm8k-v1",
                "--factory",
                "custom_gsm8k:load_environment",
                "--repository",
                "https://github.com/PrimeIntellect-ai/verifiers",
                "--revision",
                "284a868d6a9022109b749710672a0460e8a996d4",
                "--subdirectory",
                "environments/gsm8k_v1",
                "--category",
                "math-reasoning",
                "--num-tasks",
                "2",
            ]
        )
        == 0
    )
    added = json.loads(capsys.readouterr().out)
    assert added["id"] == "custom-gsm8k"
    assert added["package"] == "gsm8k-v1"
    assert added["activation"]["kind"] == "python-factory"
    assert (project / ".posttrain" / "catalog" / "environments.yaml").is_file()


def test_job_plan_aliases_work_package_plan(tmp_path: Path, capsys) -> None:
    from posttrain.project import Project

    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    package = project / ".posttrain" / "work_packages" / "cpu-check.yaml"
    package.write_text(
        """
project_id: example
work_package_id: screen/cpu-check
stage: screen
recipe:
  type: inline
  id: recipes/cpu-check@1
  revision: "1"
  stage: screen
  seats: {model: model, dataset: dataset, settings: training, training: training}
  jobs:
    - {id: validate, kind: train.sft, definition: train/trl-sft@1}
bindings:
  model: {type: ref, family: model, id: models/qwen3.5-2b@bf16}
  dataset: {type: ref, family: dataset, id: datasets/posttrain-sft-smoke@1}
  settings: {type: ref, family: training, id: qwen3.5-2b/sft-smoke-v2}
  training: {type: ref, family: training, id: training/qwen3.5-trl-lora@1}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    intent = Project.open(project).jobs.plan("cpu-check.yaml", job="validate")

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "job",
                "plan",
                "cpu-check.yaml",
                "--job",
                "validate",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["work_package_id"] == "screen/cpu-check"
    assert "provider" not in payload
    assert "pack" not in payload
    assert payload["job_id"] == "validate"
    assert payload["job_kind"] == intent.prepared.recipe_job.kind
    assert payload["job_definition_id"] == intent.prepared.definition.id
    assert payload["work_package_id"] == intent.prepared.spec.work_package_id


def test_job_package_plan_target_override_changes_nested_sft_target_and_identity(
    tmp_path: Path,
    capsys,
) -> None:
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    package = project / ".posttrain" / "work_packages" / "sft-target.yaml"
    package.write_text(
        """
project_id: example
work_package_id: train/sft-target
stage: train
recipe:
  type: inline
  id: recipes/sft-target@1
  revision: "1"
  stage: train
  seats: {model: model, dataset: dataset, settings: training, training: training}
  jobs:
    - {id: train, kind: train.sft, definition: train/trl-sft@1}
bindings:
  model: {type: ref, family: model, id: models/qwen3.5-2b@bf16}
  dataset: {type: ref, family: dataset, id: datasets/posttrain-sft-smoke@1}
  settings: {type: ref, family: training, id: qwen3.5-2b/sft-smoke-v2}
  training: {type: ref, family: training, id: training/qwen3.5-trl-lora@1}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _write_target_overlay(project)
    _write_exact_execution_config(project)

    from posttrain_cli.context import CliState
    from posttrain_cli.execution_config import PackageOverrides
    from posttrain_cli.execution_planning import _project_config_bundle, plan_job_package

    baseline = plan_job_package(
        CliState(project_root=project),
        Path("sft-target.yaml"),
        job="train",
    )
    overridden = plan_job_package(
        CliState(project_root=project),
        Path("sft-target.yaml"),
        job="train",
        overrides=PackageOverrides(target="targets/remote-rtx4090-24gb"),
    )

    assert baseline.target.id == "targets/local-cuda-8gb"
    assert overridden.target.id == "targets/remote-rtx4090-24gb"
    assert overridden.pack_plan.plan_key != baseline.pack_plan.plan_key
    selected_config = _project_config_bundle(
        overridden.layout,
        overridden.work_package_path,
        overridden.prepared,
        overridden.catalog,
    )
    assert overridden.project_config_digest == selected_config.digest
    catalog_dir = project / ".posttrain" / "catalog"
    (catalog_dir / "unrelated.yaml").write_text(
        "target:\n  targets/unrelated-8gb:\n    revision: '1'\n    device_class: nvidia-cuda\n    memory_gb: 8\n",
        encoding="utf-8",
    )
    layer_path = catalog_dir / "layer.yaml"
    layer_path.write_text(
        layer_path.read_text(encoding="utf-8").replace("  - targets.yaml\n", "  - targets.yaml\n  - unrelated.yaml\n"),
        encoding="utf-8",
    )
    assert (
        _project_config_bundle(
            overridden.layout,
            overridden.work_package_path,
            overridden.prepared,
            overridden.catalog,
        ).files
        == selected_config.files
    )
    targets_path = catalog_dir / "targets.yaml"
    targets_path.write_text(
        targets_path.read_text(encoding="utf-8").replace("memory_gb: 24", "memory_gb: 48"), encoding="utf-8"
    )
    changed_config = _project_config_bundle(
        overridden.layout,
        overridden.work_package_path,
        overridden.prepared,
        overridden.catalog,
    )
    assert changed_config.files != selected_config.files
    assert changed_config.digest != overridden.project_config_digest
    assert not (project / ".posttrain" / "state" / "pack").exists()


def test_job_pack_publishes_actual_image_without_opening_a_provider(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    package = project / ".posttrain" / "work_packages" / "cpu-check.yaml"
    package.write_text(
        """
project_id: example
work_package_id: screen/cpu-check
stage: screen
recipe:
  type: inline
  id: recipes/cpu-check@1
  revision: "1"
  stage: screen
  seats: {target: target}
  jobs:
    - {id: validate, kind: data.prepare, definition: data/cpu-check@1}
bindings:
  target: {type: ref, family: target, id: targets/local-cuda-8gb}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _write_exact_execution_config(project)
    execution_config = project / ".posttrain" / "state" / "execution.toml"
    local_execution_config = execution_config.read_text(encoding="utf-8")
    execution_config.write_text(
        local_execution_config.replace(
            "schema_version = 1\n",
            'schema_version = 1\n\n[defaults]\nprovider = "dstack"\n',
            1,
        ),
        encoding="utf-8",
    )
    execution_config.chmod(0o600)
    actual_image = RuntimeImageRef(f"registry.lan/carbonteq/posttrain-job@sha256:{'9' * 64}")

    class FakePackService:
        def __init__(self, **_kwargs) -> None:
            pass

        def pack(self, plan, _inputs):
            root = (project / ".posttrain/state/fake-context").resolve()
            root.mkdir(parents=True, exist_ok=True)
            manifest = JobPackageManifest(
                project_id=plan.spec.project_id,
                work_package_id=plan.spec.work_package_id,
                job_id=plan.spec.job_id,
                job_definition_id=plan.spec.job_definition_id,
                job_kind=plan.spec.job_kind,
                resolved_inputs_digest=plan.spec.resolved_inputs_digest,
                framework_source_digest=plan.spec.framework_source_digest,
                project_source_digest=plan.spec.project_source_digest,
                runtime_dependencies_digest="1" * 64,
                code_requirements_digest="2" * 64,
                resolved_config_digest="3" * 64,
                project_config_digest="4" * 64,
                universal_image=plan.spec.universal_image,
                kind_image=plan.spec.kind_image,
                runtime_variant=plan.spec.runtime_variant,
                expected_artifact_roles=plan.spec.expected_artifact_roles,
            )
            publication_key = hashlib.sha256(
                json.dumps(
                    {
                        "package_key": manifest.package_key,
                        "publication": plan.publication.to_payload(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            return PackedJobContext(root, manifest, "5" * 64, publication_key)

    class FakePublisher:
        remote_publications: list[str] = []
        local_publications: list[str] = []

        def __init__(self, **_kwargs) -> None:
            pass

        def publish(self, request):
            self.remote_publications.append(request.publication_key)
            receipt = (project / ".posttrain/state/fake-receipt.json").resolve()
            receipt.write_text("{}\n", encoding="utf-8")
            receipt.chmod(0o600)
            return PublishedJobImage(
                request.package_key,
                request.publication_key,
                actual_image,
                request.manifest.kind_image,
                receipt,
                False,
            )

        def publish_local(self, request):
            self.local_publications.append(request.publication_key)
            layout = (project / ".posttrain/state/fake-local-layout").resolve()
            layout.mkdir(parents=True, exist_ok=True)
            (layout / "index.json").write_text("{}\n", encoding="utf-8")
            receipt = (project / ".posttrain/state/fake-local-receipt.json").resolve()
            receipt.write_text("{}\n", encoding="utf-8")
            receipt.chmod(0o600)
            return LocalPublishedJobImage(
                request.package_key,
                request.publication_key,
                layout,
                f"posttrain-local:{request.publication_key}",
                receipt,
                False,
            )

    # The registry in this fixture holds no real images. Kind-image verification
    # is exercised directly in test_runtime_images.py; here it is stubbed so the
    # test stays about packing.
    monkeypatch.setattr(
        "posttrain_cli.commands.work_package.ensure_kind_image_ready",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "posttrain_cli.execution_planning.JobPackService",
        FakePackService,
    )
    monkeypatch.setattr(
        "posttrain_cli.execution_planning.BuildKitJobImagePublisher",
        FakePublisher,
    )
    monkeypatch.setattr(
        "posttrain_cli.execution_planning.create_execution_provider",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("job pack must not open an execution provider")),
    )

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "job",
                "pack",
                "cpu-check.yaml",
                "--job",
                "validate",
                "--host",
                f"{__name__}:create_test_host",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["images"]["actual_job"] == actual_image.value
    assert payload["package"]["cache_hit"] is False
    assert "provider" not in payload
    assert "mounts" not in payload

    from posttrain.catalog import load_project_layout
    from posttrain_cli import execution_planning
    from posttrain_cli.execution_config import load_local_execution_config

    configured = load_local_execution_config(load_project_layout(project))
    original_config_loader = execution_planning.load_local_execution_config
    monkeypatch.setattr(
        "posttrain_cli.execution_planning.load_local_execution_config",
        lambda *_args, **_kwargs: replace(configured, registry=None),
    )

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "job",
                "pack",
                "cpu-check.yaml",
                "--job",
                "validate",
                "--local",
                "--host",
                f"{__name__}:create_test_host",
            ]
        )
        == 0
    )
    local_payload = json.loads(capsys.readouterr().out)
    assert local_payload["images"]["actual_job"] is None
    assert local_payload["images"]["local_oci"]["tag"].startswith("posttrain-local:")
    assert FakePublisher.local_publications == [local_payload["package"]["publication_key"]]
    monkeypatch.setattr(
        "posttrain_cli.execution_planning.load_local_execution_config",
        original_config_loader,
    )

    assert (
        main(
            [
                "--project-root",
                str(project),
                "job",
                "plan",
                "cpu-check.yaml",
                "--job",
                "validate",
                "--host",
                f"{__name__}:create_test_host",
            ]
        )
        == 0
    )
    assert "Job intent: screen/cpu-check/validate" in capsys.readouterr().out

    execution_config.write_text(local_execution_config, encoding="utf-8")
    execution_config.chmod(0o600)

    from posttrain_cli.context import CliState
    from posttrain_cli.execution_config import LaunchOverrides
    from posttrain_cli.execution_planning import (
        plan_job_launch,
        plan_job_package,
    )

    reusable_package = plan_job_package(
        CliState(project_root=project),
        Path("cpu-check.yaml"),
        job="validate",
        host=f"{__name__}:create_test_host",
    )
    first_launch = plan_job_launch(
        reusable_package,
        overrides=LaunchOverrides(provider="local", timeout_seconds=120),
        run_id="capsule-launch-a",
    )
    second_launch = plan_job_launch(
        reusable_package,
        overrides=LaunchOverrides(provider="local", timeout_seconds=240),
        run_id="capsule-launch-b",
    )
    assert reusable_package.pack_plan.plan_key
    assert first_launch.run_spec.run_id == "capsule-launch-a"
    assert second_launch.run_spec.run_id == "capsule-launch-b"
    assert first_launch.settings.timeout_seconds == 120
    assert second_launch.settings.timeout_seconds == 240
    assert first_launch.mounts[0].instance_path.name == "capsule-launch-a"
    assert second_launch.mounts[0].instance_path.name == "capsule-launch-b"

    remote_before = tuple(FakePublisher.remote_publications)
    local = reusable_package.pack_local()
    assert local.image.layout.joinpath("index.json").is_file()
    assert tuple(FakePublisher.remote_publications) == remote_before
    assert FakePublisher.local_publications == [
        local_payload["package"]["publication_key"],
        local.context.publication_key,
    ]

    observed_requests = []

    class FakeProvider:
        def plan(self, request):
            observed_requests.append(request)
            return ExecutionPlan("fake-provider", request)

        def submit(self, plan):
            return ExecutionHandle(
                "fake-provider",
                "provider-job-1",
                plan.request.idempotency_key,
            )

    monkeypatch.setattr(
        "posttrain_cli.execution_planning.create_execution_provider",
        lambda *_args, **_kwargs: ("fake-provider", FakeProvider()),
    )
    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "job",
                "run",
                "cpu-check.yaml",
                "--job",
                "validate",
                "--host",
                f"{__name__}:create_test_host",
                "--run-id",
                "actual-image-run",
            ]
        )
        == 0
    )
    submitted = json.loads(capsys.readouterr().out)
    assert submitted["status"] == "submitted"
    assert submitted["images"]["actual_job"] == actual_image.value
    assert len(observed_requests) == 1
    assert observed_requests[0].bundle is None
    assert observed_requests[0].image == actual_image


def test_run_lifecycle_commands_use_the_canonical_run_id(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    calls: list[tuple[str, object]] = []
    handle = ExecutionHandle("local-docker", "container-1", "key-1")

    class FakeService:
        def status(self, run_id: str) -> ExecutionRecord:
            calls.append(("status", run_id))
            return ExecutionRecord(
                handle,
                "running",
                1,
                "localhost",
                datetime.now(UTC),
                "running",
            )

        def logs(
            self,
            run_id: str,
            cursor: LogCursor,
            *,
            limit: int,
        ) -> LogPage:
            calls.append(("logs", (run_id, cursor.offset, limit)))
            return LogPage(("line-one",), LogCursor(cursor.offset + 1), False)

        def cancel(self, run_id: str) -> None:
            calls.append(("cancel", run_id))

    source = object()
    terminal_record = ExecutionRecord(
        handle,
        "succeeded",
        1,
        "localhost",
        datetime.now(UTC),
        "exited",
    )

    async def fake_reconcile(service, evidence_source, run_id):
        assert isinstance(service, FakeService)
        assert evidence_source is source
        calls.append(("reconcile", run_id))
        return ExecutionReconciliation(
            run_id=run_id,
            state="consistent",
            outcome="succeeded",
            provider_record=terminal_record,
            provider_exit_code=0,
            tracking_status="succeeded",
            tracking_provider_run_id="trackio-1",
            required_artifact_roles=(),
            retained_artifacts=(),
            missing_artifact_roles=(),
            observed_at=datetime.now(UTC),
            message="provider and retained evidence are consistent",
        )

    recovery_writer = object()

    async def fake_recover(
        service,
        evidence_source,
        writer,
        run_id,
        *,
        project_id,
    ):
        assert isinstance(service, FakeService)
        assert evidence_source is source
        assert writer is recovery_writer
        assert project_id == "example"
        calls.append(("recover", run_id))
        return TrackingCancellationRecovery(
            run_id=run_id,
            disposition="recovered",
            execution_provider="local-docker",
            execution_provider_id="container-1",
            tracking_provider="trackio",
            tracking_provider_run_id="trackio-1",
            tracking_started_at=datetime.now(UTC),
            recovered_at=datetime.now(UTC),
        )

    async def fake_cleanup(
        service,
        store,
        evidence_source,
        run_id,
        *,
        diagnostic_limit,
    ):
        assert isinstance(service, FakeService)
        assert evidence_source is source
        assert diagnostic_limit == 500
        assert store.run_root(run_id).name == run_id
        calls.append(("cleanup", run_id))
        return ExecutionCleanupReceipt(
            run_id=run_id,
            outcome="succeeded",
            evidence_state="reconciled",
            provider="local-docker",
            provider_id="container-1",
            provider_disposition="removed",
            workspace_disposition="removed",
            workspace_reclaimed_bytes=64,
            reconciliation_file="reconciliation.json",
            retained_artifact_count=2,
            diagnostic_file=None,
            diagnostic_digest=None,
            diagnostic_line_count=0,
            diagnostic_truncated=False,
            completed_at=datetime.now(UTC),
        )

    monkeypatch.setattr(
        "posttrain_cli.commands.run_cmd.execution_service_for_run",
        lambda layout, run_id: FakeService(),
    )
    monkeypatch.setattr(
        "posttrain_cli.commands.run_cmd.tracking_source_for_run",
        lambda layout, run_id: source,
    )
    monkeypatch.setattr(
        "posttrain_cli.commands.run_cmd.reconciliation_source_for_run",
        lambda layout, run_id: source,
    )
    monkeypatch.setattr(
        "posttrain_cli.commands.run_cmd.reconcile_execution",
        fake_reconcile,
    )
    monkeypatch.setattr(
        "posttrain_cli.commands.run_cmd.cancelled_tracking_writer_for_run",
        lambda layout, run_id: recovery_writer,
    )
    monkeypatch.setattr(
        "posttrain_cli.commands.run_cmd.recover_cancelled_tracking",
        fake_recover,
    )
    monkeypatch.setattr(
        "posttrain_cli.commands.run_cmd.cleanup_execution",
        fake_cleanup,
    )

    for command in (
        ["run", "status", "run-1"],
        ["run", "logs", "run-1", "--offset", "3", "--limit", "1"],
        ["run", "cancel", "run-1"],
        ["run", "recover-cancelled-tracking", "run-1"],
        ["run", "reconcile", "run-1"],
        ["run", "cleanup", "run-1"],
    ):
        assert (
            main(
                [
                    "--json",
                    "--project-root",
                    str(project),
                    *command,
                ]
            )
            == 0
        )
        capsys.readouterr()

    assert calls == [
        ("status", "run-1"),
        ("logs", ("run-1", 3, 1)),
        ("cancel", "run-1"),
        ("recover", "run-1"),
        ("reconcile", "run-1"),
        ("cleanup", "run-1"),
    ]
    reconciliation = project / ".posttrain" / "state" / "executions" / "run-1" / "reconciliation.json"
    assert reconciliation.is_file()
    assert reconciliation.stat().st_mode & 0o777 == 0o600
    recovery = project / ".posttrain" / "state" / "executions" / "run-1" / "tracking-recovery.json"
    assert recovery.is_file()
    assert recovery.stat().st_mode & 0o777 == 0o600


def test_run_list_and_wait_make_submitted_runs_discoverable(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    source = ExecutionEvidenceSource(
        provider="trackio",
        source_id="trackio-example",
        project="example",
        endpoint="https://trackio.example.test",
    )
    _record_submission(project, run_id="run-older", evidence_source=source)
    _record_submission(project, run_id="run-newer", evidence_source=source)

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "run",
                "list",
            ]
        )
        == 0
    )
    listed = json.loads(capsys.readouterr().out)
    assert {item["run_id"] for item in listed} == {"run-older", "run-newer"}
    assert [item["run_id"] for item in listed] == ["run-newer", "run-older"]
    assert all(item["tracking"] == "trackio" for item in listed)

    assert (
        main(
            [
                "--project-root",
                str(project),
                "run",
                "list",
                "--limit",
                "1",
            ]
        )
        == 0
    )
    human = capsys.readouterr().out
    assert "run-newer  local-docker  state=legacy-submitted" in human
    assert "submitted=" in human

    class FakeWaitService:
        def wait(
            self,
            run_id: str,
            *,
            timeout_seconds: float,
            poll_interval_seconds: float,
            cancel_on_timeout: bool,
        ) -> ExecutionRecord:
            assert run_id == "run-newer"
            assert timeout_seconds == 30
            assert poll_interval_seconds == 0.5
            assert cancel_on_timeout is False
            return ExecutionRecord(
                ExecutionHandle("local-docker", "container-run-newer", "key"),
                "succeeded",
                1,
                "localhost",
                datetime.now(UTC),
                "exited",
            )

    monkeypatch.setattr(
        "posttrain_cli.commands.run_cmd.execution_service_for_run",
        lambda layout, run_id: FakeWaitService(),
    )
    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "run",
                "wait",
                "run-newer",
                "--timeout-seconds",
                "30",
                "--poll-interval-seconds",
                "0.5",
            ]
        )
        == 0
    )
    waited = json.loads(capsys.readouterr().out)
    assert waited["state"] == "succeeded"


def test_run_commands_keep_current_admission_visible_and_idempotent(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    _record_submission(project, run_id="historical-run", evidence_source=None)
    now = datetime.now(UTC)
    waiting = AdmissionEntry(
        run_id="waiting-run",
        state="waiting",
        plan=_cli_execution_plan("waiting-run"),
        evidence_source=None,
        queued_at=now,
        position=1,
    )
    completed = AdmissionEntry(
        run_id="completed-run",
        state="completed",
        plan=_cli_execution_plan("completed-run"),
        evidence_source=None,
        queued_at=now,
    )
    failed = AdmissionEntry(
        run_id="failed-submit-run",
        state="submission_failed",
        plan=_cli_execution_plan("failed-submit-run"),
        evidence_source=None,
        queued_at=now,
        message="RuntimeError: provider submission failed",
    )
    terminal = ExecutionRecord(
        ExecutionHandle("local-docker", "container-completed", "key"),
        "succeeded",
        1,
        "localhost",
        now,
        "exited",
    )

    class FakeAdmission:
        def list(self):
            return (waiting, completed, failed)

        def status(self, run_id):
            assert run_id == "completed-run"
            return completed, terminal

        def get(self, run_id):
            assert run_id == "completed-run"
            return completed

        def cancel(self, run_id):
            assert run_id == "completed-run"
            return completed

        def retry_submission(self, run_id):
            assert run_id == "failed-submit-run"
            return AdmissionResult(
                AdmissionEntry(
                    run_id=failed.run_id,
                    state="submitted",
                    plan=failed.plan,
                    evidence_source=None,
                    queued_at=failed.queued_at,
                )
            )

    admission = FakeAdmission()
    monkeypatch.setattr(
        "posttrain_cli.commands.run_cmd.execution_admission_service",
        lambda layout: admission,
    )

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "run",
                "list",
                "--limit",
                "1",
            ]
        )
        == 0
    )

    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["run_id"] == "waiting-run"
    assert listed[0]["queue_position"] == 1
    assert listed[0]["requested_hostnames"] == []

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "run",
                "queue",
            ]
        )
        == 0
    )
    queue = json.loads(capsys.readouterr().out)
    assert queue == [
        {
            "admission_state": "waiting",
            "assigned_hostname": None,
            "message": None,
            "provider": "local-docker",
            "provider_id": None,
            "provider_state": None,
            "queue_position": 1,
            "queue_scope": "framework",
            "queued_at": now.isoformat(),
            "requested_hostnames": [],
            "requested_target_id": "targets/local",
            "run_id": "waiting-run",
        }
    ]

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "run",
                "status",
                "completed-run",
            ]
        )
        == 0
    )
    status = json.loads(capsys.readouterr().out)
    assert status["state"] == "succeeded"
    assert status["admission_state"] == "completed"

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "run",
                "cancel",
                "completed-run",
            ]
        )
        == 0
    )
    cancelled = json.loads(capsys.readouterr().out)
    assert cancelled["status"] == "already-complete"

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "run",
                "retry-submit",
                "failed-submit-run",
            ]
        )
        == 0
    )
    retried = json.loads(capsys.readouterr().out)
    assert retried["state"] == "submitted"


def test_run_list_scopes_project_and_labels_purged_history(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    _record_submission(project, run_id="kept-run", evidence_source=None)
    now = datetime.now(UTC)
    purged = AdmissionEntry(
        run_id="purged-run",
        state="completed",
        plan=_cli_execution_plan("purged-run"),
        evidence_source=None,
        queued_at=now,
    )
    foreign_base = _cli_execution_plan("foreign-run")
    foreign_plan = replace(
        foreign_base,
        request=replace(
            foreign_base.request,
            run_spec=replace(foreign_base.request.run_spec, project_id="other-project"),
        ),
    )
    foreign = replace(purged, run_id="foreign-run", plan=foreign_plan)

    class FakeAdmission:
        def list(self):
            return (purged, foreign)

    monkeypatch.setattr(
        "posttrain_cli.commands.run_cmd.execution_admission_service",
        lambda layout: FakeAdmission(),
    )
    monkeypatch.setattr(
        "posttrain_cli.commands.run_cmd.purged_run_ids",
        lambda layout: {"purged-run"},
    )

    assert main(["--json", "--project-root", str(project), "run", "list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert {item["run_id"] for item in listed} == {"kept-run"}

    assert main(["--json", "--project-root", str(project), "run", "list", "--include-purged"]) == 0
    audit = json.loads(capsys.readouterr().out)
    assert {item["run_id"] for item in audit} == {"kept-run", "purged-run"}
    assert next(item for item in audit if item["run_id"] == "purged-run")["purged"] is True


def test_run_show_uses_project_tracking_source(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    import posttrain_observatory

    project = tmp_path / "example"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    state_root = project / ".posttrain" / "state"
    state_root.mkdir(exist_ok=True)
    (state_root / "job.env").write_text(
        "POSTTRAIN_TRACKIO_SERVER_URL=https://trackio.example.test\n",
        encoding="utf-8",
    )
    (state_root / "job.env").chmod(0o600)
    (state_root / "execution.toml").write_text(
        'schema_version = 1\nenvironment_file = "job.env"\n',
        encoding="utf-8",
    )
    (state_root / "execution.toml").chmod(0o600)
    _record_submission(
        project,
        run_id="run-1",
        evidence_source=ExecutionEvidenceSource(
            provider="trackio",
            source_id="trackio-example",
            project="example",
            endpoint="https://trackio.example.test",
        ),
    )
    observed_settings = []

    class FakeView:
        def model_dump(self, *, mode: str = "python") -> dict[str, object]:
            assert mode == "json"
            return {"run_id": "run-1", "source_id": "trackio-example"}

    class FakeService:
        async def get_run_view_response(self, locator, mode):
            assert locator.source_id == "trackio-example"
            assert locator.run_id == "run-1"
            assert mode == "auto"
            return FakeView()

    def create_service(settings):
        observed_settings.append(settings)
        return FakeService()

    monkeypatch.setattr(posttrain_observatory, "create_service", create_service)

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "run",
                "show",
                "run-1",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"run_id": "run-1", "source_id": "trackio-example"}
    assert observed_settings[0].trackio_server_url == "https://trackio.example.test"


def test_run_show_rejects_project_without_tracking(tmp_path: Path, capsys) -> None:
    project = tmp_path / "untracked"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    manifest = project / ".posttrain" / "project.toml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace('tracking = "trackio"', 'tracking = "none"'),
        encoding="utf-8",
    )
    _record_submission(project, run_id="run-1", evidence_source=None)

    assert main(["--project-root", str(project), "run", "show", "run-1"]) == 1
    assert "was submitted with tracking disabled" in capsys.readouterr().err
