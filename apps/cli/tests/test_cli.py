"""Tests for the primary posttrain command-line interface."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from posttrain.common import ExecutionTarget, RunContext
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
from posttrain.execution_pack import PackedJobContext, PublishedJobImage
from posttrain.jobs import build_job_runtime
from posttrain.tracking import RunSpec
from posttrain.work import (
    JobDefinition,
    ResolvedSeats,
    WorkPackageContext,
    WorkPackageHostRequest,
)
from posttrain_cli.cli import main


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
            image=RuntimeImageRef(
                f"registry.lan/posttrain@sha256:{'b' * 64}"
            ),
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
        *(
            f'{profile} = "{image}"'
            for profile in variants
        ),
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
    assert "carbonteq-ai/trackio.git@9b0c4af0" in pyproject
    assert "carbonteq-ai/trl.git@6e7739b8" in pyproject
    assert "selection_type: sft-settings" in settings
    assert "datasets/posttrain-sft-smoke@1" in work_package
    assert "train/trl-sft@1" in work_package
    assert "posttrain_lab" not in pyproject + settings + work_package

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
    assert "gsm8k-distill-train" in work_package
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

    assert calls == [(["/usr/bin/uv", "sync", "--python", "3.12"], project.resolve())]
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
    assert first["examples"] == 2
    assert Path(first["path"]).is_file()

    assert main(command) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["materialized"] is False
    assert second["path"] == first["path"]
    assert second["content_sha256"] == first["content_sha256"]


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
    assert {check["status"] for check in ready["checks"]} == {"ok"}

    assert main(["--json", "--project-root", str(tmp_path / "missing"), "doctor"]) == 1
    missing = json.loads(capsys.readouterr().out)
    assert missing["ok"] is False
    assert any(check["name"] == "project" and check["status"] == "error" for check in missing["checks"])


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
    _write_exact_execution_config(project)
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
                "--run-id",
                "plan-read-only",
            ]
        )
        == 0
    )
    planned = json.loads(capsys.readouterr().out)

    assert planned["run_id"] == "plan-read-only"
    assert planned["provider"] == "local"
    assert planned["runtime_profile"] == "framework/supervised@1"
    assert planned["setting_sources"]["provider"] == "job"
    assert planned["images"]["actual_job"] is None
    assert planned["pack"]["kind_profile"] == "supervised"
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
    _write_exact_execution_config(project)

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
                "--run-id",
                "grpo-static-plan",
            ]
        )
        == 0
    )
    planned = json.loads(capsys.readouterr().out)

    assert planned["run_id"] == "grpo-static-plan"
    assert planned["runtime_profile"] == "framework/online-rl-trl-py312@1"
    assert planned["images"]["actual_job"] is None
    assert planned["pack"]["kind_profile"] == "online-rl"
    assert planned["pack"]["runtime_variant"] == "online-rl-trl-py312"
    assert len(planned["pack"]["constraint_profile_digest"]) == 64
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
    assert "conflicts with the resolved backend variant online-rl-trl-py312" in (
        capsys.readouterr().err
    )

    _write_exact_execution_config(project, variants=("supervised",))
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
            ]
        )
        == 1
    )
    missing_variant = capsys.readouterr().err
    assert "runtime variant online-rl-trl-py312 is not published" in missing_variant
    assert "available: supervised" in missing_variant


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
    _write_exact_execution_config(project)

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
                "--env",
                "HF_TOKEN",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["work_package_id"] == "screen/cpu-check"
    assert payload["provider"] == "local"
    assert payload["environment_names"] == [
        "POSTTRAIN_TRACKIO_SERVER_URL",
        "TRACKIO_WRITE_TOKEN",
        "HF_TOKEN",
    ]
    assert payload["job_id"] == "validate"


def test_job_plan_target_override_changes_nested_sft_target_and_identity(
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

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "job",
                "plan",
                "sft-target.yaml",
                "--job",
                "train",
                "--run-id",
                "sft-default-target",
            ]
        )
        == 0
    )
    baseline = json.loads(capsys.readouterr().out)

    assert (
        main(
            [
                "--json",
                "--project-root",
                str(project),
                "job",
                "plan",
                "sft-target.yaml",
                "--job",
                "train",
                "--target",
                "targets/remote-rtx4090-24gb",
                "--run-id",
                "sft-remote-target",
            ]
        )
        == 0
    )
    overridden = json.loads(capsys.readouterr().out)

    assert baseline["target"]["id"] == "targets/local-cuda-8gb"
    assert overridden["target"] == {
        "id": "targets/remote-rtx4090-24gb",
        "revision": "1",
        "device_class": "nvidia-cuda",
        "memory_gb": 24.0,
    }
    assert overridden["pack"]["plan_key"] != baseline["pack"]["plan_key"]
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
        def __init__(self, **_kwargs) -> None:
            pass

        def publish(self, request):
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
        == 1
    )
    assert "dstack execution requires [providers.dstack.storage]" in capsys.readouterr().err

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
    reconciliation = (
        project
        / ".posttrain"
        / "state"
        / "executions"
        / "run-1"
        / "reconciliation.json"
    )
    assert reconciliation.is_file()
    assert reconciliation.stat().st_mode & 0o777 == 0o600
    recovery = (
        project
        / ".posttrain"
        / "state"
        / "executions"
        / "run-1"
        / "tracking-recovery.json"
    )
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
