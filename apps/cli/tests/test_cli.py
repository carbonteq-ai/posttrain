"""Tests for the primary posttrain command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

from posttrain.common import ExecutionTarget, RunContext
from posttrain.jobs import build_job_runtime
from posttrain.work import (
    JobDefinition,
    ResolvedSeats,
    WorkPackageContext,
    WorkPackageHostRequest,
)
from posttrain_cli.cli import main


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
    work_package = (project / ".posttrain" / "work_packages" / "sft.yaml").read_text(
        encoding="utf-8"
    )
    assert '"posttrain[observatory,trackio,trl]' in pyproject
    assert "carbonteq-ai/trackio.git@c5072198" in pyproject
    assert "carbonteq-ai/trl.git@5c50c69f" in pyproject
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
    assert "Job-definition preflight: complete" in validated.out


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
    work_package = (project / ".posttrain" / "work_packages" / "grpo.yaml").read_text(
        encoding="utf-8"
    )
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
    assert captured.out.index("Initialized post-training project") < captured.out.index(
        "Installing dependencies..."
    )
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
    assert payload["job_definition_preflight"] == "complete"
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
    assert validated["job_definition_preflight"] == "complete"

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
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["host"] is None
    assert result["entry"] == f"{__name__}:create_test_entry"
    assert result["jobs"][0]["value"]["target_id"] == "targets/local-cuda-8gb"


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
                "  revision: \"1\"",
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
                "math-gsm8k",
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
    assert (project / ".posttrain" / "catalog" / "environments.yaml").is_file()
