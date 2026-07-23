"""Tests for the primary posttrain command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

from posttrain_cli.cli import main


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
                "validate",
                "cpu-check.yaml",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["work_package_id"] == "screen/cpu-check"
    assert payload["resolved_seats"] == ["target"]
    assert payload["job_definition_preflight"] == "pending-host-definitions"


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
