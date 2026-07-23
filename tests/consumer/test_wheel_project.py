"""Build wheels and exercise them from a repository outside this workspace."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).resolve().parent / "fixture"
FRAMEWORK_PACKAGES = (
    "posttrain",
    "posttrain-common",
    "posttrain-data",
    "posttrain-eval",
    "posttrain-tracking",
    "posttrain-tracking-trackio",
    "posttrain-tracking-wandb",
    "posttrain-train",
    "posttrain-work",
    "posttrain-catalog",
    "posttrain-observatory",
)


def _run(*command: str, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )


def test_installed_wheels_discover_external_project_and_compose_catalog(
    tmp_path: Path,
) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("the external-consumer acceptance test requires uv")

    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    for package in FRAMEWORK_PACKAGES:
        _run(
            uv,
            "build",
            "--package",
            package,
            "--wheel",
            "--out-dir",
            str(wheelhouse),
            cwd=WORKSPACE,
        )

    environment = tmp_path / "environment"
    _run(uv, "venv", str(environment), "--python", "3.12", cwd=tmp_path)
    python = environment / "bin" / "python"
    _run(
        uv,
        "pip",
        "install",
        "--python",
        str(python),
        "--offline",
        "pydantic>=2.12,<3",
        "pyyaml>=6.0,<7",
        cwd=tmp_path,
    )
    _run(
        uv,
        "pip",
        "install",
        "--python",
        str(python),
        "--find-links",
        str(wheelhouse),
        "carbonteq-trackio @ git+https://github.com/carbonteq-ai/trackio.git@c5072198b3b1556d31ed96ffc246a03f65418ab8",
        "posttrain-catalog",
        "posttrain-data",
        "posttrain-observatory",
        "posttrain-tracking-trackio",
        "posttrain",
        cwd=tmp_path,
    )

    command = environment / "bin" / "posttrain"
    initialized_project = tmp_path / "initialized-project"
    initialized = _run(
        str(command),
        "--json",
        "init",
        str(initialized_project),
        "--project-id",
        "initialized-consumer",
        cwd=tmp_path,
    )
    initialized_layout = json.loads(initialized.stdout)
    assert initialized_layout["project_id"] == "initialized-consumer"
    assert Path(initialized_layout["root"]) == initialized_project.resolve()
    readiness = _run(
        str(command),
        "--json",
        "--project-root",
        str(initialized_project),
        "doctor",
        cwd=tmp_path,
    )
    assert json.loads(readiness.stdout)["ok"] is True

    project = tmp_path / "external-project"
    shutil.copytree(FIXTURE, project)
    nested = project / "src" / "jobs"
    nested.mkdir(parents=True)
    clean_environment = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "PYTHONPATH",
            "POSTTRAIN_PROJECT_ROOT",
            "UV_PROJECT_ENVIRONMENT",
            "VIRTUAL_ENV",
        }
    }
    clean_environment["TRACKIO_DIR"] = str(project / ".posttrain" / "state" / "trackio")
    completed = _run(
        str(python),
        str(project / "run.py"),
        cwd=nested,
        env=clean_environment,
    )
    result = json.loads(completed.stdout.splitlines()[-1])

    assert Path(result["project_root"]) == project.resolve()
    assert result["project_id"] == "external-consumer"
    assert result["base_catalog_release"] == "framework-v1"
    assert result["base_source"] == "base"
    assert result["base_target"] == "targets/local-cuda-8gb"
    assert result["overlay_source"] == "overlay"
    assert result["overlay_id"] == "external-consumer-v1"
    assert result["overlay_target"] == "targets/external-cpu"
    assert result["train_examples"] > 0
    assert result["validation_examples"] > 0
    assert result["reserve_examples"] > 0
    assert len(result["partition_digest"]) == 64
    assert result["work_package_id"] == "screen/cpu-check"
    assert result["job_status"] == "succeeded"
    assert result["tracking_status"] == "succeeded"
    assert "data/train_examples" in result["tracking_metrics"]
    assert result["observatory_mode"] == "generic"
    assert result["observatory_run_id"] == result["run_id"]

    validated = _run(
        str(command),
        "--json",
        "--project-root",
        str(project),
        "catalog",
        "validate",
        cwd=nested,
        env=clean_environment,
    )
    catalog = json.loads(validated.stdout)
    assert catalog["base_catalog_release"] == "framework-v1"
    assert catalog["project_entries"] == 1

    work_package = _run(
        str(command),
        "--json",
        "--project-root",
        str(project),
        "work-package",
        "validate",
        "cpu_check.yaml",
        cwd=nested,
        env=clean_environment,
    )
    package = json.loads(work_package.stdout)
    assert package["project_id"] == "external-consumer"
    assert package["work_package_id"] == "screen/cpu-check"
    assert package["resolved_seats"] == ["target"]
    assert package["validation_level"] == "composition"
