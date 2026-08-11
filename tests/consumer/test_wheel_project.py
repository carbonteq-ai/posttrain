"""Build wheels and exercise them from a repository outside this workspace."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest

WORKSPACE = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).resolve().parent / "fixture"
FRAMEWORK_PACKAGES = (
    "posttrain",
    "posttrain-common",
    "posttrain-data",
    "posttrain-eval",
    "posttrain-jobs",
    "posttrain-serve",
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


def _trackio_requirement() -> str:
    """Return the Trackio pin declared by the package that depends on it.

    Restating the commit here is the same drift this suite exists to catch: the
    hardcoded value silently fell behind the workspace and broke the install it
    was meant to prove.
    """
    pyproject = Path(__file__).resolve().parents[2] / "packages" / "tracking-trackio" / "pyproject.toml"
    for requirement in tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["dependencies"]:
        if requirement.startswith("carbonteq-trackio"):
            return requirement
    raise RuntimeError("posttrain-tracking-trackio no longer declares carbonteq-trackio")


def _build_wheelhouse(uv: str, root: Path) -> Path:
    """Build every workspace wheel, not a list of the ones we remembered.

    This previously enumerated packages by hand and fell six behind: the five
    execution distributions and posttrain-runtime-images. A consumer install
    then failed to resolve, which is exactly the failure this suite exists to
    detect, hidden by the suite's own stale list.
    """
    wheelhouse = root / "wheelhouse"
    wheelhouse.mkdir()
    _run(uv, "build", "--all-packages", "--wheel", "--out-dir", str(wheelhouse), cwd=WORKSPACE)
    for configured in os.environ.get("POSTTRAIN_CONSUMER_EXTRA_WHEELS", "").split(os.pathsep):
        if not configured:
            continue
        wheel = Path(configured)
        if not wheel.is_file() or wheel.suffix != ".whl":
            raise RuntimeError(f"consumer extra wheel is unavailable: {wheel}")
        shutil.copy2(wheel, wheelhouse / wheel.name)
    built = {path.name.split("-")[0].replace("_", "-") for path in wheelhouse.glob("*.whl")}
    missing = sorted(set(FRAMEWORK_PACKAGES) - built)
    if missing:
        raise RuntimeError(f"workspace build did not produce: {', '.join(missing)}")
    return wheelhouse


@pytest.fixture
def disk_tmp_path() -> Iterator[Path]:
    # Template extras include CUDA wheels; the desktop pytest tmpfs is too small.
    with tempfile.TemporaryDirectory(prefix="posttrain-consumer-", dir="/tmp") as directory:
        yield Path(directory)


def test_installed_wheels_discover_external_project_and_compose_catalog(
    tmp_path: Path,
) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("the external-consumer acceptance test requires uv")

    wheelhouse = _build_wheelhouse(uv, tmp_path)

    environment = tmp_path / "environment"
    _run(uv, "venv", str(environment), "--python", "3.13", cwd=tmp_path)
    python = environment / "bin" / "python"
    _run(
        uv,
        "pip",
        "install",
        "--python",
        str(python),
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
        _trackio_requirement(),
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
        str(command),
        "--json",
        "--project-root",
        str(project),
        "work-package",
        "run",
        "cpu_check.yaml",
        "--job",
        "validate",
        # This suite proves an installed wheel can discover a project and
        # execute a job. Packing is a separate concern needing a registry and
        # published images, which this environment deliberately has neither of.
        "--in-process",
        cwd=project,
        env=clean_environment,
    )
    execution = json.loads(completed.stdout)
    job = execution["jobs"][0]
    run_id = job["run_id"]
    result = job["value"]

    readback = _run(
        str(python),
        str(project / "run.py"),
        run_id,
        cwd=project,
        env=clean_environment,
    )
    evidence = json.loads(readback.stdout.splitlines()[-1])

    assert execution["project_id"] == "external-consumer"
    assert execution["work_package_id"] == "screen/cpu-check"
    assert execution["status"] == "succeeded"
    assert job["status"] == "succeeded"
    assert run_id
    assert result["train_examples"] > 0
    assert result["validation_examples"] > 0
    assert result["reserve_examples"] > 0
    assert len(result["partition_digest"]) == 64
    assert Path(evidence["project_root"]) == project.resolve()
    assert evidence["project_id"] == "external-consumer"
    assert evidence["base_catalog_release"] == "framework-v1"
    assert evidence["base_source"] == "base"
    assert evidence["base_target"] == "targets/local-cuda-8gb"
    assert evidence["overlay_source"] == "overlay"
    assert evidence["overlay_id"] == "external-consumer-v1"
    assert evidence["overlay_target"] == "targets/external-cpu"
    assert evidence["tracking_status"] == "succeeded"
    assert "data/train_examples" in evidence["tracking_metrics"]
    # A run that carries job identity is classified job-aware, not generic.
    # This assertion predates the job-aware Observatory views on this branch.
    assert evidence["observatory_mode"] == "job"
    assert evidence["observatory_run_id"] == run_id

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
    assert package["validation_level"] == "project"
    assert package["composition_validation"] == "complete"


def test_wheel_starters_cover_sft_and_environment_backed_paths(
    disk_tmp_path: Path,
) -> None:
    tmp_path = disk_tmp_path
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("the external-consumer acceptance test requires uv")

    wheelhouse = _build_wheelhouse(uv, tmp_path)
    bootstrap = tmp_path / "bootstrap"
    _run(uv, "venv", str(bootstrap), "--python", "3.13", cwd=tmp_path)
    python = bootstrap / "bin" / "python"
    _run(
        uv,
        "pip",
        "install",
        "--python",
        str(python),
        "--find-links",
        str(wheelhouse),
        _trackio_requirement(),
        "posttrain[observatory]",
        cwd=tmp_path,
    )
    command = bootstrap / "bin" / "posttrain"
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
    clean_environment["UV_FIND_LINKS"] = str(wheelhouse)

    sft = tmp_path / "sft-starter"
    initialized_sft = _run(
        str(command),
        "--json",
        "init",
        str(sft),
        "--template",
        "sft",
        "--project-id",
        "wheel-sft",
        cwd=tmp_path,
        env=clean_environment,
    )
    assert json.loads(initialized_sft.stdout)["project_id"] == "wheel-sft"
    sft_command = sft / ".venv" / "bin" / "posttrain"
    _run(str(sft_command), "doctor", cwd=sft, env=clean_environment)
    materialized = _run(
        str(sft_command),
        "--json",
        "dataset",
        "validate",
        "datasets/posttrain-sft-smoke@1",
        cwd=sft,
        env=clean_environment,
    )
    materialized_payload = json.loads(materialized.stdout)
    assert materialized_payload["examples"] == 2
    assert Path(materialized_payload["path"]).is_file()
    sft_preflight = _run(
        str(sft_command),
        "--json",
        "work-package",
        "validate",
        "sft.yaml",
        cwd=sft,
        env=clean_environment,
    )
    assert json.loads(sft_preflight.stdout)["composition_validation"] == "complete"
    sft_import = _run(
        str(sft / ".venv" / "bin" / "python"),
        "-c",
        "import importlib.util, posttrain.jobs; "
        "assert importlib.util.find_spec('posttrain_lab') is None; "
        "print(posttrain.jobs.__file__)",
        cwd=sft,
        env=clean_environment,
    )
    assert "posttrain/jobs" in sft_import.stdout

    grpo = tmp_path / "grpo-starter"
    initialized_grpo = _run(
        str(command),
        "--json",
        "init",
        str(grpo),
        "--template",
        "grpo",
        "--project-id",
        "wheel-grpo",
        cwd=tmp_path,
        env=clean_environment,
    )
    assert json.loads(initialized_grpo.stdout)["project_id"] == "wheel-grpo"
    grpo_command = grpo / ".venv" / "bin" / "posttrain"
    grpo_preflight = _run(
        str(grpo_command),
        "--json",
        "work-package",
        "validate",
        "grpo.yaml",
        cwd=grpo,
        env=clean_environment,
    )
    assert json.loads(grpo_preflight.stdout)["composition_validation"] == "complete"
    bridge = _run(
        str(grpo / ".venv" / "bin" / "python"),
        "-c",
        (
            "from pathlib import Path; "
            "from posttrain.catalog import discover_project, open_catalog; "
            "from posttrain.common import CatalogRef; "
            "from posttrain.train import PolicySampling; "
            "from posttrain.train.integrations import create_verifiers_training_bridge; "
            "layout=discover_project(Path.cwd()); "
            "catalog=open_catalog(scope=layout.project_id, overlays=layout.catalog_overlays); "
            "environment=catalog.resolve(CatalogRef('environment','starter-gsm8k-train')).value; "
            "bridge=create_verifiers_training_bridge("
            "environment, layout.state/'consumer-traces.jsonl', 'wheel-consumer', "
            "sampling=PolicySampling(max_tokens=384)); "
            "print(bridge.dataset.id, len(bridge.dataset.examples))"
        ),
        cwd=grpo,
        env=clean_environment,
    )
    assert bridge.stdout.strip().endswith("starter-gsm8k-train/grpo/seed-0-limit-1 1")

    starter_text = "\n".join(
        path.read_text(encoding="utf-8")
        for root in (sft, grpo)
        for path in root.rglob("*")
        if path.is_file() and ".venv" not in path.parts and path.suffix in {".py", ".toml", ".yaml", ".md"}
    )
    assert "posttrain_lab" not in starter_text
