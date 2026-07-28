"""Project initialization and starter template scaffolding."""

from __future__ import annotations

import json
import re
import socket
from importlib.metadata import PackageNotFoundError, requires, version
from pathlib import Path

from posttrain.catalog import ProjectLayout, discover_project
from posttrain.common import ContractError
from posttrain.common.selections import validate_selection_id

from ..constants import DISTRIBUTION


def workspace_root() -> Path | None:
    candidate = Path(__file__).resolve()
    for parent in candidate.parents:
        if (parent / "uv.lock").is_file() and (parent / "apps" / "cli" / "pyproject.toml").is_file():
            return parent
    return None


def installed_version() -> str:
    try:
        return version(DISTRIBUTION)
    except PackageNotFoundError:
        return "0+unknown"


def default_project_id(root: Path) -> str:
    normalized = re.sub(r"[^a-z0-9._/@:-]+", "-", root.name.lower()).strip("-._")
    if not normalized:
        raise ContractError("cannot derive project id from path; pass --project-id")
    return validate_selection_id(normalized, "project id")


def project_package_name(project_id: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", project_id.lower()).strip("_")
    if not normalized:
        raise ContractError("cannot derive Python package name from project id")
    if normalized[0].isdigit():
        normalized = f"project_{normalized}"
    return normalized


def starter_direct_references(template: str) -> tuple[str, ...]:
    selections: tuple[tuple[str, str | None], ...] = (
        ("posttrain-tracking-trackio", None),
        ("posttrain-train", "trl"),
        *((("posttrain-eval", "verifiers"),) if template == "grpo" else ()),
    )
    references: set[str] = set()
    for distribution, selected_extra in selections:
        try:
            declared = requires(distribution) or ()
        except PackageNotFoundError as error:
            raise RuntimeError(
                f"cannot generate starter dependencies because {distribution} is not installed"
            ) from error
        for raw_requirement in declared:
            requirement, separator, marker = raw_requirement.partition(";")
            requirement = requirement.strip()
            if " @ " not in requirement:
                continue
            if selected_extra is None:
                if separator and "extra" in marker:
                    continue
            elif not (f'extra == "{selected_extra}"' in marker or f"extra == '{selected_extra}'" in marker):
                continue
            references.add(requirement)
    return tuple(sorted(references))


def starter_pyproject(project_id: str, template: str) -> str:
    distribution_name = re.sub(r"[^a-z0-9]+", "-", project_id.lower()).strip("-")
    package_name = project_package_name(project_id)
    extras = "observatory,trackio,trl" if template == "sft" else "observatory,trackio,trl,verifiers"
    installed = installed_version()
    version_constraint = "" if installed == "0+unknown" else f"=={installed}"
    dependencies = (
        f"posttrain[{extras}]{version_constraint}",
        *starter_direct_references(template),
    )
    lines = [
        "[project]",
        f'name = "{distribution_name}"',
        'version = "0.1.0"',
        f'description = "Posttrain {template.upper()} starter project"',
        'requires-python = ">=3.13,<3.14"',
        "dependencies = [",
        *(f"  {json.dumps(dependency)}," for dependency in dependencies),
        "]",
        "",
        "[build-system]",
        'requires = ["hatchling"]',
        'build-backend = "hatchling.build"',
        "",
        "[tool.hatch.build.targets.wheel]",
        f'packages = ["src/{package_name}"]',
        "",
        "[tool.hatch.metadata]",
        "allow-direct-references = true",
        "",
        "[tool.posttrain.pack]",
        'project_packages = ["."]',
        'source_includes = ["pyproject.toml", "src"]',
    ]

    workspace = workspace_root()
    if workspace is not None:
        local_sources = {
            "posttrain": "apps/cli",
            "posttrain-observatory": "apps/observatory",
            "posttrain-catalog": "packages/catalog",
            "posttrain-common": "packages/common",
            "posttrain-data": "packages/data",
            "posttrain-eval": "packages/eval",
            "posttrain-jobs": "packages/jobs",
            "posttrain-serve": "packages/serve",
            "posttrain-tracking": "packages/tracking",
            "posttrain-tracking-trackio": "packages/tracking-trackio",
            "posttrain-tracking-wandb": "packages/tracking-wandb",
            "posttrain-train": "packages/train",
            "posttrain-work": "packages/work",
        }
        lines.extend(("", "[tool.uv.sources]"))
        lines.extend(
            f'{name} = {{ path = "{workspace / relative}", editable = true }}'
            for name, relative in local_sources.items()
        )
    return "\n".join((*lines, ""))


def starter_settings(template: str) -> str:
    if template == "sft":
        return "\n".join(
            (
                "training:",
                "  starter/sft@1:",
                "    selection_type: sft-settings",
                '    revision: "1"',
                "    loop:",
                "      max_steps: 1",
                "      max_length: 512",
                "",
            )
        )
    return "\n".join(
        (
            "training:",
            "  starter/grpo@1:",
            "    selection_type: grpo-settings",
            '    revision: "1"',
            "    loop:",
            "      max_steps: 1",
            "      max_length: 640",
            "      per_device_batch_size: 2",
            "      learning_rate: 0.00001",
            "    num_prompts_per_step: 1",
            "    num_generations: 2",
            "    max_prompt_length: 256",
            "    max_completion_length: 384",
            "",
        )
    )


def starter_work_package(project_id: str, template: str) -> str:
    if template == "sft":
        return "\n".join(
            (
                f"project_id: {project_id}",
                "work_package_id: train/starter-sft",
                "stage: train",
                "description: Run one bounded supervised fine-tuning update from a declarative dataset.",
                "recipe:",
                "  type: inline",
                "  id: recipes/starter-sft@1",
                '  revision: "1"',
                "  stage: train",
                "  seats:",
                "    model: model",
                "    dataset: dataset",
                "    settings: training",
                "    training: training",
                "  jobs:",
                "    - id: train",
                "      kind: train.sft",
                "      definition: train/trl-sft@1",
                "  expected_artifacts:",
                "    - trained adapter",
                "    - training summary",
                "bindings:",
                "  model:",
                "    type: ref",
                "    family: model",
                "    id: models/qwen3.5-2b@bf16",
                "  dataset:",
                "    type: ref",
                "    family: dataset",
                "    id: datasets/posttrain-sft-smoke@1",
                "  settings:",
                "    type: ref",
                "    family: training",
                "    id: starter/sft@1",
                "  training:",
                "    type: ref",
                "    family: training",
                "    id: training/qwen3.5-trl-lora@1",
                "enabled_optional_jobs: []",
                "metadata:",
                "  labels: [starter, sft]",
                "",
            )
        )
    return "\n".join(
        (
            f"project_id: {project_id}",
            "work_package_id: train/starter-grpo",
            "stage: train",
            "description: Run one bounded GRPO update against the global GSM8K environment binding.",
            "recipe:",
            "  type: inline",
            "  id: recipes/starter-grpo@1",
            '  revision: "1"',
            "  stage: train",
            "  seats:",
            "    model: model",
            "    environment: environment",
            "    settings: training",
            "    training: training",
            "    rollout_inference: inference",
            "  jobs:",
            "    - id: train",
            "      kind: train.grpo",
            "      definition: train/trl-grpo@1",
            "  expected_artifacts:",
            "    - trained adapter",
            "    - native Verifiers trajectories",
            "    - training summary",
            "bindings:",
            "  model:",
            "    type: ref",
            "    family: model",
            "    id: models/qwen3.5-0.8b@bf16",
            "  environment:",
            "    type: ref",
            "    family: environment",
            "    id: gsm8k-distill-train",
            "  settings:",
            "    type: ref",
            "    family: training",
            "    id: starter/grpo@1",
            "  training:",
            "    type: ref",
            "    family: training",
            "    id: training/qwen3.5-0.8b-trl-distill-lora@1",
            "  rollout_inference:",
            "    type: ref",
            "    family: inference",
            "    id: inference/qwen3.5-0.8b-vllm-distill-rollout@1",
            "enabled_optional_jobs: []",
            "metadata:",
            "  labels: [starter, grpo, verifiers]",
            "",
        )
    )


def initialize(
    root: Path,
    project_id: str | None,
    *,
    template: str | None = None,
) -> ProjectLayout:
    project_root = root.resolve()
    resolved_id = validate_selection_id(
        project_id if project_id is not None else default_project_id(project_root),
        "project id",
    )
    control = project_root / ".posttrain"
    manifest = control / "project.toml"
    project_brief = control / "project.yaml"
    catalog = control / "catalog"
    catalog_manifest = catalog / "layer.yaml"
    work_packages = control / "work_packages"
    work_packages_readme = work_packages / "README.md"
    control_ignore = control / ".gitignore"

    if project_root.exists() and any(project_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing project files in non-empty directory: {project_root}")

    project_root.mkdir(parents=True, exist_ok=True)
    catalog.mkdir(parents=True, exist_ok=True)
    work_packages.mkdir(parents=True, exist_ok=True)
    (control / "state").mkdir(parents=True, exist_ok=True)
    hostname = socket.gethostname().strip().lower().rstrip(".")
    execution_toml = control / "state" / "execution.toml"
    execution_toml.write_text(
        "\n".join(
            (
                "schema_version = 1",
                "",
                "[providers.local]",
                f'canonical_hostname = "{hostname}"',
                "",
            )
        ),
        encoding="utf-8",
    )
    execution_toml.chmod(0o600)
    manifest.write_text(
        "\n".join(
            (
                "schema_version = 2",
                f'project_id = "{resolved_id}"',
                'catalog_overlays = ["catalog"]',
                'work_packages = "work_packages"',
                'state = "state"',
                'tracking = "trackio"',
                'project_brief = "project.yaml"',
                "",
            )
        ),
        encoding="utf-8",
    )
    project_brief.write_text(
        "\n".join(
            (
                "schema_version: 1",
                f"objective: Improve and qualify models for the {resolved_id} project.",
                "serving: null",
                "",
            )
        ),
        encoding="utf-8",
    )
    catalog_files = ["settings.yaml"] if template is not None else []
    catalog_manifest.write_text(
        "\n".join(
            (
                "schema_version: 1",
                f"layer_id: {resolved_id}-v1",
                *(("files:", *(f"  - {name}" for name in catalog_files)) if catalog_files else ("files: []",)),
                "",
            )
        ),
        encoding="utf-8",
    )
    work_packages_readme.write_text(
        "# Work packages\n\nAdd versioned `screen`, `train`, and `qualify` work-package YAML files here.\n",
        encoding="utf-8",
    )
    control_ignore.write_text("state/\n", encoding="utf-8")
    if template is not None:
        package_name = project_package_name(resolved_id)
        package = project_root / "src" / package_name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(
            f'"""Project package for {resolved_id}."""\n',
            encoding="utf-8",
        )
        (project_root / "pyproject.toml").write_text(
            starter_pyproject(resolved_id, template),
            encoding="utf-8",
        )
        (project_root / ".gitignore").write_text(
            ".venv/\n__pycache__/\n*.py[cod]\n",
            encoding="utf-8",
        )
        (catalog / "settings.yaml").write_text(starter_settings(template), encoding="utf-8")
        work_package_name = f"{template}.yaml"
        (work_packages / work_package_name).write_text(
            starter_work_package(resolved_id, template),
            encoding="utf-8",
        )
        work_packages_readme.write_text(
            (
                "# Work packages\n\n"
                f"Validate with `posttrain work-package validate {work_package_name}` and run the "
                f"`train` job with `posttrain job run {work_package_name}` "
                "(omit `--job` when the package has exactly one).\n\n"
                "Validation and catalog materialization are CPU-safe. The generated training job is a "
                "CUDA release gate; run it only on a compatible target."
                + (
                    " Install `posttrain[vllm]` in this project before the GRPO GPU gate.\n"
                    if template == "grpo"
                    else "\n"
                )
            ),
            encoding="utf-8",
        )
    return discover_project(project_root, explicit_root=project_root)


def install_starter(project_root: Path, *, json_output: bool = False) -> None:
    from posttrain_cli import cli as cli_module

    uv = cli_module.shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to install a starter project; install uv and retry")
    stream = cli_module.sys.stderr if json_output else cli_module.sys.stdout
    print("Installing dependencies...", file=stream)
    cli_module.subprocess.run(
        [uv, "sync", "--python", "3.13"],
        cwd=project_root,
        check=True,
        stdout=stream if json_output else None,
        stderr=stream if json_output else None,
    )
    print(f"Environment ready: {project_root / '.venv'}", file=stream)
