"""Stable command-line surface for portable post-training projects."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import nullcontext, redirect_stdout
from dataclasses import fields, is_dataclass
from importlib.metadata import PackageNotFoundError, requires, version
from pathlib import Path
from typing import Any, cast

from posttrain.catalog import ProjectLayout, discover_project, open_catalog
from posttrain.common import CatalogRef, ContractError
from posttrain.common.selections import SelectionFamily, validate_selection_id
from posttrain.data import DatasetLoadPlan, materialize_dataset
from posttrain.eval import EnvironmentBinding
from posttrain.jobs import build_job_runtime, standard_definitions
from posttrain.train.integrations import preflight_verifiers_environment
from posttrain.work import (
    JobRuntime,
    ProjectEntry,
    ProjectExecutionRequest,
    WorkPackage,
    WorkPackageContext,
    WorkPackageHostFactory,
    WorkPackageHostRequest,
    load_work_package,
    resolve_work_package,
    run_work_package_job,
    validate_work_package,
)

from .overlay_write import (
    ensure_overlay_file,
    overlay_directory,
    selection_revision,
    upsert_family_entry,
)

_DISTRIBUTION = "posttrain"
_CATALOG_FAMILIES: tuple[SelectionFamily, ...] = (
    "model",
    "dataset",
    "environment",
    "inference",
    "training",
    "quantization",
    "evaluation",
    "workload",
    "target",
    "recipe",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="posttrain",
        description="Initialize, inspect, validate, and run post-training projects.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="project root containing .posttrain/project.toml; otherwise discover upward",
    )
    parser.add_argument("--json", dest="json_output", action="store_true", help="emit JSON output")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("version", help="show the installed framework CLI version")

    init = commands.add_parser("init", help="initialize a portable post-training project")
    init.add_argument("path", nargs="?", type=Path, default=Path.cwd())
    init.add_argument("--project-id", help="lowercase stable project identity; defaults from PATH")
    init.add_argument(
        "--template",
        choices=("sft", "grpo"),
        help="create and install a runnable starter project",
    )
    init.add_argument(
        "--no-install",
        action="store_true",
        help="write the selected template without creating its environment (CI/bootstrap escape hatch)",
    )

    doctor = commands.add_parser("doctor", help="check project and catalog readiness")
    doctor.add_argument(
        "--fix",
        action="store_true",
        help="also materialize datasets and preflight environments referenced by work packages",
    )
    doctor.add_argument(
        "--work-package",
        type=Path,
        help="limit --fix to one work package path",
    )
    project = commands.add_parser("project", help="inspect project configuration")
    project_commands = project.add_subparsers(dest="project_command", required=True)
    project_commands.add_parser("show", help="show the discovered project layout")

    catalog = commands.add_parser("catalog", help="inspect the composed project catalog")
    catalog_commands = catalog.add_subparsers(dest="catalog_command", required=True)
    catalog_list = catalog_commands.add_parser("list", help="list resolved catalog selections")
    catalog_list.add_argument("--family", choices=_CATALOG_FAMILIES)
    catalog_show = catalog_commands.add_parser("show", help="show one resolved catalog selection")
    catalog_show.add_argument("family", choices=_CATALOG_FAMILIES)
    catalog_show.add_argument("id")
    catalog_commands.add_parser("validate", help="load and validate the complete catalog")
    catalog_materialize = catalog_commands.add_parser(
        "materialize",
        help="materialize datasets and preflight environments referenced by work packages",
    )
    catalog_materialize.add_argument(
        "--work-package",
        type=Path,
        help="limit materialization to one work package path",
    )

    dataset = commands.add_parser("dataset", help="resolve and validate project datasets")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_validate = dataset_commands.add_parser(
        "validate",
        help="materialize and adapter-validate one dataset selection",
    )
    dataset_validate.add_argument("id")
    dataset_add = dataset_commands.add_parser("add", help="write a dataset entry into the project overlay")
    dataset_add_kinds = dataset_add.add_subparsers(dest="dataset_add_kind", required=True)
    dataset_add_hf = dataset_add_kinds.add_parser("hf", help="register a Hugging Face dataset")
    dataset_add_hf.add_argument("--id", required=True, help="catalog selection id, for example datasets/support-sft@1")
    dataset_add_hf.add_argument("--repo", required=True)
    dataset_add_hf.add_argument("--revision", required=True, help="immutable Hub revision")
    dataset_add_hf.add_argument("--split", default="train")
    dataset_add_hf.add_argument("--config")
    dataset_add_hf.add_argument(
        "--format",
        default="messages",
        choices=("auto", "messages", "prompt-completion", "alpaca", "sharegpt", "trl", "tulu", "nemo-ranked"),
    )
    dataset_add_hf.add_argument("--kind", choices=("supervised", "preference"), default="supervised")
    dataset_add_hf.add_argument("--file", default="datasets.yaml", help="overlay YAML filename")
    dataset_add_jsonl = dataset_add_kinds.add_parser("jsonl", help="register a project-relative JSONL dataset")
    dataset_add_jsonl.add_argument("--id", required=True)
    dataset_add_jsonl.add_argument("--path", required=True, help="path relative to the project root")
    dataset_add_jsonl.add_argument(
        "--format",
        default="messages",
        choices=("auto", "messages", "prompt-completion", "alpaca", "sharegpt", "trl", "tulu", "nemo-ranked"),
    )
    dataset_add_jsonl.add_argument("--kind", choices=("supervised", "preference"), default="supervised")
    dataset_add_jsonl.add_argument("--file", default="datasets.yaml")
    dataset_add_nemo = dataset_add_kinds.add_parser("nemo", help="register a project-relative NeMo JSONL dataset")
    dataset_add_nemo.add_argument("--id", required=True)
    dataset_add_nemo.add_argument("--path", required=True, help="path relative to the project root")
    dataset_add_nemo.add_argument(
        "--format",
        default="auto",
        choices=("auto", "messages", "nemo-ranked"),
    )
    dataset_add_nemo.add_argument("--kind", choices=("supervised", "preference"), default="supervised")
    dataset_add_nemo.add_argument("--file", default="datasets.yaml")

    environment = commands.add_parser("environment", help="register Verifiers environment bindings")
    environment_commands = environment.add_subparsers(dest="environment_command", required=True)
    environment_add = environment_commands.add_parser(
        "add",
        help="write an environment binding into the project overlay",
    )
    environment_add_kinds = environment_add.add_subparsers(dest="environment_add_kind", required=True)
    environment_add_local = environment_add_kinds.add_parser(
        "local",
        help="register an installed Verifiers package binding",
    )
    environment_add_local.add_argument("--id", required=True)
    environment_add_local.add_argument("--package", required=True)
    environment_add_local.add_argument("--factory", required=True)
    environment_add_local.add_argument("--repository", required=True)
    environment_add_local.add_argument("--revision", required=True, help="immutable commit SHA")
    environment_add_local.add_argument("--subdirectory")
    environment_add_local.add_argument("--category", default="custom")
    environment_add_local.add_argument("--num-tasks", type=int, default=8)
    environment_add_local.add_argument("--num-rollouts", type=int, default=1)
    environment_add_local.add_argument("--max-tokens", type=int, default=2048)
    environment_add_local.add_argument("--temperature", type=float, default=1.0)
    environment_add_local.add_argument("--file", default="environments.yaml")

    observatory = commands.add_parser("observatory", help="start the project evidence product")
    observatory_commands = observatory.add_subparsers(dest="observatory_command", required=True)
    observatory_up = observatory_commands.add_parser(
        "up",
        help="serve Observatory for the discovered project's tracking backend",
    )
    observatory_up.add_argument("--host", help="listening host; defaults to Observatory settings")
    observatory_up.add_argument("--port", type=int, help="listening port; defaults to 7861")

    work_package = commands.add_parser("work-package", help="inspect and execute work packages")
    work_package_commands = work_package.add_subparsers(
        dest="work_package_command",
        required=True,
    )
    work_package_validate = work_package_commands.add_parser(
        "validate",
        help="validate YAML, recipe structure, and catalog bindings",
    )
    work_package_validate.add_argument("path", type=Path)
    work_package_validate.add_argument(
        "--host",
        metavar="MODULE:FACTORY",
        help="also preflight concrete job definitions through this explicit project host",
    )
    work_package_validate.add_argument(
        "--entry",
        metavar="MODULE:FACTORY",
        help="override the optional project entry for this invocation",
    )
    work_package_run = work_package_commands.add_parser(
        "run",
        help="execute a validated work package through an explicit project host",
    )
    work_package_run.add_argument("path", type=Path)
    work_package_run.add_argument(
        "--job",
        required=True,
        help="execute exactly this enabled recipe job id",
    )
    work_package_run.add_argument(
        "--host",
        metavar="MODULE:FACTORY",
        help="deprecated compatibility alias for an explicit legacy host",
    )
    work_package_run.add_argument(
        "--entry",
        metavar="MODULE:FACTORY",
        help="override the optional project entry for this invocation",
    )
    return parser


def _installed_version() -> str:
    try:
        return version(_DISTRIBUTION)
    except PackageNotFoundError:
        return "0+unknown"


def _default_project_id(root: Path) -> str:
    normalized = re.sub(r"[^a-z0-9._/@:-]+", "-", root.name.lower()).strip("-._")
    if not normalized:
        raise ContractError("cannot derive project id from path; pass --project-id")
    return validate_selection_id(normalized, "project id")


def _project_package_name(project_id: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", project_id.lower()).strip("_")
    if not normalized:
        raise ContractError("cannot derive Python package name from project id")
    if normalized[0].isdigit():
        normalized = f"project_{normalized}"
    return normalized


def _workspace_root() -> Path | None:
    candidate = Path(__file__).resolve()
    for parent in candidate.parents:
        if (parent / "uv.lock").is_file() and (parent / "apps" / "cli" / "pyproject.toml").is_file():
            return parent
    return None


def _starter_direct_references(template: str) -> tuple[str, ...]:
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
            elif not (
                f'extra == "{selected_extra}"' in marker
                or f"extra == '{selected_extra}'" in marker
            ):
                continue
            references.add(requirement)
    return tuple(sorted(references))


def _starter_pyproject(project_id: str, template: str) -> str:
    distribution_name = re.sub(r"[^a-z0-9]+", "-", project_id.lower()).strip("-")
    package_name = _project_package_name(project_id)
    extras = "observatory,trackio,trl" if template == "sft" else "observatory,trackio,trl,verifiers"
    installed = _installed_version()
    version_constraint = "" if installed == "0+unknown" else f"=={installed}"
    dependencies = (
        f"posttrain[{extras}]{version_constraint}",
        *_starter_direct_references(template),
    )
    lines = [
        "[project]",
        f'name = "{distribution_name}"',
        'version = "0.1.0"',
        f'description = "Posttrain {template.upper()} starter project"',
        'requires-python = ">=3.12,<3.13"',
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
    ]

    workspace = _workspace_root()
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


def _starter_settings(template: str) -> str:
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


def _starter_work_package(project_id: str, template: str) -> str:
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


def _install_starter(project_root: Path, *, json_output: bool = False) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to install a starter project; install uv and retry")
    stream = sys.stderr if json_output else sys.stdout
    print("Installing dependencies...", file=stream)
    subprocess.run(
        [uv, "sync", "--python", "3.12"],
        cwd=project_root,
        check=True,
        stdout=stream if json_output else None,
        stderr=stream if json_output else None,
    )
    print(f"Environment ready: {project_root / '.venv'}", file=stream)


def _initialize(
    root: Path,
    project_id: str | None,
    *,
    template: str | None = None,
) -> ProjectLayout:
    project_root = root.resolve()
    resolved_id = validate_selection_id(
        project_id if project_id is not None else _default_project_id(project_root),
        "project id",
    )
    control = project_root / ".posttrain"
    manifest = control / "project.toml"
    catalog = control / "catalog"
    catalog_manifest = catalog / "layer.yaml"
    work_packages = control / "work_packages"
    work_packages_readme = work_packages / "README.md"
    control_ignore = control / ".gitignore"

    if project_root.exists() and any(project_root.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite existing project files in non-empty directory: {project_root}"
        )

    project_root.mkdir(parents=True, exist_ok=True)
    catalog.mkdir(parents=True, exist_ok=True)
    work_packages.mkdir(parents=True, exist_ok=True)
    (control / "state").mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "\n".join(
            (
                "schema_version = 1",
                f'project_id = "{resolved_id}"',
                'catalog_overlays = ["catalog"]',
                'work_packages = "work_packages"',
                'state = "state"',
                'tracking = "trackio"',
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
        package_name = _project_package_name(resolved_id)
        package = project_root / "src" / package_name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(
            f'"""Project package for {resolved_id}."""\n',
            encoding="utf-8",
        )
        (project_root / "pyproject.toml").write_text(
            _starter_pyproject(resolved_id, template),
            encoding="utf-8",
        )
        (project_root / ".gitignore").write_text(
            ".venv/\n__pycache__/\n*.py[cod]\n",
            encoding="utf-8",
        )
        (catalog / "settings.yaml").write_text(_starter_settings(template), encoding="utf-8")
        work_package_name = f"{template}.yaml"
        (work_packages / work_package_name).write_text(
            _starter_work_package(resolved_id, template),
            encoding="utf-8",
        )
        work_packages_readme.write_text(
            (
                "# Work packages\n\n"
                f"Validate with `posttrain work-package validate {work_package_name}` and run the "
                f"`train` job with `posttrain work-package run {work_package_name} --job train`.\n\n"
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


def _layout(args: argparse.Namespace) -> ProjectLayout:
    return discover_project(Path.cwd(), explicit_root=args.project_root)


def _catalog(args: argparse.Namespace) -> tuple[ProjectLayout, Any]:
    layout = _layout(args)
    return (
        layout,
        open_catalog(
            scope=layout.project_id,
            overlays=layout.catalog_overlays,
            catalog_root=layout.base_catalog,
        ),
    )


def _work_package_path(layout: ProjectLayout, configured: Path) -> Path:
    candidate = configured if configured.is_absolute() else Path.cwd() / configured
    if not candidate.is_file() and not configured.is_absolute():
        candidate = layout.work_packages / configured
    resolved = candidate.resolve()
    if not resolved.is_relative_to(layout.work_packages):
        raise ContractError(f"work-package path must remain under {layout.work_packages}: {configured}")
    return resolved


def _load_work_package(
    args: argparse.Namespace,
) -> tuple[ProjectLayout, Any, Path, WorkPackage]:
    layout, catalog = _catalog(args)
    path = _work_package_path(layout, args.path)
    package = load_work_package(path)
    if package.project_id != layout.project_id:
        raise ContractError(
            f"work package project {package.project_id!r} does not match project manifest {layout.project_id!r}"
        )
    return layout, catalog, path, package


def _load_host_factory(spec: str, *, project_root: Path) -> WorkPackageHostFactory:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute or ":" in attribute:
        raise ContractError("host must use MODULE:FACTORY syntax")
    search_root = str(project_root)
    sys.path.insert(0, search_root)
    try:
        try:
            module = importlib.import_module(module_name)
        except (ImportError, ValueError) as error:
            raise ContractError(f"cannot import work-package host module {module_name!r}: {error}") from error
    finally:
        sys.path.remove(search_root)
    try:
        factory = getattr(module, attribute)
    except AttributeError as error:
        raise ContractError(f"work-package host module {module_name!r} has no factory {attribute!r}") from error
    if not callable(factory):
        raise ContractError(f"work-package host {spec!r} is not callable")
    return cast(WorkPackageHostFactory, factory)


def _load_project_entry(spec: str, *, project_root: Path) -> ProjectEntry:
    return cast(ProjectEntry, _load_host_factory(spec, project_root=project_root))


def _host_context(
    spec: str,
    *,
    layout: ProjectLayout,
    catalog: Any,
    path: Path,
) -> WorkPackageContext:
    factory = _load_host_factory(spec, project_root=layout.root)
    request = WorkPackageHostRequest(
        project_id=layout.project_id,
        project_root=layout.root,
        state_dir=layout.state,
        work_package_path=path,
        catalog=catalog,
    )
    context = factory(request)
    if not isinstance(context, WorkPackageContext):
        raise ContractError(f"work-package host {spec!r} must return WorkPackageContext")
    if context.catalog is not catalog:
        raise ContractError("work-package host must use the catalog supplied in WorkPackageHostRequest")
    return context


def _execution_request(
    *,
    layout: ProjectLayout,
    catalog: Any,
    path: Path,
) -> ProjectExecutionRequest:
    return ProjectExecutionRequest(
        project_id=layout.project_id,
        project_root=layout.root,
        state_dir=layout.state,
        work_package_path=path,
        catalog=catalog,
    )


def _runtime_context(
    args: argparse.Namespace,
    *,
    layout: ProjectLayout,
    catalog: Any,
    path: Path,
) -> JobRuntime:
    if args.host is not None:
        return _host_context(args.host, layout=layout, catalog=catalog, path=path)
    request = _execution_request(layout=layout, catalog=catalog, path=path)
    entry_spec = args.entry or layout.entry
    if entry_spec is None:
        return build_job_runtime(request, tracking=layout.tracking)
    runtime = _load_project_entry(entry_spec, project_root=layout.root)(request)
    if not isinstance(runtime, JobRuntime):
        raise ContractError(f"project entry {entry_spec!r} must return JobRuntime")
    if runtime.catalog is not catalog:
        raise ContractError("project entry must use the catalog supplied in ProjectExecutionRequest")
    _validate_standard_definitions(runtime)
    return runtime


def _validate_standard_definitions(runtime: JobRuntime) -> None:
    for definition_id, standard in standard_definitions().items():
        configured = runtime.definitions.get(definition_id)
        if configured is None:
            raise ContractError(f"project entry omitted standard job definition: {definition_id}")
        same_operation = getattr(configured.operation, "__code__", None) is getattr(standard.operation, "__code__", None)
        if (
            configured.kind != standard.kind
            or configured.seats != standard.seats
            or not same_operation
        ):
            raise ContractError(f"project entry cannot redefine standard job definition: {definition_id}")


def _layout_payload(layout: ProjectLayout) -> dict[str, object]:
    return {
        "project_id": layout.project_id,
        "root": str(layout.root),
        "manifest": str(layout.manifest),
        "catalog_overlays": [str(path) for path in layout.catalog_overlays],
        "work_packages": str(layout.work_packages),
        "state": str(layout.state),
        "tracking": layout.tracking,
        "entry": layout.entry,
    }


def _catalog_entries(catalog: Any, family: SelectionFamily | None = None) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for ref in catalog.list(family):
        resolved = catalog.resolve(ref)
        entries.append(
            {
                "family": ref.family,
                "id": ref.id,
                "source_layer": resolved.source_layer,
                "overlay_id": resolved.overlay_id,
            }
        )
    return entries


def _catalog_summary(catalog: Any) -> dict[str, object]:
    entries = _catalog_entries(catalog)
    overlay_entries = sum(entry["source_layer"] == "overlay" for entry in entries)
    return {
        "base_catalog_release": catalog.base_id,
        "overlay_ids": list(catalog.overlay_ids),
        "entries": len(entries),
        "base_entries": len(entries) - overlay_entries,
        "project_entries": overlay_entries,
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _emit(args: argparse.Namespace, payload: object, human: str) -> None:
    if args.json_output:
        print(
            json.dumps(_json_value(payload), indent=2, sort_keys=True),
            file=args.json_stream,
            flush=True,
        )
    else:
        print(human, flush=True)


def _work_package_paths(layout: ProjectLayout, configured: Path | None) -> list[Path]:
    if configured is not None:
        return [_work_package_path(layout, configured)]
    if not layout.work_packages.is_dir():
        return []
    return sorted(path for path in layout.work_packages.glob("*.yaml") if path.is_file())


def _materialize_project_references(
    args: argparse.Namespace,
    *,
    work_package: Path | None,
) -> list[dict[str, object]]:
    layout, catalog = _catalog(args)
    results: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for path in _work_package_paths(layout, work_package):
        package = load_work_package(path)
        resolved = resolve_work_package(catalog, package)
        for seat_name, seat in resolved.seats.items():
            selection = seat.value
            if isinstance(selection, DatasetLoadPlan):
                key = ("dataset", selection.id)
                if key in seen:
                    continue
                seen.add(key)
                materialized = materialize_dataset(
                    selection,
                    state_dir=layout.state,
                    project_root=layout.root,
                )
                results.append(
                    {
                        "family": "dataset",
                        "id": materialized.selection_id,
                        "status": "materialized" if materialized.created else "cached",
                        "path": str(materialized.path),
                        "examples": materialized.examples,
                        "seat": seat_name,
                        "work_package": str(path),
                    }
                )
            elif isinstance(selection, EnvironmentBinding):
                key = ("environment", selection.id)
                if key in seen:
                    continue
                seen.add(key)
                preflight_verifiers_environment(selection)
                results.append(
                    {
                        "family": "environment",
                        "id": selection.id,
                        "status": "preflighted",
                        "package": selection.source.package,
                        "seat": seat_name,
                        "work_package": str(path),
                    }
                )
    return results


def _dataset_add(args: argparse.Namespace) -> int:
    layout = _layout(args)
    selection_id = validate_selection_id(args.id, "dataset selection id")
    kind = args.kind
    format_kind = args.format
    if args.dataset_add_kind == "hf":
        source: dict[str, object] = {
            "kind": "huggingface",
            "repo": args.repo,
            "revision": args.revision,
            "split": args.split,
        }
        if args.config:
            source["config"] = args.config
    elif args.dataset_add_kind == "jsonl":
        source = {"kind": "jsonl", "path": args.path}
    elif args.dataset_add_kind == "nemo":
        source = {"kind": "nemo", "path": args.path}
        if kind == "supervised" and format_kind not in {"auto", "messages"}:
            raise ContractError("nemo supervised format must be auto or messages")
        if kind == "preference" and format_kind not in {"auto", "nemo-ranked"}:
            raise ContractError("nemo preference format must be auto or nemo-ranked")
    else:
        raise ContractError(f"unsupported dataset add kind: {args.dataset_add_kind}")

    entry = {
        "revision": selection_revision(selection_id),
        "kind": kind,
        "source": source,
        "format": {"kind": format_kind},
    }
    overlay = overlay_directory(layout)
    path = ensure_overlay_file(overlay, args.file, layer_id=f"{layout.project_id}-v1")
    upsert_family_entry(path, family="dataset", entry_id=selection_id, entry=entry)
    # Validate decode against the composed catalog.
    _, catalog = _catalog(args)
    resolved = catalog.resolve(CatalogRef("dataset", selection_id))
    if not isinstance(resolved.value, DatasetLoadPlan):
        raise ContractError(f"wrote dataset {selection_id!r} but it did not decode as a load plan")
    payload = {
        "id": selection_id,
        "path": str(path),
        "source_layer": resolved.source_layer,
        "overlay_id": resolved.overlay_id,
        "kind": kind,
        "source_kind": resolved.value.source_kind,
    }
    _emit(args, payload, f"Added dataset {selection_id} to {path}")
    return 0


def _environment_add(args: argparse.Namespace) -> int:
    layout = _layout(args)
    if args.environment_add_kind != "local":
        raise ContractError(f"unsupported environment add kind: {args.environment_add_kind}")
    selection_id = validate_selection_id(args.id, "environment selection id")
    source: dict[str, object] = {
        "package": args.package,
        "repository": args.repository,
        "revision": args.revision,
    }
    if args.subdirectory:
        source["subdirectory"] = args.subdirectory
    entry = {
        "category": args.category,
        "source": source,
        "factory": args.factory,
        "sampling": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
        "num_tasks": args.num_tasks,
        "num_rollouts": args.num_rollouts,
    }
    overlay = overlay_directory(layout)
    path = ensure_overlay_file(overlay, args.file, layer_id=f"{layout.project_id}-v1")
    upsert_family_entry(path, family="environment", entry_id=selection_id, entry=entry)
    _, catalog = _catalog(args)
    resolved = catalog.resolve(CatalogRef("environment", selection_id))
    if not isinstance(resolved.value, EnvironmentBinding):
        raise ContractError(f"wrote environment {selection_id!r} but it did not decode as a binding")
    payload = {
        "id": selection_id,
        "path": str(path),
        "source_layer": resolved.source_layer,
        "overlay_id": resolved.overlay_id,
        "package": resolved.value.source.package,
        "factory": args.factory,
    }
    _emit(args, payload, f"Added environment {selection_id} to {path}")
    return 0


def _catalog_materialize(args: argparse.Namespace) -> int:
    results = _materialize_project_references(args, work_package=args.work_package)
    if args.json_output:
        _emit(args, {"items": results}, "")
    elif not results:
        print("No dataset or environment seats found in work packages.")
    else:
        for item in results:
            status = str(item["status"]).upper()
            detail = item.get("path") or item.get("package") or ""
            print(f"{status:12} {item['family']}/{item['id']} {detail}")
    return 0


def _doctor(args: argparse.Namespace) -> int:
    checks: list[dict[str, str]] = [
        {
            "name": "python",
            "status": "ok" if sys.version_info[:2] == (3, 12) else "error",
            "message": f"Python {sys.version_info.major}.{sys.version_info.minor}",
        }
    ]
    layout: ProjectLayout | None = None
    try:
        layout = _layout(args)
        checks.append(
            {
                "name": "project",
                "status": "ok",
                "message": f"{layout.project_id} at {layout.root}",
            }
        )
    except (ContractError, OSError) as error:
        checks.append({"name": "project", "status": "error", "message": str(error)})

    if layout is not None:
        try:
            catalog = open_catalog(
                scope=layout.project_id,
                overlays=layout.catalog_overlays,
                catalog_root=layout.base_catalog,
            )
            summary = _catalog_summary(catalog)
            checks.append(
                {
                    "name": "catalog",
                    "status": "ok",
                    "message": (f"{summary['base_catalog_release']}, {summary['entries']} resolved selections"),
                }
            )
        except (ContractError, KeyError, OSError) as error:
            checks.append({"name": "catalog", "status": "error", "message": _error_message(error)})
        checks.append(
            {
                "name": "work_packages",
                "status": "ok" if layout.work_packages.is_dir() else "error",
                "message": str(layout.work_packages),
            }
        )

    materialize_results: list[dict[str, object]] = []
    if args.fix:
        if layout is None:
            checks.append(
                {
                    "name": "materialize",
                    "status": "error",
                    "message": "cannot fix without a discovered project",
                }
            )
        else:
            try:
                materialize_results = _materialize_project_references(
                    args,
                    work_package=args.work_package,
                )
                checks.append(
                    {
                        "name": "materialize",
                        "status": "ok",
                        "message": f"{len(materialize_results)} dataset/environment seat(s) ready",
                    }
                )
            except (ContractError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                checks.append({"name": "materialize", "status": "error", "message": _error_message(error)})

    succeeded = all(check["status"] == "ok" for check in checks)
    if args.json_output:
        payload: dict[str, object] = {"ok": succeeded, "checks": checks}
        if args.fix:
            payload["materialize"] = materialize_results
        _emit(args, payload, "")
    else:
        for check in checks:
            print(f"{check['status'].upper():5} {check['name']}: {check['message']}")
    return 0 if succeeded else 1


def _observatory_up(args: argparse.Namespace) -> int:
    layout = _layout(args)
    if layout.tracking == "none":
        raise ContractError(
            "Observatory requires project tracking; set tracking to 'trackio' or 'wandb' in "
            ".posttrain/project.toml"
        )
    try:
        observatory = importlib.import_module("posttrain_observatory")
    except ImportError as error:
        raise RuntimeError(
            "Observatory is not installed; run `uv add 'posttrain[observatory]'` "
            "or install the posttrain-observatory package"
        ) from error

    settings_type = getattr(observatory, "ObservatorySettings", None)
    serve = getattr(observatory, "serve", None)
    if settings_type is None or not callable(serve):
        raise RuntimeError(
            "installed Observatory does not expose its server API; upgrade posttrain-observatory"
        )
    settings = settings_type.for_project(
        layout.project_id,
        layout.tracking,
        host=args.host,
        port=args.port,
    )
    host = f"[{settings.host}]" if ":" in settings.host and not settings.host.startswith("[") else settings.host
    url = f"http://{host}:{settings.port}"
    _emit(
        args,
        {
            "project_id": layout.project_id,
            "tracking": layout.tracking,
            "url": url,
        },
        f"Observatory listening at {url}",
    )
    serve(settings)
    return 0


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "version":
        installed = _installed_version()
        _emit(args, {"version": installed}, f"posttrain {installed}")
        return 0
    if args.command == "init":
        if args.no_install and args.template is None:
            raise ContractError("--no-install requires --template")
        layout = _initialize(
            args.path,
            args.project_id,
            template=args.template,
        )
        _emit(
            args,
            _layout_payload(layout),
            f"Initialized post-training project {layout.project_id} at {layout.root}",
        )
        if args.template is not None and not args.no_install:
            _install_starter(layout.root, json_output=args.json_output)
        return 0
    if args.command == "doctor":
        return _doctor(args)
    if args.command == "observatory" and args.observatory_command == "up":
        return _observatory_up(args)
    if args.command == "project" and args.project_command == "show":
        layout = _layout(args)
        _emit(
            args,
            _layout_payload(layout),
            "\n".join(
                (
                    f"Project: {layout.project_id}",
                    f"Root: {layout.root}",
                    f"Manifest: {layout.manifest}",
                    f"Catalog overlays: {', '.join(map(str, layout.catalog_overlays)) or '(none)'}",
                    f"Work packages: {layout.work_packages}",
                    f"State: {layout.state}",
                )
            ),
        )
        return 0
    if args.command == "catalog":
        _, catalog = _catalog(args)
        if args.catalog_command == "list":
            entries = _catalog_entries(catalog, args.family)
            _emit(
                args,
                entries,
                "\n".join(
                    f"{entry['family']}/{entry['id']} "
                    f"[{entry['source_layer']}"
                    f"{':' + str(entry['overlay_id']) if entry['overlay_id'] else ''}]"
                    for entry in entries
                )
                or "No catalog selections found.",
            )
            return 0
        if args.catalog_command == "show":
            resolved = catalog.resolve(CatalogRef(args.family, args.id))
            payload = {
                "family": resolved.ref.family,
                "id": resolved.ref.id,
                "source_layer": resolved.source_layer,
                "overlay_id": resolved.overlay_id,
                "selection": _json_value(resolved.value),
            }
            _emit(
                args,
                payload,
                json.dumps(payload, indent=2, sort_keys=True),
            )
            return 0
        if args.catalog_command == "validate":
            summary = _catalog_summary(catalog)
            _emit(
                args,
                summary,
                (
                    f"Catalog valid: {summary['base_catalog_release']}, "
                    f"{summary['base_entries']} base entries, "
                    f"{summary['project_entries']} project entries"
                ),
            )
            return 0
        if args.catalog_command == "materialize":
            return _catalog_materialize(args)
    if args.command == "dataset" and args.dataset_command == "add":
        return _dataset_add(args)
    if args.command == "environment" and args.environment_command == "add":
        return _environment_add(args)
    if args.command == "dataset" and args.dataset_command == "validate":
        layout, catalog = _catalog(args)
        resolved = catalog.resolve(CatalogRef("dataset", args.id))
        if not isinstance(resolved.value, DatasetLoadPlan):
            raise ContractError(f"catalog dataset {args.id!r} did not resolve to a dataset load plan")
        materialized = materialize_dataset(
            resolved.value,
            state_dir=layout.state,
            project_root=layout.root,
        )
        payload = {
            "id": materialized.selection_id,
            "revision": materialized.selection_revision,
            "source_layer": resolved.source_layer,
            "overlay_id": resolved.overlay_id,
            "source_kind": materialized.source_kind,
            "path": str(materialized.path),
            "manifest": str(materialized.manifest_path),
            "content_sha256": materialized.content_sha256,
            "examples": materialized.examples,
            "materialized": materialized.created,
        }
        action = "Materialized" if materialized.created else "Validated cached"
        _emit(
            args,
            payload,
            f"{action} dataset {materialized.selection_id} "
            f"({materialized.examples} examples) at {materialized.path}",
        )
        return 0
    if args.command == "work-package" and args.work_package_command == "validate":
        layout, catalog, path, package = _load_work_package(args)
        resolved = resolve_work_package(catalog, package)
        output_redirect = redirect_stdout(sys.stderr) if args.json_output else nullcontext()
        with output_redirect:
            context = _runtime_context(args, layout=layout, catalog=catalog, path=path)
            validate_work_package(context, package)
        validation_level = "host" if args.host is not None else "project"
        preflight = "complete"
        payload = {
            "path": str(path),
            "project_id": package.project_id,
            "work_package_id": package.work_package_id,
            "stage": package.stage,
            "recipe_id": resolved.recipe.id,
            "resolved_seats": sorted(resolved.seats),
            "jobs": [
                {
                    "id": job.id,
                    "kind": job.kind,
                    "definition": job.definition,
                    "optional": job.optional,
                    "enabled": not job.optional or job.id in package.enabled_optional_jobs,
                }
                for job in resolved.recipe.jobs
            ],
            "validation_level": validation_level,
            "job_definition_preflight": preflight,
        }
        _emit(
            args,
            payload,
            (
                f"Work package composition valid: {package.work_package_id} "
                f"({len(resolved.seats)} resolved seats, {len(resolved.recipe.jobs)} jobs)\n"
                f"Job-definition preflight: {preflight.replace('-', ' ')}"
            ),
        )
        return 0
    if args.command == "work-package" and args.work_package_command == "run":
        layout, catalog, path, package = _load_work_package(args)
        output_redirect = redirect_stdout(sys.stderr) if args.json_output else nullcontext()
        with output_redirect:
            context = _runtime_context(args, layout=layout, catalog=catalog, path=path)
            result = run_work_package_job(context, package, args.job)
        payload = {
            "path": str(path),
            "entry": args.entry or layout.entry,
            "host": args.host,
            "project_id": result.project_id,
            "work_package_id": result.work_package_id,
            "selected_job": args.job,
            "status": "succeeded",
            "jobs": [
                {
                    "id": job.job_id,
                    "kind": job.kind,
                    "definition": job.definition,
                    "status": job.status,
                    "run_id": job.run_id,
                    "value": _json_value(job.value),
                }
                for job in result.jobs
            ],
        }
        lines = [f"Work package succeeded: {result.work_package_id}"]
        lines.extend(
            f"{job.status.upper():9} {job.job_id} [{job.kind}]"
            + (f" run={job.run_id}" if job.run_id is not None else "")
            for job in result.jobs
        )
        _emit(args, payload, "\n".join(lines))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


def _error_message(error: BaseException) -> str:
    if isinstance(error, KeyError) and error.args:
        return str(error.args[0])
    return str(error)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.json_stream = sys.stdout
    if args.json_output and argv is None:
        # Keep provider/background logs off stdout for the real console entrypoint.
        # The saved stream remains the sole destination for the JSON document.
        sys.stdout = sys.stderr
    try:
        return _dispatch(args)
    except (
        ContractError,
        FileExistsError,
        KeyError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        ValueError,
    ) as error:
        print(f"error: {_error_message(error)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
