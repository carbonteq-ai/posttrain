"""Environment registration commands."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

import typer
from posttrain.common import CatalogRef, ContractError
from posttrain.common.selections import validate_selection_id
from posttrain.eval import EnvironmentBinding

from ..context import CliState
from ..output import emit
from ..overlay_write import ensure_overlay_file, overlay_directory, upsert_family_entry

_ENVIRONMENT_NAME = re.compile(r"^[a-z][a-z0-9-]*$")


def register(app: typer.Typer) -> None:
    environment_app = typer.Typer(
        rich_markup_mode=None, no_args_is_help=True, help="register Verifiers environment bindings"
    )
    app.add_typer(environment_app, name="environment")

    add_app = typer.Typer(
        rich_markup_mode=None, no_args_is_help=True, help="write an environment binding into the project overlay"
    )
    environment_app.add_typer(add_app, name="add")

    @environment_app.command("new", help="scaffold an installable Verifiers environment package")
    def environment_new_cmd(
        ctx: typer.Context,
        name: Annotated[str, typer.Argument(help="environment directory and distribution suffix")],
    ) -> None:
        if not _ENVIRONMENT_NAME.fullmatch(name):
            raise ContractError("environment name must use lowercase letters, digits, and hyphens")
        state: CliState = ctx.obj
        root = state.layout().root / "environments" / name
        if root.exists():
            raise FileExistsError(f"environment scaffold already exists: {root}")
        module = name.replace("-", "_")
        package = f"{module}_env"
        _write_scaffold(
            root,
            {
                "pyproject.toml": f'''[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{name}-env"
version = "0.1.0"
description = "Project-local Verifiers environment"
requires-python = ">=3.12"
dependencies = ["verifiers"]

[tool.hatch.build.targets.wheel]
packages = ["src/{package}"]
''',
                f"src/{package}/__init__.py": '"""Project-local Verifiers environment package."""\n',
                f"src/{package}/taskset.py": '''"""Define the taskset and rewards for this environment."""


def create_environment():
    """Construct this environment's Verifiers taskset.

    Replace this placeholder with a native Verifiers v1 taskset factory, then
    bind it from the project catalog using a project-path environment source.
    """

    raise NotImplementedError("implement the project-local Verifiers environment")
''',
                "README.md": f"# {name} environment\n\nImplement `create_environment` in `src/{package}/taskset.py`, then bind this package from the project catalog.\n",
            },
        )
        emit(
            state,
            {"name": name, "path": str(root), "package": f"{name}-env", "module": package},
            f"Created Verifiers environment scaffold: {root}",
        )

    @add_app.command("local", help="register an installed Verifiers package binding")
    def environment_add_local_cmd(
        ctx: typer.Context,
        selection_id: Annotated[str, typer.Option("--id")],
        package: Annotated[str, typer.Option("--package")],
        factory_ref: Annotated[
            str,
            typer.Option(
                "--factory-ref",
                "--factory",
                help="importable module:callable reference; --factory is a compatibility alias",
            ),
        ],
        repository: Annotated[str, typer.Option("--repository")],
        revision: Annotated[str, typer.Option("--revision", help="immutable commit SHA")],
        subdirectory: Annotated[str | None, typer.Option("--subdirectory")] = None,
        category: Annotated[str, typer.Option("--category")] = "custom",
        num_tasks: Annotated[int, typer.Option("--num-tasks")] = 8,
        num_rollouts: Annotated[int, typer.Option("--num-rollouts")] = 1,
        max_tokens: Annotated[int, typer.Option("--max-tokens")] = 2048,
        temperature: Annotated[float, typer.Option("--temperature")] = 1.0,
        file: Annotated[str, typer.Option("--file")] = "environments.yaml",
    ) -> None:
        state: CliState = ctx.obj
        validated_id = validate_selection_id(selection_id, "environment selection id")
        source: dict[str, object] = {
            "package": package,
            "repository": repository,
            "revision": revision,
        }
        if subdirectory:
            source["subdirectory"] = subdirectory
        entry = {
            "category": category,
            "source": source,
            "activation": {
                "kind": "python-factory",
                "reference": factory_ref,
            },
            "sampling": {
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            "num_tasks": num_tasks,
            "num_rollouts": num_rollouts,
        }
        layout = state.layout()
        overlay = overlay_directory(layout)
        path = ensure_overlay_file(overlay, file, layer_id=f"{layout.project_id}-v1")
        upsert_family_entry(path, family="environment", entry_id=validated_id, entry=entry)
        _, catalog = state.open_catalog()
        resolved = catalog.resolve(CatalogRef("environment", validated_id))
        if not isinstance(resolved.value, EnvironmentBinding):
            raise ContractError(f"wrote environment {validated_id!r} but it did not decode as a binding")
        payload = {
            "id": validated_id,
            "path": str(path),
            "source_layer": resolved.source_layer,
            "overlay_id": resolved.overlay_id,
            "package": resolved.value.source.package,
            "activation": resolved.value.activation.to_payload(),
        }
        emit(state, payload, f"Added environment {validated_id} to {path}")


def _write_scaffold(root: Path, files: dict[str, str]) -> None:
    root.mkdir(parents=True)
    for relative, contents in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents, encoding="utf-8")
