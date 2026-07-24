"""Environment registration commands."""

from __future__ import annotations

from typing import Annotated

import typer
from posttrain.common import CatalogRef, ContractError
from posttrain.common.selections import validate_selection_id
from posttrain.eval import EnvironmentBinding

from ..context import CliState
from ..output import emit
from ..overlay_write import ensure_overlay_file, overlay_directory, upsert_family_entry


def register(app: typer.Typer) -> None:
    environment_app = typer.Typer(rich_markup_mode=None, no_args_is_help=True, help="register Verifiers environment bindings")
    app.add_typer(environment_app, name="environment")

    add_app = typer.Typer(rich_markup_mode=None, no_args_is_help=True, help="write an environment binding into the project overlay")
    environment_app.add_typer(add_app, name="add")

    @add_app.command("local", help="register an installed Verifiers package binding")
    def environment_add_local_cmd(
        ctx: typer.Context,
        selection_id: Annotated[str, typer.Option("--id")],
        package: Annotated[str, typer.Option("--package")],
        factory: Annotated[str, typer.Option("--factory")],
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
            "factory": factory,
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
            "factory": factory,
        }
        emit(state, payload, f"Added environment {validated_id} to {path}")
