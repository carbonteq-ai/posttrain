"""Init command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import click
import typer
from posttrain.common import ContractError

from ..constants import TEMPLATE_CHOICES
from ..context import CliState
from ..output import emit
from ..project import layout_payload
from ..scaffolding.init_project import initialize, install_starter

TEMPLATE_CHOICE = click.Choice(TEMPLATE_CHOICES)


def register(app: typer.Typer) -> None:
    @app.command("init", help="initialize a portable post-training project")
    def init_cmd(
        ctx: typer.Context,
        path: Annotated[
            Path | None,
            typer.Argument(help="project directory"),
        ] = None,
        project_id: Annotated[
            str | None,
            typer.Option("--project-id", help="lowercase stable project identity; defaults from PATH"),
        ] = None,
        template: Annotated[
            str | None,
            typer.Option(
                "--template",
                click_type=TEMPLATE_CHOICE,
                help="create and install a runnable starter project",
            ),
        ] = None,
        no_install: Annotated[
            bool,
            typer.Option(
                "--no-install",
                help="write the selected template without creating its environment (CI/bootstrap escape hatch)",
            ),
        ] = False,
    ) -> None:
        state: CliState = ctx.obj
        if no_install and template is None:
            raise ContractError("--no-install requires --template")
        root = path if path is not None else Path.cwd()
        layout = initialize(root, project_id, template=template)
        emit(
            state,
            layout_payload(layout),
            f"Initialized post-training project {layout.project_id} at {layout.root}",
        )
        if template is not None and not no_install:
            install_starter(layout.root, json_output=state.json_output)
