"""Version command."""

from __future__ import annotations

import typer

from ..context import CliState
from ..output import emit
from ..scaffolding.init_project import installed_version


def register(app: typer.Typer) -> None:
    @app.command("version", help="show the installed framework CLI version")
    def version_cmd(ctx: typer.Context) -> None:
        state: CliState = ctx.obj
        installed = installed_version()
        emit(state, {"version": installed}, f"posttrain {installed}")
