"""Typer application composition for the posttrain CLI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, TextIO

import typer

from .commands import (
    catalog,
    controller,
    dataset,
    doctor,
    environment,
    init_cmd,
    job,
    machine,
    observatory,
    project_cmd,
    purge_cmd,
    run_cmd,
    runtime,
    state,
    version,
    work_package,
    workers,
    workload,
)
from .context import CliState
from .scaffolding.init_project import installed_version


def _version_callback(*, value: bool) -> None:
    if not value:
        return
    typer.echo(f"posttrain {installed_version()}")
    raise typer.Exit(code=0)


def create_app(*, json_stream: TextIO | None = None) -> typer.Typer:
    app = typer.Typer(
        rich_markup_mode=None,
        no_args_is_help=True,
        help="Initialize, inspect, validate, and run post-training projects.",
    )

    @app.callback()
    def root_callback(
        ctx: typer.Context,
        project_root: Annotated[
            Path | None,
            typer.Option(
                "--project-root",
                help="project root containing .posttrain/project.toml; otherwise discover upward",
            ),
        ] = None,
        env_file: Annotated[
            Path | None,
            typer.Option(
                "--env-file",
                help="use this protected runtime-value file instead of project-root posttrain.env",
            ),
        ] = None,
        json_output: Annotated[
            bool,
            typer.Option("--json", help="emit JSON output"),
        ] = False,
        show_traceback: Annotated[
            bool,
            typer.Option(
                "--traceback",
                help="print a full Python traceback for expected CLI failures",
            ),
        ] = False,
        show_version: Annotated[
            bool,
            typer.Option(
                "--version",
                "-V",
                help="show the installed framework CLI version and exit",
                callback=_version_callback,
                is_eager=True,
            ),
        ] = False,
    ) -> None:
        del show_version
        ctx.ensure_object(dict)
        ctx.obj = CliState(
            project_root=project_root,
            env_file=env_file,
            json_output=json_output,
            traceback=show_traceback,
            json_stream=json_stream or sys.stdout,
        )

    version.register(app)
    init_cmd.register(app)
    machine.register(app)
    doctor.register(app)
    project_cmd.register(app)
    purge_cmd.register(app)
    catalog.register(app)
    dataset.register(app)
    environment.register(app)
    work_package.register(app)
    workload.register(app)
    job.register(app)
    run_cmd.register(app)
    controller.register(app)
    workers.register(app)
    runtime.register(app)
    state.register(app)
    observatory.register(app)

    return app
