"""Typer application composition for the posttrain CLI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, TextIO

import typer

from .commands import (
    catalog,
    dataset,
    doctor,
    environment,
    init_cmd,
    job,
    observatory,
    project_cmd,
    run_cmd,
    runtime,
    version,
    work_package,
    workers,
)
from .context import CliState


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
        json_output: Annotated[
            bool,
            typer.Option("--json", help="emit JSON output"),
        ] = False,
    ) -> None:
        ctx.ensure_object(dict)
        ctx.obj = CliState(
            project_root=project_root,
            json_output=json_output,
            json_stream=json_stream or sys.stdout,
        )

    version.register(app)
    init_cmd.register(app)
    doctor.register(app)
    project_cmd.register(app)
    catalog.register(app)
    dataset.register(app)
    environment.register(app)
    work_package.register(app)
    job.register(app)
    run_cmd.register(app)
    workers.register(app)
    runtime.register(app)
    observatory.register(app)

    return app
