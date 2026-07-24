"""Job convenience aliases for work-package plan and run."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ..context import CliState
from .work_package import run_work_package_cmd, validate_work_package_cmd


def register(app: typer.Typer) -> None:
    job_app = typer.Typer(rich_markup_mode=None, no_args_is_help=True, help="convenience aliases for work-package plan and run")
    app.add_typer(job_app, name="job")

    @job_app.command("plan", help="alias for work-package validate (composition/preflight plan)")
    def job_plan_cmd(
        ctx: typer.Context,
        path: Annotated[Path, typer.Argument()],
        host: Annotated[
            str | None,
            typer.Option(
                "--host",
                metavar="MODULE:FACTORY",
                help="also preflight concrete job definitions through this explicit project host",
            ),
        ] = None,
        entry: Annotated[
            str | None,
            typer.Option(
                "--entry",
                metavar="MODULE:FACTORY",
                help="override the optional project entry for this invocation",
            ),
        ] = None,
    ) -> None:
        state: CliState = ctx.obj
        validate_work_package_cmd(state, path, host=host, entry=entry)

    @job_app.command("run", help="alias for work-package run")
    def job_run_cmd(
        ctx: typer.Context,
        path: Annotated[Path, typer.Argument()],
        job: Annotated[
            str,
            typer.Option("--job", help="execute exactly this enabled recipe job id"),
        ],
        host: Annotated[
            str | None,
            typer.Option(
                "--host",
                metavar="MODULE:FACTORY",
                help="deprecated compatibility alias for an explicit legacy host",
            ),
        ] = None,
        entry: Annotated[
            str | None,
            typer.Option(
                "--entry",
                metavar="MODULE:FACTORY",
                help="override the optional project entry for this invocation",
            ),
        ] = None,
    ) -> None:
        state: CliState = ctx.obj
        run_work_package_cmd(state, path, job=job, host=host, entry=entry)
