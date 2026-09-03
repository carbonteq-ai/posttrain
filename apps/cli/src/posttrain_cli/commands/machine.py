"""Machine configuration commands."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import click
import typer

from ..context import CliState
from ..execution_config import load_machine_config
from ..output import emit
from ..scaffolding.init_machine import add_machine_project, initialize_machine


def register(app: typer.Typer) -> None:
    machine_app = typer.Typer(
        rich_markup_mode=None,
        no_args_is_help=True,
        help="initialize and inspect reusable defaults for this machine",
    )
    app.add_typer(machine_app, name="machine")
    project_app = typer.Typer(
        rich_markup_mode=None,
        no_args_is_help=True,
        help="manage projects served by this machine",
    )
    machine_app.add_typer(project_app, name="project")

    @machine_app.command("show", help="show resolved machine defaults without credential values")
    def machine_show_cmd(ctx: typer.Context) -> None:
        state: CliState = ctx.obj
        machine_config = load_machine_config()
        if machine_config is None:
            raise RuntimeError("machine configuration is missing; run posttrain machine init")
        emit(
            state,
            machine_config,
            "\n".join(
                (
                    f"Machine: {machine_config.name}",
                    f"Config: {machine_config.path}",
                    f"Default provider: {machine_config.defaults.provider}",
                    f"Projects: {len(machine_config.projects)}",
                    f"Tracking: {machine_config.tracking.kind if machine_config.tracking else '(not configured)'}",
                    f"dstack: {'configured' if machine_config.dstack else 'not configured'}",
                    f"Credential sources: {len(machine_config.credentials)} (values redacted)",
                )
            ),
        )

    @machine_app.command("init", help="initialize machine defaults and protected credential sources")
    def machine_init_cmd(
        ctx: typer.Context,
        projects: Annotated[
            list[Path] | None,
            typer.Option("--project", help="register an initialized project; repeat for multiple projects"),
        ] = None,
        machine_name: Annotated[
            str | None,
            typer.Option("--machine-name", help="canonical machine identity; defaults from the hostname"),
        ] = None,
        default_provider: Annotated[
            str,
            typer.Option(
                "--default-provider",
                click_type=click.Choice(("local", "dstack")),
                help="default execution provider",
            ),
        ] = "local",
        trackio_endpoint: Annotated[
            str | None,
            typer.Option("--trackio-endpoint", help="shared credential-free Trackio endpoint"),
        ] = None,
        python_index_url: Annotated[
            str | None,
            typer.Option("--python-index-url", help="shared credential-free Python package index"),
        ] = None,
        job_registry: Annotated[
            str | None,
            typer.Option("--job-registry", help="default OCI repository prefix for project jobs"),
        ] = None,
        job_builder_endpoint: Annotated[
            str | None,
            typer.Option(
                "--job-builder-endpoint",
                help="remote service that builds and publishes project job images",
            ),
        ] = None,
        dstack_project: Annotated[
            str | None,
            typer.Option("--dstack-project", help="dstack project selected by the client adapter"),
        ] = None,
        dstack_python: Annotated[
            Path | None,
            typer.Option("--dstack-python", help="Python executable containing the dstack SDK"),
        ] = None,
    ) -> None:
        state: CliState = ctx.obj
        initialized = initialize_machine(
            projects=tuple(projects or ()),
            machine_name=machine_name,
            default_provider=default_provider,
            trackio_endpoint=trackio_endpoint,
            python_index_url=python_index_url,
            job_registry=job_registry,
            job_builder_endpoint=job_builder_endpoint,
            dstack_project=dstack_project,
            dstack_python=dstack_python or (Path(sys.executable) if dstack_project is not None else None),
        )
        payload = {
            "config": initialized.config,
            "credential_files": initialized.credential_files,
        }
        emit(
            state,
            payload,
            "\n".join(
                (
                    f"Initialized machine configuration at {initialized.config}",
                    f"Protected credential sources: {initialized.config.parent / 'credentials'}",
                    "Fill only the credentials this machine uses; empty sources are valid.",
                )
            ),
        )

    @project_app.command("add", help="register an initialized project with this machine")
    def machine_project_add_cmd(
        ctx: typer.Context,
        project: Annotated[Path, typer.Argument(help="initialized project root")],
    ) -> None:
        state: CliState = ctx.obj
        config, changed = add_machine_project(project)
        resolved = project.expanduser().resolve()
        emit(
            state,
            {"config": config, "project": resolved, "changed": changed},
            (
                f"Registered project {resolved} in {config}"
                if changed
                else f"Project {resolved} is already registered in {config}"
            ),
        )


__all__ = ["register"]
