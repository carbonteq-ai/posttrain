"""Work-package inspect and execute commands."""

from __future__ import annotations

import sys
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from typing import Annotated

import typer
from posttrain.work import resolve_work_package, run_work_package_job, validate_work_package

from ..context import CliState
from ..output import emit, json_value
from ..work_runtime import load_work_package_bundle, runtime_context


def validate_work_package_cmd(
    state: CliState,
    path: Path,
    *,
    host: str | None = None,
    entry: str | None = None,
) -> None:
    layout, catalog, resolved_path, package = load_work_package_bundle(state, path)
    resolved = resolve_work_package(catalog, package)
    output_redirect = redirect_stdout(sys.stderr) if state.json_output else nullcontext()
    with output_redirect:
        context = runtime_context(
            layout=layout,
            catalog=catalog,
            path=resolved_path,
            host=host,
            entry=entry,
        )
        validate_work_package(context, package)
    validation_level = "host" if host is not None else "project"
    preflight = "complete"
    payload = {
        "path": str(resolved_path),
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
    emit(
        state,
        payload,
        (
            f"Work package composition valid: {package.work_package_id} "
            f"({len(resolved.seats)} resolved seats, {len(resolved.recipe.jobs)} jobs)\n"
            f"Job-definition preflight: {preflight.replace('-', ' ')}"
        ),
    )


def run_work_package_cmd(
    state: CliState,
    path: Path,
    *,
    job: str,
    host: str | None = None,
    entry: str | None = None,
) -> None:
    layout, catalog, resolved_path, package = load_work_package_bundle(state, path)
    output_redirect = redirect_stdout(sys.stderr) if state.json_output else nullcontext()
    with output_redirect:
        context = runtime_context(
            layout=layout,
            catalog=catalog,
            path=resolved_path,
            host=host,
            entry=entry,
        )
        result = run_work_package_job(context, package, job)
    payload = {
        "path": str(resolved_path),
        "entry": entry or layout.entry,
        "host": host,
        "project_id": result.project_id,
        "work_package_id": result.work_package_id,
        "selected_job": job,
        "status": "succeeded",
        "jobs": [
            {
                "id": job_result.job_id,
                "kind": job_result.kind,
                "definition": job_result.definition,
                "status": job_result.status,
                "run_id": job_result.run_id,
                "value": json_value(job_result.value),
            }
            for job_result in result.jobs
        ],
    }
    lines = [f"Work package succeeded: {result.work_package_id}"]
    lines.extend(
        f"{job_result.status.upper():9} {job_result.job_id} [{job_result.kind}]"
        + (f" run={job_result.run_id}" if job_result.run_id is not None else "")
        for job_result in result.jobs
    )
    emit(state, payload, "\n".join(lines))


def register(app: typer.Typer) -> None:
    work_package_app = typer.Typer(
        rich_markup_mode=None, no_args_is_help=True, help="inspect and execute work packages"
    )
    app.add_typer(work_package_app, name="work-package")

    @work_package_app.command("validate", help="validate YAML, recipe structure, and catalog bindings")
    def work_package_validate_cmd(
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

    @work_package_app.command("run", help="execute a validated work package through an explicit project host")
    def work_package_run_cmd(
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
