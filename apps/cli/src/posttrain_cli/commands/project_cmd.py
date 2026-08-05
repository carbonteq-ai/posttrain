"""Project inspect commands."""

from __future__ import annotations

import typer

from ..context import CliState
from ..output import emit
from ..project import layout_payload
from ..purge_surface import render_plan, save_project_preview


def register(app: typer.Typer) -> None:
    project_app = typer.Typer(rich_markup_mode=None, no_args_is_help=True, help="inspect project configuration")
    app.add_typer(project_app, name="project")

    @project_app.command("show", help="show the discovered project layout")
    def project_show_cmd(ctx: typer.Context) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        payload = layout_payload(layout)
        emit(
            state,
            payload,
            "\n".join(
                (
                    f"Project: {layout.project_id}",
                    f"Root: {layout.root}",
                    f"Manifest: {layout.manifest}",
                    f"Catalog overlays: {', '.join(map(str, layout.catalog_overlays)) or '(none)'}",
                    f"Work packages: {layout.work_packages}",
                    f"State: {layout.state}",
                    f"Project brief: {layout.project_brief or '(not configured)'}",
                    f"Project brief digest: {payload['project_brief_digest'] or '(none)'}",
                    (
                        "Serving requirements: "
                        + ("configured" if payload["serving_requirements"] == "configured" else "not configured")
                    ),
                )
            ),
        )

    @project_app.command("purge", help="preview destructive deletion for the opened project")
    def project_purge_cmd(ctx: typer.Context) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        plan = save_project_preview(layout)
        emit(state, plan, render_plan(plan))
