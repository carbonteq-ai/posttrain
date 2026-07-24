"""Project inspect commands."""

from __future__ import annotations

import typer

from ..context import CliState
from ..output import emit
from ..project import layout_payload


def register(app: typer.Typer) -> None:
    project_app = typer.Typer(rich_markup_mode=None, no_args_is_help=True, help="inspect project configuration")
    app.add_typer(project_app, name="project")

    @project_app.command("show", help="show the discovered project layout")
    def project_show_cmd(ctx: typer.Context) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        emit(
            state,
            layout_payload(layout),
            "\n".join(
                (
                    f"Project: {layout.project_id}",
                    f"Root: {layout.root}",
                    f"Manifest: {layout.manifest}",
                    f"Catalog overlays: {', '.join(map(str, layout.catalog_overlays)) or '(none)'}",
                    f"Work packages: {layout.work_packages}",
                    f"State: {layout.state}",
                )
            ),
        )
