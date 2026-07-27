"""Observatory server commands."""

from __future__ import annotations

import importlib
from typing import Annotated

import typer
from posttrain.common import ContractError

from ..context import CliState
from ..output import emit
from ..tracking_config import project_observatory_settings


def register(app: typer.Typer) -> None:
    observatory_app = typer.Typer(
        rich_markup_mode=None, no_args_is_help=True, help="start the project evidence product"
    )
    app.add_typer(observatory_app, name="observatory")

    @observatory_app.command("up", help="serve Observatory for the discovered project's tracking backend")
    def observatory_up_cmd(
        ctx: typer.Context,
        host: Annotated[
            str | None,
            typer.Option("--host", help="listening host; defaults to Observatory settings"),
        ] = None,
        port: Annotated[
            int | None,
            typer.Option("--port", help="listening port; defaults to 7861"),
        ] = None,
    ) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        if layout.tracking == "none":
            raise ContractError(
                "Observatory requires project tracking; set tracking to 'trackio' or 'wandb' in .posttrain/project.toml"
            )
        try:
            observatory = importlib.import_module("posttrain_observatory")
        except ImportError as error:
            raise RuntimeError(
                "Observatory is not installed; run `uv add 'posttrain[observatory]'` "
                "or install the posttrain-observatory package"
            ) from error

        settings_type = getattr(observatory, "ObservatorySettings", None)
        serve = getattr(observatory, "serve", None)
        if settings_type is None or not callable(serve):
            raise RuntimeError("installed Observatory does not expose its server API; upgrade posttrain-observatory")
        settings = project_observatory_settings(
            layout,
            settings_type,
            host=host,
            port=port,
        )
        display_host = (
            f"[{settings.host}]" if ":" in settings.host and not settings.host.startswith("[") else settings.host
        )
        url = f"http://{display_host}:{settings.port}"
        emit(
            state,
            {
                "project_id": layout.project_id,
                "tracking": layout.tracking,
                "url": url,
            },
            f"Observatory listening at {url}",
        )
        serve(settings)
