"""Run inspection commands via Observatory."""

from __future__ import annotations

import asyncio
import importlib
import json
from typing import Annotated, Any

import click
import typer
from posttrain.common import ContractError

from ..constants import RUN_MODE_CHOICES
from ..context import CliState
from ..output import emit

RUN_MODE_CHOICE = click.Choice(RUN_MODE_CHOICES)


def register(app: typer.Typer) -> None:
    run_app = typer.Typer(rich_markup_mode=None, no_args_is_help=True, help="inspect recorded runs via Observatory")
    app.add_typer(run_app, name="run")

    @run_app.command("show", help="show one recorded run view")
    def run_show_cmd(
        ctx: typer.Context,
        run_id: Annotated[str, typer.Argument(help="canonical run id")],
        source: Annotated[
            str | None,
            typer.Option(
                "--source",
                help="Observatory source id; defaults to the project's tracking source",
            ),
        ] = None,
        mode: Annotated[
            str,
            typer.Option("--mode", click_type=RUN_MODE_CHOICE),
        ] = "auto",
    ) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        if layout.tracking == "none":
            raise ContractError(
                "run show requires project tracking; set tracking to 'trackio' or 'wandb' in "
                ".posttrain/project.toml"
            )
        try:
            observatory = importlib.import_module("posttrain_observatory")
        except ImportError as error:
            raise RuntimeError(
                "Observatory is not installed; run `uv add 'posttrain[observatory]'` "
                "or install the posttrain-observatory package"
            ) from error
        settings_type = getattr(observatory, "ObservatorySettings", None)
        create_service = getattr(observatory, "create_service", None)
        run_locator_type = getattr(observatory, "RunLocator", None)
        if settings_type is None or not callable(create_service) or run_locator_type is None:
            raise RuntimeError(
                "installed Observatory does not expose its query API; upgrade posttrain-observatory"
            )
        settings = settings_type.for_project(layout.project_id, layout.tracking)
        source_id = source or settings.source_id
        service: Any = create_service(settings)

        async def _load() -> Any:
            return await service.get_run_view_response(
                run_locator_type(source_id=source_id, run_id=run_id),
                mode,
            )

        view = asyncio.run(_load())
        payload = view.model_dump(mode="json") if hasattr(view, "model_dump") else view
        emit(
            state,
            payload,
            json.dumps(payload, indent=2, sort_keys=True, default=str),
        )
