"""Catalog inspect commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import click
import typer
from posttrain.common import CatalogRef
from posttrain.common.selections import SelectionFamily

from ..constants import CATALOG_FAMILIES
from ..context import CliState
from ..materialize import materialize_project_references
from ..output import emit, json_value
from ..project import catalog_entries, catalog_summary

FAMILY_CHOICE = click.Choice(CATALOG_FAMILIES)


def register(app: typer.Typer) -> None:
    catalog_app = typer.Typer(rich_markup_mode=None, no_args_is_help=True, help="inspect the composed project catalog")
    app.add_typer(catalog_app, name="catalog")

    @catalog_app.command("list", help="list resolved catalog selections")
    def catalog_list_cmd(
        ctx: typer.Context,
        family: Annotated[
            SelectionFamily | None,
            typer.Option("--family", click_type=FAMILY_CHOICE),
        ] = None,
    ) -> None:
        state: CliState = ctx.obj
        _, catalog = state.open_catalog()
        entries = catalog_entries(catalog, family)
        emit(
            state,
            entries,
            "\n".join(
                f"{entry['family']}/{entry['id']} "
                f"[{entry['source_layer']}"
                f"{':' + str(entry['overlay_id']) if entry['overlay_id'] else ''}]"
                for entry in entries
            )
            or "No catalog selections found.",
        )

    @catalog_app.command("show", help="show one resolved catalog selection")
    def catalog_show_cmd(
        ctx: typer.Context,
        family: Annotated[SelectionFamily, typer.Argument(click_type=FAMILY_CHOICE)],
        selection_id: Annotated[str, typer.Argument(metavar="id")],
    ) -> None:
        state: CliState = ctx.obj
        _, catalog = state.open_catalog()
        resolved = catalog.resolve(CatalogRef(family, selection_id))
        payload = {
            "family": resolved.ref.family,
            "id": resolved.ref.id,
            "source_layer": resolved.source_layer,
            "overlay_id": resolved.overlay_id,
            "selection": json_value(resolved.value),
        }
        emit(
            state,
            payload,
            json.dumps(payload, indent=2, sort_keys=True),
        )

    @catalog_app.command("validate", help="load and validate the complete catalog")
    def catalog_validate_cmd(ctx: typer.Context) -> None:
        state: CliState = ctx.obj
        _, catalog = state.open_catalog()
        summary = catalog_summary(catalog)
        emit(
            state,
            summary,
            (
                f"Catalog valid: {summary['base_catalog_release']}, "
                f"{summary['base_entries']} base entries, "
                f"{summary['project_entries']} project entries"
            ),
        )

    @catalog_app.command(
        "materialize",
        help="materialize datasets and preflight environments referenced by work packages",
    )
    def catalog_materialize_cmd(
        ctx: typer.Context,
        work_package: Annotated[
            Path | None,
            typer.Option("--work-package", help="limit materialization to one work package path"),
        ] = None,
    ) -> None:
        state: CliState = ctx.obj
        results = materialize_project_references(state, work_package=work_package)
        if state.json_output:
            emit(state, {"items": results}, "")
        elif not results:
            print("No dataset or environment seats found in work packages.")
        else:
            for item in results:
                status = str(item["status"]).upper()
                detail = item.get("path") or item.get("package") or ""
                print(f"{status:12} {item['family']}/{item['id']} {detail}")
