"""Workload-owned record population commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from posttrain.common import CatalogRef
from posttrain.serve import materialize_workload, verify_workload

from ..context import CliState
from ..output import emit


def register(app: typer.Typer) -> None:
    """Register generic workload materialization and verification commands."""

    workload_app = typer.Typer(
        rich_markup_mode=None,
        no_args_is_help=True,
        help="build and verify workload-owned record populations",
    )
    app.add_typer(workload_app, name="workload")

    @workload_app.command("materialize", help="build a workload's record population")
    def workload_materialize_cmd(
        ctx: typer.Context,
        workload_id: Annotated[str, typer.Argument(metavar="id")],
        output: Annotated[
            Path | None,
            typer.Option("--output", help="directory or JSONL output path; defaults to project state"),
        ] = None,
    ) -> None:
        state: CliState = ctx.obj
        layout, workload = _resolve_workload(state, workload_id)
        target = output or layout.state / "workloads" / _workload_slug(workload.id)
        result = materialize_workload(workload, output=target).to_payload()
        emit(
            state,
            result,
            f"Materialized workload {workload_id}: {result['record_count']} records, "
            f"{result['content_sha256']} at {result['path']}",
        )

    @workload_app.command("verify", help="rebuild a workload population and verify packaged bytes")
    def workload_verify_cmd(
        ctx: typer.Context,
        workload_id: Annotated[str, typer.Argument(metavar="id")],
    ) -> None:
        state: CliState = ctx.obj
        _, workload = _resolve_workload(state, workload_id)
        result = verify_workload(workload).to_payload()
        emit(
            state,
            {**result, "verified": True},
            f"Verified workload {workload_id}: {result['record_count']} records, {result['content_sha256']}",
        )


def _resolve_workload(state: CliState, workload_id: str):
    layout, catalog = state.open_catalog()
    resolved = catalog.resolve(CatalogRef("workload", workload_id))
    return layout, resolved.value


def _workload_slug(workload_id: str) -> str:
    return workload_id.rsplit("/", 1)[-1].replace("@", "-")


__all__ = ["register"]
