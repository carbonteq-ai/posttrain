"""Review and apply machine-scoped purge plans."""

from __future__ import annotations

from typing import Annotated

import typer

from ..context import CliState
from ..output import emit, json_value
from ..purge_surface import apply_saved_plan, load_saved_plan, load_saved_tombstone, render_plan


def register(app: typer.Typer) -> None:
    purge_app = typer.Typer(
        rich_markup_mode=None, no_args_is_help=True, help="review and apply destructive purge plans"
    )
    app.add_typer(purge_app, name="purge")

    @purge_app.command("show", help="show an immutable purge plan without contacting providers")
    def purge_show_cmd(
        ctx: typer.Context,
        purge_id: Annotated[str, typer.Argument(help="content-addressed purge id")],
    ) -> None:
        state: CliState = ctx.obj
        layout = state.layout()
        plan = load_saved_plan(layout, purge_id)
        tombstone = load_saved_tombstone(layout, purge_id)
        payload = json_value(plan)
        assert isinstance(payload, dict)
        payload["tombstone"] = json_value(tombstone) if tombstone is not None else None
        emit(state, payload, render_plan(plan, tombstone=tombstone))

    @purge_app.command("apply", help="apply a reviewed purge plan")
    def purge_apply_cmd(
        ctx: typer.Context,
        purge_id: Annotated[str, typer.Argument(help="content-addressed purge id")],
        expect_digest: Annotated[
            str | None,
            typer.Option("--expect-digest", help="complete SHA-256 plan digest for automation"),
        ] = None,
        yes: Annotated[bool, typer.Option("--yes", help="confirm non-interactive destructive apply")] = False,
    ) -> None:
        state: CliState = ctx.obj
        receipt = apply_saved_plan(
            state,
            purge_id,
            expected_digest=expect_digest,
            assume_yes=yes,
        )
        emit(state, receipt, f"Purge applied: {purge_id}")
