"""Safe project-local state inspection and migration commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ..context import CliState
from ..output import emit
from ..state_layout import migrate_state, prune_cache


def register(app: typer.Typer) -> None:
    state_app = typer.Typer(rich_markup_mode=None, no_args_is_help=True, help="migrate and inspect local project state")
    app.add_typer(state_app, name="state")

    @state_app.command("migrate", help="split cache from durable execution records without deleting the source")
    def migrate_cmd(
        ctx: typer.Context,
        from_project_root: Annotated[
            Path | None,
            typer.Option("--from-project-root", help="copy verified execution records from this old project root"),
        ] = None,
        dry_run: Annotated[bool, typer.Option("--dry-run", help="report changes without writing")] = False,
    ) -> None:
        state: CliState = ctx.obj
        report = migrate_state(state.layout(), source_project_root=from_project_root, dry_run=dry_run)
        payload = report.as_json()
        summary = "nothing to do" if not report.changed else "state migration complete"
        if report.protected_entries:
            summary += "; protected: " + ", ".join(report.protected_entries)
        emit(state, payload, summary)

    @state_app.command("cache-prune", help="report or remove recognized rebuildable cache state")
    def cache_prune_cmd(
        ctx: typer.Context,
        state_root: Annotated[
            Path | None,
            typer.Option("--state-root", help="a .posttrain/state directory; defaults to this project"),
        ] = None,
        apply: Annotated[
            bool,
            typer.Option("--apply", help="remove classified cache entries; without this the command is a dry run"),
        ] = False,
    ) -> None:
        state: CliState = ctx.obj
        report = prune_cache(state.layout(), state_root=state_root, apply=apply)
        payload = report.as_json()
        mode = "cache prune applied" if apply else "cache prune dry run"
        emit(
            state,
            payload,
            f"{mode}; reclaimable bytes: {report.reclaimable_bytes}; removed bytes: {report.removed_bytes}",
        )
