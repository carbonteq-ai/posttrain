"""Safe project-local state inspection and migration commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ..context import CliState
from ..output import emit
from ..state_layout import migrate_state


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
