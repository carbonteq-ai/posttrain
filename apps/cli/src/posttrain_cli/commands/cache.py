"""Developer-facing inspection and pruning of project-local cache material."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from ..context import CliState
from ..execution_config import MachineCachePolicy, load_machine_config
from ..output import emit
from ..state_layout import explain_cache, migrate_legacy_pack_cache, prune_cache


def register(app: typer.Typer) -> None:
    cache_app = typer.Typer(
        rich_markup_mode=None,
        no_args_is_help=True,
        help="inspect and safely reclaim project-local rebuildable cache",
    )
    app.add_typer(cache_app, name="cache")

    @cache_app.command("status", help="show protected and reclaimable cache material")
    def status_cmd(
        ctx: typer.Context,
        state_root: Annotated[
            Path | None,
            typer.Option("--state-root", help="a .posttrain/state directory; defaults to this project"),
        ] = None,
    ) -> None:
        state: CliState = ctx.obj
        report = prune_cache(state.layout(), state_root=state_root)
        machine = load_machine_config()
        policy = machine.cache if machine is not None else MachineCachePolicy()
        payload = report.as_json()
        payload["policy"] = {
            "total_budget_bytes": policy.total_budget_bytes,
            "minimum_free_bytes": policy.minimum_free_bytes,
            "reusable_max_age_seconds": policy.reusable_max_age_seconds,
            "failed_debug_max_age_seconds": policy.failed_debug_max_age_seconds,
            "retain_failed_debug": policy.retain_failed_debug,
        }
        emit(
            state,
            payload,
            (
                f"Cache status: {report.total_bytes} bytes observed; "
                f"{report.reclaimable_bytes} reclaimable; {report.protected_bytes} protected"
            ),
        )

    @cache_app.command("explain", help="explain one cache path, basename, or content key")
    def explain_cmd(
        ctx: typer.Context,
        selector: Annotated[str, typer.Argument(help="path, basename, or package/publication key")],
        state_root: Annotated[
            Path | None,
            typer.Option("--state-root", help="a .posttrain/state directory; defaults to this project"),
        ] = None,
    ) -> None:
        state: CliState = ctx.obj
        entries = explain_cache(state.layout(), selector, state_root=state_root)
        payload = {
            "selector": selector,
            "entries": [
                {
                    "path": str(entry.path),
                    "classification": entry.classification,
                    "reason": entry.reason,
                    "bytes": entry.bytes,
                    "removed": entry.removed,
                }
                for entry in entries
            ],
        }
        emit(state, payload, "\n".join(f"{entry.classification}: {entry.path} ({entry.reason})" for entry in entries))

    @cache_app.command("prune", help="report or remove safe cache candidates")
    def prune_cmd(
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
        emit(
            state,
            report.as_json(),
            f"Cache prune {'applied' if apply else 'dry run'}; reclaimable bytes: {report.reclaimable_bytes}; removed bytes: {report.removed_bytes}",
        )

    @cache_app.command("migrate", help="verify and record safely recoverable legacy package contexts")
    def migrate_cmd(
        ctx: typer.Context,
        apply: Annotated[
            bool,
            typer.Option("--apply", help="write compact records; without this the command is a read-only plan"),
        ] = False,
    ) -> None:
        state: CliState = ctx.obj
        docker = shutil.which("docker")
        verification: dict[str, bool] = {}

        def verify(image: str) -> bool:
            if image in verification:
                return verification[image]
            if docker is None:
                verification[image] = False
                return False
            try:
                result = subprocess.run(
                    [docker, "buildx", "imagetools", "inspect", image],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
            except (OSError, subprocess.TimeoutExpired):
                verification[image] = False
            else:
                verification[image] = result.returncode == 0
            return verification[image]

        report = migrate_legacy_pack_cache(state.layout(), verify_registry_image=verify, apply=apply)
        emit(
            state,
            report.as_json(),
            (
                f"Legacy package migration {'applied' if apply else 'dry run'}; "
                f"migratable bytes: {report.migratable_bytes}; migrated bytes: {report.migrated_bytes}"
            ),
        )
