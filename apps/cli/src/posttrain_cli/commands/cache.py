"""Developer-facing inspection and pruning of project-local cache material."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from posttrain_execution_buildkit import BuildxCli

from ..context import CliState
from ..execution_config import MachineCachePolicy, load_machine_config
from ..output import emit
from ..state_layout import explain_cache, migrate_legacy_pack, prune_cache


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

    @cache_app.command(
        "migrate-legacy-pack",
        help="prepare this project's historical pack cache for safe pruning",
    )
    def migrate_legacy_pack_cmd(
        ctx: typer.Context,
        apply: Annotated[
            bool,
            typer.Option(
                "--apply",
                help="commit compact migration evidence; without this the command is a dry run",
            ),
        ] = False,
    ) -> None:
        state: CliState = ctx.obj
        gateway = BuildxCli()

        def verify_remote(image: str) -> bool:
            expected = image.rpartition("@sha256:")[2]
            if len(expected) != 64:
                return False
            output = gateway.invoke(("imagetools", "inspect", image, "--format", "{{json .Manifest.Digest}}"))
            try:
                observed = json.loads(output)
            except json.JSONDecodeError:
                return False
            return observed == f"sha256:{expected}"

        report = migrate_legacy_pack(state.layout(), verify_remote=verify_remote, apply=apply)
        emit(
            state,
            report.as_json(),
            (
                f"Legacy pack migration {'applied' if apply else 'dry run'}; "
                f"migratable bytes: {report.migratable_bytes}; "
                f"protected bytes: {report.protected_bytes}; "
                f"records committed: {report.records_committed}; "
                f"discard records committed: {report.discard_records_committed}"
            ),
        )
