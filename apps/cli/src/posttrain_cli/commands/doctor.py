"""Doctor command."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from posttrain.catalog import open_catalog
from posttrain.common import ContractError

from ..checks import registry_check, runtime_images_check
from ..context import CliState
from ..errors import error_message
from ..materialize import materialize_project_references
from ..output import emit
from ..project import catalog_summary


def register(app: typer.Typer) -> None:
    @app.command("doctor", help="check project and catalog readiness")
    def doctor_cmd(
        ctx: typer.Context,
        fix: Annotated[
            bool,
            typer.Option(
                "--fix",
                help="also materialize datasets and preflight environments referenced by work packages",
            ),
        ] = False,
        work_package: Annotated[
            Path | None,
            typer.Option("--work-package", help="limit --fix to one work package path"),
        ] = None,
    ) -> None:
        state: CliState = ctx.obj
        checks: list[dict[str, str]] = [
            {
                "name": "python",
                "status": "ok" if sys.version_info[:2] == (3, 13) else "error",
                "message": f"Python {sys.version_info.major}.{sys.version_info.minor}",
            }
        ]
        layout = None
        try:
            layout = state.layout()
            checks.append(
                {
                    "name": "project",
                    "status": "ok",
                    "message": f"{layout.project_id} at {layout.root}",
                }
            )
        except (ContractError, OSError) as error:
            checks.append({"name": "project", "status": "error", "message": str(error)})

        if layout is not None:
            try:
                catalog = open_catalog(
                    scope=layout.project_id,
                    overlays=layout.catalog_overlays,
                    catalog_root=layout.base_catalog,
                )
                summary = catalog_summary(catalog)
                checks.append(
                    {
                        "name": "catalog",
                        "status": "ok",
                        "message": (f"{summary['base_catalog_release']}, {summary['entries']} resolved selections"),
                    }
                )
            except (ContractError, KeyError, OSError) as error:
                checks.append({"name": "catalog", "status": "error", "message": error_message(error)})
            checks.append(
                {
                    "name": "work_packages",
                    "status": "ok" if layout.work_packages.is_dir() else "error",
                    "message": str(layout.work_packages),
                }
            )

        materialize_results: list[dict[str, object]] = []
        if fix:
            if layout is None:
                checks.append(
                    {
                        "name": "materialize",
                        "status": "error",
                        "message": "cannot fix without a discovered project",
                    }
                )
            else:
                try:
                    materialize_results = materialize_project_references(
                        state,
                        work_package=work_package,
                    )
                    checks.append(
                        {
                            "name": "materialize",
                            "status": "ok",
                            "message": f"{len(materialize_results)} dataset/environment seat(s) ready",
                        }
                    )
                except (ContractError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                    checks.append({"name": "materialize", "status": "error", "message": error_message(error)})

        if layout is not None:
            checks.append(registry_check(state).as_dict())
            checks.append(runtime_images_check(state).as_dict())

        succeeded = all(check["status"] != "error" for check in checks)
        if state.json_output:
            payload: dict[str, object] = {"ok": succeeded, "checks": checks}
            if fix:
                payload["materialize"] = materialize_results
            emit(state, payload, "")
        else:
            for check in checks:
                print(f"{check['status'].upper():5} {check['name']}: {check['message']}")
        if not succeeded:
            raise typer.Exit(code=1)
