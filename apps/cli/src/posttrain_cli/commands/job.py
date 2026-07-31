"""Job convenience aliases for work-package plan and run."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from ..context import CliState
from ..execution_config import PackageOverrides
from .work_package import (
    _overrides,
    diff_work_package_cmd,
    pack_work_package_cmd,
    plan_work_package_cmd,
    run_work_package_cmd,
)


class _ProviderChoice(StrEnum):
    LOCAL = "local"
    DSTACK = "dstack"


def register(app: typer.Typer) -> None:
    job_app = typer.Typer(
        rich_markup_mode=None,
        no_args_is_help=True,
        help="plan, package, and submit one selected job",
    )
    app.add_typer(job_app, name="job")

    @job_app.command("plan", help="resolve one job without packing or submitting it")
    def job_plan_cmd(
        ctx: typer.Context,
        path: Annotated[
            Path,
            typer.Argument(help="work-package file or project-relative work-package path"),
        ],
        job: Annotated[
            str | None,
            typer.Option(
                "--job",
                help="enabled recipe job id; omit when the package has exactly one",
            ),
        ] = None,
        provider: Annotated[
            _ProviderChoice | None,
            typer.Option(
                "--provider",
                help="override [tool.posttrain.execution] provider for this launch",
            ),
        ] = None,
        target: Annotated[
            str | None,
            typer.Option(
                "--target",
                help="override the selected execution-target id before packaging",
            ),
        ] = None,
        runtime_profile: Annotated[
            str | None,
            typer.Option(
                "--runtime-profile",
                help="override the selected job-kind runtime profile",
            ),
        ] = None,
        timeout_seconds: Annotated[
            int | None,
            typer.Option(
                "--timeout-seconds",
                min=1,
                help="override the provider wall-clock timeout",
            ),
        ] = None,
        max_attempts: Annotated[
            int | None,
            typer.Option(
                "--max-attempts",
                min=1,
                help="override the framework execution-attempt limit",
            ),
        ] = None,
        priority: Annotated[
            int | None,
            typer.Option("--priority", help="override provider scheduling priority"),
        ] = None,
        environment_names: Annotated[
            list[str] | None,
            typer.Option(
                "--env",
                help="require and forward this named environment variable; repeatable",
            ),
        ] = None,
        run_id: Annotated[
            str | None,
            typer.Option(
                "--run-id",
                help="use this durable run identity and idempotency namespace",
            ),
        ] = None,
        host: Annotated[
            str | None,
            typer.Option(
                "--host",
                metavar="MODULE:FACTORY",
                help="also statically validate concrete job definitions through this explicit project host",
                hidden=True,
            ),
        ] = None,
        entry: Annotated[
            str | None,
            typer.Option(
                "--entry",
                metavar="MODULE:FACTORY",
                help="override the optional project entry for this invocation",
            ),
        ] = None,
        project_packages: Annotated[
            list[str] | None,
            typer.Option(
                "--project-package",
                help="override [tool.posttrain.pack].project_packages; repeatable",
            ),
        ] = None,
        source_includes: Annotated[
            list[str] | None,
            typer.Option(
                "--source-include",
                help="override [tool.posttrain.pack].source_includes; repeatable",
            ),
        ] = None,
    ) -> None:
        state: CliState = ctx.obj
        plan_work_package_cmd(
            state,
            path,
            job=job,
            overrides=_overrides(
                provider=(provider.value if provider is not None else None),
                target=target,
                runtime_profile=runtime_profile,
                timeout_seconds=timeout_seconds,
                max_attempts=max_attempts,
                priority=priority,
                environment_names=environment_names,
            ),
            run_id=run_id,
            host=host,
            entry=entry,
            project_packages=(tuple(project_packages) if project_packages is not None else None),
            source_includes=(tuple(source_includes) if source_includes is not None else None),
        )

    @job_app.command(
        "diff",
        help="explain why two packed job packages have different identities",
    )
    def job_diff_cmd(
        ctx: typer.Context,
        path: Annotated[
            Path,
            typer.Argument(help="work-package file or project-relative work-package path"),
        ],
        job: Annotated[
            str | None,
            typer.Option(
                "--job",
                help="enabled recipe job id; omit when the package has exactly one",
            ),
        ] = None,
        from_key: Annotated[
            str | None,
            typer.Option("--from", help="earlier package key, or an unambiguous prefix"),
        ] = None,
        to_key: Annotated[
            str | None,
            typer.Option("--to", help="later package key, or an unambiguous prefix"),
        ] = None,
    ) -> None:
        diff_work_package_cmd(
            ctx.obj,
            path,
            job=job,
            from_key=from_key,
            to_key=to_key,
        )

    @job_app.command("pack", help="materialize one registry image or local OCI layout")
    def job_pack_cmd(
        ctx: typer.Context,
        path: Annotated[
            Path,
            typer.Argument(help="work-package file or project-relative work-package path"),
        ],
        job: Annotated[
            str | None,
            typer.Option(
                "--job",
                help="enabled recipe job id; omit when the package has exactly one",
            ),
        ] = None,
        target: Annotated[
            str | None,
            typer.Option(
                "--target",
                help="override the selected execution-target id before packaging",
            ),
        ] = None,
        runtime_profile: Annotated[
            str | None,
            typer.Option(
                "--runtime-profile",
                help="override the selected job-kind runtime profile",
            ),
        ] = None,
        host: Annotated[
            str | None,
            typer.Option(
                "--host",
                metavar="MODULE:FACTORY",
                help="also statically validate definitions through this compatibility host",
                hidden=True,
            ),
        ] = None,
        entry: Annotated[
            str | None,
            typer.Option(
                "--entry",
                metavar="MODULE:FACTORY",
                help="override the optional project entry for this invocation",
            ),
        ] = None,
        project_packages: Annotated[
            list[str] | None,
            typer.Option(
                "--project-package",
                help="override [tool.posttrain.pack].project_packages; repeatable",
            ),
        ] = None,
        source_includes: Annotated[
            list[str] | None,
            typer.Option(
                "--source-include",
                help="override [tool.posttrain.pack].source_includes; repeatable",
            ),
        ] = None,
        build_missing: Annotated[
            bool,
            typer.Option(
                "--build-missing",
                help="rebuild absent or drifted job-kind images from the shipped definitions",
            ),
        ] = False,
        local: Annotated[
            bool,
            typer.Option(
                "--local",
                help="export a verified local OCI layout without publishing to a registry",
            ),
        ] = False,
    ) -> None:
        state: CliState = ctx.obj
        pack_work_package_cmd(
            state,
            path,
            job=job,
            overrides=PackageOverrides(
                target=target,
                runtime_profile=runtime_profile,
            ),
            host=host,
            entry=entry,
            project_packages=(tuple(project_packages) if project_packages is not None else None),
            source_includes=(tuple(source_includes) if source_includes is not None else None),
            build_missing=build_missing,
            local=local,
        )

    @job_app.command("run", help="pack if needed and submit one selected job")
    def job_run_cmd(
        ctx: typer.Context,
        path: Annotated[
            Path,
            typer.Argument(help="work-package file or project-relative work-package path"),
        ],
        job: Annotated[
            str | None,
            typer.Option(
                "--job",
                help="enabled recipe job id; omit when the package has exactly one",
            ),
        ] = None,
        provider: Annotated[
            _ProviderChoice | None,
            typer.Option(
                "--provider",
                help="override [tool.posttrain.execution] provider for this launch",
            ),
        ] = None,
        target: Annotated[
            str | None,
            typer.Option(
                "--target",
                help="override the selected execution-target id before packaging",
            ),
        ] = None,
        runtime_profile: Annotated[
            str | None,
            typer.Option(
                "--runtime-profile",
                help="override the selected job-kind runtime profile",
            ),
        ] = None,
        timeout_seconds: Annotated[
            int | None,
            typer.Option(
                "--timeout-seconds",
                min=1,
                help="override the provider wall-clock timeout",
            ),
        ] = None,
        max_attempts: Annotated[
            int | None,
            typer.Option(
                "--max-attempts",
                min=1,
                help="override the framework execution-attempt limit",
            ),
        ] = None,
        priority: Annotated[
            int | None,
            typer.Option("--priority", help="override provider scheduling priority"),
        ] = None,
        environment_names: Annotated[
            list[str] | None,
            typer.Option(
                "--env",
                help="require and forward this named environment variable; repeatable",
            ),
        ] = None,
        run_id: Annotated[
            str | None,
            typer.Option(
                "--run-id",
                help="use this durable run identity and idempotency namespace",
            ),
        ] = None,
        host: Annotated[
            str | None,
            typer.Option(
                "--host",
                metavar="MODULE:FACTORY",
                help="deprecated compatibility alias for an explicit legacy host",
                hidden=True,
            ),
        ] = None,
        entry: Annotated[
            str | None,
            typer.Option(
                "--entry",
                metavar="MODULE:FACTORY",
                help="override the optional project entry for this invocation",
            ),
        ] = None,
        in_process: Annotated[
            bool,
            typer.Option(
                "--in-process",
                help="temporary compatibility mode",
                hidden=True,
            ),
        ] = False,
        project_packages: Annotated[
            list[str] | None,
            typer.Option(
                "--project-package",
                help="override [tool.posttrain.pack].project_packages; repeatable",
            ),
        ] = None,
        source_includes: Annotated[
            list[str] | None,
            typer.Option(
                "--source-include",
                help="override [tool.posttrain.pack].source_includes; repeatable",
            ),
        ] = None,
        build_missing: Annotated[
            bool,
            typer.Option(
                "--build-missing",
                help="rebuild absent or drifted job-kind images from the shipped definitions",
            ),
        ] = False,
    ) -> None:
        state: CliState = ctx.obj
        run_work_package_cmd(
            state,
            path,
            job=job,
            host=host,
            entry=entry,
            in_process=in_process,
            overrides=_overrides(
                provider=(provider.value if provider is not None else None),
                target=target,
                runtime_profile=runtime_profile,
                timeout_seconds=timeout_seconds,
                max_attempts=max_attempts,
                priority=priority,
                environment_names=environment_names,
            ),
            run_id=run_id,
            project_packages=(tuple(project_packages) if project_packages is not None else None),
            source_includes=(tuple(source_includes) if source_includes is not None else None),
            build_missing=build_missing,
        )
