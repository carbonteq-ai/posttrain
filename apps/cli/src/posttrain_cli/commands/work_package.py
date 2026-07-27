"""Work-package inspect and execute commands."""

from __future__ import annotations

import sys
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from typing import Annotated

import typer
from posttrain.work import resolve_work_package, run_work_package_job, validate_work_package

from ..context import CliState
from ..execution_config import ExecutionOverrides, PackageOverrides
from ..execution_planning import (
    PackedJobExecution,
    PackedJobPackage,
    PlannedJobExecution,
    PlannedJobPackage,
    plan_job_execution,
    plan_job_package,
)
from ..execution_provider import (
    evidence_source_for_project,
    execution_admission_service,
)
from ..output import emit, json_value
from ..runtime_images import ensure_kind_image_ready
from ..work_runtime import load_work_package_bundle, runtime_context

_EMPTY_OVERRIDES = ExecutionOverrides()
_EMPTY_PACKAGE_OVERRIDES = PackageOverrides()


def validate_work_package_cmd(
    state: CliState,
    path: Path,
    *,
    host: str | None = None,
    entry: str | None = None,
) -> None:
    layout, catalog, resolved_path, package = load_work_package_bundle(state, path)
    resolved = resolve_work_package(catalog, package)
    output_redirect = redirect_stdout(sys.stderr) if state.json_output else nullcontext()
    with output_redirect:
        context = runtime_context(
            layout=layout,
            catalog=catalog,
            path=resolved_path,
            host=host,
            entry=entry,
            activate=False,
        )
        validate_work_package(context, package)
    validation_level = "host" if host is not None else "project"
    composition_validation = "complete"
    payload = {
        "path": str(resolved_path),
        "project_id": package.project_id,
        "work_package_id": package.work_package_id,
        "stage": package.stage,
        "recipe_id": resolved.recipe.id,
        "resolved_seats": sorted(resolved.seats),
        "jobs": [
            {
                "id": job.id,
                "kind": job.kind,
                "definition": job.definition,
                "optional": job.optional,
                "enabled": not job.optional or job.id in package.enabled_optional_jobs,
            }
            for job in resolved.recipe.jobs
        ],
        "validation_level": validation_level,
        "composition_validation": composition_validation,
    }
    emit(
        state,
        payload,
        (
            f"Work package composition valid: {package.work_package_id} "
            f"({len(resolved.seats)} resolved seats, {len(resolved.recipe.jobs)} jobs)\n"
            f"Static composition validation: {composition_validation}"
        ),
    )


def plan_work_package_cmd(
    state: CliState,
    path: Path,
    *,
    job: str,
    overrides: ExecutionOverrides = _EMPTY_OVERRIDES,
    run_id: str | None = None,
    host: str | None = None,
    entry: str | None = None,
    project_packages: tuple[str, ...] | None = None,
    source_includes: tuple[str, ...] | None = None,
) -> PlannedJobExecution:
    planned = plan_job_execution(
        state,
        path,
        job=job,
        overrides=overrides,
        run_id=run_id,
        host=host,
        entry=entry,
        project_packages=project_packages,
        source_includes=source_includes,
    )
    payload = _execution_plan_payload(planned)
    emit(
        state,
        payload,
        "\n".join(
            (
                f"Execution plan: {planned.launch.run_spec.run_id}",
                f"Provider: {planned.settings.provider}",
                f"Target: {planned.target.id}@{planned.target.revision}",
                f"Universal image: {planned.package.pack_plan.spec.universal_image.value}",
                f"Job-kind image: {planned.package.pack_plan.spec.kind_image.value}",
                f"Runtime variant: {planned.package.pack_plan.spec.runtime_variant}",
                f"Pack plan: {planned.package.pack_plan.plan_key}",
            )
        ),
    )
    return planned


def pack_work_package_cmd(
    state: CliState,
    path: Path,
    *,
    job: str,
    overrides: PackageOverrides = _EMPTY_PACKAGE_OVERRIDES,
    host: str | None = None,
    entry: str | None = None,
    project_packages: tuple[str, ...] | None = None,
    source_includes: tuple[str, ...] | None = None,
    build_missing: bool = False,
) -> PackedJobPackage:
    """Pack and publish one job without submitting it to a provider."""

    planned = plan_job_package(
        state,
        path,
        job=job,
        overrides=overrides,
        host=host,
        entry=entry,
        project_packages=project_packages,
        source_includes=source_includes,
    )
    _require_verified_kind_image(planned, build_missing=build_missing)
    packed = planned.pack()
    emit(
        state,
        _packed_job_payload(packed),
        "\n".join(
            (
                f"Job image ready: {packed.image.image.value}",
                f"Package: {packed.context.manifest.package_key}",
                f"Job kind parent: {packed.context.manifest.kind_image.value}",
                f"Publication cache: {'hit' if packed.image.cache_hit else 'built'}",
                f"Receipt: {packed.image.receipt}",
            )
        ),
    )
    return packed


def run_work_package_cmd(
    state: CliState,
    path: Path,
    *,
    job: str,
    host: str | None = None,
    entry: str | None = None,
    in_process: bool = False,
    overrides: ExecutionOverrides = _EMPTY_OVERRIDES,
    run_id: str | None = None,
    project_packages: tuple[str, ...] | None = None,
    source_includes: tuple[str, ...] | None = None,
    build_missing: bool = False,
) -> None:
    if not in_process:
        planned = plan_job_execution(
            state,
            path,
            job=job,
            overrides=overrides,
            run_id=run_id,
            host=host,
            entry=entry,
            project_packages=project_packages,
            source_includes=source_includes,
        )
        _require_verified_kind_image(planned, build_missing=build_missing)
        packed = planned.pack()
        prepared_submission = packed.prepare_submission()
        admission = execution_admission_service(planned.package.layout)
        admitted = admission.enqueue(
            prepared_submission.provider_plan,
            evidence_source=evidence_source_for_project(planned.package.layout),
            initial_service=prepared_submission.service,
        )
        submission = admitted.submission
        admission_entry = admitted.entry
        payload = {
            **_packed_job_payload(packed),
            "status": admission_entry.state,
            "queue_position": admission_entry.position,
            "submission": (
                json_value(submission) if submission is not None else None
            ),
        }
        provider_detail = (
            f"Provider run: {submission.provider_id}"
            if submission is not None
            else f"Queue position: {admission_entry.position}"
        )
        emit(
            state,
            payload,
            "\n".join(
                (
                    f"Execution {admission_entry.state}: {admission_entry.run_id}",
                    f"Image: {packed.image.image.value}",
                    f"Provider: {prepared_submission.provider_plan.provider}",
                    provider_detail,
                    f"Status: posttrain run status {admission_entry.run_id}",
                )
            ),
        )
        return

    layout, catalog, resolved_path, package = load_work_package_bundle(state, path)
    output_redirect = redirect_stdout(sys.stderr) if state.json_output else nullcontext()
    with output_redirect:
        context = runtime_context(
            layout=layout,
            catalog=catalog,
            path=resolved_path,
            host=host,
            entry=entry,
        )
        result = run_work_package_job(context, package, job)
    payload = {
        "path": str(resolved_path),
        "entry": entry or layout.entry,
        "host": host,
        "project_id": result.project_id,
        "work_package_id": result.work_package_id,
        "selected_job": job,
        "status": "succeeded",
        "jobs": [
            {
                "id": job_result.job_id,
                "kind": job_result.kind,
                "definition": job_result.definition,
                "status": job_result.status,
                "run_id": job_result.run_id,
                "value": json_value(job_result.value),
            }
            for job_result in result.jobs
        ],
    }
    lines = [f"Work package succeeded: {result.work_package_id}"]
    lines.extend(
        f"{job_result.status.upper():9} {job_result.job_id} [{job_result.kind}]"
        + (f" run={job_result.run_id}" if job_result.run_id is not None else "")
        for job_result in result.jobs
    )
    emit(state, payload, "\n".join(lines))



def _require_verified_kind_image(
    planned: PlannedJobPackage | PlannedJobExecution,
    *,
    build_missing: bool,
) -> None:
    """Confirm the job-kind image is this release's before anything is packed.

    This runs before packing and before any provider object exists, so a
    drifted image costs nothing but a clear error.
    """
    package = planned.package if isinstance(planned, PlannedJobExecution) else planned
    registry = package.local_config.registry
    if registry is None:
        raise RuntimeError("planned job is missing its registry configuration")
    ensure_kind_image_ready(
        registry,
        package.pack_plan.spec.runtime_variant,
        build_missing=build_missing,
    )


def _package_plan_payload(planned: PlannedJobPackage) -> dict[str, object]:
    registry = planned.local_config.registry
    if registry is None:
        raise RuntimeError("planned job is missing its registry configuration")
    runtime_variant = planned.pack_plan.spec.runtime_variant
    constraint = registry.constraint_profiles[runtime_variant]
    return {
        "project_id": planned.prepared.spec.project_id,
        "work_package_id": planned.prepared.spec.work_package_id,
        "job_id": planned.prepared.recipe_job.id,
        "job_kind": planned.prepared.recipe_job.kind,
        "job_definition_id": planned.prepared.definition.id,
        "runtime_profile": f"framework/{runtime_variant}@1",
        "images": {
            "universal": planned.pack_plan.spec.universal_image.value,
            "job_kind": planned.pack_plan.spec.kind_image.value,
            "actual_job": None,
        },
        "pack": {
            "plan_key": planned.pack_plan.plan_key,
            "publication_plan_key": planned.pack_plan.publication_plan_key,
            "framework_source_digest": planned.pack_plan.spec.framework_source_digest,
            "project_source_digest": planned.pack_plan.spec.project_source_digest,
            "kind_profile": planned.pack_plan.spec.kind_profile,
            "runtime_variant": planned.pack_plan.spec.runtime_variant,
            "constraint_profile_digest": constraint.digest,
            "provided_packages": list(constraint.provided_packages),
            "publication_repository": planned.pack_plan.publication.repository,
            "datasets": [
                request.to_payload() for request in planned.pack_plan.spec.datasets
            ],
            "environment_sources": [
                {
                    "repository": source.repository,
                    "revision": source.revision,
                    "subdirectories": source.subdirectories,
                }
                for source in planned.pack_plan.spec.git_sources
            ],
        },
    }


def _execution_plan_payload(planned: PlannedJobExecution) -> dict[str, object]:
    payload = _package_plan_payload(planned.package)
    settings = planned.settings
    payload.update(
        {
            "run_id": planned.launch.run_spec.run_id,
            "provider": settings.provider,
            "target": {
                "id": planned.target.id,
                "revision": planned.target.revision,
                "device_class": planned.target.device_class,
                "memory_gb": planned.target.memory_gb,
            },
            "runtime_profile": settings.runtime_profile,
            "policy": {
                "timeout_seconds": settings.timeout_seconds,
                "max_attempts": settings.max_attempts,
                "priority": settings.priority,
            },
            "environment_names": settings.environment_names,
            "setting_sources": settings.sources,
            "mounts": [
                {
                    "purpose": mount.purpose,
                    "instance_path": str(mount.instance_path),
                    "container_path": str(mount.container_path),
                    "optional": mount.optional,
                }
                for mount in planned.mounts
            ],
        }
    )
    return payload


def _packed_job_payload(
    packed: PackedJobExecution | PackedJobPackage,
) -> dict[str, object]:
    planned = packed.planned
    payload = (
        _execution_plan_payload(planned)
        if isinstance(planned, PlannedJobExecution)
        else _package_plan_payload(planned)
    )
    payload["images"] = {
        "universal": packed.context.manifest.universal_image.value,
        "job_kind": packed.context.manifest.kind_image.value,
        "actual_job": packed.image.image.value,
    }
    payload["package"] = {
        "package_key": packed.context.manifest.package_key,
        "context_digest": packed.context.context_digest,
        "publication_key": packed.image.publication_key,
        "cache_hit": packed.image.cache_hit,
        "receipt": str(packed.image.receipt),
        "context": str(packed.context.root),
    }
    return payload


def _overrides(
    *,
    provider: str | None,
    target: str | None,
    runtime_profile: str | None,
    timeout_seconds: int | None,
    max_attempts: int | None,
    priority: int | None,
    environment_names: list[str] | None,
) -> ExecutionOverrides:
    return ExecutionOverrides(
        provider=provider,
        target=target,
        runtime_profile=runtime_profile,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        priority=priority,
        environment_names=(
            tuple(environment_names)
            if environment_names is not None
            else None
        ),
    )


def register(app: typer.Typer) -> None:
    work_package_app = typer.Typer(
        rich_markup_mode=None, no_args_is_help=True, help="inspect and execute work packages"
    )
    app.add_typer(work_package_app, name="work-package")

    @work_package_app.command("validate", help="validate YAML, recipe structure, and catalog bindings")
    def work_package_validate_cmd(
        ctx: typer.Context,
        path: Annotated[Path, typer.Argument()],
        host: Annotated[
            str | None,
            typer.Option(
                "--host",
                metavar="MODULE:FACTORY",
                help="also statically validate concrete job definitions through this explicit project host",
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
    ) -> None:
        state: CliState = ctx.obj
        validate_work_package_cmd(state, path, host=host, entry=entry)

    @work_package_app.command(
        "plan",
        help="resolve one job into a read-only local or dstack execution plan",
    )
    def work_package_plan_cmd(
        ctx: typer.Context,
        path: Annotated[Path, typer.Argument()],
        job: Annotated[str, typer.Option("--job", help="plan exactly this enabled job id")],
        provider: Annotated[str | None, typer.Option("--provider")] = None,
        target: Annotated[str | None, typer.Option("--target")] = None,
        runtime_profile: Annotated[
            str | None,
            typer.Option("--runtime-profile"),
        ] = None,
        timeout_seconds: Annotated[
            int | None,
            typer.Option("--timeout-seconds", min=1),
        ] = None,
        max_attempts: Annotated[
            int | None,
            typer.Option("--max-attempts", min=1),
        ] = None,
        priority: Annotated[int | None, typer.Option("--priority")] = None,
        environment_names: Annotated[
            list[str] | None,
            typer.Option(
                "--env",
                help="require and forward this named environment variable; repeatable",
            ),
        ] = None,
        run_id: Annotated[str | None, typer.Option("--run-id")] = None,
        host: Annotated[
            str | None,
            typer.Option(
                "--host",
                metavar="MODULE:FACTORY",
                help="deprecated compatibility host used only during static job validation",
            ),
        ] = None,
        entry: Annotated[
            str | None,
            typer.Option(
                "--entry",
                metavar="MODULE:FACTORY",
                help="override the optional project entry during static validation",
            ),
        ] = None,
    ) -> None:
        state: CliState = ctx.obj
        plan_work_package_cmd(
            state,
            path,
            job=job,
            overrides=_overrides(
                provider=provider,
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
        )

    @work_package_app.command(
        "run",
        help="submit one work-package job through the configured execution provider",
    )
    def work_package_run_cmd(
        ctx: typer.Context,
        path: Annotated[Path, typer.Argument()],
        job: Annotated[
            str,
            typer.Option("--job", help="execute exactly this enabled recipe job id"),
        ],
        host: Annotated[
            str | None,
            typer.Option(
                "--host",
                metavar="MODULE:FACTORY",
                help="deprecated compatibility alias for an explicit legacy host",
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
        provider: Annotated[str | None, typer.Option("--provider")] = None,
        target: Annotated[str | None, typer.Option("--target")] = None,
        runtime_profile: Annotated[
            str | None,
            typer.Option("--runtime-profile"),
        ] = None,
        timeout_seconds: Annotated[
            int | None,
            typer.Option("--timeout-seconds", min=1),
        ] = None,
        max_attempts: Annotated[
            int | None,
            typer.Option("--max-attempts", min=1),
        ] = None,
        priority: Annotated[int | None, typer.Option("--priority")] = None,
        environment_names: Annotated[
            list[str] | None,
            typer.Option(
                "--env",
                help="require and forward this named environment variable; repeatable",
            ),
        ] = None,
        run_id: Annotated[str | None, typer.Option("--run-id")] = None,
        in_process: Annotated[
            bool,
            typer.Option(
                "--in-process",
                help="temporary compatibility mode; execute in the CLI process",
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
                provider=provider,
                target=target,
                runtime_profile=runtime_profile,
                timeout_seconds=timeout_seconds,
                max_attempts=max_attempts,
                priority=priority,
                environment_names=environment_names,
            ),
            run_id=run_id,
        )
