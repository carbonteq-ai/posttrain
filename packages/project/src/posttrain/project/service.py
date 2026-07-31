"""Project application service shared by the CLI and Python callers."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from posttrain.catalog import ProjectLayout, discover_project, load_project_layout, open_catalog
from posttrain.common import Catalog, ContractError
from posttrain.jobs import build_job_runtime, standard_definitions
from posttrain.work import (
    JobRuntime,
    PreparedWorkPackageJob,
    ProjectEntry,
    ProjectExecutionRequest,
    WorkPackage,
    WorkPackageContext,
    WorkPackageHostFactory,
    WorkPackageHostRequest,
    load_project_brief,
    load_work_package,
    prepare_work_package_job,
    resolve_work_package,
)


@dataclass(frozen=True, slots=True)
class JobIntent:
    """One resolved, statically validated job before provider or image selection.

    An intent binds a project layout, composed catalog, selected work-package
    source, and the standard runtime definitions that validated the job. It is
    deliberately provider-free: callers may inspect it without a registry,
    credentials, or a running worker.
    """

    layout: ProjectLayout
    catalog: Catalog
    work_package_path: Path
    work_package: WorkPackage
    job_id: str
    context: WorkPackageContext
    prepared: PreparedWorkPackageJob

    def __post_init__(self) -> None:
        if self.catalog.scope != self.layout.project_id:
            raise ContractError("job intent catalog scope must match the project id")
        if self.work_package.project_id != self.layout.project_id:
            raise ContractError("job intent work package must match the project id")
        if not self.work_package_path.is_absolute():
            raise ContractError("job intent work-package path must be absolute")
        if self.prepared.recipe_job.id != self.job_id:
            raise ContractError("job intent id must match its prepared recipe job")


@dataclass(frozen=True, slots=True)
class Project:
    """A project root with its layout and composed, read-only catalog."""

    layout: ProjectLayout
    catalog: Catalog

    def __post_init__(self) -> None:
        if self.catalog.scope != self.layout.project_id:
            raise ContractError("project catalog scope must match the project id")

    @classmethod
    def open(cls, root: str | Path) -> Project:
        """Open the project rooted at ``root`` without consulting a shell setting."""

        return cls._from_layout(load_project_layout(Path(root)))

    @classmethod
    def discover(cls, start: str | Path) -> Project:
        """Discover the nearest project above ``start`` for command-line convenience."""

        return cls._from_layout(discover_project(Path(start)))

    @classmethod
    def _from_layout(cls, layout: ProjectLayout) -> Project:
        return cls(
            layout=layout,
            catalog=open_catalog(
                scope=layout.project_id,
                overlays=layout.catalog_overlays,
                catalog_root=layout.base_catalog,
            ),
        )

    @property
    def jobs(self) -> JobService:
        """Return provider-free job operations for this project."""

        return JobService(self)


@dataclass(frozen=True, slots=True)
class JobService:
    """Read-only job composition operations for one opened project."""

    project: Project

    def plan(
        self,
        work_package: str | Path,
        *,
        job: str | None = None,
        entry: str | None = None,
        host: str | None = None,
    ) -> JobIntent:
        """Resolve and statically validate one enabled job without side effects."""

        path = _work_package_path(self.project.layout, Path(work_package))
        package = load_work_package(path)
        if package.project_id != self.project.layout.project_id:
            raise ContractError(
                f"work package project {package.project_id!r} does not match project manifest "
                f"{self.project.layout.project_id!r}"
            )
        job_id = _resolve_job_id(self.project.catalog, package, job)
        context = _runtime_context(
            layout=self.project.layout,
            catalog=self.project.catalog,
            path=path,
            host=host,
            entry=entry,
        )
        prepared = prepare_work_package_job(context, package, job_id)
        return JobIntent(
            layout=self.project.layout,
            catalog=self.project.catalog,
            work_package_path=path,
            work_package=package,
            job_id=job_id,
            context=context,
            prepared=prepared,
        )


def _work_package_path(layout: ProjectLayout, configured: Path) -> Path:
    candidate = configured if configured.is_absolute() else Path.cwd() / configured
    if not candidate.is_file() and not configured.is_absolute():
        candidate = layout.work_packages / configured
    resolved = candidate.resolve()
    if not resolved.is_relative_to(layout.work_packages):
        raise ContractError(f"work-package path must remain under {layout.work_packages}: {configured}")
    return resolved


def _resolve_job_id(catalog: Catalog, package: WorkPackage, requested: str | None) -> str:
    resolved = resolve_work_package(catalog, package)
    enabled = tuple(item.id for item in resolved.recipe.jobs if not item.optional or item.id in package.enabled_optional_jobs)
    if requested is not None:
        if requested not in enabled:
            available = ", ".join(enabled) if enabled else "(none)"
            raise ContractError(f"job {requested!r} is not enabled; available: {available}")
        return requested
    if len(enabled) == 1:
        return enabled[0]
    if not enabled:
        raise ContractError("work package has no enabled jobs")
    raise ContractError(f"pass job=; work package has {len(enabled)} enabled jobs: {', '.join(enabled)}")


def _runtime_context(
    *,
    layout: ProjectLayout,
    catalog: Catalog,
    path: Path,
    host: str | None,
    entry: str | None,
) -> WorkPackageContext:
    if host is not None:
        context = _host_context(host, layout=layout, catalog=catalog, path=path)
        return replace(context, seat_resolver=None)
    request = _execution_request(layout=layout, catalog=catalog, path=path)
    entry_spec = entry or layout.entry
    if entry_spec is None:
        return replace(build_job_runtime(request, tracking=layout.tracking), seat_resolver=None)
    runtime = _load_project_entry(entry_spec, project_root=layout.root)(request)
    if not isinstance(runtime, JobRuntime):
        raise ContractError(f"project entry {entry_spec!r} must return JobRuntime")
    if runtime.catalog is not catalog:
        raise ContractError("project entry must use the catalog supplied in ProjectExecutionRequest")
    _validate_standard_definitions(runtime)
    return replace(runtime, seat_resolver=None)


def _execution_request(
    *,
    layout: ProjectLayout,
    catalog: Catalog,
    path: Path,
) -> ProjectExecutionRequest:
    return ProjectExecutionRequest(
        project_id=layout.project_id,
        project_root=layout.root,
        state_dir=layout.state,
        work_package_path=path,
        catalog=catalog,
        project_brief=(load_project_brief(layout.project_brief) if layout.project_brief is not None else None),
    )


def _host_context(
    spec: str,
    *,
    layout: ProjectLayout,
    catalog: Catalog,
    path: Path,
) -> WorkPackageContext:
    factory = _load_host_factory(spec, project_root=layout.root)
    request = WorkPackageHostRequest(
        project_id=layout.project_id,
        project_root=layout.root,
        state_dir=layout.state,
        work_package_path=path,
        catalog=catalog,
        project_brief=(load_project_brief(layout.project_brief) if layout.project_brief is not None else None),
    )
    context = factory(request)
    if not isinstance(context, WorkPackageContext):
        raise ContractError(f"work-package host {spec!r} must return WorkPackageContext")
    if context.catalog is not catalog:
        raise ContractError("work-package host must use the catalog supplied in WorkPackageHostRequest")
    if context.project_brief is None and request.project_brief is not None:
        return replace(context, project_brief=request.project_brief)
    if context.project_brief != request.project_brief:
        raise ContractError("work-package host project brief conflicts with the discovered project")
    return context


def _load_project_entry(spec: str, *, project_root: Path) -> ProjectEntry:
    return cast(ProjectEntry, _load_host_factory(spec, project_root=project_root))


def _load_host_factory(spec: str, *, project_root: Path) -> WorkPackageHostFactory:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute or ":" in attribute:
        raise ContractError("host must use MODULE:FACTORY syntax")
    sys.path.insert(0, str(project_root))
    try:
        try:
            module = importlib.import_module(module_name)
        except (ImportError, ValueError) as error:
            raise ContractError(f"cannot import work-package host module {module_name!r}: {error}") from error
    finally:
        sys.path.remove(str(project_root))
    try:
        factory: Any = getattr(module, attribute)
    except AttributeError as error:
        raise ContractError(f"work-package host module {module_name!r} has no factory {attribute!r}") from error
    if not callable(factory):
        raise ContractError(f"work-package host {spec!r} is not callable")
    return cast(WorkPackageHostFactory, factory)


def _validate_standard_definitions(runtime: JobRuntime) -> None:
    for definition_id, standard in standard_definitions().items():
        configured = runtime.definitions.get(definition_id)
        if configured is None:
            raise ContractError(f"project entry omitted standard job definition: {definition_id}")
        same_operation = getattr(configured.operation, "__code__", None) is getattr(standard.operation, "__code__", None)
        same_static_validator = getattr(configured.static_validator, "__code__", None) is getattr(
            standard.static_validator, "__code__", None
        )
        if (
            configured.kind != standard.kind
            or configured.seats != standard.seats
            or configured.selection_seats != standard.selection_seats
            or not same_operation
            or not same_static_validator
        ):
            raise ContractError(f"project entry cannot redefine standard job definition: {definition_id}")
