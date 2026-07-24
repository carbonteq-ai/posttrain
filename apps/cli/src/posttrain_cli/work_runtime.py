"""Work-package host loading and runtime construction."""

from __future__ import annotations

import importlib
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from posttrain.catalog import ProjectLayout
from posttrain.common import ContractError
from posttrain.jobs import build_job_runtime, standard_definitions
from posttrain.work import (
    JobRuntime,
    ProjectEntry,
    ProjectExecutionRequest,
    WorkPackage,
    WorkPackageContext,
    WorkPackageHostFactory,
    WorkPackageHostRequest,
    load_project_brief,
    load_work_package,
)

from .context import CliState
from .project import work_package_path


def load_work_package_bundle(
    state: CliState,
    path: Path,
) -> tuple[ProjectLayout, Any, Path, WorkPackage]:
    layout, catalog = state.open_catalog()
    resolved_path = work_package_path(layout, path)
    package = load_work_package(resolved_path)
    if package.project_id != layout.project_id:
        raise ContractError(
            f"work package project {package.project_id!r} does not match project manifest {layout.project_id!r}"
        )
    return layout, catalog, resolved_path, package


def load_host_factory(spec: str, *, project_root: Path) -> WorkPackageHostFactory:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute or ":" in attribute:
        raise ContractError("host must use MODULE:FACTORY syntax")
    search_root = str(project_root)
    sys.path.insert(0, search_root)
    try:
        try:
            module = importlib.import_module(module_name)
        except (ImportError, ValueError) as error:
            raise ContractError(f"cannot import work-package host module {module_name!r}: {error}") from error
    finally:
        sys.path.remove(search_root)
    try:
        factory = getattr(module, attribute)
    except AttributeError as error:
        raise ContractError(f"work-package host module {module_name!r} has no factory {attribute!r}") from error
    if not callable(factory):
        raise ContractError(f"work-package host {spec!r} is not callable")
    return cast(WorkPackageHostFactory, factory)


def load_project_entry(spec: str, *, project_root: Path) -> ProjectEntry:
    return cast(ProjectEntry, load_host_factory(spec, project_root=project_root))


def host_context(
    spec: str,
    *,
    layout: ProjectLayout,
    catalog: Any,
    path: Path,
) -> WorkPackageContext:
    factory = load_host_factory(spec, project_root=layout.root)
    request = WorkPackageHostRequest(
        project_id=layout.project_id,
        project_root=layout.root,
        state_dir=layout.state,
        work_package_path=path,
        catalog=catalog,
        project_brief=(
            load_project_brief(layout.project_brief)
            if layout.project_brief is not None
            else None
        ),
    )
    context = factory(request)
    if not isinstance(context, WorkPackageContext):
        raise ContractError(f"work-package host {spec!r} must return WorkPackageContext")
    if context.catalog is not catalog:
        raise ContractError("work-package host must use the catalog supplied in WorkPackageHostRequest")
    if context.project_brief is None and request.project_brief is not None:
        context = replace(context, project_brief=request.project_brief)
    elif context.project_brief != request.project_brief:
        raise ContractError("work-package host project brief conflicts with the discovered project")
    return context


def execution_request(
    *,
    layout: ProjectLayout,
    catalog: Any,
    path: Path,
) -> ProjectExecutionRequest:
    return ProjectExecutionRequest(
        project_id=layout.project_id,
        project_root=layout.root,
        state_dir=layout.state,
        work_package_path=path,
        catalog=catalog,
        project_brief=(
            load_project_brief(layout.project_brief)
            if layout.project_brief is not None
            else None
        ),
    )


def runtime_context(
    *,
    layout: ProjectLayout,
    catalog: Any,
    path: Path,
    host: str | None,
    entry: str | None,
) -> JobRuntime:
    if host is not None:
        return host_context(host, layout=layout, catalog=catalog, path=path)
    request = execution_request(layout=layout, catalog=catalog, path=path)
    entry_spec = entry or layout.entry
    if entry_spec is None:
        return build_job_runtime(request, tracking=layout.tracking)
    runtime = load_project_entry(entry_spec, project_root=layout.root)(request)
    if not isinstance(runtime, JobRuntime):
        raise ContractError(f"project entry {entry_spec!r} must return JobRuntime")
    if runtime.catalog is not catalog:
        raise ContractError("project entry must use the catalog supplied in ProjectExecutionRequest")
    validate_standard_definitions(runtime)
    return runtime


def validate_standard_definitions(runtime: JobRuntime) -> None:
    for definition_id, standard in standard_definitions().items():
        configured = runtime.definitions.get(definition_id)
        if configured is None:
            raise ContractError(f"project entry omitted standard job definition: {definition_id}")
        same_operation = getattr(configured.operation, "__code__", None) is getattr(
            standard.operation, "__code__", None
        )
        if configured.kind != standard.kind or configured.seats != standard.seats or not same_operation:
            raise ContractError(f"project entry cannot redefine standard job definition: {definition_id}")
