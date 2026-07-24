"""Dataset and environment materialization for work packages."""

from __future__ import annotations

from pathlib import Path

from posttrain.catalog import ProjectLayout
from posttrain.data import DatasetLoadPlan, materialize_dataset
from posttrain.eval import EnvironmentBinding
from posttrain.train.integrations import preflight_verifiers_environment
from posttrain.work import load_work_package, resolve_work_package

from .context import CliState
from .project import work_package_path


def work_package_paths(layout: ProjectLayout, configured: Path | None) -> list[Path]:
    if configured is not None:
        return [work_package_path(layout, configured)]
    if not layout.work_packages.is_dir():
        return []
    return sorted(path for path in layout.work_packages.glob("*.yaml") if path.is_file())


def materialize_project_references(
    state: CliState,
    *,
    work_package: Path | None,
) -> list[dict[str, object]]:
    layout, catalog = state.open_catalog()
    results: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for path in work_package_paths(layout, work_package):
        package = load_work_package(path)
        resolved = resolve_work_package(catalog, package)
        for seat_name, seat in resolved.seats.items():
            selection = seat.value
            if isinstance(selection, DatasetLoadPlan):
                key = ("dataset", selection.id)
                if key in seen:
                    continue
                seen.add(key)
                materialized = materialize_dataset(
                    selection,
                    state_dir=layout.state,
                    project_root=layout.root,
                )
                results.append(
                    {
                        "family": "dataset",
                        "id": materialized.selection_id,
                        "status": "materialized" if materialized.created else "cached",
                        "path": str(materialized.path),
                        "examples": materialized.examples,
                        "seat": seat_name,
                        "work_package": str(path),
                    }
                )
            elif isinstance(selection, EnvironmentBinding):
                key = ("environment", selection.id)
                if key in seen:
                    continue
                seen.add(key)
                preflight_verifiers_environment(selection)
                results.append(
                    {
                        "family": "environment",
                        "id": selection.id,
                        "status": "preflighted",
                        "package": selection.source.package,
                        "seat": seat_name,
                        "work_package": str(path),
                    }
                )
    return results
