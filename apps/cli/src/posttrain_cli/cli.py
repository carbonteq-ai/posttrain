"""Stable command-line surface for portable post-training projects."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from posttrain.catalog import ProjectLayout, discover_project, open_catalog
from posttrain.common import CatalogRef, ContractError
from posttrain.common.selections import SelectionFamily, validate_selection_id
from posttrain.work import load_work_package, resolve_work_package

_DISTRIBUTION = "posttrain"
_CATALOG_FAMILIES: tuple[SelectionFamily, ...] = (
    "model",
    "dataset",
    "environment",
    "inference",
    "training",
    "quantization",
    "evaluation",
    "workload",
    "target",
    "recipe",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="posttrain",
        description="Initialize, inspect, validate, and run post-training projects.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="project root containing .posttrain/project.toml; otherwise discover upward",
    )
    parser.add_argument("--json", dest="json_output", action="store_true", help="emit JSON output")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("version", help="show the installed framework CLI version")

    init = commands.add_parser("init", help="initialize a portable post-training project")
    init.add_argument("path", nargs="?", type=Path, default=Path.cwd())
    init.add_argument("--project-id", help="lowercase stable project identity; defaults from PATH")

    commands.add_parser("doctor", help="check project and catalog readiness")

    project = commands.add_parser("project", help="inspect project configuration")
    project_commands = project.add_subparsers(dest="project_command", required=True)
    project_commands.add_parser("show", help="show the discovered project layout")

    catalog = commands.add_parser("catalog", help="inspect the composed project catalog")
    catalog_commands = catalog.add_subparsers(dest="catalog_command", required=True)
    catalog_list = catalog_commands.add_parser("list", help="list resolved catalog selections")
    catalog_list.add_argument("--family", choices=_CATALOG_FAMILIES)
    catalog_show = catalog_commands.add_parser("show", help="show one resolved catalog selection")
    catalog_show.add_argument("family", choices=_CATALOG_FAMILIES)
    catalog_show.add_argument("id")
    catalog_commands.add_parser("validate", help="load and validate the complete catalog")

    work_package = commands.add_parser("work-package", help="inspect and execute work packages")
    work_package_commands = work_package.add_subparsers(
        dest="work_package_command",
        required=True,
    )
    work_package_validate = work_package_commands.add_parser(
        "validate",
        help="validate YAML, recipe structure, and catalog bindings",
    )
    work_package_validate.add_argument("path", type=Path)
    return parser


def _installed_version() -> str:
    try:
        return version(_DISTRIBUTION)
    except PackageNotFoundError:
        return "0+unknown"


def _default_project_id(root: Path) -> str:
    normalized = re.sub(r"[^a-z0-9._/@:-]+", "-", root.name.lower()).strip("-._")
    if not normalized:
        raise ContractError("cannot derive project id from path; pass --project-id")
    return validate_selection_id(normalized, "project id")


def _initialize(root: Path, project_id: str | None) -> ProjectLayout:
    project_root = root.resolve()
    resolved_id = validate_selection_id(
        project_id if project_id is not None else _default_project_id(project_root),
        "project id",
    )
    control = project_root / ".posttrain"
    manifest = control / "project.toml"
    catalog = control / "catalog"
    catalog_manifest = catalog / "layer.yaml"
    work_packages = control / "work_packages"
    work_packages_readme = work_packages / "README.md"
    control_ignore = control / ".gitignore"

    conflicts = [path for path in (manifest, catalog_manifest, work_packages_readme, control_ignore) if path.exists()]
    if conflicts:
        names = ", ".join(str(path) for path in conflicts)
        raise FileExistsError(f"refusing to overwrite existing project files: {names}")

    project_root.mkdir(parents=True, exist_ok=True)
    catalog.mkdir(parents=True, exist_ok=True)
    work_packages.mkdir(parents=True, exist_ok=True)
    (control / "state").mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "\n".join(
            (
                "schema_version = 1",
                f'project_id = "{resolved_id}"',
                'catalog_overlays = ["catalog"]',
                'work_packages = "work_packages"',
                'state = "state"',
                "",
            )
        ),
        encoding="utf-8",
    )
    catalog_manifest.write_text(
        "\n".join(
            (
                "schema_version: 1",
                f"layer_id: {resolved_id}-v1",
                "files: []",
                "",
            )
        ),
        encoding="utf-8",
    )
    work_packages_readme.write_text(
        "# Work packages\n\nAdd versioned `screen`, `train`, and `qualify` work-package YAML files here.\n",
        encoding="utf-8",
    )
    control_ignore.write_text("state/\n", encoding="utf-8")
    return discover_project(project_root, explicit_root=project_root)


def _layout(args: argparse.Namespace) -> ProjectLayout:
    return discover_project(Path.cwd(), explicit_root=args.project_root)


def _catalog(args: argparse.Namespace) -> tuple[ProjectLayout, Any]:
    layout = _layout(args)
    return (
        layout,
        open_catalog(
            scope=layout.project_id,
            overlays=layout.catalog_overlays,
            catalog_root=layout.base_catalog,
        ),
    )


def _work_package_path(layout: ProjectLayout, configured: Path) -> Path:
    candidate = configured if configured.is_absolute() else Path.cwd() / configured
    if not candidate.is_file() and not configured.is_absolute():
        candidate = layout.work_packages / configured
    resolved = candidate.resolve()
    if not resolved.is_relative_to(layout.work_packages):
        raise ContractError(f"work-package path must remain under {layout.work_packages}: {configured}")
    return resolved


def _layout_payload(layout: ProjectLayout) -> dict[str, object]:
    return {
        "project_id": layout.project_id,
        "root": str(layout.root),
        "manifest": str(layout.manifest),
        "catalog_overlays": [str(path) for path in layout.catalog_overlays],
        "work_packages": str(layout.work_packages),
        "state": str(layout.state),
    }


def _catalog_entries(catalog: Any, family: SelectionFamily | None = None) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for ref in catalog.list(family):
        resolved = catalog.resolve(ref)
        entries.append(
            {
                "family": ref.family,
                "id": ref.id,
                "source_layer": resolved.source_layer,
                "overlay_id": resolved.overlay_id,
            }
        )
    return entries


def _catalog_summary(catalog: Any) -> dict[str, object]:
    entries = _catalog_entries(catalog)
    overlay_entries = sum(entry["source_layer"] == "overlay" for entry in entries)
    return {
        "base_catalog_release": catalog.base_id,
        "overlay_ids": list(catalog.overlay_ids),
        "entries": len(entries),
        "base_entries": len(entries) - overlay_entries,
        "project_entries": overlay_entries,
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _emit(args: argparse.Namespace, payload: object, human: str) -> None:
    if args.json_output:
        print(json.dumps(_json_value(payload), indent=2, sort_keys=True))
    else:
        print(human)


def _doctor(args: argparse.Namespace) -> int:
    checks: list[dict[str, str]] = [
        {
            "name": "python",
            "status": "ok" if sys.version_info[:2] == (3, 12) else "error",
            "message": f"Python {sys.version_info.major}.{sys.version_info.minor}",
        }
    ]
    layout: ProjectLayout | None = None
    try:
        layout = _layout(args)
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
            summary = _catalog_summary(catalog)
            checks.append(
                {
                    "name": "catalog",
                    "status": "ok",
                    "message": (f"{summary['base_catalog_release']}, {summary['entries']} resolved selections"),
                }
            )
        except (ContractError, KeyError, OSError) as error:
            checks.append({"name": "catalog", "status": "error", "message": _error_message(error)})
        checks.append(
            {
                "name": "work_packages",
                "status": "ok" if layout.work_packages.is_dir() else "error",
                "message": str(layout.work_packages),
            }
        )

    succeeded = all(check["status"] == "ok" for check in checks)
    if args.json_output:
        print(json.dumps({"ok": succeeded, "checks": checks}, indent=2, sort_keys=True))
    else:
        for check in checks:
            print(f"{check['status'].upper():5} {check['name']}: {check['message']}")
    return 0 if succeeded else 1


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "version":
        installed = _installed_version()
        _emit(args, {"version": installed}, f"posttrain {installed}")
        return 0
    if args.command == "init":
        layout = _initialize(args.path, args.project_id)
        _emit(
            args,
            _layout_payload(layout),
            f"Initialized post-training project {layout.project_id} at {layout.root}",
        )
        return 0
    if args.command == "doctor":
        return _doctor(args)
    if args.command == "project" and args.project_command == "show":
        layout = _layout(args)
        _emit(
            args,
            _layout_payload(layout),
            "\n".join(
                (
                    f"Project: {layout.project_id}",
                    f"Root: {layout.root}",
                    f"Manifest: {layout.manifest}",
                    f"Catalog overlays: {', '.join(map(str, layout.catalog_overlays)) or '(none)'}",
                    f"Work packages: {layout.work_packages}",
                    f"State: {layout.state}",
                )
            ),
        )
        return 0
    if args.command == "catalog":
        _, catalog = _catalog(args)
        if args.catalog_command == "list":
            entries = _catalog_entries(catalog, args.family)
            _emit(
                args,
                entries,
                "\n".join(
                    f"{entry['family']}/{entry['id']} "
                    f"[{entry['source_layer']}"
                    f"{':' + str(entry['overlay_id']) if entry['overlay_id'] else ''}]"
                    for entry in entries
                )
                or "No catalog selections found.",
            )
            return 0
        if args.catalog_command == "show":
            resolved = catalog.resolve(CatalogRef(args.family, args.id))
            payload = {
                "family": resolved.ref.family,
                "id": resolved.ref.id,
                "source_layer": resolved.source_layer,
                "overlay_id": resolved.overlay_id,
                "selection": _json_value(resolved.value),
            }
            _emit(
                args,
                payload,
                json.dumps(payload, indent=2, sort_keys=True),
            )
            return 0
        if args.catalog_command == "validate":
            summary = _catalog_summary(catalog)
            _emit(
                args,
                summary,
                (
                    f"Catalog valid: {summary['base_catalog_release']}, "
                    f"{summary['base_entries']} base entries, "
                    f"{summary['project_entries']} project entries"
                ),
            )
            return 0
    if args.command == "work-package" and args.work_package_command == "validate":
        layout, catalog = _catalog(args)
        path = _work_package_path(layout, args.path)
        package = load_work_package(path)
        if package.project_id != layout.project_id:
            raise ContractError(
                f"work package project {package.project_id!r} does not match project manifest {layout.project_id!r}"
            )
        resolved = resolve_work_package(catalog, package)
        payload = {
            "path": str(path),
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
            "validation_level": "composition",
            "job_definition_preflight": "pending-host-definitions",
        }
        _emit(
            args,
            payload,
            (
                f"Work package composition valid: {package.work_package_id} "
                f"({len(resolved.seats)} resolved seats, {len(resolved.recipe.jobs)} jobs)\n"
                "Job-definition preflight: pending host definitions"
            ),
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


def _error_message(error: BaseException) -> str:
    if isinstance(error, KeyError) and error.args:
        return str(error.args[0])
    return str(error)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _dispatch(args)
    except (ContractError, FileExistsError, KeyError, OSError) as error:
        print(f"error: {_error_message(error)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
