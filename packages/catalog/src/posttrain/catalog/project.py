"""Portable project discovery and path ownership."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from posttrain.common import ContractError
from posttrain.common.selections import validate_selection_id
from pydantic import BaseModel, ConfigDict, ValidationError

_CONTROL_DIRECTORY = ".posttrain"
_MANIFEST = "project.toml"
_PROJECT_ROOT_ENV = "POSTTRAIN_PROJECT_ROOT"


class _ProjectManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    project_id: str
    catalog_overlays: tuple[str, ...] = ("catalog",)
    work_packages: str = "work_packages"
    state: str = "state"
    tracking: Literal["trackio", "wandb", "none"] = "trackio"
    entry: str | None = None
    project_brief: str | None = None


@dataclass(frozen=True, slots=True)
class ProjectLayout:
    """Resolved source and local-state paths for one post-training project."""

    project_id: str
    root: Path
    control_dir: Path
    manifest: Path
    catalog_overlays: tuple[Path, ...]
    work_packages: Path
    state: Path
    project_brief: Path | None = None
    tracking: Literal["trackio", "wandb", "none"] = "trackio"
    entry: str | None = None
    base_catalog: Path | None = None

    def __post_init__(self) -> None:
        validate_selection_id(self.project_id, "project id")
        for field_name in ("root", "control_dir", "manifest", "work_packages", "state"):
            if not getattr(self, field_name).is_absolute():
                raise ContractError(f"project layout {field_name} path must be absolute")
        if not all(path.is_absolute() for path in self.catalog_overlays):
            raise ContractError("project layout catalog overlay paths must be absolute")
        if self.base_catalog is not None and not self.base_catalog.is_absolute():
            raise ContractError("project layout base catalog path must be absolute")
        if self.project_brief is not None and not self.project_brief.is_absolute():
            raise ContractError("project layout project brief path must be absolute")
        if self.entry is not None:
            _validate_entry(self.entry)

    @classmethod
    def legacy(cls, repository: Path, project_id: str) -> ProjectLayout:
        """Represent the current root catalog/work-package layout explicitly."""

        root = repository.resolve()
        catalog_root = root / "catalog"
        overlay = catalog_root / "projects" / project_id.replace("/", "__")
        return cls(
            project_id=project_id,
            root=root,
            control_dir=root,
            manifest=root / "pyproject.toml",
            catalog_overlays=(overlay,) if overlay.is_dir() else (),
            work_packages=root / "work_packages",
            state=root / ".posttrain" / "state",
            tracking="trackio",
            base_catalog=catalog_root if (catalog_root / "base").is_dir() else None,
        )


def discover_project(
    start: Path,
    *,
    explicit_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> ProjectLayout:
    """Discover a project by explicit root, environment, then upward search."""

    if explicit_root is not None:
        return load_project_layout(explicit_root)

    values = os.environ if environ is None else environ
    if configured := values.get(_PROJECT_ROOT_ENV):
        return load_project_layout(Path(configured))

    candidate = start.resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / _CONTROL_DIRECTORY / _MANIFEST).is_file():
            return load_project_layout(directory)
    raise ContractError(
        "post-training project not found; pass an explicit project root, set "
        f"{_PROJECT_ROOT_ENV}, or create {_CONTROL_DIRECTORY}/{_MANIFEST}"
    )


def load_project_layout(root: Path) -> ProjectLayout:
    """Load and validate one `.posttrain/project.toml` project manifest."""

    project_root = root.resolve()
    control_dir = project_root / _CONTROL_DIRECTORY
    manifest_path = control_dir / _MANIFEST
    try:
        payload = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ContractError(f"post-training project manifest not found: {manifest_path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ContractError(f"invalid post-training project manifest {manifest_path}: {error}") from error

    try:
        manifest = _ProjectManifest.model_validate(payload)
    except ValidationError as error:
        raise ContractError(f"invalid post-training project manifest {manifest_path}: {error}") from error
    if manifest.schema_version not in {1, 2}:
        raise ContractError(
            f"unsupported post-training project schema_version {manifest.schema_version}; expected 1 or 2"
        )
    if manifest.schema_version == 1 and manifest.project_brief is not None:
        raise ContractError("post-training project_brief requires project schema_version 2")
    if manifest.schema_version == 2 and manifest.project_brief is None:
        raise ContractError("post-training project schema_version 2 requires project_brief")
    validate_selection_id(manifest.project_id, "project id")
    if len(set(manifest.catalog_overlays)) != len(manifest.catalog_overlays):
        raise ContractError("post-training project catalog overlays must be unique")

    overlays = tuple(
        _project_source_path(project_root, control_dir, value, "catalog overlay") for value in manifest.catalog_overlays
    )
    work_packages = _project_source_path(
        project_root,
        control_dir,
        manifest.work_packages,
        "work_packages",
    )
    state = _state_path(control_dir, manifest.state)
    project_brief = (
        _project_source_path(project_root, control_dir, manifest.project_brief, "project_brief")
        if manifest.project_brief is not None
        else None
    )
    if project_brief is not None and not project_brief.is_file():
        raise ContractError(f"post-training project brief not found: {project_brief}")
    return ProjectLayout(
        project_id=manifest.project_id,
        root=project_root,
        control_dir=control_dir,
        manifest=manifest_path,
        catalog_overlays=overlays,
        work_packages=work_packages,
        state=state,
        project_brief=project_brief,
        tracking=manifest.tracking,
        entry=_validate_entry(manifest.entry) if manifest.entry is not None else None,
    )


def _project_source_path(root: Path, control_dir: Path, value: str, field: str) -> Path:
    if not value.strip():
        raise ContractError(f"post-training project {field} path cannot be empty")
    configured = Path(value)
    if configured.is_absolute():
        raise ContractError(f"post-training project {field} path must be relative to .posttrain")
    resolved = (control_dir / configured).resolve()
    if not resolved.is_relative_to(root):
        raise ContractError(f"post-training project {field} path escapes the project root: {value}")
    return resolved


def _state_path(control_dir: Path, value: str) -> Path:
    if not value.strip():
        raise ContractError("post-training project state path cannot be empty")
    configured = Path(value)
    return configured.resolve() if configured.is_absolute() else (control_dir / configured).resolve()


def _validate_entry(value: str) -> str:
    module, separator, attribute = value.partition(":")
    if not separator or not module or not attribute or ":" in attribute:
        raise ContractError("post-training project entry must use MODULE:CALLABLE syntax")
    if not all(part.isidentifier() for part in module.split(".")) or not attribute.isidentifier():
        raise ContractError("post-training project entry contains an invalid Python name")
    return value


__all__ = ["ProjectLayout", "discover_project", "load_project_layout"]
