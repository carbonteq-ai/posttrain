"""Developer-facing source selection for framework job packing."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from posttrain.catalog import ProjectLayout
from posttrain.common import ContractError
from posttrain.execution_pack import SourceSnapshotRequest


@dataclass(frozen=True, slots=True)
class ProjectPackConfig:
    """Committed project source selection, with optional CLI overrides."""

    pyproject: Path
    project_packages: tuple[str, ...]
    source_includes: tuple[str, ...]

    def source_request(self, root: Path) -> SourceSnapshotRequest:
        return SourceSnapshotRequest(
            root=root,
            includes=self.source_includes,
            install_roots=self.project_packages,
        )


def load_project_pack_config(
    layout: ProjectLayout,
    *,
    project_packages: tuple[str, ...] | None = None,
    source_includes: tuple[str, ...] | None = None,
) -> ProjectPackConfig:
    """Load `[tool.posttrain.pack]`; explicit CLI values win when supplied."""

    pyproject = layout.root / "pyproject.toml"
    try:
        document = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ContractError(
            f"job packing requires a project pyproject.toml: {pyproject}"
        ) from error
    except tomllib.TOMLDecodeError as error:
        raise ContractError(f"invalid project pyproject.toml: {error}") from error
    tool = document.get("tool", {})
    if not isinstance(tool, dict):
        raise ContractError("pyproject [tool] must be a table")
    posttrain = tool.get("posttrain", {})
    if not isinstance(posttrain, dict):
        raise ContractError("pyproject [tool.posttrain] must be a table")
    pack = posttrain.get("pack", {})
    if not isinstance(pack, dict):
        raise ContractError("pyproject [tool.posttrain.pack] must be a table")
    if unknown := sorted(set(pack) - {"project_packages", "source_includes"}):
        raise ContractError(
            "pyproject [tool.posttrain.pack] has unknown fields: "
            + ", ".join(unknown)
        )

    configured_packages = (
        project_packages
        if project_packages is not None
        else _string_tuple(
            pack.get("project_packages", ["."]),
            "project_packages",
        )
    )
    configured_includes = (
        source_includes
        if source_includes is not None
        else _string_tuple(
            pack.get(
                "source_includes",
                _default_includes(layout.root, configured_packages, document),
            ),
            "source_includes",
        )
    )
    packages = _normalized_paths(configured_packages, "project package")
    includes = _normalized_paths(configured_includes, "source include")
    for package in packages:
        root = (
            layout.root
            if package == "."
            else layout.root.joinpath(*package.split("/"))
        )
        if not root.is_dir() or not (root / "pyproject.toml").is_file():
            raise ContractError(
                f"project package has no pyproject.toml: {package}"
            )
    for included in includes:
        path = (
            layout.root
            if included == "."
            else layout.root.joinpath(*included.split("/"))
        )
        if not path.exists():
            raise ContractError(f"project source include does not exist: {included}")
    return ProjectPackConfig(pyproject, packages, includes)


def _default_includes(
    root: Path,
    packages: tuple[str, ...],
    pyproject: dict[str, object],
) -> list[str]:
    if packages != (".",):
        return list(packages)
    includes = ["pyproject.toml"]
    if (root / "src").is_dir():
        includes.append("src")
    project = pyproject.get("project", {})
    if isinstance(project, dict):
        readme = project.get("readme")
        readme_path: str | None = None
        if isinstance(readme, str):
            readme_path = readme
        elif isinstance(readme, dict) and isinstance(readme.get("file"), str):
            readme_path = readme["file"]
        if readme_path is not None and (root / readme_path).is_file():
            includes.append(readme_path)
    return includes


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise ContractError(
            f"pyproject [tool.posttrain.pack].{field} must be a string array"
        )
    return tuple(value)


def _normalized_paths(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        if not value or value != value.strip() or "\\" in value:
            raise ContractError(f"{label} must be a normalized relative path")
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or any(part in {"", ".."} for part in path.parts)
            or (path.as_posix() != "." and "." in path.parts)
            or path.as_posix() != value
            or ".git" in path.parts
            or ".posttrain" in path.parts
        ):
            raise ContractError(
                f"{label} must be a project source path outside control state"
            )
        normalized.append(value)
    result = tuple(sorted(normalized))
    if not result or len(set(result)) != len(result):
        raise ContractError(f"{label} values must be non-empty and unique")
    return result
