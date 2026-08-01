"""Manifest-controlled release metadata and dependency-lock generation."""

from __future__ import annotations

import hashlib
import re
import shutil
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from posttrain.runtime_images.manifest import ManifestError, load_manifest

_MANIFEST = Path("release/manifest.toml")
_TRAINING_CATALOG = Path("packages/catalog/src/posttrain/catalog/base/training.yaml")
_DEPENDENCY_LOCKS = Path("packages/catalog/src/posttrain/catalog/base/locks.toml")
_TRAIN_PROJECT = Path("packages/train/pyproject.toml")
_SOURCE_VERSION = "0.0.0"
_VERSION = re.compile(r"^[0-9]+[.][0-9]+[.][0-9]+(?:[a-zA-Z0-9.-]+)?$")
_PROJECT_VERSION_LINE = re.compile(r'(?m)^(version\s*=\s*)"0[.]0[.]0"\s*$')
_LOCK_PACKAGE_BLOCK = re.compile(r"(?ms)^\[\[package\]\]\n.*?(?=^\[\[package\]\]|\Z)")
_LOCK_PACKAGE_NAME = re.compile(r'(?m)^name\s*=\s*"(?P<name>[^"]+)"\s*$')
_LOCK_PACKAGE_VERSION = re.compile(r'(?m)^(version\s*=\s*)"(?P<version>[^"]+)"\s*$')
_INTERNAL_REQUIREMENT = re.compile(
    r"^(?P<requirement>posttrain(?:-[A-Za-z0-9._-]+)?(?:\[[^]]+\])?)(?P<constraint>[^;\s]*)(?P<marker>\s*;.*)?$"
)
_TRL_SOURCE = re.compile(r"^trl\s*@\s*git\+https://github[.]com/carbonteq-ai/trl[.]git@(?P<revision>[0-9a-f]{40})$")
_LOCK_REFERENCE = "trl-fork@current"


@dataclass(frozen=True, slots=True)
class ReleaseManifest:
    schema_version: int
    version: str


@dataclass(frozen=True, slots=True)
class ReleaseCheck:
    version: str
    package_count: int
    internal_pin_count: int


def load_release_manifest(repository_root: Path) -> ReleaseManifest:
    path = repository_root / _MANIFEST
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"release manifest not found: {path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"invalid release manifest {path}: {error}") from error
    schema_version = payload.get("schema_version")
    version = payload.get("version")
    if schema_version != 1:
        raise ValueError(f"unsupported release manifest schema_version {schema_version!r}; expected 1")
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise ValueError(f"invalid release version: {version!r}")
    return ReleaseManifest(schema_version=1, version=version)


def workspace_pyprojects(repository_root: Path) -> tuple[Path, ...]:
    paths = [repository_root / "pyproject.toml"]
    paths.extend(repository_root.glob("apps/*/pyproject.toml"))
    paths.extend(repository_root.glob("packages/*/pyproject.toml"))
    return tuple(
        sorted((path.resolve() for path in paths), key=lambda path: path.relative_to(repository_root).as_posix())
    )


def publishable_pyprojects(repository_root: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in workspace_pyprojects(repository_root)
        if isinstance(tomllib.loads(path.read_text(encoding="utf-8")).get("build-system"), dict)
    )


def _publishable_project_names(paths: tuple[Path, ...]) -> set[str]:
    names: set[str] = set()
    for path in paths:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        project = payload.get("project")
        name = project.get("name") if isinstance(project, dict) else None
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path}: publishable project must declare a non-empty name")
        if name in names:
            raise ValueError(f"duplicate publishable project name: {name!r}")
        names.add(name)
    return names


def check_release(repository_root: Path) -> ReleaseCheck:
    root = repository_root.resolve()
    manifest = load_release_manifest(root)
    errors: list[str] = []
    pin_count = 0
    # The repository root is a virtual uv workspace, not a package.  Release
    # metadata belongs solely to the publishable workspace members; inspecting
    # every ``pyproject.toml`` would incorrectly require a root ``[project]``
    # table and make that virtual-workspace shape invalid.
    publishable = publishable_pyprojects(root)
    for path in publishable:
        relative = path.relative_to(root)
        try:
            text = path.read_text(encoding="utf-8")
            payload = tomllib.loads(text)
        except (OSError, tomllib.TOMLDecodeError) as error:
            errors.append(f"{relative}: cannot read package metadata: {error}")
            continue
        project = payload.get("project")
        observed = project.get("version") if isinstance(project, dict) else None
        if observed != _SOURCE_VERSION:
            errors.append(f"{relative}: source project.version is {observed!r}, expected template {_SOURCE_VERSION!r}")
        project_metadata: dict[str, Any] = project if isinstance(project, dict) else {}
        for requirement in _project_requirements(project_metadata):
            match = _INTERNAL_REQUIREMENT.fullmatch(requirement)
            if match is None:
                continue
            if match.group("constraint"):
                errors.append(
                    f"{relative}: source dependency {requirement!r} contains a release pin; "
                    "pins belong only in staged release metadata"
                )
        try:
            rendered, rendered_pins = render_project_metadata(text, manifest.version, relative)
            rendered_payload = tomllib.loads(rendered)
        except (ValueError, tomllib.TOMLDecodeError) as error:
            errors.append(str(error))
            continue
        rendered_project = rendered_payload.get("project")
        rendered_version = rendered_project.get("version") if isinstance(rendered_project, dict) else None
        if rendered_version != manifest.version:
            errors.append(f"{relative}: staged project.version is {rendered_version!r}, expected {manifest.version!r}")
        pin_count += rendered_pins

    errors.extend(_dependency_lock_errors(root))
    try:
        load_manifest()
    except (OSError, ValueError, ManifestError) as error:
        errors.append(f"runtime image manifest is invalid: {error}")

    if errors:
        raise ValueError("release consistency check failed:\n- " + "\n- ".join(errors))
    return ReleaseCheck(version=manifest.version, package_count=len(publishable), internal_pin_count=pin_count)


def prepare_release(repository_root: Path, version: str) -> ReleaseCheck:
    """Change only the authored release version; package metadata stays templated."""

    if not _VERSION.fullmatch(version):
        raise ValueError(f"invalid release version: {version!r}")
    root = repository_root.resolve()
    manifest_path = root / _MANIFEST
    if not manifest_path.is_file():
        raise ValueError(f"release manifest not found: {manifest_path}")
    manifest_path.write_text(f'schema_version = 1\nversion = "{version}"\n', encoding="utf-8")
    return check_release(root)


def render_project_metadata(text: str, version: str, relative: Path) -> tuple[str, int]:
    """Render one publishable source template into standards-readable metadata."""

    payload = tomllib.loads(text)
    project = payload.get("project")
    if not isinstance(project, dict) or project.get("version") != _SOURCE_VERSION:
        raise ValueError(f"{relative}: release rendering requires source version {_SOURCE_VERSION}")
    rendered, version_count = _PROJECT_VERSION_LINE.subn(rf'\g<1>"{version}"', text, count=1)
    if version_count != 1:
        raise ValueError(f"{relative}: cannot locate the source [project] version")
    pin_count = 0
    for requirement in dict.fromkeys(_project_requirements(project)):
        match = _INTERNAL_REQUIREMENT.fullmatch(requirement)
        if match is None:
            continue
        if match.group("constraint"):
            raise ValueError(f"{relative}: source dependency {requirement!r} must not carry a release constraint")
        marker = match.group("marker") or ""
        pinned = f"{match.group('requirement')}=={version}{marker}"
        quoted = f'"{requirement}"'
        occurrences = rendered.count(quoted)
        if occurrences < 1:
            raise ValueError(f"{relative}: cannot render TOML dependency {requirement!r}")
        rendered = rendered.replace(quoted, f'"{pinned}"')
        pin_count += occurrences
    return rendered, pin_count


def stage_release(repository_root: Path, destination: Path) -> ReleaseCheck:
    """Copy the source tree and expand release metadata only in that copy."""

    root = repository_root.resolve()
    target = destination.resolve()
    if target.exists():
        raise ValueError(f"release staging destination already exists: {target}")

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored = {
            name
            for name in names
            if name in {".git", ".venv", ".venvs", "__pycache__", ".pytest_cache", ".ruff_cache", "dist"}
        }
        if Path(directory).name == ".posttrain" and "state" in names:
            ignored.add("state")
        return ignored

    shutil.copytree(root, target, ignore=ignore)
    manifest = load_release_manifest(target)
    for path in publishable_pyprojects(target):
        relative = path.relative_to(target)
        rendered, _ = render_project_metadata(path.read_text(encoding="utf-8"), manifest.version, relative)
        path.write_text(rendered, encoding="utf-8")
    lock_path = target / "uv.lock"
    if not lock_path.is_file():
        raise ValueError(f"release staging requires a workspace lock: {lock_path}")
    rendered_lock = render_workspace_lock(
        lock_path.read_text(encoding="utf-8"), manifest.version, publishable_pyprojects(target)
    )
    lock_path.write_text(rendered_lock, encoding="utf-8")
    return _check_staged_release(target, manifest.version)


def render_workspace_lock(text: str, version: str, publishable: tuple[Path, ...]) -> str:
    """Project the source workspace lock into the staged release metadata.

    ``uv`` records a workspace package's version in the lock.  Rendering just
    the package metadata therefore leaves a staged tree internally
    inconsistent: ``uv sync --locked`` correctly refuses to install it.  The
    dependency graph and all third-party artifacts remain source-lock bytes;
    this projection changes only the versions of editable first-party package
    records, precisely as ``uv lock`` would after the metadata projection.
    """

    names = _publishable_project_names(publishable)
    seen: set[str] = set()

    def render_block(match: re.Match[str]) -> str:
        block = match.group(0)
        name_match = _LOCK_PACKAGE_NAME.search(block)
        if name_match is None or name_match.group("name") not in names:
            return block
        name = name_match.group("name")
        if "source = { editable = " not in block:
            raise ValueError(f"uv.lock: staged package {name!r} is not an editable workspace record")
        version_matches = tuple(_LOCK_PACKAGE_VERSION.finditer(block))
        if len(version_matches) != 1:
            raise ValueError(f"uv.lock: staged package {name!r} must have exactly one version")
        version_match = version_matches[0]
        if version_match.group("version") != _SOURCE_VERSION:
            raise ValueError(
                f"uv.lock: source package {name!r} has version {version_match.group('version')!r}, "
                f"expected template {_SOURCE_VERSION!r}"
            )
        if name in seen:
            raise ValueError(f"uv.lock: duplicate workspace package record for {name!r}")
        seen.add(name)
        return block[: version_match.start()] + f'{version_match.group(1)}"{version}"' + block[version_match.end() :]

    rendered = _LOCK_PACKAGE_BLOCK.sub(render_block, text)
    missing = sorted(names - seen)
    if missing:
        raise ValueError(f"uv.lock: missing editable workspace records for {missing!r}")
    return rendered


def lock_dependencies(repository_root: Path) -> str:
    """Regenerate the one catalog lock record from pinned source inputs."""

    root = repository_root.resolve()
    revision = _trl_revision(root / _TRAIN_PROJECT)
    digest = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    path = root / _DEPENDENCY_LOCKS
    path.write_text(
        "\n".join(
            (
                "# Generated by `posttrain-release lock-dependencies`; do not edit.",
                "schema_version = 1",
                "",
                f'[locks."{_LOCK_REFERENCE}"]',
                'source = "uv.lock"',
                f'source_revision = "{revision}"',
                f'dependency_lock_sha256 = "{digest}"',
                "",
            )
        ),
        encoding="utf-8",
    )
    return digest


def _check_staged_release(root: Path, version: str) -> ReleaseCheck:
    pin_count = 0
    publishable = publishable_pyprojects(root)
    errors: list[str] = []
    for path in publishable:
        relative = path.relative_to(root)
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        project = payload.get("project")
        observed = project.get("version") if isinstance(project, dict) else None
        if observed != version:
            errors.append(f"{relative}: staged project.version is {observed!r}, expected {version!r}")
        project_metadata: dict[str, Any] = project if isinstance(project, dict) else {}
        for requirement in _project_requirements(project_metadata):
            match = _INTERNAL_REQUIREMENT.fullmatch(requirement)
            if match is None:
                continue
            pin_count += 1
            if match.group("constraint") != f"=={version}":
                errors.append(f"{relative}: staged dependency {requirement!r} is not pinned to {version}")
    if errors:
        raise ValueError("staged release metadata check failed:\n- " + "\n- ".join(errors))
    return ReleaseCheck(version=version, package_count=len(publishable), internal_pin_count=pin_count)


def _dependency_lock_errors(root: Path) -> list[str]:
    errors: list[str] = []
    lock_path = root / _DEPENDENCY_LOCKS
    try:
        payload = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        return [f"{_DEPENDENCY_LOCKS}: cannot read generated lock table: {error}"]
    locks = payload.get("locks")
    lock = locks.get(_LOCK_REFERENCE) if isinstance(locks, dict) else None
    if not isinstance(lock, dict):
        return [f"{_DEPENDENCY_LOCKS}: missing {_LOCK_REFERENCE!r}"]
    expected_revision = _trl_revision(root / _TRAIN_PROJECT)
    expected_digest = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    if lock.get("source") != "uv.lock":
        errors.append(f"{_DEPENDENCY_LOCKS}: {_LOCK_REFERENCE} source must be 'uv.lock'")
    if lock.get("source_revision") != expected_revision:
        errors.append(
            f"{_DEPENDENCY_LOCKS}: source revision is {lock.get('source_revision')!r}, expected {expected_revision!r}"
        )
    if lock.get("dependency_lock_sha256") != expected_digest:
        errors.append(
            f"{_DEPENDENCY_LOCKS}: digest is {lock.get('dependency_lock_sha256')!r}, expected {expected_digest!r}"
        )
    references = set(
        re.findall(r"(?m)^\s*dependency_lock:\s*([^\s#]+)\s*$", (root / _TRAINING_CATALOG).read_text(encoding="utf-8"))
    )
    if references != {_LOCK_REFERENCE}:
        errors.append(f"{_TRAINING_CATALOG}: dependency lock references are {sorted(references)!r}")
    return errors


def _trl_revision(path: Path) -> str:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    project = payload.get("project")
    for requirement in _project_requirements(project if isinstance(project, dict) else {}):
        match = _TRL_SOURCE.fullmatch(requirement)
        if match is not None:
            return match.group("revision")
    raise ValueError(f"cannot locate the pinned CarbonTeq TRL revision in {path}")


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _project_requirements(project: dict[str, Any]) -> Iterator[str]:
    yield from _strings(project.get("dependencies"))
    yield from _strings(project.get("optional-dependencies"))


__all__ = [
    "ReleaseCheck",
    "ReleaseManifest",
    "check_release",
    "load_release_manifest",
    "lock_dependencies",
    "prepare_release",
    "publishable_pyprojects",
    "render_project_metadata",
    "render_workspace_lock",
    "stage_release",
    "workspace_pyprojects",
]
