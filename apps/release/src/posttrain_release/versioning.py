"""Deterministic lockstep version expansion for first-party distributions."""

from __future__ import annotations

import hashlib
import re
import subprocess
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from posttrain.runtime_images.manifest import ManifestError, load_manifest

_MANIFEST = Path("release/manifest.toml")
_TRAINING_CATALOG = Path("packages/catalog/src/posttrain/catalog/base/training.yaml")
_VERSION = re.compile(r"^[0-9]+[.][0-9]+[.][0-9]+(?:[a-zA-Z0-9.-]+)?$")
_INTERNAL_REQUIREMENT = re.compile(r"^(?P<name>posttrain(?:-[A-Za-z0-9._-]+)?)(?:\[[^]]+\])?==(?P<version>[^;\s]+)")
_PROJECT_VERSION_LINE = re.compile(r'(?m)^(version\s*=\s*)"[^"]+"')
_INTERNAL_PIN = re.compile(r'(posttrain(?:-[A-Za-z0-9._-]+)?(?:\[[^]]+\])?==)[^";\s]+')
_LOCK_DIGEST = re.compile(r"(?m)^(\s*dependency_lock_sha256:\s*)[0-9a-f]{64}\s*$")


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


def check_release(repository_root: Path) -> ReleaseCheck:
    root = repository_root.resolve()
    manifest = load_release_manifest(root)
    errors: list[str] = []
    pin_count = 0
    pyprojects = workspace_pyprojects(root)
    for path in pyprojects:
        relative = path.relative_to(root)
        try:
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as error:
            errors.append(f"{relative}: cannot read package metadata: {error}")
            continue
        project = payload.get("project")
        observed = project.get("version") if isinstance(project, dict) else None
        if observed != manifest.version:
            errors.append(f"{relative}: project.version is {observed!r}, expected {manifest.version!r}")
        for requirement in _strings(payload):
            match = _INTERNAL_REQUIREMENT.match(requirement)
            if match is None:
                continue
            pin_count += 1
            if match.group("version") != manifest.version:
                errors.append(
                    f"{relative}: {match.group('name')} is pinned to {match.group('version')!r}, "
                    f"expected {manifest.version!r}"
                )

    catalog = root / _TRAINING_CATALOG
    expected_lock = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    catalog_text = catalog.read_text(encoding="utf-8")
    digests = set(re.findall(r"dependency_lock_sha256:\s*([0-9a-f]{64})", catalog_text))
    if digests != {expected_lock}:
        errors.append(
            f"{_TRAINING_CATALOG}: dependency lock digests are {sorted(digests)!r}, expected {expected_lock!r}"
        )

    try:
        load_manifest()
    except (OSError, ValueError, ManifestError) as error:
        errors.append(f"runtime image manifest is invalid: {error}")

    if errors:
        raise ValueError("release consistency check failed:\n- " + "\n- ".join(errors))
    return ReleaseCheck(version=manifest.version, package_count=len(pyprojects), internal_pin_count=pin_count)


def prepare_release(repository_root: Path, version: str) -> ReleaseCheck:
    if not _VERSION.fullmatch(version):
        raise ValueError(f"invalid release version: {version!r}")
    root = repository_root.resolve()
    manifest_path = root / _MANIFEST
    if not manifest_path.is_file():
        raise ValueError(f"release manifest not found: {manifest_path}")
    manifest_path.write_text(f'schema_version = 1\nversion = "{version}"\n', encoding="utf-8")
    for path in workspace_pyprojects(root):
        text = path.read_text(encoding="utf-8")
        updated, count = _PROJECT_VERSION_LINE.subn(rf'\g<1>"{version}"', text, count=1)
        if count != 1:
            raise ValueError(f"cannot locate [project] version in {path.relative_to(root)}")
        updated = _INTERNAL_PIN.sub(rf"\g<1>{version}", updated)
        path.write_text(updated, encoding="utf-8")

    result = subprocess.run(
        ["uv", "lock"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"uv lock failed: {detail}")
    digest = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    catalog_path = root / _TRAINING_CATALOG
    catalog = catalog_path.read_text(encoding="utf-8")
    catalog, count = _LOCK_DIGEST.subn(rf"\g<1>{digest}", catalog)
    if count == 0:
        raise ValueError(f"no dependency lock digests found in {_TRAINING_CATALOG}")
    catalog_path.write_text(catalog, encoding="utf-8")
    return check_release(root)


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


__all__ = [
    "ReleaseCheck",
    "ReleaseManifest",
    "check_release",
    "load_release_manifest",
    "prepare_release",
    "workspace_pyprojects",
]
