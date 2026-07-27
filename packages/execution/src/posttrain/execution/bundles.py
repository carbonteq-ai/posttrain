"""Deterministic, secret-averse execution bundles verified before job startup."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from posttrain.common import ContractError

from .contracts import BundleRef
from .manifest import JOB_MANIFEST_PATH, ExecutionJobManifest

_MANIFEST = ".posttrain/bundle.json"


@dataclass(frozen=True, slots=True)
class ExecutionBundlePlan:
    """Content identity of a bundle before any destination is created."""

    digest: str
    file_count: int
    size_bytes: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ContractError(f"bundle destination must be a normalized relative path: {value!r}")
    if str(path) in {_MANIFEST, str(JOB_MANIFEST_PATH)}:
        raise ContractError("bundle payload cannot write framework metadata")
    if path.parts[0] == ".posttrain" and (
        len(path.parts) < 2
        or path.parts[1] not in {"project.toml", "project.yaml", "catalog", "work_packages"}
    ):
        raise ContractError("bundle payload can include only tracked posttrain project configuration")
    return path


def _payload_files(inputs: Mapping[str, Path]) -> list[tuple[PurePosixPath, Path]]:
    files: list[tuple[PurePosixPath, Path]] = []
    claimed: set[PurePosixPath] = set()
    for destination, raw_source in sorted(inputs.items()):
        target = _relative_path(destination)
        source = raw_source.resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        if raw_source.is_symlink() or source.is_symlink():
            raise ContractError("execution bundles do not accept symlink inputs")
        candidates = [source] if source.is_file() else sorted(source.rglob("*"))
        for candidate in candidates:
            if candidate.is_symlink():
                raise ContractError(f"execution bundles do not accept symlinks: {candidate}")
            if not candidate.is_file():
                continue
            relative = target if source.is_file() else target / candidate.relative_to(source).as_posix()
            if relative in claimed:
                raise ContractError(f"duplicate bundle destination: {relative}")
            claimed.add(relative)
            files.append((relative, candidate))
    return sorted(files, key=lambda item: str(item[0]))


def _manifest_entries(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.relative_to(root).as_posix() == _MANIFEST:
            continue
        if path.is_symlink():
            raise ContractError(f"execution bundle contains a symlink: {path}")
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return entries


def _bundle_digest(entries: list[dict[str, Any]]) -> str:
    encoded = json.dumps(entries, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def plan_execution_bundle(
    inputs: Mapping[str, Path],
    manifest: ExecutionJobManifest,
) -> ExecutionBundlePlan:
    """Hash the exact execution payload without creating a staging directory."""

    payload = _payload_files(inputs)
    if not payload:
        raise ContractError("execution bundle payload cannot be empty")
    entries = [
        {
            "path": relative.as_posix(),
            "sha256": _sha256(source),
            "size_bytes": source.stat().st_size,
        }
        for relative, source in payload
    ]
    manifest_bytes = manifest.to_bytes()
    entries.append(
        {
            "path": str(JOB_MANIFEST_PATH),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "size_bytes": len(manifest_bytes),
        }
    )
    entries.sort(key=lambda entry: str(entry["path"]))
    return ExecutionBundlePlan(
        digest=_bundle_digest(entries),
        file_count=len(entries),
        size_bytes=sum(int(entry["size_bytes"]) for entry in entries),
    )


def build_bundle(inputs: Mapping[str, Path], destination: Path) -> BundleRef:
    """Copy an explicit payload selection into a new immutable bundle directory."""

    return _build_bundle(inputs, destination, framework_files={})


def build_execution_bundle(
    inputs: Mapping[str, Path],
    destination: Path,
    manifest: ExecutionJobManifest,
) -> BundleRef:
    """Build a bundle containing one verified framework-owned worker manifest."""

    return _build_bundle(
        inputs,
        destination,
        framework_files={JOB_MANIFEST_PATH: manifest.to_bytes()},
    )


def _build_bundle(
    inputs: Mapping[str, Path],
    destination: Path,
    *,
    framework_files: Mapping[PurePosixPath, bytes],
) -> BundleRef:
    if not destination.is_absolute():
        raise ContractError("execution bundle destination must be absolute")
    if destination.exists():
        raise FileExistsError(destination)
    payload = _payload_files(inputs)
    if not payload:
        raise ContractError("execution bundle payload cannot be empty")

    destination.mkdir(parents=True, mode=0o700)
    try:
        for relative, source in payload:
            output = destination / Path(*relative.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, output)
            output.chmod(0o755 if source.stat().st_mode & 0o111 else 0o644)
        for relative, content in sorted(framework_files.items(), key=lambda item: str(item[0])):
            output = destination / Path(*relative.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(content)
            output.chmod(0o644)
        entries = _manifest_entries(destination)
        digest = _bundle_digest(entries)
        manifest = destination / _MANIFEST
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(
                {
                    "schema": "posttrain.execution-bundle.v1",
                    "digest": digest,
                    "files": entries,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest.chmod(0o644)
    except BaseException:
        shutil.rmtree(destination)
        raise
    return BundleRef(destination, digest)


def verify_bundle(bundle: BundleRef) -> None:
    """Fail before execution when the manifest or any payload byte has drifted."""

    manifest_path = bundle.path / _MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ContractError("execution bundle manifest is missing or invalid") from error
    entries = _manifest_entries(bundle.path)
    digest = _bundle_digest(entries)
    if manifest.get("schema") != "posttrain.execution-bundle.v1":
        raise ContractError("execution bundle schema is unsupported")
    if manifest.get("files") != entries:
        raise ContractError("execution bundle file manifest does not match payload")
    if manifest.get("digest") != digest or bundle.digest != digest:
        raise ContractError("execution bundle digest does not match payload")
