"""Content-addressed runtime materialization passed from candidate to final release."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

_SCHEMA = "posttrain.release-materialization.v1"
_MANIFEST = PurePosixPath("published.toml")
_LOCK_ROOT = PurePosixPath("runtime-locks")
_SOURCE_MANIFEST = Path("packages/runtime-images/src/posttrain/runtime_images/published.toml")
_SOURCE_LOCK_ROOT = Path("packages/runtime-images/src/posttrain/runtime_images/containers/posttrain-job-kinds/locks")
_SHA = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_READINESS_CHECKS = {"release", "tests", "lint", "format", "types", "imports"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return payload


def _verify_readiness_binding(receipt: dict[str, object], readiness_receipt: Path) -> None:
    readiness_bytes = readiness_receipt.resolve().read_bytes()
    readiness = json.loads(readiness_bytes)
    if not isinstance(readiness, dict) or readiness.get("schema") != "posttrain.release-readiness.v2":
        raise ValueError("materialization requires a v2 release readiness receipt")
    if receipt.get("readiness_sha256") != hashlib.sha256(readiness_bytes).hexdigest():
        raise ValueError("materialization readiness receipt digest does not match")
    bindings = {
        "target_version": "framework_version",
        "source_sha": "source_sha",
        "source_tree": "source_tree",
    }
    for materialization_key, readiness_key in bindings.items():
        if receipt.get(materialization_key) != readiness.get(readiness_key):
            raise ValueError(f"materialization {materialization_key} differs from its readiness receipt")
    checks = readiness.get("checks")
    if not isinstance(checks, list):
        raise ValueError("materialization readiness receipt has no checks")
    successful = {
        item.get("name")
        for item in checks
        if isinstance(item, dict) and item.get("status") == "success" and isinstance(item.get("name"), str)
    }
    missing = sorted(_READINESS_CHECKS - successful)
    if missing:
        raise ValueError(f"materialization readiness receipt is missing successful checks: {', '.join(missing)}")


def _validate_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError("materialization file path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise ValueError(f"unsafe materialization file path: {value!r}")
    if path != _MANIFEST and _LOCK_ROOT not in path.parents:
        raise ValueError(f"materialization contains an unsupported file: {value!r}")
    return path


def _runtime_files(root: Path) -> tuple[Path, ...]:
    manifest = root / _SOURCE_MANIFEST
    locks = root / _SOURCE_LOCK_ROOT
    if not manifest.is_file():
        raise ValueError(f"runtime image manifest not found: {manifest}")
    if not locks.is_dir():
        raise ValueError(f"runtime lock directory not found: {locks}")
    lock_files = tuple(sorted(path for path in locks.rglob("*") if path.is_file()))
    if not lock_files:
        raise ValueError(f"runtime lock directory is empty: {locks}")
    if any(path.is_symlink() for path in (manifest, *lock_files)):
        raise ValueError("runtime materialization inputs must not be symbolic links")
    return (manifest, *lock_files)


def _content_digest(receipt: dict[str, object]) -> str:
    content = {key: value for key, value in receipt.items() if key not in {"created_at", "materialization_sha256"}}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def create_materialization(
    repository_root: Path,
    readiness_receipt: Path,
    destination: Path,
    *,
    candidate_version: str,
) -> dict[str, object]:
    """Copy generated runtime inputs and bind them to exact candidate source evidence."""

    root = repository_root.resolve()
    target = destination.resolve()
    if target.exists():
        raise ValueError(f"materialization destination already exists: {target}")
    readiness_bytes = readiness_receipt.resolve().read_bytes()
    readiness = json.loads(readiness_bytes)
    if not isinstance(readiness, dict) or readiness.get("schema") != "posttrain.release-readiness.v2":
        raise ValueError("materialization requires a v2 release readiness receipt")
    target_version = readiness.get("framework_version")
    source_sha = readiness.get("source_sha")
    source_tree = readiness.get("source_tree")
    if not isinstance(target_version, str) or not target_version:
        raise ValueError("readiness receipt has no framework version")
    if not isinstance(source_sha, str) or _SHA.fullmatch(source_sha) is None:
        raise ValueError("readiness receipt has no valid source SHA")
    if not isinstance(source_tree, str) or _SHA.fullmatch(source_tree) is None:
        raise ValueError("readiness receipt has no valid source tree")
    if re.fullmatch(rf"{re.escape(target_version)}rc[1-9][0-9]*", candidate_version) is None:
        raise ValueError(f"candidate version {candidate_version!r} is not an RC of {target_version!r}")

    target.mkdir(parents=True)
    files: list[dict[str, object]] = []
    try:
        for source in _runtime_files(root):
            if source == root / _SOURCE_MANIFEST:
                relative = _MANIFEST
            else:
                relative = _LOCK_ROOT / PurePosixPath(source.relative_to(root / _SOURCE_LOCK_ROOT).as_posix())
            copied = target / Path(relative)
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, copied, follow_symlinks=False)
            files.append(
                {
                    "path": relative.as_posix(),
                    "sha256": _sha256(copied),
                    "size": copied.stat().st_size,
                }
            )
        receipt: dict[str, object] = {
            "schema": _SCHEMA,
            "target_version": target_version,
            "candidate_version": candidate_version,
            "source_sha": source_sha,
            "source_tree": source_tree,
            "readiness_sha256": hashlib.sha256(readiness_bytes).hexdigest(),
            "files": files,
            "created_at": datetime.now(UTC).isoformat(),
        }
        receipt["materialization_sha256"] = _content_digest(receipt)
        _verify_readiness_binding(receipt, readiness_receipt)
        (target / "materialization.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return receipt


def verify_materialization(
    materialization_root: Path,
    *,
    target_version: str | None = None,
    source_sha: str | None = None,
    source_tree: str | None = None,
    readiness_receipt: Path | None = None,
) -> dict[str, object]:
    """Verify schema, identity, exact file set, and every retained runtime byte."""

    root = materialization_root.resolve()
    receipt = _load_json(root / "materialization.json")
    if receipt.get("schema") != _SCHEMA:
        raise ValueError("unsupported release materialization receipt")
    expected = {
        "target_version": target_version,
        "source_sha": source_sha,
        "source_tree": source_tree,
    }
    for name, value in expected.items():
        if value is not None and receipt.get(name) != value:
            raise ValueError(f"materialization {name} does not match the accepted release")
    if _SHA.fullmatch(str(receipt.get("source_sha", ""))) is None:
        raise ValueError("materialization source SHA is invalid")
    if _SHA.fullmatch(str(receipt.get("source_tree", ""))) is None:
        raise ValueError("materialization source tree is invalid")
    if _DIGEST.fullmatch(str(receipt.get("readiness_sha256", ""))) is None:
        raise ValueError("materialization readiness digest is invalid")
    receipt_target = receipt.get("target_version")
    candidate_version = receipt.get("candidate_version")
    if not isinstance(receipt_target, str) or not isinstance(candidate_version, str):
        raise ValueError("materialization versions are invalid")
    if re.fullmatch(rf"{re.escape(receipt_target)}rc[1-9][0-9]*", candidate_version) is None:
        raise ValueError("materialization candidate version does not belong to its target")

    entries = receipt.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("materialization receipt has no files")
    declared: set[PurePosixPath] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("materialization file entry is invalid")
        relative = _validate_relative_path(entry.get("path"))
        if relative in declared:
            raise ValueError(f"duplicate materialization file: {relative}")
        declared.add(relative)
        path = root / Path(relative)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"materialization file is missing or unsafe: {relative}")
        if entry.get("sha256") != _sha256(path) or entry.get("size") != path.stat().st_size:
            raise ValueError(f"materialization file does not match receipt: {relative}")
    if _MANIFEST not in declared or not any(_LOCK_ROOT in path.parents for path in declared):
        raise ValueError("materialization must contain the manifest and at least one runtime lock")
    manifest = tomllib.loads((root / Path(_MANIFEST)).read_text(encoding="utf-8"))
    if manifest.get("framework_version") != receipt_target:
        raise ValueError("materialization manifest framework version does not match its target")
    observed = {
        PurePosixPath(path.relative_to(root).as_posix())
        for path in root.rglob("*")
        if path.is_file() and path.name != "materialization.json"
    }
    if observed != declared:
        extra = sorted(str(path) for path in observed - declared)
        missing = sorted(str(path) for path in declared - observed)
        raise ValueError(f"materialization file set differs from receipt: extra={extra}, missing={missing}")
    if receipt.get("materialization_sha256") != _content_digest(receipt):
        raise ValueError("release materialization content digest does not match")
    if readiness_receipt is not None:
        _verify_readiness_binding(receipt, readiness_receipt)
    return receipt


def apply_materialization(
    materialization_root: Path,
    destination_root: Path,
    *,
    target_version: str | None = None,
    source_sha: str | None = None,
    source_tree: str | None = None,
    readiness_receipt: Path | None = None,
) -> dict[str, object]:
    """Verify first, then replace only the staged manifest and runtime lock directory."""

    receipt = verify_materialization(
        materialization_root,
        target_version=target_version,
        source_sha=source_sha,
        source_tree=source_tree,
        readiness_receipt=readiness_receipt,
    )
    source = materialization_root.resolve()
    destination = destination_root.resolve()
    target_manifest = destination / _SOURCE_MANIFEST
    target_locks = destination / _SOURCE_LOCK_ROOT
    if not target_manifest.is_file() or not target_locks.is_dir():
        raise ValueError("materialization destination is not a staged Posttrain source tree")

    temporary_root = Path(tempfile.mkdtemp(prefix=".posttrain-materialization-", dir=target_locks.parent))
    try:
        temporary_locks = temporary_root / "locks"
        shutil.copytree(source / Path(_LOCK_ROOT), temporary_locks)
        temporary_manifest = temporary_root / "published.toml"
        shutil.copyfile(source / Path(_MANIFEST), temporary_manifest)
        backup_locks = temporary_root / "previous-locks"
        backup_manifest = temporary_root / "previous-published.toml"
        os.replace(target_locks, backup_locks)
        os.replace(target_manifest, backup_manifest)
        try:
            os.replace(temporary_locks, target_locks)
            os.replace(temporary_manifest, target_manifest)
        except BaseException:
            if target_locks.exists():
                shutil.rmtree(target_locks)
            if target_manifest.exists():
                target_manifest.unlink()
            if backup_locks.exists():
                os.replace(backup_locks, target_locks)
            if backup_manifest.exists():
                os.replace(backup_manifest, target_manifest)
            raise
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    return receipt


__all__ = ["apply_materialization", "create_materialization", "verify_materialization"]
