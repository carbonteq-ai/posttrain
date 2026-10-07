"""Retain collected native artifact bytes without recapturing solver runtimes."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from posttrain.common import ContractError, JsonValue, LocalArtifactRef, ProducedArtifact


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish(temporary: Path, target: Path, digest: str) -> None:
    try:
        os.link(temporary, target)
    except FileExistsError:
        if not target.is_file() or _digest_file(target) != digest:
            raise ContractError("retained assessment artifact was modified") from None
    directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _retain_bytes(directory: Path, filename: str, value: bytes) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".retain-", dir=directory)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        _publish(temporary, directory / filename, hashlib.sha256(value).hexdigest())
    finally:
        temporary.unlink(missing_ok=True)


def retain_episode_artifacts(episode: Any, root: Path) -> dict[str, Any]:
    """Save existing State.artifacts once by digest and retain their source mapping.

    This preserves runtime-collected archives as opaque bytes. It does not resolve
    external assessment URIs, extract archives, or claim overall evidence completeness.
    Call before transient native state is discarded. Repeated preservation is safe;
    altered bytes produce another immutable manifest rather than overwrite history.
    """
    entries: list[dict[str, Any]] = []
    for trace in episode.traces:
        state = getattr(trace, "state", None)
        artifacts = getattr(state, "artifacts", {})
        for source_path, value in sorted(artifacts.items()):
            entry: dict[str, Any] = {
                "trace_id": trace.id,
                "source_path": source_path,
                "status": "unavailable" if value is None else "retained",
                "digest": None,
                "size_bytes": None,
                "member": None,
            }
            if value is not None:
                if not isinstance(value, bytes):
                    raise ContractError("native collected artifacts must contain bytes or None")
                digest = hashlib.sha256(value).hexdigest()
                _retain_bytes(root / "blobs", digest, value)
                entry.update(digest=digest, size_bytes=len(value), member=f"blobs/{digest}")
            entries.append(entry)
    entries.sort(key=lambda entry: (entry["trace_id"], entry["source_path"]))
    manifest = {
        "schema_version": 1,
        "format": "verifiers-collected-artifact-evidence",
        "episode_id": episode.id,
        "artifacts": entries,
        "external_references_resolved": False,
    }
    if entries:
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        digest = hashlib.sha256(encoded).hexdigest()
        _retain_bytes(root / "manifests", f"{digest}.json", encoded)
    return manifest


def seal_assessment_artifacts(root: Path, *, name: str) -> ProducedArtifact | None:
    """Seal manifests and referenced blobs into a deterministic streaming bundle.

    The caller must serialize sealing with preservation. Bundle member paths are
    generated from digests, never from user-controlled runtime artifact paths.
    """
    manifests = sorted((root / "manifests").glob("*.json"))
    if not manifests:
        return None
    members: dict[str, Path] = {}
    episode_ids: set[str] = set()
    unavailable = 0
    for path in manifests:
        if _digest_file(path) != path.stem:
            raise ContractError("retained assessment artifact manifest was modified")
        manifest = json.loads(path.read_bytes())
        episode_ids.add(manifest["episode_id"])
        members[f"manifests/{path.name}"] = path
        for entry in manifest["artifacts"]:
            if entry["status"] == "unavailable":
                unavailable += 1
                continue
            digest = entry["digest"]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ContractError("invalid assessment artifact digest")
            blob = root / "blobs" / digest
            if not blob.is_file() or blob.stat().st_size != entry["size_bytes"] or _digest_file(blob) != digest:
                raise ContractError("retained assessment artifact bytes are missing or modified")
            members[f"blobs/{digest}"] = blob
    root.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".bundle-", dir=root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|") as archive:
                    for member, path in sorted(members.items()):
                        info = tarfile.TarInfo(member)
                        info.size = path.stat().st_size
                        info.mode = 0o600
                        with path.open("rb") as stream:
                            archive.addfile(info, stream)
            raw.flush()
            os.fsync(raw.fileno())
        digest = _digest_file(temporary)
        target = root / f"{digest}.tar.gz"
        _publish(temporary, target, digest)
    finally:
        temporary.unlink(missing_ok=True)
    retained_episode_ids: list[JsonValue] = list(sorted(episode_ids))
    return ProducedArtifact(
        name=name,
        kind="evaluation-traces",
        reference=LocalArtifactRef(target.resolve(), digest),
        metadata={
            "format": "verifiers-collected-artifact-bundle",
            "schema_version": 1,
            "compression": "gzip",
            "size_bytes": target.stat().st_size,
            "episode_ids": retained_episode_ids,
            "manifest_count": len(manifests),
            "blob_count": sum(member.startswith("blobs/") for member in members),
            "unavailable_artifact_count": unavailable,
            "external_references_resolved": False,
        },
    )
