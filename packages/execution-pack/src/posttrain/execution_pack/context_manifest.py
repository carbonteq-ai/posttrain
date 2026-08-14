"""Canonical, safe descriptions of an already-materialized job context."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from posttrain.common import ContractError

from .service import PackedJobContext

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCHEMA = "posttrain.job-context-manifest.v1"


@dataclass(frozen=True, slots=True)
class ContextFile:
    """One regular file admitted to an actual-job build context."""

    path: PurePosixPath
    sha256: str
    size_bytes: int
    mode: int

    def __post_init__(self) -> None:
        if self.path.is_absolute() or not self.path.parts or any(part in {"", ".", ".."} for part in self.path.parts):
            raise ContractError("job context file path must be a safe relative POSIX path")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ContractError("job context file digest must be SHA-256")
        if self.size_bytes < 0:
            raise ContractError("job context file size cannot be negative")
        if self.mode & ~0o777 or self.mode & 0o600 != 0o600:
            raise ContractError("job context file mode must retain owner read/write permissions")

    def to_payload(self) -> dict[str, object]:
        return {
            "path": self.path.as_posix(),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "mode": self.mode,
        }

    @classmethod
    def from_payload(cls, payload: object) -> ContextFile:
        if not isinstance(payload, dict) or set(payload) != {"path", "sha256", "size_bytes", "mode"}:
            raise ContractError("job context file payload is invalid")
        path = payload["path"]
        digest = payload["sha256"]
        size = payload["size_bytes"]
        mode = payload["mode"]
        if (
            not isinstance(path, str)
            or not isinstance(digest, str)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or not isinstance(mode, int)
            or isinstance(mode, bool)
        ):
            raise ContractError("job context file payload has invalid field types")
        return cls(PurePosixPath(path), digest, size, mode)


@dataclass(frozen=True, slots=True)
class JobContextManifest:
    """Content-addressed transfer contract derived from ``PackedJobContext``."""

    package_key: str
    publication_key: str
    context_digest: str
    files: tuple[ContextFile, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("package", self.package_key),
            ("publication", self.publication_key),
            ("context", self.context_digest),
        ):
            if _SHA256.fullmatch(value) is None:
                raise ContractError(f"job context {name} digest must be SHA-256")
        paths = tuple(file.path for file in self.files)
        if not paths or paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ContractError("job context manifest files must be non-empty, unique, and sorted")

    @property
    def total_bytes(self) -> int:
        return sum(file.size_bytes for file in self.files)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "package_key": self.package_key,
            "publication_key": self.publication_key,
            "context_digest": self.context_digest,
            "files": [file.to_payload() for file in self.files],
        }

    def to_bytes(self) -> bytes:
        return (json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":")) + "\n").encode()

    @classmethod
    def from_payload(cls, payload: object) -> JobContextManifest:
        if not isinstance(payload, dict) or set(payload) != {
            "schema",
            "package_key",
            "publication_key",
            "context_digest",
            "files",
        }:
            raise ContractError("job context manifest payload is invalid")
        if payload["schema"] != _SCHEMA:
            raise ContractError("job context manifest schema is unsupported")
        package_key = payload["package_key"]
        publication_key = payload["publication_key"]
        context_digest = payload["context_digest"]
        files = payload["files"]
        if not all(
            isinstance(value, str) for value in (package_key, publication_key, context_digest)
        ) or not isinstance(files, list):
            raise ContractError("job context manifest payload has invalid field types")
        return cls(
            cast(str, package_key),
            cast(str, publication_key),
            cast(str, context_digest),
            tuple(ContextFile.from_payload(item) for item in files),
        )

    @classmethod
    def from_packed_context(cls, context: PackedJobContext) -> JobContextManifest:
        files: list[ContextFile] = []
        for path in sorted(context.root.rglob("*")):
            if path.is_symlink():
                raise ContractError("packed job context must not contain symbolic links")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ContractError("packed job context must contain only regular files and directories")
            relative = PurePosixPath(path.relative_to(context.root).as_posix())
            metadata = path.stat()
            files.append(
                ContextFile(
                    relative,
                    _file_sha256(path),
                    metadata.st_size,
                    stat.S_IMODE(metadata.st_mode),
                )
            )
        return cls(context.manifest.package_key, context.publication_key, context.context_digest, tuple(files))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["ContextFile", "JobContextManifest"]
