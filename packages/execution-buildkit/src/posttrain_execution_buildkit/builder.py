"""Build and verify immutable job images through Docker Buildx Bake."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from posttrain.common import ContractError
from posttrain.execution import RuntimeImageRef

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@-]*$")
_SCHEMA = "posttrain.runtime-build-receipt.v1"
_IGNORED_PARTS = frozenset({"__pycache__", ".git", ".venv", ".venvs"})
_IGNORED_SUFFIXES = frozenset({".pyc", ".pyo"})


class RemoteImageNotFoundError(RuntimeError):
    """The registry confirms that an immutable image no longer exists."""


def digest_runtime_sources(root: Path, includes: Sequence[Path]) -> str:
    """Hash the explicit framework files that form a job-runtime layer."""

    if not root.is_absolute() or not root.is_dir():
        raise ContractError("runtime source root must be an existing absolute directory")
    entries: list[dict[str, object]] = []
    claimed: set[Path] = set()
    for configured in includes:
        source = configured if configured.is_absolute() else root / configured
        resolved = source.resolve()
        if not resolved.is_relative_to(root):
            raise ContractError("runtime source path escapes the source root")
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        candidates = [resolved] if resolved.is_file() else sorted(resolved.rglob("*"))
        for candidate in candidates:
            if (
                not candidate.is_file()
                or any(part in _IGNORED_PARTS for part in candidate.relative_to(root).parts)
                or candidate.suffix in _IGNORED_SUFFIXES
            ):
                continue
            if candidate.is_symlink():
                raise ContractError(f"runtime sources do not accept symlinks: {candidate}")
            relative = candidate.relative_to(root)
            if relative in claimed:
                continue
            claimed.add(relative)
            entries.append(
                {
                    "path": relative.as_posix(),
                    "sha256": _file_digest(candidate),
                    "executable": bool(candidate.stat().st_mode & 0o111),
                }
            )
    if not entries:
        raise ContractError("runtime source selection cannot be empty")
    entries.sort(key=lambda entry: str(entry["path"]))
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class BuildxGateway(Protocol):
    def invoke(self, arguments: Sequence[str]) -> str: ...


class BuildxCli:
    """Non-shell gateway to the selected Docker Buildx implementation."""

    def __init__(self, executable: str = "docker") -> None:
        self._executable = executable

    def invoke(self, arguments: Sequence[str]) -> str:
        # Stream progress live (bake --progress=plain) while retaining a full
        # transcript for failure diagnosis. Capturing alone hid multi-hour hangs.
        process = subprocess.Popen(
            [self._executable, "buildx", *arguments],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        chunks: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            chunks.append(line)
            sys.stderr.write(line)
            sys.stderr.flush()
        returncode = process.wait()
        output = "".join(chunks)
        if returncode != 0:
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            errors = [
                line
                for line in lines
                if line.lower().startswith("error:") or " error:" in line.lower() or line.startswith("ERROR")
            ]
            tail = "\n".join(lines[-20:])
            detail = (f"{errors[-1]}\n{tail}" if errors else (tail or "no diagnostic was returned"))[-3000:]
            if arguments[:2] == ("imagetools", "inspect") and any(
                marker in detail.lower()
                for marker in (
                    "manifest unknown",
                    "manifest not found",
                    "name unknown",
                    "not found",
                    "no such manifest",
                )
            ):
                raise RemoteImageNotFoundError(f"published image is absent: {detail}")
            # Buildx echoes the whole failing RUN script in its own error, which
            # crowds the actual failure message out of any bounded tail. Retain
            # the full output so a diagnosis does not depend on guessing.
            transcript = Path(tempfile.gettempdir()) / f"posttrain-buildx-{uuid.uuid4().hex[:12]}.log"
            try:
                transcript.write_text(
                    f"$ docker buildx {' '.join(arguments)}\n\n{output}",
                    encoding="utf-8",
                )
                location = f" Full output: {transcript}"
            except OSError:
                location = ""
            raise RuntimeError(f"docker buildx {arguments[0] if arguments else 'command'} failed: {detail}.{location}")
        return output


@dataclass(frozen=True, slots=True)
class RuntimeBuildRequest:
    profile: str
    bake_file: Path
    context: Path
    target: str
    repository: str
    source_digest: str
    lock_digest: str
    base_image: RuntimeImageRef
    builder: str | None = None
    variables: Mapping[str, str] = field(default_factory=dict)
    # Registry refs (tag or digest) whose layers seed the build after a local
    # cache wipe. Not part of build_key: they must not change the published
    # result, only how quickly BuildKit reaches it.
    cache_from: tuple[str, ...] = ()
    # Provenance/SBOM attestation manifests dominate push time for multi-GB
    # images. Off by default; pass attestations=True when policy requires them.
    attestations: bool = False
    compression_level: int = 1
    force_compression: bool = False
    # Optional machine-owned PEM bundle used to establish image trust while
    # building against private HTTPS package indexes. The local path is never
    # part of image identity; the bundle bytes are.
    trust_bundle: Path | None = None

    def __post_init__(self) -> None:
        if not self.bake_file.is_absolute() or not self.bake_file.is_file():
            raise ContractError("runtime build Bake file must be an existing absolute path")
        if not self.context.is_absolute() or not self.context.is_dir():
            raise ContractError("runtime build context must be an existing absolute directory")
        for label, value in (
            ("profile", self.profile),
            ("target", self.target),
            ("repository", self.repository),
        ):
            if not _SAFE_NAME.fullmatch(value):
                raise ContractError(f"runtime build {label} is invalid")
        if "@" in self.repository or ":" in self.repository.rsplit("/", 1)[-1]:
            raise ContractError("runtime build repository must not include a tag or digest")
        if not _SHA256.fullmatch(self.source_digest) or not _SHA256.fullmatch(self.lock_digest):
            raise ContractError("runtime build source and lock digests must be SHA-256")
        if self.builder is not None and not _SAFE_NAME.fullmatch(self.builder):
            raise ContractError("runtime build builder name is invalid")
        if any(
            not _SAFE_NAME.fullmatch(name)
            or not value
            or any(token in name.upper() for token in ("TOKEN", "PASSWORD", "SECRET", "KEY"))
            for name, value in self.variables.items()
        ):
            raise ContractError("runtime build variables must be non-secret named values")
        if self.compression_level < 0 or self.compression_level > 22:
            raise ContractError("runtime build compression_level must be between 0 and 22")
        for ref in self.cache_from:
            if not ref or any(token in ref.upper() for token in ("TOKEN", "PASSWORD", "SECRET")):
                raise ContractError("runtime build cache_from refs must be non-secret image references")
        if self.trust_bundle is not None and (not self.trust_bundle.is_absolute() or not self.trust_bundle.is_file()):
            raise ContractError("runtime build trust_bundle must be an existing absolute file")

    @property
    def build_key(self) -> str:
        payload = {
            "profile": self.profile,
            "target": self.target,
            "repository": self.repository,
            "source_digest": self.source_digest,
            "lock_digest": self.lock_digest,
            "base_image": self.base_image.value,
            "variables": dict(sorted(self.variables.items())),
            "attestations": self.attestations,
            "compression_level": self.compression_level,
            "force_compression": self.force_compression,
            "trust_bundle_sha256": (_file_digest(self.trust_bundle) if self.trust_bundle is not None else None),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @property
    def tag(self) -> str:
        return f"build-{self.build_key[:16]}"


@dataclass(frozen=True, slots=True)
class RuntimeBuildResult:
    profile: str
    build_key: str
    image: RuntimeImageRef
    source_digest: str
    lock_digest: str
    base_image: RuntimeImageRef
    receipt: Path


class BuildKitRuntimeBuilder:
    def __init__(
        self,
        gateway: BuildxGateway | None = None,
        *,
        receipt_root: Path,
    ) -> None:
        if not receipt_root.is_absolute():
            raise ValueError("runtime build receipt root must be absolute")
        self._gateway = gateway or BuildxCli()
        self._root = receipt_root

    def check(self, request: RuntimeBuildRequest) -> None:
        """Validate a definition without producing an image.

        A runtime image has no identity until it is published, so this is the
        only meaningful operation short of a real build: it resolves the Bake
        graph and evaluates the Dockerfile without emitting layers.
        """
        self._gateway.invoke(self._outline_arguments(request))

    def build(self, request: RuntimeBuildRequest) -> RuntimeBuildResult:
        receipt = self.receipt_path(request)
        if receipt.is_file():
            result = self._load_receipt(receipt)
            self._ensure_matches(request, result)
            self._verify_remote(result.image)
            return result

        self._gateway.invoke(self._outline_arguments(request))
        self._root.mkdir(parents=True, exist_ok=True)
        metadata = self._root / f".metadata-{uuid.uuid4().hex}.json"
        try:
            self._gateway.invoke(self._build_arguments(request, metadata))
            image = RuntimeImageRef(f"{request.repository}@sha256:{_metadata_digest(metadata, request.target)}")
            self._verify_remote(image)
            result = RuntimeBuildResult(
                profile=request.profile,
                build_key=request.build_key,
                image=image,
                source_digest=request.source_digest,
                lock_digest=request.lock_digest,
                base_image=request.base_image,
                receipt=receipt,
            )
            self._write_receipt(result)
            return result
        finally:
            metadata.unlink(missing_ok=True)

    def receipt_path(self, request: RuntimeBuildRequest) -> Path:
        return self._root / f"{request.build_key}.json"

    def has_receipt(self, request: RuntimeBuildRequest) -> bool:
        return self.receipt_path(request).is_file()

    def _outline_arguments(self, request: RuntimeBuildRequest) -> list[str]:
        return [
            "bake",
            "--file",
            str(request.bake_file),
            "--allow",
            f"fs.read={request.context}",
            *self._builder_arguments(request),
            "--set",
            f"{request.target}.context={request.context}",
            "--set",
            f"{request.target}.tags={request.repository}:{request.tag}",
            "--set",
            f"{request.target}.output=type=cacheonly",
            *self._trust_arguments(request),
            *self._variable_arguments(request),
            "--call",
            "check",
            request.target,
        ]

    def _build_arguments(
        self,
        request: RuntimeBuildRequest,
        metadata: Path,
    ) -> list[str]:
        compression = (
            "type=image,push=true,"
            f"compression=zstd,compression-level={request.compression_level},"
            f"force-compression={'true' if request.force_compression else 'false'},"
            "oci-mediatypes=true"
        )
        arguments = [
            "bake",
            "--file",
            str(request.bake_file),
            "--allow",
            f"fs.read={request.context}",
            *self._builder_arguments(request),
            "--progress",
            "plain",
            "--push",
            "--provenance",
            "mode=max" if request.attestations else "false",
            "--sbom",
            "true" if request.attestations else "false",
            "--metadata-file",
            str(metadata),
            "--set",
            f"{request.target}.context={request.context}",
            "--set",
            f"{request.target}.tags={request.repository}:{request.tag}",
            "--set",
            f"{request.target}.output={compression}",
        ]
        if not request.attestations:
            arguments.extend(("--set", f"{request.target}.attest="))
        for ref in request.cache_from:
            arguments.extend(("--set", f"{request.target}.cache-from=type=registry,ref={ref}"))
        arguments.extend(self._trust_arguments(request))
        arguments.extend(self._variable_arguments(request))
        arguments.append(request.target)
        return arguments

    def _builder_arguments(self, request: RuntimeBuildRequest) -> list[str]:
        return ["--builder", request.builder] if request.builder is not None else []

    def _trust_arguments(self, request: RuntimeBuildRequest) -> list[str]:
        if request.trust_bundle is None:
            return []
        return [
            "--allow",
            f"fs.read={request.trust_bundle}",
            "--set",
            f"{request.target}.secrets=id=posttrain_ca_bundle,src={request.trust_bundle}",
        ]

    def _variable_arguments(self, request: RuntimeBuildRequest) -> list[str]:
        arguments = [
            "--var",
            f"BASE_IMAGE={request.base_image.value}",
            "--var",
            f"SOURCE_DIGEST={request.source_digest}",
            "--var",
            f"LOCK_DIGEST={request.lock_digest}",
        ]
        for name, value in sorted(request.variables.items()):
            arguments.extend(("--var", f"{name}={value}"))
        return arguments

    def verify_remote(self, image: RuntimeImageRef) -> None:
        self._verify_remote(image)

    def _verify_remote(self, image: RuntimeImageRef) -> None:
        output = self._gateway.invoke(("imagetools", "inspect", image.value, "--format", "{{json .Manifest.Digest}}"))
        try:
            observed = json.loads(output)
        except json.JSONDecodeError as error:
            raise RuntimeError("Buildx returned invalid remote-image metadata") from error
        expected = image.value.rsplit("@", 1)[1]
        if observed != expected:
            raise RuntimeError(f"published runtime digest mismatch: expected {expected}, observed {observed}")

    def _write_receipt(self, result: RuntimeBuildResult) -> None:
        payload = {
            "schema": _SCHEMA,
            "profile": result.profile,
            "build_key": result.build_key,
            "image": result.image.value,
            "source_digest": result.source_digest,
            "lock_digest": result.lock_digest,
            "base_image": result.base_image.value,
        }
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        temporary = self._root / f".receipt-{uuid.uuid4().hex}.tmp"
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, result.receipt)
        except FileExistsError:
            existing = self._load_receipt(result.receipt)
            if existing != result:
                raise ContractError("runtime build receipt conflicts with an existing build") from None
        finally:
            temporary.unlink(missing_ok=True)

    def _load_receipt(self, path: Path) -> RuntimeBuildResult:
        if path.stat().st_mode & 0o077:
            raise ContractError(f"runtime build receipt must not be accessible by group or other: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ContractError(f"runtime build receipt is invalid: {path}") from error
        if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
            raise ContractError(f"runtime build receipt schema is unsupported: {path}")
        try:
            return RuntimeBuildResult(
                profile=str(payload["profile"]),
                build_key=str(payload["build_key"]),
                image=RuntimeImageRef(str(payload["image"])),
                source_digest=str(payload["source_digest"]),
                lock_digest=str(payload["lock_digest"]),
                base_image=RuntimeImageRef(str(payload["base_image"])),
                receipt=path,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError(f"runtime build receipt fields are invalid: {path}") from error

    @staticmethod
    def _ensure_matches(
        request: RuntimeBuildRequest,
        result: RuntimeBuildResult,
    ) -> None:
        if (
            result.profile != request.profile
            or result.build_key != request.build_key
            or result.source_digest != request.source_digest
            or result.lock_digest != request.lock_digest
            or result.base_image != request.base_image
            or result.image.value.rsplit("@", 1)[0] != request.repository
        ):
            raise ContractError("runtime build receipt does not match the requested build")


def _metadata_digest(path: Path, target: str) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload[target]
        digest = metadata["containerimage.digest"]
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("Buildx did not return a published image digest") from error
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise RuntimeError("Buildx returned an invalid published image digest")
    value = digest.removeprefix("sha256:")
    if not _SHA256.fullmatch(value):
        raise RuntimeError("Buildx returned an invalid published image digest")
    return value


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
