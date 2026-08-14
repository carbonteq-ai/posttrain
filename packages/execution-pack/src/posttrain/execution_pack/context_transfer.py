"""Provider-neutral values for remote actual-job context transfer.

These values describe only an already materialized job package.  They do not
carry a Dockerfile, registry credential, BuildKit option, model, checkpoint,
or any source bytes.  The service application owns admission and persistence;
the concrete remote client owns HTTP.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from posttrain.common import ContractError
from posttrain.execution import JobPackageManifest, RuntimeImageRef

from .context_manifest import ContextFile, JobContextManifest
from .planning import ImagePublicationSpec
from .publication import publication_key_for

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_SCHEMA_VERSION = "v1"


class JobPublicationState(StrEnum):
    """Public lifecycle states for one immutable actual-job publication."""

    BLOCKED = "blocked"
    UPLOAD_REQUIRED = "upload-required"
    QUEUED = "queued"
    BUILDING = "building"
    REUSED = "reused"
    PUBLISHED = "published"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class JobBuilderCapabilities:
    """A server-installed builder definition bundle and bounded admission policy."""

    api_versions: tuple[str, ...]
    release_manifest_digests: tuple[str, ...]
    build_definition_digests: tuple[str, ...]
    platforms: tuple[str, ...]
    max_context_bytes: int
    max_file_count: int
    max_blob_bytes: int
    queue_available: bool

    def __post_init__(self) -> None:
        if not self.api_versions or _SCHEMA_VERSION not in self.api_versions:
            raise ContractError("job builder must advertise API v1")
        if tuple(sorted(set(self.api_versions))) != self.api_versions:
            raise ContractError("job builder API versions must be unique and sorted")
        for label, digests in (
            ("release manifest", self.release_manifest_digests),
            ("build definition", self.build_definition_digests),
        ):
            if not digests or tuple(sorted(set(digests))) != digests or any(_SHA256.fullmatch(value) is None for value in digests):
                raise ContractError(f"job builder {label} digests must be unique sorted SHA-256 values")
        if not self.platforms or tuple(sorted(set(self.platforms))) != self.platforms:
            raise ContractError("job builder platforms must be unique and sorted")
        if any(not value.startswith("linux/") for value in self.platforms):
            raise ContractError("job builder platforms must be Linux OCI platforms")
        if min(self.max_context_bytes, self.max_file_count, self.max_blob_bytes) <= 0:
            raise ContractError("job builder limits must be positive")
        if self.max_blob_bytes > self.max_context_bytes:
            raise ContractError("job builder blob limit cannot exceed context limit")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "posttrain.job-builder-capabilities.v1",
            "api_versions": list(self.api_versions),
            "release_manifest_digests": list(self.release_manifest_digests),
            "build_definition_digests": list(self.build_definition_digests),
            "platforms": list(self.platforms),
            "max_context_bytes": self.max_context_bytes,
            "max_file_count": self.max_file_count,
            "max_blob_bytes": self.max_blob_bytes,
            "queue_available": self.queue_available,
        }


@dataclass(frozen=True, slots=True)
class JobPublicationPlanRequest:
    """Manifest-first request for a previously packed actual-job context."""

    manifest: JobPackageManifest
    publication: ImagePublicationSpec
    context: JobContextManifest
    release_manifest_digest: str
    build_definition_digest: str

    def __post_init__(self) -> None:
        for label, value in (
            ("release manifest", self.release_manifest_digest),
            ("build definition", self.build_definition_digest),
        ):
            if _SHA256.fullmatch(value) is None:
                raise ContractError(f"job builder {label} digest must be SHA-256")
        if self.context.package_key != self.manifest.package_key:
            raise ContractError("job builder context package key differs from the manifest")
        expected = publication_key_for(self.manifest, self.publication)
        if self.context.publication_key != expected:
            raise ContractError("job builder context publication key differs from the publication spec")

    @property
    def package_key(self) -> str:
        return self.manifest.package_key

    @property
    def publication_key(self) -> str:
        return self.context.publication_key

    @property
    def project_id(self) -> str:
        return self.manifest.project_id

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "posttrain.job-publication-plan.v1",
            "manifest": self.manifest.to_payload(),
            "publication": self.publication.to_payload(),
            "context": self.context.to_payload(),
            "release_manifest_digest": self.release_manifest_digest,
            "build_definition_digest": self.build_definition_digest,
        }

    @classmethod
    def from_payload(cls, payload: object) -> JobPublicationPlanRequest:
        if not isinstance(payload, dict) or set(payload) != {
            "schema",
            "manifest",
            "publication",
            "context",
            "release_manifest_digest",
            "build_definition_digest",
        }:
            raise ContractError("job publication plan payload is invalid")
        if payload["schema"] != "posttrain.job-publication-plan.v1":
            raise ContractError("job publication plan schema is unsupported")
        release_manifest_digest = payload["release_manifest_digest"]
        build_definition_digest = payload["build_definition_digest"]
        if not isinstance(release_manifest_digest, str) or not isinstance(build_definition_digest, str):
            raise ContractError("job publication plan payload has invalid digest fields")
        return cls(
            JobPackageManifest.from_payload(payload["manifest"]),
            ImagePublicationSpec.from_payload(payload["publication"]),
            JobContextManifest.from_payload(payload["context"]),
            cast(str, release_manifest_digest),
            cast(str, build_definition_digest),
        )


@dataclass(frozen=True, slots=True)
class JobContextTransferPlan:
    """Server response to plan-before-upload admission."""

    publication_key: str
    state: JobPublicationState
    missing_blobs: tuple[ContextFile, ...] = ()
    retry_after_seconds: float | None = None
    safe_error_code: str | None = None

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.publication_key) is None:
            raise ContractError("job context transfer publication key must be SHA-256")
        if tuple(sorted(self.missing_blobs, key=lambda item: item.path)) != self.missing_blobs:
            raise ContractError("job context transfer blobs must be sorted by path")
        if self.state is not JobPublicationState.UPLOAD_REQUIRED and self.missing_blobs:
            raise ContractError("only an upload-required transfer may name missing blobs")
        if self.retry_after_seconds is not None and self.retry_after_seconds < 0:
            raise ContractError("job context transfer retry delay cannot be negative")
        if self.safe_error_code is not None and _IDENTITY.fullmatch(self.safe_error_code) is None:
            raise ContractError("job context transfer error code is invalid")

    @property
    def expected_upload_bytes(self) -> int:
        return sum(item.size_bytes for item in self.missing_blobs)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "posttrain.job-context-transfer-plan.v1",
            "publication_key": self.publication_key,
            "state": self.state.value,
            "missing_blobs": [item.to_payload() for item in self.missing_blobs],
            "expected_upload_bytes": self.expected_upload_bytes,
            "retry_after_seconds": self.retry_after_seconds,
            "safe_error_code": self.safe_error_code,
        }

    @classmethod
    def from_payload(cls, payload: object) -> JobContextTransferPlan:
        if not isinstance(payload, dict) or set(payload) != {
            "schema",
            "publication_key",
            "state",
            "missing_blobs",
            "expected_upload_bytes",
            "retry_after_seconds",
            "safe_error_code",
        }:
            raise ContractError("job context transfer plan payload is invalid")
        if payload["schema"] != "posttrain.job-context-transfer-plan.v1":
            raise ContractError("job context transfer plan schema is unsupported")
        publication_key = payload["publication_key"]
        state = payload["state"]
        missing_blobs = payload["missing_blobs"]
        retry_after_seconds = payload["retry_after_seconds"]
        safe_error_code = payload["safe_error_code"]
        expected_upload_bytes = payload["expected_upload_bytes"]
        if (
            not isinstance(publication_key, str)
            or not isinstance(state, str)
            or not isinstance(missing_blobs, list)
            or (retry_after_seconds is not None and not isinstance(retry_after_seconds, int | float))
            or isinstance(retry_after_seconds, bool)
            or (safe_error_code is not None and not isinstance(safe_error_code, str))
            or not isinstance(expected_upload_bytes, int)
            or isinstance(expected_upload_bytes, bool)
        ):
            raise ContractError("job context transfer plan payload has invalid field types")
        try:
            result = cls(
                publication_key,
                JobPublicationState(state),
                tuple(ContextFile.from_payload(item) for item in missing_blobs),
                float(retry_after_seconds) if retry_after_seconds is not None else None,
                safe_error_code,
            )
        except ValueError as error:
            raise ContractError("job context transfer plan state is unsupported") from error
        if expected_upload_bytes != result.expected_upload_bytes:
            raise ContractError("job context transfer plan byte count is invalid")
        return result


@dataclass(frozen=True, slots=True)
class JobContextTransferReceipt:
    """Bounded, non-sensitive evidence of one remote context transfer."""

    publication_key: str
    state: JobPublicationState
    context_manifest_digest: str
    source_context_digest: str
    declared_file_count: int
    declared_bytes: int
    uploaded_blob_count: int
    uploaded_bytes: int
    reused_blob_count: int
    reused_bytes: int
    receipt_digest: str | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("publication", self.publication_key),
            ("context manifest", self.context_manifest_digest),
            ("source context", self.source_context_digest),
        ):
            if _SHA256.fullmatch(value) is None:
                raise ContractError(f"job context transfer {label} digest must be SHA-256")
        if self.receipt_digest is not None and _SHA256.fullmatch(self.receipt_digest) is None:
            raise ContractError("job context transfer receipt digest must be SHA-256")
        counts = (
            self.declared_file_count,
            self.declared_bytes,
            self.uploaded_blob_count,
            self.uploaded_bytes,
            self.reused_blob_count,
            self.reused_bytes,
        )
        if any(value < 0 for value in counts):
            raise ContractError("job context transfer counters cannot be negative")
        if self.uploaded_bytes + self.reused_bytes > self.declared_bytes:
            raise ContractError("job context transfer byte counters exceed the declaration")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "posttrain.job-context-transfer-receipt.v1",
            "publication_key": self.publication_key,
            "state": self.state.value,
            "context_manifest_digest": self.context_manifest_digest,
            "source_context_digest": self.source_context_digest,
            "declared_file_count": self.declared_file_count,
            "declared_bytes": self.declared_bytes,
            "uploaded_blob_count": self.uploaded_blob_count,
            "uploaded_bytes": self.uploaded_bytes,
            "reused_blob_count": self.reused_blob_count,
            "reused_bytes": self.reused_bytes,
            "receipt_digest": self.receipt_digest,
        }


@dataclass(frozen=True, slots=True)
class JobPublicationImage:
    """Server-safe result of a verified immutable actual-job publication."""

    image: RuntimeImageRef
    kind_image: RuntimeImageRef
    cache_hit: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "image": self.image.value,
            "kind_image": self.kind_image.value,
            "cache_hit": self.cache_hit,
        }

    @classmethod
    def from_payload(cls, payload: object) -> JobPublicationImage:
        if not isinstance(payload, dict) or set(payload) != {"image", "kind_image", "cache_hit"}:
            raise ContractError("job publication image payload is invalid")
        image = payload["image"]
        kind_image = payload["kind_image"]
        cache_hit = payload["cache_hit"]
        if not isinstance(image, str) or not isinstance(kind_image, str) or not isinstance(cache_hit, bool):
            raise ContractError("job publication image payload has invalid field types")
        return cls(RuntimeImageRef(image), RuntimeImageRef(kind_image), cache_hit)


__all__ = [
    "JobBuilderCapabilities",
    "JobContextTransferPlan",
    "JobContextTransferReceipt",
    "JobPublicationPlanRequest",
    "JobPublicationImage",
    "JobPublicationState",
]
